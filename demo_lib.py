import os
import jax
import jax.numpy as jnp
import flax

# We use HuggingFace's tokenizer and input pipeline.
import transformers
import datasets
import numpy as np

# We use the Flax model.
import data
import modeling as flax_models
import training

os.environ['TOKENIZERS_PARALLELISM'] = 'true'

import ml_collections

import transformers

import import_weights
from flax import nn
from flax import optim

from tensorflow.io import gfile

from transformers import BertTokenizerFast

import time

def get_config(init_checkpoint, dataset):
  hf_config = transformers.AutoConfig.from_pretrained(init_checkpoint)
  model_config = ml_collections.ConfigDict({
      'vocab_size': hf_config.vocab_size,
      'hidden_size': hf_config.hidden_size,
      'num_hidden_layers': hf_config.num_hidden_layers,
      'num_attention_heads': hf_config.num_attention_heads,
      'hidden_act':  hf_config.hidden_act,
      'intermediate_size': hf_config.intermediate_size,
      'hidden_dropout_prob': hf_config.hidden_dropout_prob,
      'attention_probs_dropout_prob': hf_config.attention_probs_dropout_prob,
      'max_position_embeddings': hf_config.max_position_embeddings,
      'type_vocab_size': hf_config.type_vocab_size,
    })
  is_regression_task = (
    dataset['train'].features['label'].dtype == 'float32')
  if is_regression_task:
    num_classes = 1
  else:
    num_classes = dataset['train'].features['label'].num_classes
  config = ml_collections.ConfigDict({
      'init_checkpoint': init_checkpoint,
      'tokenizer': init_checkpoint,
      'model': model_config,
      'warmup_proportion': 0.1,
      'max_seq_length': 128,
      'num_classes': num_classes
  })
  return config


def import_pretrained_params(config):
  return import_weights.load_params(
      init_checkpoint=config.init_checkpoint,
      hidden_size=config.model.hidden_size,
      num_attention_heads=config.model.num_attention_heads,
      num_classes=config.num_classes)


def create_model(config, initial_params):
  model_kwargs = dict(
      config=config.model,
      n_classes=config.num_classes,
  )
  model_def = flax_models.BertForSequenceClassification.partial(**model_kwargs)
  model = nn.Model(model_def, initial_params)
  return model

def create_optimizer(config, model, initial_params):
  """Create a model, starting with a pre-trained checkpoint."""
  common_kwargs = dict(
    learning_rate=config.learning_rate,
    beta1=0.9,
    beta2=0.999,
    eps=1e-6,
  )
  optimizer_decay_def = optim.Adam(
    weight_decay=0.01, **common_kwargs)
  optimizer_no_decay_def = optim.Adam(
    weight_decay=0.0, **common_kwargs)
  decay = optim.ModelParamTraversal(lambda path, _: 'bias' not in path)
  no_decay = optim.ModelParamTraversal(lambda path, _: 'bias' in path)
  optimizer_def = optim.MultiOptimizer(
    (decay, optimizer_decay_def), (no_decay, optimizer_no_decay_def))
  optimizer = optimizer_def.create(model)
  optimizer = optimizer.replicate()
  del model  # don't keep a copy of the initial model
  return optimizer


def get_num_train_steps(config, data_pipeline):
  num_train_examples = len(data_pipeline.dataset['train'])
  print(f'Num train examples: {num_train_examples}')
  num_train_steps = int(
      num_train_examples * config.num_train_epochs // config.train_batch_size)
  return num_train_steps


def get_learning_rate_fn(config, num_train_steps):
  warmup_steps = int(config.warmup_proportion * num_train_steps)
  cooldown_steps = num_train_steps - warmup_steps
  learning_rate_fn = training.create_learning_rate_scheduler(
      factors='constant * linear_warmup * linear_decay',
      base_learning_rate=config.learning_rate,
      warmup_steps=warmup_steps,
      steps_per_cycle=cooldown_steps,
  )
  return learning_rate_fn


def compute_loss_and_metrics(model, batch, rng):
  """Compute cross-entropy loss for classification tasks."""
  with nn.stochastic(rng):
    metrics = model(
        batch['input_ids'],
        (batch['input_ids'] > 0).astype(np.int32),
        batch['token_type_ids'],
        batch['label'])
  return metrics['loss'], metrics


def compute_classification_stats(model, batch):
  with nn.stochastic(jax.random.PRNGKey(0)):
    y = model(
        batch['input_ids'],
        (batch['input_ids'] > 0).astype(np.int32),
        batch['token_type_ids'],
        deterministic=True)
  return {
      'idx': batch['idx'],
      'label': batch['label'],
      'prediction': y.argmax(-1)
  }


def run_train(optimizer, data_pipeline, tokenizer, config):
  tokenizer.model_max_length = config.max_seq_length
  num_train_steps = get_num_train_steps(config, data_pipeline)

  learning_rate_fn = get_learning_rate_fn(config, num_train_steps)
  train_history = training.TrainStateHistory(learning_rate_fn)
  train_state = train_history.initial_state()

  t0 = time.time()
  train_step_fn = training.create_train_step(
    compute_loss_and_metrics, clip_grad_norm=1.0)
  train_iter = data_pipeline.get_inputs(
    split='train', batch_size=config.train_batch_size, training=True)
  print(f'Input pipeline: {time.time()-t0}')

  if jax.host_id() == 0:
    print(f'\nStarting training on {config.dataset_name} for {num_train_steps} '
          f'steps ({config.num_train_epochs:.0f} epochs)...\n')

  for _, batch in zip(range(0, num_train_steps), train_iter):
    optimizer, train_state = train_step_fn(optimizer, batch, train_state)

  if jax.host_id() == 0:
    print('\nFinished training.')

  return optimizer

def run_eval(optimizer, data_pipeline, config):
  # if jax.host_id() != 0:
  #    return

  eval_step = training.create_eval_fn(compute_classification_stats)

  if config.dataset_name == 'mnli':
    validation_splits = ['validation_matched', 'validation_mismatched']
  else:
    validation_splits = ['validation']

  for split in validation_splits:
    eval_iter = data_pipeline.get_inputs(
        split='validation', batch_size=config.eval_batch_size, training=False)
    eval_stats = eval_step(optimizer, eval_iter)
    eval_metric = datasets.load_metric(config.dataset_path, config.dataset_name)
    eval_metric.add_batch(
      predictions=eval_stats['prediction'],
      references=eval_stats['label'])
    eval_metrics = eval_metric.compute()
    print('\nRunning eval...\n')
    prefix = 'eval_mismatched' if split == 'validation_mismatched' else 'eval'
    for name, val in sorted(eval_metrics.items()):
      print(f'{prefix}_{name} = {val:.06f}', flush=True)
