# Copyright 2020 The FlaxBERT Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Run masked LM/next sentence masked_lm pre-training for BERT."""

import datetime
import functools
import itertools
import json
import os
from typing import Any

from absl import app
from absl import flags
from flax import nn
from flax import optim
import data
import import_weights
from modeling import BertForPreTraining
import training
# from flax.metrics import tensorboard
from flax.training import checkpoints
from flax.core.frozen_dict import unfreeze
import jax
import jax.numpy as jnp
from jax import random
import numpy as np
from ml_collections import ConfigDict
from tensorflow.io import gfile

import datasets
from transformers import BertTokenizerFast

from ml_collections.config_flags import config_flags

PRNGKey = Any
FLAGS = flags.FLAGS

flags.DEFINE_string(
    'output_dir', None,
    'The output directory where the model checkpoints will be written.')

config_flags.DEFINE_config_file(
  'config', None,
  'Hyperparameter configuration')


def get_output_dir(config: ConfigDict):
  """Get output directory location."""
  del config
  output_dir = FLAGS.output_dir
  if output_dir is None:
    output_name = 'pretrain_{timestamp}'.format(
        timestamp=datetime.datetime.now().strftime('%Y%m%d_%H%M'),
    )
    output_dir = os.path.join('~', 'efficient_transformers', output_name)
    output_dir = os.path.expanduser(output_dir)
    print()
    print('No --output_dir specified')
    print('Using default output_dir:', output_dir, flush=True)
  return output_dir


def get_model_and_params(config: ConfigDict, rng: PRNGKey):
  """Create a model, starting with a pre-trained checkpoint."""
  if config.init_checkpoint:
    initial_params = import_weights.load_params(
        init_checkpoint=config.init_checkpoint,
        hidden_size=config.model.hidden_size,
        num_attention_heads=config.model.num_attention_heads,
        keep_masked_lm_head=True)
  else:
    rng, dropout_rng = random.split(rng)
    rngs = {'params': rng, 'dropout': dropout_rng}
    seq_input = jnp.ones((1, config.max_seq_length), jnp.int32)
    predict_input = jnp.ones((1, config.max_predictions_per_seq), jnp.int32)
    model = BertForPreTraining(FLAGS.config.model)
    initial_params = model.init(
        rngs, seq_input, seq_input, seq_input, predict_input,
        deterministic=True)['params']
    def fixup_for_tpu(x, i=[0]):
      """HACK to fix incorrect param initialization on TPU."""
      if isinstance(x, jax.ShapeDtypeStruct):
        i[0] += 1
        if len(x.shape) == 2:
          return jnp.zeros(x.shape, x.dtype)
        else:
          return nn.linear.default_kernel_init(random.PRNGKey(i[0]), x.shape, x.dtype)
      else:
        return x
    initial_params = jax.tree_map(fixup_for_tpu, initial_params)
  return model, initial_params


def create_optimizer(config: ConfigDict, initial_params):
  common_kwargs = dict(
    learning_rate=config.learning_rate,
    beta1=0.9,
    beta2=0.999,
    eps=1e-6,
  )
  optimizer_decay_def = optim.Adam(weight_decay=0.01, **common_kwargs)
  optimizer_no_decay_def = optim.Adam(weight_decay=0.0, **common_kwargs)
  decay = optim.ModelParamTraversal(lambda path, _: 'bias' not in path)
  no_decay = optim.ModelParamTraversal(lambda path, _: 'bias' in path)
  optimizer_def = optim.MultiOptimizer(
    (decay, optimizer_decay_def), (no_decay, optimizer_no_decay_def))
  optimizer = optimizer_def.create(initial_params)
  return optimizer


def compute_pretraining_loss_and_metrics(params, batch, model):
  """Compute cross-entropy loss for classification tasks."""
  metrics = model.apply({'params': params},
      params, batch['input_ids'], (batch['input_ids'] > 0).astype(np.int32),
      batch['token_type_ids'], batch['masked_lm_positions'], 
      batch['masked_lm_ids'], batch['masked_lm_weights'], 
      batch['next_sentence_label'])
  return metrics['loss'], metrics


def compute_pretraining_stats(params, batch, model):
  """Used for computing eval metrics during pre-training."""
  masked_lm_logits, next_sentence_logits = model.apply({'params': params},
      batch['input_ids'],
      (batch['input_ids'] > 0).astype(np.int32),
      batch['token_type_ids'],
      batch['masked_lm_positions'],
      deterministic=True)
  stats = BertForPreTraining.compute_metrics(
      masked_lm_logits, next_sentence_logits,
      batch['masked_lm_ids'],
      batch['masked_lm_weights'],
      batch['next_sentence_label'])

  masked_lm_correct = jnp.sum(
      (masked_lm_logits.argmax(-1) == batch['masked_lm_ids'].reshape((-1,))
       ) * batch['masked_lm_weights'].reshape((-1,)))
  next_sentence_labels = batch['next_sentence_label'].reshape((-1,))
  next_sentence_correct = jnp.sum(
      next_sentence_logits.argmax(-1) == next_sentence_labels)
  stats = {
      'masked_lm_correct': masked_lm_correct,
      'masked_lm_total': jnp.sum(batch['masked_lm_weights']),
      'next_sentence_correct': next_sentence_correct,
      'next_sentence_total': jnp.sum(jnp.ones_like(next_sentence_labels)),
      **stats
  }
  return stats


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  config = FLAGS.config

  rng = random.PRNGKey(0)
  model, initial_params = get_model_and_params(config, rng)
  optimizer = create_optimizer(config, initial_params)

  output_dir = get_output_dir(config)
  gfile.makedirs(output_dir)

  # Restore from a local checkpoint, if one exists.
  optimizer = checkpoints.restore_checkpoint(output_dir, optimizer)
  start_step = int(optimizer.state[0].step)

  optimizer = optimizer.replicate()

  os.environ['TOKENIZERS_PARALLELISM'] = 'true'
  tokenizer = BertTokenizerFast.from_pretrained(config.tokenizer)
  tokenizer.model_max_length = config.max_seq_length

  # The commented lines below correspond to a data pipeline that uses publicly
  # available data, in the form of English Wikipedia as processed and hosted by
  # the HuggingFace datasets library. The pipeline works, and downstream task
  # performance shows a benefit to pre-training, but I (Nikita) have yet to
  # confirm that final model quality is on par with the original BERT.
  #
  dataset = datasets.load_dataset('wikipedia', '20200501.en')['train']
  data_pipeline = data.PretrainingDataPipelineV1(
    dataset, tokenizer,
    max_predictions_per_seq=config.max_predictions_per_seq)

  # The data pipeline below relies on having text files of Wikipedia + Books in
  # the same format as the original BERT data. That original data is not
  # publicly available, so you will need to provide your own. I (Nikita) have
  # had success using data from Gong et al. "Efficient Training of BERT by
  # Progressively Stacking", but this data was also obtained through private
  # correspondence and may not be generally available.
  # The data_files argument may be a list, if data is split across multiple
  # input files.
  # dataset = datasets.load_dataset(
  #   'bert_data.py',
  #   data_files=os.path.expanduser('~/data/bert/corpus.train.tok')
  # )['train']
  # data_pipeline = data.PretrainingDataPipeline(
  #   dataset, tokenizer,
  #   max_predictions_per_seq=config.max_predictions_per_seq)

  datasets.logging.set_verbosity_error()

  learning_rate_fn = training.create_learning_rate_scheduler(
      factors='constant * linear_warmup * linear_decay',
      base_learning_rate=config.learning_rate,
      warmup_steps=config.num_warmup_steps,
      steps_per_cycle=config.num_train_steps - config.num_warmup_steps,
  )

  train_history = training.TrainStateHistory(learning_rate_fn)
  train_state = train_history.initial_state()

  if config.do_train:
    train_iter = data_pipeline.get_inputs(
        batch_size=config.train_batch_size, training=True)
    metrics_fn = functools.partial(compute_pretraining_loss_and_metrics, 
                                   model=model)
    train_step_fn = training.create_train_step(metrics_fn, clip_grad_norm=1.0)

    for step, batch in zip(range(start_step, config.num_train_steps),
                           train_iter):
      optimizer, train_state = train_step_fn(optimizer, batch, train_state)
      if jax.host_id() == 0 and (step % config.save_checkpoints_steps == 0
                                 or step == config.num_train_steps-1):
        checkpoints.save_checkpoint(output_dir,
                                    optimizer.unreplicate(),
                                    step)
        config_path = os.path.join(output_dir, 'config.json')
        if not os.path.exists(config_path):
          with open(config_path, 'w') as f:
            json.dump({'model_type': 'bert', **config.model}, f)

  if config.do_eval:
    eval_iter = data_pipeline.get_inputs(batch_size=config.eval_batch_size)
    eval_iter = itertools.islice(eval_iter, config.max_eval_steps)
    stats_fn = functools.partial(compute_pretraining_stats, model=model)
    eval_fn = training.create_eval_fn(stats_fn, sample_feature_name='input_ids')
    eval_stats = eval_fn(optimizer, eval_iter)

    eval_metrics = {
        'loss': jnp.mean(eval_stats['loss']),
        'masked_lm_loss': jnp.mean(eval_stats['masked_lm_loss']),
        'next_sentence_loss': jnp.mean(eval_stats['next_sentence_loss']),
        'masked_lm_accuracy': jnp.sum(
            eval_stats['masked_lm_correct']
            ) / jnp.sum(eval_stats['masked_lm_total']),
        'next_sentence_accuracy': jnp.sum(
            eval_stats['next_sentence_correct']
            ) / jnp.sum(eval_stats['next_sentence_total']),
    }

    eval_results = []
    for name, val in sorted(eval_metrics.items()):
      line = f'{name} = {val:.06f}'
      print(line, flush=True)
      eval_results.append(line)

    eval_results_path = os.path.join(output_dir, 'eval_results.txt')
    with gfile.GFile(eval_results_path, 'w') as f:
      for line in eval_results:
        f.write(line + '\n')


if __name__ == '__main__':
  app.run(main)
