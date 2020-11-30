# General imports.
import os
import time
import jax_pod_setup
import jax
import jax.numpy as jnp
import flax

# Huggingface datasets and transformers libraries.
import datasets
from transformers import BertTokenizerFast

# flax_bert-specific imports.
from flax import optim
import data
import modeling as flax_models
from demo_lib import get_config, import_pretrained_params, create_model, create_optimizer, run_train, run_eval

os.environ['TOKENIZERS_PARALLELISM'] = 'true'

print(jax.device_count())

tic = time.perf_counter()

train_settings = {
    'train_batch_size': 32,
    'eval_batch_size': 8,
    'learning_rate': 5e-5,
    'num_train_epochs': 3,
    'dataset_path': 'glue',
    'dataset_name': 'mrpc'  # ['cola', 'mrpc', 'sst2', 'stsb', 'qnli', 'rte']
}

# Load the GLUE task.
dataset = datasets.load_dataset('glue', train_settings['dataset_name'])
print(f'Jax host count: {jax.host_count()}')
print(f'Jax host id: {jax.host_id()}')

# Get pre-trained config and update it with the train configuration.
config = get_config('bert-base-uncased', dataset)
config.update(train_settings)

# Load HuggingFace tokenizer and data pipeline.
tokenizer = BertTokenizerFast.from_pretrained(config.tokenizer)
data_pipeline = data.ClassificationDataPipeline(dataset, tokenizer)

# Create Flax model and optimizer.
pretrained_params = import_pretrained_params(config)
model = create_model(config, pretrained_params)
optimizer = create_optimizer(config, model, pretrained_params)

tic = time.perf_counter()
optimizer = run_train(optimizer, data_pipeline, tokenizer, config)
toc = time.perf_counter()

print(f"Training took {toc - tic:0.4f} seconds")

run_eval(optimizer, data_pipeline, config)
