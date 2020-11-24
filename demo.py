# General imports.
import os
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
from demo_lib import get_config, import_pretrained_params, create_optimizer, create_model, run_train, run_eval

os.environ['TOKENIZERS_PARALLELISM'] = 'true'

glue_task = 'mrpc' #@param ['cola', 'mrpc', 'qqp', 'sst2', 'stsb', 'qnli', 'rte']
train_batch_size = 31 #@param {type:'integer'}
eval_batch_size = 32 #@param {type:'integer'}
learning_rate = 0.00005 #@param {type:'number'}
num_train_epochs = 3 #@param {type:'number'}

train_settings = {
    'train_batch_size': train_batch_size,
    'eval_batch_size': eval_batch_size,
    'learning_rate': learning_rate,
    'num_train_epochs': num_train_epochs,
    'dataset_path': 'glue',
    'dataset_name': glue_task
}


dataset = datasets.load_dataset('glue', glue_task)

config = get_config('bert-base-uncased', dataset)
config.update(train_settings)

tokenizer = BertTokenizerFast.from_pretrained(config.tokenizer)
data_pipeline = data.ClassificationDataPipeline(dataset, tokenizer)

pretrained_params = import_pretrained_params(config)
model = create_model(config, pretrained_params)
optimizer = create_optimizer(config, model, pretrained_params)

optimizer = run_train(optimizer, data_pipeline, tokenizer, config)
run_eval(optimizer, data_pipeline, config)