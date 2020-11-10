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

"""Transformer models."""

from flax import linen as nn
import efficient_attention
import layers
from flax.training.common_utils import onehot
import jax.numpy as jnp
from ml_collections import ConfigDict
from typing import Tuple


NEG_INFINITY = -10000.0
LAYER_NORM_EPSILON = 1e-12


def get_hidden_activation(config: ConfigDict):
  #TODO(kitaev): implement for other values of the config
  assert config.hidden_act == 'gelu'
  return nn.gelu


def get_kernel_init(config: ConfigDict):
  #TODO(kitaev): make this actually configurable
  return nn.initializers.xavier_uniform()


class BertModel(nn.Module):
  """BERT model without any task-specific heads."""
  config: ConfigDict

  @nn.compact  
  def __call__(self, 
               input_ids: jnp.ndarray, 
               input_mask: jnp.ndarray,
               type_ids: jnp.ndarray, 
               *,
               deterministic: bool = False) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Applies BERT model on the inputs."""

    config = self.config
    word_embeddings = nn.Embed(
        num_embeddings=config.vocab_size, 
        features=config.hidden_size, 
        embedding_init=get_kernel_init(config),
        name='word_embeddings')(input_ids)
    position_embeddings = layers.PositionalEncoding(
        max_len=config.max_position_embeddings,
        posemb_init=get_kernel_init(config),
        name='position_embeddings')(word_embeddings)
    type_embeddings = nn.Embed(
        num_embeddings=config.type_vocab_size,
        features=config.hidden_size,
        embedding_init=get_kernel_init(config),
        name='type_embeddings')(type_ids)

    embeddings = word_embeddings + position_embeddings + type_embeddings
    embeddings = nn.LayerNorm(
        epsilon=LAYER_NORM_EPSILON, 
        name='embeddings_layer_norm')(embeddings)
    embeddings = nn.Dropout(
        rate=config.hidden_dropout_prob)(embeddings,
                                         deterministic=deterministic)

    # Transformer blocks
    feed_forward = layers.FeedForward(
        d_ff=config.intermediate_size,
        dropout_rate=config.hidden_dropout_prob,
        intermediate_activation=get_hidden_activation(config),
        kernel_init=get_kernel_init(config),
        name='feed_forward')

    attention = nn.SelfAttention(
        num_heads=config.num_attention_heads,
        qkv_features=config.hidden_size,
        dropout_rate=config.attention_probs_dropout_prob,
        kernel_init=get_kernel_init(config),
        deterministic=deterministic,
        name='self_attention')

    hidden_states = embeddings
    mask = input_mask.astype(jnp.int32)
    for layer_num in range(config.num_hidden_layers):
      hidden_states = layers.TransformerBlock(
        feed_forward=feed_forward,
        attention=attention,
        name=f'encoder_layer_{layer_num}')(hidden_states, mask, 
                                           deterministic=deterministic)

    pooled_output = nn.Dense(
        kernel_init=get_kernel_init(config),
        name='pooler',
        features=config.hidden_size)(hidden_states[:, 0])
    pooled_output = jnp.tanh(pooled_output)

    return hidden_states, pooled_output

  def get_embedding_table(self, **unused_kwargs):
    return self.variables['params']['word_embeddings']['embedding']


class GatherIndexes(nn.Module):
  """Gathers the vectors at the specific positions."""

  @nn.compact
  def __call__(self, sequence_tensor: jnp.ndarray, positions: jnp.ndarray):
    """Applies gather indexes layer.

    Args:
      sequence_tensor: Sequence output of `BertModel` layer of shape
        (`batch_size`, `seq_length`, num_hidden) where num_hidden is number of
        hidden units of `BertModel` layer.
      positions: Positions ids of tokens in sequence to mask for pretraining
        of with dimension (batch_size, num_predictions) where
        `num_predictions` is maximum number of tokens to mask out and predict
        per each sequence.

    Returns:
      Masked out sequence tensor of shape (batch_size * num_predictions,
      num_hidden).
    """
    batch_size, seq_length, width = sequence_tensor.shape
    flat_offsets = jnp.reshape(jnp.arange(batch_size) * seq_length, [-1, 1])
    flat_positions = jnp.reshape(positions + flat_offsets, [-1])
    flat_sequence_tensor = jnp.reshape(sequence_tensor,
                                       [batch_size * seq_length, width])
    output_tensor = jnp.take(flat_sequence_tensor, flat_positions, axis=0)

    return output_tensor


class BertForSequenceClassification(nn.Module):
  """Bert model for sequence classification."""
  config: ConfigDict
  n_classes: int

  @nn.compact
  def __call__(self, 
               input_ids: jnp.ndarray,
               input_mask: jnp.ndarray,
               type_ids: jnp.ndarray,
               labels: jnp.ndarray = None,
               *,
               deterministic: bool = False):
    """Applies BERT for sequence classification."""
    bert = BertModel(config=self.config, name='bert')
    _, pooled_output = bert(input_ids, input_mask, type_ids, 
                            deterministic=deterministic)
    pooled_output = nn.Dropout(
        rate=self.config.hidden_dropout_prob,
        deterministic=deterministic)(pooled_output)
    logits = layers.OutputProjection(
        n_out=self.n_classes, kernel_init=get_kernel_init(self.config),
        name='classification')(pooled_output)

    if labels is None:
      return logits
    elif logits.shape[-1] == 1:
      # Regression task
      loss = jnp.mean((logits[..., 0] - labels) ** 2)
      return {'loss': loss}
    else:
      # Classification task
      logits = nn.log_softmax(logits)
      loss = -jnp.mean(jnp.sum(
          onehot(labels, logits.shape[-1]) * logits, axis=-1))
      return {'loss': loss}


class BertForPreTraining(nn.Module):
  """Bert model for pre-training."""
  config: ConfigDict

  @nn.compact
  def __call__(self,
               input_ids: jnp.ndarray,
               input_mask: jnp.ndarray,
               type_ids: jnp.ndarray,
               masked_lm_positions: jnp.ndarray = None,
               masked_lm_labels: jnp.ndarray = None,
               masked_lm_weights: jnp.ndarray = None,
               next_sentence_labels: jnp.ndarray = None,
               *,
               deterministic: bool = False):
    """Applies BERT for pre-training."""
    config = self.config
    bert = BertModel(config=config, name='bert')
    sequence_output, pooled_output = bert(
        input_ids, input_mask, type_ids, deterministic=deterministic)
    if masked_lm_positions is None:
      return sequence_output, pooled_output

    # Masked LM
    masked_lm_input = GatherIndexes()(sequence_output, masked_lm_positions)
    masked_lm_input = nn.Dense(
        features=config.hidden_size,
        kernel_init=get_kernel_init(config),
        name='predictions_transform_dense')(masked_lm_input)
    masked_lm_input = get_hidden_activation(config)(masked_lm_input)
    masked_lm_input = nn.LayerNorm(epsilon=LAYER_NORM_EPSILON,
        name='predictions_transform_layernorm')(masked_lm_input)
    masked_lm_logits = layers.OutputProjection(
        name='predictions_output')(masked_lm_input, bert.get_embedding_table())

    # Next-sentence prediction
    next_sentence_logits = layers.OutputProjection(
        n_out=2, kernel_init=get_kernel_init(config),
        name='classification')(pooled_output)

    if masked_lm_labels is None or next_sentence_labels is None:
      return masked_lm_logits, next_sentence_logits
    else:
      return BertForPreTraining.compute_metrics(
          masked_lm_logits, next_sentence_logits,
          masked_lm_labels, masked_lm_weights, next_sentence_labels)

  @staticmethod
  def compute_metrics(masked_lm_logits: jnp.ndarray,
                      next_sentence_logits: jnp.ndarray,
                      masked_lm_labels: jnp.ndarray,
                      masked_lm_weights: jnp.ndarray,
                      next_sentence_labels: jnp.ndarray):
    """Computes the pre-training loss and its components."""
    masked_lm_logits = nn.log_softmax(masked_lm_logits)
    masked_lm_labels = onehot(
        masked_lm_labels.reshape((-1,)), masked_lm_logits.shape[-1])
    masked_lm_weights = masked_lm_weights.reshape((-1,))
    masked_lm_loss = -jnp.sum(
        jnp.sum(masked_lm_logits * masked_lm_labels,
                axis=-1) * masked_lm_weights) / jnp.sum(masked_lm_weights)

    next_sentence_logits = nn.log_softmax(next_sentence_logits)
    next_sentence_labels = next_sentence_labels.reshape((-1,))
    next_sentence_loss = -jnp.mean(jnp.sum(
        onehot(
            next_sentence_labels, next_sentence_logits.shape[-1]
            ) * next_sentence_logits, axis=-1))
    return {
        'loss': masked_lm_loss + next_sentence_loss,
        'masked_lm_loss': masked_lm_loss,
        'next_sentence_loss': next_sentence_loss,
    }