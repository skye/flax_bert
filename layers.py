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

"""Layers used in a Transformer."""

from typing import Any, Callable

from flax import linen as nn
import jax.numpy as jnp


LAYER_NORM_EPSILON = 1e-6


class PositionalEncoding(nn.Module):
  """Learned positional embeddings for the Transformer."""
  max_len: int = 2048
  posemb_init: Callable[..., Any] = nn.initializers.xavier_normal()

  @nn.compact
  def __call__(self, inputs: jnp.ndarray):
    """Applies PositionalEncoding module."""
    assert inputs.ndim == 3, (
        f'Number of dimention should be 3, but it is: {inputs.ndim}')
    length = inputs.shape[1]
    pos_emb_shape = (1, self.max_len, inputs.shape[-1])
    pos_embedding = self.param('embedding', self.posemb_init, pos_emb_shape)
    return pos_embedding[:, :length, :]


class FeedForward(nn.Module):
  """Feed-forward layer for a Transformer model."""
  # TODO(kitaev): support chunking
  d_ff: int
  dropout_rate: float = 0.0
  intermediate_activation: Callable[..., Any] = nn.gelu
  # TODO(kitaev): chunk_size hparam for chunking
  kernel_init: Callable[..., Any] = nn.initializers.xavier_uniform()
  
  @nn.compact
  def __call__(self,
               hidden_states: jnp.ndarray,
               *,
               deterministic: bool = False):
    """Applies FeedForward module."""
    d_model = hidden_states.shape[-1]
    hidden_states = nn.Dense(
        self.d_ff,
        kernel_init=self.kernel_init,
        name='intermediate')(hidden_states)
    hidden_states = self.intermediate_activation(hidden_states)
    hidden_states = nn.Dense(
        d_model,
        kernel_init=self.kernel_init,
        name='output')(hidden_states)
    hidden_states = nn.Dropout(
        rate=self.dropout_rate)(hidden_states, 
                                deterministic=deterministic)
    return hidden_states


class TransformerBlock(nn.Module):
  """Post-norm transformer block.."""
  feed_forward: Callable[..., Any]
  attention: Callable[..., Any]

  @nn.compact
  def __call__(self, 
               hidden_states: jnp.ndarray,
               mask: jnp.ndarray = None,
               *,
               deterministic: bool = False):
    """Applies TransformerBlock module."""
    attention_output = self.attention(hidden_states, mask)
    hidden_states = nn.LayerNorm(
        epsilon=LAYER_NORM_EPSILON,
        name='self_attention_layer_norm')(hidden_states + attention_output)
    feed_forward_output = self.feed_forward(hidden_states,
                                            deterministic=deterministic)
    hidden_states = nn.LayerNorm(
        epsilon=LAYER_NORM_EPSILON,
        name='output_layer_norm')(hidden_states + feed_forward_output)

    return hidden_states


class OutputProjection(nn.Module):
  """A dense projection layer for computing output logits."""
  n_out: jnp.ndarray = None
  use_bias: bool = True
  kernel_init: Callable[..., Any] = nn.initializers.lecun_normal()
  bias_init: Callable[..., Any] = nn.initializers.zeros

  @nn.compact
  def __call__(self, inputs: jnp.ndarray, kernel: jnp.ndarray = None):
    """Applies OutputProjection module."""
    if kernel is None:
      assert self.n_out is not None, (
          'n_out argument is required when not re-using an embedding matrix')
      kernel = self.param('kernel', self.kernel_init, 
                                    (self.n_out, inputs.shape[-1]))
    y = jnp.matmul(inputs, jnp.transpose(kernel, (1, 0)))
    if self.use_bias:
      bias = self.param('bias', self.bias_init, (y.shape[-1],))
      y = y + bias
    return y
