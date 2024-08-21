# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================
"""A decoder that performs beam search."""

import collections

from typing import Callable, Optional

import numpy as np
import tensorflow as tf

from typeguard import typechecked

from opennmt.tfa.seq2seq import attention_wrapper, decoder
from opennmt.tfa.utils import keras_utils
from opennmt.tfa.utils.types import FloatTensorLike, Number, TensorLike


def _tile_batch(t, multiplier):
    """Core single-tensor implementation of tile_batch."""
    t = tf.convert_to_tensor(t, name="t")
    shape_t = tf.shape(t)
    if t.shape.ndims is None or t.shape.ndims < 1:
        raise ValueError("t must have statically known rank")
    tiling = [1] * (t.shape.ndims + 1)
    tiling[1] = multiplier
    tiled_static_batch_size = (
        t.shape[0] * multiplier if t.shape[0] is not None else None
    )
    tiled = tf.tile(tf.expand_dims(t, 1), tiling)
    tiled = tf.reshape(tiled, tf.concat(([shape_t[0] * multiplier], shape_t[1:]), 0))
    tiled.set_shape(tf.TensorShape([tiled_static_batch_size]).concatenate(t.shape[1:]))
    return tiled


def tile_batch(t: TensorLike, multiplier: int, name: Optional[str] = None) -> tf.Tensor:
    """Tiles the batch dimension of a (possibly nested structure of) tensor(s).

    For each tensor t in a (possibly nested structure) of tensors,
    this function takes a tensor t shaped `[batch_size, s0, s1, ...]` composed
    of minibatch entries `t[0], ..., t[batch_size - 1]` and tiles it to have a
    shape `[batch_size * multiplier, s0, s1, ...]` composed of minibatch
    entries `t[0], t[0], ..., t[1], t[1], ...` where each minibatch entry is
    repeated `multiplier` times.

    Args:
      t: `Tensor` shaped `[batch_size, ...]`.
      multiplier: Python int.
      name: Name scope for any created operations.

    Returns:
      A (possibly nested structure of) `Tensor` shaped
      `[batch_size * multiplier, ...]`.

    Raises:
      ValueError: if tensor(s) `t` do not have a statically known rank or
      the rank is < 1.
    """
    with tf.name_scope(name or "tile_batch"):
        return tf.nest.map_structure(lambda t_: _tile_batch(t_, multiplier), t)
