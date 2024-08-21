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


def _check_ndims(t):
    if t.shape.ndims is None:
        raise ValueError(
            "Expected tensor (%s) to have known rank, but ndims == None." % t
        )


def _check_static_batch_beam_maybe(shape, batch_size, beam_width):
    """Raises an exception if dimensions are known statically and can not be
    reshaped to [batch_size, beam_size, -1]."""
    reshaped_shape = tf.TensorShape([batch_size, beam_width, None])
    assert len(shape.dims) > 0
    if batch_size is None or shape[0] is None:
        return True  # not statically known => no check
    if shape[0] == batch_size * beam_width:
        return True  # flattened, matching
    has_second_dim = shape.ndims >= 2 and shape[1] is not None
    if has_second_dim and shape[0] == batch_size and shape[1] == beam_width:
        return True  # non-flattened, matching
    # Otherwise we could not find a match and warn:
    tf.get_logger().warn(
        "TensorArray reordering expects elements to be "
        "reshapable to %s which is incompatible with the "
        "current shape %s. Consider setting "
        "reorder_tensor_arrays to False to disable TensorArray "
        "reordering during the beam search." % (reshaped_shape, shape)
    )
    return False


def _check_batch_beam(t, batch_size, beam_width):
    """Returns an Assert operation checking that the elements of the stacked
    TensorArray can be reshaped to [batch_size, beam_size, -1].

    At this point, the TensorArray elements have a known rank of at
    least 1.
    """
    error_message = (
        "TensorArray reordering expects elements to be "
        "reshapable to [batch_size, beam_size, -1] which is "
        "incompatible with the dynamic shape of %s elements. "
        "Consider setting reorder_tensor_arrays to False to disable "
        "TensorArray reordering during the beam search."
        % (t if tf.executing_eagerly() else t.name)
    )
    rank = t.shape.ndims
    shape = tf.shape(t)
    if rank == 2:
        condition = tf.equal(shape[1], batch_size * beam_width)
    else:
        condition = tf.logical_or(
            tf.equal(shape[1], batch_size * beam_width),
            tf.logical_and(
                tf.equal(shape[1], batch_size), tf.equal(shape[2], beam_width)
            ),
        )
    return tf.Assert(condition, [error_message])


def _as_shape(value):
    """Converts the argument to a TensorShape if not already one."""
    if not isinstance(value, tf.TensorShape):
        if isinstance(value, tf.Tensor):
            value = tf.get_static_value(value)
        value = tf.TensorShape(value)
    return value


def get_attention_probs(next_cell_state, coverage_penalty_weight):
    """Get attention probabilities from the cell state.

    Args:
      next_cell_state: The next state from the cell, e.g. an instance of
        AttentionWrapperState if the cell is attentional.
      coverage_penalty_weight: Float weight to penalize the coverage of source
        sentence. Disabled with 0.0.

    Returns:
      The attention probabilities with shape
        `[batch_size, beam_width, max_time]` if coverage penalty is enabled.
        Otherwise, returns None.

    Raises:
      ValueError: If no cell is attentional but coverage penalty is enabled.
    """
    if coverage_penalty_weight == 0.0:
        return None

    # Attention probabilities of each attention layer. Each with shape
    # `[batch_size, beam_width, max_time]`.
    probs_per_attn_layer = []
    if isinstance(next_cell_state, attention_wrapper.AttentionWrapperState):
        probs_per_attn_layer = [attention_probs_from_attn_state(next_cell_state)]
    elif isinstance(next_cell_state, tuple):
        for state in next_cell_state:
            if isinstance(state, attention_wrapper.AttentionWrapperState):
                probs_per_attn_layer.append(attention_probs_from_attn_state(state))

    if not probs_per_attn_layer:
        raise ValueError(
            "coverage_penalty_weight must be 0.0 if no cell is attentional."
        )

    if len(probs_per_attn_layer) == 1:
        attention_probs = probs_per_attn_layer[0]
    else:
        # Calculate the average attention probabilities from all attention
        # layers.
        attention_probs = [tf.expand_dims(prob, -1) for prob in probs_per_attn_layer]
        attention_probs = tf.concat(attention_probs, -1)
        attention_probs = tf.reduce_mean(attention_probs, -1)

    return attention_probs


def _get_scores(
    log_probs,
    sequence_lengths,
    length_penalty_weight,
    coverage_penalty_weight,
    finished,
    accumulated_attention_probs,
):
    """Calculates scores for beam search hypotheses.

    Args:
      log_probs: The log probabilities with shape
        `[batch_size, beam_width, vocab_size]`.
      sequence_lengths: The array of sequence lengths.
      length_penalty_weight: Float weight to penalize length. Disabled with
        0.0.
      coverage_penalty_weight: Float weight to penalize the coverage of source
        sentence. Disabled with 0.0.
      finished: A boolean tensor of shape `[batch_size, beam_width]` that
        specifies which elements in the beam are finished already.
      accumulated_attention_probs: Accumulated attention probabilities up to
        the current time step, with shape `[batch_size, beam_width, max_time]`
        if coverage_penalty_weight is not 0.0.

    Returns:
      The scores normalized by the length_penalty and coverage_penalty.

    Raises:
      ValueError: accumulated_attention_probs is None when coverage penalty is
        enabled.
    """
    length_penalty_ = _length_penalty(
        sequence_lengths=sequence_lengths, penalty_factor=length_penalty_weight
    )
    length_penalty_ = tf.cast(length_penalty_, dtype=log_probs.dtype)
    scores = log_probs / length_penalty_

    coverage_penalty_weight = tf.convert_to_tensor(
        coverage_penalty_weight, name="coverage_penalty_weight"
    )
    if coverage_penalty_weight.shape.ndims != 0:
        raise ValueError(
            "coverage_penalty_weight should be a scalar, "
            "but saw shape: %s" % coverage_penalty_weight.shape
        )

    if tf.get_static_value(coverage_penalty_weight) == 0.0:
        return scores

    if accumulated_attention_probs is None:
        raise ValueError(
            "accumulated_attention_probs can be None only if coverage penalty "
            "is disabled."
        )

    # Add source sequence length mask before computing coverage penalty.
    accumulated_attention_probs = tf.where(
        tf.equal(accumulated_attention_probs, 0.0),
        tf.ones_like(accumulated_attention_probs),
        accumulated_attention_probs,
    )

    # coverage penalty =
    #     sum over `max_time` {log(min(accumulated_attention_probs, 1.0))}
    coverage_penalty = tf.reduce_sum(
        tf.math.log(tf.minimum(accumulated_attention_probs, 1.0)), 2
    )
    # Apply coverage penalty to finished predictions.
    coverage_penalty *= tf.cast(finished, tf.float32)
    weighted_coverage_penalty = coverage_penalty * coverage_penalty_weight
    # Reshape from [batch_size, beam_width] to [batch_size, beam_width, 1]
    weighted_coverage_penalty = tf.expand_dims(weighted_coverage_penalty, 2)
    return scores + weighted_coverage_penalty


def attention_probs_from_attn_state(attention_state):
    """Calculates the average attention probabilities.

    Args:
      attention_state: An instance of `AttentionWrapperState`.

    Returns:
      The attention probabilities in the given AttentionWrapperState.
      If there're multiple attention mechanisms, return the average value from
      all attention mechanisms.
    """
    # Attention probabilities over time steps, with shape
    # `[batch_size, beam_width, max_time]`.
    attention_probs = attention_state.alignments
    if isinstance(attention_probs, tuple):
        attention_probs = [tf.expand_dims(prob, -1) for prob in attention_probs]
        attention_probs = tf.concat(attention_probs, -1)
        attention_probs = tf.reduce_mean(attention_probs, -1)
    return attention_probs


def _length_penalty(sequence_lengths, penalty_factor):
    """Calculates the length penalty. See https://arxiv.org/abs/1609.08144.

    Returns the length penalty tensor:
    ```
    [(5+sequence_lengths)/6]**penalty_factor
    ```
    where all operations are performed element-wise.

    Args:
      sequence_lengths: `Tensor`, the sequence lengths of each hypotheses.
      penalty_factor: A scalar that weights the length penalty.

    Returns:
      If the penalty is `0`, returns the scalar `1.0`.  Otherwise returns
      the length penalty factor, a tensor with the same shape as
      `sequence_lengths`.
    """
    penalty_factor = tf.convert_to_tensor(penalty_factor, name="penalty_factor")
    penalty_factor.set_shape(())  # penalty should be a scalar.
    static_penalty = tf.get_static_value(penalty_factor)
    if static_penalty is not None and static_penalty == 0:
        return 1.0
    return tf.math.divide(
        (5.0 + tf.cast(sequence_lengths, tf.float32)) ** penalty_factor,
        (5.0 + 1.0) ** penalty_factor,
    )


def _mask_probs(probs, eos_token, finished):
    """Masks log probabilities.

    The result is that finished beams allocate all probability mass to eos and
    unfinished beams remain unchanged.

    Args:
      probs: Log probabilities of shape `[batch_size, beam_width, vocab_size]`
      eos_token: An int32 id corresponding to the EOS token to allocate
        probability to.
      finished: A boolean tensor of shape `[batch_size, beam_width]` that
        specifies which elements in the beam are finished already.

    Returns:
      A tensor of shape `[batch_size, beam_width, vocab_size]`, where
      unfinished beams stay unchanged and finished beams are replaced with a
      tensor with all probability on the EOS token.
    """
    vocab_size = tf.shape(probs)[2]
    # All finished examples are replaced with a vector that has all
    # probability on EOS
    finished_row = tf.one_hot(
        eos_token,
        vocab_size,
        dtype=probs.dtype,
        on_value=tf.convert_to_tensor(0.0, dtype=probs.dtype),
        off_value=probs.dtype.min,
    )
    finished_probs = tf.tile(
        tf.reshape(finished_row, [1, 1, -1]), tf.concat([tf.shape(finished), [1]], 0)
    )
    finished_mask = tf.tile(tf.expand_dims(finished, 2), [1, 1, vocab_size])

    return tf.where(finished_mask, finished_probs, probs)


def _maybe_tensor_gather_helper(
    gather_indices, gather_from, batch_size, range_size, gather_shape
):
    """Maybe applies _tensor_gather_helper.

    This applies _tensor_gather_helper when the gather_from dims is at least as
    big as the length of gather_shape. This is used in conjunction with nest so
    that we don't apply _tensor_gather_helper to inapplicable values like
    scalars.

    Args:
      gather_indices: The tensor indices that we use to gather.
      gather_from: The tensor that we are gathering from.
      batch_size: The batch size.
      range_size: The number of values in each range. Likely equal to
        beam_width.
      gather_shape: What we should reshape gather_from to in order to preserve
        the correct values. An example is when gather_from is the attention
        from an AttentionWrapperState with shape
        [batch_size, beam_width, attention_size]. There, we want to preserve
        the attention_size elements, so gather_shape is
        [batch_size * beam_width, -1]. Then, upon reshape, we still have the
        attention_size as desired.

    Returns:
      output: Gathered tensor of shape
        tf.shape(gather_from)[:1+len(gather_shape)] or the original tensor if
        its dimensions are too small.
    """
    if isinstance(gather_from, tf.TensorArray):
        return gather_from
    _check_ndims(gather_from)
    if gather_from.shape.ndims >= len(gather_shape):
        return _tensor_gather_helper(
            gather_indices=gather_indices,
            gather_from=gather_from,
            batch_size=batch_size,
            range_size=range_size,
            gather_shape=gather_shape,
        )
    else:
        return gather_from


def _tensor_gather_helper(
    gather_indices, gather_from, batch_size, range_size, gather_shape, name=None
):
    """Helper for gathering the right indices from the tensor.

    This works by reshaping gather_from to gather_shape (e.g. [-1]) and then
    gathering from that according to the gather_indices, which are offset by
    the right amounts in order to preserve the batch order.

    Args:
      gather_indices: The tensor indices that we use to gather.
      gather_from: The tensor that we are gathering from.
      batch_size: The input batch size.
      range_size: The number of values in each range. Likely equal to
        beam_width.
      gather_shape: What we should reshape gather_from to in order to preserve
        the correct values. An example is when gather_from is the attention
        from an AttentionWrapperState with shape
        [batch_size, beam_width, attention_size]. There, we want to preserve
        the attention_size elements, so gather_shape is
        [batch_size * beam_width, -1]. Then, upon reshape, we still have the
        attention_size as desired.
      name: The tensor name for set of operations. By default this is
        'tensor_gather_helper'. The final output is named 'output'.

    Returns:
      output: Gathered tensor of shape
        tf.shape(gather_from)[:1+len(gather_shape)]
    """
    with tf.name_scope(name or "tensor_gather_helper"):
        range_ = tf.expand_dims(tf.range(batch_size) * range_size, 1)
        gather_indices = tf.reshape(gather_indices + range_, [-1])
        output = tf.gather(tf.reshape(gather_from, gather_shape), gather_indices)
        final_shape = tf.shape(gather_from)[: 1 + len(gather_shape)]
        static_batch_size = tf.get_static_value(batch_size)
        final_static_shape = tf.TensorShape([static_batch_size]).concatenate(
            gather_from.shape[1 : 1 + len(gather_shape)]
        )
        output = tf.reshape(output, final_shape, name="output")
        output.set_shape(final_static_shape)
        return output
