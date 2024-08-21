# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Objects sampling from the decoder output distribution and producing the next input."""

import abc

from typing import Callable, Optional

import tensorflow as tf

from typeguard import typechecked

from opennmt.tfa.seq2seq import decoder
from opennmt.tfa.utils import types
from opennmt.tfa.utils.types import Initializer, TensorLike

_transpose_batch_time = decoder._transpose_batch_time


class Sampler(metaclass=abc.ABCMeta):
    """Interface for implementing sampling in seq2seq decoders.

    Sampler classes implement the logic of sampling from the decoder output distribution
    and producing the inputs for the next decoding step. In most cases, they should not be
    used directly but passed to a `tfa.seq2seq.BasicDecoder` instance that will manage the
    sampling.

    Here is an example using a training sampler directly to implement a custom decoding
    loop:

    >>> batch_size = 4
    >>> max_time = 7
    >>> hidden_size = 16
    >>>
    >>> sampler = tfa.seq2seq.TrainingSampler()
    >>> cell = tf.keras.layers.LSTMCell(hidden_size)
    >>>
    >>> input_tensors = tf.random.uniform([batch_size, max_time, hidden_size])
    >>> initial_finished, initial_inputs = sampler.initialize(input_tensors)
    >>>
    >>> cell_input = initial_inputs
    >>> cell_state = cell.get_initial_state(initial_inputs)
    >>>
    >>> for time_step in tf.range(max_time):
    ...     cell_output, cell_state = cell(cell_input, cell_state)
    ...     sample_ids = sampler.sample(time_step, cell_output, cell_state)
    ...     finished, cell_input, cell_state = sampler.next_inputs(
    ...         time_step, cell_output, cell_state, sample_ids)
    ...     if tf.reduce_all(finished):
    ...         break
    """

    @abc.abstractmethod
    def initialize(self, inputs, **kwargs):
        """initialize the sampler with the input tensors.

        This method must be invoked exactly once before calling other
        methods of the Sampler.

        Args:
          inputs: A (structure of) input tensors, it could be a nested tuple or
            a single tensor.
          **kwargs: Other kwargs for initialization. It could contain tensors
            like mask for inputs, or non tensor parameter.

        Returns:
          `(initial_finished, initial_inputs)`.
        """
        pass

    @abc.abstractmethod
    def sample(self, time, outputs, state):
        """Returns `sample_ids`."""
        pass

    @abc.abstractmethod
    def next_inputs(self, time, outputs, state, sample_ids):
        """Returns `(finished, next_inputs, next_state)`."""
        pass

    @abc.abstractproperty
    def batch_size(self):
        """Batch size of tensor returned by `sample`.

        Returns a scalar int32 tensor. The return value might not
        available before the invocation of initialize(), in this case,
        ValueError is raised.
        """
        raise NotImplementedError("batch_size has not been implemented")

    @abc.abstractproperty
    def sample_ids_shape(self):
        """Shape of tensor returned by `sample`, excluding the batch dimension.

        Returns a `TensorShape`. The return value might not available
        before the invocation of initialize().
        """
        raise NotImplementedError("sample_ids_shape has not been implemented")

    @abc.abstractproperty
    def sample_ids_dtype(self):
        """DType of tensor returned by `sample`.

        Returns a DType. The return value might not available before the
        invocation of initialize().
        """
        raise NotImplementedError("sample_ids_dtype has not been implemented")


class CustomSampler(Sampler):
    """Base abstract class that allows the user to customize sampling."""

    @typechecked
    def __init__(
        self,
        initialize_fn: Initializer,
        sample_fn: Callable,
        next_inputs_fn: Callable,
        sample_ids_shape: Optional[TensorLike] = None,
        sample_ids_dtype: types.AcceptableDTypes = None,
    ):
        """Initializer.

        Args:
          initialize_fn: callable that returns `(finished, next_inputs)` for
            the first iteration.
          sample_fn: callable that takes `(time, outputs, state)` and emits
            tensor `sample_ids`.
          next_inputs_fn: callable that takes
            `(time, outputs, state, sample_ids)` and emits
            `(finished, next_inputs, next_state)`.
          sample_ids_shape: Either a list of integers, or a 1-D Tensor of type
            `int32`, the shape of each value in the `sample_ids` batch.
            Defaults to a scalar.
          sample_ids_dtype: The dtype of the `sample_ids` tensor. Defaults to
            int32.
        """
        self._initialize_fn = initialize_fn
        self._sample_fn = sample_fn
        self._next_inputs_fn = next_inputs_fn
        self._batch_size = None
        self._sample_ids_shape = tf.TensorShape(sample_ids_shape or [])
        self._sample_ids_dtype = sample_ids_dtype or tf.int32

    @property
    def batch_size(self):
        if self._batch_size is None:
            raise ValueError("batch_size accessed before initialize was called")
        return self._batch_size

    @property
    def sample_ids_shape(self):
        return self._sample_ids_shape

    @property
    def sample_ids_dtype(self):
        return self._sample_ids_dtype

    def initialize(self, inputs, **kwargs):
        (finished, next_inputs) = self._initialize_fn(inputs, **kwargs)
        if self._batch_size is None:
            self._batch_size = tf.size(finished)
        return (finished, next_inputs)

    def sample(self, time, outputs, state):
        return self._sample_fn(time=time, outputs=outputs, state=state)

    def next_inputs(self, time, outputs, state, sample_ids):
        return self._next_inputs_fn(
            time=time, outputs=outputs, state=state, sample_ids=sample_ids
        )


class TrainingSampler(Sampler):
    """A training sampler that simply reads its inputs.

    Returned sample_ids are the argmax of the RNN output logits.
    """

    @typechecked
    def __init__(self, time_major: bool = False):
        """Initializer.

        Args:
          time_major: Python bool.  Whether the tensors in `inputs` are time
            major. If `False` (default), they are assumed to be batch major.

        Raises:
          ValueError: if `sequence_length` is not a 1D tensor or `mask` is
            not a 2D boolean tensor.
        """
        self.time_major = time_major
        self._batch_size = None

    @property
    def batch_size(self):
        if self._batch_size is None:
            raise ValueError("batch_size accessed before initialize was called")
        return self._batch_size

    @property
    def sample_ids_shape(self):
        return tf.TensorShape([])

    @property
    def sample_ids_dtype(self):
        return tf.int32

    def initialize(self, inputs, sequence_length=None, mask=None):
        """Initialize the TrainSampler.

        Args:
          inputs: A (structure of) input tensors.
          sequence_length: An int32 vector tensor.
          mask: A boolean 2D tensor.

        Returns:
          (finished, next_inputs), a tuple of two items. The first item is a
            boolean vector to indicate whether the item in the batch has
            finished. The second item is the first slide of input data based on
            the timestep dimension (usually the second dim of the input).
        """
        self.inputs = tf.convert_to_tensor(inputs, name="inputs")
        if not self.time_major:
            inputs = tf.nest.map_structure(_transpose_batch_time, inputs)

        self._batch_size = tf.shape(tf.nest.flatten(inputs)[0])[1]

        self.input_tas = tf.nest.map_structure(_unstack_ta, inputs)
        if sequence_length is not None and mask is not None:
            raise ValueError(
                "sequence_length and mask can't be provided at the same time."
            )
        if sequence_length is not None:
            self.sequence_length = tf.convert_to_tensor(
                sequence_length, name="sequence_length"
            )
            if self.sequence_length.shape.ndims != 1:
                raise ValueError(
                    "Expected sequence_length to be vector, but received "
                    "shape: %s" % self.sequence_length.shape
                )
        elif mask is not None:
            mask = tf.convert_to_tensor(mask)
            if mask.shape.ndims != 2:
                raise ValueError(
                    "Expected mask to a 2D tensor, but received shape: %s" % mask
                )
            if not mask.dtype.is_bool:
                raise ValueError(
                    "Expected mask to be a boolean tensor, but received "
                    "dtype: %s" % repr(mask.dtype)
                )

            axis = 1 if not self.time_major else 0
            with tf.control_dependencies(
                [_check_sequence_is_right_padded(mask, self.time_major)]
            ):
                self.sequence_length = tf.math.reduce_sum(
                    tf.cast(mask, tf.int32), axis=axis, name="sequence_length"
                )
        else:
            # As the input tensor has been converted to time major,
            # the maximum sequence length should be inferred from
            # the first dimension.
            max_seq_len = tf.shape(tf.nest.flatten(inputs)[0])[0]
            self.sequence_length = tf.fill(
                [self.batch_size], max_seq_len, name="sequence_length"
            )

        self.zero_inputs = tf.nest.map_structure(
            lambda inp: tf.zeros_like(inp[0, :]), inputs
        )

        finished = tf.equal(0, self.sequence_length)
        all_finished = tf.reduce_all(finished)
        next_inputs = tf.cond(
            all_finished,
            lambda: self.zero_inputs,
            lambda: tf.nest.map_structure(lambda inp: inp.read(0), self.input_tas),
        )
        return (finished, next_inputs)

    def sample(self, time, outputs, state):
        del state
        sample_ids = tf.cast(tf.argmax(outputs, axis=-1), tf.int32)
        return sample_ids

    def next_inputs(self, time, outputs, state, sample_ids):
        del sample_ids
        next_time = time + 1
        finished = next_time >= self.sequence_length
        all_finished = tf.reduce_all(finished)

        def read_from_ta(inp):
            return inp.read(next_time)

        next_inputs = tf.cond(
            all_finished,
            lambda: self.zero_inputs,
            lambda: tf.nest.map_structure(read_from_ta, self.input_tas),
        )
        return (finished, next_inputs, state)


class InferenceSampler(Sampler):
    """An inference sampler that uses a custom sampling function."""

    @typechecked
    def __init__(
        self,
        sample_fn: Callable,
        sample_shape: TensorLike,
        sample_dtype: types.AcceptableDTypes,
        end_fn: Callable,
        next_inputs_fn: Optional[Callable] = None,
    ):
        """Initializer.

        Args:
          sample_fn: A callable that takes `outputs` and emits tensor
            `sample_ids`.
          sample_shape: Either a list of integers, or a 1-D Tensor of type
            `int32`, the shape of the each sample in the batch returned by
            `sample_fn`.
          sample_dtype: the dtype of the sample returned by `sample_fn`.
          end_fn: A callable that takes `sample_ids` and emits a `bool` vector
            shaped `[batch_size]` indicating whether each sample is an end
            token.
          next_inputs_fn: (Optional) A callable that takes `sample_ids` and
            returns the next batch of inputs. If not provided, `sample_ids` is
            used as the next batch of inputs.
        """
        self.sample_fn = sample_fn
        self.sample_shape = tf.TensorShape(sample_shape)
        self.sample_dtype = sample_dtype
        self.end_fn = end_fn
        self.next_inputs_fn = next_inputs_fn
        self._batch_size = None

    @property
    def batch_size(self):
        if self._batch_size is None:
            raise ValueError("batch_size accessed before initialize was called")
        return self._batch_size

    @property
    def sample_ids_shape(self):
        return self.sample_shape

    @property
    def sample_ids_dtype(self):
        return self.sample_dtype

    def initialize(self, start_inputs):
        self.start_inputs = tf.convert_to_tensor(start_inputs, name="start_inputs")
        self._batch_size = tf.shape(start_inputs)[0]
        finished = tf.tile([False], [self._batch_size])
        return (finished, self.start_inputs)

    def sample(self, time, outputs, state):
        del time, state  # unused by sample
        return self.sample_fn(outputs)

    def next_inputs(self, time, outputs, state, sample_ids):
        del time, outputs  # unused by next_inputs
        if self.next_inputs_fn is None:
            next_inputs = sample_ids
        else:
            next_inputs = self.next_inputs_fn(sample_ids)
        finished = self.end_fn(sample_ids)
        return (finished, next_inputs, state)


# The following sample functions (_call_sampler, bernoulli_sample,
# categorical_sample) mimic TensorFlow Probability distribution semantics.
def _call_sampler(sample_n_fn, sample_shape, name=None):
    """Reshapes vector of samples."""
    with tf.name_scope(name or "call_sampler"):
        sample_shape = tf.convert_to_tensor(
            sample_shape, dtype=tf.int32, name="sample_shape"
        )
        # Ensure sample_shape is a vector (vs just a scalar).
        pad = tf.cast(tf.equal(tf.rank(sample_shape), 0), tf.int32)
        sample_shape = tf.reshape(
            sample_shape,
            tf.pad(tf.shape(sample_shape), paddings=[[pad, 0]], constant_values=1),
        )
        samples = sample_n_fn(tf.reduce_prod(sample_shape))
        batch_event_shape = tf.shape(samples)[1:]
        final_shape = tf.concat([sample_shape, batch_event_shape], 0)
        return tf.reshape(samples, final_shape)


def bernoulli_sample(
    probs=None, logits=None, dtype=tf.int32, sample_shape=(), seed=None
):
    """Samples from Bernoulli distribution."""
    if probs is None:
        probs = tf.sigmoid(logits, name="probs")
    else:
        probs = tf.convert_to_tensor(probs, name="probs")
    batch_shape_tensor = tf.shape(probs)

    def _sample_n(n):
        """Sample vector of Bernoullis."""
        new_shape = tf.concat([[n], batch_shape_tensor], 0)
        uniform = tf.random.uniform(new_shape, seed=seed, dtype=probs.dtype)
        return tf.cast(tf.less(uniform, probs), dtype)

    return _call_sampler(_sample_n, sample_shape)


def categorical_sample(logits, dtype=tf.int32, sample_shape=(), seed=None):
    """Samples from categorical distribution."""
    logits = tf.convert_to_tensor(logits, name="logits")
    event_size = tf.shape(logits)[-1]
    batch_shape_tensor = tf.shape(logits)[:-1]

    def _sample_n(n):
        """Sample vector of categoricals."""
        if logits.shape.ndims == 2:
            logits_2d = logits
        else:
            logits_2d = tf.reshape(logits, [-1, event_size])
        sample_dtype = tf.int64 if logits.dtype.size > 4 else tf.int32
        draws = tf.random.categorical(logits_2d, n, dtype=sample_dtype, seed=seed)
        draws = tf.reshape(tf.transpose(draws), tf.concat([[n], batch_shape_tensor], 0))
        return tf.cast(draws, dtype)

    return _call_sampler(_sample_n, sample_shape)


def _unstack_ta(inp):
    return tf.TensorArray(
        dtype=inp.dtype, size=tf.shape(inp)[0], element_shape=inp.shape[1:]
    ).unstack(inp)


def _check_sequence_is_right_padded(mask, time_major):
    """Returns an Assert operation checking that if the mask tensor is right
    padded."""
    if time_major:
        mask = tf.transpose(mask)
    sequence_length = tf.math.reduce_sum(tf.cast(mask, tf.int32), axis=1)
    max_seq_length = tf.shape(mask)[1]
    right_padded_mask = tf.sequence_mask(
        sequence_length, maxlen=max_seq_length, dtype=tf.bool
    )
    all_equal = tf.math.equal(mask, right_padded_mask)

    condition = tf.math.reduce_all(all_equal)
    error_message = "The input sequence should be right padded."

    return tf.Assert(condition, [error_message])
