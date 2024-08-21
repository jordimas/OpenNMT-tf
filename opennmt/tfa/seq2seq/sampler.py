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
