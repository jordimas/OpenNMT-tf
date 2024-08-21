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
"""Utilities for tf.keras."""

import tensorflow as tf


def assert_like_rnncell(cell_name, cell):
    """Raises a TypeError if cell is not like a
    tf.keras.layers.AbstractRNNCell.

    Args:
      cell_name: A string to give a meaningful error referencing to the name
        of the function argument.
      cell: The object which should behave like a
        tf.keras.layers.AbstractRNNCell.

    Raises:
      TypeError: A human-friendly exception.
    """
    conditions = [
        _hasattr(cell, "output_size"),
        _hasattr(cell, "state_size"),
        _hasattr(cell, "get_initial_state"),
        callable(cell),
    ]

    errors = [
        "'output_size' property is missing",
        "'state_size' property is missing",
        "'get_initial_state' method is required",
        "is not callable",
    ]

    if not all(conditions):
        errors = [error for error, cond in zip(errors, conditions) if not cond]
        raise TypeError(
            "The argument {!r} ({}) is not an RNNCell: {}.".format(
                cell_name, cell, ", ".join(errors)
            )
        )


def _hasattr(obj, attr_name):
    # If possible, avoid retrieving the attribute as the object might run some
    # lazy computation in it.
    if attr_name in dir(obj):
        return True
    try:
        getattr(obj, attr_name)
    except AttributeError:
        return False
    else:
        return True
