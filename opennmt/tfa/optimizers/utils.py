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
"""Additional Utilities used for tfa.optimizers."""

import re

from typing import List

import tensorflow as tf


def get_variable_name(variable) -> str:
    """Get the variable name from the variable tensor."""
    param_name = variable.name
    m = re.match("^(.*):\\d+$", param_name)
    if m is not None:
        param_name = m.group(1)
    return param_name


def is_variable_matched_by_regexes(variable, regexes: List[str]) -> bool:
    """Whether variable is matched in regexes list by its name."""
    if regexes:
        # var_name = get_variable_name(variable)
        var_name = variable.name
        for r in regexes:
            if re.search(r, var_name):
                return True
    return False
