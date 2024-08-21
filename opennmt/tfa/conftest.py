import numpy as np
import pytest
import tensorflow as tf

import opennmt.tfa as tfa

from opennmt.tfa.utils.test_utils import (  # noqa: F401
    data_format,
    device,
    maybe_run_functions_eagerly,
    only_run_functions_eagerly,
    pytest_addoption,
    pytest_collection_modifyitems,
    pytest_configure,
    pytest_generate_tests,
    pytest_make_parametrize_id,
    run_custom_and_py_ops,
    run_with_mixed_precision_policy,
    set_global_variables,
    set_seeds,
)

# fixtures present in this file will be available
# when running tests and can be referenced with strings
# https://docs.pytest.org/en/latest/fixture.html#conftest-py-sharing-fixture-functions


@pytest.fixture(autouse=True)
def add_doctest_namespace(doctest_namespace):
    doctest_namespace["np"] = np
    doctest_namespace["tf"] = tf
    doctest_namespace["tfa"] = tfa
