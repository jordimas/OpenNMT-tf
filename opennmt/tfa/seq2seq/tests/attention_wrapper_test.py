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
"""Tests for tfa.seq2seq.attention_wrapper."""

import collections

import numpy as np
import pytest
import tensorflow as tf

from packaging.version import Version

from opennmt.tfa.seq2seq import attention_wrapper as wrapper
from opennmt.tfa.seq2seq import sampler as sampler_py


class DummyData:
    def __init__(self):
        self.batch = 10
        self.timestep = 5
        self.memory_size = 6
        self.units = 8

        self.memory = np.random.randn(
            self.batch, self.timestep, self.memory_size
        ).astype(np.float32)
        self.memory_length = np.random.randint(
            low=1, high=self.timestep + 1, size=(self.batch,)
        )
        self.query = np.random.randn(self.batch, self.units).astype(np.float32)
        self.state = np.random.randn(self.batch, self.timestep).astype(np.float32)


attention_classes = [
    wrapper.LuongAttention,
]


@pytest.mark.parametrize("attention_cls", attention_classes)
def test_attention_shape_inference(attention_cls):
    dummy_data = DummyData()
    attention = attention_cls(dummy_data.units, dummy_data.memory)
    attention_score = attention([dummy_data.query, dummy_data.state])
    assert len(attention_score) == 2
    assert attention_score[0].shape == (dummy_data.batch, dummy_data.timestep)
    assert attention_score[1].shape == (dummy_data.batch, dummy_data.timestep)


@pytest.mark.parametrize("attention_cls", attention_classes)
def test_get_config(attention_cls):
    dummy_data = DummyData()
    attention = attention_cls(dummy_data.units, dummy_data.memory)
    config = attention.get_config()

    attention_from_config = attention_cls.from_config(config)
    config_from_clone = attention_from_config.get_config()

    assert config == config_from_clone


@pytest.mark.parametrize("attention_cls", attention_classes)
def test_layer_output(attention_cls):
    dummy_data = DummyData()
    attention = attention_cls(dummy_data.units, dummy_data.memory)
    score = attention([dummy_data.query, dummy_data.state])

    assert len(score) == 2
    assert score[0].shape == (dummy_data.batch, dummy_data.timestep)
    assert score[1].shape == (dummy_data.batch, dummy_data.timestep)


@pytest.mark.parametrize("attention_cls", attention_classes)
def test_passing_memory_from_call(attention_cls):
    dummy_data = DummyData()
    attention = attention_cls(dummy_data.units, dummy_data.memory)
    weights_before_query = attention.get_weights()
    ref_score = attention([dummy_data.query, dummy_data.state])

    all_weights = attention.get_weights()
    config = attention.get_config()
    # Simulate the twice invocation of calls here.
    attention_from_config = attention_cls.from_config(config)
    attention_from_config.build(dummy_data.memory.shape)
    attention_from_config.set_weights(weights_before_query)
    attention_from_config(dummy_data.memory, setup_memory=True)
    attention_from_config.build([dummy_data.query.shape, dummy_data.state.shape])
    attention_from_config.set_weights(all_weights)
    score = attention_from_config([dummy_data.query, dummy_data.state])

    np.testing.assert_allclose(ref_score, score)


@pytest.mark.parametrize("attention_cls", attention_classes)
def test_save_load_layer(attention_cls):
    dummy_data = DummyData()
    vocab = 20
    embedding_dim = 6
    inputs = tf.keras.Input(shape=[dummy_data.timestep])
    encoder_input = tf.keras.layers.Embedding(vocab, embedding_dim, mask_zero=True)(
        inputs
    )
    encoder_output = tf.keras.layers.LSTM(
        dummy_data.memory_size, return_sequences=True
    )(encoder_input)

    attention = attention_cls(dummy_data.units, encoder_output)
    query = tf.keras.Input(shape=[dummy_data.units])
    state = tf.keras.Input(shape=[dummy_data.timestep])

    score = attention([query, state])

    x_test = np.random.randint(vocab, size=(dummy_data.batch, dummy_data.timestep))
    model = tf.keras.Model([inputs, query, state], score)
    # Fall back to v1 style Keras training loop until issue with
    # using outputs of a layer in another layer's constructor.
    model.compile("rmsprop", "mse")
    y_ref = model.predict_on_batch([x_test, dummy_data.query, dummy_data.state])

    if Version(tf.__version__) >= Version("2.13"):
        model.use_legacy_config = True

    config = model.get_config()
    weights = model.get_weights()
    loaded_model = tf.keras.Model.from_config(
        config, custom_objects={attention_cls.__name__: attention_cls}
    )
    loaded_model.set_weights(weights)

    # Fall back to v1 style Keras training loop until issue with
    # using outputs of a layer in another layer's constructor.
    loaded_model.compile("rmsprop", "mse")

    y = loaded_model.predict_on_batch([x_test, dummy_data.query, dummy_data.state])

    np.testing.assert_allclose(y_ref, y)


@pytest.mark.parametrize("attention_cls", attention_classes)
def test_manual_memory_reset(attention_cls):
    dummy_data = DummyData()
    attention = attention_cls(dummy_data.units)

    def _compute_score(batch_size=None):
        if batch_size is None:
            batch_size = dummy_data.batch
        memory = dummy_data.memory[:batch_size]
        attention.setup_memory(
            memory, memory_sequence_length=dummy_data.memory_length[:batch_size]
        )
        assert attention.values.shape.as_list() == list(memory.shape)
        assert attention.keys.shape.as_list() == list(memory.shape)[:-1] + [
            dummy_data.units
        ]
        return attention([dummy_data.query[:batch_size], dummy_data.state[:batch_size]])

    _compute_score(batch_size=dummy_data.batch)
    variables = list(attention.variables)
    _compute_score(batch_size=dummy_data.batch - 1)

    # No new variables were created.
    for var_1, var_2 in zip(variables, list(attention.variables)):
        assert var_1 is var_2


def test_masking():
    memory = tf.ones([4, 4, 5], dtype=tf.float32)
    memory_sequence_length = tf.constant([1, 2, 3, 4], dtype=tf.int32)
    query = tf.ones([4, 5], dtype=tf.float32)
    state = None
    attention = wrapper.LuongAttention(5, memory, memory_sequence_length)
    alignment, _ = attention([query, state])
    assert np.sum(np.triu(alignment, k=1)) == 0


@pytest.mark.parametrize("attention_cls", attention_classes)
def test_memory_re_setup(attention_cls):
    class MyModel(tf.keras.models.Model):
        def __init__(self, vocab, embedding_dim, memory_size, units):
            super().__init__()
            self.emb = tf.keras.layers.Embedding(vocab, embedding_dim, mask_zero=True)
            self.encoder = tf.keras.layers.LSTM(memory_size, return_sequences=True)
            self.attn_mch = attention_cls(units)

        def call(self, inputs):
            enc_input, query, state = inputs
            mask = self.emb.compute_mask(enc_input)
            enc_input = self.emb(enc_input)
            enc_output = self.encoder(enc_input, mask=mask)
            # To ensure manual resetting also works in the graph mode,
            # we call the attention mechanism twice.
            self.attn_mch(enc_output, mask=mask, setup_memory=True)
            self.attn_mch(enc_output, mask=mask, setup_memory=True)
            score = self.attn_mch([query, state])
            return score

    vocab = 20
    embedding_dim = 6
    num_batches = 5

    dummy_data = DummyData()
    model = MyModel(vocab, embedding_dim, dummy_data.memory_size, dummy_data.units)
    model.compile("rmsprop", "mse")

    x = np.random.randint(
        vocab, size=(num_batches * dummy_data.batch, dummy_data.timestep)
    )
    x_test = np.random.randint(
        vocab, size=(num_batches * dummy_data.batch, dummy_data.timestep)
    )
    y = np.random.randn(num_batches * dummy_data.batch, dummy_data.timestep)

    query = np.tile(dummy_data.query, [num_batches, 1])
    state = np.tile(dummy_data.state, [num_batches, 1])

    model.fit([x, query, state], (y, y), batch_size=dummy_data.batch)
    model.predict_on_batch([x_test, query, state])


class ResultSummary(
    collections.namedtuple("ResultSummary", ("shape", "dtype", "mean"))
):
    pass


def get_result_summary(x):
    if isinstance(x, np.ndarray):
        return ResultSummary(x.shape, x.dtype, x.mean())
    return x


def assert_allclose_or_equal(x, y, **kwargs):
    if isinstance(x, np.ndarray) or isinstance(x, float):
        np.testing.assert_allclose(x, y, atol=1e-3, **kwargs)
    else:
        assert x == y


class DummyData2:
    def __init__(self):
        self.batch = 64
        self.units = 128
        self.encoder_timestep = 10
        self.encoder_dim = 256
        self.decoder_timestep = 12
        self.encoder_outputs = np.random.randn(
            self.batch, self.encoder_timestep, self.encoder_dim
        )
        self.encoder_sequence_length = np.random.randint(
            1, high=self.encoder_timestep, size=(self.batch,)
        ).astype(np.int32)
        self.decoder_inputs = np.random.randn(
            self.batch, self.decoder_timestep, self.units
        )
        self.decoder_sequence_length = np.random.randint(
            self.decoder_timestep, size=(self.batch,)
        ).astype(np.int32)


def test_custom_attention_layer():
    dummy_data = DummyData2()
    attention_mechanism = wrapper.LuongAttention(dummy_data.units)
    cell = tf.keras.layers.LSTMCell(dummy_data.units)
    attention_layer = tf.keras.layers.Dense(
        dummy_data.units * 2, use_bias=False, activation=tf.math.tanh
    )
    attention_wrapper = wrapper.AttentionWrapper(
        cell, attention_mechanism, attention_layer=attention_layer
    )
    with pytest.raises(ValueError):
        # Should fail because the attention mechanism has not been
        # initialized.
        attention_wrapper.get_initial_state(
            batch_size=dummy_data.batch, dtype=tf.float32
        )
    attention_mechanism.setup_memory(
        dummy_data.encoder_outputs.astype(np.float32),
        memory_sequence_length=dummy_data.encoder_sequence_length,
    )
    initial_state = attention_wrapper.get_initial_state(
        batch_size=dummy_data.batch, dtype=tf.float32
    )
    assert initial_state.attention.shape[-1] == dummy_data.units * 2
    first_input = dummy_data.decoder_inputs[:, 0].astype(np.float32)
    output, _ = attention_wrapper(first_input, initial_state)
    assert output.shape[-1] == dummy_data.units * 2


def set_random_state_for_tf_and_np():
    """Since the results of the tests have been hardcoded, we need to make sure,
    when we refactor code that the random state is the same. Meaning that all
    random functions should be called in the same order.
    """
    tf.random.set_seed(87654321)
    np.random.seed(87654321)
    DummyData2()


def test_attention_state_with_keras_rnn():
    # See https://github.com/tensorflow/addons/issues/1095.
    cell = tf.keras.layers.LSTMCell(8)

    mechanism = wrapper.LuongAttention(units=8, memory=tf.ones((2, 4, 8)))

    cell = wrapper.AttentionWrapper(cell=cell, attention_mechanism=mechanism)

    layer = tf.keras.layers.RNN(cell)
    _ = layer(inputs=tf.ones((2, 4, 8)))

    # Make sure the explicit initial_state also works.
    initial_state = cell.get_initial_state(batch_size=2, dtype=tf.float32)
    _ = layer(inputs=tf.ones((2, 4, 8)), initial_state=initial_state)


def test_attention_state_with_variable_length_input():
    cell = tf.keras.layers.LSTMCell(3)
    mechanism = wrapper.LuongAttention(units=3)
    cell = wrapper.AttentionWrapper(cell, mechanism)

    var_len = tf.random.uniform(shape=(), minval=2, maxval=10, dtype=tf.int32)
    lengths = tf.random.uniform(
        shape=(var_len,), minval=1, maxval=var_len + 1, dtype=tf.int32
    )
    data = tf.ones(shape=(var_len, var_len, 3))
    mask = tf.sequence_mask(lengths, maxlen=var_len)

    mechanism.setup_memory(data)
    layer = tf.keras.layers.RNN(cell)

    _ = layer(data, mask=mask)


def test_attention_wrapper_with_gru_cell():
    mechanism = wrapper.LuongAttention(units=3)
    cell = tf.keras.layers.GRUCell(3)
    cell = wrapper.AttentionWrapper(cell, mechanism)
    memory = tf.ones([2, 5, 3])
    inputs = tf.ones([2, 3])
    mechanism.setup_memory(memory)
    initial_state = cell.get_initial_state(inputs=inputs)
    _, state = cell(inputs, initial_state)
    tf.nest.assert_same_structure(initial_state, state)


def test_attention_wrapper_with_multiple_attention_mechanisms():
    cell = tf.keras.layers.LSTMCell(5)
    mechanisms = [wrapper.LuongAttention(units=3), wrapper.LuongAttention(units=3)]
    # We simply test that the wrapper creation makes no error.
    wrapper.AttentionWrapper(cell, mechanisms, attention_layer_size=[4, 5])
    wrapper.AttentionWrapper(
        cell,
        mechanisms,
        attention_layer=[tf.keras.layers.Dense(4), tf.keras.layers.Dense(5)],
    )
