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
"""Additional layers for sequence to sequence models."""

from opennmt.tfa.seq2seq.attention_wrapper import AttentionMechanism
from opennmt.tfa.seq2seq.attention_wrapper import AttentionWrapper
from opennmt.tfa.seq2seq.attention_wrapper import AttentionWrapperState
from opennmt.tfa.seq2seq.attention_wrapper import BahdanauAttention
from opennmt.tfa.seq2seq.attention_wrapper import BahdanauMonotonicAttention
from opennmt.tfa.seq2seq.attention_wrapper import LuongAttention
from opennmt.tfa.seq2seq.attention_wrapper import LuongMonotonicAttention
from opennmt.tfa.seq2seq.attention_wrapper import hardmax
from opennmt.tfa.seq2seq.attention_wrapper import monotonic_attention
from opennmt.tfa.seq2seq.attention_wrapper import safe_cumprod

from opennmt.tfa.seq2seq.basic_decoder import BasicDecoder
from opennmt.tfa.seq2seq.basic_decoder import BasicDecoderOutput

from opennmt.tfa.seq2seq.beam_search_decoder import BeamSearchDecoder
from opennmt.tfa.seq2seq.beam_search_decoder import BeamSearchDecoderOutput
from opennmt.tfa.seq2seq.beam_search_decoder import BeamSearchDecoderState
from opennmt.tfa.seq2seq.beam_search_decoder import FinalBeamSearchDecoderOutput
from opennmt.tfa.seq2seq.beam_search_decoder import gather_tree
from opennmt.tfa.seq2seq.beam_search_decoder import gather_tree_from_array
from opennmt.tfa.seq2seq.beam_search_decoder import tile_batch

from opennmt.tfa.seq2seq.decoder import BaseDecoder
from opennmt.tfa.seq2seq.decoder import Decoder
from opennmt.tfa.seq2seq.decoder import dynamic_decode

from opennmt.tfa.seq2seq.loss import SequenceLoss
from opennmt.tfa.seq2seq.loss import sequence_loss

from opennmt.tfa.seq2seq.sampler import CustomSampler
from opennmt.tfa.seq2seq.sampler import GreedyEmbeddingSampler
from opennmt.tfa.seq2seq.sampler import InferenceSampler
from opennmt.tfa.seq2seq.sampler import SampleEmbeddingSampler
from opennmt.tfa.seq2seq.sampler import Sampler
from opennmt.tfa.seq2seq.sampler import ScheduledEmbeddingTrainingSampler
from opennmt.tfa.seq2seq.sampler import ScheduledOutputTrainingSampler
from opennmt.tfa.seq2seq.sampler import TrainingSampler
