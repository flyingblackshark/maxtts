"""
Copyright 2023 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

     https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

"""Operations used by Grain"""

import dataclasses
import threading
import warnings
from typing import Dict
from threading import current_thread
import datasets
from datasets.distributed import split_dataset_by_node
import grain.python as grain
import numpy as np
import tensorflow as tf
import max_logging
import tokenizer
from transformers import AutoTokenizer
Features = Dict[str, tf.Tensor]
AUTOTUNE = tf.data.experimental.AUTOTUNE

########## Functions used by TFDS pipeline

def normalize_features(x, column_name):
  return {"inputs": x[column_name], "targets": x[column_name]}

def get_tokenizer(tokenizer_path, add_bos, add_eos):
  # Load tokenizer
  tokenizer_model = tokenizer.build_tokenizer(tokenizer_path, add_bos, add_eos)
  return tokenizer_model

def truncate_to_max_allowable_length(x, max_length):
  x["inputs"] = x["inputs"][:max_length]
  x["targets"] = x["targets"][:max_length]
  return x

def shift_data_by_truncation(x):
  x["inputs"] = x["inputs"][:-1]
  x["targets"] = x["targets"][1:]
  return x

########## Functions used by HF pipeline

def tokenization(example, hf_tokenizer, max_length, column_name):
  """Tokenize a HuggingFace dataset"""
  return hf_tokenizer(example[column_name], truncation=True, max_length=max_length)


@dataclasses.dataclass
class HFNormalizeFeatures(grain.MapTransform):
  """Normalize feature keys for HuggingFace input"""
  def __init__(self, column_name):
    self.column_name = column_name
  def map(self, features):
    return {
        "inputs": np.asarray(features[self.column_name], dtype=np.int32),
        "targets": np.asarray(features[self.column_name], dtype=np.int32),
    }


class HFDataSource(grain.RandomAccessDataSource):
  """A class that makes HuggingFace IterableDataset a grain datasource without random access support"""

  def __init__(self,
                dataset: datasets.IterableDataset,
                dataloading_host_index: int,
                dataloading_host_count: int,
                num_threads: int,
                generate_padding_example: bool,
                max_target_length: int,
                data_column_name: str
                ):
    self.dataset = dataset
    self.num_threads = num_threads
    self.dataloading_host_count = dataloading_host_count
    self.dataloading_host_index = dataloading_host_index
    self.generate_padding_example = generate_padding_example
    self.max_target_lenth = max_target_length
    self.data_column_name = data_column_name
    self.n_shards = dataset.n_shards
    self._check_shard_count()
    self.dataset_shards = [dataloading_host_index * self.num_threads + i for i in range(self.num_threads)]
    self.datasets = [split_dataset_by_node(dataset, world_size=self.n_shards, rank=x) for x in self.dataset_shards]
    self.data_iters = []
    self.out_of_data =False

  def _check_shard_count(self):
    if self.n_shards < (self.dataloading_host_count * self.num_threads):
      warnings.warn(f"WARNING: Inefficient dataloading. Your train or eval dataset contains {self.n_shards} shards, "
                      "smaller than number of host loading data. This is known to lead to inefficient dataloading. " 
                      "see https://github.com/google/maxtext/blob/main/getting_started/Data_Input_Pipeline.md#multihost-dataloading-best-practice"
                      )
      self.n_shards = self.dataloading_host_count * self.num_threads

  def _update_shard(self, idx):
    new_shard = self.dataset_shards[idx] + self.dataloading_host_count * self.num_threads
    if new_shard < self.n_shards:
      max_logging.log(f"Updating host {self.dataloading_host_index} dataset {idx}, was on shard {self.dataset_shards[idx]}")
      max_logging.log(f"New shard is {new_shard}")
      self.dataset_shards[idx] = new_shard
      self.datasets[idx] = split_dataset_by_node(self.dataset, world_size=self.n_shards, rank=self.dataset_shards[idx])
      self.data_iters[idx] = iter(self.datasets[idx])
    else:
      max_logging.log(f"Run out of shards on host {self.dataloading_host_index}, shard {self.dataset_shards[idx]} is not available")
      self.out_of_data = True
      if self.generate_padding_example:
        max_logging.log(f"Host {self.dataloading_host_index} will start generating all-0 padding examples until step number is met.")


  def __len__(self):
    """Return length of the HF dataset. Since HuggingFace IterableDataset does not have length,
    a fake length bigger than the dataset is returned"""
    return 10_000_000_000

  def __getitem__(self, index):
    """Since HuggingFace IterableDataset does not support random access by index.
    The next item in the iterator is returned."""
    if not self.data_iters:
      self.data_iters = [iter(x) for x in self.datasets]
    idx = int(current_thread().name.split("_")[1])

    while True:
      try:
        if self.out_of_data:
          if self.generate_padding_example:
            return {self.data_column_name: np.zeros(self.max_target_lenth, dtype=np.int32)}
          else:
            return None
        data = next(self.data_iters[idx])
        return data
      except StopIteration:
        self._update_shard(idx)

class RemoveTooLongElements(grain.FilterTransform):
  def __init__(self, max_length):
    self.max_length = max_length

  def filter(self, features) -> bool:
    return features["inputs"].shape[0] < self.max_length
  
@dataclasses.dataclass
class ParseTextAndSemanticFeatures(grain.MapTransform):
  def map(self, features):
    parsed = tf.io.parse_example(features, {
      "text_tokens": tf.io.FixedLenFeature([], dtype=tf.string),
      "semantics_tokens": tf.io.FixedLenFeature([], dtype=tf.string)
      })
    text_tokens = tf.io.parse_tensor(parsed["text_tokens"],tf.int32).numpy()
    semantics_tokens = tf.io.parse_tensor(parsed["semantics_tokens"],tf.int32).numpy().transpose(1,0)
    #codebook_targets = tf.io.parse_tensor(parsed["tokens"],tf.int64).numpy().transpose(1,0)[:-1]
    #prompt_length = parsed["prompt_length"].numpy()

    return {
        "text_tokens": text_tokens,
        "semantics_tokens": semantics_tokens,
}
@dataclasses.dataclass
class LogicalCombineSegment(grain.MapTransform):
  def map(self, features):
    inputs_segment_length = np.sum(features["inputs_segment_ids"]!=0)
    targets_segment_length = np.sum(features["targets_segment_ids"]!=0)
    features["inputs_segment_ids"][:inputs_segment_length] = np.ones(inputs_segment_length)
    features["targets_segment_ids"][:targets_segment_length] = np.ones(targets_segment_length)
    features["inputs_positions"][:inputs_segment_length] = np.arange(inputs_segment_length)
    features["targets_positions"][:inputs_segment_length] = np.arange(inputs_segment_length)
    return features
########## Functions used by Grain pipeline
@dataclasses.dataclass
class ParseAndNormalizeFeatures(grain.MapTransform):
  def map(self, features):
    parsed = tf.io.parse_example(features, {
      "tokens": tf.io.FixedLenFeature([], dtype=tf.string),
      "prompt_length": tf.io.FixedLenFeature([], dtype=tf.int64)
      })
    inputs = tf.io.parse_tensor(parsed["tokens"],tf.int64).numpy().transpose(1,0)[:-1]
    targets = tf.io.parse_tensor(parsed["tokens"],tf.int64).numpy().transpose(1,0)[1:]
    #codebook_targets = tf.io.parse_tensor(parsed["tokens"],tf.int64).numpy().transpose(1,0)[:-1]
    prompt_length = parsed["prompt_length"].numpy()

    return {
        "inputs": inputs,
        "targets": targets,
        "prompt_length":prompt_length,
    }



@dataclasses.dataclass
class ParseFeatures(grain.MapTransform):
  """Parse serialized example"""
  def __init__(self, data_column, tokenize):
    self.data_column = data_column
    if tokenize:
      self.dtype = tf.string
    else:
      self.dtype = tf.int64

  def map(self, features):
    def _parse(example):
      parsed = tf.io.parse_example(example, {
        self.data_column: tf.io.FixedLenSequenceFeature([], dtype=self.dtype, allow_missing=True)
        })
      return parsed

    return _parse(features)

@dataclasses.dataclass
class NormalizeFeatures(grain.MapTransform):
  """Normalize text feature keys."""
  def __init__(self, column_name, tokenize):
    self.column_name = column_name
    self.tokenize = tokenize

  def map(self, features):
    if self.tokenize:
      return {"inputs": features[self.column_name].numpy()[0].decode(), "targets": features[self.column_name].numpy()[0].decode()}
    else:
      return {"inputs": features[self.column_name].numpy(), "targets": features[self.column_name].numpy()}


@dataclasses.dataclass
class ReformatPacking(grain.MapTransform):
  """Reformat packing outputs."""

  def map(self, data):
    return {
        "inputs": data[0]["inputs"],
        "targets": data[0]["targets"],
        "inputs_segmentation": data[1]["inputs"],
        "targets_segmentation": data[1]["targets"],
        "inputs_position": data[2]["inputs"],
        "targets_position": data[2]["targets"],
    }

@dataclasses.dataclass
class CreateToken(grain.MapTransform):
  """Pads each input to the specified length"""
  # def __post_init__(self):
  #   self.text_tokenizer = None
  #   self._initialize_processor_lock = threading.Lock()

  def __init__(self,codebook_dim):
    self.codebook_dim = codebook_dim
    string_prefix = "<|im_start|>user\n"
    string_suffix = "<|im_end|><|im_start|>assistant\n"
    string_end = "<|im_end|>"
    # if self.text_tokenizer is None:
    #   with self._initialize_processor_lock:
    #     if self.text_tokenizer is None:  # Ensures only one thread initializes SPP.
    self.text_tokenizer = AutoTokenizer.from_pretrained("fishaudio/fish-speech-1")
    self.semantic_value = self.text_tokenizer.convert_tokens_to_ids("<|semantic|>")
    self.encoded_prefix = self.text_tokenizer.encode(
        string_prefix,
        add_special_tokens=False,
        max_length=10**6,
        truncation=False,
    )

    self.encoded_suffix = self.text_tokenizer.encode(
        string_suffix,
        add_special_tokens=False,
        max_length=10**6,
        truncation=False,
    )

    self.encoded_end = self.text_tokenizer.encode(
        string_end,
        add_special_tokens=False,
        max_length=10**6,
        truncation=False,
    )
    #self.semantic_value = semantic_value

  def map(self, data):
    text_tokens = data["text_tokens"]
    text_encoded = np.concatenate([self.encoded_prefix,text_tokens,self.encoded_suffix],axis=0)
    text_encoded = text_encoded[:,np.newaxis]
    padded_text_encoded = np.pad(text_encoded,((0,0),(0,self.codebook_dim)),mode="constant",constant_values=0)
    semantics_tokens = data["semantics_tokens"]
    padded_semantics_tokens = np.pad(semantics_tokens,((0,0),(1,0)),mode="constant",constant_values=self.semantic_value)
    padded_end_encoded = np.expand_dims(np.pad(self.encoded_end,((0,self.codebook_dim)),mode="constant",constant_values=0),0)
    encoded = np.concatenate([padded_text_encoded,padded_semantics_tokens,padded_end_encoded],axis=0)
    tokens = encoded
    semantics_mask = (np.arange(tokens.shape[0])>=text_encoded.shape[0]).astype(np.int32)
    semantics_mask[-1] = 0
    inputs = tokens[:-1]
    targets = tokens[1:]
    input_semantics_mask = semantics_mask[:-1]
    targets_semantics_mask = semantics_mask[1:]
    return {
        "inputs": inputs,
        "targets": targets,
        "input_semantics_mask":input_semantics_mask,
        "targets_semantics_mask":targets_semantics_mask,
        # "text_length": text_tokens.shape[0],
        # "semantics_length": semantics_tokens.shape[0],
    }

  
@dataclasses.dataclass
class PadToMaxLength(grain.MapTransform):
  """Pads each input to the specified length"""

  def __init__(self, max_length):
    self.max_length = max_length

  def map(self, data):
    """map to each element"""

    def _pad(x, max_length):
      pad_amount = max(max_length - x.shape[0], 0)
      pad_amount = [(0, pad_amount)] + [(0, 0)] * (len(x.shape) - 1)
      return np.pad(x, pad_amount)

    data["inputs_segmentation"] = np.ones(data["inputs"].shape[0], dtype=np.int32)
    data["inputs_position"] = np.arange(data["inputs"].shape[0], dtype=np.int32)
    data["targets_segmentation"] = np.ones(data["targets"].shape[0], dtype=np.int32)
    data["targets_position"] = np.arange(data["targets"].shape[0], dtype=np.int32)
    data["prompt_length"] = (data["inputs_position"] >= data["prompt_length"]-1).astype(np.int32)
    data["targets_segmentation_codebook"] = np.ones(data["inputs"].shape[0]-1, dtype=np.int32)
    for key, _ in data.items():
      data[key] = _pad(data[key], self.max_length)
    return data


def shift_right(x, axis=1):
  """Shift the input to the right by padding and slicing on axis."""
  pad_widths = [(0, 0)] * len(x.shape)
  pad_widths[axis] = (1, 0)
  slices = [
      slice(None),
  ] * len(x.shape)
  slices[axis] = slice(0, -1)
  padded = np.pad(x, pad_widths, mode="constant", constant_values=x.dtype.type(0))
  return padded[tuple(slices)]


def shift_and_refine(x, axis=1):
  """Shift inputs, set segmentation to 0 when target element is 0.
  Replace EOS by 0 for packed inputs."""
  x["inputs"] = shift_right(x["inputs"], axis=axis)
  targets_nonzero = x["targets"] != 0
  x["inputs_segmentation"] *= targets_nonzero
  x["targets_segmentation"] *= targets_nonzero
  # For packed targets, the first shifted token of a new sequence is made
  # 0, rather than being the EOS token for the last sequence.
  x["inputs"] *= x["inputs_segmentation"] == shift_right(x["inputs_segmentation"], axis=axis)

  return x


@dataclasses.dataclass
class ShiftData(grain.MapTransform):
  """Shift inputs and refine annotations."""

  def __init__(self, axis=1):
    self.axis = axis

  def map(self, data):
    return shift_and_refine(data, axis=self.axis)

