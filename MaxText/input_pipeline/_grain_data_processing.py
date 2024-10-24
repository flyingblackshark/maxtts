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

"""Input pipeline using Grain."""

import glob

import ml_collections
import jax
import grain.python as grain
import grain.python_lazy_dataset as grain_lazy
from input_pipeline import _input_pipeline_utils
from input_pipeline import _grain_tokenizer
from grain._src.python import options as grain_options
import multihost_dataloading

def get_datasets(data_file_pattern):
  """Load dataset from array_record files for using with grain"""
  data_files = glob.glob(data_file_pattern)
  dataset = grain.ArrayRecordDataSource(data_files)
  return dataset

def preprocessing_pipeline(
    dataset,
    speaker_pattern,
    tokenizer_path,
    global_batch_size: int,
    global_mesh,
    max_target_length: int,
    grain_worker_count: int,
    dataloading_host_index,
    dataloading_host_count,
    data_column,
    shuffle: bool = False,
    data_shuffle_seed=0,
    tokenize=False,
    add_bos=False,
    add_eos=False,
    num_epochs=None,
    packing=False,
    shift=False,
    drop_remainder=True,
):
  """Use grain to pre-process the dataset and return iterators"""
  assert global_batch_size % global_mesh.size == 0, "Batch size should be divisible number of global devices."

  all_ds = []
  speaker_files = glob.glob(speaker_pattern)
  parse_transform = _input_pipeline_utils.ParseTextAndSemanticFeatures()
  combine_transform = _input_pipeline_utils.LogicalCombineSegment()
  create_token_transform = _input_pipeline_utils.CreateToken(codebook_dim=9)
  length_struct = {"inputs": max_target_length, "targets": max_target_length,"input_semantics_mask": max_target_length, "targets_semantics_mask": max_target_length}
  for ds in speaker_files:
    speaker_dataset = grain.ArrayRecordDataSource(ds)
    speaker_dataset = grain_lazy.SourceLazyMapDataset(speaker_dataset).seed(data_shuffle_seed)
    speaker_dataset = grain_lazy.RepeatLazyMapDataset(speaker_dataset,num_epochs=None)
    speaker_dataset = grain_lazy.ShuffleLazyMapDataset(speaker_dataset)
    speaker_dataset = speaker_dataset.map(parse_transform)
    speaker_dataset = speaker_dataset.map(create_token_transform)
    speaker_dataset = grain_lazy.FirstFitPackLazyIterDataset(
      speaker_dataset,
      num_packing_bins=2,
      length_struct=length_struct,
      shuffle_bins=True,
      meta_features=("input_semantics_mask","targets_semantics_mask")
    )
    speaker_dataset = speaker_dataset.map(combine_transform)
    all_ds.append(speaker_dataset)
  dataset = grain_lazy.SourceLazyMapDataset(dataset).seed(data_shuffle_seed)
  dataset = grain_lazy.RepeatLazyMapDataset(dataset,num_epochs=None)
  dataset = grain_lazy.ShuffleLazyMapDataset(dataset)
  dataset = dataset.map(parse_transform)
  dataset = dataset.map(create_token_transform)
  dataset = grain_lazy.FirstFitPackLazyIterDataset(
      dataset,
      num_packing_bins=2,
      length_struct=length_struct,
      shuffle_bins=True,
      meta_features=("input_semantics_mask","targets_semantics_mask")
  )
  all_ds.append(dataset)
  dataset = grain_lazy.MixedLazyIterDataset(all_ds)
  

  dataset = dataset.batch(batch_size=global_batch_size // jax.process_count(),drop_remainder=drop_remainder)
  prefecth_options = grain_options.MultiprocessingOptions(num_workers=grain_worker_count,per_worker_buffer_size=128)
  dataset = dataset.prefetch(multiprocessing_options=prefecth_options)

  multihost_gen = multihost_dataloading.MultiHostDataLoadIterator(dataset, global_mesh)

  # Return multi-host jax.Array prep iterator
  return multihost_gen

def make_grain_iterator(
    config: ml_collections.ConfigDict,
    global_mesh,
    process_indices,
):
  """Load, preprocess dataset and return iterators"""
  train_ds = get_datasets(config.grain_train_files)
  train_iter = preprocessing_pipeline(
    dataset=train_ds,
    speaker_pattern=config.grain_train_speaker_files,
    tokenizer_path=config.tokenizer_path,
    global_batch_size=config.global_batch_size_to_load,
    global_mesh=global_mesh,
    max_target_length=config.max_target_length,
    grain_worker_count=config.grain_worker_count,
    dataloading_host_index=process_indices.index(jax.process_index()),
    dataloading_host_count=len(process_indices),
    data_column=config.train_data_column,
    shuffle=config.enable_data_shuffling,
    data_shuffle_seed=config.data_shuffle_seed,
    tokenize=config.tokenize_train_data,
    add_bos=config.add_bos,
    add_eos=config.add_eos,
  )

  if config.eval_interval > 0:
    eval_ds = get_datasets(config.grain_eval_files)
    eval_iter = preprocessing_pipeline(
      dataset=eval_ds,
      speaker_pattern=config.grain_train_speaker_files,
      tokenizer_path=config.tokenizer_path,
      global_batch_size=config.global_batch_size_to_load,
      global_mesh=global_mesh,
      max_target_length=config.max_target_length,
      grain_worker_count=config.grain_worker_count,
      dataloading_host_index=process_indices.index(jax.process_index()),
      dataloading_host_count=len(process_indices),
      data_column=config.eval_data_column,
      shuffle=False,
      data_shuffle_seed=config.data_shuffle_seed,
      tokenize=config.tokenize_eval_data,
      add_bos=config.add_bos,
      add_eos=config.add_eos,
    )
  else:
    eval_iter = None
  return train_iter, eval_iter
