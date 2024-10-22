
import grain.python as grain
import multihost_dataloading
import tensorflow as tf
import glob
from array_record.python.array_record_module import ArrayRecordWriter

class ReadFeatures(grain.MapTransform):
  """Normalize feature keys for HuggingFace input"""
  def map(self, features):
    parsed = tf.io.parse_example(features, {
      "text_tokens": tf.io.FixedLenFeature([], dtype=tf.string),
      "semantics_tokens": tf.io.FixedLenFeature([], dtype=tf.string),
      "speaker": tf.io.FixedLenFeature([], dtype=tf.string)
      })
    text_tokens = tf.io.parse_tensor(parsed["text_tokens"],tf.int32).numpy()
    semantics_tokens = tf.io.parse_tensor(parsed["semantics_tokens"],tf.int32).numpy()
    return {
        "text_tokens": text_tokens,
        "semantics_tokens": semantics_tokens,
        "speaker":parsed["speaker"],
    }   

if __name__ == "__main__":
    data_files = glob.glob("/bucket/new_dataset/hifi_tts/*.arrayrecord")
    dataset = grain.ArrayRecordDataSource(data_files)
    operations = []
    operations.append(ReadFeatures())

    dummy_index_sampler = grain.IndexSampler(
      num_records=len(dataset),
      num_epochs=1,
      shard_options=grain.ShardOptions(
          shard_index=0, shard_count=1, drop_remainder=True
      ),
      shuffle=False,
      seed=0,
    )

    dataloader = grain.DataLoader(
        data_source=dataset,
        operations=operations,
        sampler=dummy_index_sampler,
        worker_count=1,  # only supports one worker for now, more workers results in duplicated data
        worker_buffer_size=1,
        read_options=grain.ReadOptions(num_threads=1, prefetch_buffer_size=128),
    )
    


    i = 0
    writer_dict = {}
    for item in dataloader:
        speaker = item["speaker"].numpy().decode('UTF-8')
        print(f"round {i}")
        if speaker not in writer_dict:
            writer_dict[speaker] = ArrayRecordWriter(f"/bucket/speaker_dataset/speaker_{speaker}.arrayrecord", 'group_size:1')
            

        i+=1
        example = tf.train.Example(
                features=tf.train.Features(
                    feature={
                        'text_tokens': tf.train.Feature(
                            bytes_list=tf.train.BytesList(value=[tf.io.serialize_tensor(item["text_tokens"]).numpy()])),
                        'semantics_tokens':tf.train.Feature(
                            bytes_list=tf.train.BytesList(value=[tf.io.serialize_tensor(item["semantics_tokens"]).numpy()])
                        ),
                    }
                )
            )
        writer_dict[speaker].write(example.SerializeToString())
    # writer.close() 