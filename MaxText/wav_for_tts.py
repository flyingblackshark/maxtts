
import transformers
from jax.experimental import mesh_utils
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from functools import partial

import jax
from jax import numpy as jnp
import numpy as np
import dac_jax
import tensorflow as tf
import argparse
import os
from transformers import WhisperProcessor
from whisper_jax import FlaxWhisperForConditionalGeneration
from array_record.python.array_record_module import ArrayRecordWriter
import librosa
# from jax.experimental.compilation_cache import compilation_cache as cc
# cc.set_cache_dir("./jax_cache")
MAX_LENGTH_AUDIO = 30 * 44100
MAX_LENGTH_AUDIO_16K = 30 * 16000
MAX_LENGTH_TEXT = 10000
GLOBAL_BATCH_SIZE = 64
CODEBOOK_PAD_TOKEN_ID = 0

def create_pair(semantics,text_tokens,n_frames,tokenizer):
    semantics_slice = semantics[:,:n_frames]
    string_prefix = "<|im_start|>user\n"
    string_suffix = "<|im_end|><|im_start|>smtc\n"

    encoded_prefix = tokenizer.encode(
        string_prefix,
        add_special_tokens=False,
        max_length=10**6,
        truncation=False,
    )

    encoded_suffix = tokenizer.encode(
        string_suffix,
        add_special_tokens=False,
        max_length=10**6,
        truncation=False,
    )
    encoded = encoded_prefix + np.asarray(text_tokens).tolist() + encoded_suffix
    num_codebooks = 9
    semantic_token_id = tokenizer.convert_tokens_to_ids("<|semantic|>")
    semantic_length = semantics_slice.shape[1]
    tokens = (
        encoded
        + [semantic_token_id] * semantic_length
        + tokenizer.convert_tokens_to_ids(["<|im_end|>"])
    )
    prompt_length = len(encoded)
    codes = [[CODEBOOK_PAD_TOKEN_ID] * prompt_length for _ in range(num_codebooks)]
    for book_idx, book in zip(range(num_codebooks), semantics_slice):
        for j in book:
            codes[book_idx].append(int(j) + 1)

    for book in codes:
        book.extend([CODEBOOK_PAD_TOKEN_ID] * 1)
    tokens = [tokens] + codes
    tokens = np.asarray(tokens)
    return (tokens,prompt_length)
def batch_process_tts(files,batch_size,outPath,wavPath,spks,mesh):
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        "fishaudio/fish-speech-1",
        add_bos_token=False,
        add_eos_token=False,
        model_max_length=15000,
        legacy=False,
        token="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",
    )
    whisper_processor = WhisperProcessor.from_pretrained("openai/whisper-large-v3")
    whispeer_model, whisper_params = FlaxWhisperForConditionalGeneration.from_pretrained(
        "openai/whisper-large-v3", dtype=jnp.bfloat16, _do_init=False,
    )
    x_sharding = NamedSharding(mesh,PartitionSpec('data'))
    i = 0
    dac_model, dac_variables = dac_jax.load_model(model_type="44khz",model_bitrate="16kbps")
    @partial(jax.jit, in_shardings=x_sharding,out_shardings=x_sharding)
    def encode_to_codes(x: jnp.ndarray):
        codes, scale = dac_model.apply(
            dac_variables,
            x,
            method="encode",
        )
        return codes, scale
    @partial(jax.jit, in_shardings=x_sharding,out_shardings=x_sharding)
    def whisper_generate_fn(input_features):
        pred_ids = whispeer_model.generate(
            input_features, task="transcribe", return_timestamps=False, max_length=whispeer_model.config.max_length, params=whisper_params,
        )
        return pred_ids.sequences
    batch_data = []
    batch_data_16k = []
    batch_length = []
    file_name_arr = []
    os.makedirs(f"{outPath}/{spks}",exist_ok=True)
    writer = ArrayRecordWriter(f"{outPath}/{spks}/part_0.arrayrecord", 'group_size:1')
    while i < len(files):
        print(f"{i+1}/{len(files)}")
        file = files[i][:-4]
        file_name_arr.append(file)
        wav, sr = librosa.load(f"{wavPath}/{spks}/{file}.wav", sr=44100, mono=True)
        wav_16k = librosa.resample(wav, orig_sr=sr, target_sr=16000)
        n_frame = wav.shape[0] // 512

        batch_length.append(n_frame)
        wav = np.pad(wav,(0,MAX_LENGTH_AUDIO-wav.shape[0]))
        wav_16k = np.pad(wav_16k,(0,MAX_LENGTH_AUDIO_16K-wav_16k.shape[0]))
        batch_data.append(wav)
        batch_data_16k.append(wav_16k)
        i+=1
        if len(batch_data) >= batch_size:
            batch_data = np.stack(batch_data)
            batch_data_16k = np.stack(batch_data_16k)
            #Genrate DAC Codec Codes
            batch_codes , _ = encode_to_codes(jnp.expand_dims(batch_data,1))
            batch_whisper_input_features = whisper_processor(batch_data_16k,sampling_rate=16000, return_tensors="np").input_features
            batch_pred_ids = whisper_generate_fn(batch_whisper_input_features)
            batch_pred_ids = np.asarray(batch_pred_ids)
            #transcribe audios
            batch_transcription = whisper_processor.batch_decode(batch_pred_ids, skip_special_tokens=True)
            for j in range(len(batch_transcription)):
                if batch_transcription[j][-1] not in ("!",",","?","."):
                    batch_transcription[j] = batch_transcription[j] + "."
            
            #tokenize
            batch_tokens = tokenizer(batch_transcription)['input_ids']

            for semantic,single_token,single_length in zip(batch_codes,batch_tokens,batch_length):

                final_token,prompt_length = partial(create_pair,tokenizer=tokenizer)(semantic,single_token,single_length)
                example = tf.train.Example(
                    features=tf.train.Features(
                        feature={
                            'tokens': tf.train.Feature(
                                bytes_list=tf.train.BytesList(value=[tf.io.serialize_tensor(final_token).numpy()])),
                            # 'targets': tf.train.Feature(
                            #     bytes_list=tf.train.BytesList(value=[tf.io.serialize_tensor(labels).numpy()])),
                            'prompt_length':tf.train.Feature(
                               int64_list=tf.train.Int64List(value=[prompt_length])
                            )
                            #'speaker':tf.train.Feature(bytes_list=tf.train.BytesList(value=[item["speaker"].encode('utf-8')]))
                        }
                    )
                )   
                writer.write(example.SerializeToString())

            batch_data_16k = []
            batch_data = []
            batch_length = []
            file_name_arr = []
    writer.close() 
if __name__ == "__main__":
    device_mesh = mesh_utils.create_device_mesh((4, 1))
    mesh = Mesh(device_mesh, axis_names=("data", "model")) 

    parser = argparse.ArgumentParser()
    parser.add_argument("-w", "--wav", help="wav", dest="wav", required=True)
    parser.add_argument("-o", "--out", help="out", dest="out", required=True)
    parser.add_argument("-bs", "--batch_size",type=int, default=4)

    args = parser.parse_args()
    device_mesh = mesh_utils.create_device_mesh((jax.local_device_count(),))
    mesh = Mesh(devices=device_mesh, axis_names=('data'))
    os.makedirs(args.out, exist_ok=True)
    wavPath = args.wav
    outPath = args.out
    batch_size = args.batch_size
    spk_files = {}
    for spks in os.listdir(wavPath):
        if os.path.isdir(f"{wavPath}/{spks}"):
            files = [f for f in os.listdir(f"{wavPath}/{spks}") if f.endswith(".wav")]
            batch_process_tts(files,batch_size,outPath,wavPath,spks,mesh)
