# Copyright 2024 Google LLC
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

"""CLI utility for running inference on a single stream"""

from functools import partial
import jax

import max_utils
import maxengine

import os
import pyconfig
import sys
from transformers import AutoTokenizer
import jax.numpy as jnp
#import dac_jax
# soundfile as sf
from jax.experimental.compilation_cache import compilation_cache as cc
cc.set_cache_dir("./jax_cache")
CODEBOOK_PAD_TOKEN_ID = 0
def encode_tokens(
    tokenizer,
    string,
    prompt_tokens=None,
    num_codebooks=9,
):
    #string = clean_text(string)
    string = f"<|im_start|>user\n{string}<|im_end|><|im_start|>assistant\n"

    new_tokens = tokenizer.encode(
        string,
        add_special_tokens=False,
        max_length=10**6,
        truncation=False,
    )
    tokens = jnp.asarray([new_tokens], dtype=jnp.int32)
    true_length = tokens.shape[1]
    # Codebooks
    zeros = (
        jnp.ones((num_codebooks, tokens.shape[1]), dtype=jnp.int32)
        * CODEBOOK_PAD_TOKEN_ID
    )
    prompt = jnp.concatenate((tokens, zeros), axis=0)

    if prompt_tokens is None:
        return prompt,true_length

    # Get prompt tokens
    # if prompt_tokens.ndim == 3:
    #     assert (
    #         prompt_tokens.shape[0] == 1
    #     ), f"3 dim prompt tokens should have shape (1, num_codebooks, seq_len)"
    #     prompt_tokens = prompt_tokens[0]

    # assert prompt_tokens.ndim == 2
    data = prompt_tokens + 1

    # if prompt_tokens.shape[0] > num_codebooks:
    #     logger.warning(
    #         f"Prompt tokens shape {prompt_tokens.shape} is larger than num_codebooks {num_codebooks}, getting first {num_codebooks} codebooks"
    #     )
    #     data = data[:num_codebooks]

    # Add pad token for each codebook
    data = jnp.concatenate(
        (data, jnp.zeros((data.shape[0], 1), dtype=jnp.int32)),
        axis=1,
    )

    # Since 1.0, we use <|semantic|>
    s0_token_id = tokenizer.convert_tokens_to_ids("<|semantic|>")
    end_token_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    main_token_ids = (
        jnp.ones((1, data.shpae[1]), dtype=jnp.int32) * s0_token_id
    )
    main_token_ids[0, -1] = end_token_id

    data = jnp.concatenate((main_token_ids, data), axis=0)
    prompt = jnp.concatenate((prompt, data), axis=1)

    return prompt


def main(config):
    engine = maxengine.MaxEngine(config)
    params = engine.load_params()
    text = config.prompt
    #metadata = engine.get_tokenizer()

    tokenizer_model = AutoTokenizer.from_pretrained("fishaudio/fish-speech-1")
    im_end_id = tokenizer_model.convert_tokens_to_ids("<|im_end|>")
    tokens,true_length = encode_tokens(tokenizer_model,text)
    tokens = tokens.transpose(1,0)
    padding = config.max_prefill_predict_length - tokens.shape[0]
    padded_tokens = jnp.pad(tokens, ((0, padding),(0,0)))
    #tokenizer_model = engine.build_tokenizer(metadata)
    # tokens, true_length = tokenizer_model.encode(
    #     text, is_bos=True, prefill_lengths=[config.max_prefill_predict_length]
    # )
    #assert true_length <= config.max_prefill_predict_length, "can't take too many tokens"
    #assert config.quantization != "fp8", "fp8 on NVIDIA GPUs is not supported in decode.py yet"
    #base_params , codebook_params = params
    prefill_result, first_token = engine.prefill(params=params, padded_tokens=padded_tokens, true_length=true_length)
    slot = 0

    decode_state = engine.init_decode_state()
    decode_state = engine.insert(prefill_result, decode_state, slot=slot)

    steps = range(config.max_prefill_predict_length, config.max_target_length)
    sampled_tokens_list = []
    sampled_tokens_list.append(first_token)
    for _ in steps:
        decode_state, sampled_tokens = engine.generate(params, decode_state)
        if sampled_tokens.get_result_at_slot(slot).tokens[0].squeeze(0)[0] == im_end_id:
            break
        sampled_tokens_list.append(sampled_tokens)
    results = [sampled_tokens.get_result_at_slot(slot).tokens[0].squeeze(0) for sampled_tokens in sampled_tokens_list]
    results = jnp.stack(results,axis=0)[:,1:]   

    # model, variables = dac_jax.load_model(model_type="44khz")
    # @partial(jax.jit, static_argnums=(1, 2))
    # def decode_from_codes(codes: jnp.ndarray, scale, length: int = None):
    #     recons = model.apply(
    #         variables,
    #         codes,
    #         scale,
    #         length,
    #         method="decode",
    #     )
    #     return recons
    # audio_output = decode_from_codes(jnp.expand_dims(results.transpose(1,0),0)- 1,None).squeeze((0,1) )
    # sf.write("test.wav",audio_output,samplerate=44100)

def validate_config(config):
  assert config.load_full_state_path == "", (
      "Decode doesn't operate on full states! Convert to parameter checkpoint first." "Using generate_param_only_checkpoint."
  )


if __name__ == "__main__":
  jax.config.update("jax_default_prng_impl", "unsafe_rbg")
  os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"
  pyconfig.initialize(sys.argv)
  cfg = pyconfig.config
  validate_config(cfg)  
  max_utils.print_system_information()
  main(cfg)
