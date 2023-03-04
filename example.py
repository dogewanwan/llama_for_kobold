# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.

from typing import Tuple
import os
import sys
import torch
import fire
import time
import json
import logging
from datetime import datetime

from flask import Flask, request
from pathlib import Path

from fairscale.nn.model_parallel.initialize import initialize_model_parallel

from llama import ModelArgs, Transformer, Tokenizer, LLaMA

app = Flask(__name__)

@app.errorhandler(400)
def handle_400_error(e):
    app.logger.error(f"Bad request: {e}")
    return "Bad request", 400

def setup_model_parallel() -> Tuple[int, int]:
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    world_size = int(os.environ.get("WORLD_SIZE", -1))

    torch.distributed.init_process_group("gloo")
    initialize_model_parallel(world_size)
    torch.cuda.set_device(local_rank)

    # seed must be the same in all processes
    # torch.manual_seed(1)
    return local_rank, world_size


def load(
    ckpt_dir: str,
    tokenizer_path: str
) -> LLaMA:
    start_time = time.time()
    checkpoints = sorted(Path(ckpt_dir).glob("*.pth"))
    """
    assert world_size == len(
        checkpoints
    ), f"Loading a checkpoint for MP={len(checkpoints)} but world size is {world_size}"
    """

    """
    for k in checkpoints_data[0].keys():
        if "norm" not in k and "rope" not in k:
            checkpoints_data[0][k] = torch.cat(
                [d[k] for d in checkpoints_data], dim=key_to_dim[k.split(".")[-2]]
            )
        for d in checkpoints_data[1:]:
            del d[k]
    checkpoint = checkpoints_data[0]
    for i in range(1, len(checkpoints_data)):
        del checkpoints_data[i]
    """

    with open(Path(ckpt_dir) / "params.json", "r") as f:
        params = json.loads(f.read())

    model_args: ModelArgs = ModelArgs(max_seq_len=2048, max_batch_size=1, **params)
    tokenizer = Tokenizer(model_path=tokenizer_path)
    model_args.vocab_size = tokenizer.n_words
    torch.set_default_tensor_type(torch.cuda.HalfTensor)
    model = Transformer(model_args).cuda().half()
    torch.set_default_tensor_type(torch.FloatTensor)

    key_to_dim = {
        "w1": 0,
        "w2": -1,
        "w3": 0,
        "wo": -1,
        "wq": 0,
        "wk": 0,
        "wv": 0,
        "output": 0,
        "tok_embeddings": -1,
        "ffn_norm": None,
        "attention_norm": None,
        "norm": None,
        "rope": None,
    }

    # load the state dict incrementally, to avoid memory problems
    for i, ckpt in enumerate(checkpoints):
        print(f"Loading checkpoint {i}")
        checkpoint = torch.load(ckpt, map_location="cpu")
        for parameter_name, parameter in model.named_parameters():
            short_name = parameter_name.split(".")[-2]
            if key_to_dim[short_name] is None and i == 0:
                parameter.data = checkpoint[parameter_name]
            elif key_to_dim[short_name] == 0:
                size = checkpoint[parameter_name].size(0)
                parameter.data[size * i : size * (i + 1), :] = checkpoint[
                    parameter_name
                ]
            elif key_to_dim[short_name] == -1:
                size = checkpoint[parameter_name].size(-1)
                parameter.data[:, size * i : size * (i + 1)] = checkpoint[
                    parameter_name
                ]
        del checkpoint

    #model.load_state_dict(checkpoint, strict=False)

    generator = LLaMA(model, tokenizer)
    print(f"Loaded in {time.time() - start_time:.2f} seconds")
    return generator

generator = None

@app.route("/v1/engines", methods=["GET"])
def engines():
    return { "data": [{"ready": True, "id": "llama"}] }

@app.route("/v1/engines/llama/completions", methods=["POST"])
def chat():
    try:
        # important for stable work on 16gb gpu
        import gc
        gc.collect()
        try:
            torch.cuda.empty_cache()
        except:
            pass

        prompt = request.json.get("prompt", "Hello, world!")
        max_tokens = request.json.get("max_tokens", 120)
        temperature = request.json.get("temperature", 0.9)
        
        results = generator.generate([prompt], max_gen_len=max_tokens, temperature=temperature, top_p=0.95)[0].removeprefix(prompt)
        
    except Exception as e:
        print(f"Error: {e}")
        results = ""

    curr_dt = datetime.now()
    return {
        "id": "I don't need id, lmao",
        "object": "text_completion",
        "created": int(round(curr_dt.timestamp())),
        "model": "llama-7B",
        "choices": [
            {
                "text": results,
                "index": 0,
                "logprobs": None,
                "finish_reason": "length"
            }
        ]
    }

def main(ckpt_dir: str = "../llama_model/7B", tokenizer_path: str = "../llama_model/tokenizer.model"):
    #local_rank, world_size = setup_model_parallel()
    #if local_rank > 0:
    #    sys.stdout = open(os.devnull, "w")

    global generator
    generator = load(ckpt_dir, tokenizer_path)

    app.run(host="0.0.0.0", port=1234)

if __name__ == "__main__":
    fire.Fire(main)
