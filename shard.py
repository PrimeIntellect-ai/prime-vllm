"""
Script to shard a model into multiple shards, save them to disk, and upload them to Hugging Face Hub.

Check usage:
uv run script/shard.py -h
"""
import autorootcwd  # noqa: F401

import os
import argparse
from time import perf_counter
from contextlib import contextmanager
from typing import Optional

import torch
import torch.nn as nn

from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import create_repo, upload_folder

def rhasattr(obj, attr_path):
    attrs = attr_path.split('.')
    current = obj
    
    for attr in attrs:
        if not hasattr(current, attr):
            return False
        current = getattr(current, attr)
    
    return True

def rgetattr(obj, attr_path):
    attrs = attr_path.split('.')
    current = obj
    
    for attr in attrs:
        if not hasattr(current, attr):
            raise AttributeError(f"'{type(current).__name__}' object has no attribute '{attr}'")
        current = getattr(current, attr)
    
    return current


def rsetattr(obj, attr_path, value):
    attrs = attr_path.split('.')
    current = obj
    
    for attr in attrs[:-1]:
        if not hasattr(current, attr):
            raise AttributeError(f"'{type(current).__name__}' object has no attribute '{attr}'")
        current = getattr(current, attr)
    
    setattr(current, attrs[-1], value)

class FakeEmbedding(nn.Module):
    def __init__(self, hidden_size, dtype):
        super().__init__()
        self.hidden_size = hidden_size
        self.dtype = dtype

    def forward(self, x):
        return torch.zeros(*x.shape, self.hidden_size, device=x.device, dtype=self.dtype)

class FakeLMHead(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.vocab_size = vocab_size
        self.quant_method = self

    def forward(self, x, *args, **kwargs):
        return x

    def apply(self, module, hidden_states, bias=None):
        return self.forward(hidden_states)

def create_shard(
    model_name: str,
    cache_dir: Optional[str],
    shard_idx: int,
    num_shards: int,
    model_layers_key: str = "model.layers",
    config_layers_key: str = "num_hidden_layers",
    lm_head_key: str = "lm_head",
    embed_tokens_key: str = "model.embed_tokens",
):
    # Load the full model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=cache_dir, torch_dtype="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    
    # Get the transformer blocks
    if not rhasattr(model, model_layers_key):
        raise ValueError(f"{model_layers_key} not found in model")
    transformer_blocks = rgetattr(model, model_layers_key)
    num_blocks = len(transformer_blocks)
    assert num_blocks % num_shards == 0, f"Number of blocks must be divisible by number of shards but got {num_blocks} and {num_shards}"
    blocks_per_shard = num_blocks // num_shards

    # Determine the start and end block for this shard
    start_block = shard_idx * blocks_per_shard
    end_block = (shard_idx + 1) * blocks_per_shard if shard_idx < num_shards - 1 else num_blocks
    
    # Create a new model instance
    shard = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=cache_dir, torch_dtype="auto")
    
    # Update config to match actual number of layers
    num_layers = end_block - start_block
    if not hasattr(shard.config, "num_hidden_layers"):
        raise ValueError("num_hidden_layers not found in config")
    
    # Set the transformer blocks for this shard
    rsetattr(shard, model_layers_key, nn.ModuleList(transformer_blocks[start_block:end_block]))
    rsetattr(shard.config, config_layers_key, num_layers)
    shard.config.shard_idx = shard_idx
    shard.config.num_shards = num_shards

    # Get the state dict
    # state_dict = shard.state_dict()
    
    # # Handle embedding and LM head based on shard position
    # if shard_idx > 0:
    #     # Remove embedding on all but first shard
    #     # del state_dict[f"{embed_tokens_key}.weight"]
    #     rsetattr(shard, embed_tokens_key, FakeEmbedding(shard.config.hidden_size, shard.dtype))
    # if shard_idx < num_shards - 1:
    #     # Remove LM head on all but last shard
    #     # del state_dict[f"{lm_head_key}.weight"]
    #     rsetattr(shard, lm_head_key, FakeLMHead(shard.config.vocab_size))
    
    # Load the modified state dict back to the model
    # shard.load_state_dict(state_dict, strict=False)

    return shard, tokenizer

@contextmanager
def time(action: str):
    print(f"{action}...", end="")
    start = perf_counter()
    yield
    end = perf_counter()
    print(f" ({end - start:.2f}s)")

def main(
    model_name: str,
    num_shards: int,
    cache_dir: Optional[str],
    local_dir: Optional[str],
    remote_dir: Optional[str],
    model_layers_key: str = "model.layers",
    config_layers_key: str = "num_hidden_layers",
    lm_head_key: str = "lm_head",
    embed_tokens_key: str = "model.embed_tokens",
):
    # Create local output directory
    if local_dir is not None:
        os.makedirs(local_dir, exist_ok=True)

    # Create HF repository if it doesn't exist
    if remote_dir is not None:
        with time(f"Creating Hugging Face repository {remote_dir}"):
            create_repo(remote_dir, exist_ok=True)
    
    # Create and save shards
    for shard_idx in range(num_shards):
        # Create a new model instance
        with time(f"Creating shard {shard_idx+1}/{num_shards}"):
            shard, tokenizer = create_shard(
                model_name,
                cache_dir,
                shard_idx,
                num_shards,
                model_layers_key,
                config_layers_key,
                lm_head_key,
                embed_tokens_key,
            )

        # Save the shard locally
        if local_dir is not None:
            shard_path = os.path.join(local_dir, f"shard_{shard_idx}")
            with time(f"Saving shard {shard_idx+1}/{num_shards} to {shard_path}"):
                shard.save_pretrained(shard_path)
                tokenizer.save_pretrained(shard_path)

        # Upload to Hugging Face Hub
        if remote_dir is not None:
            with time(f"Uploading shard {shard_idx+1}/{num_shards} to {remote_dir}/shard_{shard_idx}"):
                try:
                    upload_folder(
                        folder_path=shard_path,
                        repo_id=remote_dir,
                        repo_type="model",
                        path_in_repo=f"shard_{shard_idx}",
                    )
                except Exception as e:
                    raise Exception(f"Error uploading shard {shard_idx+1}/{num_shards}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, required=True, help="Model name to shard")
    parser.add_argument("--num-shards", type=int, required=True, help="Number of shards to create"  )
    parser.add_argument("--cache-dir", type=str, default=None, help="Cache directory for model weights")
    parser.add_argument("--local-dir", type=str, default=None, help="Local directory to save shards")
    parser.add_argument("--remote-dir", type=str, default=None, help="Remote directory to upload shards")
    parser.add_argument("--model-layers-key", type=str, default="model.layers")
    parser.add_argument("--config-layers-key", type=str, default="num_hidden_layers")
    parser.add_argument("--lm-head-key", type=str, default="lm_head")
    parser.add_argument("--embed-tokens-key", type=str, default="model.embed_tokens")
    args = parser.parse_args()

    assert os.environ.get("HF_TOKEN") is not None, "HF_TOKEN is not set"
    
    main(**vars(args)) 