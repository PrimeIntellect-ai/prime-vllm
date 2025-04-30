"""
Script to shard a model into multiple shards, save them to disk, and upload them to Hugging Face Hub.

Check usage:
uv run script/shard.py -h
"""
import autorootcwd  # noqa: F401

import os
import argparse
from typing import Optional

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

def create_shard(
    model_name: str,
    cache_dir: Optional[str],
    shard_idx: int,
    num_shards: int,
    model_layers_key: str = "model.layers",
    config_layers_key: str = "num_hidden_layers",
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

    return shard, tokenizer

def main(
    model: str,
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
        print(f"Creating local directory {local_dir}")
        os.makedirs(local_dir, exist_ok=True)
    
    # Create and save shards
    for shard_idx in range(num_shards):
        # Create a new model instance
        print(f"Creating shard {shard_idx+1}/{num_shards}")
        shard, tokenizer = create_shard(
            model,
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
            print(f"Saving shard {shard_idx+1}/{num_shards} to {local_dir}/shard_{shard_idx}")
            shard_path = os.path.join(local_dir, f"shard_{shard_idx}")
            shard.save_pretrained(shard_path)
            tokenizer.save_pretrained(shard_path)

            # Upload to Hugging Face Hub
            if remote_dir is not None:
                shard_remote_dir = f"{remote_dir}-{shard_idx}.{num_shards}"
                print(f"Uploading shard {shard_idx+1}/{num_shards} to {shard_remote_dir}")
                try:
                    create_repo(shard_remote_dir, exist_ok=True)
                except Exception as e:
                    raise Exception(f"Error creating remote directory {shard_remote_dir}: {e}")
                try:
                    upload_folder(folder_path=shard_path, repo_id=shard_remote_dir, repo_type="model")
                except Exception as e:
                    raise Exception(f"Error uploading shard {shard_idx+1}/{num_shards}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Model name to shard")
    parser.add_argument("--num-shards", type=int, required=True, help="Number of shards to create"  )
    parser.add_argument("--cache-dir", type=str, default=os.environ.get("CACHE_DIR", None), help="Cache directory for model weights")
    parser.add_argument("--local-dir", type=str, default=None, help="Local directory to save shards")
    parser.add_argument("--remote-dir", type=str, default=None, help="Remote directory to upload shards")
    parser.add_argument("--model-layers-key", type=str, default="model.layers")
    parser.add_argument("--config-layers-key", type=str, default="num_hidden_layers")
    args = parser.parse_args()
    
    main(**vars(args)) 