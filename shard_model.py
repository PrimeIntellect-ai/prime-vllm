"""
Script to shard a model into multiple shards, save them to disk, and upload them to Hugging Face Hub.

Usage:
python shard_model.py --model-name <model_name> --persistent-dir <persistent_dir> --num-shards <num_shards>

Example:
python shard_model.py --model-name meta-llama/llama-2-7b-chat-hf --persistent-dir /workspace --num-shards 2
"""
import os
import argparse
from typing import Optional

import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import create_repo, upload_folder, whoami
from transformers.utils.logging import disable_progress_bar

from contextlib import contextmanager
from time import perf_counter

disable_progress_bar()

@contextmanager
def time(action: str):
    action, rest = action.split(" ", 1)
    print(f"{action}ing {rest}...")
    start = perf_counter()
    yield
    end = perf_counter()
    print(f"{action}ed {rest} took {end - start:.2f} seconds")


def main(
    model_name: str,
    persistent_dir: str,
    num_shards: int,
    token: Optional[str] = None,
    transformer_block_name: str = "layers",
):
    # Load the full model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=persistent_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=persistent_dir)
    
    # Get the transformer blocks
    if not hasattr(model.model, transformer_block_name):
        raise ValueError(f"{transformer_block_name} not found in model")
    transformer_blocks = getattr(model.model, transformer_block_name)
    num_blocks = len(transformer_blocks)
    blocks_per_shard = num_blocks // num_shards
    print(f"Found {num_blocks} blocks in {transformer_block_name}")
    print(f"Blocks per shard: {blocks_per_shard}")

    # Create output directory
    output_dir = os.path.join(persistent_dir, f"{model_name}-{num_shards}")
    os.makedirs(output_dir, exist_ok=True)

    # Create output repository if it doesn't exist
    assert os.environ.get("HF_TOKEN") is not None, "HF_TOKEN is not set"
    username = whoami()["name"]
    repo_name = f"{username}/{'-'.join(model_name.split('/'))}-{num_shards}"
    print(f"Creating Hugging Face repository {repo_name}")
    create_repo(repo_name, exist_ok=True)
    
    # Create and save shards
    for i in range(num_shards):
        # Create a new model instance
        with time(f"Creat shard {i+1}/{num_shards}"):
            start_block = i * blocks_per_shard
            end_block = (i + 1) * blocks_per_shard if i < num_shards - 1 else num_blocks
            shard = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=persistent_dir)

            # Update config to match actual number of layers
            num_layers = end_block - start_block
            if not hasattr(shard.config, "num_hidden_layers"):
                raise ValueError("num_hidden_layers not found in config")
            setattr(shard.config, "num_hidden_layers", num_layers)
            
            # Set the transformer blocks for this shard
            setattr(shard.model, transformer_block_name, nn.ModuleList(transformer_blocks[start_block:end_block]))
        
        # Save the shard
        with time(f"Sav shard {i+1}/{num_shards}"):
            shard_path = os.path.join(output_dir, f"shard_{i}")
            shard.save_pretrained(shard_path)
            tokenizer.save_pretrained(shard_path)

        # Upload to Hugging Face Hub
        with time(f"Upload shard {i+1}/{num_shards}"):
            try:
                upload_folder(
                    folder_path=shard_path,
                    repo_id=repo_name,
                    repo_type="model",
                    path_in_repo=f"shard_{i}",
                )
            except Exception as e:
                raise Exception(f"Error uploading shard {i+1}/{num_shards}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--persistent-dir", type=str, required=True)
    parser.add_argument("--num-shards", type=int, required=True)
    args = parser.parse_args()
    
    main(**vars(args)) 