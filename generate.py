import os
import argparse
import pickle
import torch.nn as nn
import time
from functools import partial
from typing import Optional

from loguru import logger
from iroh_py import Node

import autorootcwd  # noqa: F401

# Global communication node
node = None

def print_layer_input(module, input, layer_index: int):
    logger.debug(f"\nLayer: {layer_index}")
    logger.debug(input[1].shape)
    logger.debug(input[1])

def print_layer_output(module, args, output, layer_index: int):
    logger.debug(f"\nLayer: {layer_index}")
    logger.debug(output[0].shape)
    logger.debug(output[0])

def send_intermediate_states(module, args, output):
    hidden_states, residual = output
    node.send(pickle.dumps(hidden_states.to("cpu")))
    node.send(pickle.dumps(residual.to("cpu")))

def recv_intermediate_states(module, input):
    positions, _, _ = input
    device = positions.device
    hidden_states = pickle.loads(node.recv()).to(device)
    residual = pickle.loads(node.recv()).to(device)

    return positions, hidden_states, residual

def recv_output(module, args, output):
    output = pickle.loads(node.recv())
    return output

def send_output(module, args, output):
    node.send(pickle.dumps(output))

def main(
    model: str,
    download_dir: Optional[str] = None,
):
    # Setting environment variables (default to single device)
    rank, local_rank, world_size = int(os.environ.get("RANK", 0)), int(os.environ.get("LOCAL_RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))
    os.environ["CUDA_VISIBLE_DEVICES"] = str(local_rank)
    os.environ["VLLM_USE_V1"] = "0"

    # Create communication (for multi-node settings)
    if world_size > 1:
        logger.info("Setting up P2P communication")
        global node
        node = Node()
        logger.info(f"Connect to node with: {node.node_id()}")
        time.sleep(1)

        # Connect to remote node
        peer_id = os.environ.get("IROH_PEER_ID")
        if peer_id is None:
            logger.info("Didn't find IROH_PEER_ID environment variable, please enter the peer's public key: ")
            peer_id = input().strip()
        logger.info(f"Connecting to {peer_id}")
        node.connect(peer_id)

        # Wait for connection
        while not node.is_ready():
            time.sleep(0.1)
        logger.info("Connected!")

    # Create vLLM class
    logger.info("Setting up vLLM")
    from vllm import LLM, SamplingParams
    llm = LLM(
        model=model,
        download_dir=download_dir,
        max_model_len=1024,
        enforce_eager=True
    )

    if world_size > 1:
        model : nn.Module = llm.llm_engine.model_executor.driver_worker.model_runner.model

        if rank == 0:
            # First stage
            lasy_layer : nn.Module = model.model.layers[-1]
            lasy_layer.register_forward_hook(send_intermediate_states)

            sampler : nn.Module = model.sampler
            sampler.register_forward_hook(recv_output)
        elif rank == world_size - 1:
            # Last stage
            first_layer : nn.Module = model.model.layers[0]
            first_layer.register_forward_pre_hook(recv_intermediate_states)

            sampler : nn.Module = model.sampler
            sampler.register_forward_hook(send_output)
        else:
            # Intermediate stage
            first_layer : nn.Module = model.model.layers[0]
            last_layer : nn.Module = model.model.layers[-1]
            first_layer.register_forward_pre_hook(recv_intermediate_states)
            last_layer.register_forward_hook(send_intermediate_states)

        for layer_idx, layer in enumerate(model.model.layers):
            layer : nn.Module = layer
            layer.register_forward_pre_hook(partial(print_layer_input, layer_index=layer_idx))
            layer.register_forward_hook(partial(print_layer_output, layer_index=layer_idx))

    parallel_prompts, num_new_tokens = 1, 256
    prompts = ["Hi, my name is"]
    batch_size = len(prompts)
    sampling_params = SamplingParams(n=parallel_prompts, max_tokens=num_new_tokens, min_tokens=num_new_tokens, seed=69)
    start_generate = time.perf_counter()
    outputs = llm.generate(prompts, sampling_params=sampling_params, use_tqdm=False)
    generate_time = time.perf_counter() - start_generate
    token_generated = batch_size * parallel_prompts * num_new_tokens

    if rank == world_size - 1:
        for completion in outputs:
            print(f"{completion.prompt}{completion.outputs[0].text}")

    print(f"Time to generate: {generate_time:.2f} seconds")
    print(f"Tokens generated: {token_generated}")
    print(f"Tokens per second: {token_generated / generate_time:.2f}")

if __name__ == "__main__":
    assert os.environ.get("HF_TOKEN") is not None, "Set HF_TOKEN environment variable"

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--download-dir", type=str, default=None)

    args = parser.parse_args()
    main(**vars(args))