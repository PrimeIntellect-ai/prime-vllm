import os
import sys
import argparse
import pickle
from torch.distributed import destroy_process_group
import torch.nn as nn
import time
from functools import partial
from typing import Optional, Dict, Any, List

from iroh_py import Node

import autorootcwd  # noqa: F401

# Globals
logger = None
node = None

def print_layer_input(_, input, layer_index: int):
    logger.debug(f"Layer: {layer_index} input: {input[1].shape}")

def print_layer_output(_, __, output, layer_index: int):
    logger.debug(f"Layer: {layer_index} output: {output[0].shape}")

def send_intermediate_states(_, __, output):
    hidden_states, residual = output
    node.isend(pickle.dumps(hidden_states.to("cpu")), tag=0, latency=None).wait()
    node.isend(pickle.dumps(residual.to("cpu")), tag=0, latency=None).wait()

def recv_intermediate_states(_, input):
    positions, _, _ = input
    device = positions.device
    hidden_states = pickle.loads(node.irecv(tag=0).wait()).to(device)
    residual = pickle.loads(node.irecv(tag=0).wait()).to(device)

    return positions, hidden_states, residual

def recv_output(_, __, ___):
    output = pickle.loads(node.irecv(tag=0).wait())
    return output

def send_output(_, __, output):
    node.isend(pickle.dumps(output), tag=0, latency=None).wait()

def main(
    prompts: List[str],
    log_level: str,
    engine_args: Optional[Dict[str, Any]] = None,
    sampling_args: Optional[Dict[str, Any]] = None,
):
    # Setting environment variables (default to single device)
    rank, world_size = int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    os.environ["CUDA_VISIBLE_DEVICES"] = str(local_rank)
    if log_level != "DEBUG":
        os.environ["VLLM_CONFIGURE_LOGGING"] = "0"
    os.environ["VLLM_USE_V1"] = "0"

    # Setup logging
    from loguru import logger as loguru_logger
    loguru_logger.remove()  # Remove default handlers
    format = f"[Rank {rank}] <green>{{time:YYYY-MM-DD HH:mm:ss}}</green> | <level>{{level}}</level> | <level>{{message}}</level>"
    loguru_logger.add(sys.stdout, format=format, colorize=True, level=log_level)
    global logger
    logger = loguru_logger.bind(rank=rank)

    # Setup communication (for multi-node settings)
    if world_size > 1:
        logger.info("Setting up P2P communication")
        global node
        seed = os.environ.get("IROH_SEED")
        node = Node.with_seed(num_streams=1, seed=int(seed) if seed is not None else seed)
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
    
    # Initialize engine with provided args
    llm = LLM(**(engine_args or {}), enforce_eager=True)

    if world_size > 1:
        model : nn.Module = llm.llm_engine.model_executor.driver_worker.model_runner.model

        if rank == 0:
            # First stage
            last_layer : nn.Module = model.model.layers[-1]
            last_layer.register_forward_hook(send_intermediate_states)

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

    # Start generation
    logger.info("Generating...")
    sampling_params = SamplingParams(**(sampling_args or {}))
    start_generate = time.perf_counter()
    outputs = llm.generate(prompts, sampling_params=sampling_params, use_tqdm=False)
    generate_time = time.perf_counter() - start_generate

    # Print generations
    tokens_generated = 0
    if rank == world_size - 1:
        for completion in outputs:
            for output in completion.outputs:
                tokens_generated += len(output.token_ids)
                logger.info(f"{completion.prompt}{output.text}")

        # Print throughput
        throughput = tokens_generated / generate_time
        logger.info(f"Generated {tokens_generated} tokens in {generate_time:.2f} seconds")
        logger.info(f"Throughput: {throughput:.2f} tokens/second")

    # Destroy torch.distributed process group
    destroy_process_group()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Generation arguments
    parser.add_argument("--prompts", type=str, nargs="+", default=["Hi, my name is"])
    parser.add_argument("--log-level", type=str, default="INFO")

    # Engine arguments
    engine_group = parser.add_argument_group("Engine Arguments")
    engine_group.add_argument("--model", type=str, required=True)
    engine_group.add_argument("--download-dir", type=str, default=os.environ.get("CACHE_DIR", None))
    engine_group.add_argument("--tensor-parallel-size", type=int, default=1)
    engine_group.add_argument("--gpu-memory-utilization", type=float, default=0.90)

    # Sampling arguments
    sampling_group = parser.add_argument_group("Sampling Arguments")
    sampling_group.add_argument("--n", type=int, default=1)
    sampling_group.add_argument("--temperature", type=float, default=1.0)
    sampling_group.add_argument("--top-p", type=float, default=1.0)
    sampling_group.add_argument("--top-k", type=int, default=-1)
    sampling_group.add_argument("--max-tokens", type=int, default=16)
    sampling_group.add_argument("--min-tokens", type=int, default=0)
    sampling_group.add_argument("--seed", type=int, default=None)
    
    args = parser.parse_args()
    
    # Separate engine and sampling arguments
    engine_args = {}
    sampling_args = {}
    
    # Collect engine arguments
    for arg in ["model", "download_dir", "tensor_parallel_size", "gpu_memory_utilization"]:
        value = getattr(args, arg)
        if value is not None:
            engine_args[arg] = value
    
    # Collect sampling arguments
    for arg in ["n", "temperature", "top_p", "top_k", "max_tokens", "min_tokens", "seed"]:
        value = getattr(args, arg)
        if value is not None:
            sampling_args[arg] = value
    
    main(
        prompts=args.prompts,
        log_level=args.log_level,
        engine_args=engine_args,
        sampling_args=sampling_args,
    )