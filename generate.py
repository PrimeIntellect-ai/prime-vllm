import os
import sys
import argparse
import pickle
from torch.distributed import destroy_process_group
import torch.nn as nn
import time
from typing import Optional, Dict, Any, List
import json

from prime_iroh import Node

import autorootcwd  # noqa: F401

# Globals
logger = None
node = None

def send_intermediate_states(_, __, output):
    hidden_states, residual = output
    serialized_hidden_states = pickle.dumps(hidden_states.to("cpu"))
    serialized_residual = pickle.dumps(residual.to("cpu"))
    node.isend(serialized_hidden_states, tag=0, latency=None).wait()
    node.isend(serialized_residual, tag=0, latency=None).wait()
    logger.debug(f"Sent hidden_states: {hidden_states.shape} ({len(serialized_hidden_states)} bytes sent) and residual: {residual.shape} ({len(serialized_residual)} bytes sent)")

def recv_intermediate_states(_, input):
    positions, _, _ = input
    device = positions.device
    serialized_hidden_states = node.irecv(tag=0).wait()
    serialized_residual = node.irecv(tag=0).wait()
    hidden_states = pickle.loads(serialized_hidden_states).to(device)
    residual = pickle.loads(serialized_residual).to(device)
    logger.debug(f"Got hidden_states: {hidden_states.shape} ({len(serialized_hidden_states)} bytes sent), residual: {residual.shape} ({len(serialized_residual)} bytes sent) and positions {positions.shape}")

    return positions, hidden_states, residual

def main(
    prompts: List[str],
    prompt_file: Optional[str],
    output_file: Optional[str],
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
        # Model runner owns model and sampler
        model_runner : nn.Module = llm.llm_engine.model_executor.driver_worker.model_runner
        model : nn.Module = model_runner.model.model

        # First and last layers (pre/post-hook to recv/send intermediate states)
        first_layer : nn.Module = model.layers[0]
        last_layer : nn.Module = model.layers[-1]

        # Sampler (pre/ post-hook to recv/send outputs)
        # sampler : nn.Module = model_runner.sampler

        if rank == 0: # First stage
            # Send intermediate states to next stage (post-hook)
            last_layer.register_forward_hook(send_intermediate_states)

            # Receive outputs from last stage (post-hook)
            # sampler : nn.Module = model_runner.sampler
            # sampler.register_forward_hook(recv_output)

        elif rank == world_size - 1: # Last stage
            # Receive intermediate states from previous stage (pre-hook)
            first_layer.register_forward_pre_hook(recv_intermediate_states)

            # Send outputs to first  stage (post-hook)
            # sampler.register_forward_hook(send_output)
        else:
            # Receive intermediate states from previous stage and send positions to next stage (pre-hook)
            first_layer.register_forward_pre_hook(recv_intermediate_states)

            # Send intermediate states to next stage (post-hook)
            last_layer.register_forward_hook(send_intermediate_states)

        # Monkey-patch ModelRunner.execute_model function
        from vllm.worker.model_runner import ModelRunner
        original_execute_model = ModelRunner.execute_model
        def patched_execute_model(self, *args, **kwargs):
            outputs = original_execute_model(self, *args, **kwargs)
            if rank == world_size - 1: # Last stage (only send)
                serialized_output = pickle.dumps(outputs)
                node.isend(serialized_output, tag=0, latency=None).wait()
                logger.debug(f"Sent outputs ({len(serialized_output)} bytes sent)")
            elif rank == world_size - 2: # Second last stage (only receive)
                serialized_output = node.irecv(tag=0).wait()
                logger.debug(f"Received outputs ({len(serialized_output)} bytes sent)")
                outputs = pickle.loads(serialized_output)
            else: # All, but last and second last stage (receive and relay)
                serialized_output = node.irecv(tag=0).wait()
                logger.debug(f"Received outputs ({len(serialized_output)} bytes sent)")
                node.isend(serialized_output, tag=0, latency=None).wait()
                logger.debug(f"Sent outputs ({len(serialized_output)} bytes sent)")
                outputs = pickle.loads(serialized_output)

            return outputs

        ModelRunner.execute_model = patched_execute_model
            
    # Read prompts from file
    if prompt_file is not None:
        logger.info(f"Reading prompts from {prompt_file}")
        with open(prompt_file, "r") as f:
            prompts = [json.loads(line)["prompt"] for line in f if line.strip()]

    # Start generation
    logger.info("Generating...")
    sampling_params = SamplingParams(**(sampling_args or {}))
    start_generate = time.perf_counter()
    completions = llm.generate(prompts, sampling_params=sampling_params, use_tqdm=False)
    generate_time = time.perf_counter() - start_generate

    # Write outputs to file
    if output_file is not None:
        with open(output_file, 'w') as f:
            for completion in completions:
                for output in completion.outputs:
                    f.write(json.dumps({"prompt": completion.prompt, "completion": output.text}))
                    f.write("\n")

    # Print performance
    if rank == world_size - 1:
        tokens_generated = 0
        for completion in completions:
            for output in completion.outputs:
                tokens_generated += len(output.token_ids)

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
    parser.add_argument("--prompt-file", type=str, default=None)
    parser.add_argument("--output-file", type=str, default=None)
    parser.add_argument("--log-level", type=str, default="INFO")

    # Engine arguments
    engine_group = parser.add_argument_group("Engine Arguments")
    engine_group.add_argument("--model", type=str, required=True)
    engine_group.add_argument("--download-dir", type=str, default=os.environ.get("CACHE_DIR", None))
    engine_group.add_argument("--tensor-parallel-size", type=int, default=1)
    engine_group.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    engine_group.add_argument("--max-model-len", type=int, default=None)

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
    for arg in ["model", "download_dir", "tensor_parallel_size", "gpu_memory_utilization", "max_model_len"]:
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
        prompt_file=args.prompt_file,
        output_file=args.output_file,
        log_level=args.log_level,
        engine_args=engine_args,
        sampling_args=sampling_args,
    )