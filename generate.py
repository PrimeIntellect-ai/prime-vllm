"""
Script to generate text using vlLM + Iroh.
"""
import os
import argparse
from pprint import pprint
import pickle
import torch.nn as nn
from iroh_py import Node
import time

def create_comm() -> None:
    global node
    
    # Initialize communication node
    seed = os.environ.get("IROH_SEED")
    node = Node.with_seed(1, seed=int(seed) if seed is not None else None)
    print(f"Connect to node {node.node_id()}")
    time.sleep(1)

    # Connect to remote node
    peer_id = os.environ.get("IROH_PEER_ID")
    if peer_id is None:
        print("Didn't find IROH_PEER_ID environment variable, please enter the peer's public key: ")
        peer_id = input().strip()
    print(f"Connecting to {peer_id}")
    node.connect(peer_id)

    # Wait for connection to be established
    while not node.is_ready():
        time.sleep(1 / 100)
    print("Connected!")
    return node


def main(model_path: str):
    rank, world_size = int(os.environ.get("RANK")), int(os.environ.get("WORLD_SIZE"))

    os.environ["VLLM_USE_V1"] = "0"
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("LOCAL_RANK", str(rank))

    # Create communication node
    if world_size > 1:
        node = create_comm()

    def print_layer_input(module, input, layer_index: int):
        print(f"\nLayer: {layer_index}")
        print(input[1].shape)
        print(input[1])

    def print_layer_output(module, args, output, layer_index: int):
        print(f"\nLayer: {layer_index}")
        print(output[0].shape)
        print(output[0])

    def send_intermediate_states(module, args, output):
        hidden_states, residual = output
        node.isend(pickle.dumps(hidden_states.to("cpu")), tag=0, latency=0).wait()
        node.isend(pickle.dumps(residual.to("cpu")), tag=0, latency=0).wait()

    def recv_intermediate_states(module, input):
        positions, _, _ = input
        device = positions.device
        hidden_states = pickle.loads(node.irecv(tag=0).wait()).to(device)
        residual = pickle.loads(node.irecv(tag=0).wait()).to(device)

        return positions, hidden_states, residual

    def recv_output(module, args, output):
        output = pickle.loads(node.irecv(tag=0).wait())
        return output

    def send_output(module, args, output):
        node.isend(pickle.dumps(output), tag=0, latency=0).wait()

    # Create vLLM class
    from vllm import LLM, SamplingParams

    if world_size > 1:
        model_path = os.path.join(f"{model_path}-{world_size}", f"shard_{rank}")
    assert os.path.exists(model_path), f"Model path {model_path} does not exist"
    llm = LLM(model=model_path, enforce_eager=True)

    if world_size > 1:
        model : nn.Module = llm.llm_engine.model_executor.driver_worker.model_runner.model

        # for layer_index, layer in enumerate(model.model.layers):
        #     if layer_index == 0 or layer_index == len(model.model.layers) - 1:
        #         layer.register_forward_pre_hook(partial(print_layer_input, layer_index=layer_index))
        #         layer.register_forward_hook(partial(print_layer_output, layer_index=layer_index))

        if rank == 0:
            final_layer : nn.Module = model.model.layers[-1]
            final_layer.register_forward_hook(send_intermediate_states)

            sampler : nn.Module = model.sampler
            sampler.register_forward_hook(recv_output)
        
        if rank == world_size - 1:
            first_layer : nn.Module = model.model.layers[0]
            first_layer.register_forward_pre_hook(recv_intermediate_states)

            sampler : nn.Module = model.sampler
            sampler.register_forward_hook(send_output)

    parallel_prompts, num_new_tokens = 1, 256
    prompts = ["Hi, my name is"] # , "What is the capital of France?"]
    batch_size = len(prompts)
    sampling_params = SamplingParams(n=parallel_prompts, max_tokens=num_new_tokens, min_tokens=num_new_tokens, seed=69)
    start_generate = time.perf_counter()
    outputs = llm.generate(prompts, sampling_params=sampling_params, use_tqdm=False)
    generate_time = time.perf_counter() - start_generate
    token_generated = batch_size * parallel_prompts * num_new_tokens

    if rank == world_size - 1:
        completions_list = [completion.__dict__ for completion in outputs]
        pprint(completions_list)

    print(f"Time to generate: {generate_time:.2f} seconds")
    print(f"Tokens generated: {token_generated}")
    print(f"Tokens per second: {token_generated / generate_time:.2f}")


if __name__ == "__main__":
    assert os.environ.get("HF_TOKEN") is not None, "Set HF_TOKEN environment variable"
    assert os.environ.get("RANK") is not None, "Set RANK environment variable"
    assert os.environ.get("WORLD_SIZE") is not None, "Set WORLD_SIZE environment variable"

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)

    args = parser.parse_args()
    main(**vars(args))