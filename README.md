<p align="center">
</p>

<img src="https://github.com/user-attachments/assets/51e44795-5206-49d6-a12a-ecacd2799df2" alt="Prime Intellect" style="width: 100%; height: auto;"/>

---

<p align="center">

<h3 align="center">
PRIME-VLLM: Run vLLM pipeline parallel inference over public networks
</h3>

---

This codebase integrates vLLM with our custom P2P communication library `prime-iroh`, in order to run pipeline parallel inference across geographically distributed nodes. It works by hijacking the pipeline parallel vLLM implementation to support sending and receiving intermediate results between nodes that are hooked into vLLM using PyTorch pre- and post-hooks. The codebase has two main entrypoints: 

- `generate.py` is used to generate text for a specific model, set of prompts and generation parameters, possibly pipelining over public networks.
- `shard.py` is used to pre-shard models and save them locally or upload them to Hugging Face Hub.

# Usage

## Installation

**Quick Install:** Run the following command for a quick install:

```
curl -sSL https://raw.githubusercontent.com/PrimeIntellect-ai/prime-vllm/refs/heads/master/script/install.sh | bash
```

**Manual Install:** First, install `uv` and `cargo` to build the project.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
```

```bash
curl https://sh.rustup.rs -sSf | sh
source $HOME/.cargo/env
```

Then, clone the repository and install the dependencies.

```bash
git clone git@github.com:PrimeIntellect-ai/prime-vllm.git && cd prime-vllm
uv sync
```

Also, if you plan to use a private model, you will need to set the `HF_TOKEN` environment variable. Also, we recommend setting the `CACHE_DIR` environment variable to a local directory with enough disk space to store the model weights.

```bash
export CACHE_DIR=<path-to-cache-dir>
export HF_TOKEN=<your-token>
```

## Inference

### Single Node

To check that your installation has succeeded, you can run the following command to generate text with a small model on a single node:

```bash
RANK=0 WORLD_SIZE=1 uv run generate.py --model <model_name>
```

For example, to generate text from `Qwen/Qwen3-14B` using the model's default generation parameters from a simple prompt file, you can run the following commands:

```bash
# Create a simple prompt file
echo '{"prompt": "What is Prime Intellect?"}' > prompt-file.jsonl
```

```bash
# Run inference
RANK=0 WORLD_SIZE=1 uv run generate.py \
    --model Qwen/Qwen3-14B \
    --temperature 0.6 \
	--top-p 0.95 \
	--top-k 20 \
    --prompt-file prompt-file.jsonl \
    --output-file output-file.jsonl
```

*If you are running low on memory, you should try reducing the model's context window by setting the `max_model_len` argument to a smaller value.*

To explore more command line arguments, run `uv run python generate.py --help`.

### Multi-Node

To run distributed inference, you will need to pre-shard your model. This can be done by running the `shard.py` script (see [Sharding](#sharding) for more information). 

Once you sharded your model into as many shards as you have GPUs, running distributed inference is as easy as setting environment variables. For example, to run a model on two nodes, set the following environment variables on each node. 

```bash
# On the first node
export RANK=0
export LOCAL_RANK=0
export WORLD_SIZE=2
export IROH_SEED=0
export IROH_PEER_ID=ff87a0b0a3c7c0ce827e9cada5ff79e75a44a0633bfcb5b50f99307ddb26b337
```

```bash
# On the second node
export RANK=1
export LOCAL_RANK=0
export WORLD_SIZE=2
export IROH_SEED=1
export IROH_PEER_ID=ee1aa49a4459dfe813a3cf6eb882041230c7b2558469de81f87c9bf23bf10a03
```

*In the example above `IROH_PEER_ID` is the ID of the partner node meaning that node 1's ID is `ee1aa49a4459dfe813a3cf6eb882041230c7b2558469de81f87c9bf23bf10a03`. The `IROH_SEED` ensures we always generate the same ID for the current node. You can skip seeding the generation to obtain a random ID, in which case `prime-vllm` will prompt you to manually enter the peer ID.*

Then, to run inference, run the following command on each node.

```bash
uv run generate.py --model <shard_name>
```

For example, to run inference for `Qwen/Qwen3-14B` on two nodes, run the following command on each node.

```bash
uv run generate.py \
	--model mikasenghaas/Qwen3-14B-$RANK.$WORLD_SIZE \
	--temperature 0.6 \
	--top-p 0.95 \
	--top-k 20 \
	--prompt-file prompt-file.jsonl \
	--output-file output-file.jsonl
```

Again, to explore more command line arguments, run `uv run python generate.py --help`.


## Sharding

If you want to run inference on a custom model, you have to pre-shard it using the `shard.py` script. Specify the model name by its HF Hub name, the number of shards you want to create, and the local and remote directories to save and upload the shards.

```bash
uv run shard.py \
	--model <model_name> \
	--num-shards <num_shards> \
	--local-dir <local_dir> \
	--remote-dir <username>/<repo_name>
```

*You can skip uploading shards to the remote directory by not setting the CLI argument `--remote-dir`. In this case, you will have to reference the local directory for the `--model` flag for the `generate.py` script.*

For example, to shard `Qwen/Qwen3-14B` into two shards and save them to the local directory `./shards` and upload them to the remote directory `<hf_username>/<repo_name>`, run the following command.

```bash
uv run shard.py \
	--model Qwen/Qwen3-14B \
	--num-shards 2 \
	--local-dir ./shards \
	--remote-dir <hf_username>/<repo_name>
```

This will automatically save the shards to subdirectories `shard_<shard_ix>` in the local directory and upload them to `<hf_username>/<repo_name>-<shard_ix>.<num_shards>` on the remote directory, e.g. like `mikasenghaas/Qwen3-14B-0.2` and `mikasenghaas/Qwen3-14B-1.2`.


