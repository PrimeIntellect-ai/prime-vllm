<p align="center">
</p>

<img src="https://github.com/user-attachments/assets/51e44795-5206-49d6-a12a-ecacd2799df2" alt="Prime Intellect" style="width: 100%; height: auto;"/>

---

<p align="center">

<h3 align="center">
[WIP] vLLM-Iroh
</h3>

---

This codebase integrates vLLM with our custom P2P communication library, in order to run pipeline parallel inference across geographically distributed nodes. It works
by hijacking the pipeline parallel vLLM implementation to support sending and receiving intermediate activations, and sampling outputs between nodes using blocking
send and receive primitives that are hooked into vLLM using PyTorch pre- and post-hooks. The codebase has two main entrypoints: 

- `generate.py` is used to generate text given model and generation parameters possibly pipelining over public networks.
- `shard.py` is used to pre-shard models and save them locally or upload them to the Hugging Face Hub.

# Usage

## Installation

**Quick Install:** Run the following command for a quick install:

```
curl -sSL https://raw.githubusercontent.com/PrimeIntellect-ai/vllm-iroh/refs/heads/main/script/install.sh | bash
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
git clone git@github.com:PrimeIntellect-ai/vllm-iroh.git && cd vllm-iroh
uv sync
```

Also, if you plan to use a private model, you will need to set the `HF_TOKEN` environment variable. Also, we recommend setting the `CACHE_DIR` environment variable to a local directory with enough disk space to store the model weights.

```bash
export CACHE_DIR=<path-to-cache-dir>
export HF_TOKEN=<your-token>
```

## Inference

To check that your installation has succeeded, you can run the following command to generate text with a small model on a single node:

```bash
RANK=0 WORLD_SIZE=1 uv run generate.py --model <model_name>
```

Run `uv run python generate.py --help` for more information on the available options.


Running distributed inference is as easy as adjusting the environment variables to your setup and loading a pre-sharded model from HF (see [Sharding](#sharding) for more information). For example, to test distributed inference on two nodes, you can run the following commands. Make sure, that `<shard_name>` corresponds to valid Hf repositories holding the first and second shards of the model, respectively.

```bash
# On the first node
export IROH_SEED=0
export IROH_PEER_ID=ff87a0b0a3c7c0ce827e9cada5ff79e75a44a0633bfcb5b50f99307ddb26b337
RANK=0 WORLD_SIZE=2 uv run generate.py --model <shard_0_name>
```

```bash
# On the second node
export IROH_SEED=1
export IROH_PEER_ID=ee1aa49a4459dfe813a3cf6eb882041230c7b2558469de81f87c9bf23bf10a03
RANK=1 WORLD_SIZE=2 uv run generate.py --model <shard_1_name>
```

## Sharding

If you want to run inference on a custom model, you have to pre-shard it using the `shard.py` script.

```bash
uv run shard.py --model <model_name> --num-shards <num_shards> --local-dir <local_dir> --remote-dir <username>/<repo_name>
```

Here, `<model_name>` is the name of the model you want to shard, `<num_shards>` is the number of shards you want to create, `<local_dir>` is the local directory to save the shards, and `<username>/<repo_name>` is the remote directory to upload the shards. This will automatically save the shards to the local directory and upload them to the remote directory under the name `<username>/<repo_name>-<shard_index>.<num_shards>`.
