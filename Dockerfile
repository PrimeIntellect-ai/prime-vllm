# Build stage
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel
LABEL maintainer="prime intellect"
LABEL repository="prime-vllm"

# Set en_US.UTF-8 locale by default
RUN echo "LC_ALL=en_US.UTF-8" >> /etc/environment

# Set CUDA_HOME and update PATH
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=$PATH:/usr/local/cuda/bin

RUN apt-get update && apt-get install -y --no-install-recommends \
  build-essential \
  curl \
  git \
  wget \
  && apt-get clean autoclean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Install uv
ADD https://astral.sh/uv/install.sh /uv-installer.sh
RUN sh /uv-installer.sh && rm /uv-installer.sh
ENV PATH="/root/.local/bin/:$PATH"

WORKDIR /root/prime-vllm

# Copy everything
COPY . .

# Install dependencies
RUN uv venv --python 3.10 .venv && \
    . .venv/bin/activate && \
    uv sync

ENV PYTHONPATH=/root/prime-vllm
# Allow setting HF_TOKEN and CACHE_DIR at runtime

ENTRYPOINT ["uv", "run", "generate.py"]
