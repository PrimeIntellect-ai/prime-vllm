#!/usr/bin/env bash

set -e

# Colors for output
GREEN='\033[0;32m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

main() {
    log_info "Cloning repository..."
    if git ls-remote git@github.com:PrimeIntellect-ai/prime-vllm.git &>/dev/null; then
        git clone git@github.com:PrimeIntellect-ai/prime-vllm.git
    else
        git clone https://github.com/PrimeIntellect-ai/prime-vllm.git
    fi
    
    log_info "Entering project directory..."
    cd prime-vllm
    
    log_info "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh

    log_info "Sourcing uv environment..."
    if ! command -v uv &> /dev/null; then
        source $HOME/.local/bin/env
    fi
    
    log_info "Creating virtual environment..."
    uv venv
    
    log_info "Activating virtual environment..."
    source .venv/bin/activate
    
    log_info "Installing dependencies..."
    uv sync
        
    log_info "Installation completed! Check out the README for more information."
}

main