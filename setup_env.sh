#!/bin/bash

# Download the Miniconda installer script
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

# Make the installer executable
chmod +x Miniconda3-latest-Linux-x86_64.sh

# Run the installer
./Miniconda3-latest-Linux-x86_64.sh -b

# Create the conda environment
conda create \
    python=3.11 \
    pytorch-cuda=12.1 \
    pytorch cudatoolkit xformers -c pytorch -c nvidia -c xformers \
    -y

# Activate the conda environment
source ~/miniconda3/bin/activate

# Install dependencies
pip install git+https://github.com/huggingface/trl.git@e95f9fb74a3c3647b86f251b7e230ec51c64b72b
pip install --no-deps peft accelerate bitsandbytes
pip install diffusers
pip install git+https://github.com/AnswerDotAI/RAGatouille.git
pip install vllm
pip install --upgrade pillow
pip install flash-attn --no-build-isolation

# Clone verl repository
git clone https://github.com/volcengine/verl.git && cd verl && pip install -e .