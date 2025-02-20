#!/bin/bash

# Create the conda environment
conda create --name unsloth_env \
    python=3.11 \
    pytorch-cuda=12.1 \
    pytorch cudatoolkit xformers -c pytorch -c nvidia -c xformers \
    -y

# Activate the conda environment
conda activate unsloth_env

# Install unsloth and other dependencies
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install git+https://github.com/huggingface/trl.git@e95f9fb74a3c3647b86f251b7e230ec51c64b72b
pip install --no-deps peft accelerate bitsandbytes
pip install diffusers
pip install git+https://github.com/AnswerDotAI/RAGatouille.git
pip install vllm
pip install --upgrade pillow
pip install wandb