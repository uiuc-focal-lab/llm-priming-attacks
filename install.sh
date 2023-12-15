#!/bin/bash

# Install Llama 2
git clone https://github.com/facebookresearch/llama.git
mv llama llama_repo
cd llama_repo
pip install -e .
chmod u+x download.sh
echo "Downloading Llama 2 models..."
./download.sh
cd ..

# Move Llama 2 files to correct place
mv llama_repo/llama .
mv generation_attack.py llama

# Install Llama Guard
git clone https://github.com/facebookresearch/PurpleLlama.git
cd PurpleLlama/Llama-Guard
chmod u+x download.sh
echo "Downloading Llama Guard model..."
./download.sh
cd ../..