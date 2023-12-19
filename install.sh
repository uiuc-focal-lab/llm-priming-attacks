#!/bin/bash

# Install Llama 2
git clone https://github.com/facebookresearch/llama.git
mv llama llama_repo
cd llama_repo
pip install -e .

read -p "Download Llama 2 models? (y/n): " choice

if [ "$choice" = "y" ]; then
    chmod u+x download.sh
    ./download.sh
fi

cd ..

# Move Llama 2 files to correct place
mv llama_repo/llama .
mv generation_attack.py llama
rm -rf llama_repo

# Install Llama Guard
read -p "Download Llama Guard model? (y/n): " choice

if [ "$choice" = "y" ]; then
    git clone https://github.com/facebookresearch/PurpleLlama.git
    cd PurpleLlama/Llama-Guard
    chmod u+x download.sh
    ./download.sh
    cd ../..
    rm -rf PurpleLlama
fi
