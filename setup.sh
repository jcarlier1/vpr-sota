#!/bin/bash

# VPR SOTA Setup Script
echo "Setting up VPR SOTA environment..."

# Create conda environment
echo "Creating conda environment..."
conda env create -f environment.yml

# Activate environment
echo "Activating environment..."
conda activate vpr-sota

# Create algorithm directories
mkdir -p algorithms/netvlad
mkdir -p algorithms/ap-gem
mkdir -p algorithms/delg
mkdir -p algorithms/cosplace
mkdir -p algorithms/eigenplaces

echo "Environment setup complete!"
echo "To activate the environment, run: conda activate vpr-sota"
