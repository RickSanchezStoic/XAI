#!/bin/bash

# Set environment name
ENV_NAME="XAI"

echo "Creating the Conda environment..."

# Create the Conda environment with a specific Python version (use the desired version)
conda create -n $ENV_NAME python=3.9 -y

# Activate the environment
echo "Activating the environment..."
conda activate $ENV_NAME

# Install pip packages from requirements.txt
echo "Installing pip packages from requirements.txt..."
pip install -r requirements.txt

# Finished message
echo "Environment created successfully!"
echo "You can now activate it using: conda activate $ENV_NAME"
