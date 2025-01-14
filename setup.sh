#!/bin/bash

# Set environment name
ENV_NAME="XAI"

echo "Creating the Conda environment using environment.yml..."

# Create the Conda environment from the environment.yml file
conda env create -f environment.yml -y

# Activate the environment
echo "Activating the environment..."
conda activate $ENV_NAME

# Finished message
echo "Environment created successfully!"
echo "You can now activate it using: conda activate $ENV_NAME"
