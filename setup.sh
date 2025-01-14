#!/bin/bash
echo "Creating the Conda environment..."
conda env create -f environment.yml
echo "Environment created successfully!"
echo "Activate the environment using: conda activate project_env"
