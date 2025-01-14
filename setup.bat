@echo off
echo Creating the Conda environment... after completion activate using: conda activate XAI

:: Create the Conda environment with a specific Python version (adjust Python version as needed)
conda create -n XAI python=3.9 -y

:: Activate the environment
echo Activating the environment...
call conda activate XAI

:: Install pip packages from requirements.txt
echo Installing pip packages from requirements.txt...
pip install -r requirements.txt

:: Finished message
echo Environment created successfully!
echo You can now activate it using: conda activate XAI
pause
