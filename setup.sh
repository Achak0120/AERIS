#!/bin/bash
# Initial setup for your RPi project

echo "Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

echo "Installing Python requirements..."
pip install -r requirements.txt

echo "Creating folder structure..."
mkdir -p data/raw/images
mkdir -p data/processed/images_resized
mkdir -p data/logs
mkdir -p models notebooks vision ai_model microcontrollers control sim analysis research

echo "Setup complete!"
