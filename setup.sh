#!/bin/bash

# EMMA Setup Script for Linux/Mac
# Run this script to complete the setup

echo "======================================"
echo "  EMMA Setup Script"
echo "  Gamahea / LEMM Project"
echo "======================================"
echo ""

# Check Python version
echo "Checking Python version..."
if [ -f ".venv/bin/python" ]; then
    .venv/bin/python --version
else
    echo "  Virtual environment not found!"
    echo "  Please run: python -m venv .venv"
    exit 1
fi

echo ""

# Activate virtual environment
source .venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

echo ""

# Create necessary directories
echo "Creating directories..."
dirs=("data" "logs" "output" "output/clips" "models")
for dir in "${dirs[@]}"; do
    if [ ! -d "$dir" ]; then
        mkdir -p "$dir"
        echo "  Created: $dir"
    else
        echo "  Exists: $dir"
    fi
done

echo ""

# Create .env file if it doesn't exist
if [ ! -f ".env" ]; then
    echo "Creating .env file..."
    cp .env.example .env
    echo "  .env file created from .env.example"
    echo "  Please edit .env to configure your settings"
else
    echo ".env file already exists"
fi

echo ""

# Display next steps
echo "======================================"
echo "  Setup Complete!"
echo "======================================"
echo ""
echo "Next Steps:"
echo ""
echo "1. Download AI model weights to models/ directory"
echo "   See README.md for model download links"
echo ""
echo "2. Configure settings in config.yaml and .env"
echo ""
echo "3. Run EMMA:"
echo "   python app.py"
echo ""
echo "4. Open browser to: http://localhost:7860"
echo ""
echo "For help, see README.md or visit the repository"
echo ""
