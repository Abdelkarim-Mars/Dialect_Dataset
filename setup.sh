#!/bin/bash
# Setup script for Cartesia Sonic 3 Arabic Dataset Generator
# Natural, Expressive Speech Dataset Creation

set -e

echo "=========================================="
echo "Cartesia Sonic 3 Dataset Generator Setup"
echo "=========================================="
echo ""

# Check Python version
echo "Checking Python version..."
python3 --version || { echo "Error: Python 3 not found"; exit 1; }
echo ""



# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip
echo ""

# Install dependencies
echo "Installing Python dependencies..."
pip install -r requirements.txt
echo "✓ All dependencies installed"
echo ""

# Check for ffmpeg (required by pydub)
echo "Checking for ffmpeg..."
if command -v ffmpeg &> /dev/null; then
    echo "✓ ffmpeg found: $(ffmpeg -version | head -n1)"
else
    echo "⚠ Warning: ffmpeg not found"
    echo "  pydub requires ffmpeg for audio processing"
    echo "  Install with:"
    echo "    - Ubuntu/Debian: sudo apt-get install ffmpeg"
    echo "    - macOS: brew install ffmpeg"
    echo "    - Windows: Download from https://ffmpeg.org/"
fi
echo ""

# Create .env file if it doesn't exist
if [ ! -f ".env" ]; then
    echo "Creating .env file from template..."
    cp .env.template .env
    echo "✓ .env file created"
    echo "⚠ IMPORTANT: Edit .env and add your CARTESIA_API_KEY"
else
    echo "✓ .env file already exists"
fi
echo ""

# Create dataset directory structure
echo "Creating dataset directory structure..."
python3 config.py
echo ""

echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. Edit .env and add your Cartesia API key"
echo "  2. Run voice discovery: python discover_voices.py"
echo "  3. Generate dataset: python generate_dataset.py"
echo ""
echo "To activate virtual environment in future sessions:"
echo "  source venv/bin/activate"
echo ""
