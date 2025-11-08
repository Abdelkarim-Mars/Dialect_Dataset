#!/bin/bash
# Quick Start: Generate Arabic Dialect Dataset with Cartesia Sonic 3
# This script runs the complete pipeline from setup to generation

set -e

echo "=========================================="
echo "Arabic Dialect Dataset Generator"
echo "Cartesia Sonic 3 - Natural, Expressive TTS"
echo "=========================================="
echo ""

# Step 1: Check if .env exists
if [ ! -f ".env" ]; then
    echo "❌ Error: .env file not found"
    echo ""
    echo "Please create .env file with your Cartesia API key:"
    echo "  cp .env.template .env"
    echo "  nano .env  # Add your API key"
    echo ""
    exit 1
fi

# Check if API key is set
source .env
if [ -z "$CARTESIA_API_KEY" ] || [ "$CARTESIA_API_KEY" = "your_api_key_here" ]; then
    echo "❌ Error: CARTESIA_API_KEY not set in .env"
    echo ""
    echo "Please edit .env and add your Cartesia API key"
    exit 1
fi

echo "✓ API key configured"
echo ""

# Step 2: Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Virtual environment not found. Run setup.sh first:"
    echo "  ./setup.sh"
    exit 1
fi

# Activate virtual environment
source venv/bin/activate
echo "✓ Virtual environment activated"
echo ""

# Step 3: Check if voice mapping exists
if [ ! -f "dataset/voice_mapping.json" ]; then
    echo "=========================================="
    echo "STEP 1: Voice Discovery"
    echo "=========================================="
    echo ""
    echo "Voice mapping not found. Running voice discovery..."
    echo "This will test available Arabic voices and create mappings."
    echo ""

    read -p "Continue with voice discovery? (y/n) " -n 1 -r
    echo ""

    if [[ $REPLY =~ ^[Yy]$ ]]; then
        python discover_voices.py

        if [ ! -f "dataset/voice_mapping.json" ]; then
            echo ""
            echo "❌ Voice mapping not created. Please run discovery manually:"
            echo "  python discover_voices.py"
            exit 1
        fi
    else
        echo "Skipping voice discovery. Please run manually:"
        echo "  python discover_voices.py"
        exit 1
    fi
fi

echo "✓ Voice mapping exists"
echo ""

# Step 4: Generate dataset
echo "=========================================="
echo "STEP 2: Dataset Generation"
echo "=========================================="
echo ""
echo "This will generate 2,500 Arabic speech utterances."
echo "Estimated time: 4-5 hours (with rate limiting)"
echo ""

read -p "Start dataset generation? (y/n) " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    python generate_dataset.py

    echo ""
    echo "=========================================="
    echo "Generation Complete!"
    echo "=========================================="
    echo ""
    echo "Output files:"
    echo "  - Audio files: dataset/<dialect>/<class>/<token>/"
    echo "  - Manifest: dataset/manifest.csv"
    echo "  - README: dataset/README.md"
    echo "  - Validation report: dataset/validation_report.txt"
    echo ""
    echo "Next steps:"
    echo "  - Review validation report"
    echo "  - Listen to sample files"
    echo "  - Use manifest.csv for ML training"
    echo ""
else
    echo "Generation cancelled. Run manually when ready:"
    echo "  python generate_dataset.py"
fi
