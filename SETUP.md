# Setup Guide: Cartesia Sonic 3 Arabic Dataset Generator

Generate 2,500 natural, expressive Arabic speech utterances across 5 authentic dialects using Cartesia's Sonic 3 TTS model.

---

## Prerequisites

### 1. System Requirements
- **Python**: 3.8 or higher
- **ffmpeg**: Required for audio processing
  ```bash
  # Ubuntu/Debian
  sudo apt-get install ffmpeg

  # macOS
  brew install ffmpeg

  # Windows
  # Download from https://ffmpeg.org/
  ```

### 2. Cartesia API Access
- Sign up at [Cartesia AI](https://play.cartesia.ai/)
- Generate an API key from your dashboard
- Ensure you have sufficient API credits for 2,500+ TTS requests

---

## Quick Setup (Automated)

```bash
# Make setup script executable
chmod +x setup.sh

# Run setup
./setup.sh
```

The setup script will:
- Create Python virtual environment
- Install all dependencies
- Create `.env` file from template
- Set up dataset directory structure

---

## Manual Setup

### Step 1: Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate  # Linux/macOS
# or
venv\Scripts\activate     # Windows
```

### Step 2: Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**Dependencies installed:**
- `requests` - Cartesia API communication
- `pandas` - CSV manifest management
- `soundfile` - WAV file I/O
- `pydub` - Audio normalization & processing
- `numpy` - Numerical operations
- `tqdm` - Progress tracking
- `python-dotenv` - Environment variables

### Step 3: Configure API Key

```bash
# Copy template
cp .env.template .env

# Edit .env and add your API key
nano .env  # or use your preferred editor
```

`.env` should contain:
```bash
CARTESIA_API_KEY=your_actual_api_key_here
```

### Step 4: Verify Configuration

```bash
python config.py
```

Expected output:
```
✓ Configuration valid
✓ API Key: sk-abc123...
✓ Model: sonic-multilingual
✓ Language: ar
✓ Total dialects: 5
✓ Total speakers: 125
✓ Total utterances: 2,500
Created directory structure at dataset/
```

---

## Project Structure

After setup, your directory should look like:

```
Dialect_Dataset/
├── venv/                      # Virtual environment
├── dataset/                   # Output directory (created by setup)
│   ├── EGY/                  # Egyptian Arabic
│   ├── LAV/                  # Levantine Arabic
│   ├── GLF/                  # Gulf Arabic
│   ├── MOR/                  # Moroccan Arabic
│   └── TUN/                  # Tunisian Arabic
├── config.py                  # Configuration module
├── requirements.txt           # Python dependencies
├── .env                       # API key (DO NOT COMMIT)
├── .env.template              # Template for .env
├── .gitignore                 # Git ignore rules
└── SETUP.md                   # This file
```

---

## Verify Installation

### Test API Connection

```bash
python -c "
from config import validate_config
validate_config()
print('✓ Configuration valid!')
"
```

### Test Dependencies

```bash
python -c "
import requests
import pandas
import soundfile
import pydub
import numpy
import tqdm
print('✓ All dependencies imported successfully!')
"
```

### Test ffmpeg

```bash
ffmpeg -version
```

---

## Next Steps

Once setup is complete:

1. **Discover Voices** (30 min):
   ```bash
   python discover_voices.py
   ```
   - Finds Arabic-capable Cartesia voices
   - Tests sample utterances
   - Creates `voice_mapping.json`

2. **Generate Dataset** (4-5 hours):
   ```bash
   python generate_dataset.py
   ```
   - Generates 2,500 WAV files
   - Creates `manifest.csv`
   - Auto-generates `README.md`

3. **Validate Output** (15 min):
   ```bash
   python generate_dataset.py --validate-only
   ```
   - Verifies file counts
   - Checks audio format
   - Generates QA report

---

## Troubleshooting

### Error: "CARTESIA_API_KEY not set"
- Ensure `.env` file exists
- Check that API key is not `your_api_key_here`
- Verify no extra spaces around the key

### Error: "ffmpeg not found"
- Install ffmpeg using system package manager
- Restart terminal after installation
- Verify with `ffmpeg -version`

### Error: "Module not found"
- Ensure virtual environment is activated
- Re-run `pip install -r requirements.txt`
- Check Python version (3.8+ required)

### Rate Limiting (429 errors)
- Normal during generation (2,500 requests)
- Script includes automatic retry with exponential backoff
- Generation will continue, just slower

---

## Cost Estimation

- **Total requests**: ~2,500 TTS calls
- **Cartesia pricing**: Check current rates at [Cartesia Pricing](https://cartesia.ai/pricing)
- **Estimated time**: 4-5 hours (with rate limiting)

---

## Support

- **Cartesia API docs**: https://docs.cartesia.ai/
- **Issues**: Check project repository
- **API support**: support@cartesia.ai

---

## Security Notes

- **NEVER commit `.env` file** (already in `.gitignore`)
- **Keep API key confidential**
- **Rotate key if exposed**
- **Monitor API usage** on Cartesia dashboard

---

Ready to generate natural, expressive Arabic speech data! 🎙️
