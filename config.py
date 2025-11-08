"""
Configuration module for Cartesia Sonic 3 Arabic Dataset Generator
Centralized settings for natural, expressive audio generation
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ============================================================================
# API Configuration
# ============================================================================

CARTESIA_API_KEY = os.getenv("CARTESIA_API_KEY")
CARTESIA_API_VERSION = os.getenv("CARTESIA_API_VERSION", "2024-06-10")
CARTESIA_BASE_URL = "https://api.cartesia.ai"

# API Endpoints
VOICES_ENDPOINT = f"{CARTESIA_BASE_URL}/voices"
TTS_ENDPOINT = f"{CARTESIA_BASE_URL}/tts/bytes"

# ============================================================================
# Model Configuration - Sonic 3 for Natural, Expressive Audio
# ============================================================================

MODEL_ID = "sonic-multilingual"  # Cartesia Sonic 3 multilingual model
LANGUAGE_CODE = "ar"             # Arabic language code

# Output Format for High-Quality, Natural Audio
OUTPUT_FORMAT = {
    "container": "wav",
    "encoding": "pcm_s16le",     # 16-bit PCM (telephony standard)
    "sample_rate": 16000         # 16 kHz (IVR/ASR optimal)
}

# ============================================================================
# Audio Post-Processing - Preserving Natural Expressiveness
# ============================================================================

TARGET_SAMPLE_RATE = int(os.getenv("TARGET_SAMPLE_RATE", "16000"))
TARGET_PEAK_DBFS = float(os.getenv("TARGET_PEAK_DBFS", "-3.0"))
SILENCE_PADDING_MS = int(os.getenv("SILENCE_PADDING_MS", "150"))

# Audio Quality Targets
TARGET_CHANNELS = 1              # Mono
TARGET_BIT_DEPTH = 16            # 16-bit
TARGET_NOISE_FLOOR = -50.0       # dBFS
TARGET_AVG_LOUDNESS = -20.0      # dBFS

# ============================================================================
# Dataset Structure
# ============================================================================

# Project paths
PROJECT_ROOT = Path(__file__).parent
DATASET_DIR = PROJECT_ROOT / "dataset"
MANIFEST_PATH = DATASET_DIR / "manifest.csv"
VOICE_MAPPING_PATH = DATASET_DIR / "voice_mapping.json"
README_PATH = DATASET_DIR / "README.md"

# Dialect Configuration - 5 Authentic Arabic Regional Dialects
DIALECTS = {
    "EGY": {
        "name": "Egyptian Arabic",
        "speakers": 30,
        "description": "Cairo/Delta region dialect"
    },
    "LAV": {
        "name": "Levantine Arabic",
        "speakers": 25,
        "description": "Syrian/Lebanese/Palestinian dialect"
    },
    "GLF": {
        "name": "Gulf Arabic",
        "speakers": 20,
        "description": "Saudi/Emirati/Kuwaiti dialect"
    },
    "MOR": {
        "name": "Moroccan Arabic (Darija)",
        "speakers": 25,
        "description": "Western North African dialect"
    },
    "TUN": {
        "name": "Tunisian Arabic",
        "speakers": 25,
        "description": "Eastern North African dialect"
    }
}

# Total speakers across all dialects
TOTAL_SPEAKERS = sum(d["speakers"] for d in DIALECTS.values())  # 125

# ============================================================================
# Lexicon - Fixed Banking/IVR Vocabulary
# ============================================================================

# Banking Domain Words (10 tokens)
WORDS = [
    "التنشيط",    # activation
    "التحويل",    # transfer
    "الرصيد",     # balance
    "التسديد",    # settlement/payment
    "نعم",        # yes
    "لا",         # no
    "التمويل",    # financing
    "البيانات",   # data
    "الحساب",     # account
    "إنتهاء"      # finished/end
]

# Number Tokens (10 tokens)
NUMBERS = [
    "صفر",        # 0
    "واحد",       # 1
    "اثنان",      # 2
    "ثلاثة",      # 3
    "أربعة",      # 4
    "خمسة",       # 5
    "ستة",        # 6
    "سبعة",       # 7
    "ثمانية",     # 8
    "تسعة"        # 9
]

# Token classes
TOKEN_CLASSES = {
    "word": WORDS,
    "number": NUMBERS
}

# Total tokens per speaker
TOKENS_PER_SPEAKER = len(WORDS) + len(NUMBERS)  # 20

# ============================================================================
# Generation Settings
# ============================================================================

# Total dataset size
TOTAL_UTTERANCES = TOTAL_SPEAKERS * TOKENS_PER_SPEAKER  # 2,500

# Rate limiting
MAX_RETRIES = 5
RETRY_DELAY_BASE = 2.0           # seconds (exponential backoff)
REQUEST_TIMEOUT = 30.0           # seconds
RATE_LIMIT_DELAY = 1.0           # seconds between requests

# Generation options
SKIP_EXISTING = True             # Resume capability
VALIDATE_AFTER_GENERATION = True

# ============================================================================
# Validation Settings
# ============================================================================

VALIDATION_CHECKS = {
    "file_count": True,
    "audio_format": True,
    "manifest_completeness": True,
    "audio_quality": True,
    "file_existence": True
}

# ============================================================================
# Manifest Schema
# ============================================================================

MANIFEST_COLUMNS = [
    "file_path",
    "dialect",
    "speaker_id",
    "class",
    "token",
    "sample_rate",
    "channels",
    "bit_depth",
    "duration_sec",
    "voice_id",
    "gender",
    "age_group"
]

# ============================================================================
# File Naming Convention
# ============================================================================

def get_filename(dialect: str, speaker_id: str, token_class: str, token: str, take: int = 1) -> str:
    """
    Generate standardized filename for audio files.

    Format: <DIALECT>_<SPKID>_<CLASS>_<TOKEN>_take<NN>.wav
    Example: EGY_spk001_word_التنشيط_take01.wav
    """
    return f"{dialect}_spk{speaker_id:03d}_{token_class}_{token}_take{take:02d}.wav"

def get_file_path(dialect: str, token_class: str, token: str, filename: str) -> Path:
    """
    Generate full file path for audio files.

    Structure: dataset/<DIALECT>/<class>/<token>/<filename>.wav
    """
    return DATASET_DIR / dialect / token_class / token / filename

# ============================================================================
# Logging Configuration
# ============================================================================

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# ============================================================================
# Validation
# ============================================================================

def validate_config():
    """Validate configuration before running generation."""
    errors = []

    if not CARTESIA_API_KEY or CARTESIA_API_KEY == "your_api_key_here":
        errors.append("CARTESIA_API_KEY not set. Copy .env.template to .env and add your API key.")

    if TARGET_PEAK_DBFS > 0:
        errors.append(f"TARGET_PEAK_DBFS must be negative (got {TARGET_PEAK_DBFS})")

    if TARGET_SAMPLE_RATE not in [8000, 16000, 22050, 44100, 48000]:
        errors.append(f"Unusual sample rate: {TARGET_SAMPLE_RATE} Hz")

    if errors:
        raise ValueError("Configuration errors:\n" + "\n".join(f"  - {e}" for e in errors))

    return True

# ============================================================================
# Helper Functions
# ============================================================================

def create_directory_structure():
    """Create dataset directory structure."""
    DATASET_DIR.mkdir(exist_ok=True)

    for dialect in DIALECTS.keys():
        for token_class, tokens in TOKEN_CLASSES.items():
            for token in tokens:
                token_dir = DATASET_DIR / dialect / token_class / token
                token_dir.mkdir(parents=True, exist_ok=True)

    print(f"Created directory structure at {DATASET_DIR}")

if __name__ == "__main__":
    # Test configuration
    try:
        validate_config()
        print("✓ Configuration valid")
        print(f"✓ API Key: {CARTESIA_API_KEY[:10]}...")
        print(f"✓ Model: {MODEL_ID}")
        print(f"✓ Language: {LANGUAGE_CODE}")
        print(f"✓ Total dialects: {len(DIALECTS)}")
        print(f"✓ Total speakers: {TOTAL_SPEAKERS}")
        print(f"✓ Total utterances: {TOTAL_UTTERANCES}")
    except ValueError as e:
        print(f"✗ {e}")
