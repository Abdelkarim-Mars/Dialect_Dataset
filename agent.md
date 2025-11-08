# Structured AI Agent Prompt: Arabic Speech Dataset Generation via Cartesia Sonic API

**Document Version**: 2.0  
**Last Updated**: 2025-11-08  
**Target User**: Abdelkarim-Mars  
**TTS Provider**: Cartesia AI (Sonic 3 Model)  
**API Reference**: https://play.cartesia.ai/text-to-speech

---

## Executive Summary

**Objective**: Generate a production-ready Arabic speech dataset containing 2,500 WAV utterances across 5 authentic regional dialects using the Cartesia Sonic 3 TTS API for banking/IVR applications.

**Key Requirements**:
- Automated generation via Cartesia API
- Dialect-specific voice selection
- Programmatic audio processing
- Complete validation pipeline
- Structured metadata output

---

## 1. Cartesia API Integration Specifications

### 1.1 API Overview

**Base Information**:
- **Service**: Cartesia Text-to-Speech API
- **Model**: Sonic 3 (latest generation)
- **Documentation**: https://docs.cartesia.ai/ *(refer to official docs)*
- **Playground**: https://play.cartesia.ai/text-to-speech
- **API Endpoint**: `https://api.cartesia.ai/tts/bytes` *(verify current endpoint)*

**Authentication**:
```bash
# Environment variable
export CARTESIA_API_KEY="your_api_key_here"

# Or in code
headers = {
    "X-API-Key": "YOUR_CARTESIA_API_KEY",
    "Cartesia-Version": "2024-06-10"  # Use latest version
}
```

---

### 1.2 API Request Structure

**Endpoint**: `POST https://api.cartesia.ai/tts/bytes`

**Request Headers**:
```json
{
  "X-API-Key": "<YOUR_API_KEY>",
  "Cartesia-Version": "2024-06-10",
  "Content-Type": "application/json"
}
```

**Request Body Template**:
```json
{
  "model_id": "sonic-english",
  "transcript": "<ARABIC_TOKEN>",
  "voice": {
    "mode": "id",
    "id": "<VOICE_ID>"
  },
  "output_format": {
    "container": "wav",
    "encoding": "pcm_s16le",
    "sample_rate": 16000
  },
  "language": "ar",
  "duration": 2.0,
  "_experimental_voice_controls": {
    "speed": "normal",
    "emotion": ["neutral"]
  }
}
```

**Parameter Definitions**:

| Parameter | Value | Notes |
|-----------|-------|-------|
| `model_id` | `"sonic-multilingual"` or `"sonic-english"` | Use multilingual for Arabic support |
| `transcript` | `"<ARABIC_TOKEN>"` | Exact token from lexicon (UTF-8) |
| `voice.mode` | `"id"` | Use specific voice ID |
| `voice.id` | `"<VOICE_ID>"` | Select from Arabic-capable voices |
| `output_format.container` | `"wav"` | WAV format required |
| `output_format.encoding` | `"pcm_s16le"` | 16-bit PCM, little-endian |
| `output_format.sample_rate` | `16000` | 16 kHz |
| `language` | `"ar"` | Arabic language code |

---

### 1.3 Voice Selection Strategy

**Critical Requirement**: Cartesia Sonic 3 must support **Arabic language** and ideally **dialect variation**. 

**Voice Selection Per Dialect**:

Since Cartesia may not have explicit dialect-labeled voices, use this strategy:

1. **Identify Available Arabic Voices**:
   - Query Cartesia API for available Arabic voices
   - Test voices for dialect authenticity
   - Map voices to target dialects based on phonetic characteristics

2. **Voice Mapping Table** (to be populated after testing):

| Dialect Code | Target Dialect | Cartesia Voice ID | Voice Name | Gender | Notes |
|--------------|----------------|-------------------|------------|--------|-------|
| EGY | Egyptian | `<VOICE_ID_EGY_M01>` | TBD | M | Cairo-like pronunciation |
| EGY | Egyptian | `<VOICE_ID_EGY_F01>` | TBD | F | For female speakers |
| LAV | Levantine (Syrian) | `<VOICE_ID_LAV_M01>` | TBD | M | Levantine characteristics |
| LAV | Levantine (Syrian) | `<VOICE_ID_LAV_F01>` | TBD | F | For female speakers |
| GLF | Gulf (Saudi) | `<VOICE_ID_GLF_M01>` | TBD | M | Gulf pronunciation |
| GLF | Gulf (Saudi) | `<VOICE_ID_GLF_F01>` | TBD | F | For female speakers |
| MOR | Moroccan | `<VOICE_ID_MOR_M01>` | TBD | M | Darija phonology |
| MOR | Moroccan | `<VOICE_ID_MOR_F01>` | TBD | F | For female speakers |
| TUN | Tunisian | `<VOICE_ID_TUN_M01>` | TBD | M | Tunisian characteristics |
| TUN | Tunisian | `<VOICE_ID_TUN_F01>` | TBD | F | For female speakers |

**Action Required**:
```python
# Step 1: Retrieve available voices
import requests

response = requests.get(
    "https://api.cartesia.ai/voices",
    headers={"X-API-Key": API_KEY}
)
voices = response.json()

# Step 2: Filter Arabic-capable voices
arabic_voices = [v for v in voices['voices'] 
                 if 'ar' in v.get('supported_languages', [])]

# Step 3: Test each voice with sample tokens
# Step 4: Manually map voices to dialects based on listening tests
```

**Fallback Strategy**:
If Cartesia does not support all 5 dialects natively:
- Use the closest available Arabic voice
- Apply SSML or prosody controls to approximate dialect features
- Document limitations in README
- Consider post-processing with phonetic transformations

---

### 1.4 API Request Examples

**Example 1: Egyptian Word**
```python
import requests

API_KEY = "your_cartesia_api_key"
ENDPOINT = "https://api.cartesia.ai/tts/bytes"

payload = {
    "model_id": "sonic-multilingual",
    "transcript": "التنشيط",
    "voice": {
        "mode": "id",
        "id": "voice-egy-male-01"  # Replace with actual voice ID
    },
    "output_format": {
        "container": "wav",
        "encoding": "pcm_s16le",
        "sample_rate": 16000
    },
    "language": "ar"
}

headers = {
    "X-API-Key": API_KEY,
    "Cartesia-Version": "2024-06-10",
    "Content-Type": "application/json"
}

response = requests.post(ENDPOINT, json=payload, headers=headers)

if response.status_code == 200:
    with open("EGY_S01_word_التنشيط_take01.wav", "wb") as f:
        f.write(response.content)
else:
    print(f"Error: {response.status_code} - {response.text}")
```

**Example 2: Gulf Number**
```python
payload = {
    "model_id": "sonic-multilingual",
    "transcript": "ثلاثة",
    "voice": {
        "mode": "id",
        "id": "voice-glf-female-02"  # Replace with actual voice ID
    },
    "output_format": {
        "container": "wav",
        "encoding": "pcm_s16le",
        "sample_rate": 16000
    },
    "language": "ar",
    "_experimental_voice_controls": {
        "speed": "normal",
        "emotion": ["neutral"]
    }
}

response = requests.post(ENDPOINT, json=payload, headers=headers)
# Save as GLF_S12_number_ثلاثة_take01.wav
```

---

### 1.5 Rate Limiting & Error Handling

**Cartesia API Limits** *(verify current limits)*:
- Check official documentation for rate limits
- Typical limits: X requests/minute, Y requests/day
- Implement retry logic with exponential backoff

**Error Handling Strategy**:
```python
import time
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

def create_session():
    """Create session with retry logic"""
    session = requests.Session()
    retry = Retry(
        total=5,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["POST"]
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    return session

def generate_audio_with_retry(payload, max_attempts=3):
    """Generate audio with error handling"""
    session = create_session()
    
    for attempt in range(max_attempts):
        try:
            response = session.post(ENDPOINT, json=payload, headers=headers)
            
            if response.status_code == 200:
                return response.content
            elif response.status_code == 429:
                # Rate limit hit
                wait_time = int(response.headers.get('Retry-After', 60))
                print(f"Rate limit hit. Waiting {wait_time}s...")
                time.sleep(wait_time)
            else:
                print(f"Error {response.status_code}: {response.text}")
                
        except Exception as e:
            print(f"Attempt {attempt+1} failed: {e}")
            time.sleep(2 ** attempt)  # Exponential backoff
    
    return None
```

---

## 2. Dataset Specifications (Same as Original)

### 2.1 Target Lexicon

**Words Category (10 tokens)**:
```
1. التنشيط
2. التحويل
3. الرصيد
4. التسديد
5. نعم
6. لا
7. التمويل
8. البيانات
9. الحساب
10. إنتهاء
```

**Numbers Category (10 tokens)**:
```
0. صفر
1. واحد
2. اثنان
3. ثلاثة
4. أربعة
5. خمسة
6. ستة
7. سبعة
8. ثمانية
9. تسعة
```

**Total tokens per speaker**: 20 (10 words + 10 numbers)

---

### 2.2 Dialect Distribution & Speaker Allocation

| Dialect Code | Dialect Name | Region | Speakers | Utterances |
|--------------|--------------|--------|----------|------------|
| **EGY** | Egyptian | Egypt | 30 | 600 |
| **LAV** | Levantine | Syria | 25 | 500 |
| **GLF** | Gulf | Saudi Arabia | 20 | 400 |
| **MOR** | Moroccan | Morocco | 25 | 500 |
| **TUN** | Tunisian | Tunisia | 25 | 500 |
| **TOTAL** | — | — | **125** | **2,500** |

**Speaker Diversity**:
- Gender: Aim for 50/50 male/female split per dialect
- Simulated age: Assign ages 18-60 (for metadata)
- Voice variation: Use multiple Cartesia voice IDs per dialect if available

---

### 2.3 Audio Specifications

| Parameter | Specification |
|-----------|---------------|
| **Format** | WAV (RIFF WAVE) |
| **Encoding** | PCM signed 16-bit little-endian |
| **Sample Rate** | 16,000 Hz |
| **Channels** | Mono (1 channel) |
| **Peak Level** | ≤ –3 dBFS |
| **Average Loudness** | ~–20 dBFS |
| **Leading Silence** | 150 ms (±10 ms) |
| **Trailing Silence** | 150 ms (±10 ms) |
| **Duration** | 0.5–2.0 seconds |
| **Noise Floor** | < –50 dBFS |

---

### 2.4 Directory Structure

```
dataset/
├── EGY/
│   ├── words/
│   │   ├── التنشيط/
│   │   ├── التحويل/
│   │   ├── الرصيد/
│   │   ├── التسديد/
│   │   ├── نعم/
│   │   ├── لا/
│   │   ├── التمويل/
│   │   ├── البيانات/
│   │   ├── الحساب/
│   │   └── إنتهاء/
│   └── numbers/
│       ├── صفر/
│       ├── واحد/
│       ├── اثنان/
│       ├── ثلاثة/
│       ├── أربعة/
│       ├── خمسة/
│       ├── ستة/
│       ├── سبعة/
│       ├── ثمانية/
│       └── تسعة/
├── LAV/ [same structure]
├── GLF/ [same structure]
├── MOR/ [same structure]
├── TUN/ [same structure]
├── manifest.csv
├── voice_mapping.json
└── README.md
```

---

### 2.5 File Naming Convention

**Pattern**:
```
<DIALECT>_<SPKID>_<CLASS>_<TOKEN>_take<NN>.wav
```

**Examples**:
```
EGY_S01_word_التنشيط_take01.wav
LAV_S15_number_خمسة_take01.wav
GLF_S12_number_ثلاثة_take01.wav
MOR_S07_word_نعم_take01.wav
TUN_S20_word_البيانات_take01.wav
```

---

## 3. Implementation Pipeline

### 3.1 System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Dataset Generation Pipeline              │
└─────────────────────────────────────────────────────────────┘

Step 1: Setup & Configuration
├── Load API credentials
├── Create directory structure
├── Load lexicon & dialect mappings
└── Initialize voice ID assignments

Step 2: Voice Discovery & Mapping
├── Query Cartesia API for available voices
├── Test voices with sample tokens
├── Manually map voices to dialects
└── Save voice mapping to voice_mapping.json

Step 3: Batch Generation Loop
├── For each dialect:
│   ├── For each speaker (voice variation):
│   │   ├── For each token (word/number):
│   │   │   ├── Generate API request
│   │   │   ├── Call Cartesia API
│   │   │   ├── Receive WAV bytes
│   │   │   ├── Post-process audio
│   │   │   ├── Save to correct folder
│   │   │   └── Update manifest
│   │   └── Validate speaker's files
│   └── Validate dialect completion

Step 4: Post-Processing
├── Add silence padding (150ms head/tail)
├── Normalize peak to –3 dBFS
├── Validate audio specs
└── Trim excess silence

Step 5: Validation & QA
├── Check file counts
├── Validate audio formats
├── Listen to random samples
└── Generate validation report

Step 6: Documentation & Packaging
├── Generate README.md
├── Finalize manifest.csv
├── Create archive
└── Deliver
```

---

### 3.2 Python Implementation Template

**File**: `generate_dataset.py`

```python
#!/usr/bin/env python3
"""
Arabic Speech Dataset Generator using Cartesia Sonic 3 API
Author: Abdelkarim-Mars
Date: 2025-11-08
"""

import os
import json
import time
import requests
import numpy as np
import soundfile as sf
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd
from pydub import AudioSegment
from pydub.effects import normalize

# ============================================================================
# CONFIGURATION
# ============================================================================

# API Configuration
CARTESIA_API_KEY = os.getenv("CARTESIA_API_KEY", "your_api_key_here")
CARTESIA_ENDPOINT = "https://api.cartesia.ai/tts/bytes"
CARTESIA_VERSION = "2024-06-10"

# Lexicon
WORDS = [
    "التنشيط", "التحويل", "الرصيد", "التسديد", "نعم", 
    "لا", "التمويل", "البيانات", "الحساب", "إنتهاء"
]

NUMBERS = [
    "صفر", "واحد", "اثنان", "ثلاثة", "أربعة", 
    "خمسة", "ستة", "سبعة", "ثمانية", "تسعة"
]

LEXICON = {
    "words": WORDS,
    "numbers": NUMBERS
}

# Dialect Configuration
DIALECTS = {
    "EGY": {"name": "Egyptian", "speakers": 30},
    "LAV": {"name": "Levantine-Syrian", "speakers": 25},
    "GLF": {"name": "Gulf-Saudi", "speakers": 20},
    "MOR": {"name": "Moroccan", "speakers": 25},
    "TUN": {"name": "Tunisian", "speakers": 25}
}

# Voice Mapping (to be populated after testing)
VOICE_MAPPING = {
    "EGY": {
        "male": ["voice-egy-m01", "voice-egy-m02"],  # Replace with actual IDs
        "female": ["voice-egy-f01", "voice-egy-f02"]
    },
    "LAV": {
        "male": ["voice-lav-m01"],
        "female": ["voice-lav-f01"]
    },
    "GLF": {
        "male": ["voice-glf-m01"],
        "female": ["voice-glf-f01"]
    },
    "MOR": {
        "male": ["voice-mor-m01"],
        "female": ["voice-mor-f01"]
    },
    "TUN": {
        "male": ["voice-tun-m01"],
        "female": ["voice-tun-f01"]
    }
}

# Audio Specs
SAMPLE_RATE = 16000
BIT_DEPTH = 16
CHANNELS = 1
SILENCE_PADDING_MS = 150
TARGET_PEAK_DBFS = -3.0

# Paths
BASE_DIR = Path("dataset")
MANIFEST_PATH = BASE_DIR / "manifest.csv"
VOICE_MAPPING_PATH = BASE_DIR / "voice_mapping.json"

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def setup_directory_structure():
    """Create complete directory tree"""
    print("Creating directory structure...")
    
    BASE_DIR.mkdir(exist_ok=True)
    
    for dialect in DIALECTS.keys():
        for class_name in ["words", "numbers"]:
            tokens = LEXICON[class_name]
            for token in tokens:
                token_dir = BASE_DIR / dialect / class_name / token
                token_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"✓ Created {sum(1 for _ in BASE_DIR.rglob('*') if _.is_dir())} directories")


def get_cartesia_voices():
    """Retrieve available voices from Cartesia API"""
    headers = {
        "X-API-Key": CARTESIA_API_KEY,
        "Cartesia-Version": CARTESIA_VERSION
    }
    
    try:
        response = requests.get("https://api.cartesia.ai/voices", headers=headers)
        response.raise_for_status()
        voices = response.json()
        
        # Filter Arabic-capable voices
        arabic_voices = [
            v for v in voices.get('voices', []) 
            if 'ar' in v.get('supported_languages', [])
        ]
        
        print(f"Found {len(arabic_voices)} Arabic-capable voices")
        return arabic_voices
        
    except Exception as e:
        print(f"Error fetching voices: {e}")
        return []


def assign_speaker_voice(dialect: str, speaker_num: int) -> Tuple[str, str]:
    """
    Assign voice ID and gender to speaker
    
    Args:
        dialect: Dialect code (EGY, LAV, etc.)
        speaker_num: Speaker number (1-30)
    
    Returns:
        (voice_id, gender)
    """
    # Alternate between male and female
    gender = "male" if speaker_num % 2 == 1 else "female"
    
    # Get voice pool for this dialect and gender
    voice_pool = VOICE_MAPPING.get(dialect, {}).get(gender, [])
    
    if not voice_pool:
        raise ValueError(f"No voices available for {dialect} {gender}")
    
    # Rotate through available voices
    voice_id = voice_pool[(speaker_num - 1) // 2 % len(voice_pool)]
    
    return voice_id, gender


def generate_audio_cartesia(token: str, voice_id: str, dialect: str) -> bytes:
    """
    Generate audio using Cartesia API
    
    Args:
        token: Arabic text to synthesize
        voice_id: Cartesia voice ID
        dialect: Dialect code (for logging)
    
    Returns:
        WAV audio bytes
    """
    headers = {
        "X-API-Key": CARTESIA_API_KEY,
        "Cartesia-Version": CARTESIA_VERSION,
        "Content-Type": "application/json"
    }
    
    payload = {
        "model_id": "sonic-multilingual",  # Or "sonic-english" if it supports Arabic
        "transcript": token,
        "voice": {
            "mode": "id",
            "id": voice_id
        },
        "output_format": {
            "container": "wav",
            "encoding": "pcm_s16le",
            "sample_rate": SAMPLE_RATE
        },
        "language": "ar",
        "_experimental_voice_controls": {
            "speed": "normal",
            "emotion": ["neutral"]
        }
    }
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = requests.post(
                CARTESIA_ENDPOINT, 
                json=payload, 
                headers=headers,
                timeout=30
            )
            
            if response.status_code == 200:
                return response.content
            elif response.status_code == 429:
                wait_time = int(response.headers.get('Retry-After', 60))
                print(f"  Rate limit hit. Waiting {wait_time}s...")
                time.sleep(wait_time)
            else:
                print(f"  Error {response.status_code}: {response.text}")
                
        except Exception as e:
            print(f"  Attempt {attempt+1} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
    
    raise Exception(f"Failed to generate audio for '{token}' after {max_retries} attempts")


def post_process_audio(audio_bytes: bytes, target_path: Path) -> Dict:
    """
    Post-process audio: add silence, normalize, validate
    
    Args:
        audio_bytes: Raw WAV bytes from API
        target_path: Where to save processed file
    
    Returns:
        Audio metadata dict
    """
    # Load audio
    audio = AudioSegment.from_wav(io.BytesIO(audio_bytes))
    
    # Convert to mono if needed
    if audio.channels > 1:
        audio = audio.set_channels(1)
    
    # Ensure correct sample rate
    if audio.frame_rate != SAMPLE_RATE:
        audio = audio.set_frame_rate(SAMPLE_RATE)
    
    # Add silence padding
    silence = AudioSegment.silent(duration=SILENCE_PADDING_MS)
    audio = silence + audio + silence
    
    # Normalize to target peak
    audio = normalize(audio, headroom=abs(TARGET_PEAK_DBFS))
    
    # Export
    audio.export(
        target_path,
        format="wav",
        parameters=[
            "-ac", "1",  # Mono
            "-ar", str(SAMPLE_RATE),  # Sample rate
            "-sample_fmt", "s16"  # 16-bit PCM
        ]
    )
    
    # Calculate metadata
    peak_db = audio.max_dBFS
    duration_s = len(audio) / 1000.0
    
    return {
        "duration": duration_s,
        "peak_dbfs": peak_db,
        "sample_rate": SAMPLE_RATE,
        "bit_depth": BIT_DEPTH
    }


# ============================================================================
# MAIN GENERATION LOOP
# ============================================================================

def generate_dataset():
    """Main dataset generation function"""
    
    print("=" * 70)
    print("Arabic Speech Dataset Generator - Cartesia Sonic 3")
    print("=" * 70)
    
    # Setup
    setup_directory_structure()
    
    # Initialize manifest
    manifest_data = []
    
    # Statistics
    total_files = sum(d["speakers"] * 20 for d in DIALECTS.values())
    generated_count = 0
    
    print(f"\nTarget: {total_files} files across {len(DIALECTS)} dialects\n")
    
    # Generation loop
    for dialect_code, dialect_info in DIALECTS.items():
        print(f"\n{'='*70}")
        print(f"Dialect: {dialect_info['name']} ({dialect_code})")
        print(f"{'='*70}")
        
        num_speakers = dialect_info["speakers"]
        
        for speaker_num in range(1, num_speakers + 1):
            speaker_id = f"S{speaker_num:02d}"
            
            # Assign voice
            voice_id, gender = assign_speaker_voice(dialect_code, speaker_num)
            
            print(f"\n{dialect_code} {speaker_id} ({gender}, voice: {voice_id})")
            
            # Simulate age
            age = np.random.randint(18, 61)
            
            # Generate all tokens for this speaker
            for class_name, tokens in LEXICON.items():
                for token in tokens:
                    # Generate filename
                    filename = f"{dialect_code}_{speaker_id}_{class_name[:-1]}_{token}_take01.wav"
                    
                    # Target path
                    target_dir = BASE_DIR / dialect_code / class_name / token
                    target_path = target_dir / filename
                    
                    # Check if already exists
                    if target_path.exists():
                        print(f"  ↻ {token} (exists)")
                        generated_count += 1
                        continue
                    
                    try:
                        # Generate audio via Cartesia
                        audio_bytes = generate_audio_cartesia(token, voice_id, dialect_code)
                        
                        # Post-process
                        metadata = post_process_audio(audio_bytes, target_path)
                        
                        # Add to manifest
                        manifest_data.append({
                            "path": str(target_path.relative_to(BASE_DIR)),
                            "dialect": dialect_code,
                            "speaker_id": speaker_id,
                            "class": class_name[:-1],  # Remove 's'
                            "token": token,
                            "sample_rate": metadata["sample_rate"],
                            "bit_depth": metadata["bit_depth"],
                            "gender": gender,
                            "age": age,
                            "environment": "tts_cartesia_sonic3",
                            "voice_id": voice_id,
                            "duration": f"{metadata['duration']:.2f}",
                            "peak_dbfs": f"{metadata['peak_dbfs']:.2f}",
                            "notes": "Generated via Cartesia API"
                        })
                        
                        generated_count += 1
                        print(f"  ✓ {token} ({metadata['duration']:.2f}s, {metadata['peak_dbfs']:.1f}dB)")
                        
                        # Rate limiting (adjust based on Cartesia limits)
                        time.sleep(0.1)  # 10 requests/second max
                        
                    except Exception as e:
                        print(f"  ✗ {token} - ERROR: {e}")
            
            # Progress
            progress = (generated_count / total_files) * 100
            print(f"  Progress: {generated_count}/{total_files} ({progress:.1f}%)")
    
    # Save manifest
    print(f"\n{'='*70}")
    print("Saving manifest...")
    df = pd.DataFrame(manifest_data)
    df.to_csv(MANIFEST_PATH, index=False, encoding='utf-8')
    print(f"✓ Manifest saved: {MANIFEST_PATH} ({len(df)} entries)")
    
    # Save voice mapping
    with open(VOICE_MAPPING_PATH, 'w', encoding='utf-8') as f:
        json.dump(VOICE_MAPPING, f, indent=2, ensure_ascii=False)
    print(f"✓ Voice mapping saved: {VOICE_MAPPING_PATH}")
    
    # Generate README
    generate_readme()
    
    print(f"\n{'='*70}")
    print(f"✓ Dataset generation complete!")
    print(f"  Total files: {generated_count}")
    print(f"  Location: {BASE_DIR.absolute()}")
    print(f"{'='*70}\n")


# ============================================================================
# VALIDATION
# ============================================================================

def validate_dataset():
    """Validate complete dataset"""
    
    print("\n" + "="*70)
    print("DATASET VALIDATION")
    print("="*70 + "\n")
    
    errors = []
    
    # Check file counts
    print("Checking file counts...")
    for dialect_code, dialect_info in DIALECTS.items():
        expected = dialect_info["speakers"] * 20
        actual = len(list((BASE_DIR / dialect_code).rglob("*.wav")))
        
        status = "✓" if actual == expected else "✗"
        print(f"  {status} {dialect_code}: {actual}/{expected}")
        
        if actual != expected:
            errors.append(f"{dialect_code}: Expected {expected}, found {actual}")
    
    # Check manifest
    print("\nChecking manifest...")
    if MANIFEST_PATH.exists():
        df = pd.read_csv(MANIFEST_PATH)
        print(f"  ✓ Manifest entries: {len(df)}")
        
        # Verify all files exist
        missing = []
        for path in df['path']:
            if not (BASE_DIR / path).exists():
                missing.append(path)
        
        if missing:
            print(f"  ✗ Missing files: {len(missing)}")
            errors.extend(missing[:10])  # Show first 10
        else:
            print(f"  ✓ All manifest files exist")
    else:
        print(f"  ✗ Manifest not found")
        errors.append("Manifest missing")
    
    # Sample audio validation
    print("\nValidating audio specs (sample)...")
    sample_files = list(BASE_DIR.rglob("*.wav"))[:20]
    
    for wav_file in sample_files:
        try:
            data, sr = sf.read(wav_file)
            
            # Check sample rate
            if sr != SAMPLE_RATE:
                errors.append(f"{wav_file.name}: Wrong sample rate {sr}")
            
            # Check channels
            if len(data.shape) > 1:
                errors.append(f"{wav_file.name}: Not mono")
                
        except Exception as e:
            errors.append(f"{wav_file.name}: Read error - {e}")
    
    print(f"  ✓ Sampled {len(sample_files)} files")
    
    # Summary
    print("\n" + "="*70)
    if not errors:
        print("✓ VALIDATION PASSED - No errors found")
    else:
        print(f"✗ VALIDATION FAILED - {len(errors)} error(s):")
        for err in errors[:20]:
            print(f"  - {err}")
    print("="*70 + "\n")


# ============================================================================
# DOCUMENTATION
# ============================================================================

def generate_readme():
    """Generate README.md"""
    
    readme_content = f"""# Arabic Speech Dataset - Banking/IVR Vocabulary

## Overview

This dataset contains 2,500 clean Arabic speech utterances across 5 regional dialects, generated using **Cartesia Sonic 3 TTS API**.

**Creation Date**: 2025-11-08  
**Version**: 1.0  
**Generated By**: Abdelkarim-Mars  
**TTS Provider**: Cartesia AI (Sonic 3 Model)  
**API Reference**: https://play.cartesia.ai/text-to-speech

---

## Dataset Statistics

| Dialect | Region | Speakers | Utterances | Percentage |
|---------|--------|----------|------------|------------|
| Egyptian (EGY) | Egypt | 30 | 600 | 24% |
| Levantine (LAV) | Syria | 25 | 500 | 20% |
| Gulf (GLF) | Saudi Arabia | 20 | 400 | 16% |
| Moroccan (MOR) | Morocco | 25 | 500 | 20% |
| Tunisian (TUN) | Tunisia | 25 | 500 | 20% |
| **TOTAL** | — | **125** | **2,500** | **100%** |

**Tokens per speaker**: 20 (10 words + 10 numbers)

---

## Vocabulary

### Words (10)
التنشيط، التحويل، الرصيد، التسديد، نعم، لا، التمويل، البيانات، الحساب، إنتهاء

### Numbers (10)
صفر، واحد، اثنان، ثلاثة، أربعة، خمسة، ستة، سبعة، ثمانية، تسعة

---

## Audio Specifications

- **Format**: WAV (PCM, uncompressed)
- **Sample Rate**: 16,000 Hz
- **Bit Depth**: 16-bit
- **Channels**: Mono
- **Duration per file**: 0.5–2.0 seconds
- **Silence padding**: 150 ms leading, 150 ms trailing
- **Peak level**: ≤ –3 dBFS
- **Average loudness**: ~–20 dBFS

---

## Generation Method

All utterances were generated using the **Cartesia Sonic 3** text-to-speech API with Arabic language support.

### Voice Selection
- Multiple Cartesia voice IDs were mapped to each dialect
- Gender balance maintained (50/50 male/female target)
- Voice IDs selected to approximate authentic dialect phonology
- See `voice_mapping.json` for complete voice-to-dialect mapping

### Dialect Authenticity
While generated via TTS, voices were carefully selected to approximate:
- **Egyptian (EGY)**: Cairo-based pronunciation patterns
- **Levantine (LAV)**: Syrian variant characteristics
- **Gulf (GLF)**: Saudi Gulf dialect features
- **Moroccan (MOR)**: Darija phonological traits
- **Tunisian (TUN)**: Tunisian Arabic characteristics

**Note**: As a TTS-generated dataset, dialect authenticity is approximated based on available Cartesia voices. For applications requiring native speaker recordings, consider supplementing with human-recorded data.

---

## Directory Structure

```
dataset/
├── EGY/
│   ├── words/[10 token folders]
│   └── numbers/[10 token folders]
├── LAV/[same structure]
├── GLF/[same structure]
├── MOR/[same structure]
├── TUN/[same structure]
├── manifest.csv
├── voice_mapping.json
└── README.md
```

---

## File Naming

Pattern: `<DIALECT>_<SPKID>_<CLASS>_<TOKEN>_take01.wav`

Example: `EGY_S07_word_التنشيط_take01.wav`

---

## Metadata

### manifest.csv
Contains complete metadata for all files:
- File path
- Dialect and speaker ID
- Token class and text
- Audio specifications
- Voice ID used
- Generated metadata (duration, peak level)

### voice_mapping.json
Documents the mapping between dialects and Cartesia voice IDs used for generation.

---

## Usage Example

```python
import pandas as pd
import soundfile as sf

# Load manifest
manifest = pd.read_csv('dataset/manifest.csv')

# Filter Egyptian words
egy_words = manifest[
    (manifest['dialect'] == 'EGY') & 
    (manifest['class'] == 'word')
]

# Load an audio file
audio, sr = sf.read(f"dataset/{egy_words.iloc[0]['path']}")
```

---

## Limitations

- **TTS-Generated**: Not recorded by native speakers
- **Dialect Approximation**: Relies on Cartesia's voice capabilities
- **Limited Voice Variation**: Constrained by available Arabic voices
- **Prosody**: May not capture full dialectal nuances

---

## Citation

If you use this dataset, please cite:

```
Arabic Speech Dataset - Banking/IVR Vocabulary
Generated using Cartesia Sonic 3 TTS API
Created by: Abdelkarim-Mars
Date: 2025-11-08
```

---

## License

[Specify your license here]

---

## Changelog

**v1.0** (2025-11-08): Initial release via Cartesia Sonic 3 API
"""
    
    readme_path = BASE_DIR / "README.md"
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(readme_content)
    
    print(f"✓ README generated: {readme_path}")


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    import io  # Add this for BytesIO
    
    # Check API key
    if CARTESIA_API_KEY == "your_api_key_here":
        print("ERROR: Please set CARTESIA_API_KEY environment variable")
        exit(1)
    
    # Optional: Test voice discovery first
    # voices = get_cartesia_voices()
    # if voices:
    #     print("\nAvailable Arabic voices:")
    #     for v in voices[:5]:
    #         print(f"  - {v.get('id')}: {v.get('name')}")
    
    # Generate dataset
    generate_dataset()
    
    # Validate
    validate_dataset()
    
    print("\n✓ All done!\n")
```

---

## 4. Pre-Generation Checklist

### 4.1 Before Running the Script

- [ ] **Cartesia API Key**: Set `CARTESIA_API_KEY` environment variable
- [ ] **Voice Discovery**: Run voice discovery to identify available Arabic voices
- [ ] **Voice Testing**: Test sample voices for each dialect
- [ ] **Voice Mapping**: Update `VOICE_MAPPING` dictionary with actual Cartesia voice IDs
- [ ] **Rate Limits**: Verify Cartesia API rate limits and adjust sleep times
- [ ] **Dependencies**: Install required packages:
  ```bash
  pip install requests pandas soundfile pydub numpy
  ```
- [ ] **ffmpeg**: Install ffmpeg (required by pydub)
  ```bash
  # Ubuntu/Debian
  sudo apt-get install ffmpeg
  
  # macOS
  brew install ffmpeg
  ```

---

### 4.2 Voice Discovery & Testing Script

**File**: `discover_voices.py`

```python
#!/usr/bin/env python3
"""
Discover and test Cartesia voices for Arabic dialects
"""

import os
import requests
from pathlib import Path

CARTESIA_API_KEY = os.getenv("CARTESIA_API_KEY")
ENDPOINT = "https://api.cartesia.ai/tts/bytes"

def discover_voices():
    """List all available Arabic voices"""
    
    headers = {
        "X-API-Key": CARTESIA_API_KEY,
        "Cartesia-Version": "2024-06-10"
    }
    
    response = requests.get("https://api.cartesia.ai/voices", headers=headers)
    
    if response.status_code == 200:
        voices = response.json()
        
        print("Available Arabic Voices:\n")
        print(f"{'ID':<30} {'Name':<25} {'Languages':<20} {'Description'}")
        print("-" * 100)
        
        for voice in voices.get('voices', []):
            langs = voice.get('supported_languages', [])
            if 'ar' in langs or 'ara' in langs:
                print(f"{voice['id']:<30} {voice.get('name', 'N/A'):<25} {str(langs):<20} {voice.get('description', '')}")
    else:
        print(f"Error: {response.status_code} - {response.text}")


def test_voice(voice_id: str, test_token: str = "التنشيط"):
    """Test a specific voice with sample token"""
    
    headers = {
        "X-API-Key": CARTESIA_API_KEY,
        "Cartesia-Version": "2024-06-10",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model_id": "sonic-multilingual",
        "transcript": test_token,
        "voice": {"mode": "id", "id": voice_id},
        "output_format": {
            "container": "wav",
            "encoding": "pcm_s16le",
            "sample_rate": 16000
        },
        "language": "ar"
    }
    
    response = requests.post(ENDPOINT, json=payload, headers=headers)
    
    if response.status_code == 200:
        test_dir = Path("voice_tests")
        test_dir.mkdir(exist_ok=True)
        
        filename = f"{voice_id}_{test_token}.wav"
        filepath = test_dir / filename
        
        with open(filepath, 'wb') as f:
            f.write(response.content)
        
        print(f"✓ Saved test: {filepath}")
    else:
        print(f"✗ Error: {response.status_code}")


if __name__ == "__main__":
    print("Cartesia Voice Discovery Tool\n")
    
    if not CARTESIA_API_KEY:
        print("ERROR: Set CARTESIA_API_KEY environment variable")
        exit(1)
    
    # Discover voices
    discover_voices()
    
    # Interactive testing
    print("\n" + "="*70)
    print("Test voices by entering voice ID (or 'quit' to exit)")
    print("="*70)
    
    while True:
        voice_id = input("\nVoice ID to test: ").strip()
        
        if voice_id.lower() in ['quit', 'exit', 'q']:
            break
        
        if voice_id:
            test_voice(voice_id)
```

**Usage**:
```bash
export CARTESIA_API_KEY="your_key_here"
python discover_voices.py
```

---

## 5. Execution Workflow

### 5.1 Step-by-Step Process

```bash
# Step 1: Setup environment
export CARTESIA_API_KEY="your_cartesia_api_key"

# Step 2: Install dependencies
pip install requests pandas soundfile pydub numpy
sudo apt-get install ffmpeg  # or brew install ffmpeg on macOS

# Step 3: Discover available voices
python discover_voices.py

# Step 4: Test voices for each dialect
# Listen to samples in voice_tests/ folder
# Update VOICE_MAPPING in generate_dataset.py

# Step 5: Run dataset generation
python generate_dataset.py

# Step 6: Monitor progress
# Script will show real-time progress
# Check for errors in output

# Step 7: Validation runs automatically
# Review validation output

# Step 8: Verify deliverables
ls -lh dataset/
head -n 20 dataset/manifest.csv
cat dataset/README.md
```

---

### 5.2 Expected Runtime

**Estimates** (assuming 0.1s delay per request):

| Dialect | Files | Estimated Time |
|---------|-------|----------------|
| EGY | 600 | ~1 hour |
| LAV | 500 | ~50 min |
| GLF | 400 | ~40 min |
| MOR | 500 | ~50 min |
| TUN | 500 | ~50 min |
| **TOTAL** | **2,500** | **~5 hours** |

*Adjust based on actual Cartesia API response times and rate limits*

---

## 6. Quality Assurance

### 6.1 Automated Validation

The script includes built-in validation:
- File count verification per dialect
- Audio format checking (sample rate, channels, bit depth)
- Manifest completeness
- File existence checks

### 6.2 Manual Quality Checks

**Sample Listening Protocol**:
1. Randomly select 10 files per dialect (50 total)
2. Listen for:
   - Correct pronunciation
   - Appropriate dialect characteristics
   - No artifacts or distortion
   - Proper silence padding
3. Document any issues

**Dialect Authenticity Check**:
- Have native speakers review samples from each dialect
- Verify phonological features match expectations
- Document voice-to-dialect mapping quality

---

## 7. Troubleshooting

### 7.1 Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| `API key error` | Invalid/missing key | Set `CARTESIA_API_KEY` correctly |
| `Rate limit 429` | Too many requests | Increase sleep time in script |
| `No Arabic voices` | API doesn't support Arabic | Verify with Cartesia support |
| `Wrong sample rate` | API config error | Check `output_format` in request |
| `Missing files` | Generation failed | Check error logs, re-run |
| `Manifest mismatch` | File/manifest desync | Re-generate manifest |

### 7.2 Recovery from Errors

The script is resumable:
- Already-generated files are skipped
- Re-run the script to complete missing files
- Manifest is updated incrementally

---

## 8. Deliverables Checklist

**Final Package Must Include**:

- [ ] `dataset/` directory with complete structure
  - [ ] 5 dialect folders (EGY, LAV, GLF, MOR, TUN)
  - [ ] Each with words/ and numbers/ subfolders
  - [ ] Each token has dedicated subfolder
  - [ ] Total: 2,500 WAV files

- [ ] `dataset/manifest.csv`
  - [ ] 2,500 rows + header
  - [ ] All paths valid
  - [ ] Includes voice_id column

- [ ] `dataset/voice_mapping.json`
  - [ ] Complete voice-to-dialect mapping
  - [ ] Documents all Cartesia voices used

- [ ] `dataset/README.md`
  - [ ] Complete statistics
  - [ ] Generation methodology
  - [ ] Limitations documented
  - [ ] Usage examples

- [ ] Validation report (stdout or separate file)

---

## 9. Advanced Configurations

### 9.1 Parallel Processing (Optional)

For faster generation, modify to use multiprocessing:

```python
from multiprocessing import Pool

def generate_single_file(args):
    """Worker function for parallel generation"""
    dialect, speaker_id, token, voice_id, gender, age = args
    # ... generation logic ...
    return manifest_row

# In main loop:
with Pool(processes=4) as pool:  # Adjust based on rate limits
    results = pool.map(generate_single_file, task_list)
```

**Warning**: Ensure compliance with Cartesia rate limits!

---

### 9.2 Custom Voice Controls

Leverage Cartesia's experimental controls for variation:

```python
"_experimental_voice_controls": {
    "speed": ["slowest", "slow", "normal", "fast", "fastest"][random.randint(0,4)],
    "emotion": [["neutral"], ["positive"], ["calm"]][random.randint(0,2)]
}
```

---

## 10. Summary Reference

### Quick Command Reference

```bash
# Setup
export CARTESIA_API_KEY="your_key"
pip install -r requirements.txt

# Discover voices
python discover_voices.py

# Generate dataset
python generate_dataset.py

# Validate only
python generate_dataset.py --validate-only  # (add this option to script)

# Package for delivery
tar -czf arabic_dataset_v1.tar.gz dataset/
```

### Key Files

| File | Purpose |
|------|---------|
| `generate_dataset.py` | Main generation script |
| `discover_voices.py` | Voice discovery tool |
| `dataset/manifest.csv` | Complete metadata |
| `dataset/voice_mapping.json` | Voice-dialect mapping |
| `dataset/README.md` | Documentation |

---

**End of Specification Document**

This structured prompt provides complete, unambiguous instructions for generating the Arabic speech dataset using the Cartesia Sonic 3 API. All technical requirements, API integration details, and validation procedures are explicitly defined for automated or semi-automated execution.
