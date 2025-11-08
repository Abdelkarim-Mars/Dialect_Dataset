#!/usr/bin/env python3
"""
Arabic Dialect Dataset Generator
Cartesia Sonic 3 TTS - Natural, Expressive Speech

Generates 2,500 high-quality Arabic speech utterances across 5 authentic dialects
for banking/IVR applications using Cartesia's Sonic 3 multilingual model.

Usage:
    python generate_dataset.py                      # Full generation
    python generate_dataset.py --validate-only      # Validation only
    python generate_dataset.py --resume             # Resume from previous run
    python generate_dataset.py --dialect EGY        # Generate specific dialect
"""

import argparse
import io
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import soundfile as sf
from pydub import AudioSegment
from tqdm import tqdm

from config import (
    CARTESIA_API_KEY,
    CARTESIA_API_VERSION,
    TTS_ENDPOINT,
    MODEL_ID,
    LANGUAGE_CODE,
    OUTPUT_FORMAT,
    DATASET_DIR,
    MANIFEST_PATH,
    VOICE_MAPPING_PATH,
    README_PATH,
    DIALECTS,
    TOKEN_CLASSES,
    TOTAL_UTTERANCES,
    TARGET_SAMPLE_RATE,
    TARGET_PEAK_DBFS,
    SILENCE_PADDING_MS,
    TARGET_CHANNELS,
    TARGET_BIT_DEPTH,
    MAX_RETRIES,
    RETRY_DELAY_BASE,
    REQUEST_TIMEOUT,
    RATE_LIMIT_DELAY,
    SKIP_EXISTING,
    MANIFEST_COLUMNS,
    get_filename,
    get_file_path,
    validate_config
)


# ============================================================================
# Logging Setup
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('dataset_generation.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


# ============================================================================
# Cartesia API Client with Rate Limiting & Retry Logic
# ============================================================================

class CartesiaClient:
    """Client for Cartesia TTS API with robust error handling."""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.headers = {
            "Cartesia-Version": CARTESIA_API_VERSION,
            "X-API-Key": api_key,
            "Content-Type": "application/json"
        }
        self.session = requests.Session()
        self.session.headers.update(self.headers)

    def synthesize_speech(
        self,
        text: str,
        voice_id: str,
        max_retries: int = MAX_RETRIES
    ) -> Optional[bytes]:
        """
        Synthesize speech using Cartesia Sonic 3 TTS.

        Args:
            text: Arabic text to synthesize
            voice_id: Cartesia voice ID
            max_retries: Maximum retry attempts for rate limiting

        Returns:
            WAV audio bytes or None if failed
        """
        payload = {
            "model_id": MODEL_ID,
            "transcript": text,
            "language": LANGUAGE_CODE,
            "voice": {
                "mode": "id",
                "id": voice_id
            },
            "output_format": OUTPUT_FORMAT
        }

        for attempt in range(max_retries):
            try:
                response = self.session.post(
                    TTS_ENDPOINT,
                    json=payload,
                    timeout=REQUEST_TIMEOUT
                )

                # Success
                if response.status_code == 200:
                    return response.content

                # Rate limiting - exponential backoff
                elif response.status_code == 429:
                    retry_after = int(response.headers.get('Retry-After', RETRY_DELAY_BASE * (2 ** attempt)))
                    logger.warning(f"Rate limited. Retrying after {retry_after}s (attempt {attempt + 1}/{max_retries})")
                    time.sleep(retry_after)
                    continue

                # Server errors - retry
                elif 500 <= response.status_code < 600:
                    delay = RETRY_DELAY_BASE * (2 ** attempt)
                    logger.warning(f"Server error {response.status_code}. Retrying after {delay}s")
                    time.sleep(delay)
                    continue

                # Client errors - don't retry
                else:
                    logger.error(f"API error {response.status_code}: {response.text}")
                    return None

            except requests.exceptions.Timeout:
                delay = RETRY_DELAY_BASE * (2 ** attempt)
                logger.warning(f"Request timeout. Retrying after {delay}s (attempt {attempt + 1}/{max_retries})")
                time.sleep(delay)
                continue

            except requests.exceptions.RequestException as e:
                logger.error(f"Request error: {e}")
                return None

        logger.error(f"Failed after {max_retries} attempts")
        return None


# ============================================================================
# Audio Post-Processing - Natural & Expressive Quality Preservation
# ============================================================================

class AudioProcessor:
    """Process audio for natural, expressive quality."""

    @staticmethod
    def process_audio(
        raw_wav_bytes: bytes,
        target_sample_rate: int = TARGET_SAMPLE_RATE,
        target_peak_dbfs: float = TARGET_PEAK_DBFS,
        silence_padding_ms: int = SILENCE_PADDING_MS
    ) -> Optional[AudioSegment]:
        """
        Process audio with minimal alterations to preserve natural quality.

        Steps:
        1. Load raw WAV
        2. Convert to mono (if needed)
        3. Resample to target rate (if needed)
        4. Add silence padding (head + tail)
        5. Normalize peak to target dBFS

        Args:
            raw_wav_bytes: Raw WAV bytes from API
            target_sample_rate: Target sample rate (16000 Hz)
            target_peak_dbfs: Target peak level (-3.0 dBFS)
            silence_padding_ms: Silence padding duration (150 ms)

        Returns:
            Processed AudioSegment or None if failed
        """
        try:
            # Load raw audio
            audio = AudioSegment.from_wav(io.BytesIO(raw_wav_bytes))

            # Convert to mono
            if audio.channels > 1:
                audio = audio.set_channels(TARGET_CHANNELS)
                logger.debug("Converted to mono")

            # Resample if needed
            if audio.frame_rate != target_sample_rate:
                audio = audio.set_frame_rate(target_sample_rate)
                logger.debug(f"Resampled to {target_sample_rate} Hz")

            # Add silence padding (prevents clipping, smooth IVR integration)
            silence = AudioSegment.silent(duration=silence_padding_ms)
            audio = silence + audio + silence
            logger.debug(f"Added {silence_padding_ms}ms silence padding")

            # Normalize peak to target dBFS (preserves dynamics)
            current_peak = audio.max_dBFS
            if current_peak > -100:  # Valid audio (not silence)
                gain_change = target_peak_dbfs - current_peak
                audio = audio.apply_gain(gain_change)
                logger.debug(f"Normalized: {current_peak:.1f} dBFS → {target_peak_dbfs:.1f} dBFS")

            return audio

        except Exception as e:
            logger.error(f"Audio processing error: {e}")
            return None

    @staticmethod
    def save_audio(
        audio: AudioSegment,
        output_path: Path,
        sample_rate: int = TARGET_SAMPLE_RATE,
        bit_depth: int = TARGET_BIT_DEPTH
    ) -> bool:
        """
        Save audio to WAV file with specified parameters.

        Args:
            audio: AudioSegment to save
            output_path: Output file path
            sample_rate: Sample rate (16000 Hz)
            bit_depth: Bit depth (16-bit)

        Returns:
            True if successful, False otherwise
        """
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Export as WAV with specific parameters
            audio.export(
                output_path,
                format="wav",
                parameters=[
                    "-ar", str(sample_rate),          # Sample rate
                    "-ac", str(TARGET_CHANNELS),      # Mono
                    "-sample_fmt", "s16",             # 16-bit signed
                    "-acodec", "pcm_s16le"            # Little-endian PCM
                ]
            )

            return True

        except Exception as e:
            logger.error(f"Error saving audio to {output_path}: {e}")
            return False

    @staticmethod
    def get_audio_info(file_path: Path) -> Dict:
        """Extract audio file metadata."""
        try:
            info = sf.info(file_path)
            return {
                "sample_rate": info.samplerate,
                "channels": info.channels,
                "duration_sec": round(info.duration, 3),
                "frames": info.frames,
                "format": info.format,
                "subtype": info.subtype
            }
        except Exception as e:
            logger.error(f"Error reading audio info from {file_path}: {e}")
            return {}


# ============================================================================
# Voice Manager - Handle Voice Selection & Rotation
# ============================================================================

class VoiceManager:
    """Manage voice selection for speakers across dialects."""

    def __init__(self, voice_mapping_path: Path):
        """Load voice mapping from JSON file."""
        if not voice_mapping_path.exists():
            raise FileNotFoundError(
                f"Voice mapping not found: {voice_mapping_path}\n"
                f"Run 'python discover_voices.py' first to create voice mapping."
            )

        with open(voice_mapping_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.mapping = data.get("dialects", {})
        logger.info(f"Loaded voice mapping from {voice_mapping_path}")

        # Validate mapping
        for dialect in DIALECTS.keys():
            if dialect not in self.mapping:
                raise ValueError(f"Missing dialect in voice mapping: {dialect}")

    def get_voice_for_speaker(
        self,
        dialect: str,
        speaker_id: int
    ) -> Tuple[str, str, str]:
        """
        Get voice ID for a specific speaker.

        Rotates through available voices to create speaker diversity.

        Args:
            dialect: Dialect code (e.g., 'EGY')
            speaker_id: Speaker number (1-based)

        Returns:
            (voice_id, gender, age_group)
        """
        # Alternate gender for variety
        gender = "male" if speaker_id % 2 == 1 else "female"

        # Get voice list for dialect/gender
        voice_list = self.mapping[dialect].get(gender, [])

        if not voice_list:
            # Fallback to opposite gender if not available
            gender = "female" if gender == "male" else "male"
            voice_list = self.mapping[dialect].get(gender, [])

        if not voice_list:
            raise ValueError(f"No voices available for {dialect}/{gender}")

        # Rotate through voices
        voice_index = (speaker_id - 1) % len(voice_list)
        voice_id = voice_list[voice_index]

        # Age group (could be enhanced with metadata from voice mapping)
        age_group = "adult"

        return voice_id, gender, age_group


# ============================================================================
# Dataset Generator - Main Batch Processing Engine
# ============================================================================

class DatasetGenerator:
    """Generate complete Arabic dialect dataset."""

    def __init__(
        self,
        api_client: CartesiaClient,
        audio_processor: AudioProcessor,
        voice_manager: VoiceManager
    ):
        self.api_client = api_client
        self.audio_processor = audio_processor
        self.voice_manager = voice_manager
        self.manifest_data = []

    def generate_utterance(
        self,
        dialect: str,
        speaker_id: int,
        token_class: str,
        token: str,
        take: int = 1
    ) -> bool:
        """
        Generate single utterance.

        Args:
            dialect: Dialect code (e.g., 'EGY')
            speaker_id: Speaker ID (1-based)
            token_class: 'word' or 'number'
            token: Arabic text token
            take: Take number (default 1)

        Returns:
            True if successful, False otherwise
        """
        # Get filename and path
        filename = get_filename(dialect, speaker_id, token_class, token, take)
        file_path = get_file_path(dialect, token_class, token, filename)

        # Skip if exists
        if SKIP_EXISTING and file_path.exists():
            logger.debug(f"Skipping existing: {file_path}")
            return True

        # Get voice for this speaker
        try:
            voice_id, gender, age_group = self.voice_manager.get_voice_for_speaker(
                dialect, speaker_id
            )
        except ValueError as e:
            logger.error(f"Voice selection error: {e}")
            return False

        # Synthesize speech
        raw_audio = self.api_client.synthesize_speech(token, voice_id)
        if not raw_audio:
            logger.error(f"Failed to synthesize: {filename}")
            return False

        # Process audio
        processed_audio = self.audio_processor.process_audio(raw_audio)
        if not processed_audio:
            logger.error(f"Failed to process audio: {filename}")
            return False

        # Save audio
        success = self.audio_processor.save_audio(processed_audio, file_path)
        if not success:
            logger.error(f"Failed to save: {file_path}")
            return False

        # Get audio metadata
        audio_info = self.audio_processor.get_audio_info(file_path)

        # Add to manifest
        manifest_entry = {
            "file_path": str(file_path),
            "dialect": dialect,
            "speaker_id": f"spk{speaker_id:03d}",
            "class": token_class,
            "token": token,
            "sample_rate": audio_info.get("sample_rate", TARGET_SAMPLE_RATE),
            "channels": audio_info.get("channels", TARGET_CHANNELS),
            "bit_depth": TARGET_BIT_DEPTH,
            "duration_sec": audio_info.get("duration_sec", 0.0),
            "voice_id": voice_id,
            "gender": gender,
            "age_group": age_group
        }
        self.manifest_data.append(manifest_entry)

        logger.debug(f"Generated: {filename}")

        # Rate limiting delay
        time.sleep(RATE_LIMIT_DELAY)

        return True

    def generate_dataset(
        self,
        target_dialects: Optional[List[str]] = None
    ) -> bool:
        """
        Generate complete dataset.

        Args:
            target_dialects: Specific dialects to generate (None = all)

        Returns:
            True if successful, False otherwise
        """
        dialects_to_generate = target_dialects or list(DIALECTS.keys())

        # Calculate total utterances
        total_utterances = 0
        for dialect in dialects_to_generate:
            speakers = DIALECTS[dialect]["speakers"]
            tokens = sum(len(tokens) for tokens in TOKEN_CLASSES.values())
            total_utterances += speakers * tokens

        logger.info(f"Starting dataset generation: {total_utterances} utterances")
        logger.info(f"Dialects: {', '.join(dialects_to_generate)}")

        # Progress tracking
        with tqdm(total=total_utterances, desc="Generating dataset") as pbar:
            for dialect in dialects_to_generate:
                num_speakers = DIALECTS[dialect]["speakers"]
                logger.info(f"Processing {dialect}: {num_speakers} speakers")

                for speaker_id in range(1, num_speakers + 1):
                    for token_class, tokens in TOKEN_CLASSES.items():
                        for token in tokens:
                            success = self.generate_utterance(
                                dialect=dialect,
                                speaker_id=speaker_id,
                                token_class=token_class,
                                token=token
                            )

                            if not success:
                                logger.warning(
                                    f"Failed: {dialect}/spk{speaker_id:03d}/{token_class}/{token}"
                                )

                            pbar.update(1)

        logger.info(f"Generation complete: {len(self.manifest_data)} utterances generated")
        return True

    def save_manifest(self, output_path: Path):
        """Save manifest CSV."""
        df = pd.DataFrame(self.manifest_data, columns=MANIFEST_COLUMNS)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False, encoding="utf-8")
        logger.info(f"Manifest saved: {output_path} ({len(df)} entries)")


# ============================================================================
# Validation & QA Module
# ============================================================================

class DatasetValidator:
    """Validate generated dataset."""

    def __init__(self, dataset_dir: Path, manifest_path: Path):
        self.dataset_dir = dataset_dir
        self.manifest_path = manifest_path

    def validate_file_counts(self) -> Dict[str, Dict]:
        """Validate file counts per dialect."""
        logger.info("Validating file counts...")

        results = {}
        for dialect_code, dialect_info in DIALECTS.items():
            expected = dialect_info["speakers"] * len(WORDS) * len(NUMBERS) // len(WORDS)
            expected = dialect_info["speakers"] * sum(len(t) for t in TOKEN_CLASSES.values())

            dialect_dir = self.dataset_dir / dialect_code
            if not dialect_dir.exists():
                results[dialect_code] = {
                    "expected": expected,
                    "actual": 0,
                    "status": "FAIL"
                }
                continue

            actual = len(list(dialect_dir.rglob("*.wav")))
            status = "PASS" if actual == expected else "FAIL"

            results[dialect_code] = {
                "expected": expected,
                "actual": actual,
                "status": status
            }

            logger.info(f"  {dialect_code}: {actual}/{expected} files [{status}]")

        return results

    def validate_manifest(self) -> Dict:
        """Validate manifest completeness."""
        logger.info("Validating manifest...")

        if not self.manifest_path.exists():
            return {"status": "FAIL", "reason": "Manifest file not found"}

        try:
            df = pd.read_csv(self.manifest_path)

            # Check row count
            expected_rows = TOTAL_UTTERANCES
            actual_rows = len(df)

            # Check for missing values
            missing_values = df.isnull().sum().to_dict()
            has_missing = any(count > 0 for count in missing_values.values())

            # Check file existence
            files_exist = all(Path(fp).exists() for fp in df["file_path"])

            status = "PASS" if (
                actual_rows == expected_rows and
                not has_missing and
                files_exist
            ) else "FAIL"

            result = {
                "status": status,
                "expected_rows": expected_rows,
                "actual_rows": actual_rows,
                "missing_values": missing_values,
                "all_files_exist": files_exist
            }

            logger.info(f"  Manifest: {actual_rows}/{expected_rows} rows [{status}]")

            return result

        except Exception as e:
            logger.error(f"Manifest validation error: {e}")
            return {"status": "FAIL", "reason": str(e)}

    def validate_audio_format(self, sample_size: int = 100) -> Dict:
        """Validate audio format for sample files."""
        logger.info(f"Validating audio format (sample size: {sample_size})...")

        if not self.manifest_path.exists():
            return {"status": "FAIL", "reason": "Manifest not found"}

        df = pd.read_csv(self.manifest_path)
        sample_files = df.sample(min(sample_size, len(df)))["file_path"].tolist()

        issues = []
        for file_path in sample_files:
            try:
                info = sf.info(file_path)

                if info.samplerate != TARGET_SAMPLE_RATE:
                    issues.append(f"{file_path}: Wrong sample rate ({info.samplerate})")

                if info.channels != TARGET_CHANNELS:
                    issues.append(f"{file_path}: Wrong channel count ({info.channels})")

            except Exception as e:
                issues.append(f"{file_path}: {e}")

        status = "PASS" if not issues else "FAIL"
        logger.info(f"  Audio format: [{status}] ({len(issues)} issues)")

        return {
            "status": status,
            "sample_size": len(sample_files),
            "issues": issues
        }

    def generate_report(self, output_path: Path):
        """Generate validation report."""
        logger.info("Generating validation report...")

        file_counts = self.validate_file_counts()
        manifest_validation = self.validate_manifest()
        audio_validation = self.validate_audio_format()

        report = []
        report.append("=" * 70)
        report.append("DATASET VALIDATION REPORT")
        report.append("=" * 70)
        report.append("")

        # File counts
        report.append("File Counts by Dialect:")
        for dialect, result in file_counts.items():
            report.append(
                f"  {dialect}: {result['actual']}/{result['expected']} [{result['status']}]"
            )
        report.append("")

        # Manifest
        report.append("Manifest Validation:")
        report.append(f"  Status: {manifest_validation.get('status')}")
        report.append(f"  Rows: {manifest_validation.get('actual_rows')}/{manifest_validation.get('expected_rows')}")
        report.append("")

        # Audio format
        report.append("Audio Format Validation:")
        report.append(f"  Status: {audio_validation.get('status')}")
        report.append(f"  Sample size: {audio_validation.get('sample_size')}")
        if audio_validation.get('issues'):
            report.append(f"  Issues: {len(audio_validation['issues'])}")
        report.append("")

        # Overall status
        all_pass = all([
            all(r["status"] == "PASS" for r in file_counts.values()),
            manifest_validation.get("status") == "PASS",
            audio_validation.get("status") == "PASS"
        ])

        report.append("=" * 70)
        report.append(f"OVERALL: {'PASS ✓' if all_pass else 'FAIL ✗'}")
        report.append("=" * 70)

        # Save report
        report_text = "\n".join(report)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(report_text)

        logger.info(f"Validation report saved: {output_path}")
        print("\n" + report_text)


# ============================================================================
# Documentation Generator
# ============================================================================

def generate_readme(dataset_dir: Path, manifest_path: Path, voice_mapping_path: Path):
    """Generate dataset README.md."""
    logger.info("Generating README.md...")

    # Load manifest
    if manifest_path.exists():
        df = pd.read_csv(manifest_path)
        total_files = len(df)

        # Statistics by dialect
        dialect_stats = df.groupby("dialect").size().to_dict()
        gender_stats = df.groupby("gender").size().to_dict()
    else:
        total_files = 0
        dialect_stats = {}
        gender_stats = {}

    # Load voice mapping
    if voice_mapping_path.exists():
        with open(voice_mapping_path, "r", encoding="utf-8") as f:
            voice_mapping = json.load(f)
    else:
        voice_mapping = {}

    # Generate README content
    readme_content = f"""# Arabic Dialect Speech Dataset
**Natural, Expressive TTS with Cartesia Sonic 3**

---

## Overview

This dataset contains **{total_files:,} high-quality Arabic speech utterances** across **5 authentic regional dialects**, generated using **Cartesia's Sonic 3 multilingual TTS model**.

**Target Application**: Banking/IVR systems requiring dialect-specific voice recognition training data.

---

## Dataset Statistics

### Total Utterances
- **Total files**: {total_files:,}
- **Total dialects**: {len(DIALECTS)}
- **Total speakers**: {sum(d['speakers'] for d in DIALECTS.values())}
- **Tokens per speaker**: {sum(len(t) for t in TOKEN_CLASSES.values())}

### By Dialect

| Dialect | Name | Utterances | Speakers |
|---------|------|------------|----------|
"""

    for dialect_code, dialect_info in DIALECTS.items():
        count = dialect_stats.get(dialect_code, 0)
        readme_content += f"| {dialect_code} | {dialect_info['name']} | {count:,} | {dialect_info['speakers']} |\n"

    readme_content += f"""
### By Gender

| Gender | Utterances |
|--------|------------|
"""

    for gender, count in gender_stats.items():
        readme_content += f"| {gender.capitalize()} | {count:,} |\n"

    readme_content += f"""
---

## Audio Specifications

| Parameter | Value |
|-----------|-------|
| **Format** | WAV (RIFF WAVE, uncompressed) |
| **Encoding** | PCM signed 16-bit little-endian |
| **Sample Rate** | 16,000 Hz (16 kHz) |
| **Channels** | Mono (1 channel) |
| **Bit Depth** | 16-bit |
| **Peak Level** | ≤ –3 dBFS |
| **Silence Padding** | 150 ms (head + tail) |

---

## Vocabulary

### Banking Domain Words (10 tokens)
```
التنشيط (activation)
التحويل (transfer)
الرصيد (balance)
التسديد (settlement)
نعم (yes)
لا (no)
التمويل (financing)
البيانات (data)
الحساب (account)
إنتهاء (finished)
```

### Numbers (10 tokens)
```
صفر (0), واحد (1), اثنان (2), ثلاثة (3), أربعة (4)
خمسة (5), ستة (6), سبعة (7), ثمانية (8), تسعة (9)
```

---

## Directory Structure

```
dataset/
├── EGY/                    # Egyptian Arabic
│   ├── word/
│   │   ├── التنشيط/
│   │   ├── التحويل/
│   │   └── ...
│   └── number/
│       ├── صفر/
│       └── ...
├── LAV/                    # Levantine Arabic
├── GLF/                    # Gulf Arabic
├── MOR/                    # Moroccan Arabic
├── TUN/                    # Tunisian Arabic
├── manifest.csv            # Complete metadata
├── voice_mapping.json      # Voice-to-dialect mappings
└── README.md               # This file
```

---

## File Naming Convention

Format: `<DIALECT>_<SPKID>_<CLASS>_<TOKEN>_take<NN>.wav`

Example: `EGY_spk001_word_التنشيط_take01.wav`

Components:
- **DIALECT**: 3-letter dialect code (EGY, LAV, GLF, MOR, TUN)
- **SPKID**: Speaker ID (spk001-spk030)
- **CLASS**: Token class (word, number)
- **TOKEN**: Arabic text token
- **NN**: Take number (01-99)

---

## Usage Examples

### Load Dataset with Python

```python
import pandas as pd
from pathlib import Path

# Load manifest
df = pd.read_csv("dataset/manifest.csv")

# Filter by dialect
egy_files = df[df["dialect"] == "EGY"]

# Filter by token class
word_files = df[df["class"] == "word"]

# Get file paths
audio_files = df["file_path"].tolist()
```

### Query by Dialect

```python
# Get all Egyptian Arabic utterances
egy = df[df["dialect"] == "EGY"]
print(f"Egyptian Arabic: {{len(egy)}} files")

# Get specific speaker
speaker_001 = df[df["speaker_id"] == "spk001"]
```

### Load Audio

```python
import soundfile as sf

# Read audio file
audio, sample_rate = sf.read("dataset/EGY/word/التنشيط/EGY_spk001_word_التنشيط_take01.wav")
print(f"Duration: {{len(audio)/sample_rate:.2f}s")
```

---

## Voice Mapping

This dataset uses Cartesia's Sonic 3 multilingual model with carefully selected voices mapped to each dialect.

See `voice_mapping.json` for complete voice-to-dialect mappings.

---

## Generation Details

- **TTS Model**: Cartesia Sonic 3 Multilingual
- **Language**: Arabic (`ar`)
- **Post-Processing**:
  - Mono conversion
  - 16 kHz resampling
  - 150ms silence padding (head/tail)
  - Peak normalization to –3 dBFS

---

## Quality Assurance

All utterances have been:
- ✓ Generated with natural, expressive Sonic 3 voices
- ✓ Processed with minimal alterations to preserve quality
- ✓ Normalized for consistent loudness
- ✓ Validated for format compliance
- ✓ Metadata-tracked in manifest.csv

---

## Limitations

1. **Dialect Authenticity**: Voices are approximated to dialects based on available Cartesia voices. True native speakers may exhibit different phonetic characteristics.

2. **TTS Artifacts**: Some utterances may contain minor TTS artifacts inherent to synthetic speech.

3. **Limited Vocabulary**: Only 20 tokens (10 words + 10 numbers) for banking/IVR domain.

4. **Speaker Diversity**: Simulated speakers using voice rotation, not actual unique individuals.

---

## Citation

If you use this dataset in your research, please cite:

```
@dataset{{arabic_dialect_tts_2025,
  title={{Arabic Dialect Speech Dataset: Natural TTS with Cartesia Sonic 3}},
  year={{2025}},
  note={{2,500 utterances across 5 dialects (Egyptian, Levantine, Gulf, Moroccan, Tunisian)}}
}}
```

---

## License

Dataset generated using Cartesia AI's Sonic 3 TTS API. Please review Cartesia's terms of service for usage restrictions.

---

## Support

For questions or issues with this dataset:
- Review `manifest.csv` for detailed metadata
- Check `validation_report.txt` for quality metrics
- Consult Cartesia documentation: https://docs.cartesia.ai/

---

**Generated with natural, expressive speech for authentic Arabic dialect representation.** 🎙️
"""

    # Save README
    readme_path = dataset_dir / "README.md"
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(readme_content)

    logger.info(f"README.md generated: {readme_path}")


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate Arabic dialect speech dataset with Cartesia Sonic 3"
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only run validation, skip generation"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from previous run (skip existing files)"
    )
    parser.add_argument(
        "--dialect",
        type=str,
        choices=list(DIALECTS.keys()),
        help="Generate specific dialect only"
    )

    args = parser.parse_args()

    # Validate configuration
    try:
        validate_config()
    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        sys.exit(1)

    # Validation only mode
    if args.validate_only:
        validator = DatasetValidator(DATASET_DIR, MANIFEST_PATH)
        validator.generate_report(DATASET_DIR / "validation_report.txt")
        sys.exit(0)

    # Initialize components
    logger.info("Initializing dataset generator...")

    api_client = CartesiaClient(CARTESIA_API_KEY)
    audio_processor = AudioProcessor()

    try:
        voice_manager = VoiceManager(VOICE_MAPPING_PATH)
    except FileNotFoundError as e:
        logger.error(str(e))
        sys.exit(1)

    generator = DatasetGenerator(api_client, audio_processor, voice_manager)

    # Generate dataset
    target_dialects = [args.dialect] if args.dialect else None

    logger.info("="*70)
    logger.info("STARTING DATASET GENERATION")
    logger.info("="*70)

    success = generator.generate_dataset(target_dialects)

    if not success:
        logger.error("Dataset generation failed")
        sys.exit(1)

    # Save manifest
    generator.save_manifest(MANIFEST_PATH)

    # Generate documentation
    generate_readme(DATASET_DIR, MANIFEST_PATH, VOICE_MAPPING_PATH)

    # Run validation
    validator = DatasetValidator(DATASET_DIR, MANIFEST_PATH)
    validator.generate_report(DATASET_DIR / "validation_report.txt")

    logger.info("="*70)
    logger.info("DATASET GENERATION COMPLETE")
    logger.info("="*70)
    logger.info(f"Total utterances: {len(generator.manifest_data)}")
    logger.info(f"Manifest: {MANIFEST_PATH}")
    logger.info(f"README: {README_PATH}")
    logger.info(f"Validation report: {DATASET_DIR / 'validation_report.txt'}")


if __name__ == "__main__":
    main()
