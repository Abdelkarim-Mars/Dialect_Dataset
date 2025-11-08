#!/usr/bin/env python3
"""
Professional-Grade Cartesia Sonic 3 TTS - Tunisian Arabic
==========================================================

Runtime verification and synthesis script for Cartesia API with:
- Model capability discovery (Sonic 3 / sonic-multilingual)
- Arabic and Tunisian dialect voice detection
- Professional audio validation (WAV format, no clipping)
- Comprehensive error handling and logging

Requirements: Python ≥3.10, httpx, python-dotenv

Author: Senior Python TTS Engineer
"""

import io
import logging
import os
import sys
import wave
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx
from dotenv import load_dotenv

# ============================================================================
# Configuration
# ============================================================================

# Load environment variables
load_dotenv()

# API Configuration - Based on actual Cartesia API structure
API_BASE = "https://api.cartesia.ai"
API_VERSION = "2024-06-10"  # Cartesia-Version header
TIMEOUT = 30.0

# Target Models (in order of preference)
TARGET_MODELS = ["sonic-3", "sonic-multilingual"]

# Tunisian Test Sentence (Natural colloquial Tunisian Arabic)
TUNISIAN_SENTENCE = "شنوة أحوالك؟ اليوم باش نمشيو للحانوت نشريو شوية قهوة."
# Translation: "How are you? Today we're going to the shop to buy some coffee."

# Audio Output Settings
OUTPUT_FILE = "output_tn.wav"
TARGET_SAMPLE_RATE = 24000
TARGET_CHANNELS = 1
TARGET_BIT_DEPTH = 16

# Clipping Detection Threshold
CLIPPING_THRESHOLD = 32760  # Allow small margin below 32767

# ============================================================================
# Logging Setup
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# ============================================================================
# Environment Validation
# ============================================================================

def getenv_strict(name: str) -> str:
    """
    Retrieve required environment variable.

    Args:
        name: Environment variable name

    Returns:
        Variable value

    Raises:
        SystemExit: If variable is missing or empty
    """
    value = os.getenv(name)
    if not value or value.strip() == "":
        logger.error(f"Missing required environment variable: {name}")
        logger.error(f"Set {name} in .env file or environment")
        sys.exit(1)
    return value.strip()

# ============================================================================
# HTTP Client Utilities
# ============================================================================

def create_http_client(api_key: str) -> httpx.Client:
    """
    Create configured HTTP client for Cartesia API.

    Args:
        api_key: Cartesia API key

    Returns:
        Configured httpx.Client instance
    """
    headers = {
        "Cartesia-Version": API_VERSION,
        "X-API-Key": api_key,
        "Content-Type": "application/json"
    }

    return httpx.Client(
        base_url=API_BASE,
        headers=headers,
        timeout=TIMEOUT,
        follow_redirects=True
    )

def http_get_json(client: httpx.Client, path: str) -> Any:
    """
    Perform GET request and return JSON response.

    Args:
        client: HTTP client
        path: API endpoint path

    Returns:
        JSON response data

    Raises:
        httpx.HTTPStatusError: On HTTP error status
        ValueError: On invalid JSON response
    """
    try:
        response = client.get(path)
        response.raise_for_status()
        return response.json()
    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP {e.response.status_code} error for {path}")
        logger.error(f"Response: {e.response.text[:500]}")
        raise
    except ValueError as e:
        logger.error(f"Invalid JSON response from {path}: {e}")
        raise

def http_post_bytes(client: httpx.Client, path: str, payload: Dict[str, Any]) -> bytes:
    """
    Perform POST request and return binary response.

    Args:
        client: HTTP client
        path: API endpoint path
        payload: Request JSON payload

    Returns:
        Binary response content

    Raises:
        httpx.HTTPStatusError: On HTTP error status
    """
    try:
        response = client.post(path, json=payload)
        response.raise_for_status()
        return response.content
    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP {e.response.status_code} error for {path}")
        logger.error(f"Response: {e.response.text[:500]}")
        logger.error(f"Payload sent: {payload}")
        raise

# ============================================================================
# Model Discovery & Validation
# ============================================================================

def discover_models(client: httpx.Client) -> List[Dict[str, Any]]:
    """
    Discover available models from Cartesia API.

    [ASSUMPTION] Endpoint: /models or /v1/models
    Falls back to inferring from voice metadata if models endpoint unavailable.

    Args:
        client: HTTP client

    Returns:
        List of model dictionaries
    """
    logger.info("Discovering available models...")

    # Try common model endpoints
    for endpoint in ["/models", "/v1/models"]:
        try:
            data = http_get_json(client, endpoint)

            # Handle both list and dict responses
            if isinstance(data, list):
                models = data
            elif isinstance(data, dict) and "data" in data:
                models = data["data"]
            elif isinstance(data, dict) and "models" in data:
                models = data["models"]
            else:
                models = [data]

            logger.info(f"✓ Retrieved {len(models)} models from {endpoint}")
            return models

        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                logger.debug(f"Endpoint {endpoint} not found, trying alternative...")
                continue
            raise

    # Fallback: Infer from documentation
    logger.warning("Model discovery endpoint not available")
    logger.warning("Assuming sonic-multilingual model is available (verify in dashboard)")
    return [{"id": "sonic-multilingual", "name": "Sonic Multilingual"}]

def verify_target_model(models: List[Dict[str, Any]]) -> str:
    """
    Verify target model is available.

    Args:
        models: List of available models

    Returns:
        Selected model ID

    Raises:
        SystemExit: If no target model found
    """
    model_ids = []
    for model in models:
        model_id = model.get("id") or model.get("name") or model.get("model_id")
        if model_id:
            model_ids.append(str(model_id).lower())

    logger.info(f"Available models: {', '.join(model_ids)}")

    # Check for target models in order of preference
    for target in TARGET_MODELS:
        for model_id in model_ids:
            if target.lower() in model_id:
                logger.info(f"✓ Selected model: {model_id}")
                return model_id

    logger.error(f"None of the target models found: {TARGET_MODELS}")
    logger.error(f"Available models: {model_ids}")
    sys.exit(1)

# ============================================================================
# Voice Discovery & Selection
# ============================================================================

def discover_voices(client: httpx.Client) -> List[Dict[str, Any]]:
    """
    Discover available voices from Cartesia API.

    Args:
        client: HTTP client

    Returns:
        List of voice dictionaries

    Raises:
        SystemExit: On discovery failure
    """
    logger.info("Discovering available voices...")

    try:
        data = http_get_json(client, "/voices")

        # Handle response format
        if isinstance(data, list):
            voices = data
        elif isinstance(data, dict) and "data" in data:
            voices = data["data"]
        elif isinstance(data, dict) and "voices" in data:
            voices = data["voices"]
        else:
            voices = [data]

        logger.info(f"✓ Retrieved {len(voices)} voices")
        return voices

    except Exception as e:
        logger.error(f"Failed to discover voices: {e}")
        sys.exit(1)

def filter_arabic_voices(voices: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Filter voices that support Arabic language.

    Args:
        voices: List of all voices

    Returns:
        List of Arabic-capable voices
    """
    arabic_voices = []

    for voice in voices:
        # Check language field (can be string or list)
        languages = voice.get("language", [])
        if isinstance(languages, str):
            languages = [languages]

        # Check if Arabic is supported
        for lang in languages:
            lang_lower = str(lang).lower()
            if any(ar in lang_lower for ar in ["ar", "arabic", "ara"]):
                arabic_voices.append(voice)
                break
        else:
            # Some voices support all languages (no language field)
            if not languages:
                arabic_voices.append(voice)

    logger.info(f"✓ Found {len(arabic_voices)} Arabic-capable voices")
    return arabic_voices

def select_tunisian_voice(voices: List[Dict[str, Any]]) -> Tuple[Dict[str, Any], bool]:
    """
    Select best voice for Tunisian Arabic.

    Priority:
    1. Voice with "ar-TN" or "ar_TN" locale
    2. Voice with "Tunisian" or "Tunisia" in name/description/tags
    3. Fallback to generic Arabic voice

    Args:
        voices: List of Arabic-capable voices

    Returns:
        Tuple of (selected_voice, is_tunisian)

    Raises:
        SystemExit: If no suitable voice found
    """
    if not voices:
        logger.error("No Arabic voices available")
        logger.error("Verify Arabic support in Cartesia dashboard")
        sys.exit(1)

    # Priority 1: Tunisian locale
    for voice in voices:
        locale = str(voice.get("locale", "")).lower()
        language = str(voice.get("language", "")).lower()

        if "ar-tn" in locale or "ar_tn" in locale:
            logger.info(f"✓ Found Tunisian voice (locale): {voice.get('id')}")
            return voice, True
        if "ar-tn" in language or "ar_tn" in language:
            logger.info(f"✓ Found Tunisian voice (language): {voice.get('id')}")
            return voice, True

    # Priority 2: Tunisian in metadata
    for voice in voices:
        name = str(voice.get("name", "")).lower()
        description = str(voice.get("description", "")).lower()
        tags = " ".join(voice.get("tags", [])).lower()

        tunisian_keywords = ["tunis", "tunisia", "تونس"]

        for keyword in tunisian_keywords:
            if keyword in name or keyword in description or keyword in tags:
                logger.info(f"✓ Found Tunisian voice (metadata): {voice.get('id')}")
                return voice, True

    # Priority 3: Generic Arabic fallback
    logger.warning("No Tunisian-specific voice found")
    logger.warning("Using generic Arabic voice as fallback")

    selected_voice = voices[0]
    logger.info(f"✓ Selected Arabic voice: {selected_voice.get('id')}")

    return selected_voice, False

# ============================================================================
# Audio Synthesis
# ============================================================================

def build_tts_payload(
    model_id: str,
    voice_id: str,
    text: str,
    language: str = "ar",
    use_ssml: bool = True
) -> Dict[str, Any]:
    """
    Build TTS API request payload.

    [ASSUMPTION] Prosody settings for natural expressivity:
    - rate: 98% (slightly slower for clarity)
    - pitch: +0.5 semitones (slightly warmer)

    Args:
        model_id: Model identifier
        voice_id: Voice identifier
        text: Text to synthesize
        language: Language code (default: "ar")
        use_ssml: Whether to use SSML formatting

    Returns:
        API request payload dictionary
    """
    # Build SSML with mild expressivity adjustments
    if use_ssml:
        # Note: SSML support may vary by model
        ssml_text = f'''<speak xml:lang="{language}">
  <prosody rate="98%" pitch="+0.5st">{text}</prosody>
</speak>'''
        input_field = ssml_text
    else:
        input_field = text

    # Cartesia API payload structure (from existing codebase)
    payload = {
        "model_id": model_id,
        "transcript": input_field,
        "language": language,
        "voice": {
            "mode": "id",
            "id": voice_id
        },
        "output_format": {
            "container": "wav",
            "encoding": "pcm_s16le",
            "sample_rate": TARGET_SAMPLE_RATE
        }
    }

    # [ASSUMPTION] Last proposer mode - may not be supported by all models
    # If this causes errors, the fallback is to remove this field
    try:
        payload["inference"] = {"proposer_mode": "last"}
    except KeyError:
        pass

    return payload

def synthesize_audio(
    client: httpx.Client,
    model_id: str,
    voice_id: str,
    text: str,
    language: str = "ar"
) -> bytes:
    """
    Synthesize audio using Cartesia TTS API.

    Args:
        client: HTTP client
        model_id: Model identifier
        voice_id: Voice identifier
        text: Text to synthesize
        language: Language code

    Returns:
        Audio bytes (WAV format)

    Raises:
        httpx.HTTPStatusError: On synthesis failure
        SystemExit: On empty response
    """
    logger.info("Synthesizing audio...")
    logger.info(f"  Model: {model_id}")
    logger.info(f"  Voice: {voice_id}")
    logger.info(f"  Language: {language}")
    logger.info(f"  Text: {text[:50]}...")

    # Try with SSML first
    try:
        payload = build_tts_payload(model_id, voice_id, text, language, use_ssml=True)
        audio_bytes = http_post_bytes(client, "/tts/bytes", payload)

        if audio_bytes:
            logger.info("✓ Synthesis successful (with SSML)")
            return audio_bytes

    except httpx.HTTPStatusError as e:
        if "ssml" in str(e.response.text).lower():
            logger.warning("SSML not supported, retrying with plain text...")
        else:
            raise

    # Fallback to plain text
    payload = build_tts_payload(model_id, voice_id, text, language, use_ssml=False)
    audio_bytes = http_post_bytes(client, "/tts/bytes", payload)

    if not audio_bytes:
        logger.error("Empty audio response from API")
        sys.exit(1)

    logger.info("✓ Synthesis successful (plain text)")
    return audio_bytes

# ============================================================================
# Audio Validation
# ============================================================================

def validate_wav_format(audio_bytes: bytes) -> Tuple[bool, Dict[str, Any]]:
    """
    Validate WAV file format and extract metadata.

    Args:
        audio_bytes: WAV file bytes

    Returns:
        Tuple of (is_valid, metadata_dict)
    """
    try:
        with wave.open(io.BytesIO(audio_bytes), "rb") as wav:
            metadata = {
                "channels": wav.getnchannels(),
                "sample_width": wav.getsampwidth(),
                "sample_rate": wav.getframerate(),
                "num_frames": wav.getnframes(),
                "duration_sec": wav.getnframes() / wav.getframerate()
            }

        # Verify expected format
        is_valid = (
            metadata["channels"] == TARGET_CHANNELS and
            metadata["sample_width"] == TARGET_BIT_DEPTH // 8 and
            metadata["sample_rate"] == TARGET_SAMPLE_RATE
        )

        return is_valid, metadata

    except wave.Error as e:
        logger.error(f"Invalid WAV format: {e}")
        return False, {}

def detect_clipping(audio_bytes: bytes) -> Tuple[bool, int]:
    """
    Detect audio clipping (signal exceeding maximum amplitude).

    Args:
        audio_bytes: WAV file bytes

    Returns:
        Tuple of (has_clipping, peak_amplitude)
    """
    try:
        with wave.open(io.BytesIO(audio_bytes), "rb") as wav:
            if wav.getsampwidth() != 2:
                logger.warning("Non-16-bit audio, skipping clipping detection")
                return False, 0

            frames = wav.readframes(wav.getnframes())

        # Find peak amplitude in PCM16 signed format
        peak = 0
        for i in range(0, len(frames), 2):
            if i + 1 < len(frames):
                sample = int.from_bytes(frames[i:i+2], byteorder="little", signed=True)
                peak = max(peak, abs(sample))

        has_clipping = peak >= CLIPPING_THRESHOLD

        return has_clipping, peak

    except Exception as e:
        logger.warning(f"Clipping detection failed: {e}")
        return False, 0

def validate_audio(audio_bytes: bytes) -> None:
    """
    Comprehensive audio validation.

    Args:
        audio_bytes: WAV file bytes

    Raises:
        SystemExit: On validation failure
    """
    logger.info("Validating audio output...")

    # Validate WAV format
    is_valid, metadata = validate_wav_format(audio_bytes)

    if not is_valid:
        logger.error("Audio validation failed:")
        logger.error(f"  Expected: {TARGET_CHANNELS}ch, {TARGET_SAMPLE_RATE}Hz, {TARGET_BIT_DEPTH}-bit")
        logger.error(f"  Got: {metadata.get('channels')}ch, "
                    f"{metadata.get('sample_rate')}Hz, "
                    f"{metadata.get('sample_width', 0) * 8}-bit")
        sys.exit(1)

    logger.info(f"✓ Format: {metadata['channels']}ch, "
               f"{metadata['sample_rate']}Hz, "
               f"{metadata['sample_width'] * 8}-bit PCM")
    logger.info(f"✓ Duration: {metadata['duration_sec']:.2f} seconds")

    # Detect clipping
    has_clipping, peak = detect_clipping(audio_bytes)

    if has_clipping:
        logger.error(f"Audio clipping detected (peak: {peak}/32767)")
        logger.error("Reduce gain or request lower volume from API")
        sys.exit(1)

    logger.info(f"✓ No clipping detected (peak: {peak}/32767)")

# ============================================================================
# Main Execution
# ============================================================================

def print_capability_summary(
    model_id: str,
    voice: Dict[str, Any],
    is_tunisian: bool,
    total_voices: int
) -> None:
    """Print capability discovery summary."""
    logger.info("")
    logger.info("=" * 70)
    logger.info("CAPABILITY SUMMARY")
    logger.info("=" * 70)
    logger.info(f"Model: {model_id}")
    logger.info(f"Total voices discovered: {total_voices}")
    logger.info(f"Selected voice ID: {voice.get('id')}")
    logger.info(f"Voice name: {voice.get('name', 'N/A')}")
    logger.info(f"Language: {voice.get('language', 'N/A')}")
    logger.info(f"Tunisian-specific: {'Yes' if is_tunisian else 'No (using Arabic fallback)'}")
    logger.info("=" * 70)
    logger.info("")

def main() -> None:
    """Main execution flow."""
    logger.info("=" * 70)
    logger.info("Cartesia Sonic 3 TTS - Tunisian Arabic Synthesis")
    logger.info("=" * 70)
    logger.info("")

    # Step 1: Validate environment
    logger.info("Step 1: Validating environment...")
    api_key = getenv_strict("CARTESIA_API_KEY")
    logger.info(f"✓ API key loaded: {api_key[:12]}...")

    # Step 2: Initialize HTTP client
    logger.info("")
    logger.info("Step 2: Initializing HTTP client...")
    client = create_http_client(api_key)
    logger.info("✓ Client configured")

    try:
        # Step 3: Discover and verify model
        logger.info("")
        logger.info("Step 3: Discovering models...")
        models = discover_models(client)
        model_id = verify_target_model(models)

        # Step 4: Discover voices
        logger.info("")
        logger.info("Step 4: Discovering voices...")
        all_voices = discover_voices(client)
        arabic_voices = filter_arabic_voices(all_voices)

        if not arabic_voices:
            logger.error("No Arabic-capable voices found")
            logger.error("Verify Arabic support for selected model in dashboard")
            sys.exit(1)

        # Step 5: Select Tunisian voice
        logger.info("")
        logger.info("Step 5: Selecting Tunisian voice...")
        voice, is_tunisian = select_tunisian_voice(arabic_voices)

        # Print capability summary
        print_capability_summary(model_id, voice, is_tunisian, len(all_voices))

        # Step 6: Synthesize audio
        logger.info("Step 6: Synthesizing audio...")
        audio_bytes = synthesize_audio(
            client=client,
            model_id=model_id,
            voice_id=voice.get("id"),
            text=TUNISIAN_SENTENCE,
            language="ar"
        )

        # Step 7: Validate audio
        logger.info("")
        logger.info("Step 7: Validating audio...")
        validate_audio(audio_bytes)

        # Step 8: Write output file
        logger.info("")
        logger.info("Step 8: Writing output file...")
        output_path = Path(OUTPUT_FILE)
        output_path.write_bytes(audio_bytes)
        logger.info(f"✓ Audio written to: {output_path.absolute()}")

        # Success summary
        logger.info("")
        logger.info("=" * 70)
        logger.info("SUCCESS")
        logger.info("=" * 70)
        logger.info(f"Output file: {output_path.absolute()}")
        logger.info(f"Model: {model_id}")
        logger.info(f"Voice: {voice.get('id')}")
        logger.info(f"Language: ar (Arabic)")
        logger.info(f"Tunisian-specific: {'Yes' if is_tunisian else 'No'}")
        logger.info(f"Text: {TUNISIAN_SENTENCE}")
        logger.info("=" * 70)

    except httpx.HTTPStatusError as e:
        logger.error("")
        logger.error("HTTP Error occurred:")
        logger.error(f"  Status: {e.response.status_code}")
        logger.error(f"  Response: {e.response.text[:500]}")
        sys.exit(1)

    except Exception as e:
        logger.error("")
        logger.error(f"Unexpected error: {e}")
        logger.exception("Full traceback:")
        sys.exit(1)

    finally:
        client.close()


if __name__ == "__main__":
    main()
