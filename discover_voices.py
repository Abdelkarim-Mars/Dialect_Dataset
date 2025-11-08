#!/usr/bin/env python3
"""
Voice Discovery & Dialect Mapping Tool
Cartesia Sonic 3 Arabic Dataset Generator

Discovers Arabic-capable voices from Cartesia API and creates dialect mappings
for natural, expressive speech generation.

Usage:
    python discover_voices.py                    # Interactive discovery mode
    python discover_voices.py --list-only        # List voices without testing
    python discover_voices.py --test-voice <id>  # Test specific voice
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import requests
from tqdm import tqdm

from config import (
    CARTESIA_API_KEY,
    CARTESIA_API_VERSION,
    VOICES_ENDPOINT,
    TTS_ENDPOINT,
    LANGUAGE_CODE,
    MODEL_ID,
    OUTPUT_FORMAT,
    VOICE_MAPPING_PATH,
    DIALECTS,
    WORDS,
    NUMBERS,
    validate_config
)


class VoiceDiscovery:
    """Discover and test Cartesia voices for Arabic speech generation."""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.headers = {
            "Cartesia-Version": CARTESIA_API_VERSION,
            "X-API-Key": api_key
        }
        self.session = requests.Session()
        self.session.headers.update(self.headers)

    def get_all_voices(self) -> List[Dict]:
        """Fetch all available voices from Cartesia API."""
        print("Fetching voices from Cartesia API...")

        try:
            response = self.session.get(VOICES_ENDPOINT, timeout=30)
            response.raise_for_status()
            voices = response.json()

            print(f"✓ Retrieved {len(voices)} total voices")
            return voices

        except requests.exceptions.RequestException as e:
            print(f"✗ Error fetching voices: {e}")
            sys.exit(1)

    def filter_arabic_voices(self, voices: List[Dict]) -> List[Dict]:
        """Filter voices that support Arabic language."""
        print("\nFiltering for Arabic-capable voices...")

        arabic_voices = []

        for voice in voices:
            # Check if voice supports Arabic
            supported_languages = voice.get("language", [])

            # Cartesia may use different formats: "ar", "arabic", etc.
            if any(lang.lower() in ["ar", "arabic", "ara"] for lang in supported_languages):
                arabic_voices.append(voice)
            # If no language field, include for testing (some voices support all languages)
            elif not supported_languages:
                arabic_voices.append(voice)

        print(f"✓ Found {len(arabic_voices)} Arabic-capable voices")
        return arabic_voices

    def test_voice(self, voice_id: str, text: str, output_path: Optional[Path] = None) -> bool:
        """
        Test a voice by generating sample audio.

        Args:
            voice_id: Cartesia voice ID
            text: Arabic text to synthesize
            output_path: Optional path to save WAV file

        Returns:
            True if successful, False otherwise
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

        try:
            response = self.session.post(
                TTS_ENDPOINT,
                json=payload,
                timeout=30
            )
            response.raise_for_status()

            # Save audio if output path provided
            if output_path:
                output_path.parent.mkdir(parents=True, exist_ok=True)
                with open(output_path, "wb") as f:
                    f.write(response.content)

            return True

        except requests.exceptions.RequestException as e:
            print(f"  ✗ Error testing voice {voice_id}: {e}")
            return False

    def interactive_voice_selection(self, voices: List[Dict]) -> Dict[str, Dict]:
        """
        Interactive mode: test voices and create dialect mapping.

        Returns:
            voice_mapping: Dictionary mapping dialects to voice IDs
        """
        print("\n" + "="*70)
        print("INTERACTIVE VOICE SELECTION")
        print("="*70)
        print("\nWe'll test voices and you can select the best ones for each dialect.")
        print("Audio samples will be saved to: dataset/voice_samples/")
        print("\nSample texts:")
        print(f"  Word: {WORDS[0]} (activation)")
        print(f"  Number: {NUMBERS[1]} (one)")
        print()

        # Create samples directory
        samples_dir = Path("dataset/voice_samples")
        samples_dir.mkdir(parents=True, exist_ok=True)

        # Initialize voice mapping
        voice_mapping = {
            dialect: {"male": [], "female": []}
            for dialect in DIALECTS.keys()
        }

        print(f"Testing {len(voices)} voices...")
        print("Listen to samples and note down voice IDs that sound natural and expressive.")
        print()

        # Test each voice with sample texts
        successful_voices = []

        for voice in tqdm(voices, desc="Generating voice samples"):
            voice_id = voice.get("id")
            voice_name = voice.get("name", "Unknown")

            # Test with word sample
            word_path = samples_dir / f"{voice_id}_word.wav"
            success_word = self.test_voice(voice_id, WORDS[0], word_path)

            # Small delay to avoid rate limiting
            time.sleep(0.5)

            # Test with number sample
            number_path = samples_dir / f"{voice_id}_number.wav"
            success_number = self.test_voice(voice_id, NUMBERS[1], number_path)

            if success_word and success_number:
                successful_voices.append({
                    "id": voice_id,
                    "name": voice_name,
                    "description": voice.get("description", ""),
                    "word_sample": str(word_path),
                    "number_sample": str(number_path)
                })

            # Small delay between voices
            time.sleep(0.5)

        print(f"\n✓ Successfully generated samples for {len(successful_voices)} voices")
        print(f"\nSamples saved to: {samples_dir}")

        # Display voice list
        print("\n" + "="*70)
        print("AVAILABLE VOICES")
        print("="*70)

        for i, voice in enumerate(successful_voices, 1):
            print(f"\n{i}. Voice ID: {voice['id']}")
            print(f"   Name: {voice['name']}")
            if voice['description']:
                print(f"   Description: {voice['description']}")
            print(f"   Samples: {voice['word_sample']}")
            print(f"            {voice['number_sample']}")

        # Save voice list for reference
        voice_list_path = samples_dir / "voice_list.json"
        with open(voice_list_path, "w", encoding="utf-8") as f:
            json.dump(successful_voices, f, ensure_ascii=False, indent=2)

        print(f"\n✓ Voice list saved to: {voice_list_path}")

        # Interactive mapping
        print("\n" + "="*70)
        print("CREATE DIALECT MAPPING")
        print("="*70)
        print("\nNow, listen to the samples and assign voices to each dialect.")
        print("For each dialect, provide at least 2 male and 2 female voice IDs.")
        print("(Press Enter to skip, type 'done' when finished with a dialect)")
        print()

        for dialect_code, dialect_info in DIALECTS.items():
            print(f"\n{dialect_info['name']} ({dialect_code}):")
            print(f"  Need {dialect_info['speakers']} speakers")
            print()

            # Collect male voices
            print("  Male voices (provide voice IDs, one per line, 'done' when finished):")
            while True:
                voice_id = input("    > ").strip()
                if voice_id.lower() == "done":
                    break
                if voice_id and voice_id not in voice_mapping[dialect_code]["male"]:
                    voice_mapping[dialect_code]["male"].append(voice_id)
                    print(f"      Added: {voice_id}")

            # Collect female voices
            print("  Female voices (provide voice IDs, one per line, 'done' when finished):")
            while True:
                voice_id = input("    > ").strip()
                if voice_id.lower() == "done":
                    break
                if voice_id and voice_id not in voice_mapping[dialect_code]["female"]:
                    voice_mapping[dialect_code]["female"].append(voice_id)
                    print(f"      Added: {voice_id}")

        return voice_mapping

    def auto_mapping(self, voices: List[Dict]) -> Dict[str, Dict]:
        """
        Automatic mode: create basic mapping using available voices.
        Use this as a starting point for manual refinement.
        """
        print("\n" + "="*70)
        print("AUTOMATIC VOICE MAPPING (Starter Template)")
        print("="*70)
        print("\nCreating basic mapping using available voices...")
        print("⚠ You should manually test and refine this mapping for best quality!")
        print()

        # Initialize mapping
        voice_mapping = {
            dialect: {"male": [], "female": []}
            for dialect in DIALECTS.keys()
        }

        # Distribute voices across dialects
        # This is a basic round-robin distribution - not optimal for quality!
        voice_ids = [v.get("id") for v in voices if v.get("id")]

        dialect_codes = list(DIALECTS.keys())

        for i, voice_id in enumerate(voice_ids):
            dialect_idx = i % len(dialect_codes)
            gender = "male" if i % 2 == 0 else "female"

            dialect_code = dialect_codes[dialect_idx]
            voice_mapping[dialect_code][gender].append(voice_id)

        return voice_mapping

    def save_mapping(self, voice_mapping: Dict[str, Dict], output_path: Path):
        """Save voice mapping to JSON file."""
        # Add metadata
        mapping_with_metadata = {
            "version": "1.0",
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "model": MODEL_ID,
            "language": LANGUAGE_CODE,
            "dialects": voice_mapping
        }

        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(mapping_with_metadata, f, ensure_ascii=False, indent=2)

        print(f"\n✓ Voice mapping saved to: {output_path}")

    def display_mapping_summary(self, voice_mapping: Dict[str, Dict]):
        """Display summary of voice mapping."""
        print("\n" + "="*70)
        print("VOICE MAPPING SUMMARY")
        print("="*70)

        for dialect_code, voices in voice_mapping.items():
            dialect_name = DIALECTS[dialect_code]["name"]
            male_count = len(voices["male"])
            female_count = len(voices["female"])
            total_voices = male_count + female_count

            print(f"\n{dialect_name} ({dialect_code}):")
            print(f"  Male voices: {male_count}")
            print(f"  Female voices: {female_count}")
            print(f"  Total unique voices: {total_voices}")

            if total_voices < 4:
                print(f"  ⚠ Warning: Only {total_voices} voices (recommend at least 4)")


def main():
    """Main entry point for voice discovery."""
    parser = argparse.ArgumentParser(
        description="Discover and map Cartesia voices for Arabic dataset generation"
    )
    parser.add_argument(
        "--list-only",
        action="store_true",
        help="Only list available voices without testing"
    )
    parser.add_argument(
        "--test-voice",
        type=str,
        metavar="VOICE_ID",
        help="Test a specific voice ID"
    )
    parser.add_argument(
        "--auto",
        action="store_true",
        help="Automatic mapping (not recommended - use for testing only)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=VOICE_MAPPING_PATH,
        help=f"Output path for voice mapping (default: {VOICE_MAPPING_PATH})"
    )

    args = parser.parse_args()

    # Validate configuration
    try:
        validate_config()
    except ValueError as e:
        print(f"✗ Configuration error: {e}")
        sys.exit(1)

    # Initialize discovery
    discovery = VoiceDiscovery(CARTESIA_API_KEY)

    # Test specific voice
    if args.test_voice:
        print(f"Testing voice: {args.test_voice}")
        output_path = Path(f"dataset/voice_samples/{args.test_voice}_test.wav")
        success = discovery.test_voice(args.test_voice, WORDS[0], output_path)

        if success:
            print(f"✓ Test successful! Audio saved to: {output_path}")
        else:
            print(f"✗ Test failed for voice: {args.test_voice}")

        sys.exit(0 if success else 1)

    # Get all voices
    all_voices = discovery.get_all_voices()

    # Filter for Arabic
    arabic_voices = discovery.filter_arabic_voices(all_voices)

    if not arabic_voices:
        print("✗ No Arabic-capable voices found!")
        sys.exit(1)

    # List only mode
    if args.list_only:
        print("\n" + "="*70)
        print("ARABIC-CAPABLE VOICES")
        print("="*70)

        for voice in arabic_voices:
            print(f"\nID: {voice.get('id')}")
            print(f"Name: {voice.get('name', 'Unknown')}")
            print(f"Description: {voice.get('description', 'N/A')}")
            print(f"Languages: {', '.join(voice.get('language', []))}")

        sys.exit(0)

    # Create voice mapping
    if args.auto:
        voice_mapping = discovery.auto_mapping(arabic_voices)
    else:
        voice_mapping = discovery.interactive_voice_selection(arabic_voices)

    # Save mapping
    discovery.save_mapping(voice_mapping, args.output)

    # Display summary
    discovery.display_mapping_summary(voice_mapping)

    print("\n" + "="*70)
    print("NEXT STEPS")
    print("="*70)
    print(f"\n1. Review voice mapping: {args.output}")
    print("2. Listen to samples in: dataset/voice_samples/")
    print("3. Edit mapping if needed (it's JSON)")
    print("4. Run dataset generation: python generate_dataset.py")
    print()


if __name__ == "__main__":
    main()
