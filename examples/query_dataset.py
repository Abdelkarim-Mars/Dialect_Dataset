#!/usr/bin/env python3
"""
Example: Query and analyze the Arabic dialect dataset

Demonstrates how to:
- Load the manifest
- Filter by dialect, speaker, or token class
- Analyze dataset statistics
- Load and process audio files
"""

import pandas as pd
import soundfile as sf
from pathlib import Path
from collections import Counter


def load_manifest(manifest_path: str = "../dataset/manifest.csv") -> pd.DataFrame:
    """Load the dataset manifest."""
    df = pd.read_csv(manifest_path)
    print(f"✓ Loaded manifest: {len(df)} utterances")
    return df


def analyze_dataset(df: pd.DataFrame):
    """Print dataset statistics."""
    print("\n" + "="*70)
    print("DATASET STATISTICS")
    print("="*70)

    print(f"\nTotal utterances: {len(df):,}")

    # By dialect
    print("\nBy Dialect:")
    dialect_counts = df['dialect'].value_counts().sort_index()
    for dialect, count in dialect_counts.items():
        print(f"  {dialect}: {count:,} utterances")

    # By gender
    print("\nBy Gender:")
    gender_counts = df['gender'].value_counts()
    for gender, count in gender_counts.items():
        print(f"  {gender.capitalize()}: {count:,} utterances")

    # By class
    print("\nBy Token Class:")
    class_counts = df['class'].value_counts()
    for cls, count in class_counts.items():
        print(f"  {cls.capitalize()}: {count:,} utterances")

    # Unique speakers
    print(f"\nUnique speakers: {df['speaker_id'].nunique()}")
    print(f"Unique voices: {df['voice_id'].nunique()}")

    # Audio stats
    print("\nAudio Statistics:")
    print(f"  Avg duration: {df['duration_sec'].mean():.2f}s")
    print(f"  Min duration: {df['duration_sec'].min():.2f}s")
    print(f"  Max duration: {df['duration_sec'].max():.2f}s")
    print(f"  Total duration: {df['duration_sec'].sum() / 3600:.2f} hours")


def query_by_dialect(df: pd.DataFrame, dialect: str):
    """Get all utterances for a specific dialect."""
    filtered = df[df['dialect'] == dialect]
    print(f"\n{dialect} dialect: {len(filtered)} utterances")
    return filtered


def query_by_speaker(df: pd.DataFrame, speaker_id: str):
    """Get all utterances for a specific speaker."""
    filtered = df[df['speaker_id'] == speaker_id]
    print(f"\nSpeaker {speaker_id}: {len(filtered)} utterances")
    return filtered


def query_by_token(df: pd.DataFrame, token: str):
    """Get all utterances for a specific token."""
    filtered = df[df['token'] == token]
    print(f"\nToken '{token}': {len(filtered)} utterances")
    return filtered


def get_random_sample(df: pd.DataFrame, n: int = 10):
    """Get random sample of utterances."""
    sample = df.sample(n)
    print(f"\nRandom sample of {n} utterances:")
    for _, row in sample.iterrows():
        print(f"  {row['file_path']}")
    return sample


def load_audio_file(file_path: str):
    """Load and display audio file information."""
    audio, sample_rate = sf.read(file_path)

    print(f"\nAudio file: {Path(file_path).name}")
    print(f"  Sample rate: {sample_rate} Hz")
    print(f"  Duration: {len(audio) / sample_rate:.2f}s")
    print(f"  Samples: {len(audio):,}")
    print(f"  Shape: {audio.shape}")

    return audio, sample_rate


def main():
    """Main demonstration."""
    print("="*70)
    print("Arabic Dialect Dataset Query Examples")
    print("="*70)

    # Load manifest
    df = load_manifest()

    # Overall statistics
    analyze_dataset(df)

    # Example queries
    print("\n" + "="*70)
    print("EXAMPLE QUERIES")
    print("="*70)

    # Query by dialect
    egy_files = query_by_dialect(df, "EGY")

    # Query by speaker
    speaker_files = query_by_speaker(df, "spk001")

    # Query by token
    token_files = query_by_token(df, "نعم")

    # Combined query: Egyptian dialect, male speakers, word class
    print("\n" + "="*70)
    print("Combined Query: Egyptian + Male + Words")
    print("="*70)

    combined = df[
        (df['dialect'] == 'EGY') &
        (df['gender'] == 'male') &
        (df['class'] == 'word')
    ]
    print(f"Results: {len(combined)} utterances")

    # Random sample
    print("\n" + "="*70)
    print("Random Sample")
    print("="*70)
    get_random_sample(df, n=5)

    # Load audio file example
    if len(df) > 0:
        print("\n" + "="*70)
        print("Load Audio Example")
        print("="*70)

        sample_file = df.iloc[0]['file_path']

        try:
            load_audio_file(sample_file)
        except FileNotFoundError:
            print(f"File not found: {sample_file}")

    print("\n" + "="*70)
    print("Query Examples Complete")
    print("="*70)


if __name__ == "__main__":
    main()
