#!/usr/bin/env python3
"""
Example: Export subset of dataset

Create a smaller subset of the dataset for specific use cases:
- Single dialect
- Specific speakers
- Specific tokens
- Random sample
"""

import argparse
import shutil
from pathlib import Path
import pandas as pd
from tqdm import tqdm


def export_subset(
    manifest_path: str,
    output_dir: str,
    dialect: str = None,
    speaker_ids: list = None,
    token_class: str = None,
    sample_size: int = None
):
    """
    Export subset of dataset.

    Args:
        manifest_path: Path to manifest.csv
        output_dir: Output directory for subset
        dialect: Filter by dialect (e.g., 'EGY')
        speaker_ids: Filter by speaker IDs (e.g., ['spk001', 'spk002'])
        token_class: Filter by class ('word' or 'number')
        sample_size: Random sample size (if None, use all matching)
    """
    # Load manifest
    df = pd.read_csv(manifest_path)
    print(f"Total utterances in dataset: {len(df)}")

    # Apply filters
    filtered = df.copy()

    if dialect:
        filtered = filtered[filtered['dialect'] == dialect]
        print(f"Filtered by dialect '{dialect}': {len(filtered)} utterances")

    if speaker_ids:
        filtered = filtered[filtered['speaker_id'].isin(speaker_ids)]
        print(f"Filtered by speakers {speaker_ids}: {len(filtered)} utterances")

    if token_class:
        filtered = filtered[filtered['class'] == token_class]
        print(f"Filtered by class '{token_class}': {len(filtered)} utterances")

    # Random sample if specified
    if sample_size and sample_size < len(filtered):
        filtered = filtered.sample(sample_size)
        print(f"Random sample: {len(filtered)} utterances")

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Copy files
    print(f"\nCopying {len(filtered)} files to {output_dir}...")

    for _, row in tqdm(filtered.iterrows(), total=len(filtered)):
        src_file = Path(row['file_path'])

        # Recreate directory structure
        rel_path = src_file.relative_to('dataset')
        dst_file = output_path / rel_path

        # Create parent directories
        dst_file.parent.mkdir(parents=True, exist_ok=True)

        # Copy file
        if src_file.exists():
            shutil.copy2(src_file, dst_file)

    # Save subset manifest
    subset_manifest_path = output_path / "manifest.csv"

    # Update file paths in manifest
    filtered_copy = filtered.copy()
    filtered_copy['file_path'] = filtered_copy['file_path'].apply(
        lambda p: str(output_path / Path(p).relative_to('dataset'))
    )

    filtered_copy.to_csv(subset_manifest_path, index=False)

    print(f"\n✓ Subset exported successfully!")
    print(f"  Files: {len(filtered)}")
    print(f"  Location: {output_path}")
    print(f"  Manifest: {subset_manifest_path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Export subset of Arabic dialect dataset"
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default="dataset/manifest.csv",
        help="Path to manifest.csv"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory for subset"
    )
    parser.add_argument(
        "--dialect",
        type=str,
        choices=['EGY', 'LAV', 'GLF', 'MOR', 'TUN'],
        help="Filter by dialect"
    )
    parser.add_argument(
        "--speakers",
        type=str,
        nargs='+',
        help="Filter by speaker IDs (e.g., spk001 spk002)"
    )
    parser.add_argument(
        "--class",
        type=str,
        choices=['word', 'number'],
        dest='token_class',
        help="Filter by token class"
    )
    parser.add_argument(
        "--sample",
        type=int,
        help="Random sample size"
    )

    args = parser.parse_args()

    export_subset(
        manifest_path=args.manifest,
        output_dir=args.output,
        dialect=args.dialect,
        speaker_ids=args.speakers,
        token_class=args.token_class,
        sample_size=args.sample
    )


if __name__ == "__main__":
    main()
