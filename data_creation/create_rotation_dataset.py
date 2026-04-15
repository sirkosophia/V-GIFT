#!/usr/bin/env python3
"""
Script to create a pre-generated rotation dataset for LLaVA training.
This script generates a JSON file compatible with LLaVA's custom data format.
WITH DIRECT DEGREE ANSWERS - model answers directly with degrees (e.g., "0", "90", "180", "270")

Supports a --reconstruct mode that re-generates images using the compact
rotation_metadata.json (reconstruction_metadata/rotation_metadata.json), which stores
per-entry: id (UUID), source_image (COCO filename), rotation_degrees.
This ensures output filenames match exactly what llava_v1_5_v_gift.json references.
"""

import os
import json
import random
import argparse
from pathlib import Path
from typing import List, Dict
from PIL import Image
import uuid


def get_image_files(image_dir: str) -> List[Path]:
    """Get all image files from directory."""
    image_dir = Path(image_dir)
    if not image_dir.exists():
        raise ValueError(f"Image directory {image_dir} does not exist")

    extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    image_files = []

    for ext in extensions:
        image_files.extend(list(image_dir.glob(f'*{ext}')))
        image_files.extend(list(image_dir.glob(f'*{ext.upper()}')))

    return sorted(image_files)


def rotate_and_save_image(image_path: Path, rotation_degrees: int, output_dir: Path, sample_id: str) -> str:
    """Rotate image and save to output directory. Returns relative path to rotated image."""
    try:
        # Load image
        image = Image.open(image_path).convert('RGB')

        # Apply rotation
        if rotation_degrees == 0:
            rotated_image = image
        elif rotation_degrees == 90:
            rotated_image = image.rotate(-90, expand=True)  # PIL rotates counter-clockwise
        elif rotation_degrees == 180:
            rotated_image = image.rotate(-180, expand=True)
        elif rotation_degrees == 270:
            rotated_image = image.rotate(-270, expand=True)  # equivalent to 90 clockwise
        else:
            raise ValueError(f"Unsupported rotation: {rotation_degrees}")

        # Save rotated image
        output_filename = f"{sample_id}_rot{rotation_degrees}.jpg"
        output_path = output_dir / output_filename
        rotated_image.save(output_path, 'JPEG', quality=95)

        return output_filename

    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None


def create_rotation_qa(rotation_degrees: int) -> tuple[str, str]:
    """Create question and answer for rotation task with DIRECT degree answers."""

    question = "I give you an image that can be rotated by a multiple of 90 degrees clockwise. Answer with the number of degrees directly. Here are the possible answers: 0, 90, 180, 270."

    # Answer is just the number
    answer = str(rotation_degrees)

    return question, answer


def create_rotation_dataset_reconstruct(
    input_dir: str,
    output_dir: str,
    output_json: str,
    metadata_json: str,
):
    """
    Reconstruct rotation dataset using the compact rotation_metadata.json, which stores
    per-entry: id (UUID), source_image (COCO filename), rotation_degrees.
    Output filenames match exactly what the final dataset references.
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Loading rotation metadata from {metadata_json}...")
    with open(metadata_json) as f:
        metadata = json.load(f)
    print(f"Found {len(metadata)} entries to reconstruct")

    # Build mapping: uuid -> (coco_filename, rotation_degrees)
    mapping = {e["id"]: (e["source_image"], e["rotation_degrees"]) for e in metadata}

    dataset_samples = []
    skipped = 0

    print(f"Regenerating {len(mapping)} rotation images...")
    for i, (sample_id, (coco_filename, rotation_degrees)) in enumerate(mapping.items()):
        if i % 1000 == 0:
            print(f"Processing {i+1}/{len(mapping)}")

        image_path = input_path / coco_filename
        if not image_path.exists():
            print(f"WARNING: source image not found: {image_path}")
            skipped += 1
            continue

        rotated_image_filename = rotate_and_save_image(
            image_path, rotation_degrees, output_path, sample_id
        )

        if rotated_image_filename is None:
            skipped += 1
            continue

        question, answer = create_rotation_qa(rotation_degrees)

        sample = {
            "id": sample_id,
            "image": rotated_image_filename,
            "conversations": [
                {
                    "from": "human",
                    "value": f"<image>\n{question}"
                },
                {
                    "from": "gpt",
                    "value": answer
                }
            ]
        }

        dataset_samples.append(sample)

    print(f"Created {len(dataset_samples)} samples ({skipped} skipped)")

    print(f"Saving dataset to {output_json}")
    with open(output_json, 'w') as f:
        json.dump(dataset_samples, f, indent=2)

    print("Reconstruction complete!")
    print(f"Images saved to: {output_dir}")
    print(f"Dataset JSON: {output_json}")


def create_rotation_dataset(
    input_dir: str,
    output_dir: str,
    output_json: str,
    num_images: int = 10000,
    rotations_per_image: int = 4,
    rotations: List[int] = [0, 90, 180, 270]
):
    """
    Create rotation dataset with pre-generated images.

    Args:
        input_dir: Directory containing source images (e.g., COCO train2017)
        output_dir: Directory to save rotated images
        output_json: Path for output JSON file
        num_images: Number of source images to use
        rotations_per_image: How many rotations to create per source image
        rotations: List of rotation angles to use
    """

    # Setup paths
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Get all source images
    print(f"Finding images in {input_dir}...")
    all_images = get_image_files(input_dir)
    print(f"Found {len(all_images)} total images")

    # Select subset if specified
    if num_images > 0 and num_images < len(all_images):
        selected_images = random.sample(all_images, num_images)
        print(f"Selected {len(selected_images)} images for dataset creation")
    else:
        selected_images = all_images
        print(f"Using all {len(selected_images)} images")

    # Generate dataset samples
    dataset_samples = []

    print("Generating rotation dataset with direct degree answers...")
    for i, image_path in enumerate(selected_images):
        if i % 1000 == 0:
            print(f"Processing image {i+1}/{len(selected_images)}")

        # Create multiple rotations per image
        for rotation_idx in range(rotations_per_image):
            # Choose rotation (can be random or systematic)
            if rotations_per_image == len(rotations):
                # Use all rotations systematically
                rotation = rotations[rotation_idx]
            else:
                # Random selection
                rotation = random.choice(rotations)

            # Generate unique sample ID
            sample_id = str(uuid.uuid4())

            # Rotate and save image
            rotated_image_filename = rotate_and_save_image(
                image_path, rotation, output_path, sample_id
            )

            if rotated_image_filename is None:
                continue  # Skip if rotation failed

            # Create Q&A with direct degree answer
            question, answer = create_rotation_qa(rotation)

            # Create sample in LLaVA format
            sample = {
                "id": sample_id,
                "image": rotated_image_filename,  # Relative path to rotated image
                "conversations": [
                    {
                        "from": "human",
                        "value": f"<image>\n{question}"
                    },
                    {
                        "from": "gpt",
                        "value": answer
                    }
                ]
            }

            dataset_samples.append(sample)

    print(f"Created {len(dataset_samples)} total samples")

    # Print rotation distribution statistics
    rotation_counts = {0: 0, 90: 0, 180: 0, 270: 0}
    for sample in dataset_samples:
        answer_text = sample["conversations"][1]["value"]
        # Extract the degree number from answer
        for deg in [0, 90, 180, 270]:
            if str(deg) in answer_text:
                rotation_counts[deg] += 1
                break

    print("\nRotation angle distribution:")
    for deg, count in sorted(rotation_counts.items()):
        percentage = (count / len(dataset_samples)) * 100
        print(f"  {deg}°: {count} ({percentage:.1f}%)")

    # Show some example Q&A pairs
    print("\nExample Q&A pairs:")
    for i in range(min(5, len(dataset_samples))):
        sample = dataset_samples[i]
        print(f"\nExample {i+1}:")
        print(f"  Q: {sample['conversations'][0]['value'].replace('<image>', '[IMAGE]')[:80]}...")
        print(f"  A: {sample['conversations'][1]['value']}")

    # Save dataset JSON
    print(f"\nSaving dataset to {output_json}")
    with open(output_json, 'w') as f:
        json.dump(dataset_samples, f, indent=2)

    print("Dataset creation complete!")
    print(f"Images saved to: {output_dir}")
    print(f"Dataset JSON: {output_json}")
    print(f"Total samples: {len(dataset_samples)}")


def main():
    parser = argparse.ArgumentParser(description="Create rotation dataset for LLaVA training with direct degree answers")
    parser.add_argument("--input_dir", type=str, required=True,
                       help="Directory containing source images (e.g., COCO train2017)")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Directory to save rotated images")
    parser.add_argument("--output_json", type=str, required=True,
                       help="Path for output JSON file")

    # Original mode arguments
    parser.add_argument("--num_images", type=int, default=10000,
                       help="Number of source images to use (0 = all) [original mode only]")
    parser.add_argument("--rotations_per_image", type=int, default=4,
                       help="Number of rotations per source image [original mode only]")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility [original mode only]")

    # Reconstruct mode arguments
    _repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    parser.add_argument("--reconstruct", action="store_true",
                       help="Reconstruct dataset using rotation_metadata.json")
    parser.add_argument("--metadata_json", type=str,
                       default=os.path.join(_repo_root, "reconstruction_metadata", "rotation_metadata.json"),
                       help="Path to rotation_metadata.json [reconstruct mode]")

    args = parser.parse_args()

    if args.reconstruct:
        create_rotation_dataset_reconstruct(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            output_json=args.output_json,
            metadata_json=args.metadata_json,
        )
    else:
        random.seed(args.seed)
        create_rotation_dataset(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            output_json=args.output_json,
            num_images=args.num_images,
            rotations_per_image=args.rotations_per_image
        )


if __name__ == "__main__":
    main()
