"""
Convert LLaVA-format JSON + images to Megatron Energon webdataset format.

The LLaVA JSON format (input):
    [{"id": "...", "image": "rel/path/to/img.jpg", "conversations": [
        {"from": "human", "value": "<image>\nQuestion?"},
        {"from": "gpt", "value": "Answer"}
    ]}, ...]

The Energon webdataset format (output):
    A directory with:
      - ssl-0000.tar, ssl-0001.tar, ... (each containing {key}.jpg + {key}.json pairs)
      - .nv-meta/dataset.yaml
      - .nv-meta/split.yaml
      - .nv-meta/.info.yaml
      - .nv-meta/sample_loader.py

Usage:
    python convert_llava_json_to_webdataset.py \\
        --input_json  /path/to/ssl_tasks.json \\
        --image_base  /path/to/image/base/dir \\
        --output_dir  /path/to/output/webdataset \\
        --shard_size  10000

    # To extract only SSL samples from a mixed JSON (filter by image path substring):
    python convert_llava_json_to_webdataset.py \\
        --input_json  /path/to/llava_mix.json \\
        --image_base  /path/to/image/base/dir \\
        --output_dir  /path/to/output/webdataset \\
        --filter_substr  "correspondence"
"""

import argparse
import io
import json
import os
import tarfile
from pathlib import Path

from PIL import Image
import yaml


ROLE_MAP = {"human": "user", "gpt": "assistant"}


def convert_conversations(conversations):
    """Convert LLaVA conversations to Energon texts format."""
    texts = []
    for turn in conversations:
        role = ROLE_MAP.get(turn["from"], turn["from"])
        texts.append({"role": role, "content": turn["value"]})
    return texts


def build_sample_json(sample, image_ext="jpg"):
    """Build the Energon JSON payload for a single sample."""
    texts = convert_conversations(sample["conversations"])
    return {
        "texts": texts,
        "media": "image",
        "name": [image_ext],   # matches the ext stored in the tar sample dict
    }


def write_webdataset(samples, image_base, output_dir, shard_size=10000):
    """Write samples to a webdataset directory with shard tar files."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    shard_files = []          # list of (tar_name, count)
    current_shard = []
    shard_idx = 0
    global_idx = 0

    def flush_shard(shard_samples, shard_idx):
        tar_name = f"ssl-{shard_idx:04d}.tar"
        tar_path = output_dir / tar_name
        with tarfile.open(tar_path, "w") as tf:
            for key, img_bytes, json_bytes in shard_samples:
                # Image file: {key}.jpg
                img_info = tarfile.TarInfo(name=f"{key}.jpg")
                img_info.size = len(img_bytes)
                tf.addfile(img_info, io.BytesIO(img_bytes))
                # JSON file: {key}.json
                json_info = tarfile.TarInfo(name=f"{key}.json")
                json_info.size = len(json_bytes)
                tf.addfile(json_info, io.BytesIO(json_bytes))
        return tar_name, len(shard_samples)

    skipped = 0
    for sample in samples:
        img_rel = sample.get("image", "")
        if isinstance(img_rel, list):
            img_rel = img_rel[0] if img_rel else ""
        img_path = Path(image_base) / img_rel

        if not img_path.exists():
            print(f"  [WARN] Image not found, skipping: {img_path}")
            skipped += 1
            continue

        # Re-encode image as JPEG to normalise format
        try:
            pil_img = Image.open(img_path).convert("RGB")
            buf = io.BytesIO()
            pil_img.save(buf, format="JPEG", quality=95)
            img_bytes = buf.getvalue()
        except Exception as e:
            print(f"  [WARN] Failed to read image {img_path}: {e}")
            skipped += 1
            continue

        key = f"{global_idx:08d}"
        json_bytes = json.dumps(build_sample_json(sample)).encode("utf-8")
        current_shard.append((key, img_bytes, json_bytes))
        global_idx += 1

        if len(current_shard) >= shard_size:
            name, count = flush_shard(current_shard, shard_idx)
            shard_files.append((name, count))
            print(f"  Wrote shard {name} ({count} samples)")
            current_shard = []
            shard_idx += 1

    # Write remaining samples
    if current_shard:
        name, count = flush_shard(current_shard, shard_idx)
        shard_files.append((name, count))
        print(f"  Wrote shard {name} ({count} samples)")

    total = sum(c for _, c in shard_files)
    print(f"\nTotal samples written: {total}  (skipped: {skipped})")
    return shard_files


def write_nv_meta(output_dir, shard_files):
    """Write the Energon .nv-meta metadata files."""
    meta_dir = Path(output_dir) / ".nv-meta"
    meta_dir.mkdir(exist_ok=True)

    # dataset.yaml — identifies the sample type and loaders
    dataset_yaml = {
        "sample_type": {
            "__module__": "aiak_training_llm.data.multimodal",
            "__class__": "MultiMixQASample",
        },
        "part_filter": "sample_loader.py:part_filter",
        "sample_loader": "sample_loader.py:sample_loader",
    }
    with open(meta_dir / "dataset.yaml", "w") as f:
        yaml.dump(dataset_yaml, f, default_flow_style=False)

    # split.yaml — lists tar files per split
    train_tars = [name for name, _ in shard_files]
    split_yaml = {
        "exclude": [],
        "split_parts": {
            "test": [],
            "train": train_tars,
            "val": [],
        },
    }
    with open(meta_dir / "split.yaml", "w") as f:
        yaml.dump(split_yaml, f, default_flow_style=False)

    # .info.yaml — shard sample counts (required by Energon)
    info_yaml = {"shard_counts": {name: count for name, count in shard_files}}
    with open(meta_dir / ".info.yaml", "w") as f:
        yaml.dump(info_yaml, f, default_flow_style=False)

    # sample_loader.py — custom loader that energon calls per sample
    sample_loader_code = '''\
def sample_loader(sample: dict) -> dict:
    messages = []
    system = None
    for message in sample["json"]["texts"]:
        assert message["role"] in ["system", "user", "assistant"]
        if message["role"] == "system":
            system = message["content"]
            continue
        messages.append(dict(role=message["role"], content=message["content"]))
    image = []
    video = []
    if sample["json"]["media"] == "image":
        for name in sample["json"]["name"]:
            image.append(sample.get(name))
    elif sample["json"]["media"] == "video":
        for name in sample["json"]["name"]:
            video.append(sample.get(name))
    return dict(
        __key__=sample["__key__"],
        __restore_key__=sample["__restore_key__"],
        video=video if len(video) > 0 else None,
        image=image if len(image) > 0 else None,
        system=system,
        messages=messages,
    )


def part_filter(part: str) -> bool:
    return True
'''
    with open(meta_dir / "sample_loader.py", "w") as f:
        f.write(sample_loader_code)

    print(f"Wrote .nv-meta to {meta_dir}")


def main():
    parser = argparse.ArgumentParser(description="Convert LLaVA JSON to Energon webdataset")
    parser.add_argument("--input_json", required=True,
                        help="LLaVA-format JSON file (list of samples)")
    parser.add_argument("--image_base", required=True,
                        help="Base directory for resolving relative image paths")
    parser.add_argument("--output_dir", required=True,
                        help="Output webdataset directory")
    parser.add_argument("--shard_size", type=int, default=10000,
                        help="Max samples per tar shard (default: 10000)")
    parser.add_argument("--filter_substr", default=None,
                        help="If set, only include samples whose 'image' path contains this substring")
    args = parser.parse_args()

    print(f"Loading {args.input_json} ...")
    with open(args.input_json) as f:
        data = json.load(f)
    print(f"  Loaded {len(data)} samples")

    if args.filter_substr:
        data = [s for s in data if args.filter_substr in s.get("image", "")]
        print(f"  After filtering for '{args.filter_substr}': {len(data)} samples")

    print(f"\nConverting to webdataset at {args.output_dir} ...")
    shard_files = write_webdataset(data, args.image_base, args.output_dir, args.shard_size)
    write_nv_meta(args.output_dir, shard_files)
    print("\nDone.")


if __name__ == "__main__":
    main()
