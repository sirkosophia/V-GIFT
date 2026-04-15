import os
import json
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps
from pathlib import Path
import uuid
from tqdm import tqdm
import random
import matplotlib.colors as mcolors
from skimage.color import rgb2lab

# Set seeds for reproducibility
random.seed(42)
np.random.seed(42)

# Pre-build xkcd color lookup table in CIELAB space (perceptually uniform)
_XKCD_LAB = {}
for name, hex_val in mcolors.XKCD_COLORS.items():
    rgb = mcolors.to_rgb(hex_val)
    # rgb2lab expects (1,1,3) float array in [0,1]
    lab = rgb2lab(np.array([[list(rgb)]]))[0, 0]
    _XKCD_LAB[name.replace("xkcd:", "")] = lab


def rgb_to_name(r, g, b):
    """Convert RGB to the closest xkcd color name using CIELAB distance."""
    lab = rgb2lab(np.array([[[r / 255.0, g / 255.0, b / 255.0]]]))[0, 0]
    min_dist = float("inf")
    closest = "black"
    for name, ref_lab in _XKCD_LAB.items():
        dist = np.sum((lab - ref_lab) ** 2)
        if dist < min_dist:
            min_dist = dist
            closest = name
    return closest


def format_rgb(r, g, b):
    """Format RGB as a readable string with color name."""
    name = rgb_to_name(r, g, b)
    return f"RGB({r}, {g}, {b}) ({name})"


class ColorizationDatasetCreator:
    def __init__(
        self,
        images_dir,
        output_dir,
        num_points=5,
        patch_radius=1,
        target_size=512,
    ):
        self.images_dir = Path(images_dir)
        self.output_dir = Path(output_dir)
        self.num_points = num_points
        self.patch_radius = patch_radius  # average color from a small patch around the point
        self.target_size = target_size

        # Create output directories
        self.images_output_dir = self.output_dir / "images"
        self.images_output_dir.mkdir(parents=True, exist_ok=True)

        self.dataset = []

    def get_image_list(self):
        """Get all jpg images in the images directory."""
        images = sorted([
            f.name for f in self.images_dir.iterdir()
            if f.suffix.lower() in ('.jpg', '.jpeg', '.png')
        ])
        return images

    def sample_color_at_point(self, img_array, x, y):
        """Sample average color in a small patch around (x, y)."""
        h, w, _ = img_array.shape
        r = self.patch_radius

        y_min = max(0, y - r)
        y_max = min(h, y + r + 1)
        x_min = max(0, x - r)
        x_max = min(w, x + r + 1)

        patch = img_array[y_min:y_max, x_min:x_max].astype(np.float64)
        avg_color = patch.mean(axis=(0, 1))
        return int(avg_color[0]), int(avg_color[1]), int(avg_color[2])

    def colors_are_distinct(self, colors, min_distance=40):
        """Check that all colors are sufficiently different from each other."""
        for i in range(len(colors)):
            for j in range(i + 1, len(colors)):
                dist = np.sqrt(sum((a - b) ** 2 for a, b in zip(colors[i], colors[j])))
                if dist < min_distance:
                    return False
        return True

    def sample_points_with_distinct_colors(self, img_array, margin=20, max_attempts=50):
        """Sample num_points points whose colors are sufficiently distinct.

        Returns list of (x, y, r, g, b) tuples or None if couldn't find distinct set.
        """
        h, w, _ = img_array.shape

        for _ in range(max_attempts):
            points = []
            colors = []
            for _ in range(self.num_points):
                x = random.randint(margin, w - margin - 1)
                y = random.randint(margin, h - margin - 1)
                r, g, b = self.sample_color_at_point(img_array, x, y)
                points.append((x, y))
                colors.append((r, g, b))

            if self.colors_are_distinct(colors):
                return [(x, y, r, g, b) for (x, y), (r, g, b) in zip(points, colors)]

        return None

    def draw_points_on_grayscale(self, gray_img, points_xy, labels):
        """Draw labeled points on a grayscale image. Returns a new RGB image."""
        # Convert grayscale to RGB so we can draw colored markers
        img = gray_img.convert("RGB")
        draw = ImageDraw.Draw(img)

        color = "red"
        radius = 8

        try:
            font = ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 28
            )
        except Exception:
            font = ImageFont.load_default()

        for (x, y), label in zip(points_xy, labels):
            # Draw filled circle
            draw.ellipse(
                [x - radius, y - radius, x + radius, y + radius],
                fill=color,
                outline="white",
                width=2,
            )

            # Draw label with background
            text_pos = (x + radius + 5, y - radius - 3)
            bbox = draw.textbbox(text_pos, label, font=font)
            padding = 3
            draw.rectangle(
                [bbox[0] - padding, bbox[1] - padding, bbox[2] + padding, bbox[3] + padding],
                fill="white",
            )
            draw.text(text_pos, label, fill=color, font=font)

        return img

    def process_image(self, image_name):
        """Process a single image to create a colorization matching sample."""
        img_path = self.images_dir / image_name

        try:
            img_raw = ImageOps.exif_transpose(Image.open(img_path))
        except Exception as e:
            print(f"Error opening {image_name}: {e}")
            return None

        # Skip grayscale images (mode 'L' or 'LA', or RGB where all channels are identical)
        if img_raw.mode in ("L", "LA"):
            return None
        img_pil = img_raw.convert("RGB")
        arr_check = np.array(img_pil)
        if (arr_check[:, :, 0] == arr_check[:, :, 1]).all() and \
           (arr_check[:, :, 1] == arr_check[:, :, 2]).all():
            return None

        # Resize to target size (keep aspect ratio, fit in target_size box)
        w, h = img_pil.size
        scale = self.target_size / max(w, h)
        new_w, new_h = int(w * scale), int(h * scale)
        img_resized = img_pil.resize((new_w, new_h), Image.BILINEAR)
        img_array = np.array(img_resized)

        # Sample points with distinct colors
        sampled = self.sample_points_with_distinct_colors(img_array)
        if sampled is None:
            return None

        # Separate coordinates and colors
        points_xy = [(x, y) for x, y, _, _, _ in sampled]
        colors = [(r, g, b) for _, _, r, g, b in sampled]

        # Create grayscale version
        gray_img = img_resized.convert("L")  # single channel grayscale

        # Labels for the points: A, B, C, D, E
        point_labels = [chr(ord("A") + i) for i in range(self.num_points)]

        # Draw points on the grayscale image
        annotated_img = self.draw_points_on_grayscale(gray_img, points_xy, point_labels)

        # Shuffle colors to create the matching task
        shuffled_indices = list(range(self.num_points))
        random.shuffle(shuffled_indices)
        shuffled_colors = [colors[i] for i in shuffled_indices]

        # Build the correct answer mapping: point label -> color number
        answer_mapping = {}
        for point_idx in range(self.num_points):
            for slot_idx in range(self.num_points):
                if shuffled_indices[slot_idx] == point_idx:
                    answer_mapping[point_labels[point_idx]] = slot_idx + 1  # 1-indexed
                    break

        # Save annotated image
        sample_id = str(uuid.uuid4())
        img_filename = f"{sample_id}_colorization.jpg"
        img_output_path = self.images_output_dir / img_filename
        annotated_img.save(img_output_path, quality=95)

        # Format color list for the prompt
        color_list_str = "\n".join(
            f"  {i+1}. {format_rgb(*c)}" for i, c in enumerate(shuffled_colors)
        )

        # Format point coordinates
        point_coord_str = ", ".join(
            f"{label} at ({x}, {y})" for label, (x, y) in zip(point_labels, points_xy)
        )

        # Format answer
        answer_str = ", ".join(
            f"{label}-{answer_mapping[label]}" for label in point_labels
        )

        # Build conversation
        question = (
            f"<image>\n"
            f"This is a grayscale version of a color photograph. "
            f"There are {self.num_points} marked points on the image: {point_coord_str}.\n"
            f"Below are {self.num_points} colors sampled from the original color image at these points, "
            f"but listed in shuffled order:\n{color_list_str}\n\n"
            f"Match each point to its original color. "
            f"Answer with the format: A-number, B-number, C-number, D-number, E-number."
        )

        entry = {
            "id": sample_id,
            "image": img_filename,
            "source_image": image_name,
            "points": [
                {"label": label, "x": x, "y": y, "original_color": list(colors[i])}
                for i, (label, (x, y)) in enumerate(zip(point_labels, points_xy))
            ],
            "shuffled_colors": [list(c) for c in shuffled_colors],
            "answer_mapping": answer_mapping,
            "conversations": [
                {"from": "human", "value": question},
                {"from": "gpt", "value": answer_str},
            ],
        }

        return entry

    def create_dataset(self, max_samples=None):
        """Create the full colorization matching dataset."""
        images = self.get_image_list()
        print(f"Found {len(images)} images in {self.images_dir}")

        if max_samples:
            images = images[:max_samples]
            print(f"Processing first {max_samples} images")

        for image_name in tqdm(images, desc="Processing images"):
            try:
                entry = self.process_image(image_name)
                if entry is not None:
                    self.dataset.append(entry)
            except Exception as e:
                print(f"Error processing {image_name}: {e}")
                continue

        print(f"Successfully created {len(self.dataset)} dataset entries")

        # Save full dataset JSON (with metadata)
        json_path = self.output_dir / "colorization_dataset.json"
        with open(json_path, "w") as f:
            json.dump(self.dataset, f, indent=2)
        print(f"Dataset saved to {json_path}")

        # Convert to LLaVA format: strip metadata, keep only id/image/conversations
        llava_data = [
            {"id": e["id"], "image": e["image"], "conversations": e["conversations"]}
            for e in self.dataset
        ]
        llava_path = self.output_dir / "colorization_dataset_llava.json"
        with open(llava_path, "w") as f:
            json.dump(llava_data, f, indent=2)
        print(f"LLaVA-format dataset saved to {llava_path}")


    def reconstruct_dataset(self, raw_json: str):
        """
        Reconstruct annotated colorization images using metadata from the original raw JSON.

        The raw JSON (colorization_dataset.json) contains per-entry: id, source_image,
        points (label, x, y in resized image space), shuffled_colors, answer_mapping.
        We re-render the exact same grayscale+annotated image for each entry using those
        stored coordinates, so output filenames match what llava_v1_5_v_gift.json references.

        Args:
            raw_json: Path to the original colorization_dataset.json with full metadata.
        """
        print(f"Loading raw colorization metadata from {raw_json}...")
        with open(raw_json) as f:
            entries = json.load(f)
        print(f"Found {len(entries)} entries to reconstruct")

        skipped = 0
        reconstructed = 0

        for i, entry in enumerate(tqdm(entries, desc="Reconstructing colorization images")):
            sample_id = entry["id"]
            source_image = entry["source_image"]
            points_meta = entry["points"]   # list of {label, x, y, original_color}

            img_path = self.images_dir / source_image
            if not img_path.exists():
                print(f"WARNING: source image not found: {img_path}")
                skipped += 1
                continue

            try:
                img_raw = ImageOps.exif_transpose(Image.open(img_path))
                img_pil = img_raw.convert("RGB")

                # Resize the same way as original processing
                w, h = img_pil.size
                scale = self.target_size / max(w, h)
                new_w, new_h = int(w * scale), int(h * scale)
                img_resized = img_pil.resize((new_w, new_h), Image.BILINEAR)

                # Convert to grayscale
                gray_img = img_resized.convert("L")

                # Restore stored points
                points_xy = [(p["x"], p["y"]) for p in points_meta]
                point_labels = [p["label"] for p in points_meta]

                # Draw points on grayscale (same method as original)
                annotated_img = self.draw_points_on_grayscale(gray_img, points_xy, point_labels)

                # Save with original UUID-based filename
                img_filename = f"{sample_id}_colorization.jpg"
                img_output_path = self.images_output_dir / img_filename
                annotated_img.save(img_output_path, quality=95)

                reconstructed += 1

            except Exception as e:
                print(f"Error reconstructing {source_image}: {e}")
                skipped += 1
                continue

        print(f"Reconstruction complete: {reconstructed} images saved, {skipped} skipped")
        print(f"Images saved to: {self.images_output_dir}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Create or reconstruct colorization dataset")
    _repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    parser.add_argument("--reconstruct", action="store_true",
                        help="Reconstruct images using colorization_metadata.json")
    parser.add_argument("--metadata_json", type=str,
                        default=os.path.join(_repo_root, "reconstruction_metadata", "colorization_metadata.json"),
                        help="Path to colorization_metadata.json [reconstruct mode]")
    parser.add_argument("--images_dir", type=str, default=None,
                        help="Override default COCO images directory")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Override default output directory")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Max images to process [original mode only]")
    args = parser.parse_args()

    _repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    _coco_dir = os.path.join(_repo_root, "datasets", "COCO")
    images_dir = args.images_dir or os.path.join(_coco_dir, "images", "train2017")
    output_dir = args.output_dir or os.path.join(_repo_root, "datasets", "colorization")

    creator = ColorizationDatasetCreator(
        images_dir=images_dir,
        output_dir=output_dir,
        num_points=5,
        patch_radius=2,
        target_size=512,
    )

    if args.reconstruct:
        creator.reconstruct_dataset(raw_json=args.metadata_json)
    else:
        creator.create_dataset(max_samples=args.max_samples)


if __name__ == "__main__":
    main()
