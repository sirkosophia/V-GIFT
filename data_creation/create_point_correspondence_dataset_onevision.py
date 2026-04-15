import os
import json
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps
from pathlib import Path
import uuid
from tqdm import tqdm
import random

# torch/torchvision are only needed for original creation mode (DINO feature extraction).
# In --reconstruct mode they are never used, so we tolerate import failures.
try:
    import torch
    import torch.nn as nn
    import torchvision.transforms as transforms
    import torchvision.transforms.functional as TF
    torch.manual_seed(42)
    _TORCH_AVAILABLE = True
except (ImportError, RuntimeError):
    torch = nn = transforms = TF = None
    _TORCH_AVAILABLE = False

_NNModuleBase = nn.Module if _TORCH_AVAILABLE else object

# Set seeds for reproducibility
random.seed(42)
np.random.seed(42)


# DINOv2 models
dino_models = {
    "small": ("dinov2_vits14_reg_lc", 384, 14),
    "base": ("dinov2_vitb14_reg_lc", 768, 14),
    "large": ("dinov2_vitl14_reg_lc", 1024, 14),
}


class DinoVitFeatureExtractorFinetune(_NNModuleBase):
    """
    DINO Vision Transformer Feature Extractor.
    """

    def __init__(
        self,
        image_backbone="small",
        last_n_feat=1,
        path=None,
    ):
        super().__init__()
        model_dict = dino_models
        repo_path = "facebookresearch/dinov2"

        assert (
            image_backbone in model_dict.keys()
        ), f"DinoVitFeatureExtractor is only available for {model_dict.keys()}"

        model_name, embed_dim, patch_size = model_dict[image_backbone]
        print(f"image_backbone: {image_backbone}, model_name: {model_name}, embed_dim: {embed_dim}, patch_size: {patch_size}")

        self.last_n_feat = last_n_feat
        self.embed_dim = embed_dim * self.last_n_feat

        path = path or repo_path
        encoder = torch.hub.load(path, model_name, source='github')

        self.encoder = encoder.backbone if hasattr(encoder, 'backbone') else encoder
        self.encoder.eval()
        self.patch_size = patch_size

    def forward(self, x):
        if len(x.shape) == 3:
            x = x.unsqueeze(0)

        batch_size, _, height, width = x.size()
        assert (height % self.patch_size) == 0
        assert (width % self.patch_size) == 0
        f_height = height // self.patch_size
        f_width = width // self.patch_size

        output = self.encoder.get_intermediate_layers(x, n=self.last_n_feat, return_class_token=True)
        x, _ = output[0]
        x = x.transpose(1, 2).view(batch_size, self.embed_dim, f_height, f_width)
        return x


class PointCorrespondenceDatasetCreator:
    def __init__(
        self,
        pairs_file,
        images_dir,
        masks_dir,
        output_dir,
        image_backbone='base',
        version='v3',
        device='cuda' if (_TORCH_AVAILABLE and torch.cuda.is_available()) else 'cpu',
        reconstruct=False,
    ):
        self.pairs_file = pairs_file
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.output_dir = Path(output_dir)
        self.device = device

        # Create output directories
        self.images_output_dir = self.output_dir / 'images'
        self.images_output_dir.mkdir(parents=True, exist_ok=True)

        if reconstruct:
            # Reconstruct mode: point coordinates are read from metadata,
            # so the DINO model and masks are not needed.
            self.model = None
            self.patch_size = None
        else:
            print(f"Loading DINO{version} model with backbone: {image_backbone}")
            self.model = DinoVitFeatureExtractorFinetune(
                image_backbone=image_backbone,
                last_n_feat=1,
                version=version
            )
            self.model = self.model.to(self.device)
            self.model.eval()
            self.patch_size = self.model.patch_size

        self.dataset = []

    # ---------------------------
    # Augmenter-style geometry:
    # CenterCrop(c=min(w,h)-1) -> Resize(augment_size, augment_size)
    # ---------------------------

    def _get_center_square_crop_params(self, pil_img, crop_size: int):
        """Return (top, left, height, width) for a center square crop."""
        w, h = pil_img.size
        crop_size = min(int(crop_size), w, h)
        top = int(round((h - crop_size) / 2.0))
        left = int(round((w - crop_size) / 2.0))
        return top, left, crop_size, crop_size

    def _augmenter_image(self, pil_img, augment_size: int = 224):
        """
        Exactly matches:
            T.CenterCrop((c,c)) where c=min(w,h)-1
            T.Resize((augment_size, augment_size), BILINEAR, antialias=None)
            ToTensor + Normalize
        Returns:
            img_t: torch.FloatTensor (3,H,W) normalized
            params: dict for mapping points 224->original
        """
        w, h = pil_img.size
        c = min(w, h) - 1
        top, left, ch, cw = self._get_center_square_crop_params(pil_img, c)

        img_c = TF.crop(pil_img, top, left, ch, cw)
        img_r = TF.resize(
            img_c,
            (augment_size, augment_size),
            interpolation=transforms.InterpolationMode.BILINEAR,
            antialias=None,
        )

        img_t = TF.to_tensor(img_r)
        img_t = TF.normalize(img_t, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        # Map 224-space -> crop-space -> original:
        # crop_coord = coord_224 * (c / augment_size)
        scale = c / float(augment_size)
        params = {
            "left": left,
            "top": top,
            "scale": scale,
            "crop_size": c,
            "augment_size": augment_size,
            "orig_w": w,
            "orig_h": h,
        }
        return img_t, params

    def _augmenter_mask(self, mask_pil, orig_w: int, orig_h: int, augment_size: int = 224):
        """
        Apply same geometry to label masks:
            CenterCrop(c=min(w,h)-1) -> Resize(augment_size, augment_size) with NEAREST
        Returns:
            mask_np: numpy array (augment_size, augment_size)
        """
        # Guard: if mask is already in augmented 224-space, don't transform again.
        arr0 = np.array(mask_pil)
        if arr0.ndim == 2 and arr0.shape == (augment_size, augment_size):
            return arr0
        c = min(orig_w, orig_h) - 1
        top = int(round((orig_h - c) / 2.0))
        left = int(round((orig_w - c) / 2.0))

        m_c = TF.crop(mask_pil, top, left, c, c)
        m_r = TF.resize(
            m_c,
            (augment_size, augment_size),
            interpolation=transforms.InterpolationMode.NEAREST,
            antialias=None,
        )
        return np.array(m_r)

    def _map_point_to_original_aug(self, point_224, aug_params):
        """Map point from augmenter 224-space back to original space."""
        x, y = point_224
        x_crop = x * aug_params["scale"]
        y_crop = y * aug_params["scale"]
        x_orig = int(round(x_crop + aug_params["left"]))
        y_orig = int(round(y_crop + aug_params["top"]))
        x_orig = max(0, min(x_orig, aug_params["orig_w"] - 1))
        y_orig = max(0, min(y_orig, aug_params["orig_h"] - 1))
        return (x_orig, y_orig)


    def load_pairs(self):
        """Load image pairs from the pairs file."""
        pairs = []
        with open(self.pairs_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(': ')
                if len(parts) == 2:
                    img1 = parts[0].strip()
                    img2 = parts[1].strip()
                    pairs.append((img1, img2))
        return pairs

    def load_mask(self, image_name):
        """Load segmentation mask for an image."""
        mask_name = image_name.replace('.jpg', '.pt')
        mask_path = self.masks_dir / mask_name

        if not mask_path.exists():
            return None

        try:
            mask_data = torch.load(mask_path, map_location='cpu')

            if isinstance(mask_data, dict):
                for key in ['masks', 'mask', 'segmentation']:
                    if key in mask_data:
                        mask_tensor = mask_data[key]
                        break
                else:
                    mask_tensor = list(mask_data.values())[0]
            else:
                mask_tensor = mask_data

            if isinstance(mask_tensor, torch.Tensor):
                mask_tensor = mask_tensor.long()

            mask_tensor = mask_tensor.squeeze()

            return mask_tensor
        except Exception as e:
            print(f"Error loading mask {mask_path}: {e}")
            return None

    def extract_dense_features(self, image_input):
        """Extract dense features from an image using DINO."""
        img_tensor = image_input
        if len(img_tensor.shape) == 3:
            img_tensor = img_tensor.unsqueeze(0)
        img_tensor = img_tensor.to(self.device)

        with torch.no_grad():
            features = self.model(img_tensor)
            batch_size, embed_dim, patch_h, patch_w = features.shape
            features = features.permute(0, 2, 3, 1)
            features = features.reshape(-1, embed_dim)

        return features, patch_h, patch_w, None

    def find_shared_object_mask(self, mask1, mask2, img1_pil, img2_pil):
        """Find the biggest shared object between two segmentation masks."""
        if mask1 is None or mask2 is None:
            return None

        if torch.is_tensor(mask1):
            mask1 = mask1.numpy()
        if torch.is_tensor(mask2):
            mask2 = mask2.numpy()

        mask1 = np.squeeze(mask1)
        mask2 = np.squeeze(mask2)

        if len(mask1.shape) != 2 or len(mask2.shape) != 2:
            print(f"Warning: Unexpected mask shapes: {mask1.shape}, {mask2.shape}")
            return None

        mask1_pil = Image.fromarray(mask1.astype(np.int32), mode='I')
        mask2_pil = Image.fromarray(mask2.astype(np.int32), mode='I')

        w1, h1 = img1_pil.size
        w2, h2 = img2_pil.size

        mask1_resized = self._augmenter_mask(mask1_pil, w1, h1, augment_size=224)
        mask2_resized = self._augmenter_mask(mask2_pil, w2, h2, augment_size=224)

        labels1 = set(np.unique(mask1_resized)) - {0}
        labels2 = set(np.unique(mask2_resized)) - {0}

        shared_labels = labels1 & labels2

        if len(shared_labels) == 0:
            print("Warning: No shared object labels found between the two masks")
            return (mask1_resized > 0).astype(np.uint8), (mask2_resized > 0).astype(np.uint8)

        max_intersection = 0
        best_label = None

        for label in shared_labels:
            binary1 = (mask1_resized == label)
            binary2 = (mask2_resized == label)
            intersection_size = np.sum(binary1) + np.sum(binary2)

            if intersection_size > max_intersection:
                max_intersection = intersection_size
                best_label = label

        if best_label is None:
            print("Warning: No valid intersection found")
            return (mask1_resized > 0).astype(np.uint8), (mask2_resized > 0).astype(np.uint8)

        print(f"Selected object with label {best_label}, intersection size: {max_intersection}")

        overlap_mask1 = (mask1_resized == best_label).astype(np.uint8)
        overlap_mask2 = (mask2_resized == best_label).astype(np.uint8)

        print(f"Final overlap mask shapes: {overlap_mask1.shape}, {overlap_mask2.shape}")
        print(f"Overlap mask 1: {overlap_mask1.sum()} pixels")
        print(f"Overlap mask 2: {overlap_mask2.sum()} pixels")

        return overlap_mask1, overlap_mask2

    def sample_point_from_mask(self, mask):
        """Sample a random point from a binary mask."""
        if mask is None:
            return None

        coords = np.argwhere(mask > 0)
        if len(coords) == 0:
            return None

        idx = np.random.randint(0, len(coords))
        y, x = coords[idx]
        return (int(x), int(y))

    def find_correspondence(self, point1, features1, features2, patch_h1, patch_w1, patch_h2, patch_w2, size1, size2):
        """Find corresponding point in image2 using dense feature matching."""
        x1, y1 = point1

        patch_x1 = int(x1 / self.patch_size)
        patch_y1 = int(y1 / self.patch_size)

        patch_x1 = max(0, min(patch_x1, patch_w1 - 1))
        patch_y1 = max(0, min(patch_y1, patch_h1 - 1))

        patch_idx1 = patch_y1 * patch_w1 + patch_x1
        feature_vec1 = features1[patch_idx1:patch_idx1+1]

        features2_norm = torch.nn.functional.normalize(features2, dim=1)
        feature_vec1_norm = torch.nn.functional.normalize(feature_vec1, dim=1)

        similarities = torch.mm(feature_vec1_norm, features2_norm.t())

        best_patch_idx = similarities.argmax().item()

        patch_y2 = best_patch_idx // patch_w2
        patch_x2 = best_patch_idx % patch_w2

        x2 = int((patch_x2 + 0.5) * self.patch_size)
        y2 = int((patch_y2 + 0.5) * self.patch_size)
        x2 = max(0, min(x2, size2[0] - 1))
        y2 = max(0, min(y2, size2[1] - 1))

        return (x2, y2)

    def draw_points_on_image(self, image_path, points, labels, output_path):
        """Draw points on image with labels and save to output_path."""
        img = Image.open(image_path).convert('RGB')
        draw = ImageDraw.Draw(img)

        color = 'red'
        radius = 12

        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 50)
        except:
            try:
                font = ImageFont.truetype("arial.ttf", 50)
            except:
                font = ImageFont.load_default()

        for point, label in zip(points, labels):
            if point is None:
                continue
            x, y = point

            draw.ellipse([x-radius, y-radius, x+radius, y+radius], fill=color, outline=color, width=3)

            text_position = (x+radius+8, y-radius-5)
            bbox = draw.textbbox(text_position, label, font=font)
            padding = 5
            bg_bbox = [bbox[0]-padding, bbox[1]-padding, bbox[2]+padding, bbox[3]+padding]
            draw.rectangle(bg_bbox, fill='white')
            draw.text(text_position, label, fill=color, font=font)

        img.save(output_path)
        return img

    def process_pair(self, img1_name, img2_name):
        """Process a single image pair."""
        img1_path = self.images_dir / img1_name
        img2_path = self.images_dir / img2_name

        if not img1_path.exists() or not img2_path.exists():
            print(f"Images not found: {img1_name} or {img2_name}")
            return None

        img1_pil = ImageOps.exif_transpose(Image.open(img1_path).convert("RGB"))
        img2_pil = ImageOps.exif_transpose(Image.open(img2_path).convert("RGB"))

        mask1 = self.load_mask(img1_name)
        mask2 = self.load_mask(img2_name)

        if mask1 is None or mask2 is None:
            print(f"Masks not found for: {img1_name} or {img2_name}")
            return None

        result = self.find_shared_object_mask(mask1, mask2, img1_pil, img2_pil)
        if result is None:
            return None
        mask1_binary, mask2_binary = result

        img1_transformed, aug_params1 = self._augmenter_image(img1_pil, augment_size=224)
        img2_transformed, aug_params2 = self._augmenter_image(img2_pil, augment_size=224)

        point1 = self.sample_point_from_mask(mask1_binary)
        if point1 is None:
            print(f"No valid points in mask for {img1_name}")
            return None

        x1, y1 = point1
        if not (0 <= x1 < mask1_binary.shape[1] and 0 <= y1 < mask1_binary.shape[0]):
            print(f"Warning: Point {point1} out of bounds for image {img1_name}")
            return None
        if mask1_binary[y1, x1] == 0:
            print(f"Warning: Point {point1} not on mask for image {img1_name}")
            return None

        features1, patch_h1, patch_w1, _ = self.extract_dense_features(img1_transformed)
        features2, patch_h2, patch_w2, _ = self.extract_dense_features(img2_transformed)

        transformed_size = (224, 224)

        point2_correspondence = self.find_correspondence(
            point1, features1, features2,
            patch_h1, patch_w1, patch_h2, patch_w2,
            transformed_size, transformed_size
        )

        additional_points = []
        for _ in range(2):
            pt = self.sample_point_from_mask(mask2_binary)
            if pt is not None:
                additional_points.append(pt)

        all_points_img2 = [point2_correspondence] + additional_points
        while len(all_points_img2) < 3:
            pt = self.sample_point_from_mask(mask2_binary)
            if pt is not None:
                all_points_img2.append(pt)
        all_points_img2 = all_points_img2[:3]

        indices = list(range(3))
        random.shuffle(indices)
        all_points_img2_shuffled = [all_points_img2[i] for i in indices]
        correspondence_index = indices.index(0)

        # Map all points back to original image space
        point1_orig = self._map_point_to_original_aug(point1, aug_params1)
        all_points_img2_orig = [self._map_point_to_original_aug(pt, aug_params2) for pt in all_points_img2_shuffled]

        # Generate unique ID
        sample_id = str(uuid.uuid4())

        # --- LLaVA-OneVision: save the two images separately ---
        img1_filename = f"{sample_id}_img1.jpg"
        img2_filename = f"{sample_id}_img2.jpg"

        img1_out_path = self.images_output_dir / img1_filename
        img2_out_path = self.images_output_dir / img2_filename

        # Image 1: single query point labelled "Q"
        self.draw_points_on_image(
            img1_path,
            [point1_orig],
            ["Q"],
            img1_out_path,
        )

        # Image 2: three candidate points labelled "0", "1", "2"
        self.draw_points_on_image(
            img2_path,
            all_points_img2_orig,
            ["0", "1", "2"],
            img2_out_path,
        )

        # Format point strings for the conversation
        point1_str = f"({point1_orig[0]}, {point1_orig[1]})"
        points2_str = [f"({p[0]}, {p[1]})" for p in all_points_img2_orig]

        entry = {
            "id": sample_id,
            # List of two images — LLaVA-OneVision multi-image format
            "image": [img1_filename, img2_filename],
            "image1_name": img1_name,
            "image2_name": img2_name,
            "point_img1": point1_str,
            "points_img2": points2_str,
            "correspondence_index": correspondence_index,
            "conversations": [
                {
                    "from": "human",
                    "value": (
                        "<image>\n<image>\n"
                        f"The first image shows a query point Q at {point1_str} on a shared object. "
                        f"The second image shows three candidate points: {', '.join(points2_str)}. "
                        "Which candidate point in the second image corresponds to the query point Q in the first image? "
                        "Answer with the index (0, 1, or 2) directly."
                    ),
                },
                {
                    "from": "gpt",
                    "value": str(correspondence_index),
                },
            ],
        }

        return entry

    def reconstruct_dataset(self, raw_json: str):
        """
        Reconstruct two-image point-correspondence images using metadata from the original raw JSON.

        The raw JSON (point_correspondence_dataset.json from onevision) contains per-entry:
        id, image (list of [img1_filename, img2_filename]), image1_name, image2_name,
        point_img1 (original-space coords as string), points_img2 (list of coord strings),
        correspondence_index.

        We re-draw the query point on img1 and the 3 candidate points on img2 using the
        stored original-space coordinates, so output filenames match the original dataset.

        Args:
            raw_json: Path to the raw onevision point correspondence JSON with full metadata.
        """
        import re
        print(f"Loading raw onevision point-correspondence metadata from {raw_json}...")
        with open(raw_json) as f:
            entries = json.load(f)
        print(f"Found {len(entries)} entries to reconstruct")

        def parse_point(s):
            m = re.match(r'\((\d+),\s*(\d+)\)', s.strip())
            if m:
                return (int(m.group(1)), int(m.group(2)))
            return None

        skipped = 0
        reconstructed = 0

        for entry in tqdm(entries, desc="Reconstructing onevision point-correspondence images"):
            sample_id = entry["id"]
            img1_name = entry.get("image1_name")
            img2_name = entry.get("image2_name")
            point_img1_str = entry.get("point_img1")
            points_img2_str = entry.get("points_img2", [])

            if not (img1_name and img2_name and point_img1_str and points_img2_str):
                skipped += 1
                continue

            img1_path = self.images_dir / img1_name
            img2_path = self.images_dir / img2_name
            if not img1_path.exists() or not img2_path.exists():
                print(f"WARNING: images not found: {img1_name} / {img2_name}")
                skipped += 1
                continue

            point1 = parse_point(point_img1_str)
            points2 = [parse_point(s) for s in points_img2_str]
            if point1 is None or None in points2:
                print(f"WARNING: could not parse points for {sample_id}")
                skipped += 1
                continue

            img1_filename = f"{sample_id}_img1.jpg"
            img2_filename = f"{sample_id}_img2.jpg"
            img1_out_path = self.images_output_dir / img1_filename
            img2_out_path = self.images_output_dir / img2_filename

            try:
                self.draw_points_on_image(img1_path, [point1], ["Q"], img1_out_path)
                self.draw_points_on_image(img2_path, points2, ["0", "1", "2"], img2_out_path)
                reconstructed += 1
            except Exception as e:
                print(f"Error reconstructing {sample_id}: {e}")
                skipped += 1

        print(f"Reconstruction complete: {reconstructed} pairs saved, {skipped} skipped")
        print(f"Images saved to: {self.images_output_dir}")

    def create_dataset(self, max_samples=None):
        """Create the full point correspondence dataset."""
        pairs = self.load_pairs()
        print(f"Loaded {len(pairs)} image pairs")

        if max_samples:
            pairs = pairs[:max_samples]
            print(f"Processing first {max_samples} pairs")

        for img1, img2 in tqdm(pairs, desc="Processing pairs"):
            try:
                entry = self.process_pair(img1, img2)
                if entry is not None:
                    self.dataset.append(entry)
            except Exception as e:
                print(f"Error processing pair ({img1}, {img2}): {e}")
                continue

        print(f"Successfully created {len(self.dataset)} dataset entries")

        json_path = self.output_dir / "point_correspondence_dataset.json"
        with open(json_path, 'w') as f:
            json.dump(self.dataset, f, indent=2)

        print(f"Dataset saved to {json_path}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Create or reconstruct onevision point correspondence dataset")
    _repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    parser.add_argument("--reconstruct", action="store_true",
                        help="Reconstruct images using point_correspondence_onevision_metadata.json")
    parser.add_argument("--metadata_json", type=str,
                        default=os.path.join(_repo_root, "reconstruction_metadata", "point_correspondence_onevision_metadata.json"),
                        help="Path to point_correspondence_onevision_metadata.json [reconstruct mode]")
    parser.add_argument("--images_dir", type=str, default=None,
                        help="Override default COCO images directory")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Override default output directory")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Max pairs to process [original mode only]")
    args = parser.parse_args()

    _repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    _coco_dir = os.path.join(_repo_root, "datasets", "COCO")
    pairs_file = os.path.join(_coco_dir, "pairs", "COCO_pairs_list_train.txt")
    images_dir = args.images_dir or os.path.join(_coco_dir, "images", "train2017")
    masks_dir = os.path.join(_coco_dir, "masks", "train2017")
    output_dir = args.output_dir or os.path.join(_repo_root, "datasets", "point_correspondence_coco_llava_onevision")

    creator = PointCorrespondenceDatasetCreator(
        pairs_file=pairs_file,
        images_dir=images_dir,
        masks_dir=masks_dir,
        output_dir=output_dir,
        image_backbone='base',
        version='v2',
        reconstruct=args.reconstruct,
    )

    if args.reconstruct:
        creator.reconstruct_dataset(raw_json=args.metadata_json)
    else:
        creator.create_dataset(max_samples=args.max_samples)


if __name__ == "__main__":
    main()
