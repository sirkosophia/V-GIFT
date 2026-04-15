

---

We provide creation code and metadata to reproduce the **V-GIFT training dataset**, which combines standard visual instruction tuning data with three SSL tasks: rotation prediction, point correspondence, and colorization.

The SSL images are not distributed directly; instead, this repository provides the **reconstruction metadata** and **creation scripts** needed to regenerate them exactly from COCO train2017.


---

## Prerequisites

### 1. COCO train2017 images

```bash
mkdir -p datasets/COCO/images
wget http://images.cocodataset.org/zips/train2017.zip
unzip train2017.zip -d datasets/COCO/images/
rm train2017.zip
```

### 2. DIP segmentation masks *(point correspondence only)*

```bash
mkdir -p datasets/COCO/masks
wget https://huggingface.co/datasets/SophiaSirko/DIP_COCO_pseudolabels/resolve/main/dip_COCO_masks.zip \
     -O datasets/COCO/masks/dip_COCO_masks.zip
unzip datasets/COCO/masks/dip_COCO_masks.zip -d datasets/COCO/masks/
rm datasets/COCO/masks/dip_COCO_masks.zip
```

Masks should end up at `datasets/COCO/masks/train2017/`.

### 3. COCO pairs list *(point correspondence only)*

```bash
mkdir -p datasets/COCO/pairs
wget https://raw.githubusercontent.com/sirkosophia/DIP/main/pairs/COCO_pairs_list_train.txt \
     -O datasets/COCO/pairs/COCO_pairs_list_train.txt
```

### 4. Final training JSON

The full V-GIFT training mix (`llava_v1_5_v_gift.json`, 731 826 entries) is hosted on HuggingFace:

```bash
wget https://huggingface.co/datasets/SophiaSirko/V-GIFT/resolve/main/llava_v1_5_v_gift.json
```

This JSON references the SSL images by relative path (e.g. `rotation/images/<uuid>_rot90.jpg`). The steps below reconstruct those images so all paths resolve correctly.

---

## Reconstructing the SSL Images

Each script has a `--reconstruct` flag that reads the corresponding file from `reconstruction_metadata/` and regenerates the images with the **exact original filenames** referenced in `llava_v1_5_v_gift.json`. No model weights or random seeds are required.

All commands are run from the repo root.

### Rotation

```bash
python data_creation/create_rotation_dataset.py \
  --reconstruct \
  --input_dir  datasets/COCO/images/train2017 \
  --output_dir datasets/rotation/images \
  --output_json datasets/rotation/rotation_dataset.json
```

### Colorization

```bash
python data_creation/create_colorization_dataset.py \
  --reconstruct \
  --images_dir datasets/COCO/images/train2017 \
  --output_dir datasets/colorization
```

### Point Correspondence (LLaVA composite format)

> Requires DIP masks and COCO pairs list (see Prerequisites 2 & 3).

```bash
python data_creation/create_point_correspondence_dataset.py \
  --reconstruct \
  --images_dir datasets/COCO/images/train2017 \
  --output_dir datasets/point_correspondence
```

### Override metadata path (optional)

Each script accepts `--metadata_json` to point to a different metadata file if needed:

```bash
python data_creation/create_rotation_dataset.py \
  --reconstruct \
  --metadata_json /path/to/rotation_metadata.json \
  --input_dir  datasets/COCO/images/train2017 \
  --output_dir datasets/rotation/images \
  --output_json datasets/rotation/rotation_dataset.json
```

---

## What Is in the Metadata Files

Each `reconstruction_metadata/*.json` stores the minimal per-sample information needed to regenerate images deterministically. No conversations or model outputs are included.

| File | Per-entry fields |
|---|---|
| `rotation_metadata.json` | `id`, `source_image`, `rotation_degrees` |
| `colorization_metadata.json` | `id`, `source_image`, `points`, `shuffled_colors`, `answer_mapping` |
| `point_correspondence_metadata.json` | `id`, `image1_name`, `image2_name`, `point_img1`, `points_img2`, `correspondence_index` |

---


## LLaVA 1.5 Instruction Tuning data 
Following [LLaVA](https://github.com/haotian-liu/LLaVA)
Download the images from constituting datasets:

- COCO: [train2017](http://images.cocodataset.org/zips/train2017.zip)
- GQA: [images](https://downloads.cs.stanford.edu/nlp/data/gqa/images.zip)
- OCR-VQA: [download script](https://drive.google.com/drive/folders/1_GYPY5UkUy7HIcR0zq3ZCFgeZN7BAfm_?usp=sharing), **we save all files as `.jpg`**
- TextVQA: [train_val_images](https://dl.fbaipublicfiles.com/textvqa/images/train_val_images.zip)
- VisualGenome: [part1](https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip), [part2](https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip)

After downloading all of them, organize the data as follows in `V-GIFT/datasets`,

```
├── coco
│   └── train2017
├── gqa
│   └── images
├── ocr_vqa
│   └── images
├── textvqa
│   └── train_images
└── vg
    ├── VG_100K
    └── VG_100K_2
```

## LLaVA 1.5 OneVision 

download dataset into `./onevision/datasets/`

Download dataset from [LLaVA-558K-Webdataset](https://huggingface.co/datasets/lmms-lab/LLaVA-558K-Webdataset).

### Create SSL tasks 

### Rotation

```bash
python data_creation/create_rotation_dataset.py \
  --input_dir  datasets/COCO/images/train2017 \
  --output_dir onevision/datasets/rotation/images \
  --output_json onevision/datasets/rotation/rotation_dataset.json
```

### Colorization

```bash
python data_creation/create_colorization_dataset.py \
  --images_dir onevision/datasets/COCO/images/train2017 \
  --output_dir onevision/datasets/colorization \
  --output_json onevision/datasets/colorization/colorization_dataset.json
```

### Point Correspondence (LLaVA OneVision 2 image input format)

> Requires DIP masks and COCO pairs list (see Prerequisites 2 & 3).

```bash
python data_creation/create_point_correspondence_dataset_onevision.py \
  --images_dir onevision/datasets/COCO/images/train2017 \
  --output_dir onevision/datasets/point_correspondence \
  --output_json onevision/datasets/point_correspondence/point_correspondence_dataset.json
```


### Convert to web datasets 

### Rotation

```bash
python onevision/tools/data_preprocess/convert_llava_json_to_webdataset.py \
      --input_json  onevision/datasets/rotation/rotation_dataset.json \
      --image_base onevision/datasets/rotation/images \
      --output_dir  onevision/datasets/rotation_webdataset \
      --shard_size  10000
```

### Colorization

```bash
python onevision/tools/data_preprocess/convert_llava_json_to_webdataset.py \
  --input_json onevision/datasets/colorization/colorization_dataset.json \
  --image_base onevision/datasets/colorization/images \
  --output_dir  onevision/datasets/colorization_webdataset \
  --shard_size  10000
```

###  Point Correspondence 

```bash
python onevision/tools/data_preprocess/convert_llava_json_to_webdataset.py \
  --input_json onevision/datasets/point_correspondence/point_correspondence_dataset.json \
  --image_base onevision/datasets/point_correspondence/images \
  --output_dir  onevision/datasets/point_correspondence_webdataset \
  --shard_size  10000
```
