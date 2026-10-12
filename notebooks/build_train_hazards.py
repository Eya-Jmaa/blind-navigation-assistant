"""Builds notebooks/train_hazards.ipynb. Usage: python notebooks/build_train_hazards.py notebooks/train_hazards.ipynb"""
import json
import sys

cells = []


def md(text):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": text.strip("\n")})


def code(text):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                  "source": text.strip("\n")})


md('''
# Phase 6 — Training a street-hazard detector

Pretrained YOLO (COCO, 80 classes) has never seen **stairs, poles, pillars, bins, cones**.
We fine-tune YOLO11s on a public dataset built for visually impaired navigation:

> *Object Detection Dataset: Navigation Assistance for the Visually Impaired People using YOLOv11*
> (Mendeley Data, CC BY 4.0, DOI 10.17632/m68g3h7p87.1) — 8,114 images filmed from a pedestrian's point of view.

**Kaggle settings (right panel) before running:**
1. *Session options → Accelerator:* **GPU** (T4 or P100)
2. *Session options → Internet:* **On** (needs a phone-verified Kaggle account)
3. Run with **Save Version → Save & Run All (Commit)**: it runs in the background (you can close the
   browser), and everything written to `/kaggle/working` is kept in the *Output* tab.
''')

code('''
import os
from pathlib import Path

ON_KAGGLE = Path("/kaggle").exists()
# Kaggle keeps only /kaggle/working as output. The raw data goes to /tmp (fast, not saved).
RAW = Path("/tmp/raw") if ON_KAGGLE else Path(os.environ["LOCAL_RAW"])
DATA = Path("/tmp/hazards") if ON_KAGGLE else Path(os.environ["LOCAL_DATA"])
OUT = Path("/kaggle/working") if ON_KAGGLE else Path(os.environ["LOCAL_OUT"])

# Full training on Kaggle; a tiny smoke test anywhere else (to check the code runs).
EPOCHS = 60 if ON_KAGGLE else 1
BATCH = 32 if ON_KAGGLE else 4
FRACTION = 1.0 if ON_KAGGLE else 0.02
''')

code('''
!pip install -q ultralytics
''')

md('''
## 1. Download the dataset
''')

code('''
!mkdir -p /tmp/raw && wget -q -O /tmp/raw/data.zip "https://data.mendeley.com/public-api/zip/m68g3h7p87/download/1" && unzip -q -o /tmp/raw/data.zip -d /tmp/raw && rm /tmp/raw/data.zip
''')

code('''
import yaml

# The zip contains a folder with data.yaml + train/ valid/ test/ — find it wherever it is.
SRC = next(RAW.rglob("data.yaml")).parent
names = yaml.safe_load(open(SRC / "data.yaml"))["names"]
print("Dataset folder:", SRC)
print(len(names), "classes:", names)
''')

md('''
## 2. Look at the data before training

Always check **how many examples of each class** you have. A class with 50 examples will be
learned much worse than one with 5,000 — this tells you in advance which results to trust.
''')

code('''
from collections import Counter

for split in ["train", "valid", "test"]:
    counts = Counter()
    label_files = list((SRC / split / "labels").glob("*.txt"))
    for f in label_files:
        for line in f.read_text().splitlines():
            if line.strip():
                counts[names[int(line.split()[0])]] += 1
    print(f"{split}: {len(label_files)} images")
    for name, n in counts.most_common():
        print(f"   {name:16s} {n}")
''')

md('''
## 3. Keep only the hazards we need

Our app already uses the COCO model for people and vehicles. This second model only needs the
**obstacles COCO doesn't know**. We:
- keep 6 classes and renumber them 0..5 (*Tree* is dropped: 2 validation and 0 test examples,
  so it could not even be evaluated),
- convert the **polygon** labels (segmentation outlines) to **boxes** (min/max of the polygon),
- keep some images with no hazard as **background images** (empty label file): they teach the
  model what is *not* a hazard. But ~70% of the training images have no hazard at all: with that
  many, the model mostly learns to stay silent (and trains 3x slower). We keep 1 background for
  every 2 hazard images in *train*. Valid/test stay untouched: they must reflect reality.
''')

code('''
import random
import shutil

BACKGROUND_PER_HAZARD_IMAGE = 0.5  # train only

KEEP = {  # dataset class -> our class name
    "Stairs": "stairs",
    "Electric Pole": "pole",
    "Building Pillar": "pillar",
    "Bin": "bin",
    "Parking Cone": "cone",
    "Obstacle": "obstacle",
}
NEW_NAMES = list(KEEP.values())
OLD_TO_NEW = {names.index(old): NEW_NAMES.index(new) for old, new in KEEP.items()}


def to_box(values):
    """YOLO label values -> (cx, cy, w, h). Polygons (x1 y1 x2 y2 ...) become their bounding box."""
    if len(values) == 4:
        return values
    xs, ys = values[0::2], values[1::2]
    x1, x2, y1, y2 = max(min(xs), 0), min(max(xs), 1), max(min(ys), 0), min(max(ys), 1)
    return (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1


def link_or_copy(src, dst):
    try:
        os.link(src, dst)  # hard link: instant, no extra disk space
    except OSError:
        shutil.copy(src, dst)


def convert(label):
    """Original label file -> list of YOLO box lines for our classes only."""
    lines = []
    if label.exists():
        for line in label.read_text().splitlines():
            parts = line.split()
            if not parts or int(parts[0]) not in OLD_TO_NEW:
                continue
            cx, cy, w, h = to_box([float(v) for v in parts[1:]])
            if w * h < 1e-5:
                continue  # degenerate box (a few pixels): annotation noise
            lines.append(f"{OLD_TO_NEW[int(parts[0])]} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    return lines


random.seed(0)
stats = Counter()
for split in ["train", "valid", "test"]:
    (DATA / split / "images").mkdir(parents=True, exist_ok=True)
    (DATA / split / "labels").mkdir(parents=True, exist_ok=True)
    hazard, background = [], []
    for image in sorted((SRC / split / "images").iterdir()):
        lines = convert(SRC / split / "labels" / (image.stem + ".txt"))
        (hazard if lines else background).append((image, lines))
    if split == "train":
        background = random.sample(background, int(len(hazard) * BACKGROUND_PER_HAZARD_IMAGE))
    for image, lines in hazard + background:
        link_or_copy(image, DATA / split / "images" / image.name)
        (DATA / split / "labels" / (image.stem + ".txt")).write_text("\\n".join(lines))
        for line in lines:
            stats[f"{split}/{NEW_NAMES[int(line.split()[0])]}"] += 1
    stats[f"{split}/images_with_hazards"] = len(hazard)
    stats[f"{split}/background_images"] = len(background)

for key in sorted(stats):
    print(f"{key:28s} {stats[key]}")
''')

md('''
## 4. Visual check: do the boxes look right?

The most common silent bug in ML: labels that don't match the images (wrong class ids, shifted
boxes...). The model would train "fine" and learn garbage. Look before you train.
''')

code('''
import random
import cv2
import matplotlib.pyplot as plt

random.seed(1)
with_hazards = [p for p in (DATA / "train" / "labels").glob("*.txt") if p.read_text().strip()]
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
for ax, label in zip(axes.flat, random.sample(with_hazards, 6)):
    image_path = next((DATA / "train" / "images").glob(label.stem + ".*"))
    img = cv2.cvtColor(cv2.imread(str(image_path)), cv2.COLOR_BGR2RGB)
    H, W = img.shape[:2]
    for line in label.read_text().splitlines():
        c, cx, cy, w, h = line.split()
        cx, cy, w, h = float(cx) * W, float(cy) * H, float(w) * W, float(h) * H
        x1, y1 = int(cx - w / 2), int(cy - h / 2)
        cv2.rectangle(img, (x1, y1), (int(x1 + w), int(y1 + h)), (255, 0, 0), 3)
        cv2.putText(img, NEW_NAMES[int(c)], (x1, max(y1 - 8, 20)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    ax.imshow(img)
    ax.axis("off")
plt.tight_layout()
plt.show()
''')

md('''
## 5. Train

**Transfer learning**: we start from `yolo11s.pt`, already trained on COCO. It already knows edges,
textures, shapes — we only teach it the new classes. Much faster and better than starting from zero.

- `epochs=60`, `patience=15`: stop early if the validation score hasn't improved for 15 epochs.
- Default augmentations (mosaic, color jitter, flips, scaling) multiply the variety of the data.
- About 1.5–2 h on a Kaggle GPU.
''')

code('''
data_yaml = DATA / "hazards.yaml"
data_yaml.write_text(yaml.safe_dump({
    "path": str(DATA), "train": "train/images", "val": "valid/images", "test": "test/images",
    "names": dict(enumerate(NEW_NAMES)),
}))

from ultralytics import YOLO

model = YOLO("yolo11s.pt")
model.train(
    data=str(data_yaml), epochs=EPOCHS, batch=BATCH, fraction=FRACTION, imgsz=640,
    patience=15, seed=0, project=str(OUT / "runs"), name="hazards", exist_ok=True,
    # Data-loading processes. On Windows they must re-launch the script ("spawn"), which
    # hangs outside a normal script: load data in the main process there instead.
    workers=8 if ON_KAGGLE else 0,
)
''')

md('''
## 6. Evaluate on the TEST set (images never used during training)

- **Precision**: of what the model detects, how much is right.
- **Recall**: of what is really there, how much the model finds.
- **mAP50**: average precision when a box counts as correct if it overlaps the truth by ≥ 50%.
- **mAP50-95**: same, averaged over stricter overlaps (50% → 95%): also measures box *accuracy*.
''')

code('''
best = OUT / "runs" / "hazards" / "weights" / "best.pt"
metrics = YOLO(best).val(data=str(data_yaml), split="test", batch=BATCH, plots=True,
                         project=str(OUT / "runs"), name="hazards_test", exist_ok=True,
                         workers=8 if ON_KAGGLE else 0)

print(f"\\n{'class':10s} {'mAP50':>7s} {'mAP50-95':>9s}")
for i, name in enumerate(NEW_NAMES):
    if i in metrics.box.ap_class_index:
        j = list(metrics.box.ap_class_index).index(i)
        print(f"{name:10s} {metrics.box.ap50[j]:7.3f} {metrics.box.maps[i]:9.3f}")
print(f"{'ALL':10s} {metrics.box.map50:7.3f} {metrics.box.map:9.3f}")
''')

code('''
from IPython.display import Image, display

display(Image(str(OUT / "runs" / "hazards" / "results.png"), width=1000))
display(Image(str(OUT / "runs" / "hazards_test" / "confusion_matrix_normalized.png"), width=800))
''')

md('''
## 7. Save the model

Download `hazards.pt` from the **Output** tab and put it in `models/hazards.pt` in the project.
''')

code('''
shutil.copy(best, OUT / "hazards.pt")
print("Saved", OUT / "hazards.pt", f"({(OUT / 'hazards.pt').stat().st_size / 1e6:.1f} MB)")
''')

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                                   "name": "python3"},
                                    "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
for c in cells:  # nbformat stores source as a list of lines
    c["source"] = [l + "\n" for l in c["source"].split("\n")]
    c["source"][-1] = c["source"][-1].rstrip("\n")
json.dump(nb, open(sys.argv[1], "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print("written", sys.argv[1], len(cells), "cells")
