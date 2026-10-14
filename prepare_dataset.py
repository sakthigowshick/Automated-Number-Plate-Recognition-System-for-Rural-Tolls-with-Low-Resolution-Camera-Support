from pathlib import Path
import xml.etree.ElementTree as ET
import random
import re
import shutil
import yaml

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "google_images"
OUT = ROOT / "yolo_dataset"
SEED = 42

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def normalize(name):
    p = Path(name).name.lower()
    # Handles names such as photo.jpg.xml and photo.jpg.jpeg
    while Path(p).suffix.lower() in IMAGE_EXTS | {".xml"}:
        p = str(Path(p).with_suffix(""))
    return p

def valid_plate_label(text):
    text = (text or "").strip().upper()
    # Reject obvious non-plate labels such as CRETA, blur, TERRANO.
    return (
        6 <= len(text) <= 12
        and text.isalnum()
        and any(c.isalpha() for c in text)
        and any(c.isdigit() for c in text)
    )

images = [
    p for p in SOURCE.rglob("*")
    if p.is_file() and p.suffix.lower() in IMAGE_EXTS
]
xmls = list(SOURCE.rglob("*.xml"))

image_map = {}
for p in images:
    image_map.setdefault(normalize(p.name), []).append(p)

records = []
unmatched = []
bad = []

for xml_path in xmls:
    try:
        root = ET.parse(xml_path).getroot()
        filename = root.findtext("filename", "").strip()

        candidates = []
        if filename:
            candidates.extend(image_map.get(normalize(filename), []))

        candidates.extend(image_map.get(normalize(xml_path.name), []))

        # Deduplicate candidates.
        candidates = list(dict.fromkeys(candidates))

        if len(candidates) != 1:
            unmatched.append(xml_path.name)
            continue

        image_path = candidates[0]
        width = int(root.findtext("size/width", "0"))
        height = int(root.findtext("size/height", "0"))

        if width <= 0 or height <= 0:
            bad.append((xml_path.name, "invalid image dimensions"))
            continue

        boxes = []
        for obj in root.findall("object"):
            label = obj.findtext("name", "")
            if not valid_plate_label(label):
                continue

            box = obj.find("bndbox")
            if box is None:
                continue

            try:
                x1 = float(box.findtext("xmin"))
                y1 = float(box.findtext("ymin"))
                x2 = float(box.findtext("xmax"))
                y2 = float(box.findtext("ymax"))
            except (TypeError, ValueError):
                continue

            x1 = max(0, min(x1, width))
            x2 = max(0, min(x2, width))
            y1 = max(0, min(y1, height))
            y2 = max(0, min(y2, height))

            if x2 <= x1 or y2 <= y1:
                continue

            xc = ((x1 + x2) / 2) / width
            yc = ((y1 + y2) / 2) / height
            bw = (x2 - x1) / width
            bh = (y2 - y1) / height

            boxes.append(
                f"0 {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}"
            )

        if boxes:
            records.append((image_path, boxes))

    except Exception as exc:
        bad.append((xml_path.name, str(exc)))

print("Images found:", len(images))
print("XML files found:", len(xmls))
print("Matched images with valid plate boxes:", len(records))
print("Unmatched XML files:", len(unmatched))
print("Invalid XML records:", len(bad))

if len(records) < 10:
    print("STOP: Too few matched annotations. Check filename matching.")
    print("Unmatched examples:", unmatched[:10])
    raise SystemExit(1)

random.Random(SEED).shuffle(records)

n = len(records)
n_train = int(n * 0.8)
n_val = int(n * 0.1)

splits = {
    "train": records[:n_train],
    "val": records[n_train:n_train + n_val],
    "test": records[n_train + n_val:],
}

# Do not overwrite an existing prepared dataset.
if OUT.exists():
    raise SystemExit(
        f"{OUT} already exists. Rename or back it up before rerunning."
    )

for split, items in splits.items():
    image_dir = OUT / "images" / split
    label_dir = OUT / "labels" / split
    image_dir.mkdir(parents=True)
    label_dir.mkdir(parents=True)

    for i, (src, boxes) in enumerate(items):
        # Unique names avoid collisions between folders.
        name = f"{split}_{i:05d}"
        shutil.copy2(src, image_dir / f"{name}{src.suffix.lower()}")
        (label_dir / f"{name}.txt").write_text(
            "\n".join(boxes) + "\n", encoding="utf-8"
        )

data = {
    "path": str(OUT.resolve()),
    "train": "images/train",
    "val": "images/val",
    "test": "images/test",
    "names": {0: "number_plate"},
}

with (OUT / "data.yaml").open("w", encoding="utf-8") as f:
    yaml.safe_dump(data, f, sort_keys=False)

print("\nDataset created:", OUT)
for split, items in splits.items():
    print(f"{split}: {len(items)} images")
print("Config:", OUT / "data.yaml")
print("Review sample images and boxes before training.")
