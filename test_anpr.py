from pathlib import Path
import cv2
import torch
import easyocr
from ultralytics import YOLO

MODEL_PATH = r"C:\ANPR-Rural-Toll\runs\detect\runs\anpr\plate_detector-2\weights\best.pt"
IMAGE_DIR = Path(r"C:\ANPR-Rural-Toll\google_images")
OUTPUT_PATH = Path(r"C:\ANPR-Rural-Toll\anpr_result.jpg")

print("Loading YOLO model...", flush=True)
model = YOLO(MODEL_PATH)

print("Loading OCR...", flush=True)
reader = easyocr.Reader(["en"], gpu=torch.cuda.is_available())

# Find an image to test
extensions = {".jpg", ".jpeg", ".png", ".bmp"}
images = sorted(
    p for p in IMAGE_DIR.iterdir()
    if p.suffix.lower() in extensions
)

if not images:
    raise FileNotFoundError(f"No images found in {IMAGE_DIR}")

image_path = images[0]
print("Testing image:", image_path.name, flush=True)

image = cv2.imread(str(image_path))
if image is None:
    raise ValueError("Could not read image")

# Detect number plates using the trained GPU model
results = model.predict(
    source=image,
    device=0 if torch.cuda.is_available() else "cpu",
    imgsz=640,
    conf=0.25,
    verbose=False
)

count = 0

for box in results[0].boxes:
    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().tolist())

    h, w = image.shape[:2]
    x1, x2 = max(0, x1), min(w, x2)
    y1, y2 = max(0, y1), min(h, y2)

    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        continue

    # Enlarge and enhance the plate crop
    crop = cv2.resize(
        crop, None, fx=3, fy=3,
        interpolation=cv2.INTER_CUBIC
    )
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.createCLAHE(
        clipLimit=2.0, tileGridSize=(8, 8)
    ).apply(gray)

    ocr_results = reader.readtext(gray, detail=1)
    plate_text = " ".join(
        item[1] for item in ocr_results
        if item[2] >= 0.20
    ).strip()

    if not plate_text:
        plate_text = "TEXT NOT READ"

    confidence = float(box.conf[0].item())
    print(
        f"Plate {count + 1}: {plate_text} "
        f"| Detection confidence: {confidence:.2f}",
        flush=True
    )

    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(
        image, plate_text[:35],
        (x1, max(y1 - 10, 25)),
        cv2.FONT_HERSHEY_SIMPLEX, 0.65,
        (0, 255, 0), 2
    )
    count += 1

cv2.imwrite(str(OUTPUT_PATH), image)

print("\nANPR test completed!", flush=True)
print("Plates detected:", count, flush=True)
print("Output image:", OUTPUT_PATH, flush=True)