
import torch
from ultralytics import YOLO

print("CUDA Available:", torch.cuda.is_available())

if not torch.cuda.is_available():
    raise RuntimeError("CUDA GPU is not available!")

print("GPU:", torch.cuda.get_device_name(0))

# Create YOLO model
model = YOLO("yolov8n.pt")

# Run inference on GPU (device 0)
results = model.predict(
    source="https://ultralytics.com/images/bus.jpg",
    device=0,
    imgsz=640,
    verbose=False
)

print("YOLO GPU inference successful!")
print("Detections:", len(results[0].boxes))
