# Automated Number Plate Recognition System for Rural Tolls

## Project Overview
An AI-based Automatic Number Plate Recognition (ANPR) prototype designed to detect vehicle number plates and recognize their text, including in challenging image conditions.

## Technologies Used
- Python
- YOLOv8n for number plate detection
- EasyOCR for text recognition
- OpenCV for image preprocessing
- Streamlit for the web interface
- PyTorch for deep learning

## Features
- Upload vehicle images
- Detect number plates
- Enhance plate crops before OCR
- Display recognized text and detection confidence
- Export results as CSV

## Installation

1. Clone the repository.
2. Create and activate a Python virtual environment.
3. Install dependencies:

   pip install -r requirements.txt

4. Install a compatible PyTorch build for your system if required.
5. Place the trained model at:

   models/best.pt

6. Run the application:

   streamlit run app.py

## Model Evaluation
Preliminary validation results:
- Precision: 99.6%
- Recall: 100%
- mAP@50: 99.5%
- mAP@50-95: 86.1%

These results are from a small validation set of 43 images and 43 annotated instances. They measure plate detection, not OCR accuracy or real-world performance.

## Limitations
OCR results may be incorrect for blurry, damaged, or low-resolution plates. Further testing on diverse real-world images is required.

## Team
- Selvaganapathy K
- Sakthi Gowshick S
- Manoj K
- Vignesh S

## Disclaimer
This is a prototype for research and demonstration purposes.
