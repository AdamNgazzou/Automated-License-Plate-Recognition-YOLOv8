# License Plate Detector & Recognizer 🚗🔍

A robust end-to-end pipeline for automatic vehicle and license plate detection, tracking, and recognition in videos. Built with YOLOv8, custom-trained models, SORT tracking, and OCR, this project enables accurate extraction and export of license plate data from traffic footage.

![Python](https://img.shields.io/badge/Python-3.7%2B-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-1.10%2B-red.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.5%2B-green.svg)
![Ultralytics YOLO](https://img.shields.io/badge/Ultralytics-YOLOv8-blueviolet)
![EasyOCR](https://img.shields.io/badge/EasyOCR-%3E=1.6.2-orange)

---

## Features

- **Vehicle Detection:** Uses YOLOv8 COCO model for robust vehicle localization.
- **License Plate Detection:** Employs a custom YOLOv8 model for precise license plate detection.
- **Vehicle Tracking:** Integrates [SORT](https://github.com/abewley/sort) for multi-object tracking across frames.
- **OCR Recognition:** Utilizes EasyOCR for reading license plate text from detected plates.
- **CSV Export:** Outputs bounding boxes, recognized text, and scores to CSV for further analysis.

---

## Tech Stack

- **Python**
- **YOLOv8** (via [Ultralytics](https://github.com/ultralytics/ultralytics))
- **OpenCV**
- **NumPy**
- **EasyOCR**
- **SORT** ([repo](https://github.com/abewley/sort))

---

## Dataset & Resources

- **Custom License Plate Detector Model:**  Trained on [Roboflow License Plate Recognition Dataset](https://universe.roboflow.com/roboflow-universe-projects/license-plate-recognition-rxg4e/dataset/4)
- **SORT Tracking Module:**  Clone from [https://github.com/abewley/sort](https://github.com/abewley/sort)
- **Sample Video for Testing:**  [Pexels Traffic Flow Video](https://www.pexels.com/video/traffic-flow-in-the-highway-2103099/)

---

## Project Structure

```
main.py                # Main pipeline: detection, tracking, OCR, export
utils.py               # Utility functions (OCR, formatting, helpers)
visualise.py           # Visualization and video output
models/
  license_plate_detector.pt  # Custom-trained YOLOv8 license plate model
  yolov8n.pt                # YOLOv8 COCO model
outputs/
  main.csv             # Output CSV with results
  test.csv             # Additional output
sort/                  # SORT tracking module (clone from repo)
requirements.txt       # Python dependencies
sample.mp4             # Sample input video
out.mp4                # Output visualization video
```

---

## Installation

1. **Clone this repository:**

   ```sh
   git clone github.com/AdamNgazzou/Automated-License-Plate-Recognition-YOLOv8
   cd licence_plate_detection
   ```

2. **Install Python dependencies:**

   ```sh
   pip install -r requirements.txt
   ```

3. **Clone the SORT tracking module:**

   ```sh
   git clone https://github.com/abewley/sort.git
   ```

---

## Usage

Run inference on the sample video:

```sh
python main.py
```

- **Output CSV:** Results are saved in `outputs/main.csv`
- **Output Video:** Visualization saved as `out.mp4`

You can adjust input/output paths in `main.py` and `visualise.py` as needed.

---

## Results

- **CSV Output:**  
  All detected vehicles, license plates, bounding boxes, and recognized text are exported to `outputs/main.csv`.

- **Visualization:**  
  Annotated video with bounding boxes and recognized plates is saved as `out.mp4`.

- **Sample Output:**  
  _(Add screenshots or GIFs here)_

---

## Future Improvements

- Support for multi-language license plates
- Real-time inference and streaming support

---

## Contributing

Contributions, bug reports, and feature requests are welcome! Please open an issue or submit a pull request.

---
