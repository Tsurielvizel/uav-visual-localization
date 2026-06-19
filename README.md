# Vision-Based UAV Navigation & Pose Estimation

This repository contains a foundational pipeline for indoor 6DoF vision-based UAV navigation, utilizing geometric camera calibration, natural feature constraints, and ArUco marker tracking.

The system is designed to simulate how autonomous drones navigate in GPS-denied environments using only onboard optical sensors and spatial constraints.

---

## Overview

Accurate intrinsic matrices (`K`) and distortion models are critical prerequisites for reliable spatial pose estimation, for example when using `solvePnP`.

This project handles:

1. **Geometric Calibration**  
   Generating robust intrinsic camera matrices for heterogeneous lenses.

2. **Dynamic Resolution Scaling**  
   Automatically adjusting focal length (`fx`, `fy`) and optical center (`cx`, `cy`) if the live video stream resolution differs from the original calibration resolution.

3. **Pose Estimation & Telemetry**  
   Estimating 3D distances using ArUco markers, synchronized with raw IMU gyroscope telemetry.

4. **Interactive Dashboarding**  
   Generating a fully standalone HTML/JavaScript dashboard to visually evaluate the flight trajectory and compare uncalibrated vs. calibrated pose outputs.

---

## Performance & Accuracy

The calibration pipeline significantly improves distance estimation accuracy by compensating for lens distortion, resolution cropping, and focal discrepancies.

In a physical validation test, the camera was moved along a continuous trajectory from an initial distance of **~1.5 meters** towards an ArUco marker, stopping at a measured ground-truth distance of exactly **40.0 cm**.

* **Raw (Uncalibrated) Estimation:** Exhibited a significant drift, overestimating the final distance (calculating ~52 cm instead of 40 cm). This error highlights the vulnerability of standard pose estimation to uncompensated 16:9 vertical sensor cropping and default focal length assumptions.
* **Calibrated Estimation:** Converged much closer to the true physical distance of 40.0 cm. This successfully demonstrates the necessity and effectiveness of the dynamic matrix scaling and distortion correction algorithms implemented in this pipeline.

---

## Repository Structure

```text
.
├── ArUcoPDF/
│   └── Printable PDF with 10 ArUco markers
│       Dictionary: DICT_4X4_250
│
├── data/
│   ├── Curated calibration images
│   ├── Recorded experiment logs
│   ├── Captured frames
│   └── dashboard.html
│
├── output/
│   └── Serialized calibration matrices (.npz)
│
├── final_experiment.py
│   └── Main localization script with telemetry logging and dynamic scaling
│
└── run_calibration.py
    └── Offline geometric calibration pipeline using chessboard images
```

---

## Usage

### 1. Offline Calibration

Generate camera matrices and distortion coefficients from a curated chessboard image dataset:

```bash
python run_calibration.py
```

The calibration output is saved as serialized `.npz` files inside the `output/` directory.

---

### 2. Live Experiment & Dashboard

Before running the live experiment:

1. Make sure the IP Webcam stream is active.
2. Update the camera stream IP/URL inside `final_experiment.py`.
3. Make sure the calibration `.npz` file exists in the expected location.

Then run:

```bash
python final_experiment.py
```

The script dynamically scales the intrinsic matrix to match the live stream resolution and records a short trajectory session.

Upon completion, an interactive dashboard is generated:

```text
data/dashboard.html
```

Open this file in a browser to review the synchronized frames, distance estimates, telemetry, and pose visualization.

---

## Dashboard Features

The generated dashboard includes:

- Interactive distance-vs-time graph
- Comparison between raw and calibrated frame outputs
- Synchronized ArUco pose visualization
- IMU gyroscope telemetry display
- Clickable chart points for frame-by-frame inspection
- Fullscreen image preview by clicking on frames
- Clear legend explaining the OpenCV pose axes

---

## Pose Axes Convention

The calibrated frame displays OpenCV pose axes:

- **Red X-axis:** Right / Left direction
- **Green Y-axis:** Down / Up direction
- **Blue Z-axis:** Depth / Distance from the camera

---

## Notes

- The HTML dashboard is fully standalone and does not require an external server.
- The image paths stored in the CSV and dashboard should remain relative to the `data/` directory structure.
- If the live stream resolution differs from the calibration image resolution, the script rescales the intrinsic camera matrix automatically.
- Calibration accuracy depends strongly on the quality and diversity of the chessboard images.
- For wide-angle lenses, distortion correction is especially important because raw pose estimates can be significantly biased.

---

## Requirements

Typical dependencies include:

```bash
pip install opencv-python numpy
```

If additional plotting, data processing, or dashboard utilities are added later, install the relevant packages as needed.

---

## Project Goal

The goal of this project is to demonstrate a practical computer-vision pipeline for UAV-style indoor localization, combining:

- Camera calibration
- ArUco marker detection
- Pose estimation
- Dynamic intrinsic matrix scaling
- IMU telemetry logging
- Visual dashboard-based analysis

This provides a compact experimental framework for evaluating vision-based navigation concepts in GPS-denied environments.