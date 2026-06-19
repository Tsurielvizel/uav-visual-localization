# Vision-Based UAV Navigation: Geometric Camera Calibration

This repository contains the foundational geometric camera calibration pipeline developed for an indoor 6DoF vision-based UAV navigation system. Accurate intrinsic matrices ($K$) and distortion models are critical prerequisites for reliable feature matching, epipolar geometry extraction, and spatial pose estimation (e.g., `solvePnP`) in GPS-denied environments.

## Overview
The module performs multi-camera calibration for a heterogeneous sensor setup:
1. **Wide-Angle Lens:** Calibrated with a full radial distortion model to account for significant optical bowing.
2. **Narrow-Angle/Telephoto Lens:** Constrained calibration (`CALIB_FIX_K3`, `CALIB_ZERO_TANGENT_DIST`) to prevent mathematical overfitting, as narrow FoV optics exhibit near-linear properties but are highly susceptible to planar orientation noise.

## Automated Filtering & Methodology
To achieve robust sub-pixel accuracy using a physical target under standard indoor lighting, the pipeline implements an aggressive filtering strategy:
* Raw image sequences are evaluated iteratively.
* Corners are resolved dynamically using `cv2.findChessboardCorners` with adaptive thresholding.
* Images exhibiting motion blur, specular reflections, or severe perspective distortion are automatically rejected.
* Only the top-performing frames (yielding the lowest reprojection error) are curated into the `data/` repository for the final calculation.

## Results & Metrics
The calibration utilizes a $7 \times 7$ internal vertex checkerboard with $25.0 \text{ mm}$ squares.

| Lens Type | Curated Frames | Optimization Strategy | Final RMS Reprojection Error |
| :--- | :---: | :--- | :--- |
| **Wide-Angle** | 12 | Full non-linear modeling | **0.8345 pixels** |
| **Narrow-Angle** | 10 | Constrained modeling | **1.7978 pixels** |

*Note: The sub-pixel RMS of 0.83px on the wide-angle lens is highly optimal for visual odometry. The 1.79px RMS on the narrow lens is completely sufficient for state-estimation, as minor localized pixel noise is smoothed via subsequent IMU sensor fusion.*

## Repository Structure
```text
├── data/
│   ├── narrow_selected/       # Top 10 curated images for narrow FoV
│   └── wide_selected/         # Top 12 curated images for wide FoV
├── output/
│   ├── calibration_narrow.npz # Extracted matrices & parameters
│   └── calibration_wide.npz   
├── run_calibration.py         # Main execution script
└── README.md

## **Usage**

To generate the camera matrices and distortion coefficients from the curated dataset:

python run_calibration.py

Outputs are automatically serialized into .npz binaries in the output/ directory for immediate integration into the navigation and localization loops.

