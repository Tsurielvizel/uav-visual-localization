"""
Geometric Camera Calibration Pipeline
Author: Tsuriel Vizel

Computes intrinsic camera matrices and distortion coefficients using a curated 
dataset of chessboard images. The output matrices are serialized into .npz files 
for use in the main UAV localization node.
"""

import cv2
import numpy as np
from pathlib import Path

# ==========================================
# Calibration Constants
# ==========================================
SQUARE_SIZE = 25.0  # mm
CHECKERBOARD = (7, 7)

def calibrate_lens(lens_type):
    folder = Path(f"data/{lens_type}_selected")
    image_paths = list(folder.glob("*.jpg")) + list(folder.glob("*.png"))
    
    if not image_paths:
        print(f"[ERROR] No images found in {folder}")
        return

    print(f"\n[*] Calibrating {lens_type.upper()} lens using {len(image_paths)} curated images...")

    objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
    objp *= SQUARE_SIZE

    objpoints = []
    imgpoints = []
    standard_size = None
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FILTER_QUADS

    for img_path in image_paths:
        img = cv2.imread(str(img_path))
        h, w = img.shape[:2]
        
        if standard_size is None:
            standard_size = (w, h) if w < h else (h, w)

        # Handle portrait/landscape consistency
        if (w, h) != standard_size:
            img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, flags)
        
        if ret:
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.001)
            corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            objpoints.append(objp)
            imgpoints.append(corners_refined)

    # Apply constraint only for narrow lens to prevent overfitting
    calib_flags = cv2.CALIB_FIX_K3 + cv2.CALIB_ZERO_TANGENT_DIST if lens_type == "narrow" else 0

    rms, K, dist, _, _ = cv2.calibrateCamera(
        objpoints, imgpoints, standard_size, None, None, flags=calib_flags
    )

    print(f"[SUCCESS] {lens_type.upper()} RMS: {rms:.4f} px")
    
    # Save the output matrices
    output_path = Path(f"output/calibration_{lens_type}.npz")
    np.savez(output_path, K=K, dist=dist, rms=rms, image_size=standard_size)
    print(f"[*] Matrices saved to {output_path}")

if __name__ == "__main__":
    print("========================================")
    print("Multi-Camera Calibration Pipeline")
    print("========================================")
    calibrate_lens("narrow")
    calibrate_lens("wide")
    print("========================================")
    print("Pipeline execution complete.")