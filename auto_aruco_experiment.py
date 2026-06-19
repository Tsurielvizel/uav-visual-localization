"""
UAV Visual Localization - Automated ArUco Tracking & Telemetry Logger

This script captures synchronized video frames and IMU (Gyroscope) telemetry
from an Android device running an IP Webcam server over WiFi. It performs 
automated ArUco marker detection on both raw and camera-calibrated (undistorted) 
frames, saving the visual results and generating a consolidated experiment report.

Author: Tsuriel Vizel
Institution: Technion - Israel Institute of Technology
"""

import cv2
import cv2.aruco as aruco
import urllib.request
import json
import time
import threading
import numpy as np
from pathlib import Path

# ==============================================================================
# CONFIGURATION & CONSTANTS
# ==============================================================================
# Network settings (Update the IP address to match your Android device)
URL_BASE = "http://10.100.102.17:8080"
VIDEO_URL = f"{URL_BASE}/video"
SENSOR_URL = f"{URL_BASE}/sensors.json"

# Experiment timing configurations
DELAY_BEFORE_START = 10.0  # Seconds to wait before starting the recording
CAPTURE_INTERVAL = 2.0     # Seconds between each capture interval
TOTAL_CAPTURES = 4         # Total number of frames to capture

# Calibration and ArUco marker settings
CALIBRATION_FILE = "output/calibration_wide.npz"
ARUCO_DICT = aruco.DICT_4X4_250
MARKER_SIZE = 0.1          # Marker size in meters (100mm = 0.1m)

# Global variables for cross-thread synchronization
latest_gyro_data = None
running = True


# ==============================================================================
# TELEMETRY BACKGROUND THREAD
# ==============================================================================
def sensor_thread_worker():
    """
    Background thread to continuously fetch gyroscope telemetry via HTTP JSON.
    Runs asynchronously to prevent blocking the high-speed OpenCV video loop.
    """
    global latest_gyro_data, running
    print("[*] Sensor telemetry background thread started.")
    
    while running:
        try:
            with urllib.request.urlopen(SENSOR_URL, timeout=1.0) as url:
                data = json.loads(url.read().decode())
                
                # Verify the expected JSON structure exists before extracting
                if 'gyro' in data and 'data' in data['gyro'] and len(data['gyro']['data']) > 0:
                    # Extract the latest [gx, gy, gz] reading
                    latest_gyro_data = data['gyro']['data'][-1][1]
        except Exception:
            # Silent catch to maintain thread stability during minor network drops
            pass
        
        # Poll at ~20Hz to avoid overloading the mobile server
        time.sleep(0.05) 


# ==============================================================================
# CORE FUNCTIONS
# ==============================================================================
def load_calibration():
    """
    Loads the camera intrinsic parameters and distortion coefficients.
    Returns None if the file is missing or corrupted.
    """
    try:
        data = np.load(CALIBRATION_FILE)
        return data['K'], data['dist']
    except Exception as e:
        print(f"[ERROR] Could not load calibration file: {e}")
        return None, None

def detect_aruco(image):
    """
    Detects ArUco markers in a given BGR image.
    Returns the detected corners and marker IDs.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    dictionary = aruco.getPredefinedDictionary(ARUCO_DICT)
    parameters = aruco.DetectorParameters()
    detector = aruco.ArucoDetector(dictionary, parameters)
    
    corners, ids, rejected = detector.detectMarkers(gray)
    return corners, ids


# ==============================================================================
# MAIN EXPERIMENT PIPELINE
# ==============================================================================
def main():
    global running
    
    # 1. Initialize Calibration Data
    camera_matrix, dist_coeffs = load_calibration()
    has_calib = camera_matrix is not None and dist_coeffs is not None

    # 2. Setup Output Directory and Report File
    output_dir = Path("data/aruco_experiment")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    report_file = open(output_dir / "experiment_report.txt", "w")
    report_file.write("=== UAV ArUco Tracking Experiment Report ===\n")
    report_file.write(f"Settings: Dict=4x4, MarkerSize={MARKER_SIZE}m, Calibrated={has_calib}\n\n")

    # 3. Initialize Video Stream
    print(f"[*] Connecting to Camera Stream: {VIDEO_URL}...")
    cap = cv2.VideoCapture(VIDEO_URL)
    if not cap.isOpened():
        print("[ERROR] Could not connect to the camera stream. Check WiFi/IP.")
        return

    # 4. Start Telemetry Thread
    sensor_thread = threading.Thread(target=sensor_thread_worker, daemon=True)
    sensor_thread.start()

    # 5. Initialization Delay (Countdown)
    print("\n" + "="*50)
    print(f"⏳ INITIALIZING... Please wait {DELAY_BEFORE_START} seconds.")
    print("Move the phone into position pointing at the ArUco marker.")
    print("="*50 + "\n")

    start_wait = time.time()
    while time.time() - start_wait < DELAY_BEFORE_START:
        time_left = DELAY_BEFORE_START - (time.time() - start_wait)
        print(f"\rStarting in: {time_left:.1f}s", end="")
        cap.read() # Keep buffer clear
        cv2.waitKey(1)
        time.sleep(0.1)

    print("\n\n[🚀] EXPERIMENT STARTED! Recording...")

    # 6. Main Capture Loop
    captures_taken = 0
    last_capture_time = time.time() - CAPTURE_INTERVAL 
    
    while captures_taken < TOTAL_CAPTURES:
        ret, frame = cap.read()
        if not ret:
            print("[WARNING] Frame dropped.")
            break

        current_time = time.time()

        if current_time - last_capture_time >= CAPTURE_INTERVAL:
            timestamp = time.strftime("%H%M%S")
            base_name = f"capture_{captures_taken+1}_{timestamp}"
            
            # Instantly grab latest async telemetry data
            gyro_data = latest_gyro_data
            raw_image = frame.copy()
            
            # --- Save Raw Image ---
            cv2.imwrite(str(output_dir / f"{base_name}_raw.jpg"), raw_image)

            # --- Detect ArUco (Raw) ---
            _, ids_raw = detect_aruco(raw_image)
            raw_found = ids_raw is not None

            # --- Detect ArUco (Calibrated/Undistorted) ---
            calib_found = False
            if has_calib:
                h, w = raw_image.shape[:2]
                new_camera_matrix, _ = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w,h), 1, (w,h))
                undistorted_image = cv2.undistort(raw_image, camera_matrix, dist_coeffs, None, new_camera_matrix)
                
                cv2.imwrite(str(output_dir / f"{base_name}_calib.jpg"), undistorted_image)

                _, ids_calib = detect_aruco(undistorted_image)
                calib_found = ids_calib is not None

            # --- Log Data to Report ---
            report_file.write(f"--- Capture {captures_taken+1}/{TOTAL_CAPTURES} ({timestamp}) ---\n")
            if gyro_data:
                report_file.write(f"Gyro (X,Y,Z): {gyro_data[0]:.4f}, {gyro_data[1]:.4f}, {gyro_data[2]:.4f}\n")
            else:
                report_file.write("Gyro: No sync data in memory.\n")
            
            report_file.write(f"Raw Detection: {'FOUND' if raw_found else 'NOT FOUND'}\n")
            if has_calib:
                report_file.write(f"Calib Detection: {'FOUND' if calib_found else 'NOT FOUND'}\n")
            report_file.write("\n")

            # Console feedback
            print(f"[SNAP {captures_taken+1}/{TOTAL_CAPTURES}] Saved! Aruco Raw: {'✅' if raw_found else '❌'} | Calib: {'✅' if calib_found else '❌'} | Gyro: {'✅' if gyro_data else '❌'}")
            
            captures_taken += 1
            last_capture_time = current_time

        # Real-time display
        cv2.imshow("UAV Visual Localization - Running", frame)
        cv2.waitKey(1)

    # 7. Cleanup & Shutdown
    running = False # Signal the background thread to stop
    print(f"\n[✅] Experiment Completed! Results saved in: {output_dir}/")
    
    report_file.close()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()