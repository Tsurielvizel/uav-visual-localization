"""
UAV Visual-Inertial Localization: Dynamic Scaling & Interactive Dashboard
Author: Tsuriel Vizel

This script runs the live visual localization pipeline. It connects to an IP Webcam,
dynamically scales the intrinsic camera matrix to match the stream resolution, 
tracks an ArUco marker for 6DoF pose estimation, and logs synchronized IMU telemetry.
Outputs are compiled into an interactive HTML dashboard.
"""

import cv2
import cv2.aruco as aruco
import urllib.request
import json
import time
import threading
import numpy as np
import csv
import ssl
from pathlib import Path

# ==============================================================================
# CONFIGURATION
# ==============================================================================
URL_BASE = "http://10.100.102.21:8080"
VIDEO_URL = f"{URL_BASE}/video"
SENSOR_URL = f"{URL_BASE}/sensors.json"

DELAY_BEFORE_START = 10.0
RECORDING_DURATION = 12.0
SAMPLE_INTERVAL = 0.5

CALIBRATION_FILE = "output/calibration_narrow.npz"
ARUCO_DICT = aruco.DICT_4X4_250
MARKER_SIZE = 0.1  # 100mm

latest_gyro_data = [0.0, 0.0, 0.0]
running = True

# ==============================================================================
# THREADS & MATH
# ==============================================================================
def sensor_thread_worker():
    global latest_gyro_data, running
    while running:
        try:
            with urllib.request.urlopen(SENSOR_URL, timeout=1.0) as url:
                data = json.loads(url.read().decode())
                if 'gyro' in data and 'data' in data['gyro'] and len(data['gyro']['data']) > 0:
                    latest_gyro_data = data['gyro']['data'][-1][1]
        except Exception:
            pass
        time.sleep(0.05)

def load_calibration():
    try:
        data = np.load(CALIBRATION_FILE)
        K = data['K']
        dist = data['dist']
        img_size = data['image_size'] if 'image_size' in data.files else None
        return K, dist, img_size
    except Exception as e:
        print(f"[ERROR] Calibration file failed to load: {e}")
        return None, None, None

def estimate_pose(corners, camera_matrix, dist_coeffs):
    marker_points = np.array([
        [-MARKER_SIZE/2,  MARKER_SIZE/2, 0],
        [ MARKER_SIZE/2,  MARKER_SIZE/2, 0],
        [ MARKER_SIZE/2, -MARKER_SIZE/2, 0],
        [-MARKER_SIZE/2, -MARKER_SIZE/2, 0]
    ], dtype=np.float32)
    success, rvec, tvec = cv2.solvePnP(marker_points, corners[0], camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_IPPE_SQUARE)
    return success, rvec, tvec

# ==============================================================================
# MAIN
# ==============================================================================
def main():
    global running
    orig_camera_matrix, dist_coeffs, orig_img_size = load_calibration()
    if orig_camera_matrix is None:
        return

    # Create a UNIQUE folder for this run to prevent file overriding
    run_id = time.strftime("%H%M%S")
    output_dir = Path(f"data/final_experiment_run_{run_id}")
    frames_dir = output_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    
    dictionary = aruco.getPredefinedDictionary(ARUCO_DICT)
    parameters = aruco.DetectorParameters()
    detector = aruco.ArucoDetector(dictionary, parameters)
    
    log_data = []

    print(f"[*] Connecting to Camera Stream: {VIDEO_URL}...")
    cap = cv2.VideoCapture(VIDEO_URL)
    if not cap.isOpened():
        print("[ERROR] Camera stream not found.")
        return

    sensor_thread = threading.Thread(target=sensor_thread_worker, daemon=True)
    sensor_thread.start()

    print("\n" + "="*50)
    print(f"⏳ INITIALIZING... Please wait {DELAY_BEFORE_START} seconds.")
    print("Position phone far from the chair. Prepare to move slowly.")
    print("="*50 + "\n")

    start_wait = time.time()
    while time.time() - start_wait < DELAY_BEFORE_START:
        time_left = DELAY_BEFORE_START - (time.time() - start_wait)
        print(f"\rStarting in: {time_left:.1f}s   ", end="")
        cap.read()
        cv2.waitKey(1)
        time.sleep(0.1)

    print(f"\n\n[🚀] RECORDING TO DIRECTORY: {output_dir.name}")

    # ==========================================================
    # DYNAMIC CALIBRATION SCALING (The Fix!)
    # ==========================================================
    ret, initial_frame = cap.read()
    stream_h, stream_w = initial_frame.shape[:2]

    # Find the original width/height from the calibration process
    if orig_img_size is not None:
        calib_w, calib_h = max(orig_img_size), min(orig_img_size)
    else:
        # Fallback: estimate from principal point if image_size wasn't saved
        calib_w, calib_h = orig_camera_matrix[0, 2] * 2, orig_camera_matrix[1, 2] * 2

    # Scale the focal length and optical center to match the live video stream
    scale_x = stream_w / float(calib_w)
    scale_y = stream_h / float(calib_h)
    
    camera_matrix = orig_camera_matrix.copy()
    camera_matrix[0, 0] *= scale_x
    camera_matrix[1, 1] *= scale_y
    camera_matrix[0, 2] *= scale_x
    camera_matrix[1, 2] *= scale_y

    print(f"[*] Scaled Camera Matrix from {calib_w}x{calib_h} to Stream Size {stream_w}x{stream_h}")

    new_camera_matrix, _ = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (stream_w,stream_h), 1, (stream_w,stream_h))
    map1, map2 = cv2.initUndistortRectifyMap(camera_matrix, dist_coeffs, None, new_camera_matrix, (stream_w,stream_h), cv2.CV_16SC2)

    raw_camera_matrix = np.array([[stream_w, 0, stream_w/2], [0, stream_w, stream_h/2], [0, 0, 1]], dtype=np.float32)
    raw_dist_coeffs = np.zeros((4,1), dtype=np.float32)

    # ==========================================================

    start_record = time.time()
    next_capture_time = 0.0
    sample_index = 0

    while True:
        current_time = time.time()
        elapsed_time = current_time - start_record
        if elapsed_time > RECORDING_DURATION:
            break

        ret, frame = cap.read()
        if not ret:
            break

        if elapsed_time >= next_capture_time:
            time_str = f"{next_capture_time:.1f}s"
            file_suffix = f"{next_capture_time:.1f}s".replace('.', '_')
            gx, gy, gz = latest_gyro_data
            
            raw_distance = None
            calib_distance = None
            marker_id = -1

            raw_img = frame.copy()
            gray_raw = cv2.cvtColor(raw_img, cv2.COLOR_BGR2GRAY)
            corners_raw, ids_raw, _ = detector.detectMarkers(gray_raw)
            
            if ids_raw is not None:
                marker_id = int(ids_raw[0][0])
                aruco.drawDetectedMarkers(raw_img, corners_raw, ids_raw)
                success_raw, _, tvec_raw = estimate_pose(corners_raw, raw_camera_matrix, raw_dist_coeffs)
                if success_raw:
                    raw_distance = float(np.linalg.norm(tvec_raw))
            
            dist_text = f"Dist: {raw_distance:.2f}m" if raw_distance else "Dist: NOT FOUND"
            cv2.putText(raw_img, f"RAW FRAME | Time: {time_str}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(raw_img, dist_text, (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            raw_filename = f"frame_{file_suffix}_raw.jpg"
            cv2.imwrite(str(frames_dir / raw_filename), raw_img)

            calib_img = cv2.remap(frame, map1, map2, interpolation=cv2.INTER_LINEAR)
            gray_calib = cv2.cvtColor(calib_img, cv2.COLOR_BGR2GRAY)
            corners_calib, ids_calib, _ = detector.detectMarkers(gray_calib)
            
            if ids_calib is not None:
                aruco.drawDetectedMarkers(calib_img, corners_calib, ids_calib)
                success_calib, rvec_calib, tvec_calib = estimate_pose(corners_calib, new_camera_matrix, np.zeros((4,1)))
                if success_calib:
                    calib_distance = float(np.linalg.norm(tvec_calib))
                    cv2.drawFrameAxes(calib_img, new_camera_matrix, np.zeros((4,1)), rvec_calib, tvec_calib, 0.05)
            
            c_dist_text = f"Dist: {calib_distance:.2f}m" if calib_distance else "Dist: NOT FOUND"
            cv2.putText(calib_img, "Calibrated Pose Estimate", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(calib_img, f"Time: {time_str} | {c_dist_text}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            calib_filename = f"frame_{file_suffix}_calib.jpg"
            cv2.imwrite(str(frames_dir / calib_filename), calib_img)

            log_data.append({
                'sample_index': sample_index,
                'time_sec': round(next_capture_time, 1),
                'marker_id': marker_id,
                'raw_distance': round(raw_distance, 3) if raw_distance else "null",
                'calib_distance': round(calib_distance, 3) if calib_distance else "null",
                'gyro_x': round(gx, 4),
                'gyro_y': round(gy, 4),
                'gyro_z': round(gz, 4),
                'raw_path': f"frames/{raw_filename}",
                'calib_path': f"frames/{calib_filename}"
            })

            print(f"[INTERVAL] Saved {time_str} | Calib Dist: {calib_distance if calib_distance else 'N/A'}")
            sample_index += 1
            next_capture_time += SAMPLE_INTERVAL

        cv2.imshow("Experiment Active Pipeline...", frame)
        cv2.waitKey(1)

    running = False
    cap.release()
    cv2.destroyAllWindows()
    print("\n[✅] Trajectory data capture finished. Compiling logs...")

    
    # ==========================================================
    # HTML DASHBOARD GENERATION
    # ==========================================================
    csv_path = output_dir / "telemetry_log_v3.csv"

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "sample_index",
                "time_sec",
                "marker_id",
                "raw_distance",
                "calib_distance",
                "gyro_x",
                "gyro_y",
                "gyro_z",
                "raw_path",
                "calib_path",
            ],
        )
        writer.writeheader()

        for row in log_data:
            writer.writerow(row)

    html_path = output_dir / "dashboard.html"
    js_data_array = json.dumps(log_data)

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>UAV Visual-Inertial Localization Dashboard</title>

    <style>
        body {{
            font-family: 'Segoe UI', sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f4f6f9;
            color: #333;
        }}

        header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding-bottom: 20px;
            border-bottom: 2px solid #ddd;
            margin-bottom: 20px;
        }}

        h1 {{
            margin: 0;
            color: #1e293b;
            font-size: 24px;
        }}

        .subtitle {{
            color: #64748b;
            font-size: 14px;
        }}

        .layout {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
        }}

        .panel {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        }}

        .image-container {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 10px;
            margin-top: 15px;
        }}

        .img-box {{
            text-align: center;
        }}

        img {{
            width: 100%;
            border-radius: 6px;
            border: 1px solid #e2e8f0;
            object-fit: cover;
            cursor: zoom-in;
            transition: transform 0.2s;
        }}

        img:hover {{
            transform: scale(1.02);
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        }}

        .img-label {{
            font-weight: bold;
            margin-bottom: 5px;
            font-size: 14px;
            color: #475569;
        }}

        .axes-legend {{
            font-size: 12px;
            color: #475569;
            background: #f8fafc;
            padding: 8px;
            border-radius: 6px;
            margin-top: 10px;
            text-align: center;
            border: 1px solid #e2e8f0;
        }}

        .meta-grid {{
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 10px;
            margin-top: 15px;
            background: #f8fafc;
            padding: 12px;
            border-radius: 6px;
        }}

        .meta-item {{
            text-align: center;
        }}

        .meta-val {{
            font-size: 16px;
            font-weight: bold;
            color: #0f172a;
        }}

        .meta-lbl {{
            font-size: 11px;
            color: #64748b;
            text-transform: uppercase;
        }}

        svg {{
            width: 100%;
            height: 320px;
            background: #fff;
        }}

        .axis {{
            stroke: #cbd5e1;
            stroke-width: 1;
        }}

        .axis-text {{
            font-size: 10px;
            fill: #64748b;
        }}

        .line-calib {{
            fill: none;
            stroke: #10b981;
            stroke-width: 3;
        }}

        .line-raw {{
            fill: none;
            stroke: #ef4444;
            stroke-width: 2;
            stroke-dasharray: 4;
        }}

        .dot {{
            cursor: pointer;
            transition: r 0.2s;
        }}

        .dot:hover {{
            r: 8;
        }}

        .dot-active {{
            r: 9 !important;
            stroke: #0f172a;
            stroke-width: 2;
        }}

        .btn {{
            background: #2563eb;
            color: white;
            border: none;
            padding: 8px 16px;
            border-radius: 4px;
            cursor: pointer;
            font-weight: bold;
        }}

        .btn:hover {{
            background: #1d4ed8;
        }}

        .legend {{
            display: flex;
            gap: 15px;
            justify-content: center;
            font-size: 12px;
            margin-top: 10px;
        }}

        .legend-item {{
            display: flex;
            align-items: center;
            gap: 5px;
        }}

        /* Fullscreen Modal Styles */
        .modal {{
            display: none;
            position: fixed;
            z-index: 1000;
            left: 0;
            top: 0;
            width: 100%;
            height: 100%;
            background-color: rgba(0, 0, 0, 0.85);
            align-items: center;
            justify-content: center;
        }}

        .modal img {{
            max-width: 95%;
            max-height: 95%;
            border-radius: 8px;
            border: 2px solid white;
            object-fit: contain;
            cursor: zoom-out;
        }}

        .close-btn {{
            position: absolute;
            top: 20px;
            right: 40px;
            color: white;
            font-size: 40px;
            cursor: pointer;
            font-weight: bold;
        }}
    </style>
</head>

<body>

    <div id="imageModal" class="modal" onclick="closeModal()">
        <span class="close-btn">&times;</span>
        <img id="modalImg" src="" alt="Fullscreen preview">
    </div>

    <header>
        <div>
            <h1>UAV Visual-Inertial Localization Dashboard</h1>
            <div class="subtitle">Dynamic Camera Scaling & IMU Synchronization</div>
        </div>

        <button class="btn" id="playBtn" onclick="toggleSlideshow()">▶ Play Flight</button>
    </header>

    <div class="layout">

        <div class="panel">
            <h3>Estimated Distance vs Time (Click dots)</h3>

            <div id="chartWrapper">
                <svg id="metricChart" viewBox="0 0 600 320">
                    <line x1="50" y1="20" x2="550" y2="20" stroke="#f1f5f9" />
                    <line x1="50" y1="90" x2="550" y2="90" stroke="#f1f5f9" />
                    <line x1="50" y1="160" x2="550" y2="160" stroke="#f1f5f9" />
                    <line x1="50" y1="230" x2="550" y2="230" stroke="#f1f5f9" />
                    <line x1="50" y1="270" x2="550" y2="270" stroke="#cbd5e1" stroke-width="1.5" />

                    <line x1="50" y1="20" x2="50" y2="270" class="axis" />

                    <text x="25" y="25" class="axis-text">4.0m</text>
                    <text x="25" y="108" class="axis-text">2.6m</text>
                    <text x="25" y="193" class="axis-text">1.3m</text>
                    <text x="25" y="274" class="axis-text">0.0m</text>

                    <text x="50" y="290" class="axis-text">0s</text>
                    <text x="133" y="290" class="axis-text">2s</text>
                    <text x="216" y="290" class="axis-text">4s</text>
                    <text x="300" y="290" class="axis-text">6s</text>
                    <text x="383" y="290" class="axis-text">8s</text>
                    <text x="466" y="290" class="axis-text">10s</text>
                    <text x="550" y="290" class="axis-text">12s</text>

                    <g id="svgLines"></g>
                    <g id="svgDots"></g>
                </svg>
            </div>

            <div class="legend">
                <div class="legend-item">
                    <span style="width:12px;height:3px;background:#10b981;display:inline-block;"></span>
                    Calibrated Estimate
                </div>

                <div class="legend-item">
                    <span style="width:12px;height:3px;border-top:2px dashed #ef4444;display:inline-block;"></span>
                    Uncalibrated (Raw)
                </div>
            </div>
        </div>

        <div class="panel">
            <h3>Synchronized Frame Comparison (<span id="currentTimeLbl">0.0</span>s)</h3>

            <div class="image-container">
                <div class="img-box">
                    <div class="img-label">Raw (Generic Matrix)</div>
                    <img id="rawView" src="" alt="Raw frame" onclick="openModal(this.src)" title="Click to enlarge">
                </div>

                <div class="img-box">
                    <div class="img-label">Calibrated (Scaled Matrix)</div>
                    <img id="calibView" src="" alt="Calibrated frame" onclick="openModal(this.src)" title="Click to enlarge">
                </div>
            </div>

            <div class="axes-legend">
                <strong>Pose Axes:</strong>
                <span style="color:#ef4444; font-weight:bold;">X (Red)</span> = Right/Left |
                <span style="color:#10b981; font-weight:bold;">Y (Green)</span> = Down/Up |
                <span style="color:#3b82f6; font-weight:bold;">Z (Blue)</span> = Depth/Distance
            </div>

            <div class="meta-grid">
                <div class="meta-item">
                    <div class="meta-val" id="valCalibDist">-</div>
                    <div class="meta-lbl">Calib Dist (m)</div>
                </div>

                <div class="meta-item">
                    <div class="meta-val" id="valRawDist">-</div>
                    <div class="meta-lbl">Raw Dist (m)</div>
                </div>

                <div class="meta-item">
                    <div class="meta-val" id="valGyroY">-</div>
                    <div class="meta-lbl">Gyro Y (Yaw)</div>
                </div>

                <div class="meta-item">
                    <div class="meta-val" id="valGyroZ">-</div>
                    <div class="meta-lbl">Gyro Z (Roll)</div>
                </div>
            </div>
        </div>

    </div>

    <script>
        const dataset = {js_data_array};

        let activeIdx = 0;
        let slideshowInterval = null;

        function openModal(src) {{
            const modal = document.getElementById("imageModal");
            const modalImg = document.getElementById("modalImg");

            modal.style.display = "flex";
            modalImg.src = src;
        }}

        function closeModal() {{
            document.getElementById("imageModal").style.display = "none";
        }}

        function getX(time) {{
            return 50 + (time / 12.0) * 500;
        }}

        function getY(dist) {{
            if (dist === null || dist === undefined || isNaN(dist)) {{
                return null;
            }}

            return 270 - (dist / 4.0) * 250;
        }}

        function formatDistance(value) {{
            if (value === null || value === undefined || isNaN(value)) {{
                return "N/A";
            }}

            return value.toFixed(2) + "m";
        }}

        function formatNumber(value) {{
            if (value === null || value === undefined || isNaN(value)) {{
                return "N/A";
            }}

            return value.toFixed(2);
        }}

        function initChart() {{
            const linesG = document.getElementById("svgLines");
            const dotsG = document.getElementById("svgDots");

            let pathCalib = "";
            let pathRaw = "";
            let dotsHTML = "";

            dataset.forEach((pt, i) => {{
                const cx = getX(pt.time_sec);
                const cyCalib = getY(pt.calib_distance);
                const cyRaw = getY(pt.raw_distance);

                if (cyCalib !== null) {{
                    pathCalib += (pathCalib === "" ? "M" : "L") + ` ${{cx}} ${{cyCalib}}`;
                }}

                if (cyRaw !== null) {{
                    pathRaw += (pathRaw === "" ? "M" : "L") + ` ${{cx}} ${{cyRaw}}`;
                }}

                if (cyCalib !== null) {{
                    dotsHTML += `<circle cx="${{cx}}" cy="${{cyCalib}}" r="5" fill="#10b981" class="dot" id="dot-${{i}}" onclick="selectFrame(${{i}})" />`;
                }}
            }});

            linesG.innerHTML = `
                <path d="${{pathCalib}}" class="line-calib" />
                <path d="${{pathRaw}}" class="line-raw" />
            `;

            dotsG.innerHTML = dotsHTML;
        }}

        function selectFrame(idx) {{
            document.querySelectorAll(".dot").forEach(el => el.classList.remove("dot-active"));

            activeIdx = idx;
            const pt = dataset[idx];

            const targetDot = document.getElementById(`dot-${{idx}}`);
            if (targetDot) {{
                targetDot.classList.add("dot-active");
            }}

            document.getElementById("currentTimeLbl").innerText = formatNumber(pt.time_sec);

            document.getElementById("rawView").src = pt.raw_path || "";
            document.getElementById("calibView").src = pt.calib_path || "";

            document.getElementById("valCalibDist").innerText = formatDistance(pt.calib_distance);
            document.getElementById("valRawDist").innerText = formatDistance(pt.raw_distance);
            document.getElementById("valGyroY").innerText = formatNumber(pt.gyro_y);
            document.getElementById("valGyroZ").innerText = formatNumber(pt.gyro_z);
        }}

        function toggleSlideshow() {{
            const btn = document.getElementById("playBtn");

            if (slideshowInterval) {{
                clearInterval(slideshowInterval);
                slideshowInterval = null;
                btn.innerText = "▶ Play Flight";
            }} else {{
                btn.innerText = "⏸ Pause";

                slideshowInterval = setInterval(() => {{
                    const nextIdx = (activeIdx + 1) % dataset.length;
                    selectFrame(nextIdx);
                }}, 500);
            }}
        }}

        if (dataset.length > 0) {{
            initChart();
            selectFrame(0);
        }}
    </script>

</body>
</html>
"""

    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"\n[🏆] DASHBOARD READY! Open {{html_path}} to view.")


if __name__ == "__main__":
    main()