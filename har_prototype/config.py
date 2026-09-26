"""
Configuration settings for AI Human Activity Recognition (HAR) On-Board Assistant.
All shared constants, thresholds, camera settings, network ports, and file paths are centralized here.
"""

from pathlib import Path
import numpy as np

# Project root directory
BASE_DIR = Path(__file__).resolve().parent

# --- Protocol & Action Recognition ---
# Predefined sequence of experiment protocol steps
EXPERIMENT_STEPS = [
    "pick_red_box",
    "open_red_box",
    "pick_yellow_box",
    "open_yellow_box",
    "place_back"
]

# Pixel distance threshold between hand position (wrist) and object center to consider an interaction candidate
PROXIMITY_THRESHOLD = 50

# Consecutive frames an interaction candidate must persist to confirm the action (eliminates transient jitter/flicker)
CONFIRMATION_FRAMES = 5

# --- Camera & Processing Settings ---
CAMERA_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
TARGET_FPS = 20

# Minimum contour area (in pixels) for colored box detection to filter out ambient noise
MIN_BOX_CONTOUR_AREA = 800

# --- Color Detection Ranges (HSV) ---
# Red wraps around hue 0 and 180 in OpenCV (H: 0-179, S: 0-255, V: 0-255)
# Range 1: Lower red [0..10]
RED_LOWER_1 = np.array([0, 100, 70], dtype=np.uint8)
RED_UPPER_1 = np.array([10, 255, 255], dtype=np.uint8)

# Range 2: Upper red [170..180]
RED_LOWER_2 = np.array([170, 100, 70], dtype=np.uint8)
RED_UPPER_2 = np.array([180, 255, 255], dtype=np.uint8)

# Yellow range [20..35]
YELLOW_LOWER = np.array([20, 100, 100], dtype=np.uint8)
YELLOW_UPPER = np.array([35, 255, 255], dtype=np.uint8)

# --- Network & Streaming ---
# IP & Port for live video MJPEG streaming
STREAM_IP = "127.0.0.1"
STREAM_PORT = 5000

# Host & Port for real-time WebSocket state telemetry pushed to the dashboard
WS_HOST = "127.0.0.1"
WS_PORT = 8765

# --- Storage & Logging Paths ---
LOG_DIR = BASE_DIR / "logs"
OUTPUT_DIR = BASE_DIR / "output"
DATASET_DIR = BASE_DIR / "dataset"

# Ensure runtime directories exist
LOG_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DATASET_DIR.mkdir(parents=True, exist_ok=True)

# Structured JSON audit log and human-readable plain-text summary
LOG_FILE_PATH = str(LOG_DIR / "experiment_log.json")
SUMMARY_FILE_PATH = str(LOG_DIR / "summary.txt")

# Local recording file path (XVID AVI container)
VIDEO_OUTPUT_PATH = str(OUTPUT_DIR / "recorded_video.avi")
