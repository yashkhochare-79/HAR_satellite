# 🚀 AI Human Activity Recognition (HAR) for On-Board BAS Experiments

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-3776AB?logo=python&logoColor=white)](https://python.org)
[![Computer Vision](https://img.shields.io/badge/OpenCV-4.11-5C3EE8?logo=opencv&logoColor=white)](https://opencv.org)
[![Hand Tracking](https://img.shields.io/badge/MediaPipe-0.10.21-00897B?logo=google&logoColor=white)](https://developers.google.com/mediapipe)
[![Edge AI](https://img.shields.io/badge/Deployment-100%25%20Offline%20Edge-critical)](#)
[![Telemetry](https://img.shields.io/badge/WebSockets-16.0-blue)](https://websockets.readthedocs.io)
[![Stream](https://img.shields.io/badge/Live%20Stream-Flask%20MJPEG-black?logo=flask)](https://flask.palletsprojects.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Autonomous on-board AI assistant for astronauts conducting Biological & Physical Science (BAS) experiments in orbital space stations and lunar surface habitats. Designed for zero-cloud edge execution with communication-delay tolerance.**

---

## 📌 Table of Contents

- [Mission Background & Problem Context](#-mission-background--problem-context)
- [System Architecture](#-system-architecture)
- [Core Capabilities](#-core-capabilities)
- [Reference Experiment Protocol](#-reference-experiment-protocol)
- [Repository Structure](#-repository-structure)
- [Quick Start Guide](#-quick-start-guide)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Running the System](#running-the-system)
- [Mission Control Dashboard](#-mission-control-dashboard)
- [Custom Dataset Generation](#-custom-dataset-generation)
- [Demo Shortcuts & Judging Controls](#-demo-shortcuts--judging-controls)
- [Calibration & Tuning Guide](#-calibration--tuning-guide)
- [Known Limitations & Future Roadmap](#-known-limitations--future-roadmap)
- [License](#-license)

---

## 🛰️ Mission Background & Problem Context

During long-duration spaceflight (such as the International Space Station or Artemis Lunar Base), astronauts perform intricate Biological and Physical Science (BAS) experiments inside confined payload racks. In deep-space operations:
- **Ground communication latency** (up to seconds on the Moon, and minutes on Mars) makes real-time video support from mission control impossible.
- **Bandwidth constraints & orbital blackouts** prevent cloud-based computer vision inference.
- **Procedural fidelity is critical**: An unnoticed mistake, skipped chemical step, or wrong sample box placement invalidates entire research cycles.

This prototype provides an **on-board, real-time edge assistant** that continuously monitors the experiment workspace through a fixed-payload camera, validates action sequences against a formal protocol state machine, issues immediate offline voice alerts upon procedural deviation, saves synchronized video recordings, and broadcasts live telemetry to the habitat's local monitoring dashboard.

---

## 🏗️ System Architecture

The pipeline processes high-framerate video frames entirely in local RAM and executes synchronous computer vision coupled with asynchronous I/O workers:

```text
               ┌──────────────────────────────────────────────┐
               │    Fixed-Payload Camera (640x480 @ 20 FPS)   │
               └──────────────────────┬───────────────────────┘
                                      │
                                      ▼
               ┌──────────────────────────────────────────────┐
               │         DETECTOR LAYER (detector.py)         │
               │  • MediaPipe Hands (wrist landmark index 0)  │
               │  • Dual-range HSV morphological color masks  │
               └──────────────────────┬───────────────────────┘
                                      │ (Hands + Box Bounding Boxes)
                                      ▼
               ┌──────────────────────────────────────────────┐
               │    ACTION RECOGNITION (action_recognizer.py) │
               │  • Spatial proximity (< 50px Euclidean dist) │
               │  • Temporal hysteresis (5 consecutive frames)│
               │  • Bimanual vs Single-hand interaction mode  │
               └──────────────────────┬───────────────────────┘
                                      │ (Confirmed Actions)
                                      ▼
               ┌──────────────────────────────────────────────┐
               │    SEQUENCE VALIDATOR (sequence_validator.py)│
               │  • Deterministic 5-step protocol FSM         │
               │  • Status: WAITING / SUCCESS / OUT_OF_SEQ    │
               └──────┬───────────────────┬───────────────────┘
                      │                   │
         ┌────────────┴─────┐      ┌──────┴────────────┐
         ▼                  ▼      ▼                   ▼
┌─────────────────┐ ┌────────────┐ ┌────────────────┐ ┌────────────────┐
│   VOICE ALERT   │ │   LOGGER   │ │ VIDEO STREAMER │ │WEBSOCKET BRIDGE│
│(voice_alert.py) │ │(logger.py) │ │ (streamer.py)  │ │   (main.py)    │
│ Offline TTS     │ │Structured  │ │• Local XVID AVI│ │Asyncio Thread  │
│ non-blocking    │ │JSON audit  │ │• HTTP MJPEG    │ │ws://127.0.0.1: │
│ worker queue    │ │+ plain text│ │  stream port   │ │8765            │
└─────────────────┘ └────────────┘ └──────┬─────────┘ └───────┬────────┘
                                          │                   │
                                          └─────────┬─────────┘
                                                    │
                                                    ▼
                                     ┌─────────────────────────────┐
                                     │ MISSION DASHBOARD (GUI)     │
                                     │ http://127.0.0.1:5000/      │
                                     │ Live Feed + Sequence Tracker│
                                     └─────────────────────────────┘
```

---

## ⚡ Core Capabilities

1. **Object Detection**: Fast HSV segmentation for experiment payload containers (Red & Yellow boxes), skipping false positive noise using contour area filtering.
2. **Hand Tracking**: MediaPipe Hands tracks wrist coordinates in real time ($20+$ FPS) on standard CPU hardware.
3. **Action Recognition**: Computes Euclidean hand-object interaction vectors; applies temporal confirmation hysteresis ($5$ frames) to prevent transient false triggers.
4. **Protocol Sequence Validation**: Deterministic state machine checks actions against expected order, flagging skipped steps and out-of-order manipulations.
5. **Offline Voice Alerts**: Non-blocking offline Text-to-Speech (pyttsx3) issues instant audible warnings to astronauts without halting vision processing.
6. **Structured Audit Logging**: Maintains a tamper-resistant JSON event log with automatic error recovery and exports lightweight plain-text mission summaries.
7. **Simultaneous Stream & Local Save**: Writes local `.avi` video while streaming multipart MJPEG over IP.
8. **Dark Mission-Control Dashboard**: Interactive HTML/CSS/JS telemetry console displaying live camera feed, active sequence states, violation alerts, and auto-reconnecting WebSockets.

---

## 🧪 Reference Experiment Protocol

The benchmark experiment protocol monitors manipulation of nested experiment containers:

| Step | Action Name | Expected Astronaut Action |
|:---:|:---|:---|
| **1** | `pick_red_box` | Grasp and lift the red sample box |
| **2** | `open_red_box` | Open the red box lid (bimanual interaction) |
| **3** | `pick_yellow_box` | Grasp and lift the yellow sample box |
| **4** | `open_yellow_box` | Open the yellow box lid |
| **5** | `place_back` | Return both sample boxes to resting storage |

---

## 📁 Repository Structure

```text
har_prototype/
├── main.py                 # Pipeline orchestrator & WebSocket telemetry server
├── config.py               # Shared constants, camera specs, ports, HSV limits
├── detector.py             # MediaPipe hand tracker + HSV color box segmenter
├── action_recognizer.py    # Spatial proximity calculation & temporal hysteresis
├── sequence_validator.py   # Protocol state machine & sequence ordering validator
├── logger.py               # Structured JSON audit log & plain-text summary exporter
├── voice_alert.py          # Asynchronous offline text-to-speech engine (pyttsx3)
├── streamer.py             # Dual local AVI recording & Flask MJPEG stream server
├── gui/
│   └── dashboard.html      # Mission-control responsive telemetry dashboard
├── dataset/                # Recorded custom experiment video clips
├── scripts/
│   └── record_samples.py   # Dataset recording utility with QA frame extraction
├── logs/                   # experiment_log.json and summary.txt
├── output/                 # recorded_video.avi
├── requirements.txt        # Python dependency manifest
└── README.md               # Detailed module guide
```

---

## 🚀 Quick Start Guide

### Prerequisites
- Python 3.10 to 3.12 (64-bit)
- Standard USB webcam or built-in payload camera
- Windows, Linux, or macOS

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>/har_prototype

# 2. Create a virtual environment
python -m venv venv

# 3. Activate the virtual environment
# Windows (PowerShell):
.\venv\Scripts\Activate.ps1
# Windows (Command Prompt):
.\venv\Scripts\activate.bat
# Linux / macOS:
source venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt
```

### Running the System

```bash
python main.py
```

Once started:
- Open your browser to **[http://127.0.0.1:5000/](http://127.0.0.1:5000/)** to access the live dashboard.
- Live video stream: `http://127.0.0.1:5000/video_feed`
- WebSocket telemetry: `ws://127.0.0.1:8765`

---

## 🖥️ Mission Control Dashboard

The dashboard provides a futuristic space station telemetry aesthetic:
- **Live Video Viewport**: Displays camera stream with real-time wrist vectors, bounding boxes, and interaction distance metrics.
- **Top HUD**: Shows recording status (`● REC · LOCAL + STREAM`), resolution, target framerate, and UTC clock.
- **Protocol Sequence Tracker**: Step-by-step list updating dynamically with `PENDING`, `ACTIVE ⚡`, and `DONE ✓` tags.
- **Alert Banner**: Flashing neon red violation banner when an unexpected or out-of-order action occurs, accompanied by spoken audio.
- **Session Telemetry Cards**: Completed steps counter, protocol flags count, and mission elapsed stopwatch.
- **Event Audit Log**: Real-time table appending every event (timestamp, action, outcome status).

---

## 📹 Custom Dataset Generation

The hackathon specification requires generating a custom, focused dataset replicating the experiment setup:

```bash
python scripts/record_samples.py
```

### Features:
- Maps keyboard numbers `1` to `5` to the five protocol steps.
- On keypress, records a **~3.0 second sample clip** (`.avi`) into `dataset/<step_name>/clip_<timestamp>.avi`.
- Automatically extracts **5 evenly spaced frames** (`.jpg`) for visual Quality Assurance (QA).
- Displays on-screen recording progress bars and terminal clip tallies to ensure balanced training data.
- **Recommended dataset target**: $10$–$15$ clips per step across varied lighting and angles.

---

## 🎮 Demo Shortcuts & Judging Controls

During live presentations, room lighting or table arrangements might vary. The system includes built-in fail-safe shortcuts in the OpenCV window:

| Key | Function |
|:---:|:---|
| **`1`** | Trigger Step 1: `pick_red_box` |
| **`2`** | Trigger Step 2: `open_red_box` |
| **`3`** | Trigger Step 3: `pick_yellow_box` |
| **`4`** | Trigger Step 4: `open_yellow_box` |
| **`5`** | Trigger Step 5: `place_back` |
| **`R`** | Reset experiment sequence to Step 1 |
| **`Q`** | Graceful shutdown, saving video and exporting summaries |

*(Tip: Pressing `2` before `1` demonstrates the instant out-of-sequence audio warning and alert banner!)*

---

## ⚙️ Calibration & Tuning Guide

Before demonstrating in a new venue, calibrate settings in [`config.py`](file:///d:/OneDrive/Desktop/SIH%202026/har_prototype/config.py):

1. **HSV Color Thresholds**:
   - Red uses dual wrap-around hue ranges (`RED_LOWER_1`, `RED_UPPER_1`, `RED_LOWER_2`, `RED_UPPER_2`).
   - Yellow uses `YELLOW_LOWER`, `YELLOW_UPPER`.
   - Run `python detector.py` to view raw masks and adjust boundaries for ambient room lighting.
2. **Proximity Distance**:
   - `PROXIMITY_THRESHOLD = 50`: Increase to `70`–`90` pixels if the camera is mounted higher above the experiment tray.
3. **Camera Device**:
   - `CAMERA_INDEX = 0`: Change to `1` if using an external USB camera.

---

## 🔭 Known Limitations & Future Roadmap

This prototype is intentionally designed as an efficient edge proof-of-concept. Planned advancements beyond this prototype include:

1. **Orientation-Agnostic 3D Human Mesh Recovery (HMR)**:
   In true microgravity, astronauts float freely. Future iterations will register 3D human pose relative to the payload rack coordinate system rather than assuming gravity-down alignment.
2. **Deep Hand-Object Interaction (HOI) Models**:
   Replacing proximity heuristics with a temporal Graph Convolutional Network (GCN) or spatio-temporal Transformer trained specifically on dexterous astronaut glove manipulations.
3. **Hardware Acceleration**:
   Deploying quantized ONNX models onto radiation-hardened edge accelerators (such as Google Coral Edge TPU or Intel Myriad).

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
