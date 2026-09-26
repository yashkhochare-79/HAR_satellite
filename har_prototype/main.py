"""
Main Integration Pipeline & Orchestration Layer for BAS AI Human Activity Recognition (HAR).
Coordinates real-time computer vision detection, proximity-based action recognition,
deterministic protocol sequence validation, offline voice alerts, local video storage,
IP MJPEG streaming, and asynchronous WebSocket state synchronization for mission control.
"""

import sys
import time
import json
import asyncio
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional, Set, Any
import cv2
import numpy as np
import websockets

# Import local prototype modules
import config
from detector import Detector
from action_recognizer import ActionRecognizer
from sequence_validator import SequenceValidator
from logger import logger
from voice_alert import voice_alert
from streamer import VideoStreamer

# Global set to track connected WebSocket client sessions
connected_ws_clients: Set[Any] = set()
ws_loop: Optional[asyncio.AbstractEventLoop] = None
latest_telemetry_cache: dict = {}


# ==============================================================================
# WEBSOCKET TELEMETRY BRIDGE
# ==============================================================================
# WHY A DEDICATED THREAD & ASYNCIO LOOP:
# OpenCV's real-time capture and inference loop must execute synchronously at a
# steady TARGET_FPS without being stalled by network I/O, slow HTTP clients,
# or WebSocket TCP handshakes. Running the WebSocket server in its own thread with
# an isolated asyncio loop guarantees that telemetry broadcasts are purely
# non-blocking and decoupled from the computer vision pipeline.
# ==============================================================================

async def ws_handler(websocket):
    """
    Handles new WebSocket connections from the dashboard UI.
    Immediately transmits the latest telemetry state so the GUI renders without delay.
    """
    connected_ws_clients.add(websocket)
    try:
        if latest_telemetry_cache:
            await websocket.send(json.dumps(latest_telemetry_cache))
        # Keep connection open until client closes
        async for _ in websocket:
            pass
    except Exception:
        pass
    finally:
        connected_ws_clients.discard(websocket)


async def _async_broadcast(message_str: str):
    """Broadcasts a JSON string to all currently connected dashboard clients."""
    for client in list(connected_ws_clients):
        try:
            await client.send(message_str)
        except Exception:
            connected_ws_clients.discard(client)


def broadcast_telemetry(state_data: dict):
    """
    Thread-safe dispatcher called from the OpenCV loop.
    Pushes state_data to the WebSocket asyncio loop on the background thread.
    """
    global latest_telemetry_cache
    latest_telemetry_cache = state_data
    if ws_loop is not None and ws_loop.is_running():
        msg_str = json.dumps(state_data)
        asyncio.run_coroutine_threadsafe(_async_broadcast(msg_str), ws_loop)


def run_websocket_server():
    """Entry point for the background WebSocket server thread."""
    global ws_loop
    ws_loop = asyncio.new_event_loop()
    asyncio.set_event_loop(ws_loop)

    async def start_server():
        async with websockets.serve(ws_handler, config.WS_HOST, config.WS_PORT):
            print(f"[WEBSOCKET] Live Telemetry Server listening on ws://{config.WS_HOST}:{config.WS_PORT}")
            # Keep server running until loop is stopped
            await asyncio.Future()

    try:
        ws_loop.run_until_complete(start_server())
    except asyncio.CancelledError:
        pass
    finally:
        ws_loop.close()


# ==============================================================================
# MAIN VISION & PROTOCOL INTEGRATION PIPELINE
# ==============================================================================

def main():
    print("=" * 70)
    print("AI HUMAN ACTIVITY RECOGNITION (HAR) - ON-BOARD BAS EXPERIMENT")
    print(f"Protocol: {' -> '.join(config.EXPERIMENT_STEPS)}")
    print(f"Camera Index: {config.CAMERA_INDEX} | Resolution: {config.FRAME_WIDTH}x{config.FRAME_HEIGHT} @ {config.TARGET_FPS} FPS")
    print(f"MJPEG Live Stream: http://{config.STREAM_IP}:{config.STREAM_PORT}/video_feed")
    print(f"Dashboard UI: file:///{Path(config.BASE_DIR / 'gui' / 'dashboard.html').resolve().as_posix()}")
    print("=" * 70)
    print("Keyboard Controls:")
    print("  [Q] Exit system")
    print("  [R] Reset experiment sequence")
    print("  [1-5] Manual test trigger for steps 1 to 5")
    print("=" * 70)

    # 1. Initialize Subsystems
    print("[INIT] Initializing Vision Detector...")
    detector = Detector()

    print("[INIT] Initializing Action Recognizer...")
    action_recognizer = ActionRecognizer()

    print("[INIT] Initializing Protocol Sequence Validator...")
    validator = SequenceValidator()

    print("[INIT] Initializing Video Streamer & Local Storage...")
    streamer = VideoStreamer()

    print("[INIT] Starting Offline Voice Alert Manager...")
    voice_alert.start()
    voice_alert.speak_alert("On-board experiment assistant initialized and ready.")

    # 2. Start WebSocket Server Thread
    ws_thread = threading.Thread(
        target=run_websocket_server,
        name="WebSocketServerThread",
        daemon=True
    )
    ws_thread.start()

    # 3. Open Payload Camera Feed
    cap = cv2.VideoCapture(config.CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)

    if not cap.isOpened():
        print(f"[WARN] Unable to open camera at index {config.CAMERA_INDEX}. Entering simulation mode.")

    start_time = time.time()
    last_confirmed_step: Optional[str] = None
    flags_count = 0
    last_event = {}
    fps = float(config.TARGET_FPS)
    frame_interval = 1.0 / config.TARGET_FPS

    # Initial telemetry broadcast
    broadcast_telemetry({
        "current_step": "STANDBY",
        "next_step": validator.get_current_expected_step(),
        "status": "ready",
        "stats": {
            "done": 0,
            "total": len(validator.steps),
            "flags": 0,
            "elapsed_seconds": 0.0
        },
        "last_event": None
    })

    window_name = "BAS On-board Experiment Assistant (Local Display)"
    try:
        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    except Exception:
        pass

    try:
        while True:
            loop_start = time.time()

            # a. Capture Frame with Robust Error Handling
            ret = False
            frame = None
            if cap.isOpened():
                try:
                    ret, frame = cap.read()
                except Exception as e:
                    print(f"[WARN] Camera read exception: {e}")
                    ret = False

            if not ret or frame is None:
                # Generate synthetic test frame to maintain telemetry and stream continuity
                frame = np.zeros((config.FRAME_HEIGHT, config.FRAME_WIDTH, 3), dtype=np.uint8)
                cv2.putText(
                    frame,
                    "PAYLOAD CAMERA FEED OFFLINE / SIMULATION",
                    (40, config.FRAME_HEIGHT // 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 165, 255),
                    2
                )
            else:
                frame = cv2.resize(frame, (config.FRAME_WIDTH, config.FRAME_HEIGHT))

            # b. Run Computer Vision Detections
            hands = detector.detect_hands(frame)
            boxes = detector.detect_boxes(frame)

            # c. Run Action Recognition
            confirmed_action = action_recognizer.recognize(hands, boxes)

            # Check for Keyboard Inputs (manual overrides for demo flexibility)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == ord('Q'):
                print("\n[OPERATOR] Termination requested. Shutting down...")
                break
            elif key == ord('r') or key == ord('R'):
                print("\n[OPERATOR] Resetting experiment sequence...")
                validator.reset()
                action_recognizer.reset()
                flags_count = 0
                last_confirmed_step = None
                last_event = {"time": datetime.now().strftime("%H:%M:%S"), "step": "RESET", "status": "reset"}
                broadcast_telemetry({
                    "current_step": "STANDBY",
                    "next_step": validator.get_current_expected_step(),
                    "status": "reset",
                    "stats": {
                        "done": 0,
                        "total": len(validator.steps),
                        "flags": 0,
                        "elapsed_seconds": round(time.time() - start_time, 1)
                    },
                    "last_event": last_event
                })
                continue
            elif ord('1') <= key <= ord(str(len(config.EXPERIMENT_STEPS))):
                # Presenter shortcut to manually simulate any step
                step_idx = key - ord('1')
                manual_action = config.EXPERIMENT_STEPS[step_idx]
                print(f"[KEYBOARD SIMULATION] Manual action triggered: {manual_action}")
                confirmed_action = manual_action

            # d. Validate Sequence
            status, info = validator.validate(confirmed_action)

            # e. Handle Sequence Status Transitions
            status_changed = False
            if confirmed_action is not None and status != "waiting":
                last_confirmed_step = confirmed_action
                status_changed = True
                timestamp_str = datetime.now().strftime("%H:%M:%S")

                if status == "out_of_sequence":
                    flags_count += 1
                    print(f"\n[ALERT] Out of sequence action: '{confirmed_action}'. Expected: '{info}'")
                    logger.log_event(confirmed_action, "out_of_sequence", {"expected": info})
                    voice_alert.speak_alert(f"Warning: unexpected step. Expected {info.replace('_', ' ')}")
                    last_event = {
                        "time": timestamp_str,
                        "step": confirmed_action,
                        "status": "out_of_sequence"
                    }

                elif status == "success":
                    print(f"\n[SUCCESS] Confirmed step: '{confirmed_action}'. Next expected: '{info}'")
                    logger.log_event(confirmed_action, "success", {"next_expected": info})
                    voice_alert.speak_alert(f"Step complete. Next step: {info.replace('_', ' ')}")
                    last_event = {
                        "time": timestamp_str,
                        "step": confirmed_action,
                        "status": "success"
                    }

                elif status == "completed":
                    print(f"\n[COMPLETE] Final step '{confirmed_action}' finished. Experiment complete!")
                    logger.log_event(confirmed_action, "completed", {})
                    voice_alert.speak_alert("Experiment complete. All protocol steps verified.")
                    logger.export_summary()
                    last_event = {
                        "time": timestamp_str,
                        "step": confirmed_action,
                        "status": "completed"
                    }

            # f. Draw Annotations & Telemetry HUD on Frame
            annotated_frame = detector.draw_debug(frame, hands, boxes)

            # Render Status HUD Banner on Video Feed
            hud_bg = annotated_frame.copy()
            cv2.rectangle(hud_bg, (0, 0), (config.FRAME_WIDTH, 45), (10, 15, 25), -1)
            cv2.addWeighted(hud_bg, 0.75, annotated_frame, 0.25, 0, annotated_frame)

            curr_step_label = last_confirmed_step or "WAITING FOR INTERACTION"
            next_step_label = validator.get_current_expected_step() or "COMPLETED"
            cv2.putText(
                annotated_frame,
                f"ACTION: {curr_step_label} | EXPECTED: {next_step_label}",
                (12, 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 240, 255),
                1,
                cv2.LINE_AA
            )

            status_color = (0, 255, 0) if status in ("success", "completed") else ((0, 0, 255) if status == "out_of_sequence" else (200, 200, 200))
            cv2.putText(
                annotated_frame,
                f"STATUS: {status.upper()} | FLAGS: {flags_count} | PROGRESS: {validator.current_index}/{len(validator.steps)}",
                (12, 38),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                status_color,
                1,
                cv2.LINE_AA
            )

            # Pass frame to streamer (records locally + updates IP stream)
            streamer.update(annotated_frame)

            # g. Broadcast Telemetry over WebSocket
            # Broadcast immediately on event change OR periodically (every 5 frames) to refresh elapsed time
            elapsed_sec = round(time.time() - start_time, 1)
            if status_changed or (int(time.time() * 4) % 4 == 0):
                telemetry = {
                    "current_step": last_confirmed_step or "STANDBY",
                    "next_step": validator.get_current_expected_step(),
                    "status": status if status != "waiting" else "in_progress",
                    "stats": {
                        "done": validator.current_index,
                        "total": len(validator.steps),
                        "flags": flags_count,
                        "elapsed_seconds": elapsed_sec
                    },
                    "last_event": last_event
                }
                broadcast_telemetry(telemetry)

            # Show local OpenCV window
            try:
                cv2.imshow(window_name, annotated_frame)
            except Exception:
                pass

            # Maintain TARGET_FPS pacing
            elapsed_loop = time.time() - loop_start
            sleep_time = max(0.001, frame_interval - elapsed_loop)
            time.sleep(sleep_time)

            # Handle experiment protocol completion (keep server standing by rather than abrupt exit)
            if validator.is_completed and status == "completed":
                print("\n[MISSION PROTOCOL COMPLETED] All steps verified! Assistant standing by.")
                print("Press 'R' to reset sequence or 'Q' to quit.")

    except KeyboardInterrupt:
        print("\n[MAIN] Interrupted by keyboard.")
    except Exception as e:
        print(f"\n[MAIN ERROR] Unexpected error in main loop: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Graceful Shutdown Sequence
        print("[CLEANUP] Releasing camera capture...")
        if cap.isOpened():
            cap.release()
        cv2.destroyAllWindows()

        print("[CLEANUP] Shutting down video streamer & local recording...")
        streamer.shutdown()

        print("[CLEANUP] Shutting down offline voice alerts...")
        voice_alert.shutdown()

        print("[CLEANUP] Stopping WebSocket server loop...")
        if ws_loop is not None and ws_loop.is_running():
            ws_loop.call_soon_threadsafe(ws_loop.stop)

        print("[CLEANUP] Mission Assistant shutdown complete.")


if __name__ == "__main__":
    main()
