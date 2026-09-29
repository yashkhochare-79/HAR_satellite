"""
Real-time Video Streamer & Local Storage Engine.
Simultaneously records raw/annotated camera feed to a local AVI container
and exposes an IP-based MJPEG multipart live stream via Flask with zero-latency lock buffering.
"""

import threading
import time
import logging
from typing import Optional
import json
import cv2
from flask import Flask, Response, render_template_string, jsonify
from werkzeug.serving import make_server
import config

# Suppress Flask development server banner in console
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)


class VideoStreamer:
    """
    Handles simultaneous local file recording (OpenCV VideoWriter)
    and IP MJPEG network streaming (Flask server on background thread).
    """

    def __init__(
        self,
        stream_ip: str = config.STREAM_IP,
        stream_port: int = config.STREAM_PORT,
        video_output_path: str = config.VIDEO_OUTPUT_PATH,
        frame_width: int = config.FRAME_WIDTH,
        frame_height: int = config.FRAME_HEIGHT,
        target_fps: int = config.TARGET_FPS
    ):
        self.stream_ip = stream_ip
        self.stream_port = stream_port
        self.video_output_path = video_output_path
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.target_fps = target_fps

        # Thread-safe buffer holding only the most recent JPEG frame
        self._frame_lock = threading.Lock()
        self._latest_jpeg: Optional[bytes] = None
        self._latest_telemetry: dict = {}
        self._running = True

        # Initialize local VideoWriter (XVID container)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self._writer = cv2.VideoWriter(
            self.video_output_path,
            fourcc,
            float(self.target_fps),
            (self.frame_width, self.frame_height)
        )
        if not self._writer.isOpened():
            # Fallback to MJPG if XVID is unsupported by OS codec
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            self._writer = cv2.VideoWriter(
                self.video_output_path,
                fourcc,
                float(self.target_fps),
                (self.frame_width, self.frame_height)
            )

        # Set up Flask app
        self._app = Flask(__name__)
        self._setup_routes()

        # Start Flask HTTP server on dedicated daemon thread
        self._server = make_server(self.stream_ip, self.stream_port, self._app, threaded=True)
        self._server_thread = threading.Thread(
            target=self._server.serve_forever,
            name="MJPEGStreamerThread",
            daemon=True
        )
        self._server_thread.start()
        print(f"[STREAMER] Live MJPEG stream active at http://{self.stream_ip}:{self.stream_port}/video_feed")
        print(f"[STREAMER] Local recording path: {self.video_output_path}")

    def set_telemetry(self, telemetry_data: dict):
        """Thread-safe update of latest telemetry state for HTTP/SSE clients."""
        with self._frame_lock:
            self._latest_telemetry = dict(telemetry_data)

    def _setup_routes(self):
        """Defines Flask routes for MJPEG streaming, HTML dashboard, and SSE/HTTP telemetry."""

        @self._app.route("/video_feed")
        def video_feed():
            return Response(
                self._mjpeg_generator(),
                mimetype="multipart/x-mixed-replace; boundary=frame"
            )

        @self._app.route("/api/telemetry")
        def api_telemetry():
            with self._frame_lock:
                data = dict(self._latest_telemetry)
            resp = jsonify(data)
            resp.headers["Access-Control-Allow-Origin"] = "*"
            return resp

        @self._app.route("/api/events")
        def sse_events():
            def event_stream():
                last_encoded = ""
                while self._running:
                    with self._frame_lock:
                        curr = json.dumps(self._latest_telemetry)
                    if curr != last_encoded and self._latest_telemetry:
                        last_encoded = curr
                        yield f"data: {curr}\n\n"
                    time.sleep(0.08)
            resp = Response(event_stream(), mimetype="text/event-stream")
            resp.headers["Access-Control-Allow-Origin"] = "*"
            resp.headers["Cache-Control"] = "no-cache"
            resp.headers["X-Accel-Buffering"] = "no"
            return resp

        @self._app.route("/")
        @self._app.route("/dashboard")
        def index():
            dashboard_file = config.BASE_DIR / "gui" / "dashboard.html"
            if dashboard_file.exists():
                with open(dashboard_file, "r", encoding="utf-8") as f:
                    return f.read()
            return render_template_string("""
            <!DOCTYPE html>
            <html>
            <head><title>BAS Fixed-Payload Camera Live Feed</title></head>
            <body style="background:#090d16;color:#e2e8f0;font-family:sans-serif;text-align:center;padding:20px;">
                <h2>ON-BOARD PAYLOAD FEED (MJPEG)</h2>
                <img src="/video_feed" style="border:2px solid #00f0ff;border-radius:6px;max-width:90%;">
                <p style="color:#94a3b8;font-size:12px;">Streaming from {{ ip }}:{{ port }}</p>
            </body>
            </html>
            """, ip=self.stream_ip, port=self.stream_port)

    def _mjpeg_generator(self):
        """Yields MJPEG frames to connected HTTP clients."""
        while self._running:
            with self._frame_lock:
                frame_data = self._latest_jpeg

            if frame_data is not None:
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + frame_data + b"\r\n"
                )
            # Sleep to match target FPS pacing
            time.sleep(1.0 / self.target_fps)

    def update(self, frame):
        """
        Writes frame to local video file and updates the live stream buffer.
        Non-blocking to the vision processing loop.
        """
        if not self._running or frame is None:
            return

        # 1. Write locally
        try:
            if self._writer.isOpened():
                # Ensure frame matches target dimension
                if frame.shape[1] != self.frame_width or frame.shape[0] != self.frame_height:
                    resized = cv2.resize(frame, (self.frame_width, self.frame_height))
                    self._writer.write(resized)
                else:
                    self._writer.write(frame)
        except Exception as e:
            print(f"[STREAMER ERROR] VideoWriter write failed: {e}")

        # 2. Encode to JPEG for network streaming
        try:
            ret, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            if ret:
                with self._frame_lock:
                    self._latest_jpeg = buffer.tobytes()
        except Exception as e:
            print(f"[STREAMER ERROR] JPEG encoding failed: {e}")

    def shutdown(self):
        """Releases the VideoWriter and stops the Flask server cleanly."""
        print("[STREAMER] Shutting down streamer and releasing video resources...")
        self._running = False

        if self._writer is not None:
            try:
                self._writer.release()
                print(f"[STREAMER] Local video saved to: {self.video_output_path}")
            except Exception as e:
                print(f"[STREAMER ERROR] Failed to close VideoWriter: {e}")

        if self._server is not None:
            try:
                self._server.shutdown()
            except Exception:
                pass


if __name__ == "__main__":
    import numpy as np

    print("=" * 60)
    print("SMOKE TEST: VideoStreamer")
    print("Starting webcam capture and streaming for ~10 seconds...")
    print(f"Open http://{config.STREAM_IP}:{config.STREAM_PORT}/video_feed in your browser.")
    print("=" * 60)

    streamer = VideoStreamer()

    cap = cv2.VideoCapture(config.CAMERA_INDEX)
    start_time = time.time()
    frames_sent = 0

    try:
        while time.time() - start_time < 10.0:
            ret, frame = cap.read()
            if not ret:
                # Generate synthetic test frame if no webcam available
                frame = np.zeros((config.FRAME_HEIGHT, config.FRAME_WIDTH, 3), dtype=np.uint8)
                cv2.putText(
                    frame,
                    f"TEST SYNTHETIC STREAM {time.strftime('%H:%M:%S')}",
                    (40, 240),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2
                )

            streamer.update(frame)
            frames_sent += 1
            time.sleep(1.0 / config.TARGET_FPS)
    finally:
        cap.release()
        streamer.shutdown()
        print(f"Smoke test completed. Sent {frames_sent} frames.")
