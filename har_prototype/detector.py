"""
Real-time Vision Detection Layer for Space Station / Lunar BAS Experiments.
Combines Google MediaPipe Hand Tracking (for astronaut wrist position & gestures)
with lightweight dual-range HSV color masking (for red and yellow experiment boxes).
Optimized for 20+ FPS offline edge performance on standard camera hardware.
"""

import cv2
import numpy as np
import time
from typing import List, Dict, Tuple, Any, Optional
import config

try:
    import mediapipe as mp
    # MediaPipe solutions API
    if hasattr(mp, "solutions") and hasattr(mp.solutions, "hands"):
        mp_hands = mp.solutions.hands
        mp_drawing = mp.solutions.drawing_utils
    else:
        from mediapipe.python.solutions import hands as mp_hands
        from mediapipe.python.solutions import drawing_utils as mp_drawing
except Exception as e:
    mp_hands = None
    mp_drawing = None
    print(f"[DETECTOR WARNING] MediaPipe import warning: {e}")


class Detector:
    """
    Modular detector handling astronaut hand tracking and colored box localization.
    """

    def __init__(self, max_num_hands: int = 2, min_detection_confidence: float = 0.5):
        self.max_num_hands = max_num_hands
        self.min_detection_confidence = min_detection_confidence
        self.hands_module = None

        if mp_hands is not None:
            try:
                self.hands_module = mp_hands.Hands(
                    static_image_mode=False,
                    max_num_hands=self.max_num_hands,
                    min_detection_confidence=self.min_detection_confidence,
                    min_tracking_confidence=0.5
                )
                print("[DETECTOR] MediaPipe Hands initialized successfully.")
            except Exception as e:
                print(f"[DETECTOR ERROR] Failed to initialize MediaPipe Hands: {e}")
        else:
            print("[DETECTOR WARN] MediaPipe Hands not available in environment.")

    def detect_hands(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detects astronaut hands in the frame.

        Returns:
            List of dicts: [{"position": (cx, cy), "handedness": "Left"/"Right"}, ...]
            where (cx, cy) is the wrist landmark (index 0) in pixel coords.
            Returns empty list if no hands found.
        """
        if self.hands_module is None or frame is None:
            return []

        h, w, _ = frame.shape
        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False

        results = self.hands_module.process(rgb_frame)
        rgb_frame.flags.writeable = True

        detected_hands = []
        if results.multi_hand_landmarks:
            for idx, hand_lms in enumerate(results.multi_hand_landmarks):
                # Wrist is index 0
                wrist = hand_lms.landmark[0]
                cx, cy = int(wrist.x * w), int(wrist.y * h)

                handedness = "Right"
                if results.multi_handedness and idx < len(results.multi_handedness):
                    handedness = results.multi_handedness[idx].classification[0].label

                detected_hands.append({
                    "position": (cx, cy),
                    "handedness": handedness,
                    "landmarks": hand_lms
                })

        return detected_hands

    def detect_boxes(self, frame: np.ndarray) -> Dict[str, Dict[str, Any]]:
        """
        Detects red and yellow experiment boxes using fast HSV color segmentation.

        Returns:
            Dict containing detected boxes per color:
            {
               "red": {"center": (cx, cy), "bbox": (x, y, w, h)},
               "yellow": {"center": (cx, cy), "bbox": (x, y, w, h)}
            }
            Skips a color if nothing exceeds MIN_BOX_CONTOUR_AREA to avoid false positives.
        """
        boxes = {}
        if frame is None:
            return boxes

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        kernel = np.ones((5, 5), np.uint8)

        # 1. Red Mask (combines lower and upper red hue wraps)
        mask_r1 = cv2.inRange(hsv, config.RED_LOWER_1, config.RED_UPPER_1)
        mask_r2 = cv2.inRange(hsv, config.RED_LOWER_2, config.RED_UPPER_2)
        red_mask = cv2.bitwise_or(mask_r1, mask_r2)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)

        r_contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if r_contours:
            largest_red = max(r_contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_red)
            if area >= config.MIN_BOX_CONTOUR_AREA:
                x, y, w, h = cv2.boundingRect(largest_red)
                boxes["red"] = {
                    "center": (int(x + w / 2), int(y + h / 2)),
                    "bbox": (int(x), int(y), int(w), int(h)),
                    "area": area
                }

        # 2. Yellow Mask
        yellow_mask = cv2.inRange(hsv, config.YELLOW_LOWER, config.YELLOW_UPPER)
        yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_OPEN, kernel)
        yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_CLOSE, kernel)

        y_contours, _ = cv2.findContours(yellow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if y_contours:
            largest_yellow = max(y_contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_yellow)
            if area >= config.MIN_BOX_CONTOUR_AREA:
                x, y, w, h = cv2.boundingRect(largest_yellow)
                boxes["yellow"] = {
                    "center": (int(x + w / 2), int(y + h / 2)),
                    "bbox": (int(x), int(y), int(w), int(h)),
                    "area": area
                }

        return boxes

    def draw_debug(
        self,
        frame: np.ndarray,
        hands: List[Dict[str, Any]],
        boxes: Dict[str, Dict[str, Any]]
    ) -> np.ndarray:
        """
        Draws color-coded bounding boxes, proximity circles, and labels onto the frame.
        """
        annotated = frame.copy()

        # 1. Draw Boxes
        for color, box_data in boxes.items():
            cx, cy = box_data["center"]
            x, y, w, h = box_data["bbox"]

            if color == "red":
                box_color = (40, 40, 245)      # BGR Red
                text_color = (60, 60, 255)
            else:
                box_color = (0, 215, 255)       # BGR Yellow
                text_color = (0, 230, 255)

            # Bounding box
            cv2.rectangle(annotated, (x, y), (x + w, y + h), box_color, 2)
            # Center target crosshair
            cv2.circle(annotated, (cx, cy), 5, box_color, -1)
            # Proximity threshold radius
            cv2.circle(annotated, (cx, cy), config.PROXIMITY_THRESHOLD, box_color, 1, cv2.LINE_AA)

            label = f"{color.upper()} BOX ({cx},{cy})"
            cv2.putText(
                annotated,
                label,
                (x, max(18, y - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                text_color,
                1,
                cv2.LINE_AA
            )

        # 2. Draw Hands & Interaction Vectors
        for hand in hands:
            hx, hy = hand["position"]
            label = f"{hand.get('handedness', 'Hand')} Wrist ({hx},{hy})"

            # Hand skeleton if landmarks available
            if mp_drawing and "landmarks" in hand and hand["landmarks"] is not None:
                mp_drawing.draw_landmarks(
                    annotated,
                    hand["landmarks"],
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=1, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(0, 240, 255), thickness=1)
                )

            # Highlight wrist center
            cv2.circle(annotated, (hx, hy), 7, (255, 255, 0), -1)
            cv2.putText(
                annotated,
                label,
                (hx + 10, hy - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (255, 255, 0),
                1,
                cv2.LINE_AA
            )

            # Check distance to each detected box and draw connecting line
            for color, box_data in boxes.items():
                bx, by = box_data["center"]
                dist = np.hypot(hx - bx, hy - by)
                if dist <= config.PROXIMITY_THRESHOLD:
                    # In interaction range! Draw active green vector
                    cv2.line(annotated, (hx, hy), (bx, by), (0, 255, 0), 2, cv2.LINE_AA)
                    cv2.putText(
                        annotated,
                        f"NEAR {color.upper()}: {int(dist)}px",
                        (int((hx + bx) / 2), int((hy + by) / 2) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        (0, 255, 0),
                        1,
                        cv2.LINE_AA
                    )

        return annotated


if __name__ == "__main__":
    print("=" * 60)
    print("LIVE DEMO / TEST: Detector Layer (MediaPipe + HSV Boxes)")
    print(f"Camera Index: {config.CAMERA_INDEX} | Target: ~{config.TARGET_FPS} FPS")
    print("Press 'q' in the OpenCV window to exit.")
    print("=" * 60)

    detector = Detector()
    cap = cv2.VideoCapture(config.CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)

    prev_time = time.time()
    fps = 0.0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("[WARN] Failed to read from webcam.")
            break

        frame = cv2.resize(frame, (config.FRAME_WIDTH, config.FRAME_HEIGHT))

        # Detect
        hands = detector.detect_hands(frame)
        boxes = detector.detect_boxes(frame)

        # Annotate
        annotated = detector.draw_debug(frame, hands, boxes)

        # FPS calculation
        curr_time = time.time()
        fps = 0.9 * fps + 0.1 * (1.0 / max(0.001, curr_time - prev_time))
        prev_time = curr_time

        cv2.putText(
            annotated,
            f"FPS: {fps:.1f} | Hands: {len(hands)} | Boxes: {list(boxes.keys())}",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA
        )

        cv2.imshow("BAS HAR - Detector Debug Feed", annotated)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
