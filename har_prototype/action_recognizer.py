"""
Action Recognition Module for Space Station / Lunar BAS Experiments.
Translates raw computer vision detections (hand wrist landmarks and colored box centers)
into confirmed experiment protocol actions using spatial proximity and temporal hysteresis.
"""

import math
from typing import List, Dict, Tuple, Optional, Any, Union
import config


class ActionRecognizer:
    """
    Infers human-object interaction based on hand-box Euclidean proximity.
    Maintains a temporal confirmation counter across consecutive frames
    to prevent jitter/flicker and false positive state transitions.
    """

    def __init__(
        self,
        proximity_threshold: float = config.PROXIMITY_THRESHOLD,
        confirmation_frames: int = config.CONFIRMATION_FRAMES
    ):
        self.proximity_threshold = proximity_threshold
        self.confirmation_frames = confirmation_frames

        # Candidate action currently being tracked
        self.current_candidate: Optional[str] = None
        self.candidate_count: int = 0

        # Action confirmed in previous frame (to prevent repeated event spamming)
        self.last_confirmed_action: Optional[str] = None
        self.has_emitted_for_current_candidate: bool = False

        # Internal state tracking to distinguish sequence actions:
        # e.g., first pick vs open (or two-handed interaction for open)
        self.red_box_picked: bool = False
        self.red_box_opened: bool = False
        self.yellow_box_picked: bool = False
        self.yellow_box_opened: bool = False
        self.place_back_done: bool = False

    def _extract_positions(self, hand_positions: List[Union[Dict[str, Any], Tuple[int, int]]]) -> List[Tuple[int, int]]:
        """Normalizes hand inputs to a list of (x, y) coordinates."""
        positions = []
        for h in hand_positions:
            if isinstance(h, dict) and "position" in h:
                positions.append(h["position"])
            elif isinstance(h, (tuple, list)) and len(h) >= 2:
                positions.append((int(h[0]), int(h[1])))
        return positions

    def infer_candidate(
        self,
        hand_positions: List[Union[Dict[str, Any], Tuple[int, int]]],
        boxes: Dict[str, Dict[str, Any]]
    ) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
        """
        Calculates distance from every hand to every box center.
        If under proximity_threshold, determines the candidate interaction.

        Returns:
            Tuple[candidate_action_name, debug_metadata]
        """
        hands = self._extract_positions(hand_positions)
        if not hands or not boxes:
            return None, None

        closest_pair = None
        min_dist = float("inf")
        box_hand_counts: Dict[str, int] = {"red": 0, "yellow": 0}

        # Check distances
        for color, box_info in boxes.items():
            if not box_info or "center" not in box_info:
                continue
            bx, by = box_info["center"]

            for hx, hy in hands:
                dist = math.hypot(hx - bx, hy - by)
                if dist < min_dist:
                    min_dist = dist
                    closest_pair = (color, (hx, hy), (bx, by), dist)

                if dist <= self.proximity_threshold:
                    if color in box_hand_counts:
                        box_hand_counts[color] += 1

        if min_dist > self.proximity_threshold or closest_pair is None:
            return None, None

        interacting_color, hand_pt, box_pt, dist = closest_pair
        num_hands_on_box = box_hand_counts.get(interacting_color, 1)

        candidate = None
        debug_info = {
            "color": interacting_color,
            "dist": round(dist, 1),
            "hands_on_box": num_hands_on_box
        }

        # Protocol mapping logic:
        # 1. Two hands on a box strongly indicates an 'open' manipulation.
        # 2. One hand indicates a 'pick' (or sequence progression).
        if interacting_color == "red":
            if num_hands_on_box >= 2:
                candidate = "open_red_box"
            elif not self.red_box_picked:
                candidate = "pick_red_box"
            elif self.red_box_picked and not self.red_box_opened:
                candidate = "open_red_box"
            elif self.yellow_box_opened:
                # Placing items back after completion
                candidate = "place_back"
            else:
                candidate = "open_red_box"

        elif interacting_color == "yellow":
            if num_hands_on_box >= 2:
                candidate = "open_yellow_box"
            elif not self.yellow_box_picked:
                candidate = "pick_yellow_box"
            elif self.yellow_box_picked and not self.yellow_box_opened:
                candidate = "open_yellow_box"
            elif self.yellow_box_opened:
                candidate = "place_back"
            else:
                candidate = "open_yellow_box"

        return candidate, debug_info

    def recognize(
        self,
        hand_positions: List[Union[Dict[str, Any], Tuple[int, int]]],
        boxes: Dict[str, Dict[str, Any]]
    ) -> Optional[str]:
        """
        Processes current frame detections.
        Only confirms (returns) an action once the same candidate has persisted
        for CONFIRMATION_FRAMES consecutive frames. Resets if candidate changes or disappears.

        Returns:
            Action string once confirmed, or None.
        """
        candidate, _ = self.infer_candidate(hand_positions, boxes)

        if candidate is None:
            # Interaction lost or out of proximity -> reset streak counter
            self.current_candidate = None
            self.candidate_count = 0
            self.has_emitted_for_current_candidate = False
            return None

        if candidate == self.current_candidate:
            self.candidate_count += 1
        else:
            # Candidate changed to a different action
            self.current_candidate = candidate
            self.candidate_count = 1
            self.has_emitted_for_current_candidate = False

        # Check if consecutive frame threshold is satisfied
        if self.candidate_count >= self.confirmation_frames:
            if not self.has_emitted_for_current_candidate:
                # Emit confirmed action once per sustained gesture
                self.has_emitted_for_current_candidate = True
                self.last_confirmed_action = candidate
                self._update_internal_phase(candidate)
                return candidate

        return None

    def _update_internal_phase(self, action: str):
        """Updates internal state flags as actions are confirmed."""
        if action == "pick_red_box":
            self.red_box_picked = True
        elif action == "open_red_box":
            self.red_box_opened = True
        elif action == "pick_yellow_box":
            self.yellow_box_picked = True
        elif action == "open_yellow_box":
            self.yellow_box_opened = True
        elif action == "place_back":
            self.place_back_done = True

    def reset(self):
        """Clears all internal state counters and protocol phases."""
        self.current_candidate = None
        self.candidate_count = 0
        self.last_confirmed_action = None
        self.has_emitted_for_current_candidate = False
        self.red_box_picked = False
        self.red_box_opened = False
        self.yellow_box_picked = False
        self.yellow_box_opened = False
        self.place_back_done = False


if __name__ == "__main__":
    print("=" * 60)
    print("SMOKE TEST: ActionRecognizer")
    print(f"Proximity Threshold: {config.PROXIMITY_THRESHOLD}px | Confirmation: {config.CONFIRMATION_FRAMES} frames")
    print("=" * 60)

    recognizer = ActionRecognizer()

    # Simulated red box at (200, 200)
    mock_boxes = {
        "red": {"center": (200, 200), "bbox": (170, 170, 60, 60)},
        "yellow": {"center": (400, 200), "bbox": (370, 170, 60, 60)}
    }

    # Hand starts far away at (50, 50)
    print("\n1. Testing hand far away...")
    for f in range(3):
        res = recognizer.recognize([(50, 50)], mock_boxes)
        print(f"  Frame {f+1}: Result = {res}")

    # Hand moves close to red box at (210, 210) (distance ~14px < 50px)
    print("\n2. Testing hand approaching red box (streak accumulation)...")
    for f in range(1, config.CONFIRMATION_FRAMES + 3):
        res = recognizer.recognize([(210, 210)], mock_boxes)
        print(f"  Frame {f}: Candidate='{recognizer.current_candidate}' (count={recognizer.candidate_count}) -> Emit: {res}")

    # Now simulate two hands on red box to trigger open_red_box
    print("\n3. Testing two hands on red box...")
    for f in range(1, config.CONFIRMATION_FRAMES + 2):
        res = recognizer.recognize([(205, 205), (195, 195)], mock_boxes)
        print(f"  Frame {f}: Candidate='{recognizer.current_candidate}' (count={recognizer.candidate_count}) -> Emit: {res}")

    print("\nSmoke test completed.")
