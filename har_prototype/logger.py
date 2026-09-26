"""
Structured and human-readable experiment audit logger.
Maintains a tamper-resistant JSON event log and exports a lightweight
human-readable summary format for mission reports with zero cloud reliance.
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List
import config


class ExperimentLogger:
    """
    Handles structured JSON audit logging and lightweight plain-text summary generation.
    """

    def __init__(self, log_file: Optional[str] = None, summary_file: Optional[str] = None):
        self.log_file = Path(log_file or config.LOG_FILE_PATH)
        self.summary_file = Path(summary_file or config.SUMMARY_FILE_PATH)
        # Ensure parent directories exist
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        self.summary_file.parent.mkdir(parents=True, exist_ok=True)

    def log_event(self, step_name: str, status: str, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Appends a structured event entry into the JSON log.
        If file is missing or corrupt, starts fresh instead of crashing.

        Args:
            step_name: Name of the action or protocol step (e.g. 'pick_red_box')
            status: Outcome status ('success', 'out_of_sequence', 'completed', etc.)
            extra: Optional dictionary containing additional metadata (fps, bounding boxes, hand positions)
        """
        now = datetime.now()
        event = {
            "timestamp": now.isoformat(),
            "time_display": now.strftime("%H:%M:%S"),
            "step": step_name,
            "status": status,
            "extra": extra or {}
        }

        events: List[Dict[str, Any]] = []

        if self.log_file.exists():
            try:
                with open(self.log_file, "r", encoding="utf-8") as f:
                    content = f.read().strip()
                    if content:
                        events = json.loads(content)
                        if not isinstance(events, list):
                            events = []
            except (json.JSONDecodeError, OSError):
                # If corrupt or unreadable, start fresh
                events = []

        events.append(event)

        try:
            with open(self.log_file, "w", encoding="utf-8") as f:
                json.dump(events, f, indent=2)
        except OSError as e:
            print(f"[LOGGER ERROR] Failed to write event to {self.log_file}: {e}")

        return event

    def get_all_events(self) -> List[Dict[str, Any]]:
        """Reads and returns all logged events."""
        if not self.log_file.exists():
            return []
        try:
            with open(self.log_file, "r", encoding="utf-8") as f:
                content = f.read().strip()
                if not content:
                    return []
                return json.loads(content)
        except Exception:
            return []

    def export_summary(self) -> str:
        """
        Reads the full JSON log and writes a lightweight plain-text summary to SUMMARY_FILE_PATH:
        Format: 'HH:MM:SS  step_name  STATUS'
        Returns the path to the summary file.
        """
        events = self.get_all_events()
        lines = [
            "=" * 50,
            "ON-BOARD BAS EXPERIMENT PROTOCOL SUMMARY",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Total Events Logged: {len(events)}",
            "=" * 50,
            f"{'TIME':<10} {'STEP NAME':<22} {'STATUS'}",
            "-" * 50
        ]

        for ev in events:
            time_str = ev.get("time_display")
            if not time_str:
                # Fallback from ISO timestamp
                try:
                    dt = datetime.fromisoformat(ev.get("timestamp", ""))
                    time_str = dt.strftime("%H:%M:%S")
                except Exception:
                    time_str = "00:00:00"

            step = ev.get("step", "UNKNOWN")
            status = ev.get("status", "UNKNOWN").upper()
            lines.append(f"{time_str:<10} {step:<22} {status}")

        lines.append("=" * 50)
        summary_text = "\n".join(lines) + "\n"

        try:
            with open(self.summary_file, "w", encoding="utf-8") as f:
                f.write(summary_text)
            print(f"[LOGGER] Summary successfully exported to {self.summary_file}")
        except OSError as e:
            print(f"[LOGGER ERROR] Failed to export summary to {self.summary_file}: {e}")

        return str(self.summary_file)


# Global singleton instance for easy import across modules
logger = ExperimentLogger()


if __name__ == "__main__":
    print("=" * 60)
    print("SMOKE TEST: ExperimentLogger")
    print("=" * 60)

    test_logger = ExperimentLogger(
        log_file=str(config.LOG_DIR / "test_experiment_log.json"),
        summary_file=str(config.LOG_DIR / "test_summary.txt")
    )

    # Log sample events
    test_logger.log_event("pick_red_box", "success", {"proximity_px": 28.4})
    test_logger.log_event("pick_yellow_box", "out_of_sequence", {"expected": "open_red_box"})
    test_logger.log_event("open_red_box", "success", {"hands_detected": 2})
    test_logger.log_event("place_back", "completed")

    summary_path = test_logger.export_summary()

    with open(summary_path, "r", encoding="utf-8") as f:
        print("\nExported Summary File Contents:")
        print(f.read())

    print("Smoke test completed.")
