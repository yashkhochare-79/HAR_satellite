"""
Sequence Validator for Space Station / Lunar BAS Experiments.
Validates performed human activities against the strict predefined protocol
defined in config.EXPERIMENT_STEPS using a deterministic state machine.
"""

from collections import deque
from typing import Tuple, Optional, List
import config


class SequenceValidator:
    """
    State machine that tracks and validates astronaut actions in real time.
    """

    def __init__(self, steps: Optional[List[str]] = None, max_history: int = 10):
        self.steps: List[str] = steps if steps is not None else list(config.EXPERIMENT_STEPS)
        self.current_index: int = 0
        # Rolling history of recent confirmed actions (stores up to max_history entries)
        self.history: deque = deque(maxlen=max_history)
        self.is_completed: bool = False

    def validate(self, confirmed_action: Optional[str]) -> Tuple[str, Optional[str]]:
        """
        Validates a newly confirmed action against expected protocol order.

        Args:
            confirmed_action: The action string confirmed by ActionRecognizer,
                              or None if no action confirmed in current frame.

        Returns:
            Tuple[status, info]:
                - ("waiting", None): If no action occurred or sequence is already completed.
                - ("success", next_step_name): If action matches expected step (advances index).
                - ("out_of_sequence", expected_step_name): If action is a known action but does
                  not match current expected step (covers both wrong step and skipped step cases).
                  NOTE: In future production iterations, 'wrong step' vs 'skipped step' can be
                  differentiated by checking if confirmed_action exists ahead in self.steps.
                - ("completed", None): Once all steps in the protocol are successfully finished.
        """
        if self.is_completed:
            return "completed", None

        if confirmed_action is None:
            return "waiting", None

        # Check if already completed before processing
        if self.current_index >= len(self.steps):
            self.is_completed = True
            return "completed", None

        expected_step = self.steps[self.current_index]

        # Record into rolling history
        self.history.append({
            "action": confirmed_action,
            "expected": expected_step,
            "index_at_time": self.current_index
        })

        if confirmed_action == expected_step:
            # Step matches expectation! Advance pointer
            self.current_index += 1
            if self.current_index >= len(self.steps):
                self.is_completed = True
                return "completed", None
            else:
                next_expected = self.steps[self.current_index]
                return "success", next_expected
        else:
            # Action does not match expected step.
            # Flag out-of-sequence event (could be wrong action or skipped action).
            # Note: For hackathon prototype, we flag the violation and remind the operator of expected step.
            return "out_of_sequence", expected_step

    def get_current_expected_step(self) -> Optional[str]:
        """Returns the step currently expected, or None if completed."""
        if self.current_index < len(self.steps):
            return self.steps[self.current_index]
        return None

    def get_progress(self) -> dict:
        """Returns current progress stats."""
        total = len(self.steps)
        done = self.current_index
        return {
            "done": done,
            "total": total,
            "completed": self.is_completed,
            "current_step": self.get_current_expected_step(),
            "percent": int((done / total) * 100) if total > 0 else 0
        }

    def reset(self):
        """Restarts the experiment protocol sequence back to step 0."""
        self.current_index = 0
        self.history.clear()
        self.is_completed = False


if __name__ == "__main__":
    print("=" * 60)
    print("SMOKE TEST: SequenceValidator")
    print(f"Protocol Steps: {config.EXPERIMENT_STEPS}")
    print("=" * 60)

    validator = SequenceValidator()

    # Test test actions including an intentional out-of-sequence action
    test_actions = [
        "pick_red_box",       # Should succeed -> expect open_red_box
        "pick_yellow_box",    # Deliberate out-of-sequence error! Expected open_red_box
        "open_red_box",       # Correct next step -> expect pick_yellow_box
        "pick_yellow_box",    # Should succeed -> expect open_yellow_box
        "open_yellow_box",    # Should succeed -> expect place_back
        "place_back"          # Final step -> completed
    ]

    for i, action in enumerate(test_actions, start=1):
        status, info = validator.validate(action)
        print(f"Action #{i}: '{action}' -> Status: [{status.upper()}] | Info: {info}")

    print("\nFinal Progress:", validator.get_progress())
    print("Smoke test completed.")
