"""
Custom Dataset Recording Utility for BAS Human Activity Recognition (HAR).
Allows researchers/operators to quickly record focused synthetic/custom video samples
of the 5 experiment protocol steps via webcam, generating labeled .avi clips and QA frame extracts.
"""

import sys
import time
from pathlib import Path
from datetime import datetime
import cv2
import numpy as np

# Ensure parent directory is in Python path to import config
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config


def get_clip_counts(steps, dataset_dir):
    """Counts existing recorded .avi clips per step folder."""
    counts = {}
    for step in steps:
        step_dir = dataset_dir / step
        if step_dir.exists():
            counts[step] = len(list(step_dir.glob("clip_*.avi")))
        else:
            counts[step] = 0
    return counts


def main():
    print("=" * 65)
    print("BAS HAR EXPERIMENT: CUSTOM DATASET RECORDING UTILITY")
    print(f"Target Camera Index: {config.CAMERA_INDEX} | Resolution: {config.FRAME_WIDTH}x{config.FRAME_HEIGHT}")
    print("Controls:")
    for idx, step in enumerate(config.EXPERIMENT_STEPS, start=1):
        print(f"  [{idx}] Record 3-second clip for: '{step}'")
    print("  [Q] Exit recording utility")
    print("=" * 65)

    dataset_path = Path(config.DATASET_DIR)
    dataset_path.mkdir(parents=True, exist_ok=True)

    # Initialize folders
    for step in config.EXPERIMENT_STEPS:
        (dataset_path / step).mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(config.CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)

    if not cap.isOpened():
        print(f"[ERROR] Cannot access camera at index {config.CAMERA_INDEX}.")
        print("Please check camera connections or update CAMERA_INDEX in config.py.")
        return

    clip_counts = get_clip_counts(config.EXPERIMENT_STEPS, dataset_path)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')

    window_name = "BAS HAR - Dataset Recording Utility"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    print("\nDataset Balance:")
    for s, c in clip_counts.items():
        print(f"  • {s:<18}: {c} clips")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] Failed to grab frame from camera.")
            time.sleep(0.05)
            continue

        frame = cv2.resize(frame, (config.FRAME_WIDTH, config.FRAME_HEIGHT))
        display_frame = frame.copy()

        # Render HUD Overlay
        overlay = display_frame.copy()
        cv2.rectangle(overlay, (10, 10), (config.FRAME_WIDTH - 10, 130), (10, 15, 25), -1)
        cv2.addWeighted(overlay, 0.75, display_frame, 0.25, 0, display_frame)

        cv2.putText(
            display_frame,
            "ON-BOARD HAR DATASET RECORDER (Press 1-5 to record, Q to quit)",
            (20, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 240, 255),
            1,
            cv2.LINE_AA
        )

        for i, step in enumerate(config.EXPERIMENT_STEPS, start=1):
            col = 20 if i <= 3 else 320
            row = 55 + ((i - 1) % 3) * 22
            count = clip_counts.get(step, 0)
            text = f"[{i}] {step:<18} (Clips: {count})"
            cv2.putText(
                display_frame,
                text,
                (col, row),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (200, 220, 240),
                1,
                cv2.LINE_AA
            )

        cv2.imshow(window_name, display_frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q') or key == ord('Q'):
            print("\nExiting dataset recording utility...")
            break

        # Check key 1 to 5
        selected_step = None
        if ord('1') <= key <= ord(str(len(config.EXPERIMENT_STEPS))):
            step_idx = key - ord('1')
            selected_step = config.EXPERIMENT_STEPS[step_idx]

        if selected_step is not None:
            print(f"\n[REC] Recording ~3.0s sample for '{selected_step}'...")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            step_dir = dataset_path / selected_step
            step_dir.mkdir(parents=True, exist_ok=True)

            video_file = step_dir / f"clip_{timestamp}.avi"
            out_writer = cv2.VideoWriter(
                str(video_file),
                fourcc,
                float(config.TARGET_FPS),
                (config.FRAME_WIDTH, config.FRAME_HEIGHT)
            )

            # Record for 3 seconds: total frames = 3 * TARGET_FPS
            total_record_frames = 3 * config.TARGET_FPS
            recorded_frames = []

            for f_idx in range(total_record_frames):
                r_ret, r_frame = cap.read()
                if not r_ret:
                    continue
                r_frame = cv2.resize(r_frame, (config.FRAME_WIDTH, config.FRAME_HEIGHT))
                out_writer.write(r_frame)
                recorded_frames.append(r_frame.copy())

                # Live Recording HUD
                rec_hud = r_frame.copy()
                progress_pct = int(((f_idx + 1) / total_record_frames) * 100)
                cv2.circle(rec_hud, (30, 30), 10, (0, 0, 255), -1)
                cv2.putText(
                    rec_hud,
                    f"RECORDING [{selected_step}] {progress_pct}%",
                    (50, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA
                )
                cv2.imshow(window_name, rec_hud)
                cv2.waitKey(int(1000 / config.TARGET_FPS))

            out_writer.release()
            print(f"[REC COMPLETE] Saved clip: {video_file}")

            # Save 5 evenly spaced sample frames as .jpg for visual QA
            num_samples = 5
            if len(recorded_frames) >= num_samples:
                indices = np.linspace(0, len(recorded_frames) - 1, num_samples, dtype=int)
                for qa_idx, frame_i in enumerate(indices, start=1):
                    qa_path = step_dir / f"clip_{timestamp}_frame_{qa_idx}.jpg"
                    cv2.imwrite(str(qa_path), recorded_frames[frame_i])
                print(f"[QA EXTRACT] Saved {num_samples} sample frames to {step_dir}")

            # Update count and print summary
            clip_counts = get_clip_counts(config.EXPERIMENT_STEPS, dataset_path)
            print("\nUpdated Dataset Counts:")
            for s, c in clip_counts.items():
                print(f"  • {s:<18}: {c} clips")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
