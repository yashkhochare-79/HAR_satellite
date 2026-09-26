"""
Offline Text-to-Speech (TTS) Voice Alert System for Space Station / Lunar BAS Experiments.
Uses pyttsx3 completely offline (zero cloud latency/bandwidth dependence).
Operates asynchronously via a thread-safe Queue and dedicated background worker
to ensure the real-time computer vision processing loop never stalls.
"""

import queue
import threading
import time
from typing import Optional


class VoiceAlertManager:
    """
    Manages non-blocking voice notifications using offline TTS.
    Worker thread pulls messages from a queue and synthesizes speech sequentially.
    """

    def __init__(self, speech_rate: int = 175, volume: float = 0.95):
        self.speech_rate = speech_rate
        self.volume = volume
        self.message_queue: queue.Queue = queue.Queue()
        self._stop_event = threading.Event()
        self._worker_thread: Optional[threading.Thread] = None

        # Start the background worker thread
        self.start()

    def start(self):
        """Starts the background TTS worker thread if not already running."""
        if self._worker_thread is None or not self._worker_thread.is_alive():
            self._stop_event.clear()
            self._worker_thread = threading.Thread(
                target=self._worker_loop,
                name="VoiceAlertWorker",
                daemon=True
            )
            self._worker_thread.start()

    def _worker_loop(self):
        """
        Background worker loop that handles pyttsx3 speech synthesis.
        Note: pyttsx3 and COM/SAPI5 on Windows require initialization on the
        thread where runAndWait() is executed to avoid RPC_E_WRONG_THREAD.
        """
        engine = None
        try:
            import pyttsx3
            engine = pyttsx3.init()
            engine.setProperty("rate", self.speech_rate)
            engine.setProperty("volume", self.volume)
        except Exception as e:
            print(f"[VOICE ALERT WARNING] Offline TTS engine initialization failed: {e}")
            print("[VOICE ALERT] Alerts will fall back to audio simulation / console only.")

        while not self._stop_event.is_set():
            try:
                # Wait for next alert message with a short timeout to check stop_event
                message = self.message_queue.get(timeout=0.3)
            except queue.Empty:
                continue

            if message is None:
                # Sentinel shutdown signal
                self.message_queue.task_done()
                break

            print(f"[AUDIO ALERT SPK] >> '{message}'")

            if engine is not None:
                try:
                    engine.say(message)
                    engine.runAndWait()
                except Exception as ex:
                    print(f"[VOICE ALERT RUN ERROR] {ex}")
            else:
                # Fallback if pyttsx3 isn't available
                time.sleep(0.5)

            self.message_queue.task_done()

        if engine is not None:
            try:
                engine.stop()
            except Exception:
                pass

    def speak_alert(self, message: str):
        """
        Enqueues an alert message to be spoken. Non-blocking call.
        """
        if message and not self._stop_event.is_set():
            self.message_queue.put(message)

    def shutdown(self, timeout: float = 2.0):
        """
        Gracefully terminates the background worker thread.
        """
        self._stop_event.set()
        self.message_queue.put(None)  # Unblock get()
        if self._worker_thread is not None and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=timeout)
        print("[VOICE ALERT] Voice alert worker stopped.")


# Global singleton instance
voice_alert = VoiceAlertManager()


def speak_alert(message: str):
    """Convenience functional wrapper for singleton."""
    voice_alert.speak_alert(message)


def shutdown():
    """Convenience shutdown wrapper for singleton."""
    voice_alert.shutdown()


if __name__ == "__main__":
    print("=" * 60)
    print("SMOKE TEST: VoiceAlertManager (Offline TTS)")
    print("=" * 60)

    # Test enqueueing multiple messages back-to-back without blocking
    print("Enqueueing Message 1...")
    voice_alert.speak_alert("Warning: unexpected step. Expected pick red box.")
    print("Enqueueing Message 2...")
    voice_alert.speak_alert("Step complete. Next step: open red box.")

    # Allow worker thread time to process both messages
    print("Waiting for queue to drain...")
    voice_alert.message_queue.join()
    print("Queue successfully drained.")

    voice_alert.shutdown()
    print("Smoke test completed.")
