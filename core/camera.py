from __future__ import annotations

import threading
import time
from typing import Callable, Optional

import cv2
import numpy as np

Frame = np.ndarray
FrameCallback = Callable[[Frame], None]


class CameraCapture:
    def __init__(self, device_index: int = 0, target_fps: float = 20) -> None:
        self.device_index = device_index
        self.target_fps = target_fps
        self._capture: Optional[cv2.VideoCapture] = None
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self.frame_callback: Optional[FrameCallback] = None

    def start(self) -> None:
        if self._running:
            return
        capture = cv2.VideoCapture(self.device_index)
        if not capture.isOpened():
            capture.release()
            raise RuntimeError("Unable to open camera")
        self._capture = capture
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        if self._capture is not None:
            self._capture.release()
            self._capture = None

    def _run(self) -> None:
        assert self._capture is not None
        delay = 1.0 / self.target_fps if self.target_fps > 0 else 0.0
        while self._running:
            ret, frame = self._capture.read()
            if not ret:
                continue
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if self.frame_callback is not None:
                self.frame_callback(frame_rgb)
            if delay > 0:
                time.sleep(delay)

    def __enter__(self) -> CameraCapture:
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.stop()


def _preview_loop() -> None:
    window_name = "WakeGuard Camera Preview"
    last_frame: Optional[Frame] = None

    def on_frame(frame: Frame) -> None:
        nonlocal last_frame
        last_frame = frame

    with CameraCapture() as camera:
        camera.frame_callback = on_frame
        while True:
            if last_frame is not None:
                frame_bgr = cv2.cvtColor(last_frame, cv2.COLOR_RGB2BGR)
                cv2.imshow(window_name, frame_bgr)
            if cv2.waitKey(1) & 0xFF == 27:
                break
        cv2.destroyAllWindows()


if __name__ == "__main__":
    _preview_loop()
