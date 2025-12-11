from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Optional

import time

import numpy as np

from .camera import CameraCapture
from .landmarks import LandmarkDetector, FaceMetrics


@dataclass
class SignalConfig:
    window_duration_seconds: float = 20.0
    eye_aspect_ratio_threshold: float = 0.22
    minimum_blink_duration_seconds: float = 0.1
    minimum_long_blink_duration_seconds: float = 0.4
    baseline_blink_rate_per_minute: float = 20.0
    drowsiness_long_blink_weight: float = 35.0
    drowsiness_blink_rate_weight: float = 1.5
    distraction_face_missing_weight: float = 100.0


@dataclass
class SignalScores:
    drowsiness_score: float
    distraction_score: float
    blink_rate_per_minute: float
    long_blink_count: int
    fraction_face_missing: float


class SignalProcessor:
    def __init__(self, config: Optional[SignalConfig] = None) -> None:
        self.config = config or SignalConfig()
        self.sample_timestamps: Deque[float] = deque()
        self.sample_face_present_flags: Deque[bool] = deque()
        self.sample_count = 0
        self.sample_count_face_present = 0
        self.blink_event_timestamps: Deque[float] = deque()
        self.long_blink_event_timestamps: Deque[float] = deque()
        self.current_blink_active = False
        self.current_blink_start_timestamp: Optional[float] = None

    def update(
        self,
        timestamp: float,
        face_present: bool,
        left_eye_ear: Optional[float],
        right_eye_ear: Optional[float],
    ) -> SignalScores:
        self._prune_old(timestamp)
        self._update_samples(timestamp, face_present)
        minimum_eye_aspect_ratio: Optional[float] = None
        if (
            face_present
            and left_eye_ear is not None
            and right_eye_ear is not None
        ):
            minimum_eye_aspect_ratio = min(left_eye_ear, right_eye_ear)
        self._update_blinks(timestamp, minimum_eye_aspect_ratio)
        return self._compute_scores()

    def _update_samples(self, timestamp: float, face_present: bool) -> None:
        self.sample_timestamps.append(timestamp)
        self.sample_face_present_flags.append(face_present)
        self.sample_count += 1
        if face_present:
            self.sample_count_face_present += 1

    def _prune_old(self, timestamp: float) -> None:
        cutoff = timestamp - self.config.window_duration_seconds
        while self.sample_timestamps and self.sample_timestamps[0] < cutoff:
            old_face_present = self.sample_face_present_flags.popleft()
            self.sample_timestamps.popleft()
            self.sample_count -= 1
            if old_face_present:
                self.sample_count_face_present -= 1
        while self.blink_event_timestamps and self.blink_event_timestamps[0] < cutoff:
            self.blink_event_timestamps.popleft()
        while (
            self.long_blink_event_timestamps
            and self.long_blink_event_timestamps[0] < cutoff
        ):
            self.long_blink_event_timestamps.popleft()

    def _update_blinks(
        self,
        timestamp: float,
        minimum_eye_aspect_ratio: Optional[float],
    ) -> None:
        eye_closed = (
            minimum_eye_aspect_ratio is not None
            and minimum_eye_aspect_ratio < self.config.eye_aspect_ratio_threshold
        )
        if eye_closed and not self.current_blink_active:
            self.current_blink_active = True
            self.current_blink_start_timestamp = timestamp
        elif not eye_closed and self.current_blink_active:
            assert self.current_blink_start_timestamp is not None
            blink_duration = timestamp - self.current_blink_start_timestamp
            if blink_duration >= self.config.minimum_blink_duration_seconds:
                self.blink_event_timestamps.append(timestamp)
                if (
                    blink_duration
                    >= self.config.minimum_long_blink_duration_seconds
                ):
                    self.long_blink_event_timestamps.append(timestamp)
            self.current_blink_active = False
            self.current_blink_start_timestamp = None

    def _compute_scores(self) -> SignalScores:
        if self.sample_count <= 0:
            return SignalScores(
                drowsiness_score=0.0,
                distraction_score=0.0,
                blink_rate_per_minute=0.0,
                long_blink_count=0,
                fraction_face_missing=0.0,
            )
        if self.sample_count >= 2:
            window_span = self.sample_timestamps[-1] - self.sample_timestamps[0]
            if window_span <= 0:
                window_span = self.config.window_duration_seconds
        else:
            window_span = self.config.window_duration_seconds
        window_span = min(window_span, self.config.window_duration_seconds)
        if window_span <= 0:
            window_span = self.config.window_duration_seconds

        blink_count = len(self.blink_event_timestamps)
        long_blink_count = len(self.long_blink_event_timestamps)

        if blink_count > 0:
            blink_rate_per_minute = blink_count * 60.0 / window_span
        else:
            blink_rate_per_minute = 0.0

        if self.sample_count > 0:
            fraction_face_missing = 1.0 - (
                self.sample_count_face_present / float(self.sample_count)
            )
        else:
            fraction_face_missing = 0.0

        drowsiness_from_long_blinks = (
            self.config.drowsiness_long_blink_weight * long_blink_count
        )
        excess_blink_rate = max(
            0.0, blink_rate_per_minute - self.config.baseline_blink_rate_per_minute
        )
        drowsiness_from_blink_rate = (
            self.config.drowsiness_blink_rate_weight * excess_blink_rate
        )
        drowsiness_score = drowsiness_from_long_blinks + drowsiness_from_blink_rate
        distraction_score = (
            self.config.distraction_face_missing_weight * fraction_face_missing
        )
        drowsiness_score = max(0.0, min(100.0, drowsiness_score))
        distraction_score = max(0.0, min(100.0, distraction_score))

        return SignalScores(
            drowsiness_score=drowsiness_score,
            distraction_score=distraction_score,
            blink_rate_per_minute=blink_rate_per_minute,
            long_blink_count=long_blink_count,
            fraction_face_missing=fraction_face_missing,
        )


def _demo_loop() -> None:
    signal_processor = SignalProcessor()
    detector = LandmarkDetector()
    last_frame: Optional[np.ndarray] = None

    def on_frame(frame: np.ndarray) -> None:
        nonlocal last_frame
        last_frame = frame.copy()

    with CameraCapture() as camera:
        camera.frame_callback = on_frame
        try:
            while True:
                now = time.monotonic()
                if last_frame is not None:
                    metrics = detector.process(last_frame)
                    scores = signal_processor.update(
                        timestamp=now,
                        face_present=metrics.face_present,
                        left_eye_ear=metrics.left_eye_ear,
                        right_eye_ear=metrics.right_eye_ear,
                    )
                    print(
                        f"\rDrowsiness: {scores.drowsiness_score:6.2f}  "
                        f"Distraction: {scores.distraction_score:6.2f}  "
                        f"Blink rate: {scores.blink_rate_per_minute:6.2f}  ",
                        end="",
                        flush=True,
                    )
                time.sleep(0.05)
        except KeyboardInterrupt:
            pass
        finally:
            detector.close()
            print()


if __name__ == "__main__":
    _demo_loop()
