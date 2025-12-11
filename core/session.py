from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import time

from .camera import CameraCapture
from .landmarks import LandmarkDetector, FaceMetrics
from .signals import SignalProcessor, SignalConfig, SignalScores
from .state_machine import (
    FocusEvent,
    FocusState,
    FocusStateMachine,
    StateParams,
)


@dataclass
class SessionConfig:
    detect_drowsiness: bool = True
    detect_distraction: bool = True
    target_duration_seconds: Optional[float] = None


@dataclass
class SessionStats:
    start_time: datetime
    end_time: datetime
    duration_seconds: float
    focused_seconds: float
    drowsy_seconds: float
    distracted_seconds: float
    events: List[FocusEvent]


class FocusSession:
    def __init__(
        self,
        config: Optional[SessionConfig] = None,
        state_params: Optional[StateParams] = None,
        signal_config: Optional[SignalConfig] = None,
    ) -> None:
        self.config = config or SessionConfig()
        self.state_machine = FocusStateMachine(state_params)
        self.signal_processor = SignalProcessor(signal_config)
        self.start_time_wall: Optional[datetime] = None
        self.end_time_wall: Optional[datetime] = None
        self.start_time_mono: Optional[float] = None
        self.end_time_mono: Optional[float] = None
        self.last_timestamp: Optional[float] = None
        self.last_state: Optional[FocusState] = None
        self.last_scores: Optional[SignalScores] = None
        self.durations_by_state: Dict[FocusState, float] = {
            state: 0.0 for state in FocusState
        }
        self.events: List[FocusEvent] = []

    def start(self, timestamp: Optional[float] = None) -> None:
        now_mono = time.monotonic() if timestamp is None else timestamp
        self.start_time_wall = datetime.now()
        self.start_time_mono = now_mono
        self.last_timestamp = None
        self.last_state = self.state_machine.state
        self.last_scores = None
        for state in FocusState:
            self.durations_by_state[state] = 0.0
        self.events.clear()

    def update(
        self,
        timestamp: float,
        metrics: FaceMetrics,
    ) -> Tuple[FocusState, SignalScores, Optional[FocusEvent]]:
        if self.start_time_mono is None:
            raise RuntimeError("Session not started")
        if self.last_timestamp is not None and self.last_state is not None:
            dt = max(0.0, timestamp - self.last_timestamp)
            self.durations_by_state[self.last_state] += dt

        raw_scores = self.signal_processor.update(
            timestamp=timestamp,
            face_present=metrics.face_present,
            left_eye_ear=metrics.left_eye_ear,
            right_eye_ear=metrics.right_eye_ear,
        )
        scores = self._apply_detection_mask(raw_scores)
        state, event = self.state_machine.update(timestamp, scores)
        if event is not None:
            self.events.append(event)

        self.last_timestamp = timestamp
        self.last_state = state
        self.last_scores = scores
        return state, scores, event

    def end(self, timestamp: Optional[float] = None) -> None:
        if self.start_time_mono is None:
            raise RuntimeError("Session not started")
        now_mono = time.monotonic() if timestamp is None else timestamp
        if self.last_timestamp is not None and self.last_state is not None:
            dt = max(0.0, now_mono - self.last_timestamp)
            self.durations_by_state[self.last_state] += dt
        self.end_time_mono = now_mono
        self.end_time_wall = datetime.now()

    def summary(self) -> SessionStats:
        if (
            self.start_time_wall is None
            or self.end_time_wall is None
            or self.start_time_mono is None
            or self.end_time_mono is None
        ):
            raise RuntimeError("Session has not been fully completed")
        duration_seconds = max(0.0, self.end_time_mono - self.start_time_mono)
        focused_seconds = self.durations_by_state.get(FocusState.FOCUSED, 0.0)
        drowsy_seconds = (
            self.durations_by_state.get(FocusState.DROWSY_WARNING, 0.0)
            + self.durations_by_state.get(FocusState.DROWSY_ALARM, 0.0)
        )
        distracted_seconds = (
            self.durations_by_state.get(FocusState.DISTRACTED_WARNING, 0.0)
            + self.durations_by_state.get(FocusState.DISTRACTED_ALARM, 0.0)
        )
        return SessionStats(
            start_time=self.start_time_wall,
            end_time=self.end_time_wall,
            duration_seconds=duration_seconds,
            focused_seconds=focused_seconds,
            drowsy_seconds=drowsy_seconds,
            distracted_seconds=distracted_seconds,
            events=list(self.events),
        )

    def _apply_detection_mask(self, scores: SignalScores) -> SignalScores:
        d_score = scores.drowsiness_score if self.config.detect_drowsiness else 0.0
        dis_score = scores.distraction_score if self.config.detect_distraction else 0.0
        return SignalScores(
            drowsiness_score=d_score,
            distraction_score=dis_score,
            blink_rate_per_minute=scores.blink_rate_per_minute,
            long_blink_count=scores.long_blink_count,
            fraction_face_missing=scores.fraction_face_missing,
        )


def _console_session_demo() -> None:
    session = FocusSession(
        SessionConfig(
            detect_drowsiness=True,
            detect_distraction=True,
            target_duration_seconds=60.0,
        ),
        StateParams(),
        SignalConfig(),
    )
    detector = LandmarkDetector()
    last_metrics: Optional[FaceMetrics] = None

    def on_frame(frame: np.ndarray) -> None:
        nonlocal last_metrics
        last_metrics = detector.process(frame)

    with CameraCapture() as camera:
        camera.frame_callback = on_frame
        session.start()
        try:
            while True:
                now = time.monotonic()
                if last_metrics is not None:
                    state, scores, event = session.update(now, last_metrics)
                    line = (
                        f"\rState: {state.name:<18} "
                        f"Drowsiness: {scores.drowsiness_score:6.2f} "
                        f"Distraction: {scores.distraction_score:6.2f}"
                    )
                    print(line, end="", flush=True)
                    if event is not None:
                        print()
                        print(
                            f"Event {event.event_type.name} "
                            f"from {event.from_state.name} to {event.to_state.name}"
                        )
                if (
                    session.config.target_duration_seconds is not None
                    and session.start_time_mono is not None
                    and now - session.start_time_mono
                    >= session.config.target_duration_seconds
                ):
                    break
                time.sleep(0.05)
        except KeyboardInterrupt:
            pass
        finally:
            session.end()
            detector.close()
            print()

    stats = session.summary()
    print("Session summary")
    print(f"Start:       {stats.start_time}")
    print(f"End:         {stats.end_time}")
    print(f"Duration:    {stats.duration_seconds:.1f} s")
    print(f"Focused:     {stats.focused_seconds:.1f} s")
    print(f"Drowsy:      {stats.drowsy_seconds:.1f} s")
    print(f"Distracted:  {stats.distracted_seconds:.1f} s")
    print(f"Events:      {len(stats.events)}")


if __name__ == "__main__":
    _console_session_demo()
