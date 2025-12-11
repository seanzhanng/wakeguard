from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional, Tuple

import time
import numpy as np

from .camera import CameraCapture
from .landmarks import LandmarkDetector, FaceMetrics
from .signals import SignalProcessor, SignalConfig, SignalScores


class FocusState(Enum):
    FOCUSED = auto()
    DROWSY_WARNING = auto()
    DROWSY_ALARM = auto()
    DISTRACTED_WARNING = auto()
    DISTRACTED_ALARM = auto()


class FocusEventType(Enum):
    ENTER_FOCUSED = auto()
    ENTER_DROWSY_WARNING = auto()
    ENTER_DROWSY_ALARM = auto()
    ENTER_DISTRACTED_WARNING = auto()
    ENTER_DISTRACTED_ALARM = auto()


@dataclass
class FocusEvent:
    timestamp: float
    event_type: FocusEventType
    from_state: FocusState
    to_state: FocusState
    drowsiness_score: float
    distraction_score: float


@dataclass
class StateParams:
    drowsy_warning_threshold: float = 60.0
    drowsy_alarm_threshold: float = 80.0
    distracted_warning_threshold: float = 60.0
    distracted_alarm_threshold: float = 80.0
    min_drowsy_warning_duration: float = 3.0
    min_drowsy_alarm_duration: float = 2.0
    min_distracted_warning_duration: float = 3.0
    min_distracted_alarm_duration: float = 2.0
    drowsy_recovery_threshold: float = 40.0
    distracted_recovery_threshold: float = 40.0
    recovery_grace_duration: float = 3.0


class FocusStateMachine:
    def __init__(self, params: Optional[StateParams] = None) -> None:
        self.params = params or StateParams()
        self.state = FocusState.FOCUSED
        self.last_timestamp: Optional[float] = None
        self.time_in_state = 0.0
        self.time_above_drowsy_warning = 0.0
        self.time_above_drowsy_alarm = 0.0
        self.time_above_distracted_warning = 0.0
        self.time_above_distracted_alarm = 0.0
        self.time_below_drowsy_recovery = 0.0
        self.time_below_distracted_recovery = 0.0

    def update(
        self,
        timestamp: float,
        scores: SignalScores,
    ) -> Tuple[FocusState, Optional[FocusEvent]]:
        if self.last_timestamp is None:
            dt = 0.0
        else:
            dt = max(0.0, timestamp - self.last_timestamp)
        self.last_timestamp = timestamp
        self.time_in_state += dt
        self._update_timers(dt, scores)
        new_state = self._compute_next_state(scores)
        event: Optional[FocusEvent] = None
        if new_state != self.state:
            event_type = self._event_type_for_state(new_state)
            event = FocusEvent(
                timestamp=timestamp,
                event_type=event_type,
                from_state=self.state,
                to_state=new_state,
                drowsiness_score=scores.drowsiness_score,
                distraction_score=scores.distraction_score,
            )
            self.state = new_state
            self.time_in_state = 0.0
        return self.state, event

    def _update_timers(self, dt: float, scores: SignalScores) -> None:
        if scores.drowsiness_score >= self.params.drowsy_warning_threshold:
            self.time_above_drowsy_warning += dt
        else:
            self.time_above_drowsy_warning = 0.0
        if scores.drowsiness_score >= self.params.drowsy_alarm_threshold:
            self.time_above_drowsy_alarm += dt
        else:
            self.time_above_drowsy_alarm = 0.0
        if scores.distraction_score >= self.params.distracted_warning_threshold:
            self.time_above_distracted_warning += dt
        else:
            self.time_above_distracted_warning = 0.0
        if scores.distraction_score >= self.params.distracted_alarm_threshold:
            self.time_above_distracted_alarm += dt
        else:
            self.time_above_distracted_alarm = 0.0
        if scores.drowsiness_score <= self.params.drowsy_recovery_threshold:
            self.time_below_drowsy_recovery += dt
        else:
            self.time_below_drowsy_recovery = 0.0
        if scores.distraction_score <= self.params.distracted_recovery_threshold:
            self.time_below_distracted_recovery += dt
        else:
            self.time_below_distracted_recovery = 0.0

    def _compute_next_state(self, scores: SignalScores) -> FocusState:
        p = self.params
        s = self.state
        if s == FocusState.FOCUSED:
            if (
                scores.drowsiness_score >= p.drowsy_warning_threshold
                and self.time_above_drowsy_warning >= p.min_drowsy_warning_duration
            ):
                return FocusState.DROWSY_WARNING
            if (
                scores.distraction_score >= p.distracted_warning_threshold
                and self.time_above_distracted_warning
                >= p.min_distracted_warning_duration
            ):
                return FocusState.DISTRACTED_WARNING
            return FocusState.FOCUSED

        if s == FocusState.DROWSY_WARNING:
            if (
                scores.drowsiness_score >= p.drowsy_alarm_threshold
                and self.time_above_drowsy_alarm >= p.min_drowsy_alarm_duration
            ):
                return FocusState.DROWSY_ALARM
            if (
                scores.drowsiness_score <= p.drowsy_recovery_threshold
                and self.time_below_drowsy_recovery >= p.recovery_grace_duration
            ):
                return FocusState.FOCUSED
            return FocusState.DROWSY_WARNING

        if s == FocusState.DROWSY_ALARM:
            if (
                scores.drowsiness_score <= p.drowsy_recovery_threshold
                and self.time_below_drowsy_recovery >= p.recovery_grace_duration
            ):
                return FocusState.FOCUSED
            return FocusState.DROWSY_ALARM

        if s == FocusState.DISTRACTED_WARNING:
            if (
                scores.distraction_score >= p.distracted_alarm_threshold
                and self.time_above_distracted_alarm
                >= p.min_distracted_alarm_duration
            ):
                return FocusState.DISTRACTED_ALARM
            if (
                scores.distraction_score <= p.distracted_recovery_threshold
                and self.time_below_distracted_recovery >= p.recovery_grace_duration
            ):
                return FocusState.FOCUSED
            return FocusState.DISTRACTED_WARNING

        if s == FocusState.DISTRACTED_ALARM:
            if (
                scores.distraction_score <= p.distracted_recovery_threshold
                and self.time_below_distracted_recovery >= p.recovery_grace_duration
            ):
                return FocusState.FOCUSED
            return FocusState.DISTRACTED_ALARM

        return FocusState.FOCUSED

    @staticmethod
    def _event_type_for_state(state: FocusState) -> FocusEventType:
        if state == FocusState.FOCUSED:
            return FocusEventType.ENTER_FOCUSED
        if state == FocusState.DROWSY_WARNING:
            return FocusEventType.ENTER_DROWSY_WARNING
        if state == FocusState.DROWSY_ALARM:
            return FocusEventType.ENTER_DROWSY_ALARM
        if state == FocusState.DISTRACTED_WARNING:
            return FocusEventType.ENTER_DISTRACTED_WARNING
        return FocusEventType.ENTER_DISTRACTED_ALARM


def _demo_loop() -> None:
    signal_processor = SignalProcessor(SignalConfig())
    detector = LandmarkDetector()
    machine = FocusStateMachine()
    last_metrics: Optional[FaceMetrics] = None

    def on_frame(frame: np.ndarray) -> None:
        nonlocal last_metrics
        last_metrics = detector.process(frame)

    with CameraCapture() as camera:
        camera.frame_callback = on_frame
        try:
            while True:
                now = time.monotonic()
                if last_metrics is not None:
                    metrics = last_metrics
                    scores = signal_processor.update(
                        timestamp=now,
                        face_present=metrics.face_present,
                        left_eye_ear=metrics.left_eye_ear,
                        right_eye_ear=metrics.right_eye_ear,
                    )
                    state, event = machine.update(now, scores)
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
                time.sleep(0.05)
        except KeyboardInterrupt:
            pass
        finally:
            detector.close()
            print()


if __name__ == "__main__":
    _demo_loop()
