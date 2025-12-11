from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Dict, Optional

from .state_machine import FocusEvent, FocusEventType


class AlarmKind(Enum):
    DROWSY = auto()
    DISTRACTED = auto()


@dataclass
class AlarmConfig:
    cooldown_seconds: float = 10.0


class AlarmController:
    def __init__(self, config: Optional[AlarmConfig] = None) -> None:
        self.config = config or AlarmConfig()
        self._last_alarm_time: Dict[AlarmKind, float] = {
            AlarmKind.DROWSY: float("-inf"),
            AlarmKind.DISTRACTED: float("-inf"),
        }

    def process_event(self, event: FocusEvent) -> Optional[AlarmKind]:
        kind = self._alarm_kind_for_event(event)
        if kind is None:
            return None
        last_time = self._last_alarm_time.get(kind, float("-inf"))
        delta = event.timestamp - last_time
        if delta < self.config.cooldown_seconds:
            return None
        self._last_alarm_time[kind] = event.timestamp
        return kind

    @staticmethod
    def _alarm_kind_for_event(event: FocusEvent) -> Optional[AlarmKind]:
        if event.event_type == FocusEventType.ENTER_DROWSY_ALARM:
            return AlarmKind.DROWSY
        if event.event_type == FocusEventType.ENTER_DISTRACTED_ALARM:
            return AlarmKind.DISTRACTED
        return None
