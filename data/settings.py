from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass
class AppSettings:
    sensitivity_profile: str = "Balanced"
    alarm_volume: float = 1.0
    drowsy_alarm_enabled: bool = True
    distracted_alarm_enabled: bool = True
    show_camera_preview: bool = True


SETTINGS_FILE = Path(__file__).resolve().parent / "settings.json"


def load_settings() -> AppSettings:
    defaults = AppSettings()
    if not SETTINGS_FILE.exists():
        return defaults
    try:
        data = json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
    except Exception:
        return defaults
    return AppSettings(
        sensitivity_profile=str(data.get("sensitivity_profile", defaults.sensitivity_profile)),
        alarm_volume=float(data.get("alarm_volume", defaults.alarm_volume)),
        drowsy_alarm_enabled=bool(data.get("drowsy_alarm_enabled", defaults.drowsy_alarm_enabled)),
        distracted_alarm_enabled=bool(data.get("distracted_alarm_enabled", defaults.distracted_alarm_enabled)),
        show_camera_preview=bool(data.get("show_camera_preview", defaults.show_camera_preview)),
    )


def save_settings(settings: AppSettings) -> None:
    SETTINGS_FILE.write_text(json.dumps(asdict(settings), indent=2), encoding="utf-8")
