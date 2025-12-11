from __future__ import annotations

from typing import Optional

from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from data.settings import AppSettings


class SettingsDialog(QDialog):
    def __init__(self, current_settings: AppSettings, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._current_settings = current_settings
        self.result_settings: Optional[AppSettings] = None
        self.setWindowTitle("Settings")
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout()

        profile_layout = QHBoxLayout()
        profile_label = QLabel("Default sensitivity profile:")
        self.profile_combo = QComboBox()
        self.profile_combo.addItem("Balanced")
        self.profile_combo.addItem("Chill")
        self.profile_combo.addItem("Strict")
        index = self.profile_combo.findText(self._current_settings.sensitivity_profile)
        if index >= 0:
            self.profile_combo.setCurrentIndex(index)
        profile_layout.addWidget(profile_label)
        profile_layout.addWidget(self.profile_combo)
        layout.addLayout(profile_layout)

        volume_layout = QHBoxLayout()
        volume_label = QLabel("Alarm volume:")
        self.volume_spin = QSpinBox()
        self.volume_spin.setRange(0, 100)
        self.volume_spin.setSingleStep(5)
        self.volume_spin.setValue(int(round(self._current_settings.alarm_volume * 100)))
        volume_layout.addWidget(volume_label)
        volume_layout.addWidget(self.volume_spin)
        layout.addLayout(volume_layout)

        self.drowsy_enabled_checkbox = QCheckBox("Enable drowsy alarm sound")
        self.drowsy_enabled_checkbox.setChecked(self._current_settings.drowsy_alarm_enabled)
        layout.addWidget(self.drowsy_enabled_checkbox)

        self.distracted_enabled_checkbox = QCheckBox("Enable distracted alarm sound")
        self.distracted_enabled_checkbox.setChecked(self._current_settings.distracted_alarm_enabled)
        layout.addWidget(self.distracted_enabled_checkbox)

        self.preview_checkbox = QCheckBox("Show camera preview")
        self.preview_checkbox.setChecked(self._current_settings.show_camera_preview)
        layout.addWidget(self.preview_checkbox)

        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        button_box.accepted.connect(self._on_accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

        self.setLayout(layout)

    def _on_accept(self) -> None:
        profile = self.profile_combo.currentText()
        volume = max(0, min(100, self.volume_spin.value())) / 100.0
        drowsy_enabled = self.drowsy_enabled_checkbox.isChecked()
        distracted_enabled = self.distracted_enabled_checkbox.isChecked()
        preview = self.preview_checkbox.isChecked()
        self.result_settings = AppSettings(
            sensitivity_profile=profile,
            alarm_volume=volume,
            drowsy_alarm_enabled=drowsy_enabled,
            distracted_alarm_enabled=distracted_enabled,
            show_camera_preview=preview,
        )
        self.accept()
