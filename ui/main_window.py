from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
from PyQt6.QtCore import QTimer, Qt, QUrl
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtMultimedia import QSoundEffect
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from core.camera import CameraCapture
from core.landmarks import LandmarkDetector
from core.session import FocusSession, SessionConfig, SessionStats
from core.signals import SignalConfig
from core.state_machine import StateParams, FocusState
from core.alarms import AlarmKind
from .calibration import CalibrationDialog, CalibrationResult
from .history import HistoryDialog
from .summary import SummaryDialog


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("WakeGuard")
        self._camera: Optional[CameraCapture] = None
        self._detector: Optional[LandmarkDetector] = None
        self._session: Optional[FocusSession] = None
        self._update_timer: Optional[QTimer] = None
        self._last_frame: Optional[np.ndarray] = None
        self._calibrated_eye_threshold: Optional[float] = None
        self._drowsy_sound: Optional[QSoundEffect] = None
        self._distracted_sound: Optional[QSoundEffect] = None
        self._active_alarm_kind: Optional[AlarmKind] = None
        self._last_score_label_update_mono: float = 0.0

        self._build_ui()
        self._init_sounds()

    def _build_ui(self) -> None:
        central = QWidget(self)
        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        title_label = QLabel("WakeGuard")
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        font = title_label.font()
        font.setPointSize(18)
        font.setBold(True)
        title_label.setFont(font)
        layout.addWidget(title_label)

        duration_layout = QHBoxLayout()
        duration_label = QLabel("Session duration (minutes, 0 = unlimited):")
        self.duration_spin = QSpinBox()
        self.duration_spin.setRange(0, 240)
        self.duration_spin.setValue(25)
        duration_layout.addWidget(duration_label)
        duration_layout.addWidget(self.duration_spin)
        layout.addLayout(duration_layout)

        options_layout = QHBoxLayout()
        self.detect_drowsiness_checkbox = QCheckBox("Detect drowsiness")
        self.detect_drowsiness_checkbox.setChecked(True)
        self.detect_distraction_checkbox = QCheckBox("Detect distraction")
        self.detect_distraction_checkbox.setChecked(True)
        options_layout.addWidget(self.detect_drowsiness_checkbox)
        options_layout.addWidget(self.detect_distraction_checkbox)
        layout.addLayout(options_layout)

        sensitivity_layout = QHBoxLayout()
        sensitivity_label = QLabel("Sensitivity profile:")
        self.sensitivity_combo = QComboBox()
        self.sensitivity_combo.addItem("Balanced")
        self.sensitivity_combo.addItem("Chill")
        self.sensitivity_combo.addItem("Strict")
        self.sensitivity_combo.setCurrentIndex(0)
        sensitivity_layout.addWidget(sensitivity_label)
        sensitivity_layout.addWidget(self.sensitivity_combo)
        layout.addLayout(sensitivity_layout)

        calibration_layout = QHBoxLayout()
        self.calibrate_button = QPushButton("Calibrate EAR")
        self.calibration_status_label = QLabel("Calibration: not set")
        calibration_layout.addWidget(self.calibrate_button)
        calibration_layout.addWidget(self.calibration_status_label)
        layout.addLayout(calibration_layout)

        session_buttons_layout = QHBoxLayout()
        self.start_button = QPushButton("Start Session")
        self.history_button = QPushButton("Session history")
        session_buttons_layout.addWidget(self.start_button)
        session_buttons_layout.addWidget(self.history_button)
        layout.addLayout(session_buttons_layout)

        self.start_button.clicked.connect(self._on_start_stop_clicked)
        self.calibrate_button.clicked.connect(self._on_calibrate_clicked)
        self.history_button.clicked.connect(self._on_history_clicked)

        status_group = QVBoxLayout()
        self.state_label = QLabel("State: idle")
        self.drowsiness_label = QLabel("Drowsiness: 0")
        self.distraction_label = QLabel("Distraction: 0")
        status_group.addWidget(self.state_label)
        status_group.addWidget(self.drowsiness_label)
        status_group.addWidget(self.distraction_label)
        layout.addLayout(status_group)

        video_label_title = QLabel("Camera preview")
        layout.addWidget(video_label_title)
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setFixedSize(320, 240)
        layout.addWidget(self.video_label)

        central.setLayout(layout)
        self.setCentralWidget(central)
        self.resize(560, 540)

    def _init_sounds(self) -> None:
        base_dir = Path(__file__).resolve().parent.parent
        assets_dir = base_dir / "assets"
        drowsy_path = assets_dir / "drowsy_alarm.wav"
        distracted_path = assets_dir / "distracted_alarm.wav"

        if drowsy_path.exists():
            sound = QSoundEffect(self)
            sound.setSource(QUrl.fromLocalFile(str(drowsy_path)))
            sound.setLoopCount(1)
            sound.setVolume(1.0)
            self._drowsy_sound = sound

        if distracted_path.exists():
            sound = QSoundEffect(self)
            sound.setSource(QUrl.fromLocalFile(str(distracted_path)))
            sound.setLoopCount(1)
            sound.setVolume(1.0)
            self._distracted_sound = sound

    def _on_start_stop_clicked(self) -> None:
        if self._session is None:
            self._start_session()
        else:
            self._stop_session(manual=True)

    def _on_calibrate_clicked(self) -> None:
        if self._session is not None:
            msg = QMessageBox(self)
            msg.setWindowTitle("Calibration not available")
            msg.setText("Please end the current session before calibrating.")
            msg.exec()
            return

        dlg = CalibrationDialog(self)
        result_code = dlg.exec()
        if result_code == int(QDialog.DialogCode.Accepted) and dlg.calibration_result is not None:
            result: CalibrationResult = dlg.calibration_result
            self._calibrated_eye_threshold = result.ear_threshold
            self.calibration_status_label.setText(
                f"Calibration: EAR threshold {result.ear_threshold:.3f}"
            )

    def _on_history_clicked(self) -> None:
        if self._session is not None:
            msg = QMessageBox(self)
            msg.setWindowTitle("History available after session")
            msg.setText("Please end the current session before opening history.")
            msg.exec()
            return
        dlg = HistoryDialog(self)
        dlg.exec()

    def _start_session(self) -> None:
        detect_drowsiness = self.detect_drowsiness_checkbox.isChecked()
        detect_distraction = self.detect_distraction_checkbox.isChecked()

        minutes = self.duration_spin.value()
        if minutes <= 0:
            target_duration = None
        else:
            target_duration = float(minutes * 60)

        state_params, signal_config = self._params_for_sensitivity()

        self._session = FocusSession(
            SessionConfig(
                detect_drowsiness=detect_drowsiness,
                detect_distraction=detect_distraction,
                target_duration_seconds=target_duration,
            ),
            state_params,
            signal_config,
        )
        self._session.start()

        self._detector = LandmarkDetector()
        self._camera = CameraCapture()
        self._camera.frame_callback = self._on_camera_frame
        self._camera.start()

        self._active_alarm_kind = None
        self._stop_all_alarms()
        self._last_score_label_update_mono = 0.0

        self._update_timer = QTimer(self)
        self._update_timer.timeout.connect(self._on_update_timer)
        self._update_timer.start(50)

        self.start_button.setText("End Session")
        self.detect_drowsiness_checkbox.setEnabled(False)
        self.detect_distraction_checkbox.setEnabled(False)
        self.duration_spin.setEnabled(False)
        self.sensitivity_combo.setEnabled(False)
        self.calibrate_button.setEnabled(False)
        self.history_button.setEnabled(False)
        self.state_label.setText("State: running...")
        self.state_label.setStyleSheet("")
        self.drowsiness_label.setText("Drowsiness: 0")
        self.distraction_label.setText("Distraction: 0")

    def _stop_session(self, manual: bool) -> None:
        if self._update_timer is not None:
            self._update_timer.stop()
            self._update_timer = None

        if self._session is not None:
            self._session.end()
            session = self._session
        else:
            session = None

        if self._camera is not None:
            self._camera.stop()
            self._camera = None

        if self._detector is not None:
            self._detector.close()
            self._detector = None

        self._session = None
        self._last_frame = None
        self._active_alarm_kind = None
        self._stop_all_alarms()
        self.video_label.clear()

        self.start_button.setText("Start Session")
        self.detect_drowsiness_checkbox.setEnabled(True)
        self.detect_distraction_checkbox.setEnabled(True)
        self.duration_spin.setEnabled(True)
        self.sensitivity_combo.setEnabled(True)
        self.calibrate_button.setEnabled(True)
        self.history_button.setEnabled(True)
        self.state_label.setText("State: idle")
        self.state_label.setStyleSheet("")
        self.drowsiness_label.setText("Drowsiness: 0")
        self.distraction_label.setText("Distraction: 0")

        if session is not None:
            stats = session.summary()
            session_id = session.save_to_db()
            self._show_summary_dialog(stats, session_id, manual)

    def _on_camera_frame(self, frame: np.ndarray) -> None:
        self._last_frame = frame.copy()

    def _on_update_timer(self) -> None:
        if self._session is None or self._detector is None:
            return
        if self._last_frame is None:
            return

        now = time.monotonic()
        metrics = self._detector.process(self._last_frame)
        state, scores, event = self._session.update(now, metrics)

        self._update_state_label(state)

        if self._last_score_label_update_mono == 0.0 or now - self._last_score_label_update_mono >= 1.0:
            self._last_score_label_update_mono = now
            self._update_score_labels(scores)

        self._update_video_preview(self._last_frame)
        self._update_alarm_for_state(state)

        config = self._session.config
        if (
            config.target_duration_seconds is not None
            and self._session.start_time_mono is not None
            and now - self._session.start_time_mono >= config.target_duration_seconds
        ):
            self._stop_session(manual=False)

    def _update_state_label(self, state: FocusState) -> None:
        self.state_label.setText(f"State: {state.name}")
        if state == FocusState.FOCUSED:
            self.state_label.setStyleSheet("color: #008800; font-weight: bold;")
        elif state.name.startswith("DROWSY"):
            self.state_label.setStyleSheet("color: #cc7a00; font-weight: bold;")
        elif state.name.startswith("DISTRACTED"):
            self.state_label.setStyleSheet("color: #cc0000; font-weight: bold;")
        else:
            self.state_label.setStyleSheet("")

    def _update_score_labels(self, scores) -> None:
        d = int(round(scores.drowsiness_score))
        c = int(round(scores.distraction_score))
        self.drowsiness_label.setText(f"Drowsiness: {d}")
        self.distraction_label.setText(f"Distraction: {c}")

    def _update_video_preview(self, frame: np.ndarray) -> None:
        h, w, ch = frame.shape
        bytes_per_line = ch * w
        image = QImage(
            frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888
        )
        pixmap = QPixmap.fromImage(image)
        pixmap = pixmap.scaled(
            self.video_label.width(),
            self.video_label.height(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.video_label.setPixmap(pixmap)

    def _update_alarm_for_state(self, state: FocusState) -> None:
        if state == FocusState.FOCUSED:
            if self._active_alarm_kind is not None:
                self._stop_all_alarms()
                self._active_alarm_kind = None
            return

        desired_kind: Optional[AlarmKind] = None
        if state == FocusState.DROWSY_ALARM:
            desired_kind = AlarmKind.DROWSY
        elif state == FocusState.DISTRACTED_ALARM:
            desired_kind = AlarmKind.DISTRACTED

        if desired_kind is None:
            return

        if self._active_alarm_kind != desired_kind:
            self._stop_all_alarms()
            self._active_alarm_kind = desired_kind

        if self._active_alarm_kind is not None:
            self._play_alarm(self._active_alarm_kind)

    def _play_alarm(self, kind: AlarmKind) -> None:
        if kind == AlarmKind.DROWSY and self._drowsy_sound is not None:
            if not self._drowsy_sound.isPlaying():
                self._drowsy_sound.setLoopCount(1)
                self._drowsy_sound.play()
            return
        if kind == AlarmKind.DISTRACTED and self._distracted_sound is not None:
            if not self._distracted_sound.isPlaying():
                self._distracted_sound.setLoopCount(1)
                self._distracted_sound.play()
            return
        QApplication.beep()

    def _stop_all_alarms(self) -> None:
        if self._drowsy_sound is not None:
            self._drowsy_sound.stop()
        if self._distracted_sound is not None:
            self._distracted_sound.stop()

    def _params_for_sensitivity(self) -> tuple[StateParams, SignalConfig]:
        profile = self.sensitivity_combo.currentText()
        if profile == "Chill":
            state_params = StateParams(
                drowsy_warning_threshold=70.0,
                drowsy_alarm_threshold=85.0,
                distracted_warning_threshold=70.0,
                distracted_alarm_threshold=85.0,
                min_drowsy_warning_duration=4.0,
                min_drowsy_alarm_duration=3.0,
                min_distracted_warning_duration=4.0,
                min_distracted_alarm_duration=3.0,
                drowsy_recovery_threshold=70.0,
                distracted_recovery_threshold=70.0,
                recovery_grace_duration=0.0,
            )
            signal_config = SignalConfig(
                window_duration_seconds=25.0,
                eye_aspect_ratio_threshold=0.21,
                minimum_blink_duration_seconds=0.12,
                minimum_long_blink_duration_seconds=0.45,
                baseline_blink_rate_per_minute=18.0,
                drowsiness_long_blink_weight=30.0,
                drowsiness_blink_rate_weight=1.2,
                distraction_face_missing_weight=80.0,
            )
        elif profile == "Strict":
            state_params = StateParams(
                drowsy_warning_threshold=50.0,
                drowsy_alarm_threshold=70.0,
                distracted_warning_threshold=50.0,
                distracted_alarm_threshold=70.0,
                min_drowsy_warning_duration=2.0,
                min_drowsy_alarm_duration=1.0,
                min_distracted_warning_duration=2.0,
                min_distracted_alarm_duration=1.0,
                drowsy_recovery_threshold=50.0,
                distracted_recovery_threshold=50.0,
                recovery_grace_duration=0.0,
            )
            signal_config = SignalConfig(
                window_duration_seconds=15.0,
                eye_aspect_ratio_threshold=0.23,
                minimum_blink_duration_seconds=0.08,
                minimum_long_blink_duration_seconds=0.35,
                baseline_blink_rate_per_minute=22.0,
                drowsiness_long_blink_weight=40.0,
                drowsiness_blink_rate_weight=2.0,
                distraction_face_missing_weight=120.0,
            )
        else:
            state_params = StateParams(
                drowsy_warning_threshold=60.0,
                drowsy_alarm_threshold=80.0,
                distracted_warning_threshold=60.0,
                distracted_alarm_threshold=80.0,
                min_drowsy_warning_duration=2.0,
                min_drowsy_alarm_duration=1.5,
                min_distracted_warning_duration=2.0,
                min_distracted_alarm_duration=1.5,
                drowsy_recovery_threshold=60.0,
                distracted_recovery_threshold=60.0,
                recovery_grace_duration=0.0,
            )
            signal_config = SignalConfig(
                window_duration_seconds=10.0,
                eye_aspect_ratio_threshold=0.22,
                minimum_blink_duration_seconds=0.10,
                minimum_long_blink_duration_seconds=0.40,
                baseline_blink_rate_per_minute=20.0,
                drowsiness_long_blink_weight=35.0,
                drowsiness_blink_rate_weight=1.5,
                distraction_face_missing_weight=100.0,
            )

        if self._calibrated_eye_threshold is not None:
            signal_config.eye_aspect_ratio_threshold = self._calibrated_eye_threshold

        return state_params, signal_config

    def _show_summary_dialog(
        self,
        stats: SessionStats,
        session_id: int,
        manual: bool,
    ) -> None:
        dlg = SummaryDialog(stats, session_id, manual, self)
        dlg.exec()

    def closeEvent(self, event) -> None:
        if self._update_timer is not None:
            self._update_timer.stop()
        if self._camera is not None:
            self._camera.stop()
        if self._detector is not None:
            self._detector.close()
        self._session = None
        self._update_timer = None
        self._camera = None
        self._detector = None
        self._last_frame = None
        self._active_alarm_kind = None
        self._stop_all_alarms()
        super().closeEvent(event)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
