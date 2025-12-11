from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, List

import numpy as np
from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QProgressBar,
    QVBoxLayout,
    QWidget,
    QMessageBox,
)

from core.camera import CameraCapture
from core.landmarks import LandmarkDetector, FaceMetrics


@dataclass
class CalibrationResult:
    ear_open: float
    ear_closed: float
    ear_threshold: float


class CalibrationDialog(QDialog):
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("EAR calibration")
        self._camera: Optional[CameraCapture] = None
        self._detector: Optional[LandmarkDetector] = None
        self._update_timer: Optional[QTimer] = None
        self._last_frame: Optional[np.ndarray] = None

        self._phase = "idle"
        self._phase_start_mono: Optional[float] = None
        self._open_samples: List[float] = []
        self._closed_samples: List[float] = []
        self._open_duration_seconds = 3.0
        self._closed_duration_seconds = 3.0

        self.calibration_result: Optional[CalibrationResult] = None

        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout()

        self.instructions_label = QLabel(
            "Step 1: click Start and look at the screen with eyes open."
        )
        self.instructions_label.setWordWrap(True)
        layout.addWidget(self.instructions_label)

        self.status_label = QLabel("Waiting to start")
        layout.addWidget(self.status_label)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)

        buttons_layout = QHBoxLayout()
        self.start_button = QPushButton("Start")
        cancel_button = QPushButton("Cancel")
        buttons_layout.addWidget(self.start_button)
        buttons_layout.addWidget(cancel_button)
        layout.addLayout(buttons_layout)

        self.setLayout(layout)
        self.resize(420, 200)

        self.start_button.clicked.connect(self._on_start_clicked)
        cancel_button.clicked.connect(self._on_cancel_clicked)

    def _on_start_clicked(self) -> None:
        if self._phase == "idle":
            try:
                self._detector = LandmarkDetector()
                self._camera = CameraCapture()
                self._camera.frame_callback = self._on_camera_frame
                self._camera.start()
            except Exception as exc:
                msg = QMessageBox(self)
                msg.setWindowTitle("Camera error")
                msg.setText(str(exc))
                msg.exec()
                self._cleanup()
                return

            self._update_timer = QTimer(self)
            self._update_timer.timeout.connect(self._on_update_timer)
            self._update_timer.start(50)

            self._phase = "open"
            self._phase_start_mono = self._now_mono()
            self._open_samples.clear()
            self._closed_samples.clear()
            self.progress_bar.setValue(0)
            self.status_label.setText("Eyes open: hold steady and look at the screen")
            self.instructions_label.setText(
                "Step 1: keep your eyes open and look at the screen."
            )
            self.start_button.setEnabled(False)

        elif self._phase == "between":
            self._phase = "closed"
            self._phase_start_mono = self._now_mono()
            self._closed_samples.clear()
            self.progress_bar.setValue(0)
            self.status_label.setText(
                "Eyes closed: gently close your eyes and keep them closed"
            )
            self.instructions_label.setText(
                "Step 2: close your eyes gently and keep them closed."
            )
            self.start_button.setEnabled(False)

    def _on_cancel_clicked(self) -> None:
        self.calibration_result = None
        self.reject()

    def _on_camera_frame(self, frame: np.ndarray) -> None:
        self._last_frame = frame.copy()

    def _on_update_timer(self) -> None:
        if self._phase not in ("open", "closed"):
            return
        if self._last_frame is None or self._detector is None:
            return

        now = self._now_mono()
        metrics: FaceMetrics = self._detector.process(self._last_frame)
        ear_value = None
        if (
            metrics.face_present
            and metrics.left_eye_ear is not None
            and metrics.right_eye_ear is not None
        ):
            ear_value = min(metrics.left_eye_ear, metrics.right_eye_ear)

        if self._phase_start_mono is None:
            self._phase_start_mono = now

        elapsed = max(0.0, now - self._phase_start_mono)

        if self._phase == "open":
            target = self._open_duration_seconds
            if ear_value is not None:
                self._open_samples.append(ear_value)
            ratio = min(1.0, elapsed / target if target > 0 else 1.0)
            self.progress_bar.setValue(int(ratio * 100))
            if elapsed >= target:
                self._phase = "between"
                self._phase_start_mono = None
                self.progress_bar.setValue(0)
                self.status_label.setText(
                    "Step 1 complete. Get ready to close your eyes, then click Start."
                )
                self.instructions_label.setText(
                    "When you are ready, click Start to begin Step 2 (eyes closed)."
                )
                self.start_button.setEnabled(True)

        elif self._phase == "closed":
            target = self._closed_duration_seconds
            if ear_value is not None:
                self._closed_samples.append(ear_value)
            ratio = min(1.0, elapsed / target if target > 0 else 1.0)
            self.progress_bar.setValue(int(ratio * 100))
            if elapsed >= target:
                self._finish_calibration()

    def _finish_calibration(self) -> None:
        self._phase = "done"
        if self._update_timer is not None:
            self._update_timer.stop()
            self._update_timer = None

        if not self._open_samples or not self._closed_samples:
            self._cleanup()
            msg = QMessageBox(self)
            msg.setWindowTitle("Calibration failed")
            msg.setText("Could not collect enough samples. Please try again.")
            msg.exec()
            self.calibration_result = None
            self.reject()
            return

        ear_open = float(sum(self._open_samples) / len(self._open_samples))
        ear_closed = float(sum(self._closed_samples) / len(self._closed_samples))

        if ear_open <= ear_closed:
            ear_threshold = ear_open * 0.8
        else:
            ear_threshold = (ear_open + ear_closed) / 2.0

        self.calibration_result = CalibrationResult(
            ear_open=ear_open,
            ear_closed=ear_closed,
            ear_threshold=ear_threshold,
        )

        self.status_label.setText(
            f"Calibration complete. Threshold EAR = {ear_threshold:.3f}"
        )
        self.instructions_label.setText("Calibration complete.")
        self.progress_bar.setValue(100)

        self._cleanup()
        self.accept()

    def _cleanup(self) -> None:
        if self._update_timer is not None:
            self._update_timer.stop()
            self._update_timer = None
        if self._camera is not None:
            self._camera.stop()
            self._camera = None
        if self._detector is not None:
            self._detector.close()
            self._detector = None
        self._last_frame = None

    def _now_mono(self) -> float:
        import time

        return time.monotonic()

    def closeEvent(self, event) -> None:
        self._cleanup()
        super().closeEvent(event)
