from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np

from .camera import CameraCapture

Frame = np.ndarray

RIGHT_EYE_INDICES: Tuple[int, int, int, int, int, int] = (
    33,
    159,
    158,
    133,
    153,
    145,
)

LEFT_EYE_INDICES: Tuple[int, int, int, int, int, int] = (
    362,
    380,
    374,
    263,
    386,
    385,
)


@dataclass
class FaceMetrics:
    face_present: bool
    left_eye_ear: Optional[float]
    right_eye_ear: Optional[float]
    landmarks: Optional[np.ndarray]


class LandmarkDetector:
    def __init__(
        self,
        max_num_faces: int = 1,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        self._mp_face_mesh = mp.solutions.face_mesh
        self._mesh = self._mp_face_mesh.FaceMesh(
            static_image_mode=False,
            refine_landmarks=True,
            max_num_faces=max_num_faces,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def process(self, frame_rgb: Frame) -> FaceMetrics:
        results = self._mesh.process(frame_rgb)
        if not results.multi_face_landmarks:
            return FaceMetrics(False, None, None, None)

        face_landmarks = results.multi_face_landmarks[0]
        landmarks = np.array(
            [[lm.x, lm.y, lm.z] for lm in face_landmarks.landmark],
            dtype=np.float32,
        )

        right_ear = self._eye_aspect_ratio(landmarks, RIGHT_EYE_INDICES)
        left_ear = self._eye_aspect_ratio(landmarks, LEFT_EYE_INDICES)

        return FaceMetrics(True, left_ear, right_ear, landmarks)

    def close(self) -> None:
        self._mesh.close()

    def __enter__(self) -> LandmarkDetector:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    @staticmethod
    def _eye_aspect_ratio(
        landmarks: np.ndarray,
        indices: Tuple[int, int, int, int, int, int],
    ) -> float:
        p1, p2, p3, p4, p5, p6 = indices
        a = np.linalg.norm(landmarks[p2, :2] - landmarks[p6, :2])
        b = np.linalg.norm(landmarks[p3, :2] - landmarks[p5, :2])
        c = np.linalg.norm(landmarks[p1, :2] - landmarks[p4, :2])
        if c == 0:
            return 0.0
        return float((a + b) / (2.0 * c))


def _preview_loop() -> None:
    window_name = "WakeGuard Landmarks Preview"
    last_frame: Optional[Frame] = None
    last_metrics: Optional[FaceMetrics] = None
    detector = LandmarkDetector()

    def on_frame(frame: Frame) -> None:
        nonlocal last_frame, last_metrics
        last_frame = frame.copy()
        last_metrics = detector.process(frame)

    with CameraCapture() as camera:
        camera.frame_callback = on_frame

        while True:
            if last_frame is not None:
                frame_bgr = cv2.cvtColor(last_frame, cv2.COLOR_RGB2BGR)
                text = "No face"
                if last_metrics is not None and last_metrics.face_present:
                    h, w, _ = frame_bgr.shape
                    if last_metrics.landmarks is not None:
                        pts = last_metrics.landmarks
                        for idx in RIGHT_EYE_INDICES + LEFT_EYE_INDICES:
                            x = int(pts[idx, 0] * w)
                            y = int(pts[idx, 1] * h)
                            cv2.circle(frame_bgr, (x, y), 1, (0, 255, 0), -1)
                    le = last_metrics.left_eye_ear or 0.0
                    re = last_metrics.right_eye_ear or 0.0
                    text = f"L EAR: {le:.3f}  R EAR: {re:.3f}"
                cv2.putText(
                    frame_bgr,
                    text,
                    (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )
                cv2.imshow(window_name, frame_bgr)

            if cv2.waitKey(1) & 0xFF == 27:
                break

        detector.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    _preview_loop()
