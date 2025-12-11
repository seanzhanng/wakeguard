from __future__ import annotations

from datetime import datetime
from typing import List, Optional

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from data.db import fetch_recent_sessions, fetch_session_events, SessionRecord, EventRecord


class HistoryDialog(QDialog):
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Session history")
        self._sessions: List[SessionRecord] = []
        self._build_ui()
        self._load_sessions()

    def _build_ui(self) -> None:
        layout = QVBoxLayout()

        title_label = QLabel("Recent sessions")
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        font = title_label.font()
        font.setPointSize(14)
        font.setBold(True)
        title_label.setFont(font)
        layout.addWidget(title_label)

        self.list_widget = QListWidget()
        layout.addWidget(self.list_widget)

        buttons_layout = QHBoxLayout()
        self.details_button = QPushButton("View details")
        close_button = QPushButton("Close")
        buttons_layout.addWidget(self.details_button)
        buttons_layout.addWidget(close_button)
        layout.addLayout(buttons_layout)

        self.setLayout(layout)
        self.resize(520, 420)

        self.details_button.clicked.connect(self._on_view_details_clicked)
        close_button.clicked.connect(self.reject)
        self.list_widget.itemDoubleClicked.connect(self._on_item_double_clicked)

    def _load_sessions(self) -> None:
        self._sessions = fetch_recent_sessions(limit=50)
        self.list_widget.clear()
        for session in self._sessions:
            label = self._format_session_label(session)
            item = QListWidgetItem(label)
            item.setData(Qt.ItemDataRole.UserRole, session.id)
            self.list_widget.addItem(item)

    def _format_session_label(self, session: SessionRecord) -> str:
        start: datetime = session.start_time
        duration = session.duration_seconds
        focused = session.focused_seconds
        drowsy = session.drowsy_seconds
        distracted = session.distracted_seconds
        return (
            f"ID {session.id} | {start.strftime('%Y-%m-%d %H:%M:%S')} | "
            f"Dur {duration:4d}s | F {focused:4d}s | Dzw {drowsy:4d}s | Dis {distracted:4d}s"
        )

    def _on_item_double_clicked(self, item: QListWidgetItem) -> None:
        self._show_details_for_item(item)

    def _on_view_details_clicked(self) -> None:
        item = self.list_widget.currentItem()
        if item is None:
            msg = QMessageBox(self)
            msg.setWindowTitle("No selection")
            msg.setText("Please select a session first.")
            msg.exec()
            return
        self._show_details_for_item(item)

    def _show_details_for_item(self, item: QListWidgetItem) -> None:
        session_id = item.data(Qt.ItemDataRole.UserRole)
        session = next((s for s in self._sessions if s.id == session_id), None)
        if session is None:
            msg = QMessageBox(self)
            msg.setWindowTitle("Error")
            msg.setText("Session not found.")
            msg.exec()
            return

        events: List[EventRecord] = fetch_session_events(session_id=session.id)

        msg = QMessageBox(self)
        msg.setWindowTitle(f"Session {session.id} details")

        start = session.start_time.strftime("%Y-%m-%d %H:%M:%S")
        end = session.end_time.strftime("%Y-%m-%d %H:%M:%S")
        text_lines = [
            f"Session ID: {session.id}",
            f"Start: {start}",
            f"End:   {end}",
            f"Duration:    {session.duration_seconds} s",
            f"Focused:     {session.focused_seconds} s",
            f"Drowsy:      {session.drowsy_seconds} s",
            f"Distracted:  {session.distracted_seconds} s",
            f"Events:      {len(events)}",
        ]

        if events:
            text_lines.append("")
            text_lines.append("Events:")
            for e in events[:20]:
                ts = e.timestamp.strftime("%H:%M:%S")
                text_lines.append(f"{ts}  {e.type}")

        msg.setText("\n".join(text_lines))
        msg.exec()
