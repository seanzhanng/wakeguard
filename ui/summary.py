from __future__ import annotations

from typing import Optional

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from core.session import SessionStats
from core.state_machine import FocusEvent


class SummaryDialog(QDialog):
    def __init__(
        self,
        stats: SessionStats,
        session_id: int,
        manual: bool,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.stats = stats
        self.session_id = session_id
        self.manual = manual
        self.setWindowTitle("Session summary")
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout()

        reason = "Ended manually" if self.manual else "Completed target duration"
        reason_label = QLabel(reason)
        font = reason_label.font()
        font.setBold(True)
        reason_label.setFont(font)
        layout.addWidget(reason_label)

        summary_row = QHBoxLayout()

        text_layout = QVBoxLayout()
        id_label = QLabel(f"Session id: {self.session_id}")
        duration_label = QLabel(f"Duration: {self.stats.duration_seconds:.1f} s")
        focused_label = QLabel(f"Focused: {self.stats.focused_seconds:.1f} s")
        drowsy_label = QLabel(f"Drowsy: {self.stats.drowsy_seconds:.1f} s")
        distracted_label = QLabel(f"Distracted: {self.stats.distracted_seconds:.1f} s")
        events_label = QLabel(f"Events: {len(self.stats.events)}")
        text_layout.addWidget(id_label)
        text_layout.addWidget(duration_label)
        text_layout.addWidget(focused_label)
        text_layout.addWidget(drowsy_label)
        text_layout.addWidget(distracted_label)
        text_layout.addWidget(events_label)
        text_layout.addStretch(1)

        summary_row.addLayout(text_layout)

        figure = Figure(figsize=(3, 3))
        canvas = FigureCanvas(figure)
        ax = figure.add_subplot(111)

        values = []
        labels = []
        colors = []

        if self.stats.focused_seconds > 0:
            values.append(self.stats.focused_seconds)
            labels.append("Focused")
            colors.append("#4caf50")
        if self.stats.drowsy_seconds > 0:
            values.append(self.stats.drowsy_seconds)
            labels.append("Drowsy")
            colors.append("#ff9800")
        if self.stats.distracted_seconds > 0:
            values.append(self.stats.distracted_seconds)
            labels.append("Distracted")
            colors.append("#f44336")

        if values:
            ax.pie(values, labels=labels, colors=colors, autopct="%1.0f%%", startangle=90)
            ax.axis("equal")
        else:
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            ax.axis("off")

        summary_row.addWidget(canvas)
        layout.addLayout(summary_row)

        events_label_header = QLabel("Events")
        events_label_header.setAlignment(Qt.AlignmentFlag.AlignLeft)
        layout.addWidget(events_label_header)

        events_table = QTableWidget()
        events_table.setColumnCount(3)
        events_table.setHorizontalHeaderLabels(["Time", "Event", "From → To"])
        events_table.setRowCount(len(self.stats.events))

        for row, event in enumerate(self.stats.events):
            self._populate_event_row(events_table, row, event)

        events_table.resizeColumnsToContents()
        layout.addWidget(events_table)

        self.setLayout(layout)
        self.resize(640, 480)

    def _populate_event_row(self, table: QTableWidget, row: int, event: FocusEvent) -> None:
        time_str = ""
        if hasattr(event, "timestamp"):
            time_str = str(event.timestamp)
        event_str = event.event_type.name if hasattr(event, "event_type") else ""
        from_str = event.from_state.name if hasattr(event, "from_state") else ""
        to_str = event.to_state.name if hasattr(event, "to_state") else ""
        from_to = f"{from_str} → {to_str}" if from_str or to_str else ""

        table.setItem(row, 0, QTableWidgetItem(time_str))
        table.setItem(row, 1, QTableWidgetItem(event_str))
        table.setItem(row, 2, QTableWidgetItem(from_to))
