from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional

BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "wakeguard.db"


@dataclass
class SessionRecord:
    id: int
    start_time: datetime
    end_time: datetime
    duration_seconds: int
    focused_seconds: int
    drowsy_seconds: int
    distracted_seconds: int
    config_json: str
    notes: Optional[str]


@dataclass
class EventRecord:
    id: int
    session_id: int
    timestamp: datetime
    type: str
    details_json: str


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                start_time TEXT NOT NULL,
                end_time TEXT NOT NULL,
                duration_seconds INTEGER NOT NULL,
                focused_seconds INTEGER NOT NULL,
                drowsy_seconds INTEGER NOT NULL,
                distracted_seconds INTEGER NOT NULL,
                config_json TEXT NOT NULL,
                notes TEXT
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER NOT NULL,
                timestamp TEXT NOT NULL,
                type TEXT NOT NULL,
                details_json TEXT NOT NULL,
                FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
            )
            """
        )
        conn.commit()


def insert_session(
    start_time: datetime,
    end_time: datetime,
    duration_seconds: float,
    focused_seconds: float,
    drowsy_seconds: float,
    distracted_seconds: float,
    config_json: str,
    notes: Optional[str] = None,
) -> int:
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO sessions (
                start_time,
                end_time,
                duration_seconds,
                focused_seconds,
                drowsy_seconds,
                distracted_seconds,
                config_json,
                notes
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                start_time.isoformat(),
                end_time.isoformat(),
                int(round(duration_seconds)),
                int(round(focused_seconds)),
                int(round(drowsy_seconds)),
                int(round(distracted_seconds)),
                config_json,
                notes,
            ),
        )
        conn.commit()
        return int(cur.lastrowid)


def insert_event(
    session_id: int,
    timestamp: datetime,
    type: str,
    details_json: str,
) -> int:
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO events (
                session_id,
                timestamp,
                type,
                details_json
            )
            VALUES (?, ?, ?, ?)
            """,
            (
                session_id,
                timestamp.isoformat(),
                type,
                details_json,
            ),
        )
        conn.commit()
        return int(cur.lastrowid)


def fetch_session(session_id: int) -> Optional[SessionRecord]:
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT
                id,
                start_time,
                end_time,
                duration_seconds,
                focused_seconds,
                drowsy_seconds,
                distracted_seconds,
                config_json,
                notes
            FROM sessions
            WHERE id = ?
            """,
            (session_id,),
        )
        row = cur.fetchone()
        if row is None:
            return None
        return SessionRecord(
            id=row["id"],
            start_time=datetime.fromisoformat(row["start_time"]),
            end_time=datetime.fromisoformat(row["end_time"]),
            duration_seconds=int(row["duration_seconds"]),
            focused_seconds=int(row["focused_seconds"]),
            drowsy_seconds=int(row["drowsy_seconds"]),
            distracted_seconds=int(row["distracted_seconds"]),
            config_json=row["config_json"],
            notes=row["notes"],
        )


def fetch_session_events(session_id: int) -> List[EventRecord]:
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT
                id,
                session_id,
                timestamp,
                type,
                details_json
            FROM events
            WHERE session_id = ?
            ORDER BY timestamp ASC
            """,
            (session_id,),
        )
        rows = cur.fetchall()
        result: List[EventRecord] = []
        for row in rows:
            result.append(
                EventRecord(
                    id=row["id"],
                    session_id=row["session_id"],
                    timestamp=datetime.fromisoformat(row["timestamp"]),
                    type=row["type"],
                    details_json=row["details_json"],
                )
            )
        return result


def fetch_recent_sessions(limit: int = 20) -> List[SessionRecord]:
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute(
            f"""
            SELECT
                id,
                start_time,
                end_time,
                duration_seconds,
                focused_seconds,
                drowsy_seconds,
                distracted_seconds,
                config_json,
                notes
            FROM sessions
            ORDER BY start_time DESC
            LIMIT {int(limit)}
            """
        )
        rows = cur.fetchall()
        result: List[SessionRecord] = []
        for row in rows:
            result.append(
                SessionRecord(
                    id=row["id"],
                    start_time=datetime.fromisoformat(row["start_time"]),
                    end_time=datetime.fromisoformat(row["end_time"]),
                    duration_seconds=int(row["duration_seconds"]),
                    focused_seconds=int(row["focused_seconds"]),
                    drowsy_seconds=int(row["drowsy_seconds"]),
                    distracted_seconds=int(row["distracted_seconds"]),
                    config_json=row["config_json"],
                    notes=row["notes"],
                )
            )
        return result


def _init_db_cli() -> None:
    init_db()
    print(f"Database initialized at {DB_PATH}")


if __name__ == "__main__":
    _init_db_cli()
