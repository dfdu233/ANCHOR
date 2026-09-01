"""SQLite experiment ledger with leasing, heartbeat, and stale-run recovery."""

from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  name TEXT NOT NULL,
  fingerprint TEXT NOT NULL UNIQUE,
  command_json TEXT NOT NULL,
  output_dir TEXT NOT NULL,
  status TEXT NOT NULL CHECK(status IN ('queued','running','done','failed')),
  attempts INTEGER NOT NULL DEFAULT 0,
  max_attempts INTEGER NOT NULL DEFAULT 2,
  worker TEXT,
  created_at REAL NOT NULL,
  started_at REAL,
  heartbeat_at REAL,
  finished_at REAL,
  exit_code INTEGER,
  failure_reason TEXT
);
CREATE INDEX IF NOT EXISTS jobs_status_id ON jobs(status, id);
"""


@dataclass(frozen=True)
class Job:
    id: int
    name: str
    fingerprint: str
    command: list[str]
    output_dir: str
    status: str
    attempts: int
    max_attempts: int
    worker: str | None


class JobQueue:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path, timeout=30, isolation_level=None)
        self.connection.row_factory = sqlite3.Row
        self.connection.executescript(SCHEMA)

    @staticmethod
    def _job(row: sqlite3.Row) -> Job:
        return Job(
            id=row["id"], name=row["name"], fingerprint=row["fingerprint"],
            command=json.loads(row["command_json"]), output_dir=row["output_dir"],
            status=row["status"], attempts=row["attempts"],
            max_attempts=row["max_attempts"], worker=row["worker"],
        )

    def enqueue(
        self, name: str, fingerprint: str, command: list[str], output_dir: str,
        max_attempts: int = 2,
    ) -> int:
        cursor = self.connection.execute(
            """INSERT OR IGNORE INTO jobs
               (name,fingerprint,command_json,output_dir,status,max_attempts,created_at)
               VALUES (?,?,?,?, 'queued', ?, ?)""",
            (name, fingerprint, json.dumps(command), output_dir, max_attempts, time.time()),
        )
        if cursor.lastrowid:
            return int(cursor.lastrowid)
        row = self.connection.execute(
            "SELECT id FROM jobs WHERE fingerprint=?", (fingerprint,)
        ).fetchone()
        return int(row["id"])

    def claim(self, worker: str) -> Job | None:
        now = time.time()
        self.connection.execute("BEGIN IMMEDIATE")
        try:
            row = self.connection.execute(
                """SELECT * FROM jobs WHERE status='queued' AND attempts < max_attempts
                   ORDER BY id LIMIT 1"""
            ).fetchone()
            if row is None:
                self.connection.execute("COMMIT")
                return None
            updated = self.connection.execute(
                """UPDATE jobs SET status='running', attempts=attempts+1,
                   worker=?, started_at=?, heartbeat_at=?
                   WHERE id=? AND status='queued'""",
                (worker, now, now, row["id"]),
            )
            self.connection.execute("COMMIT")
            if updated.rowcount != 1:
                return None
            return self._job(self.connection.execute(
                "SELECT * FROM jobs WHERE id=?", (row["id"],)
            ).fetchone())
        except Exception:
            self.connection.execute("ROLLBACK")
            raise

    def heartbeat(self, job_id: int, worker: str) -> None:
        updated = self.connection.execute(
            "UPDATE jobs SET heartbeat_at=? WHERE id=? AND status='running' AND worker=?",
            (time.time(), job_id, worker),
        )
        if updated.rowcount != 1:
            raise RuntimeError(f"job {job_id} lease is not owned by {worker}")

    def finish(
        self, job_id: int, worker: str, exit_code: int, failure_reason: str | None = None
    ) -> None:
        status = "done" if exit_code == 0 else "failed"
        updated = self.connection.execute(
            """UPDATE jobs SET status=?, finished_at=?, exit_code=?, failure_reason=?
               WHERE id=? AND status='running' AND worker=?""",
            (status, time.time(), exit_code, failure_reason, job_id, worker),
        )
        if updated.rowcount != 1:
            raise RuntimeError(f"job {job_id} lease is not owned by {worker}")

    def recover_stale(self, stale_after_seconds: float) -> list[int]:
        cutoff = time.time() - stale_after_seconds
        rows = self.connection.execute(
            "SELECT id,attempts,max_attempts FROM jobs WHERE status='running' AND heartbeat_at<?",
            (cutoff,),
        ).fetchall()
        recovered = []
        for row in rows:
            status = "queued" if row["attempts"] < row["max_attempts"] else "failed"
            self.connection.execute(
                """UPDATE jobs SET status=?, worker=NULL,
                   failure_reason='stale heartbeat recovered' WHERE id=?""",
                (status, row["id"]),
            )
            recovered.append(int(row["id"]))
        return recovered

    def status(self) -> list[dict[str, Any]]:
        return [dict(row) for row in self.connection.execute(
            "SELECT * FROM jobs ORDER BY id"
        ).fetchall()]
