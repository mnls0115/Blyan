"""
Lightweight analytics store for frontend and backend usage metrics.

- Uses stdlib sqlite3 (no new deps)
- WAL mode for reliability under concurrent writes
- Privacy-focused: stores anonymous client_id and small meta JSON only
"""

from __future__ import annotations

import os
import json
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


_LOCK = threading.Lock()
_GLOBAL: Optional["AnalyticsStore"] = None


def _now_ms() -> int:
    return int(time.time() * 1000)


@dataclass
class AnalyticsConfig:
    enabled: bool
    db_path: Path
    sample_rate: float


def _load_config() -> AnalyticsConfig:
    enabled = os.getenv("ANALYTICS_ENABLED", "true").lower() in ("1", "true", "yes")
    db_path = Path(os.getenv("ANALYTICS_DB_PATH", "data/analytics.db"))
    try:
        sr = float(os.getenv("ANALYTICS_SAMPLE_RATE", "1.0"))
    except Exception:
        sr = 1.0
    sample_rate = max(0.0, min(1.0, sr))
    return AnalyticsConfig(enabled=enabled, db_path=db_path, sample_rate=sample_rate)


class AnalyticsStore:
    """SQLite-backed analytics store with basic aggregation."""

    def __init__(self, cfg: Optional[AnalyticsConfig] = None) -> None:
        self.cfg = cfg or _load_config()
        self._ensure_db()

    def _ensure_db(self) -> None:
        if not self.cfg.db_path.parent.exists():
            self.cfg.db_path.parent.mkdir(parents=True, exist_ok=True)

        with self._connect() as conn:
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS events (
                  id INTEGER PRIMARY KEY AUTOINCREMENT,
                  ts_ms INTEGER NOT NULL,
                  event_type TEXT NOT NULL,
                  client_id TEXT,
                  route TEXT,
                  model TEXT,
                  status TEXT,
                  duration_ms REAL,
                  meta_json TEXT
                );
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_events_ts ON events(ts_ms);")
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_events_type_ts ON events(event_type, ts_ms);"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_events_route_ts ON events(route, ts_ms);"
            )
            conn.commit()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(str(self.cfg.db_path), timeout=5, isolation_level=None)

    def is_enabled(self) -> bool:
        return self.cfg.enabled

    def record_event(
        self,
        *,
        event_type: str,
        client_id: Optional[str] = None,
        route: Optional[str] = None,
        model: Optional[str] = None,
        status: Optional[str] = None,
        duration_ms: Optional[float] = None,
        meta: Optional[Dict[str, Any]] = None,
        ts_ms: Optional[int] = None,
    ) -> None:
        if not self.is_enabled():
            return

        # Basic server-side sampling
        if self.cfg.sample_rate < 1.0:
            # Simple LCG-based hash on client_id+time to avoid randint deps
            basis = f"{client_id or ''}:{ts_ms or _now_ms()}:{event_type}"
            h = 0
            for ch in basis:
                h = (h * 33 + ord(ch)) & 0xFFFFFFFF
            if (h / 0xFFFFFFFF) > self.cfg.sample_rate:
                return

        # Clamp meta size to 2KB
        meta_json: Optional[str] = None
        if meta:
            try:
                meta_json = json.dumps(meta)[:2048]
            except Exception:
                meta_json = None

        with _LOCK:
            with self._connect() as conn:
                conn.execute(
                    """
                    INSERT INTO events (ts_ms, event_type, client_id, route, model, status, duration_ms, meta_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        ts_ms or _now_ms(),
                        event_type,
                        client_id,
                        route,
                        model,
                        status,
                        float(duration_ms) if duration_ms is not None else None,
                        meta_json,
                    ),
                )

    def summary(self, window_seconds: int = 24 * 3600) -> Dict[str, Any]:
        """Aggregate summary for the last window_seconds."""
        now = _now_ms()
        since = now - window_seconds * 1000

        with self._connect() as conn:
            cur = conn.cursor()

            def one(q: str, params: Tuple[Any, ...]) -> int:
                cur.execute(q, params)
                row = cur.fetchone()
                return int(row[0] or 0)

            page_views = one(
                "SELECT COUNT(*) FROM events WHERE event_type = ? AND ts_ms >= ?",
                ("page_view", since),
            )

            cur.execute(
                "SELECT COUNT(DISTINCT COALESCE(client_id, '')) FROM events WHERE ts_ms >= ?",
                (since,),
            )
            unique_clients = int(cur.fetchone()[0] or 0)

            inf_total = one(
                "SELECT COUNT(*) FROM events WHERE event_type IN ('inference_client_timing','inference_processed') AND ts_ms >= ?",
                (since,),
            )
            inf_success = one(
                "SELECT COUNT(*) FROM events WHERE event_type IN ('inference_client_timing','inference_processed') AND status = 'success' AND ts_ms >= ?",
                (since,),
            )

            # Frontend latency stats (Python percentile)
            cur.execute(
                "SELECT duration_ms FROM events WHERE event_type='inference_client_timing' AND ts_ms>=? AND duration_ms IS NOT NULL",
                (since,),
            )
            fe_vals = sorted([float(r[0]) for r in cur.fetchall() if r[0] is not None])
            fe_mean = sum(fe_vals) / len(fe_vals) if fe_vals else 0.0
            fe_p50 = _percentile(fe_vals, 0.5) if fe_vals else 0.0
            fe_p95 = _percentile(fe_vals, 0.95) if fe_vals else 0.0

            # Backend latency stats
            cur.execute(
                "SELECT duration_ms FROM events WHERE event_type='inference_processed' AND ts_ms>=? AND duration_ms IS NOT NULL",
                (since,),
            )
            be_vals = sorted([float(r[0]) for r in cur.fetchall() if r[0] is not None])
            be_mean = sum(be_vals) / len(be_vals) if be_vals else 0.0
            be_p50 = _percentile(be_vals, 0.5) if be_vals else 0.0
            be_p95 = _percentile(be_vals, 0.95) if be_vals else 0.0

            # Top routes
            cur.execute(
                """
                SELECT route, COUNT(*) as c
                FROM events
                WHERE ts_ms >= ? AND route IS NOT NULL
                GROUP BY route
                ORDER BY c DESC
                LIMIT 5
                """,
                (since,),
            )
            top_routes = [{"route": r or "", "count": int(c)} for r, c in cur.fetchall()]

            # Top models
            cur.execute(
                """
                SELECT model, COUNT(*) as c
                FROM events
                WHERE ts_ms >= ? AND model IS NOT NULL
                GROUP BY model
                ORDER BY c DESC
                LIMIT 5
                """,
                (since,),
            )
            top_models = [{"model": m or "", "count": int(c)} for m, c in cur.fetchall()]

        return {
            "window_seconds": window_seconds,
            "page_views_total": page_views,
            "unique_clients": unique_clients,
            "inference_count": inf_total,
            "inference_success_rate": (inf_success / inf_total) if inf_total else 0.0,
            "frontend_latency_ms": {"mean": fe_mean, "p50": fe_p50, "p95": fe_p95},
            "backend_latency_ms": {"mean": be_mean, "p50": be_p50, "p95": be_p95},
            "top_routes": top_routes,
            "top_models": top_models,
        }


def get_store() -> AnalyticsStore:
    global _GLOBAL
    if _GLOBAL is None:
        _GLOBAL = AnalyticsStore()
    return _GLOBAL

# Helpers
def _percentile(sorted_vals: list[float], p: float) -> float:
    if not sorted_vals:
        return 0.0
    if p <= 0:
        return sorted_vals[0]
    if p >= 1:
        return sorted_vals[-1]
    k = (len(sorted_vals) - 1) * p
    f = int(k)
    c = min(f + 1, len(sorted_vals) - 1)
    if f == c:
        return sorted_vals[f]
    d0 = sorted_vals[f] * (c - k)
    d1 = sorted_vals[c] * (k - f)
    return d0 + d1
