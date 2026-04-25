import json
import os
import sqlite3
import time
from typing import Any, Dict, List, Optional


class RouterStore:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.json_path = "{0}.json".format(db_path)
        self.backend = "sqlite"
        directory = os.path.dirname(db_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        try:
            self._init_db()
        except sqlite3.Error:
            self.backend = "json"
            self._init_json()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        return connection

    def _init_db(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS health_checks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    provider_id TEXT NOT NULL,
                    model_id TEXT NOT NULL,
                    success INTEGER NOT NULL,
                    status_code INTEGER,
                    latency_ms REAL,
                    checked_at INTEGER NOT NULL,
                    error TEXT
                );

                CREATE TABLE IF NOT EXISTS request_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    request_id TEXT NOT NULL,
                    provider_id TEXT NOT NULL,
                    model_id TEXT NOT NULL,
                    scenario TEXT NOT NULL,
                    performance TEXT NOT NULL,
                    success INTEGER NOT NULL,
                    latency_ms REAL,
                    prompt_tokens INTEGER,
                    completion_tokens INTEGER,
                    total_tokens INTEGER,
                    created_at INTEGER NOT NULL,
                    error TEXT
                );
                """
            )

    def _init_json(self) -> None:
        if not os.path.exists(self.json_path):
            with open(self.json_path, "w", encoding="utf-8") as handle:
                json.dump({"health_checks": [], "request_logs": []}, handle)

    def _json_read(self) -> Dict[str, Any]:
        self._init_json()
        with open(self.json_path, "r", encoding="utf-8") as handle:
            return json.load(handle)

    def _json_write(self, data: Dict[str, Any]) -> None:
        with open(self.json_path, "w", encoding="utf-8") as handle:
            json.dump(data, handle, ensure_ascii=True, indent=2)

    def log_health_check(
        self,
        provider_id: str,
        model_id: str,
        success: bool,
        latency_ms: Optional[float] = None,
        status_code: Optional[int] = None,
        error: Optional[str] = None,
    ) -> None:
        if self.backend == "json":
            data = self._json_read()
            data["health_checks"].append(
                {
                    "provider_id": provider_id,
                    "model_id": model_id,
                    "success": 1 if success else 0,
                    "status_code": status_code,
                    "latency_ms": latency_ms,
                    "checked_at": int(time.time()),
                    "error": error,
                }
            )
            self._json_write(data)
            return

        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO health_checks (
                    provider_id, model_id, success, status_code, latency_ms, checked_at, error
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    provider_id,
                    model_id,
                    1 if success else 0,
                    status_code,
                    latency_ms,
                    int(time.time()),
                    error,
                ),
            )

    def log_request(
        self,
        request_id: str,
        provider_id: str,
        model_id: str,
        scenario: str,
        performance: str,
        success: bool,
        latency_ms: Optional[float],
        prompt_tokens: int,
        completion_tokens: int,
        total_tokens: int,
        error: Optional[str] = None,
    ) -> None:
        if self.backend == "json":
            data = self._json_read()
            data["request_logs"].append(
                {
                    "request_id": request_id,
                    "provider_id": provider_id,
                    "model_id": model_id,
                    "scenario": scenario,
                    "performance": performance,
                    "success": 1 if success else 0,
                    "latency_ms": latency_ms,
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": total_tokens,
                    "created_at": int(time.time()),
                    "error": error,
                }
            )
            self._json_write(data)
            return

        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO request_logs (
                    request_id, provider_id, model_id, scenario, performance, success, latency_ms,
                    prompt_tokens, completion_tokens, total_tokens, created_at, error
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    request_id,
                    provider_id,
                    model_id,
                    scenario,
                    performance,
                    1 if success else 0,
                    latency_ms,
                    prompt_tokens,
                    completion_tokens,
                    total_tokens,
                    int(time.time()),
                    error,
                ),
            )

    def health_snapshot(self) -> Dict[str, Dict[str, float]]:
        if self.backend == "json":
            data = self._json_read()
            grouped: Dict[str, Dict[str, float]] = {}
            counts: Dict[str, int] = {}
            for row in data["health_checks"]:
                model_id = row["model_id"]
                grouped.setdefault(model_id, {"success_rate": 0.0, "latency_ms": 0.0})
                counts[model_id] = counts.get(model_id, 0) + 1
                grouped[model_id]["success_rate"] += float(row.get("success", 0))
                grouped[model_id]["latency_ms"] += float(row.get("latency_ms") or 0.0)
            for model_id, values in grouped.items():
                count = float(counts[model_id])
                values["success_rate"] = values["success_rate"] / count
                values["latency_ms"] = values["latency_ms"] / count
            return grouped

        query = """
            SELECT
                model_id,
                AVG(CASE WHEN success = 1 THEN 1.0 ELSE 0.0 END) AS success_rate,
                AVG(COALESCE(latency_ms, 0)) AS latency_ms
            FROM (
                SELECT *
                FROM health_checks
                ORDER BY checked_at DESC
            )
            GROUP BY model_id
        """
        with self._connect() as connection:
            rows = connection.execute(query).fetchall()
        return {
            row["model_id"]: {
                "success_rate": float(row["success_rate"] or 0.0),
                "latency_ms": float(row["latency_ms"] or 0.0),
            }
            for row in rows
        }

    def provider_status_rows(self) -> List[Dict[str, Any]]:
        if self.backend == "json":
            data = self._json_read()
            latest_by_model: Dict[str, Dict[str, Any]] = {}
            for row in data["health_checks"]:
                latest_by_model[row["model_id"]] = row
            rows = list(latest_by_model.values())
            rows.sort(key=lambda item: (item["provider_id"], item["model_id"]))
            return rows

        query = """
            SELECT
                provider_id,
                model_id,
                success,
                status_code,
                latency_ms,
                checked_at,
                error
            FROM health_checks
            WHERE id IN (
                SELECT MAX(id)
                FROM health_checks
                GROUP BY model_id
            )
            ORDER BY provider_id, model_id
        """
        with self._connect() as connection:
            rows = connection.execute(query).fetchall()
        return [dict(row) for row in rows]

    def usage_summary(self) -> Dict[str, Any]:
        if self.backend == "json":
            data = self._json_read()
            request_logs = data["request_logs"]
            by_model: Dict[str, Dict[str, Any]] = {}
            successful_requests = 0
            total_tokens = 0
            for row in request_logs:
                model_bucket = by_model.setdefault(row["model_id"], {"model_id": row["model_id"], "requests": 0, "total_tokens": 0})
                model_bucket["requests"] += 1
                model_bucket["total_tokens"] += int(row.get("total_tokens") or 0)
                total_tokens += int(row.get("total_tokens") or 0)
                if row.get("success"):
                    successful_requests += 1
            recent = list(reversed(request_logs[-25:]))
            sorted_models = sorted(by_model.values(), key=lambda item: item["requests"], reverse=True)
            return {
                "totals": {
                    "requests": len(request_logs),
                    "successful_requests": successful_requests,
                    "total_tokens": total_tokens,
                },
                "by_model": sorted_models,
                "recent_requests": recent,
            }

        with self._connect() as connection:
            totals = connection.execute(
                """
                SELECT
                    COUNT(*) AS requests,
                    SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) AS successful_requests,
                    SUM(total_tokens) AS total_tokens
                FROM request_logs
                """
            ).fetchone()
            by_model = connection.execute(
                """
                SELECT model_id, COUNT(*) AS requests, SUM(total_tokens) AS total_tokens
                FROM request_logs
                GROUP BY model_id
                ORDER BY requests DESC
                """
            ).fetchall()
            recent = connection.execute(
                """
                SELECT provider_id, model_id, scenario, success, latency_ms, created_at, error
                FROM request_logs
                ORDER BY created_at DESC
                LIMIT 25
                """
            ).fetchall()

        return {
            "totals": dict(totals) if totals else {"requests": 0, "successful_requests": 0, "total_tokens": 0},
            "by_model": [dict(row) for row in by_model],
            "recent_requests": [dict(row) for row in recent],
        }
