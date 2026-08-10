#!/usr/bin/env python3
"""Read-only campaign watcher for benchmark GPU co-tenancy.

The watcher never changes campaign state and never signals any process.  It reads
the campaign heartbeat, samples ``nvidia-smi`` while a physical GPU is assigned
to a benchmark, and writes an independent audit trail under
``CAMPAIGN_ROOT/audit/gpu_cotenancy``.

One CUDA compute application is expected for a running benchmark.  More than
one application is recorded as possible co-tenancy.  Zero applications is not
an anomaly because it is normal during benchmark import and teardown.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import datetime as dt
import fcntl
import hashlib
import json
import math
import os
import signal
import subprocess
import sys
import tempfile
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


SCHEMA_VERSION = "mergenet.cifar_resize_gpu_cotenancy_audit.v1"
SCRIPT_PATH = Path(__file__).resolve()
DEFAULT_INTERVAL_SEC = 5.0
DEFAULT_QUERY_TIMEOUT_SEC = 15.0


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def parse_utc(value: Any) -> dt.datetime | None:
    try:
        parsed = dt.datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    return parsed if parsed.tzinfo is not None else None


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _fsync_directory(directory: Path) -> None:
    descriptor = os.open(directory, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def atomic_append_jsonl_many(path: Path, payloads: Sequence[Mapping[str, Any]]) -> None:
    """Append JSONL records using replace, so readers never see a partial line."""

    if not payloads:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    previous = b""
    try:
        previous = path.read_bytes()
    except FileNotFoundError:
        pass
    if previous and not previous.endswith(b"\n"):
        raise RuntimeError(f"refusing to extend malformed JSONL without trailing newline: {path}")
    encoded = b"".join(
        (
            json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            + "\n"
        ).encode("utf-8")
        for payload in payloads
    )
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(previous)
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def atomic_append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    atomic_append_jsonl_many(path, [payload])


@contextmanager
def exclusive_lock(path: Path) -> Iterable[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"another GPU co-tenancy watcher holds {path}") from exc
        handle.seek(0)
        handle.truncate()
        handle.write(f"pid={os.getpid()} acquired_at={utc_now()}\n")
        handle.flush()
        os.fsync(handle.fileno())
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def process_exists(pid: int) -> bool:
    if pid <= 1:
        return False
    stat_path = Path("/proc") / str(pid) / "stat"
    try:
        raw = stat_path.read_text(encoding="utf-8")
        # /proc/PID/stat starts with ``pid (comm) state``; comm may contain
        # spaces or parentheses, so split only after its final closing paren.
        fields = raw[raw.rfind(")") + 1 :].strip().split()
        state = fields[0]
    except (OSError, IndexError):
        return False
    return state != "Z"


def process_is_campaign_master(pid: int, campaign_root: Path) -> bool:
    """Reject a reused PID by checking the live command line when procfs exists."""

    if not process_exists(pid):
        return False
    cmdline_path = Path("/proc") / str(pid) / "cmdline"
    try:
        fields = [
            value.decode("utf-8", errors="replace")
            for value in cmdline_path.read_bytes().split(b"\0")
            if value
        ]
    except OSError:
        return False
    command = "\n".join(fields)
    return "campaign.py" in command and str(campaign_root) in command


def _run_nvidia_smi(arguments: Sequence[str], timeout_sec: float) -> str:
    completed = subprocess.run(
        ["nvidia-smi", *arguments],
        check=True,
        capture_output=True,
        text=True,
        timeout=timeout_sec,
    )
    return completed.stdout


def parse_gpu_rows(text: str) -> dict[int, str]:
    result: dict[int, str] = {}
    for row in csv.reader(text.splitlines(), skipinitialspace=True):
        if not row or all(not field.strip() for field in row):
            continue
        if len(row) != 2:
            raise ValueError(f"unexpected nvidia-smi GPU row: {row!r}")
        index_text, gpu_uuid = (field.strip() for field in row)
        index = int(index_text)
        if not gpu_uuid.startswith("GPU-"):
            raise ValueError(f"invalid GPU UUID in nvidia-smi row: {row!r}")
        result[index] = gpu_uuid
    if not result:
        raise ValueError("nvidia-smi returned no physical GPUs")
    return result


def _memory_mib(value: str) -> int | None:
    try:
        return int(float(value.strip()))
    except ValueError:
        return None


def parse_compute_rows(text: str) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {}
    for row in csv.reader(text.splitlines(), skipinitialspace=True):
        if not row or all(not field.strip() for field in row):
            continue
        if len(row) != 4:
            raise ValueError(f"unexpected nvidia-smi compute row: {row!r}")
        gpu_uuid, pid_text, process_name, memory_text = (
            field.strip() for field in row
        )
        if not gpu_uuid.startswith("GPU-"):
            raise ValueError(f"invalid compute-app GPU UUID: {row!r}")
        try:
            pid: int | str = int(pid_text)
        except ValueError:
            pid = pid_text
        result.setdefault(gpu_uuid, []).append(
            {
                "pid": pid,
                "process_name": process_name,
                "used_gpu_memory_mib": _memory_mib(memory_text),
                "used_gpu_memory_raw": memory_text,
            }
        )
    for applications in result.values():
        applications.sort(key=lambda item: str(item["pid"]))
    return result


def query_compute_apps(timeout_sec: float) -> tuple[dict[int, str], dict[str, list[dict[str, Any]]]]:
    gpu_rows = _run_nvidia_smi(
        ["--query-gpu=index,uuid", "--format=csv,noheader,nounits"], timeout_sec
    )
    compute_rows = _run_nvidia_smi(
        [
            "--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory",
            "--format=csv,noheader,nounits",
        ],
        timeout_sec,
    )
    return parse_gpu_rows(gpu_rows), parse_compute_rows(compute_rows)


def _stable_session_id(
    campaign_id: str,
    physical_gpu: int,
    benchmark_pid: int,
    benchmark_started_at: str | None,
) -> str:
    canonical = (
        f"{campaign_id}|{physical_gpu}|{benchmark_pid}|{benchmark_started_at or 'unknown'}"
    ).encode("utf-8")
    digest = hashlib.sha256(canonical).hexdigest()[:16]
    return f"gpu{physical_gpu}_pid{benchmark_pid}_{digest}"


def _read_benchmark_started_at(
    campaign_root: Path, physical_gpu: int, benchmark_pid: int
) -> str | None:
    pid_path = campaign_root / "state" / "pids" / f"gpu{physical_gpu}.json"
    try:
        value = load_json(pid_path)
        if (
            isinstance(value, Mapping)
            and value.get("kind") == "benchmark"
            and int(value.get("pid")) == benchmark_pid
        ):
            started_at = value.get("started_at")
            return str(started_at) if started_at else None
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        pass
    return None


@dataclasses.dataclass
class Session:
    session_id: str
    campaign_id: str
    physical_gpu: int
    gpu_uuid: str | None
    benchmark_key: str
    benchmark_pid: int
    benchmark_started_at: str | None
    first_seen_at: str
    last_seen_at: str
    samples: int = 0
    anomalous_samples: int = 0
    anomaly_events: int = 0
    query_failures: int = 0
    min_compute_app_count: int | None = None
    max_compute_app_count: int = 0
    observed_compute_pids: set[str] = dataclasses.field(default_factory=set)
    anomaly_active: bool = False
    last_sample: dict[str, Any] | None = None
    ended_at: str | None = None
    end_reason: str | None = None

    def observe(self, sample: Mapping[str, Any]) -> tuple[bool, bool]:
        """Update counters and return (entered_anomaly, recovered)."""

        self.samples += 1
        self.last_seen_at = str(sample["captured_at"])
        self.last_sample = dict(sample)
        count = int(sample["compute_app_count"])
        self.max_compute_app_count = max(self.max_compute_app_count, count)
        self.min_compute_app_count = (
            count if self.min_compute_app_count is None else min(self.min_compute_app_count, count)
        )
        for application in sample.get("compute_apps", []):
            self.observed_compute_pids.add(str(application.get("pid")))
        anomalous = bool(sample.get("possible_cotenancy"))
        if anomalous:
            self.anomalous_samples += 1
        entered = anomalous and not self.anomaly_active
        recovered = not anomalous and self.anomaly_active
        if entered:
            self.anomaly_events += 1
        self.anomaly_active = anomalous
        return entered, recovered

    def observe_query_failure(self, captured_at: str, error: str) -> None:
        self.samples += 1
        self.query_failures += 1
        self.last_seen_at = captured_at
        self.last_sample = {
            "captured_at": captured_at,
            "query_ok": False,
            "query_error": error,
        }

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "session_id": self.session_id,
            "campaign_id": self.campaign_id,
            "physical_gpu": self.physical_gpu,
            "gpu_uuid": self.gpu_uuid,
            "benchmark_key": self.benchmark_key,
            "benchmark_pid": self.benchmark_pid,
            "benchmark_started_at": self.benchmark_started_at,
            "first_seen_at": self.first_seen_at,
            "last_seen_at": self.last_seen_at,
            "ended_at": self.ended_at,
            "end_reason": self.end_reason,
            "samples": self.samples,
            "anomalous_samples": self.anomalous_samples,
            "anomaly_events": self.anomaly_events,
            "query_failures": self.query_failures,
            "min_compute_app_count": self.min_compute_app_count,
            "max_compute_app_count": self.max_compute_app_count,
            "observed_compute_pids": sorted(self.observed_compute_pids),
            "possible_cotenancy_observed": self.anomalous_samples > 0,
            "anomaly_active": self.anomaly_active,
            "last_sample": self.last_sample,
            "interpretation": (
                "compute_app_count > 1 while heartbeat assigns one benchmark process "
                "is possible GPU co-tenancy; zero is allowed during startup/teardown"
            ),
        }


class Watcher:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.campaign_root = args.campaign_root.expanduser().resolve(strict=True)
        self.heartbeat_path = self.campaign_root / "state" / "heartbeat.json"
        self.audit_dir = self.campaign_root / "audit" / "gpu_cotenancy"
        self.sessions_dir = self.audit_dir / "sessions"
        self.samples_path = self.audit_dir / "samples.jsonl"
        self.events_path = self.audit_dir / "events.jsonl"
        self.status_path = self.audit_dir / "status.json"
        self.lock_path = self.audit_dir / "watcher.lock"
        self.stop_requested = False
        self.stop_signal: int | None = None
        self.started_at = utc_now()
        self.master_pid: int | None = None
        self.campaign_id: str | None = None
        self.active: dict[tuple[int, int], Session] = {}
        self.total_samples = 0
        self.total_anomaly_events = 0
        self.total_query_failures = 0
        self.last_heartbeat_at: str | None = None
        self.last_error: str | None = None

    def request_stop(self, signum: int, _frame: Any) -> None:
        self.stop_requested = True
        self.stop_signal = signum

    def _event(self, event: str, **details: Any) -> None:
        payload = {
            "schema_version": SCHEMA_VERSION,
            "captured_at": utc_now(),
            "watcher_pid": os.getpid(),
            "master_pid": self.master_pid,
            "event": event,
            **details,
        }
        atomic_append_jsonl(self.events_path, payload)

    def _write_session(self, session: Session) -> None:
        atomic_write_json(self.sessions_dir / f"{session.session_id}.json", session.as_dict())

    def _restore_session(self, session: Session) -> bool:
        """Resume counters when the watcher restarts during the same benchmark."""

        path = self.sessions_dir / f"{session.session_id}.json"
        try:
            previous = load_json(path)
        except (OSError, json.JSONDecodeError):
            return False
        if not isinstance(previous, Mapping):
            raise RuntimeError(f"existing session audit is not an object: {path}")
        identity = (
            previous.get("session_id") == session.session_id
            and int(previous.get("physical_gpu")) == session.physical_gpu
            and int(previous.get("benchmark_pid")) == session.benchmark_pid
            and previous.get("campaign_id") == session.campaign_id
        )
        if not identity:
            raise RuntimeError(f"existing session audit identity mismatch: {path}")
        previous_uuid = previous.get("gpu_uuid")
        if isinstance(previous_uuid, str) and previous_uuid.startswith("GPU-"):
            session.gpu_uuid = previous_uuid
        session.first_seen_at = str(previous.get("first_seen_at", session.first_seen_at))
        session.last_seen_at = str(previous.get("last_seen_at", session.last_seen_at))
        session.samples = int(previous.get("samples", 0))
        session.anomalous_samples = int(previous.get("anomalous_samples", 0))
        session.anomaly_events = int(previous.get("anomaly_events", 0))
        session.query_failures = int(previous.get("query_failures", 0))
        minimum = previous.get("min_compute_app_count")
        session.min_compute_app_count = None if minimum is None else int(minimum)
        session.max_compute_app_count = int(previous.get("max_compute_app_count", 0))
        session.observed_compute_pids = {
            str(value) for value in previous.get("observed_compute_pids", [])
        }
        session.anomaly_active = bool(previous.get("anomaly_active", False))
        last_sample = previous.get("last_sample")
        session.last_sample = dict(last_sample) if isinstance(last_sample, Mapping) else None
        session.ended_at = None
        session.end_reason = None
        return True

    def _write_status(self, state: str, reason: str | None = None) -> None:
        atomic_write_json(
            self.status_path,
            {
                "schema_version": SCHEMA_VERSION,
                "state": state,
                "reason": reason,
                "watcher_pid": os.getpid(),
                "launch_token": self.args.launch_token,
                "campaign_root": str(self.campaign_root),
                "campaign_id": self.campaign_id,
                "master_pid": self.master_pid,
                "started_at": self.started_at,
                "updated_at": utc_now(),
                "last_heartbeat_at": self.last_heartbeat_at,
                "interval_sec": self.args.interval_sec,
                "query_timeout_sec": self.args.query_timeout_sec,
                "active_sessions": [
                    session.session_id
                    for _, session in sorted(self.active.items())
                ],
                "total_samples": self.total_samples,
                "total_query_failures": self.total_query_failures,
                "total_anomaly_events": self.total_anomaly_events,
                "last_error": self.last_error,
                "stop_signal": self.stop_signal,
                "writes_only_under": str(self.audit_dir),
            },
        )

    def _load_heartbeat(self) -> Mapping[str, Any]:
        value = load_json(self.heartbeat_path)
        if not isinstance(value, Mapping):
            raise ValueError("heartbeat root is not an object")
        master_pid = int(value.get("master_pid"))
        if master_pid <= 1:
            raise ValueError(f"invalid heartbeat master_pid={master_pid}")
        running = value.get("running")
        if not isinstance(running, Mapping):
            raise ValueError("heartbeat.running is not an object")
        return value

    def _benchmark_assignments(
        self, heartbeat: Mapping[str, Any]
    ) -> dict[tuple[int, int], dict[str, Any]]:
        assignments: dict[tuple[int, int], dict[str, Any]] = {}
        for gpu_text, raw in heartbeat["running"].items():
            if not isinstance(raw, Mapping) or raw.get("kind") != "benchmark":
                continue
            physical_gpu = int(gpu_text)
            benchmark_pid = int(raw.get("pid"))
            if physical_gpu < 0 or benchmark_pid <= 1:
                raise ValueError(f"invalid benchmark assignment gpu={gpu_text!r}: {raw!r}")
            assignments[(physical_gpu, benchmark_pid)] = {
                "physical_gpu": physical_gpu,
                "benchmark_pid": benchmark_pid,
                "benchmark_key": str(raw.get("key", f"gpu{physical_gpu}")),
            }
        return assignments

    def _ensure_sessions(
        self,
        assignments: Mapping[tuple[int, int], Mapping[str, Any]],
        gpu_uuids: Mapping[int, str] | None,
        captured_at: str,
    ) -> None:
        removed = set(self.active) - set(assignments)
        for identity in sorted(removed):
            session = self.active.pop(identity)
            session.ended_at = captured_at
            session.end_reason = "benchmark_no_longer_in_heartbeat"
            self._write_session(session)
            self._event(
                "benchmark_session_end",
                session_id=session.session_id,
                physical_gpu=session.physical_gpu,
                benchmark_pid=session.benchmark_pid,
                end_reason=session.end_reason,
            )

        for identity, assignment in sorted(assignments.items()):
            if identity in self.active:
                continue
            physical_gpu, benchmark_pid = identity
            benchmark_started_at = _read_benchmark_started_at(
                self.campaign_root, physical_gpu, benchmark_pid
            )
            session_id = _stable_session_id(
                str(self.campaign_id), physical_gpu, benchmark_pid, benchmark_started_at
            )
            session = Session(
                session_id=session_id,
                campaign_id=str(self.campaign_id),
                physical_gpu=physical_gpu,
                gpu_uuid=gpu_uuids.get(physical_gpu) if gpu_uuids else None,
                benchmark_key=str(assignment["benchmark_key"]),
                benchmark_pid=benchmark_pid,
                benchmark_started_at=benchmark_started_at,
                first_seen_at=captured_at,
                last_seen_at=captured_at,
            )
            resumed = self._restore_session(session)
            self.active[identity] = session
            self._write_session(session)
            self._event(
                "benchmark_session_observation_resumed" if resumed else "benchmark_session_start",
                session_id=session.session_id,
                physical_gpu=physical_gpu,
                gpu_uuid=session.gpu_uuid,
                benchmark_pid=benchmark_pid,
                benchmark_key=session.benchmark_key,
                benchmark_started_at=benchmark_started_at,
            )

    def _sample_once(self, heartbeat: Mapping[str, Any]) -> None:
        captured_at = utc_now()
        sample_records: list[dict[str, Any]] = []
        assignments = self._benchmark_assignments(heartbeat)
        try:
            gpu_uuids, compute_apps = query_compute_apps(self.args.query_timeout_sec)
            query_error = None
        except (OSError, subprocess.SubprocessError, ValueError) as exc:
            gpu_uuids = {}
            compute_apps = {}
            query_error = f"{type(exc).__name__}: {exc}"
            self.last_error = query_error
            self.total_query_failures += 1
            self._event("nvidia_smi_query_failure", error=query_error)

        self._ensure_sessions(assignments, gpu_uuids, captured_at)
        for identity, session in sorted(self.active.items()):
            if identity not in assignments:
                continue
            if query_error is not None:
                session.observe_query_failure(captured_at, query_error)
                self.total_samples += 1
                self._write_session(session)
                sample_records.append(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "session_id": session.session_id,
                        "campaign_id": self.campaign_id,
                        "physical_gpu": session.physical_gpu,
                        "benchmark_pid": session.benchmark_pid,
                        "captured_at": captured_at,
                        "query_ok": False,
                        "query_error": query_error,
                    }
                )
                continue

            gpu_uuid = gpu_uuids.get(session.physical_gpu)
            if gpu_uuid is None:
                error = f"physical GPU {session.physical_gpu} missing from nvidia-smi index/UUID query"
                session.observe_query_failure(captured_at, error)
                self.total_samples += 1
                self.total_query_failures += 1
                self.last_error = error
                sample = {
                    "schema_version": SCHEMA_VERSION,
                    "session_id": session.session_id,
                    "campaign_id": self.campaign_id,
                    "physical_gpu": session.physical_gpu,
                    "benchmark_pid": session.benchmark_pid,
                    "captured_at": captured_at,
                    "query_ok": False,
                    "query_error": error,
                }
                sample_records.append(sample)
                self._event(
                    "benchmark_gpu_mapping_missing",
                    session_id=session.session_id,
                    physical_gpu=session.physical_gpu,
                    benchmark_pid=session.benchmark_pid,
                    error=error,
                )
                self._write_session(session)
                continue
            if session.gpu_uuid is not None and gpu_uuid != session.gpu_uuid:
                error = (
                    f"physical GPU {session.physical_gpu} UUID changed from "
                    f"{session.gpu_uuid} to {gpu_uuid} during benchmark session"
                )
                session.observe_query_failure(captured_at, error)
                self.total_samples += 1
                self.total_query_failures += 1
                self.last_error = error
                sample = {
                    "schema_version": SCHEMA_VERSION,
                    "session_id": session.session_id,
                    "campaign_id": self.campaign_id,
                    "physical_gpu": session.physical_gpu,
                    "benchmark_pid": session.benchmark_pid,
                    "captured_at": captured_at,
                    "query_ok": False,
                    "query_error": error,
                }
                sample_records.append(sample)
                self._event(
                    "benchmark_gpu_uuid_changed",
                    session_id=session.session_id,
                    physical_gpu=session.physical_gpu,
                    benchmark_pid=session.benchmark_pid,
                    expected_gpu_uuid=session.gpu_uuid,
                    observed_gpu_uuid=gpu_uuid,
                )
                self._write_session(session)
                continue
            applications = list(compute_apps.get(gpu_uuid, [])) if gpu_uuid else []
            sample = {
                "schema_version": SCHEMA_VERSION,
                "session_id": session.session_id,
                "campaign_id": self.campaign_id,
                "physical_gpu": session.physical_gpu,
                "gpu_uuid": gpu_uuid,
                "benchmark_pid": session.benchmark_pid,
                "benchmark_key": session.benchmark_key,
                "captured_at": captured_at,
                "heartbeat_at": heartbeat.get("at"),
                "query_ok": True,
                "compute_app_count": len(applications),
                "expected_max_compute_app_count": 1,
                "possible_cotenancy": len(applications) > 1,
                "extra_compute_app_count": max(0, len(applications) - 1),
                "compute_apps": applications,
            }
            if session.gpu_uuid is None:
                session.gpu_uuid = gpu_uuid
            entered, recovered = session.observe(sample)
            self.total_samples += 1
            sample_records.append(sample)
            if entered:
                self.total_anomaly_events += 1
                self._event(
                    "possible_gpu_cotenancy_started",
                    session_id=session.session_id,
                    physical_gpu=session.physical_gpu,
                    gpu_uuid=gpu_uuid,
                    benchmark_pid=session.benchmark_pid,
                    compute_app_count=len(applications),
                    compute_apps=applications,
                )
            elif recovered:
                self._event(
                    "possible_gpu_cotenancy_recovered",
                    session_id=session.session_id,
                    physical_gpu=session.physical_gpu,
                    gpu_uuid=gpu_uuid,
                    benchmark_pid=session.benchmark_pid,
                    compute_app_count=len(applications),
                    compute_apps=applications,
                )
            self._write_session(session)
        atomic_append_jsonl_many(self.samples_path, sample_records)

    def _close_sessions(self, reason: str) -> None:
        ended_at = utc_now()
        for identity in sorted(list(self.active)):
            session = self.active.pop(identity)
            session.ended_at = ended_at
            session.end_reason = reason
            self._write_session(session)
            self._event(
                "benchmark_session_end",
                session_id=session.session_id,
                physical_gpu=session.physical_gpu,
                benchmark_pid=session.benchmark_pid,
                end_reason=reason,
            )

    def run(self) -> int:
        heartbeat = self._load_heartbeat()
        self.master_pid = int(heartbeat["master_pid"])
        self.campaign_id = str(heartbeat.get("campaign_id", self.campaign_root.name))
        if not process_is_campaign_master(self.master_pid, self.campaign_root):
            raise RuntimeError(
                f"heartbeat master PID {self.master_pid} is not the live campaign master"
            )
        self.last_heartbeat_at = str(heartbeat.get("at"))
        self._event("watcher_start", campaign_root=str(self.campaign_root))
        self._write_status("running")

        samples_taken = 0
        stop_reason = "requested_stop"
        while not self.stop_requested:
            if not process_is_campaign_master(self.master_pid, self.campaign_root):
                stop_reason = "campaign_master_exited_or_identity_changed"
                self._event("master_exit_detected", campaign_root=str(self.campaign_root))
                break
            try:
                heartbeat = self._load_heartbeat()
            except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
                self.last_error = f"heartbeat read failed: {type(exc).__name__}: {exc}"
                self._event("heartbeat_read_failure", error=self.last_error)
                self._write_status("running")
            else:
                observed_master_pid = int(heartbeat["master_pid"])
                if observed_master_pid != self.master_pid:
                    stop_reason = "heartbeat_master_pid_changed"
                    self._event(
                        "master_identity_changed",
                        expected_master_pid=self.master_pid,
                        observed_master_pid=observed_master_pid,
                    )
                    break
                self.last_heartbeat_at = str(heartbeat.get("at"))
                heartbeat_time = parse_utc(self.last_heartbeat_at)
                if heartbeat_time is not None:
                    age = (dt.datetime.now(dt.timezone.utc) - heartbeat_time).total_seconds()
                    if age > self.args.stale_heartbeat_sec:
                        self._event(
                            "stale_heartbeat_observed",
                            heartbeat_at=self.last_heartbeat_at,
                            age_sec=age,
                            threshold_sec=self.args.stale_heartbeat_sec,
                        )
                self._sample_once(heartbeat)
                samples_taken += 1
                self._write_status("running")
                if self.args.once:
                    stop_reason = "once_complete"
                    break

            deadline = time.monotonic() + self.args.interval_sec
            while not self.stop_requested and time.monotonic() < deadline:
                time.sleep(min(0.25, max(0.0, deadline - time.monotonic())))

        if self.stop_requested:
            stop_reason = f"signal_{self.stop_signal}" if self.stop_signal else "requested_stop"
        self._close_sessions(stop_reason)
        self._event("watcher_stop", reason=stop_reason, samples_taken=samples_taken)
        self._write_status("stopped", stop_reason)
        return 0


def positive_finite(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive finite number")
    return parsed


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--campaign-root",
        type=Path,
        default=Path(os.environ.get("CAMPAIGN_ROOT", ".")),
        help="campaign root containing state/heartbeat.json",
    )
    parser.add_argument(
        "--interval-sec", type=positive_finite, default=DEFAULT_INTERVAL_SEC
    )
    parser.add_argument(
        "--query-timeout-sec", type=positive_finite, default=DEFAULT_QUERY_TIMEOUT_SEC
    )
    parser.add_argument(
        "--stale-heartbeat-sec",
        type=positive_finite,
        default=30.0,
        help="record an audit event when a live master's heartbeat exceeds this age",
    )
    parser.add_argument("--once", action="store_true", help="take one poll then exit")
    parser.add_argument(
        "--detach", action="store_true", help="spawn a detached watcher and verify startup"
    )
    parser.add_argument("--self-test", action="store_true", help="run CPU-only unit self-test")
    parser.add_argument("--launch-token", default=None, help=argparse.SUPPRESS)
    return parser.parse_args(argv)


def run_self_test() -> int:
    gpu_rows = "0, GPU-aaa\n7, GPU-bbb\n"
    compute_rows = (
        "GPU-aaa, 100, python, 5000\n"
        "GPU-aaa, 200, [Not Found], 866\n"
        "GPU-bbb, 300, python, N/A\n"
    )
    assert parse_gpu_rows(gpu_rows) == {0: "GPU-aaa", 7: "GPU-bbb"}
    parsed = parse_compute_rows(compute_rows)
    assert len(parsed["GPU-aaa"]) == 2
    assert parsed["GPU-bbb"][0]["used_gpu_memory_mib"] is None

    session = Session(
        session_id="selftest",
        campaign_id="selftest",
        physical_gpu=0,
        gpu_uuid="GPU-aaa",
        benchmark_key="gpu0",
        benchmark_pid=123,
        benchmark_started_at="2026-08-10T00:00:00+00:00",
        first_seen_at="2026-08-10T00:00:00+00:00",
        last_seen_at="2026-08-10T00:00:00+00:00",
    )
    normal = {
        "captured_at": "2026-08-10T00:00:01+00:00",
        "compute_app_count": 1,
        "possible_cotenancy": False,
        "compute_apps": [{"pid": 100}],
    }
    anomaly = {
        "captured_at": "2026-08-10T00:00:02+00:00",
        "compute_app_count": 3,
        "possible_cotenancy": True,
        "compute_apps": [{"pid": 100}, {"pid": 200}, {"pid": 300}],
    }
    assert session.observe(normal) == (False, False)
    assert session.observe(anomaly) == (True, False)
    assert session.observe(normal) == (False, True)
    assert session.samples == 3
    assert session.anomalous_samples == 1
    assert session.anomaly_events == 1
    assert session.max_compute_app_count == 3

    with tempfile.TemporaryDirectory(prefix="cotenancy-watcher-selftest-") as temporary:
        root = Path(temporary)
        atomic_write_json(root / "record.json", session.as_dict())
        atomic_append_jsonl(root / "events.jsonl", {"event": "one"})
        atomic_append_jsonl(root / "events.jsonl", {"event": "two"})
        assert load_json(root / "record.json")["max_compute_app_count"] == 3
        lines = (root / "events.jsonl").read_text(encoding="utf-8").splitlines()
        assert [json.loads(line)["event"] for line in lines] == ["one", "two"]
        lock_path = root / "watcher.lock"
        with exclusive_lock(lock_path):
            try:
                with exclusive_lock(lock_path):
                    raise AssertionError("second watcher lock unexpectedly succeeded")
            except RuntimeError:
                pass
    print("SELF_TEST_OK parser session atomic_json_jsonl single_instance_lock")
    return 0


def launch_detached(args: argparse.Namespace) -> int:
    campaign_root = args.campaign_root.expanduser().resolve(strict=True)
    audit_dir = campaign_root / "audit" / "gpu_cotenancy"
    audit_dir.mkdir(parents=True, exist_ok=True)
    log_path = audit_dir / "watcher.log"
    status_path = audit_dir / "status.json"
    launch_token = uuid.uuid4().hex
    child_args = [
        sys.executable,
        "-S",
        str(SCRIPT_PATH),
        "--campaign-root",
        str(campaign_root),
        "--interval-sec",
        str(args.interval_sec),
        "--query-timeout-sec",
        str(args.query_timeout_sec),
        "--stale-heartbeat-sec",
        str(args.stale_heartbeat_sec),
        "--launch-token",
        launch_token,
    ]
    if args.once:
        child_args.append("--once")
    with log_path.open("a", encoding="utf-8", buffering=1) as log_handle:
        process = subprocess.Popen(
            child_args,
            cwd=str(SCRIPT_PATH.parent),
            stdin=subprocess.DEVNULL,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            close_fds=True,
        )
    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(
                f"detached watcher exited during startup with rc={process.returncode}; "
                f"see {log_path}"
            )
        try:
            status = load_json(status_path)
        except (OSError, json.JSONDecodeError):
            status = None
        if (
            isinstance(status, Mapping)
            and int(status.get("watcher_pid", -1)) == process.pid
            and status.get("launch_token") == launch_token
            and status.get("state") == "running"
        ):
            print(
                json.dumps(
                    {
                        "detached": True,
                        "watcher_pid": process.pid,
                        "status": str(status_path),
                        "samples": str(audit_dir / "samples.jsonl"),
                        "events": str(audit_dir / "events.jsonl"),
                        "log": str(log_path),
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return 0
        time.sleep(0.1)
    raise RuntimeError(f"watcher startup timed out; see {log_path}")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.self_test:
        if args.detach or args.once:
            raise SystemExit("--self-test cannot be combined with --detach/--once")
        return run_self_test()
    if args.detach and args.once:
        raise SystemExit("--once is foreground-only and cannot be combined with --detach")
    if args.detach:
        return launch_detached(args)

    watcher = Watcher(args)
    signal.signal(signal.SIGINT, watcher.request_stop)
    signal.signal(signal.SIGTERM, watcher.request_stop)
    try:
        with exclusive_lock(watcher.lock_path):
            return watcher.run()
    except Exception as exc:
        print(f"WATCHER_FATAL {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
