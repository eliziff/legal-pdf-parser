#!/usr/bin/env python3
"""Run one command at BelowNormal priority with a hard process-tree timeout."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


BELOW_NORMAL_PRIORITY_CLASS = 0x00004000
CREATE_NEW_PROCESS_GROUP = 0x00000200
JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000
JOB_OBJECT_EXTENDED_LIMIT_INFORMATION = 9


class IoCounters(ctypes.Structure):
    _fields_ = [(name, ctypes.c_uint64) for name in (
        "read_operations", "write_operations", "other_operations",
        "read_bytes", "write_bytes", "other_bytes",
    )]


class BasicLimitInformation(ctypes.Structure):
    _fields_ = [
        ("per_process_time", ctypes.c_int64),
        ("per_job_time", ctypes.c_int64),
        ("limit_flags", ctypes.c_uint32),
        ("minimum_working_set", ctypes.c_size_t),
        ("maximum_working_set", ctypes.c_size_t),
        ("active_process_limit", ctypes.c_uint32),
        ("affinity", ctypes.c_size_t),
        ("priority_class", ctypes.c_uint32),
        ("scheduling_class", ctypes.c_uint32),
    ]


class ExtendedLimitInformation(ctypes.Structure):
    _fields_ = [
        ("basic", BasicLimitInformation),
        ("io", IoCounters),
        ("process_memory_limit", ctypes.c_size_t),
        ("job_memory_limit", ctypes.c_size_t),
        ("peak_process_memory", ctypes.c_size_t),
        ("peak_job_memory", ctypes.c_size_t),
    ]


def canonical_environment() -> tuple[dict[str, str], int]:
    entries: dict[str, tuple[str, str]] = {}
    duplicates = 0
    for key, value in os.environ.items():
        folded = key.casefold()
        if folded in entries:
            duplicates += 1
        entries[folded] = (key, value)
    return {key: value for key, value in entries.values()}, duplicates


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def write_receipt(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def create_kill_job(kernel32: ctypes.WinDLL, process_handle: int) -> int:
    kernel32.CreateJobObjectW.argtypes = [ctypes.c_void_p, ctypes.c_wchar_p]
    kernel32.CreateJobObjectW.restype = ctypes.c_void_p
    kernel32.SetInformationJobObject.argtypes = [
        ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_uint32,
    ]
    kernel32.SetInformationJobObject.restype = ctypes.c_int
    kernel32.AssignProcessToJobObject.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    kernel32.AssignProcessToJobObject.restype = ctypes.c_int
    kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
    kernel32.CloseHandle.restype = ctypes.c_int
    job = kernel32.CreateJobObjectW(None, None)
    if not job:
        raise OSError(ctypes.get_last_error(), "failed to create process job")
    limits = ExtendedLimitInformation()
    limits.basic.limit_flags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
    if not kernel32.SetInformationJobObject(
        job, JOB_OBJECT_EXTENDED_LIMIT_INFORMATION,
        ctypes.byref(limits), ctypes.sizeof(limits),
    ) or not kernel32.AssignProcessToJobObject(job, ctypes.c_void_p(process_handle)):
        error = ctypes.get_last_error()
        kernel32.CloseHandle(job)
        raise OSError(error, "failed to configure process job")
    return int(job)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cwd", required=True, type=Path)
    parser.add_argument("--receipt", required=True, type=Path)
    parser.add_argument("--timeout", required=True, type=float)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("a command is required after --")
    if os.name != "nt":
        raise SystemExit("run_bounded.py currently requires Windows")

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.GetCurrentProcess.restype = ctypes.c_void_p
    kernel32.SetPriorityClass.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
    kernel32.SetPriorityClass.restype = ctypes.c_int
    kernel32.GetPriorityClass.argtypes = [ctypes.c_void_p]
    kernel32.GetPriorityClass.restype = ctypes.c_uint32
    if not kernel32.SetPriorityClass(kernel32.GetCurrentProcess(), BELOW_NORMAL_PRIORITY_CLASS):
        raise OSError(ctypes.get_last_error(), "failed to set runner priority")
    environment, duplicate_count = canonical_environment()
    runner_priority = kernel32.GetPriorityClass(kernel32.GetCurrentProcess())
    started = time.perf_counter()
    process = subprocess.Popen(
        command,
        cwd=args.cwd.resolve(),
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        creationflags=BELOW_NORMAL_PRIORITY_CLASS | CREATE_NEW_PROCESS_GROUP,
    )
    try:
        job = create_kill_job(kernel32, int(process._handle))  # type: ignore[attr-defined]
    except Exception:
        process.kill()
        process.wait(timeout=5)
        raise
    observed_priority = kernel32.GetPriorityClass(int(process._handle))  # type: ignore[attr-defined]
    timed_out = False
    tree_kill_exit: int | None = None
    try:
        stdout, stderr = process.communicate(timeout=args.timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        tree_kill_exit = 0 if kernel32.CloseHandle(job) else ctypes.get_last_error()
        job = 0
        stdout, stderr = process.communicate(timeout=5)
    finally:
        if job:
            kernel32.CloseHandle(job)
    elapsed = time.perf_counter() - started
    command_bytes = json.dumps(command, ensure_ascii=False, separators=(",", ":")).encode()
    receipt: dict[str, object] = {
        "schema_version": "legalpdf.bounded-command.v1",
        "runner_sha256": sha256_bytes(Path(__file__).read_bytes()),
        "command_executable": Path(command[0]).name,
        "command_argc": len(command),
        "command_sha256": sha256_bytes(command_bytes),
        "cwd": str(args.cwd.resolve()),
        "timeout_seconds": args.timeout,
        "elapsed_seconds": round(elapsed, 6),
        "pid": process.pid,
        "timed_out": timed_out,
        "exit_code": 124 if timed_out else process.returncode,
        "tree_kill_exit": tree_kill_exit,
        "below_normal_expected": BELOW_NORMAL_PRIORITY_CLASS,
        "runner_priority_observed": runner_priority,
        "child_priority_observed": observed_priority,
        "environment_case_duplicates_removed": duplicate_count,
        "stdout_bytes": len(stdout),
        "stdout_sha256": sha256_bytes(stdout),
        "stderr_bytes": len(stderr),
        "stderr_sha256": sha256_bytes(stderr),
    }
    write_receipt(args.receipt.resolve(), receipt)
    sys.stdout.buffer.write(stdout)
    sys.stderr.buffer.write(stderr)
    if timed_out:
        return 124
    if observed_priority != BELOW_NORMAL_PRIORITY_CLASS:
        return 125
    return process.returncode


if __name__ == "__main__":
    raise SystemExit(main())
