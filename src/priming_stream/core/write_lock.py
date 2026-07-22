"""Single-writer lock for the consolidation write path.

Two write cycles (``sleep-auto`` runs, or any future consolidation
entry point) must never overlap. The old exists()-then-write lockfile
had a seconds-wide TOCTOU window — two near-simultaneous starts could
both pass it — so this module holds a REAL OS file lock:
``msvcrt.locking`` on Windows, ``fcntl.flock`` on POSIX — the same
pattern ``daemon.lifecycle.acquire_lock`` already uses.

Properties that replace the old heuristics:

* Acquisition is atomic — no check-then-write window.
* The lock dies with the holder process — no stale-age / PID-liveness
  logic needed; a crashed cycle releases implicitly.
* The lockFILE persists between cycles (its *content* is informational:
  ``pid timestamp``). Mere existence therefore means nothing — probes must
  use :func:`is_held`, never ``exists()``.

Stdlib-only, importable from any writer without further dependencies.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Historical name kept for continuity (docs / probes reference it).
LOCK_NAME = ".sleep_auto.lock"


def _lock_path(storage_dir: Path) -> Path:
    return Path(storage_dir) / LOCK_NAME


def acquire(storage_dir: Path, holder_note: str = "") -> object | None:
    """Try to take the write-cycle lock. Returns the open handle (keep it
    alive; closing releases the OS lock) or ``None`` when another live
    cycle holds it. Never raises."""
    try:
        Path(storage_dir).mkdir(parents=True, exist_ok=True)
        fh = open(_lock_path(storage_dir), "a+", encoding="utf-8")
    except OSError:
        return None
    try:
        fh.seek(0)  # msvcrt locks at the CURRENT position — pin byte 0
        if sys.platform == "win32":
            import msvcrt
            msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        try:
            fh.close()
        except OSError:
            pass
        return None
    # Informational content only — the OS lock is the guarantee.
    try:
        fh.seek(0)
        fh.truncate()
        note = f" {holder_note}" if holder_note else ""
        fh.write(f"{os.getpid()}{note}\n")
        fh.flush()
    except OSError:
        pass
    return fh


def release(handle: object) -> None:
    """Release + close. No-op on ``None``. Never raises."""
    if handle is None:
        return
    try:
        if sys.platform == "win32":
            import msvcrt
            try:
                handle.seek(0)  # type: ignore[attr-defined]
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)  # type: ignore[attr-defined]
            except OSError:
                pass
        else:
            import fcntl
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)  # type: ignore[attr-defined]
            except OSError:
                pass
    finally:
        try:
            handle.close()  # type: ignore[attr-defined]
        except Exception:
            pass


def is_held(storage_dir: Path) -> bool:
    """True when a live cycle currently holds the lock.

    Probes by attempting the OS lock (atomic, sub-ms) and releasing it
    immediately on success. This is the ONLY correct cheap gate — the
    lockfile persists between cycles, so ``exists()`` would read "held"
    forever after the first cycle ran.
    """
    handle = acquire(storage_dir)
    if handle is None:
        return True
    release(handle)
    return False
