"""Daemon lifecycle helpers — endpoint file, lockfile, pid checks, autostart.

Stdlib-only. Used by both the daemon (server.py) at startup/shutdown and
the hook client (client.py) to discover or autostart the daemon.

Conventions (spec §5):
- Windows lockfile uses ``msvcrt.locking(LK_NBLCK, 1)``; POSIX uses
  ``fcntl.flock(LOCK_EX | LOCK_NB)``.
- The lockfile handle MUST be kept alive for the duration of the lock;
  closing the file releases the lock. Caller (daemon ``main()``) owns
  the handle and calls :func:`release_lock` on shutdown.
- Endpoint file writes are atomic via ``os.replace(tmp, final)``.
- :func:`read_endpoint`, :func:`is_pid_alive`, :func:`is_endpoint_stale`
  are defensive — they swallow every exception and return ``None``/
  ``False``/``True`` rather than raise.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

DAEMON_DIR_ENV = "PRIMING_STREAM_DAEMON_DIR"
DISABLE_AUTOSTART_ENV = "PRIMING_STREAM_DISABLE_AUTOSTART"

_REQUIRED_ENDPOINT_FIELDS = ("host", "port", "pid", "started_at", "version")


def daemon_dir() -> Path:
    """Resolve the daemon state directory.

    Honors ``$PRIMING_STREAM_DAEMON_DIR`` (tests use this for
    isolation); otherwise defaults to ``%APPDATA%\\priming-stream``
    on Windows and ``~/.priming-stream`` on POSIX. Creates the
    directory if missing.
    """
    env = os.environ.get(DAEMON_DIR_ENV)
    if env:
        d = Path(env)
    elif sys.platform == "win32":
        appdata = os.environ.get("APPDATA") or os.path.expanduser("~")
        d = Path(appdata) / "priming-stream"
    else:
        d = Path.home() / ".priming-stream"
    d.mkdir(parents=True, exist_ok=True)
    return d


def endpoint_path() -> Path:
    return daemon_dir() / "daemon.json"


def lockfile_path() -> Path:
    return daemon_dir() / "daemon.lock"


def log_path() -> Path:
    return daemon_dir() / "daemon.log"


def read_endpoint() -> dict | None:
    """Return endpoint file contents as dict, or None if missing/unparseable.

    Never raises — file-not-found, JSON decode errors, permission
    issues, non-dict payloads all collapse to ``None``.
    """
    try:
        with open(endpoint_path(), "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except Exception:
        return None
    if not isinstance(data, dict):
        return None
    return data


def write_endpoint(
    host: str,
    port: int,
    pid: int,
    started_at: str,
    version: str,
) -> None:
    """Atomically write the endpoint file. JSON, UTF-8."""
    payload = {
        "host": host,
        "port": port,
        "pid": pid,
        "started_at": started_at,
        "version": version,
    }
    final = endpoint_path()
    tmp = final.with_suffix(final.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, final)


def remove_endpoint() -> None:
    """Remove the endpoint file if present. No-op if absent."""
    try:
        endpoint_path().unlink()
    except FileNotFoundError:
        pass
    except Exception:
        # Best-effort cleanup; do not raise from shutdown paths.
        pass


def acquire_lock() -> object:
    """Acquire an OS-level lock on :func:`lockfile_path`.

    Returns the open file handle, which the caller MUST keep alive for
    the duration of the lock (closing the file releases the OS lock on
    both Windows and POSIX). Raises ``BlockingIOError`` (or ``OSError``)
    if another process already holds the lock.

    Windows: ``msvcrt.locking(fh.fileno(), LK_NBLCK, 1)``.
    POSIX:  ``fcntl.flock(fh, LOCK_EX | LOCK_NB)``.
    """
    path = lockfile_path()
    # Open in append+read mode so the file is created if missing and we
    # don't truncate an existing lockfile (some platforms key the OS
    # lock by file content/handle).
    fh = open(path, "a+", encoding="utf-8")
    try:
        if sys.platform == "win32":
            import msvcrt
            # Lock the first byte; LK_NBLCK = non-blocking exclusive.
            # On a freshly created empty file there are zero bytes to
            # lock, but msvcrt happily locks past EOF.
            msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except Exception:
        fh.close()
        raise
    return fh


def release_lock(handle: object) -> None:
    """Release the lock and close the handle. No-op if handle is None."""
    if handle is None:
        return
    try:
        if sys.platform == "win32":
            import msvcrt
            try:
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)  # type: ignore[attr-defined]
            except Exception:
                pass
        else:
            import fcntl
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)  # type: ignore[attr-defined]
            except Exception:
                pass
    finally:
        try:
            handle.close()  # type: ignore[attr-defined]
        except Exception:
            pass


def is_pid_alive(pid: int) -> bool:
    """True if a process with ``pid`` exists.

    On Windows we use ``ctypes.windll.kernel32.OpenProcess`` with
    ``PROCESS_QUERY_LIMITED_INFORMATION`` (0x1000); a non-zero handle
    means the PID is alive (caller closes the handle). On POSIX we
    use ``os.kill(pid, 0)``.

    Edge cases: pid <= 0 always returns False. Never raises.
    """
    try:
        if pid is None or pid <= 0:
            return False
        if sys.platform == "win32":
            import ctypes
            PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
            kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
            handle = kernel32.OpenProcess(
                PROCESS_QUERY_LIMITED_INFORMATION, False, int(pid)
            )
            if not handle:
                return False
            try:
                # Distinguish a live process from a zombie/exited one
                # whose handle is still open: STILL_ACTIVE == 259.
                exit_code = ctypes.c_ulong(0)
                ok = kernel32.GetExitCodeProcess(
                    handle, ctypes.byref(exit_code)
                )
                if not ok:
                    return True  # handle is valid; treat as alive
                return exit_code.value == 259
            finally:
                kernel32.CloseHandle(handle)
        else:
            try:
                os.kill(int(pid), 0)
            except ProcessLookupError:
                return False
            except PermissionError:
                # Process exists but we lack signal permission.
                return True
            return True
    except Exception:
        return False


def is_endpoint_stale(info: dict | None) -> bool:
    """True if ``info`` is None, missing required fields, or pid is dead.

    Required fields: host (str), port (int), pid (int), started_at
    (str), version (str). Never raises.
    """
    try:
        if not isinstance(info, dict):
            return True
        for f in _REQUIRED_ENDPOINT_FIELDS:
            if f not in info:
                return True
        if not isinstance(info.get("host"), str):
            return True
        if not isinstance(info.get("port"), int):
            return True
        if not isinstance(info.get("pid"), int):
            return True
        if not isinstance(info.get("started_at"), str):
            return True
        if not isinstance(info.get("version"), str):
            return True
        return not is_pid_alive(info["pid"])
    except Exception:
        return True


def autostart_daemon() -> None:
    """Spawn a detached daemon subprocess and return immediately.

    Command: ``[sys.executable, "-m", "priming_stream.daemon.server"]``.

    Windows: ``CREATE_NEW_PROCESS_GROUP | DETACHED_PROCESS``, all stdio
    fds redirected to ``DEVNULL``. POSIX: ``start_new_session=True``,
    same DEVNULL redirection. Caller does not wait on the subprocess.

    Tests set ``PRIMING_STREAM_DISABLE_AUTOSTART=1`` to skip the spawn
    entirely — without that guard, hook-as-subprocess tests would leak
    a real detached daemon (which then loads fastembed + holds the
    tmp_path lockfile past pytest teardown).
    """
    if os.environ.get(DISABLE_AUTOSTART_ENV) == "1":
        return
    cmd = [sys.executable, "-m", "priming_stream.daemon.server"]
    kwargs: dict = {
        "stdin": subprocess.DEVNULL,
        "stdout": subprocess.DEVNULL,
        "stderr": subprocess.DEVNULL,
        "close_fds": True,
    }
    if sys.platform == "win32":
        creationflags = 0
        creationflags |= getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        creationflags |= getattr(subprocess, "DETACHED_PROCESS", 0)
        kwargs["creationflags"] = creationflags
    else:
        kwargs["start_new_session"] = True
    subprocess.Popen(cmd, **kwargs)
