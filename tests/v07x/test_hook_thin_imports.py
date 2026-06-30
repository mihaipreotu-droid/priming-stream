"""Static grep gate (spec §D10, §D11) — ensures the hot-path modules
remain stdlib-pure.

We read the source files as text (not via ``import``) and assert that no
heavyweight identifier appears anywhere in them. The grep is intentionally
substring-based: regardless of whether the import is ``import chromadb``,
``from chromadb import …``, or inside a deferred local import, it will
be caught.
"""
from __future__ import annotations

from pathlib import Path

import priming_stream


_HOT_PATH_FILES = [
    "hooks/user_prompt_submit.py",
    "daemon/client.py",
    "daemon/fallback_lexical.py",
    "daemon/render.py",
]

_FORBIDDEN_TOKENS = [
    "chromadb",
    "fastembed",
    "onnxruntime",
    "torch",
    "numpy",
    "priming_stream.bridge",
    "priming_stream.integrations",
]


def _src_dir() -> Path:
    return Path(priming_stream.__file__).parent


def _read_file(rel: str) -> str:
    return (_src_dir() / rel).read_text(encoding="utf-8")


def test_d10_hook_has_no_heavyweight_imports():
    src = _read_file("hooks/user_prompt_submit.py")
    for tok in _FORBIDDEN_TOKENS:
        assert tok not in src, (
            f"forbidden token {tok!r} appears in user_prompt_submit.py"
        )


def test_d11_client_has_no_heavyweight_imports():
    src = _read_file("daemon/client.py")
    for tok in _FORBIDDEN_TOKENS:
        assert tok not in src, f"forbidden token {tok!r} in daemon/client.py"


def test_d11_fallback_lexical_has_no_heavyweight_imports():
    src = _read_file("daemon/fallback_lexical.py")
    for tok in _FORBIDDEN_TOKENS:
        assert tok not in src, (
            f"forbidden token {tok!r} in daemon/fallback_lexical.py"
        )


def test_d11_render_has_no_heavyweight_imports():
    src = _read_file("daemon/render.py")
    for tok in _FORBIDDEN_TOKENS:
        assert tok not in src, f"forbidden token {tok!r} in daemon/render.py"


def test_runtime_import_does_not_load_heavyweights(tmp_path):
    """Dynamic check: importing the hook in a fresh interpreter must not
    pull the embedding / vector / bridge modules into ``sys.modules``.

    Runs in a subprocess so it cannot pollute the parent test session's
    module cache (deleting cached chromadb/fastembed modules in-process
    would break unrelated tests that depend on them).
    """
    import os
    import subprocess
    import sys

    worktree_src = Path(__file__).resolve().parents[2] / "src"
    env = os.environ.copy()
    pp = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        str(worktree_src) + (os.pathsep + pp if pp else "")
    )
    # Belt-and-braces: this test only imports the hook (doesn't call main),
    # but if any future regression turned the import into a side effect that
    # touched the daemon, we don't want a real detached daemon spawning.
    env["PRIMING_STREAM_DISABLE_AUTOSTART"] = "1"
    env["PRIMING_STREAM_DAEMON_DIR"] = str(tmp_path / "daemon_dir")
    # Subprocess body: import the hook, then dump module names that
    # belong to the forbidden set.
    code = (
        "import sys\n"
        "import priming_stream.hooks.user_prompt_submit  # noqa: F401\n"
        "HEAVY = ('chromadb', 'fastembed', 'onnxruntime', 'torch', "
        "'numpy', 'priming_stream.bridge', 'priming_stream.integrations')\n"
        "leaked = [n for n in sys.modules "
        "if any(n == p or n.startswith(p + '.') for p in HEAVY)]\n"
        "import json\n"
        "print(json.dumps(leaked))\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env=env,
        timeout=20,
    )
    assert proc.returncode == 0, (
        f"subprocess failed: stderr={proc.stderr!r}"
    )
    import json
    leaked = json.loads(proc.stdout.strip())
    assert not leaked, f"hot path leaks heavy modules: {leaked}"


def test_unknown_slash_does_not_load_heavyweights(tmp_path):
    """An unknown slash prompt (e.g. ``/typo``) must NOT pull
    ``priming_stream.bridge`` / chromadb / fastembed into sys.modules.

    Regression for M-4: previously every leading ``/`` triggered
    ``_slash_dispatch`` import, which warm-loaded the heavy command
    layer even before recognizing the command. The mini-parser now
    gates that warm-load on a KNOWN command name.
    """
    import json as _json
    import os
    import subprocess
    import sys

    worktree_src = Path(__file__).resolve().parents[2] / "src"
    env = os.environ.copy()
    pp = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        str(worktree_src) + (os.pathsep + pp if pp else "")
    )
    # Don't spawn a real detached daemon during the test and don't talk
    # to a real one on the developer's machine.
    env["PRIMING_STREAM_DISABLE_AUTOSTART"] = "1"
    env["PRIMING_STREAM_DAEMON_DIR"] = str(tmp_path / "daemon_dir")

    code = (
        "import sys, json\n"
        "import io\n"
        "sys.stdin = io.StringIO(json.dumps({'prompt': '/typo foo'}))\n"
        "from priming_stream.hooks import user_prompt_submit\n"
        "user_prompt_submit.main()\n"
        "HEAVY = ('chromadb', 'fastembed', 'onnxruntime', 'torch', "
        "'numpy', 'priming_stream.bridge', 'priming_stream.integrations')\n"
        "leaked = [n for n in sys.modules "
        "if any(n == p or n.startswith(p + '.') for p in HEAVY)]\n"
        "sys.stderr.write(json.dumps(leaked))\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env=env,
        timeout=20,
    )
    assert proc.returncode == 0, (
        f"subprocess failed: stderr={proc.stderr!r}"
    )
    # The hook itself wrote a JSON envelope to stdout (we don't care
    # what, since with no daemon + no DB the tier-3 empty {} path will
    # have fired). The module-name dump is on stderr.
    leaked = _json.loads(proc.stderr.strip())
    assert not leaked, (
        f"unknown slash warm-loaded heavy modules: {leaked}"
    )
