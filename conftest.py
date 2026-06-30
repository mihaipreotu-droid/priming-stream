import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# Hook tests drive user_prompt_submit.main() without patching load_config;
# without this, every suite run would append test echoes to the REAL
# storage/episodic/echoes.jsonl. Echo tests delenv it via monkeypatch.
os.environ.setdefault("PRIMING_STREAM_ECHOES_OFF", "1")
# Likewise for active-use telemetry: dispatching a real MCP tool in a test
# would append to the REAL storage/episodic/usage.jsonl. Usage tests delenv it.
os.environ.setdefault("PRIMING_STREAM_USAGE_OFF", "1")


@pytest.fixture
def migrated_db(tmp_path):
    from priming_stream.core.db import connect
    from priming_stream.core.schema import apply_migrations

    conn = connect(tmp_path / "graph.db")
    apply_migrations(conn)
    yield conn
    conn.close()


@pytest.fixture
def config():
    from priming_stream.core.config import load_config

    return load_config(Path("___nonexistent_settings___.toml"))
