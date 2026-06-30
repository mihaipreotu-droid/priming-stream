"""hooks — Claude Code waking-time hooks.

Each hook reads a JSON event from stdin and writes a JSON result to stdout.
Hooks read the active graph only and write exclusively to the episodic log
and small JSON state files; they never write the durable graph.
"""
