"""priming_stream.daemon — resident localhost daemon for the bridge hot path.

The hook (``priming_stream.hooks.user_prompt_submit``) stays stdlib-only and
talks to a long-running daemon process that owns the embedding model +
ChromaDB client across CC sessions. See ARCHITECTURE.md.
"""
