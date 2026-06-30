"""bridge — v0.7-x waking-time live component.

The canonical public surface is ``build_priming`` + ``priming_items``
from ``bridge.working_set`` and ``render_buckets`` from
``daemon.render``. The walk entry point is ``walk_two_seeds`` from
``bridge.spreading``.
"""
from priming_stream.bridge.working_set import build_priming, priming_items

__all__ = ["build_priming", "priming_items"]
