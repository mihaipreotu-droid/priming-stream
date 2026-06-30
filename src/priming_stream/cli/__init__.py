"""Priming Stream command-line interface.

The entry point is ``priming_stream.cli.main:main``. This package intentionally does
not re-export ``main`` — doing so imports the ``main`` submodule eagerly and
triggers a ``runpy`` RuntimeWarning when the CLI is launched via
``python -m priming_stream.cli.main`` (as the idle scheduler does).
"""
