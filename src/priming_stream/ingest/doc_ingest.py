"""Explicit document-ingestion helpers (v0.7-x-piece3, Phase C1).

Pure, side-effect-free identity + change-detection helpers shared by the
document-ingest path (the `/prime-ingest` skill: doc_plan -> doc_ingest
Workflow -> card_writer -> sleep-finalize). No LLM, no I/O beyond what the
caller passes in — every function here is unit-testable in isolation.

Three concepts:

- **doc_key** — a document's dedup identity (see ARCHITECTURE.md).
  One index_card per doc_key (enforced by the Phase-A partial unique
  index). Deterministic. Minimal by design (feedback: prefer minimal
  mechanism): an explicit key wins (vault migration passes the canonical
  kebab page-stem); otherwise the key is the absolute POSIX source path.
  Fancier derivation (DOI/URL normalisation, title-only author-year
  slugs + semantic dedup) is deferred per decision §9.

- **content_hash** — sha256 prefix over the source `.md` bytes, for
  own-doc change detection (decision §4). A regenerated/edited source
  yields a different hash -> the reconcile replaces the card.

- **card .md filename** — derived DETERMINISTICALLY from doc_key, so a
  regeneration overwrites the same file in place. This is what upholds the
  Phase-A reconcile invariant ("at most one card .md per doc_key in
  records_dir"); a random per-run rec_id filename would leave stale `.md`
  that churn the row every cycle.
"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path


def _slugify(s: str) -> str:
    """lowercase, non-alphanumerics → single hyphens, trimmed."""
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")


def _first_author_surname(authors: str | list[str]) -> str:
    """Best-effort surname of the first author. ``"Collins & Loftus"`` →
    ``collins``; ``"Anderson, J. R."`` → ``anderson``; ``"Aeschbach et al."`` →
    ``aeschbach``; a list → its first."""
    if isinstance(authors, (list, tuple)):
        authors = authors[0] if authors else ""
    first = re.split(r"[,&;]| and ", str(authors).strip())[0].strip()
    # Drop a trailing "et al" so it isn't mistaken for the surname.
    first = re.sub(r"\bet\s+al\.?\s*$", "", first, flags=re.I).strip()
    words = first.split()
    return _slugify(words[-1]) if words else ""


def canonical_doc_key(
    *,
    doi: str | None = None,
    url: str | None = None,
    authors: str | list[str] | None = None,
    year: str | int | None = None,
    title: str | None = None,
    fallback: str | None = None,
) -> str:
    """The canonical, type-agnostic document identity (piece3-B), in priority
    order DOI > URL > ``t:<author-year-titleslug>``. One key per document.

    - **DOI** (best): ``doi:<normalized>`` — protocol/``doi:`` prefix stripped,
      lowercased. Exact dedup, no semantic needed.
    - **URL** (online docs): ``url:<normalized>`` — protocol + ``www.`` +
      trailing slash stripped, lowercased.
    - **t:** universal fallback for everything else (own/partner docs, books,
      papers cited without DOI): first-author surname + 4-digit year +
      short title slug. ``fallback`` (e.g. a filename) substitutes for a
      missing title so a key can always be formed.

    Raises ``ValueError`` if no usable component is supplied.
    """
    if doi:
        d = doi.strip().lower()
        for pre in ("https://doi.org/", "http://doi.org/", "doi.org/", "doi:"):
            if d.startswith(pre):
                d = d[len(pre):]
        d = d.strip("/ ")
        if d:
            return "doi:" + d
    if url:
        u = url.strip().lower()
        for pre in ("https://", "http://"):
            if u.startswith(pre):
                u = u[len(pre):]
        if u.startswith("www."):
            u = u[4:]
        u = u.rstrip("/")
        if u:
            return "url:" + u
    parts: list[str] = []
    if authors:
        a = _first_author_surname(authors)
        if a:
            parts.append(a)
    if year is not None:
        y = re.sub(r"[^0-9]", "", str(year))[:4]
        if y:
            parts.append(y)
    name = title or fallback or ""
    if name:
        slug = _slugify(name)
        # Drop a leading citation echo so a title like "Aeschbach et al. 2025
        # Intelligence…" doesn't repeat the author/year already in the prefix
        # ("t:aeschbach-2025-aeschbach-et-al-2025-…" -> "t:aeschbach-2025-
        # intelligence-…"). Only strip when real content remains.
        if authors:
            sur = _first_author_surname(authors)
            if sur:
                stripped = re.sub(
                    rf"^{re.escape(sur)}(-et-al)?(-\d{{4}})?-?", "", slug,
                )
                if stripped:
                    slug = stripped
        slug = slug[:40].strip("-")
        if slug:
            parts.append(slug)
    key = "-".join(p for p in parts if p)
    if not key:
        raise ValueError("canonical_doc_key: no usable identity components")
    return "t:" + key


def content_hash(md: bytes | str) -> str:
    """sha256 prefix (16 hex) over the source content. Accepts bytes or
    str (str is utf-8 encoded). Stable across runs for identical input;
    any edit flips it, which is the regenerate signal for an index_card."""
    if isinstance(md, str):
        md = md.encode("utf-8")
    return hashlib.sha256(md).hexdigest()[:16]


def card_md_filename(doc_key: str) -> str:
    """Deterministic card `.md` filename for a doc_key.

    ``card_<sha256(doc_key)[:12]>.md`` — keyed by doc_key (NOT a rec_id) so
    regenerating a card overwrites the same file in place, upholding the
    Phase-A one-card-.md-per-doc_key invariant. The ``card_`` prefix keeps
    cards visually distinct from claim records (``rec_*.md``) in the same
    directory; both are reconciled uniformly by sleep-finalize.
    """
    digest = hashlib.sha256(doc_key.encode("utf-8")).hexdigest()[:12]
    return f"card_{digest}.md"


def make_card_record(
    *,
    rec_id: str,
    source_uri: str,
    doc_key: str,
    source: str | None,
    content_hash: str | None,
    created_at: str,
    body: str,
    title: str | None = None,
    provisional: bool = False,
):
    """Assemble an index_card :class:`~priming_stream.core.models.Record`.

    SQL-canonical replacement for the old ``render_card_md`` (which emitted
    ``.md`` frontmatter). Anchors are ``0/0`` — an index_card is a doc-level
    node, not a span into a chunk (tier-2 short-circuits cards and returns
    the body). ``body`` is the worker's card content as plain markdown,
    stored verbatim in ``summary``.
    """
    from priming_stream.core.models import Record

    return Record(
        id=rec_id,
        source_uri=source_uri,
        anchor_offset_start=0,
        anchor_offset_end=0,
        summary=body.strip(),
        created_at=created_at,
        kind="index_card",
        doc_key=doc_key,
        source=source or None,
        content_hash=content_hash or None,
        title=title,
        provisional=provisional,
    )


