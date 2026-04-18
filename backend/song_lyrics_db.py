"""
Curated timed-lyrics DB — intentionally empty.

The pipeline uses lrclib.net for all lyrics retrieval; no hardcoded
per-song data is kept here so every song is treated identically.
The public API is preserved so existing imports do not break.
"""
from __future__ import annotations
from typing import Optional


CURATED: dict[str, dict] = {}
_ALIASES: dict[str, str] = {}


def _norm(s: str) -> str:
    import re
    s = s.lower()
    s = re.sub(r"[\(\[].*?[\)\]]", "", s)
    s = re.sub(r"\b(feat\.?|ft\.?|featuring|versus|vs\.?)\b.*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def lookup(song: str, artist: str = "") -> Optional[list[dict]]:
    return None


def get_language(song: str) -> Optional[str]:
    return None
