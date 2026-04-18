"""
Lyrics store and time-sync utilities.

Fetches real time-synced (LRC) lyrics from lrclib.net — free, no API key.
Results are cached in a bounded LRU cache (200 songs) for the server lifetime.

Query strategy (tried in order until lyrics are found):
  1. Exact song + all artists
  2. Song + first artist only  (handles "A, B feat. C" mismatches)
  3. Song title only           (most permissive; avoids artist-name noise)
  4. Clean title (strip parentheticals) + first artist
  5. lrclib /api/search        (fuzzy full-text search as last resort)
"""
from __future__ import annotations

import bisect
import logging
import re
import time
from collections import OrderedDict
from typing import Optional

import requests

logger = logging.getLogger(__name__)

_LRCLIB_GET    = "https://lrclib.net/api/get"
_LRCLIB_SEARCH = "https://lrclib.net/api/search"
_TIMEOUT       = 6   # seconds per request
_MAX_RETRIES   = 2   # retry on timeout / 5xx

# Shared session reuses TCP connections across calls (significant latency saving)
_session = requests.Session()
_session.headers.update({"User-Agent": "Syntora/2.1 (github.com/syntora)"})

# Bounded LRU cache — evicts least-recently-used entry when full
_CACHE_MAX = 200
_CACHE: OrderedDict[str, list[dict]] = OrderedDict()


def _cache_get(key: str) -> list[dict] | None:
    if key not in _CACHE:
        return None
    _CACHE.move_to_end(key)  # mark as recently used
    return _CACHE[key]


def _cache_set(key: str, value: list[dict]) -> None:
    _CACHE[key] = value
    _CACHE.move_to_end(key)
    if len(_CACHE) > _CACHE_MAX:
        _CACHE.popitem(last=False)  # evict LRU entry


# ── LRC parser ────────────────────────────────────────────────────────────────

def parse_lrc(lrc_text: str) -> list[dict]:
    """Parse LRC-format text → sorted list of {time, text}."""
    pattern = re.compile(r"\[(\d{1,2}):(\d{2})(?:\.(\d+))?\](.*)")
    lines: list[dict] = []
    for raw in lrc_text.splitlines():
        m = pattern.match(raw.strip())
        if not m:
            continue
        centis = (m.group(3) or "00")[:2].ljust(2, "0")
        total  = int(m.group(1)) * 60 + int(m.group(2)) + int(centis) / 100
        text   = m.group(4).strip()
        if text:
            lines.append({"time": total, "text": text})
    return sorted(lines, key=lambda x: x["time"])


def _extract_lines(data: dict) -> list[dict]:
    """Pull synced or plain lyrics from a lrclib response dict."""
    synced = (data.get("syncedLyrics") or "").strip()
    if synced:
        lines = parse_lrc(synced)
        if lines:
            return lines

    plain = (data.get("plainLyrics") or "").strip()
    if plain:
        lines = [
            {"time": float(i * 3), "text": t.strip()}
            for i, t in enumerate(plain.splitlines())
            if t.strip()
        ]
        if lines:
            return lines

    return []


def _clean_title(title: str) -> str:
    """Strip parentheticals like '(From Premam)' or '[Official]'."""
    title = re.sub(r"\s*[\(\[].*?[\)\]]\s*", " ", title)
    return re.sub(r"\s{2,}", " ", title).strip()


def _first_artist(artist: str) -> str:
    """Return the primary artist — split on comma / 'feat.' / '&'."""
    return re.split(r",|feat\.|&", artist, flags=re.IGNORECASE)[0].strip()


# ── HTTP helpers with retry ───────────────────────────────────────────────────

def _get_json(url: str, params: dict) -> dict | list | None:
    """GET *url* with *params*, retrying on timeout and 5xx responses."""
    for attempt in range(_MAX_RETRIES + 1):
        try:
            r = _session.get(url, params=params, timeout=_TIMEOUT)
            if r.status_code == 200:
                return r.json()
            if r.status_code in (429, 503) and attempt < _MAX_RETRIES:
                time.sleep(2 ** attempt)
                continue
            return None
        except requests.Timeout:
            if attempt < _MAX_RETRIES:
                logger.debug("lrclib timeout on attempt %d, retrying…", attempt + 1)
                continue
            logger.debug("lrclib timed out after %d attempts", _MAX_RETRIES + 1)
        except requests.ConnectionError as exc:
            logger.debug("lrclib connection error: %s", exc)
            break
    return None


def _get_exact(params: dict) -> list[dict]:
    data = _get_json(_LRCLIB_GET, params)
    return _extract_lines(data) if isinstance(data, dict) else []


def _search(song: str, artist: str) -> list[dict]:
    """Full-text search — returns best match from result list."""
    q = f"{song} {artist}".strip()
    data = _get_json(_LRCLIB_SEARCH, {"q": q})
    if isinstance(data, list):
        for item in data:
            lines = _extract_lines(item)
            if lines:
                logger.info(
                    "lrclib search hit: %s – %s",
                    item.get("trackName"), item.get("artistName"),
                )
                return lines
    return []


# ── Public API ────────────────────────────────────────────────────────────────

def get_lyrics(song: str = "", artist: str = "") -> list[dict]:
    """
    Return time-synced lyrics for song/artist.
    Checks LRU cache first, then lrclib, returns [] when nothing found.
    """
    key = f"{song.lower().strip()}::{artist.lower().strip()}"
    cached = _cache_get(key)
    if cached is not None:
        return cached

    song   = song.strip()
    artist = artist.strip()
    clean  = _clean_title(song)
    first  = _first_artist(artist)

    # Ordered attempts: most specific → least specific
    attempts = [
        {"artist_name": artist, "track_name": song},
        {"artist_name": first,  "track_name": song},
        {"track_name": song},
        {"artist_name": first,  "track_name": clean},
        {"track_name": clean},
    ]

    seen: set[str] = set()
    for params in attempts:
        sig = str(sorted(params.items()))
        if sig in seen:
            continue
        seen.add(sig)

        lines = _get_exact(params)
        if lines:
            logger.info("lrclib: found '%s' with params %s", song, params)
            _cache_set(key, lines)
            return lines

    # Last resort: full-text search
    lines = _search(song, artist)
    if lines:
        _cache_set(key, lines)
        return lines

    logger.info("lrclib: no lyrics found for '%s' by '%s'", song, artist)
    _cache_set(key, [])
    return []


def get_current_line(lyrics: list[dict], progress_seconds: float) -> Optional[dict]:
    """
    Return the latest lyric line whose timestamp <= progress_seconds.
    O(log n) binary search — faster than linear scan for long songs.
    """
    if not lyrics:
        return None
    times = [line["time"] for line in lyrics]
    idx = bisect.bisect_right(times, progress_seconds) - 1
    return lyrics[idx] if idx >= 0 else None
