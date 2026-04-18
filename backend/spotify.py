"""
Spotify Web API — OAuth 2.0 Authorization Code Flow + playback helpers.

Credentials are read from os.environ; callers (main.py) are responsible for
calling load_dotenv() before importing this module.
"""
from __future__ import annotations

import base64
import logging
import os
import time
from urllib.parse import urlencode

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)

CLIENT_ID     = os.environ.get("SPOTIFY_CLIENT_ID", "")
CLIENT_SECRET = os.environ.get("SPOTIFY_CLIENT_SECRET", "")
REDIRECT_URI  = os.environ.get("SPOTIFY_REDIRECT_URI", "http://127.0.0.1:8000/callback")
SCOPE         = "user-read-playback-state user-read-currently-playing"

_AUTH_BASE = "https://accounts.spotify.com"
_API_BASE  = "https://api.spotify.com/v1"

# Shared session with automatic retry on transient errors (503, 429, network issues)
_retry_strategy = Retry(
    total=3,
    backoff_factor=0.5,           # waits: 0.5s, 1s, 2s
    status_forcelist={429, 500, 502, 503, 504},
    allowed_methods={"GET", "POST"},
    raise_on_status=False,
)
_adapter = HTTPAdapter(max_retries=_retry_strategy)
_session = requests.Session()
_session.mount("https://", _adapter)
_session.mount("http://", _adapter)


def get_auth_url(state: str = "") -> str:
    params = {
        "client_id":     CLIENT_ID,
        "response_type": "code",
        "redirect_uri":  REDIRECT_URI,
        "scope":         SCOPE,
        "show_dialog":   "false",
    }
    if state:
        params["state"] = state
    return f"{_AUTH_BASE}/authorize?{urlencode(params)}"


def _basic_auth_header() -> str:
    encoded = base64.b64encode(f"{CLIENT_ID}:{CLIENT_SECRET}".encode()).decode()
    return f"Basic {encoded}"


def exchange_code(code: str) -> dict:
    """Exchange the OAuth authorization code for access + refresh tokens."""
    resp = _session.post(
        f"{_AUTH_BASE}/api/token",
        headers={
            "Authorization": _basic_auth_header(),
            "Content-Type":  "application/x-www-form-urlencoded",
        },
        data={
            "grant_type":   "authorization_code",
            "code":         code,
            "redirect_uri": REDIRECT_URI,
        },
        timeout=12,
    )
    resp.raise_for_status()
    return resp.json()


def refresh_access_token(refresh_token: str) -> dict:
    """Use the refresh token to obtain a new access token."""
    resp = _session.post(
        f"{_AUTH_BASE}/api/token",
        headers={
            "Authorization": _basic_auth_header(),
            "Content-Type":  "application/x-www-form-urlencoded",
        },
        data={
            "grant_type":    "refresh_token",
            "refresh_token": refresh_token,
        },
        timeout=12,
    )
    resp.raise_for_status()
    return resp.json()


def get_currently_playing(access_token: str) -> dict | None:
    """
    Return {song, artist, progress_seconds, is_playing, preview_url, track_id}
    or None when nothing is playing / player is paused.

    Raises requests.HTTPError on non-2xx responses that exhaust retries.
    """
    resp = _session.get(
        f"{_API_BASE}/me/player/currently-playing",
        headers={"Authorization": f"Bearer {access_token}"},
        timeout=6,
    )

    # 204 = player is active but nothing is playing; empty body = same
    if resp.status_code == 204 or not resp.content:
        return None

    resp.raise_for_status()
    data = resp.json()

    if not data.get("is_playing"):
        return None

    item    = data.get("item") or {}
    artists = ", ".join(a["name"] for a in item.get("artists", []))
    return {
        "song":             item.get("name", "Unknown"),
        "artist":           artists,
        "progress_seconds": data.get("progress_ms", 0) / 1000,
        "is_playing":       True,
        "preview_url":      item.get("preview_url"),
        "track_id":         item.get("id"),
    }
