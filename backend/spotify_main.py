#!/usr/bin/env python3
"""
Spotify Lyrics Transliterator — terminal entry point.

Usage
-----
  cd backend/
  python spotify_main.py

Reads SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET from backend/.env.

Flow
----
  1. HTTP server starts on localhost:5175 to receive the OAuth callback.
  2. Spotify authorization URL is printed (and opened in the browser).
  3. After login, Spotify redirects to localhost:5175/callback.
  4. Code is exchanged for access + refresh tokens.
  5. Sync loop polls Spotify every second; prints transliterated lyrics.
"""
from __future__ import annotations

import sys
import os
from pathlib import Path

# Load .env before any module reads os.environ
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / ".env")
except ImportError:
    pass

import time
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse, parse_qs

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import spotify
import lyrics as lyrics_mod
from translator import translate

CALLBACK_PORT       = 5175
POLL_INTERVAL_SECS  = 1
AUTH_TIMEOUT_SECS   = 120
SYNC_OFFSET_SECONDS = 0.5  # compensate for API round-trip latency

_tokens: dict[str, object] = {}


def _token_expiry(data: dict) -> float:
    return time.time() + data.get("expires_in", 3600) - 60


def _ensure_valid_token() -> None:
    if _tokens.get("expires_at", 0) < time.time():
        print("🔄  Refreshing Spotify token…", flush=True)
        try:
            data = spotify.refresh_access_token(_tokens["refresh_token"])
            _tokens["access_token"] = data["access_token"]
            _tokens["expires_at"]   = _token_expiry(data)
            if "refresh_token" in data:
                _tokens["refresh_token"] = data["refresh_token"]
        except Exception as exc:
            print(f"⚠️   Token refresh failed: {exc}", flush=True)


def _make_callback_handler(auth_done: threading.Event):
    """Return a handler class that signals auth_done on successful code exchange."""

    class _Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            parsed = urlparse(self.path)
            if parsed.path != "/callback":
                self._respond(404, "Not found"); return

            params = parse_qs(parsed.query)
            if "error" in params:
                self._respond(400, f"Spotify error: {params['error'][0]}"); return

            code = params.get("code", [None])[0]
            if not code:
                self._respond(400, "Missing code parameter"); return

            try:
                data = spotify.exchange_code(code)
                _tokens["access_token"]  = data["access_token"]
                _tokens["refresh_token"] = data.get("refresh_token", "")
                _tokens["expires_at"]    = _token_expiry(data)
                self._respond(200, "✅ Logged in! You can close this tab.")
                auth_done.set()
            except Exception as exc:
                self._respond(500, f"Token exchange failed: {exc}")

        def _respond(self, code: int, body: str):
            self.send_response(code)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(
                f"<html><body style='font-family:sans-serif;font-size:1.4rem;padding:2rem'>"
                f"{body}</body></html>".encode()
            )

        def log_message(self, *_): pass

    return _Handler


def run_sync_loop() -> None:
    print("\n🎧  Starting Spotify lyric sync…  (Ctrl+C to stop)\n", flush=True)

    current_song   : str | None     = None
    current_lyrics : list[dict]     = []
    last_lyric_time: float          = -1.0
    lyric_cache    : dict[str, str] = {}

    while True:
        try:
            _ensure_valid_token()
            playback = spotify.get_currently_playing(_tokens["access_token"])

            if not playback:
                print("\r⏸   Nothing playing…              ", end="", flush=True)
                time.sleep(2)
                continue

            song     = playback["song"]
            artist   = playback["artist"]
            progress = playback["progress_seconds"] + SYNC_OFFSET_SECONDS

            if song != current_song:
                current_song    = song
                current_lyrics  = lyrics_mod.get_lyrics(song, artist)
                lyric_cache     = {}
                last_lyric_time = -1.0
                print(f"\n{'─' * 52}")
                print(f"🎵  Song:   {song}")
                print(f"🎵  Artist: {artist}")
                print(f"{'─' * 52}\n")

            line = lyrics_mod.get_current_line(current_lyrics, progress)
            if line is None or line["time"] == last_lyric_time:
                time.sleep(POLL_INTERVAL_SECS)
                continue

            last_lyric_time  = line["time"]
            original         = line["text"]
            if original not in lyric_cache:
                lyric_cache[original] = translate(original)

            print(f"🎵  Original:  {original}")
            print(f"🎵  Converted: {lyric_cache[original]}\n", flush=True)

        except KeyboardInterrupt:
            print("\n\n👋  Stopped.")
            break
        except Exception as exc:
            print(f"\n⚠️   Error: {exc}", flush=True)
            time.sleep(2)

        time.sleep(POLL_INTERVAL_SECS)


def main() -> None:
    if not spotify.CLIENT_ID or not spotify.CLIENT_SECRET:
        print(
            "❌  Missing Spotify credentials.\n"
            "    Ensure SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET are in backend/.env"
        )
        sys.exit(1)

    auth_done = threading.Event()
    server    = HTTPServer(("localhost", CALLBACK_PORT), _make_callback_handler(auth_done))
    thread    = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    auth_url = spotify.get_auth_url()
    print(f"\n🔐  Open this URL to log in to Spotify:\n\n    {auth_url}\n")
    webbrowser.open(auth_url)

    print(f"⏳  Waiting for Spotify login (timeout: {AUTH_TIMEOUT_SECS}s)…\n")
    auth_done.wait(timeout=AUTH_TIMEOUT_SECS)

    if not auth_done.is_set():
        print("❌  Login timed out. Run again and complete the browser login.")
        sys.exit(1)

    server.shutdown()
    run_sync_loop()


if __name__ == "__main__":
    main()
