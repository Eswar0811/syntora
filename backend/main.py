from __future__ import annotations

from pathlib import Path
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / ".env")
except ImportError:
    pass

import asyncio
import json
import logging
import os
import re
import time
import uuid
from collections import defaultdict
from contextlib import asynccontextmanager
from threading import Lock

from fastapi import FastAPI, HTTPException, Request, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from pydantic import BaseModel, Field, field_validator

try:
    from sse_starlette.sse import EventSourceResponse
    _SSE_AVAILABLE = True
except ImportError:
    _SSE_AVAILABLE = False

from byt5_engine import get_engine, normalize_tamil
from hindi_engine import get_hindi_engine
from malayalam_engine import get_malayalam_engine
from telugu_engine import get_telugu_engine
from song_engine import get_song_engine
import spotify as spotify_mod
import lyrics as lyrics_mod
from audio_engine import get_whisper_engine, progress_in_preview
from translator import translate_pair

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Config ──────────────────────────────────────────────────────────────────
_FRONTEND_URL   = os.environ.get("FRONTEND_URL", "http://localhost:3000")
_DEFAULT_ORIGINS = f"{_FRONTEND_URL},http://localhost:3000,http://localhost:5175"
_ALLOWED_ORIGINS = [
    o.strip()
    for o in os.environ.get("ALLOWED_ORIGINS", _DEFAULT_ORIGINS).split(",")
    if o.strip()
]

# ── Rate limiting (sliding window, in-memory) ────────────────────────────────
_rate_store: dict[str, list[float]] = defaultdict(list)
_rate_lock = Lock()


def _allow_request(key: str, max_req: int, window: float = 60.0) -> bool:
    now = time.monotonic()
    cutoff = now - window
    with _rate_lock:
        bucket = _rate_store[key]
        while bucket and bucket[0] < cutoff:
            bucket.pop(0)
        if len(bucket) >= max_req:
            return False
        bucket.append(now)
        return True


# ── Input sanitization ────────────────────────────────────────────────────────
_CTRL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def _sanitize(text: str) -> str:
    """Remove ASCII control characters; leave all Unicode (Indic scripts) intact."""
    return _CTRL_RE.sub("", text).strip()


# ── Request models ────────────────────────────────────────────────────────────

class ConvertRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=2000)
    mode: str = Field("formal")

    @field_validator("text")
    @classmethod
    def clean(cls, v: str) -> str:
        return _sanitize(v)


class HindiRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=2000)

    @field_validator("text")
    @classmethod
    def clean(cls, v: str) -> str:
        return _sanitize(v)


class MalayalamRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=2000)

    @field_validator("text")
    @classmethod
    def clean(cls, v: str) -> str:
        return _sanitize(v)


class TeluguRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=2000)

    @field_validator("text")
    @classmethod
    def clean(cls, v: str) -> str:
        return _sanitize(v)


class SongRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000)

    @field_validator("text")
    @classmethod
    def clean(cls, v: str) -> str:
        return _sanitize(v)


# ── Lifespan ──────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(_app: FastAPI):
    logger.info("Pre-loading engines…")
    get_engine()
    get_telugu_engine()
    logger.info("Engines ready.")
    yield
    logger.info("Shutting down Syntora API.")


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="Syntora API", version="2.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_ALLOWED_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
    allow_credentials=False,
)


@app.exception_handler(Exception)
async def _global_exc(request: Request, exc: Exception):
    logger.error("Unhandled error on %s %s: %s", request.method, request.url.path, exc)
    # Never expose internal details in the response body
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


# ── Transliteration endpoints ─────────────────────────────────────────────────

@app.post("/convert")
async def convert(req: ConvertRequest):
    if req.mode not in ("formal", "spoken", "slang"):
        raise HTTPException(400, "mode must be formal|spoken|slang")
    engine = get_engine()
    result = engine.convert(req.text, mode=req.mode)
    result["input_cleaned"] = normalize_tamil(req.text)
    return JSONResponse(content=result)


@app.post("/hindi/convert")
async def hindi_convert(req: HindiRequest):
    """Bidirectional Hindi ↔ Hinglish — offline rule-based engine."""
    engine = get_hindi_engine()
    return JSONResponse(content=engine.convert(req.text))


@app.post("/malayalam/convert")
async def malayalam_convert(req: MalayalamRequest):
    """Bidirectional Malayalam ↔ Manglish — offline rule-based engine."""
    engine = get_malayalam_engine()
    return JSONResponse(content=engine.convert(req.text))


@app.post("/telugu/convert")
async def telugu_convert(req: TeluguRequest):
    """Bidirectional Telugu ↔ Tenglish — rule-based + ByT5 neural correction."""
    engine = get_telugu_engine()
    return JSONResponse(content=engine.convert(req.text))


@app.post("/song/convert")
async def song_convert(req: SongRequest):
    engine = get_song_engine()
    return JSONResponse(content=engine.convert(req.text))


# ── Audio transcription ───────────────────────────────────────────────────────

_ALLOWED_AUDIO_EXTS = frozenset({"webm", "mp3", "wav", "ogg", "m4a", "mp4", "flac"})
_MAX_AUDIO_BYTES    = 10 * 1024 * 1024  # 10 MB


@app.post("/audio/transcribe")
async def audio_transcribe(request: Request, file: UploadFile = File(...)):
    ip = request.client.host if request.client else "0.0.0.0"
    if not _allow_request(f"audio:{ip}", max_req=10, window=60.0):
        raise HTTPException(429, "Rate limit: max 10 audio requests per minute")

    whisper = get_whisper_engine()
    if not whisper._ready:
        raise HTTPException(
            503,
            "Whisper not available. Install openai-whisper and ffmpeg to enable audio detection.",
        )

    audio_bytes = await file.read()
    if not audio_bytes:
        raise HTTPException(400, "Empty audio upload")
    if len(audio_bytes) > _MAX_AUDIO_BYTES:
        raise HTTPException(413, "Audio file too large (max 10 MB)")

    filename = file.filename or "audio.webm"
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else "webm"
    if ext not in _ALLOWED_AUDIO_EXTS:
        ext = "webm"

    try:
        result = whisper.transcribe_bytes(audio_bytes, ext)
    except Exception as exc:
        raise HTTPException(500, f"Transcription failed: {exc}")

    lang = result.get("language", "unknown")
    text = result.get("text", "").strip()

    base: dict = {
        "language":       lang,
        "detected_label": _LANG_LABEL.get(lang, lang.upper()),
        "text":           text,
    }
    if not text:
        base["error"] = "No speech detected in the audio"
        return JSONResponse(content=base)

    base["original"] = text
    base.update(_get_translations(text))
    return JSONResponse(content=base)


# ── Helpers ───────────────────────────────────────────────────────────────────

_LANG_LABEL = {
    "ta": "Tamil", "hi": "Hindi", "ml": "Malayalam",
    "te": "Telugu", "en": "English",
}

_ALL_LANG_PAIRS = [
    ("tamil",     "tanglish",  "tanglish"),
    ("hindi",     "hinglish",  "hinglish"),
    ("malayalam", "manglish",  "manglish"),
    ("telugu",    "tenglish",  "tenglish"),
]

# Translation cache: lyric line text → {tamil, tanglish, hindi, hinglish, …}
# Manual LRU: evict oldest entry when full.
_translation_cache: dict[str, dict] = {}
_MAX_TRANSLATION_CACHE = 400


def _get_translations(text: str) -> dict:
    if text in _translation_cache:
        return _translation_cache[text]

    out: dict = {}
    for script_key, roman_key, engine_target in _ALL_LANG_PAIRS:
        try:
            script_form, roman_form = translate_pair(text, engine_target)
            out[script_key] = script_form
            out[roman_key]  = roman_form
        except Exception as exc:
            logger.warning("translate_pair(%s) failed: %s", engine_target, exc)
            out[script_key] = text
            out[roman_key]  = text

    if len(_translation_cache) >= _MAX_TRANSLATION_CACHE:
        _translation_cache.pop(next(iter(_translation_cache)))
    _translation_cache[text] = out
    return out


def _detect_script_language(lyrics_list: list[dict]) -> str | None:
    """Detect ISO 639-1 language from Unicode script of the first few lyric lines."""
    for line in (lyrics_list or [])[:5]:
        for ch in line.get("text", ""):
            cp = ord(ch)
            if 0x0B80 <= cp <= 0x0BFF: return "ta"   # Tamil
            if 0x0900 <= cp <= 0x097F: return "hi"   # Hindi / Devanagari
            if 0x0D00 <= cp <= 0x0D7F: return "ml"   # Malayalam
            if 0x0C00 <= cp <= 0x0C7F: return "te"   # Telugu
    return None


# ── Spotify integration ───────────────────────────────────────────────────────

# ── Multi-user session store ──────────────────────────────────────────────────
# Each user gets a UUID session_id generated at /spotify/auth-url.
# It travels through the OAuth state param, gets stored in the browser's
# localStorage, and is sent back on every request as X-Session-ID header.

_sessions: dict[str, dict] = {}          # session_id → token data
_session_locks: dict[str, asyncio.Lock] = {}  # session_id → refresh lock
_SESSION_TTL = 24 * 3600                  # auto-expire sessions after 24 h

_SYNC_OFFSET = 0.4
_SAFE_REASON_RE = re.compile(r"[^a-zA-Z0-9_\-]")


def _token_expiry(data: dict) -> float:
    return time.time() + data.get("expires_in", 3600) - 60


def _get_session_id(request: Request) -> str:
    sid = request.headers.get("X-Session-ID", "").strip()
    if not sid:
        raise HTTPException(401, "No session — please connect Spotify first")
    return sid


def _purge_expired_sessions() -> None:
    now = time.time()
    expired = [sid for sid, s in _sessions.items()
               if s.get("expires_at", 0) + _SESSION_TTL < now]
    for sid in expired:
        _sessions.pop(sid, None)
        _session_locks.pop(sid, None)


async def _ensure_valid_token(session_id: str) -> str:
    """Return a valid access token for this session, refreshing if expired."""
    session = _sessions.get(session_id)
    if not session or not session.get("access_token"):
        raise HTTPException(401, "Not authenticated — please reconnect Spotify")

    if session.get("expires_at", 0) > time.time():
        return session["access_token"]

    lock = _session_locks.setdefault(session_id, asyncio.Lock())
    async with lock:
        session = _sessions.get(session_id, {})
        if session.get("expires_at", 0) > time.time():
            return session["access_token"]
        try:
            data = spotify_mod.refresh_access_token(session["refresh_token"])
            session["access_token"] = data["access_token"]
            session["expires_at"]   = _token_expiry(data)
            if "refresh_token" in data:
                session["refresh_token"] = data["refresh_token"]
            logger.info("Spotify: token refreshed for session %s", session_id[:8])
        except Exception as exc:
            _sessions.pop(session_id, None)
            raise HTTPException(401, f"Token refresh failed — please reconnect: {exc}")

    return _sessions[session_id]["access_token"]


@app.get("/callback")
async def spotify_oauth_callback(
    code: str = None, error: str = None, state: str = None
):
    """Spotify redirects here with ?code=…&state=session_id after user grants access."""
    if error:
        safe_reason = _SAFE_REASON_RE.sub("", error)[:64]
        logger.warning("Spotify OAuth denied: %s", safe_reason)
        return RedirectResponse(url=f"{_FRONTEND_URL}?spotify=error&reason={safe_reason}")
    if not code:
        logger.error("Spotify OAuth callback: no code received")
        return RedirectResponse(url=f"{_FRONTEND_URL}?spotify=error&reason=no_code")

    session_id = state if state and len(state) == 36 else str(uuid.uuid4())

    try:
        data = spotify_mod.exchange_code(code)
        _sessions[session_id] = {
            "access_token":  data["access_token"],
            "refresh_token": data.get("refresh_token", ""),
            "expires_at":    _token_expiry(data),
        }
        _purge_expired_sessions()
        logger.info("Spotify: token exchange OK for session %s", session_id[:8])
    except Exception as exc:
        logger.error("Spotify code exchange failed: %s", exc)
        return RedirectResponse(url=f"{_FRONTEND_URL}?spotify=error&reason=exchange_failed")

    return RedirectResponse(url=f"{_FRONTEND_URL}?spotify=connected&sid={session_id}")


@app.get("/spotify/auth-url")
async def spotify_auth_url():
    if not spotify_mod.CLIENT_ID:
        raise HTTPException(500, "SPOTIFY_CLIENT_ID env var not set on the server")
    session_id = str(uuid.uuid4())
    url = spotify_mod.get_auth_url(state=session_id)
    logger.info("Spotify auth URL issued for session %s", session_id[:8])
    return {"url": url, "redirect_uri": spotify_mod.REDIRECT_URI, "session_id": session_id}


@app.post("/spotify/logout")
async def spotify_logout(request: Request):
    """Clear this user's Spotify session only — other users are unaffected."""
    try:
        sid = _get_session_id(request)
        _sessions.pop(sid, None)
        _session_locks.pop(sid, None)
        logger.info("Spotify: session %s logged out", sid[:8])
    except HTTPException:
        pass
    return {"status": "ok"}


async def _get_spotify_state(access_token: str) -> dict:
    """
    Core logic shared by /spotify/current (polling) and /spotify/stream (SSE).
    Returns the full state dict ready to JSON-encode.
    Raises HTTPException(502) on Spotify API failure.
    """
    try:
        playback = spotify_mod.get_currently_playing(access_token)
    except Exception as exc:
        raise HTTPException(502, f"Spotify API error: {exc}")

    if not playback:
        return {"is_playing": False}

    lyrics_list   = lyrics_mod.get_lyrics(playback["song"], playback["artist"])
    source        = "lrclib"
    using_whisper = False
    no_preview    = False
    detected_lang: str | None = _detect_script_language(lyrics_list)

    if not lyrics_list:
        preview_url = playback.get("preview_url")
        if preview_url:
            whisper = get_whisper_engine()
            if whisper._ready:
                w_lines = whisper.transcribe_url(
                    preview_url, playback["song"], playback["artist"]
                )
                if w_lines:
                    lyrics_list   = w_lines
                    detected_lang = w_lines[0].get("language") or detected_lang
                    using_whisper = True
                    source        = "whisper"
                    logger.info(
                        "Whisper: '%s' lang=%s %d segment(s)",
                        playback["song"], detected_lang, len(w_lines),
                    )
            else:
                logger.warning("Whisper not ready — install openai-whisper + ffmpeg")
        else:
            no_preview = True

    raw_progress = playback["progress_seconds"] + _SYNC_OFFSET
    adjusted     = progress_in_preview(raw_progress) if using_whisper else raw_progress
    line         = lyrics_mod.get_current_line(lyrics_list, adjusted)

    result: dict = {
        "is_playing":       True,
        "song":             playback["song"],
        "artist":           playback["artist"],
        "progress_seconds": playback["progress_seconds"],
        "track_id":         playback.get("track_id", ""),
        "source":           source,
    }
    if detected_lang:
        result["detected_language"] = detected_lang
        result["detected_label"]    = _LANG_LABEL.get(detected_lang, detected_lang.upper())
    if using_whisper:
        result["whisper_transcribed"] = True
    if no_preview:
        result["no_preview"] = True
    if line:
        result["original"] = line["text"]
        result.update(_get_translations(line["text"]))

    return result


@app.get("/spotify/current")
async def spotify_current(request: Request):
    """Polling endpoint. Rate-limited to 120 req/min per IP."""
    ip = request.client.host if request.client else "0.0.0.0"
    if not _allow_request(f"spotify:{ip}", max_req=120, window=60.0):
        raise HTTPException(429, "Rate limit exceeded — slow down the polling interval")
    sid = _get_session_id(request)
    access_token = await _ensure_valid_token(sid)
    state = await _get_spotify_state(access_token)
    return JSONResponse(content=state)


@app.get("/spotify/stream")
async def spotify_stream(request: Request, sid: str = None):
    """
    Server-Sent Events endpoint. Pushes a JSON event whenever the playing
    track or active lyric line changes. Sends a heartbeat ping every cycle
    to keep the connection alive and detect disconnects early.
    """
    if not _SSE_AVAILABLE:
        raise HTTPException(
            501,
            "SSE not available — install sse-starlette: pip install sse-starlette",
        )

    ip = request.client.host if request.client else "0.0.0.0"
    logger.info("SSE: client connected (%s)", ip)

    # Accept session_id from query param (EventSource) or header (fetch)
    sid = sid or request.headers.get("X-Session-ID", "").strip()
    if not sid:
        raise HTTPException(401, "No session — please connect Spotify first")

    async def event_generator():
        last_key: tuple | None = None   # (track_id, lyric_line) — change detection
        consecutive_errors = 0

        while True:
            if await request.is_disconnected():
                logger.info("SSE: client disconnected (%s)", ip)
                return

            # ── Token ─────────────────────────────────────────────────────
            try:
                access_token = await _ensure_valid_token(sid)
            except HTTPException as exc:
                if exc.status_code == 401:
                    yield {
                        "event": "auth_error",
                        "data": json.dumps({"code": 401, "message": "Not authenticated"}),
                    }
                    return
                await asyncio.sleep(3)
                continue

            # ── State ─────────────────────────────────────────────────────
            try:
                state = await _get_spotify_state(access_token)
                consecutive_errors = 0
            except HTTPException as exc:
                consecutive_errors += 1
                logger.warning("SSE error #%d: %s", consecutive_errors, exc.detail)
                backoff = min(2 ** consecutive_errors, 30)
                await asyncio.sleep(backoff)
                continue

            # ── Emit only on change, heartbeat otherwise ───────────────────
            change_key = (
                (state.get("track_id"), state.get("original"))
                if state.get("is_playing")
                else ("idle", None)
            )
            if change_key != last_key:
                last_key = change_key
                yield {"event": "update", "data": json.dumps(state)}
            else:
                yield {"event": "ping", "data": ""}

            await asyncio.sleep(1.0 if state.get("is_playing") else 3.0)

    return EventSourceResponse(
        event_generator(),
        headers={
            "Cache-Control":    "no-cache, no-transform",
            "X-Accel-Buffering": "no",   # disable Nginx/proxy buffering
        },
    )


@app.get("/health")
async def health():
    tamil_engine   = get_engine()
    hindi_engine   = get_hindi_engine()
    mal_engine     = get_malayalam_engine()
    tel_engine     = get_telugu_engine()
    song_engine    = get_song_engine()
    whisper_engine = get_whisper_engine()
    return {
        "status":                 "ok",
        "tamil_neural_ready":     tamil_engine._ready,
        "hindi_neural_ready":     hindi_engine._ready,
        "malayalam_neural_ready": mal_engine._ready,
        "telugu_neural_ready":    tel_engine._ready,
        "song_neural_ready":      song_engine._ready,
        "whisper_ready":          whisper_engine._ready,
        "tamil_model":     tamil_engine.MODEL_ID,
        "hindi_model":     hindi_engine.MODEL_ID  if hindi_engine._ready  else "rule-based",
        "malayalam_model": mal_engine.MODEL_ID    if mal_engine._ready    else "rule-based",
        "telugu_model":    tel_engine.MODEL_ID    if tel_engine._ready    else "rule-based",
        "song_model":      song_engine.MODEL_ID   if song_engine._ready   else "rule-based",
        "whisper_model":   f"whisper-{whisper_engine.model_size}" if whisper_engine._ready else "unavailable",
    }
