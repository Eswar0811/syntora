from __future__ import annotations

from pathlib import Path
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / ".env")
except ImportError:
    pass

import asyncio
import bisect
import json
import logging
import os
import re
import secrets
import time
from collections import defaultdict
from contextlib import asynccontextmanager
from dataclasses import dataclass, field as _dc_field
from threading import Lock

from fastapi import FastAPI, HTTPException, Request, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from pydantic import BaseModel, Field, field_validator

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

# ── Config ────────────────────────────────────────────────────────────────────
_FRONTEND_URL    = os.environ.get("FRONTEND_URL", "http://localhost:3000")
_DEFAULT_ORIGINS = f"{_FRONTEND_URL},http://localhost:3000,http://localhost:5175"
_ALLOWED_ORIGINS = [
    o.strip()
    for o in os.environ.get("ALLOWED_ORIGINS", _DEFAULT_ORIGINS).split(",")
    if o.strip()
]
# Matches production + any Vercel preview URL (syntora-app-*.vercel.app)
# alongside the explicit list above; localhost covered for local dev
_ORIGIN_REGEX = (
    r"https://syntora[a-zA-Z0-9\-]*\.vercel\.app"
    r"|https?://(localhost|127\.0\.0\.1)(:\d+)?"
)

# ── Rate limiting ─────────────────────────────────────────────────────────────
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


# ── Per-user session store ────────────────────────────────────────────────────
_SAFE_REASON_RE = re.compile(r"[^a-zA-Z0-9_\-]")
_SAFE_SID_RE    = re.compile(r"[^A-Za-z0-9_\-]")
SESSION_HEADER  = "x-session-id"
_SESSION_TTL    = 7200   # seconds of inactivity before a session expires
_TOKEN_FILE     = Path("/tmp/syntora_sess.json")

# How far ahead of raw Spotify position to select lyrics (compensates for
# total round-trip latency: Spotify API + Render→Vercel + browser render)
_SYNC_OFFSET = 0.9


@dataclass
class _Session:
    access_token:  str  = ""
    refresh_token: str  = ""
    expires_at:    float = 0.0
    last_active:   float = _dc_field(default_factory=time.time)
    last_good:     dict  = _dc_field(default_factory=dict)


_sessions:      dict[str, _Session] = {}
_sessions_lock: Lock                 = Lock()


def _token_expiry(data: dict) -> float:
    return time.time() + data.get("expires_in", 3600) - 60


def _new_session() -> tuple[str, _Session]:
    sid = secrets.token_urlsafe(32)
    s   = _Session()
    with _sessions_lock:
        _sessions[sid] = s
    _persist_sessions()
    return sid, s


def _get_session(request: Request) -> _Session | None:
    # Accept session ID from header (dev) or query param (production proxy-safe)
    raw = request.headers.get(SESSION_HEADER, "").strip() or \
          request.query_params.get("sid", "").strip()
    sid = _SAFE_SID_RE.sub("", raw)[:64]
    if not sid:
        return None
    with _sessions_lock:
        s = _sessions.get(sid)
        if s:
            s.last_active = time.time()
    return s


def _remove_session(request: Request) -> None:
    raw = request.headers.get(SESSION_HEADER, "").strip() or \
          request.query_params.get("sid", "").strip()
    sid = _SAFE_SID_RE.sub("", raw)[:64]
    if sid:
        with _sessions_lock:
            _sessions.pop(sid, None)
        _persist_sessions()


def _prune_sessions() -> None:
    cutoff = time.time() - _SESSION_TTL
    with _sessions_lock:
        dead = [k for k, v in _sessions.items() if v.last_active < cutoff]
        for k in dead:
            del _sessions[k]


def _persist_sessions() -> None:
    try:
        with _sessions_lock:
            data = {
                sid: {
                    "access_token":  s.access_token,
                    "refresh_token": s.refresh_token,
                    "expires_at":    s.expires_at,
                    "last_active":   s.last_active,
                }
                for sid, s in _sessions.items()
            }
        _TOKEN_FILE.write_text(json.dumps(data))
    except Exception:
        pass


def _load_sessions() -> None:
    try:
        if not _TOKEN_FILE.exists():
            return
        raw   = json.loads(_TOKEN_FILE.read_text())
        cutoff = time.time() - _SESSION_TTL
        loaded = 0
        for sid, vals in raw.items():
            if vals.get("last_active", 0) > cutoff:
                _sessions[sid] = _Session(
                    access_token  = vals.get("access_token", ""),
                    refresh_token = vals.get("refresh_token", ""),
                    expires_at    = vals.get("expires_at", 0.0),
                    last_active   = vals.get("last_active", time.time()),
                )
                loaded += 1
        if loaded:
            logger.info("Restored %d Spotify session(s) from disk", loaded)
    except Exception as exc:
        logger.warning("Could not load sessions from disk: %s", exc)


async def _require_token(request: Request) -> str:
    s = _get_session(request)
    if not s or not s.access_token:
        raise HTTPException(401, "Not authenticated — please connect Spotify")
    if s.expires_at > time.time():
        return s.access_token
    try:
        data = spotify_mod.refresh_access_token(s.refresh_token)
        s.access_token = data["access_token"]
        s.expires_at   = _token_expiry(data)
        if "refresh_token" in data:
            s.refresh_token = data["refresh_token"]
        logger.info("Spotify token refreshed for session")
        _persist_sessions()
    except Exception as exc:
        _remove_session(request)
        raise HTTPException(401, f"Token refresh failed — please reconnect: {exc}")
    return s.access_token


# ── Self-ping keep-alive ──────────────────────────────────────────────────────
async def _keep_alive():
    import httpx
    host = os.environ.get("RENDER_EXTERNAL_URL", "")
    if not host:
        return
    url = f"{host}/health"
    await asyncio.sleep(60)
    while True:
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                await client.get(url)
        except Exception:
            pass
        await asyncio.sleep(240)


# ── Lifespan ──────────────────────────────────────────────────────────────────
async def _session_janitor():
    while True:
        await asyncio.sleep(600)
        _prune_sessions()
        _persist_sessions()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    logger.info("Pre-loading engines…")
    get_engine()
    get_hindi_engine()
    get_malayalam_engine()
    get_telugu_engine()
    logger.info("All 4 engines ready.")
    _load_sessions()
    task1 = asyncio.create_task(_keep_alive())
    task2 = asyncio.create_task(_session_janitor())
    yield
    task1.cancel()
    task2.cancel()
    logger.info("Shutting down Syntora API.")


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="Syntora API", version="3.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_ALLOWED_ORIGINS,
    allow_origin_regex=_ORIGIN_REGEX,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", SESSION_HEADER],
    allow_credentials=False,
)


@app.exception_handler(Exception)
async def _global_exc(request: Request, exc: Exception):
    logger.error("Unhandled error on %s %s: %s", request.method, request.url.path, exc)
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
    engine = get_hindi_engine()
    return JSONResponse(content=engine.convert(req.text))


@app.post("/malayalam/convert")
async def malayalam_convert(req: MalayalamRequest):
    engine = get_malayalam_engine()
    return JSONResponse(content=engine.convert(req.text))


@app.post("/telugu/convert")
async def telugu_convert(req: TeluguRequest):
    engine = get_telugu_engine()
    return JSONResponse(content=engine.convert(req.text))


@app.post("/song/convert")
async def song_convert(req: SongRequest):
    engine = get_song_engine()
    return JSONResponse(content=engine.convert(req.text))


# ── Audio transcription ───────────────────────────────────────────────────────

_ALLOWED_AUDIO_EXTS = frozenset({"webm", "mp3", "wav", "ogg", "m4a", "mp4", "flac"})
_MAX_AUDIO_BYTES    = 10 * 1024 * 1024


@app.post("/audio/transcribe")
async def audio_transcribe(request: Request, file: UploadFile = File(...)):
    ip = request.client.host if request.client else "0.0.0.0"
    if not _allow_request(f"audio:{ip}", max_req=10, window=60.0):
        raise HTTPException(429, "Rate limit: max 10 audio requests per minute")
    whisper = get_whisper_engine()
    if not whisper._ready:
        raise HTTPException(503, "Whisper not available — install openai-whisper and ffmpeg")
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
    for line in (lyrics_list or [])[:5]:
        for ch in line.get("text", ""):
            cp = ord(ch)
            if 0x0B80 <= cp <= 0x0BFF: return "ta"
            if 0x0900 <= cp <= 0x097F: return "hi"
            if 0x0D00 <= cp <= 0x0D7F: return "ml"
            if 0x0C00 <= cp <= 0x0C7F: return "te"
    return None


def _line_index(lyrics_list: list[dict], position: float) -> int:
    """Return index of the latest line whose time <= position (-1 if before first)."""
    times = [l["time"] for l in lyrics_list]
    return bisect.bisect_right(times, position) - 1


# _last_good is now per-session (stored on _Session.last_good)


# ── Spotify ───────────────────────────────────────────────────────────────────

@app.get("/spotify/auth-url")
async def spotify_auth_url():
    if not spotify_mod.CLIENT_ID:
        raise HTTPException(500, "SPOTIFY_CLIENT_ID env var not set on the server")
    url = spotify_mod.get_auth_url(state="")
    return {"url": url, "redirect_uri": spotify_mod.REDIRECT_URI}


@app.get("/callback")
async def spotify_oauth_callback(code: str = None, error: str = None, state: str = None):
    if error:
        safe_reason = _SAFE_REASON_RE.sub("", error)[:64]
        return RedirectResponse(url=f"{_FRONTEND_URL}?spotify=error&reason={safe_reason}")
    if not code:
        return RedirectResponse(url=f"{_FRONTEND_URL}?spotify=error&reason=no_code")
    try:
        data = spotify_mod.exchange_code(code)
        sid, s = _new_session()
        s.access_token  = data["access_token"]
        s.refresh_token = data.get("refresh_token", "")
        s.expires_at    = _token_expiry(data)
        logger.info("Spotify token exchange OK, session=%s…", sid[:8])
    except Exception as exc:
        logger.error("Token exchange failed: %s", exc)
        return RedirectResponse(url=f"{_FRONTEND_URL}?spotify=error&reason=exchange_failed")
    return RedirectResponse(url=f"{_FRONTEND_URL}?spotify=connected&sid={sid}")


@app.get("/spotify/status")
async def spotify_status(request: Request):
    s = _get_session(request)
    return {"authed": bool(s and s.access_token)}


@app.post("/spotify/logout")
async def spotify_logout(request: Request):
    _remove_session(request)
    return {"status": "ok"}


async def _get_spotify_state(access_token: str, session: _Session) -> dict:
    # ── Fetch playback from Spotify ───────────────────────────────────────────
    try:
        playback = spotify_mod.get_currently_playing(access_token)
    except Exception as exc:
        if session.last_good:
            logger.warning("Spotify error (using cached state): %s", exc)
            return {**session.last_good, "stale": True}
        raise HTTPException(502, f"Spotify API error: {exc}")

    if not playback:
        session.last_good.clear()
        return {"is_playing": False}

    # ── Lyrics ────────────────────────────────────────────────────────────────
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
                w_lines = whisper.transcribe_url(preview_url, playback["song"], playback["artist"])
                if w_lines:
                    lyrics_list   = w_lines
                    detected_lang = w_lines[0].get("language") or detected_lang
                    using_whisper = True
                    source        = "whisper"
        else:
            no_preview = True

    raw_progress = playback["progress_seconds"]
    adjusted     = progress_in_preview(raw_progress) if using_whisper else raw_progress + _SYNC_OFFSET

    # ── Current + next line selection and pre-translation ─────────────────────
    idx = _line_index(lyrics_list, adjusted)

    result: dict = {
        "is_playing":       True,
        "song":             playback["song"],
        "artist":           playback["artist"],
        "progress_seconds": raw_progress,   # raw — frontend interpolates on top
        "track_id":         playback.get("track_id", ""),
        "source":           source,
        "sync_offset":      _SYNC_OFFSET,   # tell frontend what offset we used
    }

    if detected_lang:
        result["detected_language"] = detected_lang
        result["detected_label"]    = _LANG_LABEL.get(detected_lang, detected_lang.upper())
    if using_whisper:
        result["whisper_transcribed"] = True
    if no_preview:
        result["no_preview"] = True

    # Current line
    if idx >= 0 and lyrics_list:
        current = lyrics_list[idx]
        result["original"]   = current["text"]
        result["line_time"]  = current["time"]
        result.update(_get_translations(current["text"]))

        # Next line — pre-translated so the frontend can switch instantly
        if idx + 1 < len(lyrics_list):
            nxt = lyrics_list[idx + 1]
            result["next_original"]  = nxt["text"]
            result["next_line_time"] = nxt["time"]
            next_tr = _get_translations(nxt["text"])
            for k, v in next_tr.items():
                result[f"next_{k}"] = v

    session.last_good.update(result)
    return result


@app.get("/spotify/current")
async def spotify_current(request: Request):
    # Rate-limit per session (not IP) so multiple users behind the same NAT/proxy
    # each get their own bucket of 120 req/min.
    raw_sid = (request.headers.get(SESSION_HEADER, "") or
               request.query_params.get("sid", "")).strip()
    sid_clean = _SAFE_SID_RE.sub("", raw_sid)[:64]
    rate_key  = f"spotify:sid:{sid_clean}" if sid_clean else \
                f"spotify:ip:{request.client.host if request.client else '0.0.0.0'}"
    if not _allow_request(rate_key, max_req=120, window=60.0):
        raise HTTPException(429, "Rate limit exceeded")
    session      = _get_session(request)
    access_token = await _require_token(request)
    if session is None:
        raise HTTPException(401, "Session not found")
    state = await _get_spotify_state(access_token, session)
    return JSONResponse(content=state)


@app.get("/health")
async def health():
    return {
        "status":             "ok",
        "tamil_neural_ready": get_engine()._ready,
        "telugu_neural_ready": get_telugu_engine()._ready,
    }
