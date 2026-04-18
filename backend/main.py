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
from collections import defaultdict
from contextlib import asynccontextmanager
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

# ── Rate limiting (sliding window, in-memory) ─────────────────────────────────
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


# ── Single-user token store ───────────────────────────────────────────────────
_token: dict = {}          # {access_token, refresh_token, expires_at}
_token_lock: asyncio.Lock | None = None
_SAFE_REASON_RE = re.compile(r"[^a-zA-Z0-9_\-]")
_SYNC_OFFSET = 0.4


def _get_token_lock() -> asyncio.Lock:
    global _token_lock
    if _token_lock is None:
        _token_lock = asyncio.Lock()
    return _token_lock


def _token_expiry(data: dict) -> float:
    return time.time() + data.get("expires_in", 3600) - 60


async def _ensure_valid_token() -> str:
    if not _token or not _token.get("access_token"):
        raise HTTPException(401, "Not authenticated — please connect Spotify")

    if _token.get("expires_at", 0) > time.time():
        return _token["access_token"]

    async with _get_token_lock():
        if _token.get("expires_at", 0) > time.time():
            return _token["access_token"]
        try:
            data = spotify_mod.refresh_access_token(_token["refresh_token"])
            _token["access_token"] = data["access_token"]
            _token["expires_at"]   = _token_expiry(data)
            if "refresh_token" in data:
                _token["refresh_token"] = data["refresh_token"]
            logger.info("Spotify: token refreshed")
        except Exception as exc:
            _token.clear()
            raise HTTPException(401, f"Token refresh failed — please reconnect: {exc}")

    return _token["access_token"]


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
app = FastAPI(title="Syntora API", version="3.0.0", lifespan=lifespan)

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


# ── Spotify integration ───────────────────────────────────────────────────────

@app.get("/callback")
async def spotify_oauth_callback(
    code: str = None, error: str = None, state: str = None
):
    if error:
        safe_reason = _SAFE_REASON_RE.sub("", error)[:64]
        logger.warning("Spotify OAuth denied: %s", safe_reason)
        return RedirectResponse(url=f"{_FRONTEND_URL}?spotify=error&reason={safe_reason}")
    if not code:
        logger.error("Spotify OAuth callback: no code received")
        return RedirectResponse(url=f"{_FRONTEND_URL}?spotify=error&reason=no_code")

    try:
        data = spotify_mod.exchange_code(code)
        _token.clear()
        _token.update({
            "access_token":  data["access_token"],
            "refresh_token": data.get("refresh_token", ""),
            "expires_at":    _token_expiry(data),
        })
        logger.info("Spotify: token exchange OK")
    except Exception as exc:
        logger.error("Spotify code exchange failed: %s", exc)
        return RedirectResponse(url=f"{_FRONTEND_URL}?spotify=error&reason=exchange_failed")

    return RedirectResponse(url=f"{_FRONTEND_URL}?spotify=connected")


@app.get("/spotify/auth-url")
async def spotify_auth_url():
    if not spotify_mod.CLIENT_ID:
        raise HTTPException(500, "SPOTIFY_CLIENT_ID env var not set on the server")
    url = spotify_mod.get_auth_url(state="")
    return {"url": url, "redirect_uri": spotify_mod.REDIRECT_URI}


@app.post("/spotify/logout")
async def spotify_logout():
    _token.clear()
    logger.info("Spotify: logged out")
    return {"status": "ok"}


async def _get_spotify_state(access_token: str) -> dict:
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
    ip = request.client.host if request.client else "0.0.0.0"
    if not _allow_request(f"spotify:{ip}", max_req=120, window=60.0):
        raise HTTPException(429, "Rate limit exceeded — slow down the polling interval")
    access_token = await _ensure_valid_token()
    state = await _get_spotify_state(access_token)
    return JSONResponse(content=state)


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
