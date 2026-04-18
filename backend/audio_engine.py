"""
Whisper-based audio transcription fallback for Syntora Live Lyrics.

When lrclib has no lyrics for a song this module:
  1. Downloads the Spotify 30-second preview (preview_url from the Tracks API)
  2. Runs OpenAI Whisper (multilingual small model) for transcription +
     automatic language detection
  3. Returns time-synced segments in the same {time, text} format used by
     lyrics.py so the rest of the pipeline is unchanged

Supported Indic languages (Whisper handles natively):
  Tamil (ta), Hindi (hi), Malayalam (ml), Telugu (te)

Requirements:
  pip install openai-whisper
  brew install ffmpeg   # or apt-get install ffmpeg

The model is lazy-loaded on first use and is not required at import time —
if Whisper or ffmpeg is unavailable the engine silently degrades to
"no audio fallback available".
"""
from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Optional

import requests

logger = logging.getLogger(__name__)

# How long (seconds) the Spotify preview window is
PREVIEW_DURATION = 30.0

# Whisper model to use — "small" is the best balance of speed and accuracy
# for multilingual Indic content on CPU.
# Upgrade to "medium" for better accuracy at the cost of ~2× slower inference.
WHISPER_MODEL_SIZE = "small"

# Whisper ISO-639-1 language code → Syntora transliteration target
# Used when target="auto" to pick the right engine automatically.
LANG_TO_TARGET: dict[str, str] = {
    "ta": "tanglish",
    "hi": "hinglish",
    "ml": "manglish",
    "te": "tenglish",
}

_PREVIEW_TIMEOUT = 20  # seconds to download the 30-second preview MP3
_CACHE: dict[str, list[dict]] = {}  # keyed by preview_url; only successful results cached

# Common Whisper hallucinations on music with no audible speech or pure instrumentals
_HALLUCINATIONS: frozenset[str] = frozenset({
    "thank you", "thanks for watching", "subscribe", "please like",
    "subtitles by", "transcribed by", "caption", "amara.org",
    "[music]", "[applause]", "[silence]", "[inaudible]",
    "www.", ".com", "♪", "♫", "...", "..!",
    "you", "i", "the", "a",  # single-word hallucinations
    "hmm", "uh", "um",
})


def _is_valid_segment(text: str, no_speech_prob: float) -> bool:
    """Return False for silent segments or known Whisper hallucination strings."""
    if no_speech_prob > 0.6:
        return False
    lower = text.lower().strip()
    # Too short to be a real lyric
    if len(lower) < 4:
        return False
    # Exact match against known hallucinations
    if lower in _HALLUCINATIONS:
        return False
    # Substring match against longer hallucination strings
    return not any(h in lower for h in _HALLUCINATIONS if len(h) > 4)


class WhisperEngine:
    """
    Wraps an OpenAI Whisper model for multilingual audio transcription.

    Lazy-loads on first call to transcribe_url(); if the library is missing
    or ffmpeg is not on PATH the _ready flag stays False and callers receive
    empty lists gracefully.
    """

    def __init__(self, model_size: str = WHISPER_MODEL_SIZE):
        self.model_size = model_size
        self._model     = None
        self._ready     = False
        self._try_load()

    def _try_load(self) -> None:
        try:
            import whisper  # openai-whisper
            logger.info("Loading Whisper '%s' model for audio fallback…", self.model_size)
            self._model = whisper.load_model(self.model_size)
            self._ready = True
            logger.info("Whisper model ready (size=%s).", self.model_size)
        except ImportError:
            logger.warning(
                "openai-whisper not installed. "
                "Run: pip install openai-whisper && brew install ffmpeg"
            )
        except Exception as exc:
            logger.warning("Whisper load failed (%s). Audio fallback disabled.", exc)

    # ── Public API ────────────────────────────────────────────────────────────

    def transcribe_url(
        self,
        url: str,
        song: str = "",
        artist: str = "",
    ) -> list[dict]:
        """
        Download *url* (Spotify 30-second MP3 preview) and transcribe it.

        Returns a list of::

            {"time": float, "text": str, "language": str, "source": "whisper"}

        where ``time`` is seconds from the start of the preview (0–30).
        Returns [] on any failure so callers can test truthiness safely.
        """
        if url in _CACHE:
            return _CACHE[url]

        if not self._ready:
            _CACHE[url] = []
            return []

        try:
            # 1. Download the preview
            resp = requests.get(url, timeout=_PREVIEW_TIMEOUT)
            resp.raise_for_status()

            # 2. Write to a temp file (Whisper needs a file path, not bytes)
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as fh:
                fh.write(resp.content)
                tmp = Path(fh.name)

            # Build a helpful initial prompt from available metadata
            if song and artist:
                prompt = f"Lyrics from the song '{song}' by {artist}."
            elif song:
                prompt = f"Lyrics from the song '{song}'."
            else:
                prompt = "Song lyrics in an Indic language."

            # 3. Transcribe — language is auto-detected by Whisper.
            # condition_on_previous_text=False prevents hallucination cascades
            # where one bad prediction poisons all subsequent segments.
            # temperature=0 gives greedy (most deterministic) decoding.
            result = self._model.transcribe(
                str(tmp),
                task="transcribe",
                fp16=False,
                verbose=False,
                condition_on_previous_text=False,
                no_speech_threshold=0.5,
                logprob_threshold=None,
                compression_ratio_threshold=2.4,
                temperature=0,
                initial_prompt=prompt,
            )
            tmp.unlink(missing_ok=True)

            lang     = result.get("language", "unknown")
            segments = result.get("segments") or []

            lines: list[dict] = []
            for seg in segments:
                seg_text        = (seg.get("text") or "").strip()
                no_speech_prob  = float(seg.get("no_speech_prob", 0.0))
                if seg_text and _is_valid_segment(seg_text, no_speech_prob):
                    lines.append({
                        "time":     float(seg["start"]),
                        "text":     seg_text,
                        "language": lang,
                        "source":   "whisper",
                    })

            # Fallback: Whisper gave a flat string with no segments
            if not lines:
                full = (result.get("text") or "").strip()
                if full and _is_valid_segment(full, 0.0):
                    lines = [{
                        "time":     0.0,
                        "text":     full,
                        "language": lang,
                        "source":   "whisper",
                    }]

            logger.info(
                "Whisper: '%s' → %d valid segment(s) (of %d raw), detected_lang=%s",
                song or url[:40], len(lines), len(segments), lang,
            )

            # Only cache successful non-empty results; failures will be retried
            if lines:
                _CACHE[url] = lines
            return lines

        except Exception as exc:
            logger.error("Whisper transcription failed for '%s': %s", song, exc)
            # Do NOT cache failures — allow retry on the next poll cycle
            return []

    def detected_language(self, url: str) -> str:
        """Return the ISO language code for *url*, or 'unknown'."""
        lines = _CACHE.get(url) or []
        if lines:
            return lines[0].get("language", "unknown")
        return "unknown"

    def transcribe_bytes(self, audio_bytes: bytes, ext: str = "webm") -> dict:
        """
        Transcribe raw audio bytes (from a browser MediaRecorder upload).

        Returns::
            {"language": str, "text": str, "segments": list[dict]}

        Raises on hard failures so the caller can return HTTP 500.
        """
        if not self._ready:
            return {"language": "unknown", "text": "", "segments": [],
                    "error": "Whisper not available"}

        with tempfile.NamedTemporaryFile(suffix=f".{ext}", delete=False) as fh:
            fh.write(audio_bytes)
            tmp = Path(fh.name)

        try:
            result = self._model.transcribe(
                str(tmp),
                task="transcribe",
                fp16=False,
                verbose=False,
                condition_on_previous_text=False,
                no_speech_threshold=0.4,
                logprob_threshold=None,
                compression_ratio_threshold=2.4,
                temperature=0,
            )
            tmp.unlink(missing_ok=True)

            lang     = result.get("language", "unknown")
            segments = result.get("segments") or []

            lines: list[dict] = []
            for seg in segments:
                seg_text   = (seg.get("text") or "").strip()
                no_speech  = float(seg.get("no_speech_prob", 0.0))
                if seg_text and _is_valid_segment(seg_text, no_speech):
                    lines.append({
                        "time":     float(seg["start"]),
                        "text":     seg_text,
                        "language": lang,
                    })

            full_text = " ".join(s["text"] for s in lines).strip()
            if not full_text:
                full_text = (result.get("text") or "").strip()

            logger.info("transcribe_bytes: lang=%s  segments=%d  text=%r",
                        lang, len(lines), full_text[:60])
            return {"language": lang, "text": full_text, "segments": lines}

        except Exception as exc:
            tmp.unlink(missing_ok=True)
            logger.error("transcribe_bytes failed: %s", exc)
            raise


# ── Singleton ─────────────────────────────────────────────────────────────────

_engine: Optional[WhisperEngine] = None


def get_whisper_engine() -> WhisperEngine:
    global _engine
    if _engine is None:
        _engine = WhisperEngine()
    return _engine


# ── Helpers used by main.py ───────────────────────────────────────────────────

def progress_in_preview(song_progress_seconds: float) -> float:
    """
    Map an arbitrary song position into the 30-second Whisper preview window.

    Cycles with % so lyrics keep updating throughout the song rather than
    freezing at the last segment forever.  The mismatch is acceptable because
    Whisper only has 30 seconds of audio to work with.
    """
    return song_progress_seconds % PREVIEW_DURATION
