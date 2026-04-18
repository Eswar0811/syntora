"""
Tamil Song → Hindi / Malayalam Cross-Script Transliteration
============================================================
Converts Tamil Unicode text to phonetically equivalent
Devanagari (Hindi script) and Malayalam Unicode.

Architecture:
  Layer 1 — Unicode normalizer
  Layer 2 — Character-level phoneme mapping (Tamil → Devanagari / Malayalam)
  Layer 3 — ByT5 neural correction (shared model, single load for both targets)
  Layer 4 — Post-processor

Both target scripts are generated from one engine instance sharing ByT5 weights.
"""

import re
import unicodedata
import time
import logging
from functools import lru_cache
from typing import Optional

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# LAYER 1 — Normalizer
# ─────────────────────────────────────────────

def _normalize(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\u200c", "").replace("\u200d", "")
    text = re.sub(r"[ \t]+", " ", text).strip()
    return text


# ─────────────────────────────────────────────
# LAYER 2 — Character-level phoneme mappings
# ─────────────────────────────────────────────

# Tamil → Devanagari (phonetic cross-script transliteration)
TAMIL_TO_DEVA: dict[str, str] = {
    # Independent vowels
    "அ": "अ",  "ஆ": "आ", "இ": "इ",  "ஈ": "ई",
    "உ": "उ",  "ஊ": "ऊ", "எ": "ए",  "ஏ": "ए",
    "ஐ": "ऐ", "ஒ": "ओ", "ஓ": "ओ", "ஔ": "औ",
    # Vowel markers → Devanagari matras
    "\u0BBE": "\u093E",  # ா → ा
    "\u0BBF": "\u093F",  # ி → ि
    "\u0BC0": "\u0940",  # ீ → ी
    "\u0BC1": "\u0941",  # ு → ु
    "\u0BC2": "\u0942",  # ூ → ू
    "\u0BC6": "\u0947",  # ெ → े
    "\u0BC7": "\u0947",  # ே → े
    "\u0BC8": "\u0948",  # ை → ै
    "\u0BCA": "\u094B",  # ொ → ो
    "\u0BCB": "\u094B",  # ோ → ो
    "\u0BCC": "\u094C",  # ௌ → ौ
    "\u0BCD": "\u094D",  # ் → ् (pulli → halant)
    # Consonants
    "க": "क", "ங": "ङ", "ச": "च", "ஞ": "ञ",
    "ட": "ट", "ண": "ण", "த": "त", "ந": "न",
    "ப": "प", "ம": "म", "ய": "य", "ர": "र",
    "ல": "ल", "வ": "व", "ழ": "ळ", "ள": "ळ",
    "ற": "ऱ", "ன": "न",
    # Grantha
    "ஜ": "ज", "ஷ": "ष", "ஸ": "स", "ஹ": "ह",
    # Tamil numerals
    "௦": "०", "௧": "१", "௨": "२", "௩": "३",
    "௪": "४", "௫": "५", "௬": "६", "௭": "७",
    "௮": "८", "௯": "९",
}

# Tamil → Malayalam (closely related Dravidian scripts — near 1:1 phoneme match)
TAMIL_TO_MALAYALAM: dict[str, str] = {
    # Independent vowels
    "அ": "അ",  "ஆ": "ആ", "இ": "ഇ",  "ஈ": "ഈ",
    "உ": "ഉ",  "ஊ": "ഊ", "எ": "എ",  "ஏ": "ഏ",
    "ஐ": "ഐ", "ஒ": "ഒ", "ஓ": "ഓ", "ஔ": "ഔ",
    # Vowel markers → Malayalam matras
    "\u0BBE": "\u0D3E",  # ா → ാ
    "\u0BBF": "\u0D3F",  # ி → ി
    "\u0BC0": "\u0D40",  # ீ → ീ
    "\u0BC1": "\u0D41",  # ு → ു
    "\u0BC2": "\u0D42",  # ூ → ൂ
    "\u0BC6": "\u0D46",  # ெ → െ
    "\u0BC7": "\u0D47",  # ே → േ
    "\u0BC8": "\u0D48",  # ை → ൈ
    "\u0BCA": "\u0D4A",  # ொ → ൊ
    "\u0BCB": "\u0D4B",  # ோ → ോ
    "\u0BCC": "\u0D4C",  # ௌ → ൌ
    "\u0BCD": "\u0D4D",  # ் → ് (pulli → chandrakkala)
    # Consonants
    "க": "ക", "ங": "ങ", "ச": "ച", "ஞ": "ഞ",
    "ட": "ട", "ண": "ണ", "த": "ത", "ந": "ന",
    "ப": "പ", "ம": "മ", "ய": "യ", "ர": "ര",
    "ல": "ല", "வ": "വ", "ழ": "ഴ", "ள": "ള",
    "ற": "റ", "ன": "ന",
    # Grantha
    "ஜ": "ജ", "ஷ": "ഷ", "ஸ": "സ", "ஹ": "ഹ",
    # Tamil numerals
    "௦": "൦", "௧": "൧", "௨": "൨", "௩": "൩",
    "௪": "൪", "௫": "൫", "௬": "൬", "௭": "൭",
    "௮": "൮", "௯": "൯",
}


def _char_map(text: str, mapping: dict[str, str]) -> str:
    return "".join(mapping.get(ch, ch) for ch in text)


# ─────────────────────────────────────────────
# LAYER 3 — ByT5 Neural (single shared model)
# ─────────────────────────────────────────────

class ByT5SongEngine:
    """
    Single ByT5 model instance handles both Tamil→Hindi and Tamil→Malayalam.
    Target language is expressed via the prompt prefix — no separate checkpoints.
    Rule-based character mapping runs first and is passed as a hint to ByT5.
    """

    MODEL_ID = "google/byt5-small"

    def __init__(self, use_neural: bool = False, device: str = "cpu"):
        self.use_neural = use_neural
        self.device = device
        self.model = None
        self.tokenizer = None
        self._ready = False

        if use_neural:
            self._try_load_model()

    def _try_load_model(self):
        try:
            from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
            import torch
            logger.info(f"Loading shared ByT5 song model: {self.MODEL_ID}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_ID)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                self.MODEL_ID, dtype=torch.float32
            ).to(self.device)
            self.model.eval()
            self._ready = True
            logger.info("ByT5 song model ready.")
        except Exception as e:
            logger.warning(f"ByT5 load failed ({e}). Using rule-only mode.")
            self._ready = False

    @lru_cache(maxsize=1024)
    def _neural(self, text: str, target: str, rule_hint: str) -> str:
        import torch
        prompt = f"transliterate Tamil to {target} script: {text}  hint: {rule_hint}"
        inputs = self.tokenizer(
            prompt, return_tensors="pt", max_length=512, truncation=True
        ).to(self.device)
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=256,
                num_beams=4,
                early_stopping=True,
                no_repeat_ngram_size=3,
                length_penalty=1.0,
            )
        return self.tokenizer.decode(out[0], skip_special_tokens=True).strip()

    def _transliterate(self, text: str, target: str) -> tuple[str, str]:
        mapping = TAMIL_TO_DEVA if target == "hindi" else TAMIL_TO_MALAYALAM
        rule_output = _char_map(text, mapping)

        if self._ready and self.use_neural:
            try:
                output = self._neural(text, target, rule_output)
                layer = "byt5+rules"
            except Exception as e:
                logger.error(f"ByT5 error ({target}): {e}")
                output, layer = rule_output, "rules-only (byt5 fallback)"
        else:
            output, layer = rule_output, "rules-only"

        return output, layer

    def convert(self, text: str) -> dict:
        """
        Convert Tamil song lyrics to both Hindi and Malayalam in a single call.
        Both translations share the same loaded model — single load, two outputs.
        """
        t0 = time.perf_counter()
        text = _normalize(text)

        hindi_out, hindi_layer = self._transliterate(text, "hindi")
        malayalam_out, malayalam_layer = self._transliterate(text, "malayalam")

        elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)

        return {
            "hindi":           hindi_out,
            "malayalam":       malayalam_out,
            "hindi_layer":     hindi_layer,
            "malayalam_layer": malayalam_layer,
            "time_ms":         elapsed_ms,
            "word_count":      len(text.split()),
            "model":           self.MODEL_ID if self._ready else "rule-based",
        }


# ─────────────────────────────────────────────
# Singleton
# ─────────────────────────────────────────────
_engine: Optional[ByT5SongEngine] = None


def get_song_engine() -> ByT5SongEngine:
    global _engine
    if _engine is None:
        _engine = ByT5SongEngine(use_neural=False)
    return _engine
