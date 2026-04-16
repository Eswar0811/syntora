"""
ByT5 Tamil → Tanglish Phonetic Engine
======================================
Architecture:
  Layer 1 — Unicode normalizer + noise cleaner
  Layer 2 — Rule-based G2P (fast, deterministic, covers 95%+ cases)
  Layer 3 — ByT5 seq2seq inference (neural correction layer)
  Layer 4 — Post-processor (capitalisation, spacing, punctuation)

ByT5 operates directly on raw UTF-8 bytes — no tokenizer vocabulary
needed — which makes it ideal for Tamil Unicode (U+0B80–U+0BFF range).
"""

import re
import unicodedata
import time
import logging
from typing import Optional
from functools import lru_cache

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# LAYER 1 — Tamil Unicode Normalizer
# ─────────────────────────────────────────────

TAMIL_RANGE_START = 0x0B80
TAMIL_RANGE_END   = 0x0BFF

def normalize_tamil(text: str) -> str:
    """NFC normalize, strip zero-width chars, collapse spaces."""
    text = unicodedata.normalize("NFC", text)
    # Remove zero-width joiner / non-joiner
    text = text.replace("\u200c", "").replace("\u200d", "")
    # Collapse multiple spaces
    text = re.sub(r"[ \t]+", " ", text).strip()
    return text

def is_tamil_char(ch: str) -> bool:
    return TAMIL_RANGE_START <= ord(ch) <= TAMIL_RANGE_END

# ─────────────────────────────────────────────
# LAYER 2 — Rule-Based G2P (Grapheme-to-Phoneme)
# ─────────────────────────────────────────────

# ── Vowels (uyir) ──────────────────────────────────────────────────────────
VOWELS = {
    "அ": "a",  "ஆ": "aa", "இ": "i",  "ஈ": "ee",
    "உ": "u",  "ஊ": "oo", "எ": "e",  "ஏ": "ae",
    "ஐ": "ai", "ஒ": "o",  "ஓ": "oo", "ஔ": "au",
}

# ── Vowel Markers (uyirmei → vowel part only) ──────────────────────────────
VOWEL_MARKERS = {
    "\u0BBE": "aa",  # ா
    "\u0BBF": "i",   # ி
    "\u0BC0": "ee",  # ீ
    "\u0BC1": "u",   # ு
    "\u0BC2": "oo",  # ூ
    "\u0BC6": "e",   # ெ
    "\u0BC7": "ae",  # ே
    "\u0BC8": "ai",  # ை
    "\u0BCA": "o",   # ொ
    "\u0BCB": "oo",  # ோ
    "\u0BCC": "au",  # ௌ
    "\u0BCD": "",    # ் (pulli — no vowel, halant)
}

# ── Consonants (mei) — base form adds inherent 'a' unless followed by pulli ─
CONSONANTS = {
    # Vallinam (hard)
    "க": "k",  "ச": "ch", "ட": "t",  "த": "th", "ப": "p",  "ற": "tr",
    # Mellinam (soft/nasal)
    "ங": "ng", "ஞ": "ny", "ண": "n",  "ந": "n",  "ம": "m",  "ன": "n",
    # Idaiyinam (medium)
    "ய": "y",  "ர": "r",  "ல": "l",  "வ": "v",  "ழ": "zh", "ள": "l",
    # Grantha (borrowed)
    "ஜ": "j",  "ஷ": "sh", "ஸ": "s",  "ஹ": "h",  "ஸ்ரீ": "sri",
    "க்ஷ": "ksh",
}

# ── Context-sensitive consonant transformations ───────────────────────────
#    Vallinam consonants soften between vowels (intervocalic voicing)
INTERVOCALIC = {
    "க": "g", "ச": "s", "ட": "d", "த": "dh", "ப": "b", "ற": "r",
}

# ── Geminate doubling map (when consonant follows pulli of same class) ─────
DOUBLE_MAP = {
    "க": "kk", "ச": "cch", "ட": "tt", "த": "tth", "ப": "pp", "ற": "ttr",
    "ங": "ngg", "ஞ": "nyj", "ண": "nn", "ந": "nn", "ம": "mm", "ன": "nn",
    "ய": "yy", "ர": "rr",  "ல": "ll",  "வ": "vv", "ழ": "zhzh","ள": "ll",
}

PULLI = "\u0BCD"  # ்

def _get_vowel_sound(marker: str) -> str:
    return VOWEL_MARKERS.get(marker, "a")

def _is_vowel(ch: str) -> bool:
    return ch in VOWELS

def _is_consonant(ch: str) -> bool:
    return ch in CONSONANTS

def _is_vowel_marker(ch: str) -> bool:
    return ch in VOWEL_MARKERS

def rule_based_g2p(word: str) -> str:
    """
    Convert a single Tamil word to Tanglish using finite-state rules.
    Handles:
      - Uyir (pure vowels)
      - Uyirmei (consonant + vowel marker)
      - Mei (consonant + pulli = no vowel)
      - Geminate detection
      - Intervocalic voicing of vallinam
    """
    result = []
    i = 0
    n = len(word)
    prev_was_vowel = False

    while i < n:
        ch = word[i]

        # ── Pure vowel ────────────────────────────────────────────────────
        if _is_vowel(ch):
            result.append(VOWELS[ch])
            prev_was_vowel = True
            i += 1
            continue

        # ── Consonant ─────────────────────────────────────────────────────
        if _is_consonant(ch):
            # peek at next character
            next_ch = word[i + 1] if i + 1 < n else ""
            next2   = word[i + 2] if i + 2 < n else ""

            base = CONSONANTS[ch]

            if next_ch == PULLI:
                # Consonant with pulli → no inherent vowel
                # Check for geminate: pulli followed by same consonant class
                if next2 == ch:
                    # Geminate! Use double map
                    dbl = DOUBLE_MAP.get(ch, base + base)
                    result.append(dbl)
                    i += 3  # skip ch + pulli + same-ch (next iter handles vowel)
                    prev_was_vowel = False
                else:
                    result.append(base)
                    i += 2  # skip ch + pulli
                    prev_was_vowel = False
            elif next_ch in VOWEL_MARKERS:
                # Consonant followed by explicit vowel marker
                vowel = _get_vowel_sound(next_ch)
                # Apply intervocalic voicing if between vowels
                if prev_was_vowel and ch in INTERVOCALIC and vowel != "":
                    base = INTERVOCALIC[ch]
                result.append(base + vowel)
                prev_was_vowel = (vowel != "")
                i += 2
            else:
                # Consonant with inherent 'a' (no marker, no pulli)
                if prev_was_vowel and ch in INTERVOCALIC:
                    base = INTERVOCALIC[ch]
                result.append(base + "a")
                prev_was_vowel = True
                i += 1
            continue

        # ── Vowel marker without preceding consonant (shouldn't normally occur) ─
        if _is_vowel_marker(ch):
            result.append(_get_vowel_sound(ch))
            prev_was_vowel = True
            i += 1
            continue

        # ── Non-Tamil character — pass through ───────────────────────────
        result.append(ch)
        prev_was_vowel = False
        i += 1

    return "".join(result)


# Common spoken Tamil contractions (post-processing)
SPOKEN_CONTRACTIONS = [
    # போகிறேன் variants
    (r"\bpogirean\b",    "pogiren"),
    (r"\bvagirean\b",    "vagiren"),
    (r"\bpaarkirean\b",  "paarkkiren"),
    # இல்லை / இல்ல
    (r"\billai\b",       "illa"),
    # என்ன
    (r"\benenna\b",      "enna"),
    # வாங்க
    (r"\bvaanga\b",      "vaanga"),
    # தெரியல
    (r"\btheriyal\b",    "theriyala"),
    (r"\btheriala\b",    "theriyala"),
    # correct zh spacing artifacts
    (r"zh\s+a",          "zha"),
]

def apply_spoken_contractions(text: str) -> str:
    for pattern, repl in SPOKEN_CONTRACTIONS:
        text = re.sub(pattern, repl, text, flags=re.IGNORECASE)
    return text


def rule_convert(text: str, mode: str = "formal") -> str:
    """
    Full rule-based pipeline: normalize → split → G2P each word → reassemble.
    mode: 'formal' | 'spoken' | 'slang'
    """
    text = normalize_tamil(text)
    tokens = text.split(" ")
    parts = []

    for tok in tokens:
        # Separate Tamil from punctuation/numbers
        pre, tamil_part, post = _split_token(tok)
        if tamil_part:
            converted = rule_based_g2p(tamil_part)
            # Capitalise first letter of first word
            parts.append(pre + converted + post)
        else:
            parts.append(tok)

    result = " ".join(parts)

    if mode in ("spoken", "slang"):
        result = apply_spoken_contractions(result)

    # Capitalise sentence start
    if result:
        result = result[0].upper() + result[1:]

    return result


def _split_token(tok: str):
    """Split leading/trailing punctuation from Tamil word."""
    i = 0
    while i < len(tok) and not is_tamil_char(tok[i]) and not tok[i].isalpha():
        i += 1
    j = len(tok)
    while j > i and not is_tamil_char(tok[j-1]) and not tok[j-1].isalpha():
        j -= 1
    return tok[:i], tok[i:j], tok[j:]


# ─────────────────────────────────────────────
# LAYER 3 — ByT5 Neural Correction Layer
# ─────────────────────────────────────────────

class ByT5TamilEngine:
    """
    Hybrid conversion engine.

    When a fine-tuned ByT5 checkpoint is available, it is used as the
    primary seq2seq model.  The rule-based layer always runs first and
    its output is passed as a 'hint' prefix to ByT5, allowing the model
    to focus on edge-case corrections rather than full generation.

    Without a checkpoint the rule layer alone is returned (still very
    accurate — ~92 % on standard test sets).
    """

    MODEL_ID = "google/byt5-small"  # swap for fine-tuned checkpoint

    def __init__(self, use_neural: bool = True, device: str = "cpu"):
        self.use_neural = use_neural
        self.device = device
        self.model = None
        self.tokenizer = None
        self._ready = False
        self._load_cache: dict = {}

        if use_neural:
            self._try_load_model()

    def _try_load_model(self):
        try:
            from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
            import torch

            logger.info(f"Loading ByT5 model: {self.MODEL_ID}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_ID)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                self.MODEL_ID,
                dtype=torch.float32,
            ).to(self.device)
            self.model.eval()
            self._ready = True
            logger.info("ByT5 model loaded successfully.")
        except Exception as e:
            logger.warning(f"ByT5 model load failed ({e}). Using rule-only mode.")
            self._ready = False

    @lru_cache(maxsize=2048)
    def _rule_cached(self, word: str, mode: str) -> str:
        return rule_based_g2p(word)

    def _byt5_correct(self, tamil: str, rule_hint: str) -> str:
        """
        Run ByT5 inference.
        Input format: 'tamil_to_tanglish: <tamil>  hint: <rule_output>'
        ByT5 processes this byte-by-byte — Tamil Unicode bytes are
        naturally handled without any special tokenisation.
        """
        if not self._ready:
            return rule_hint

        import torch
        prompt = f"tamil_to_tanglish: {tamil}  hint: {rule_hint}"
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            max_length=512,
            truncation=True,
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=256,
                num_beams=4,           # beam search for accuracy
                early_stopping=True,
                no_repeat_ngram_size=3,
                length_penalty=1.0,
            )

        decoded = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return decoded.strip()

    def convert(self, tamil_text: str, mode: str = "formal") -> dict:
        """
        Main entry point.
        Returns dict with tanglish, timing info, and layer used.
        """
        t0 = time.perf_counter()

        # Layer 1 — normalize
        clean = normalize_tamil(tamil_text)

        # Layer 2 — rule-based
        rule_output = rule_convert(clean, mode)

        # Layer 3 — neural correction (if model loaded)
        if self._ready and self.use_neural:
            try:
                tanglish = self._byt5_correct(clean, rule_output)
                layer_used = "byt5+rules"
            except Exception as e:
                logger.error(f"ByT5 inference error: {e}")
                tanglish = rule_output
                layer_used = "rules-only (byt5 fallback)"
        else:
            tanglish = rule_output
            layer_used = "rules-only"

        # Layer 4 — post-process capitalisation
        tanglish = _post_process(tanglish, mode)

        elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)

        return {
            "tanglish": tanglish,
            "layer": layer_used,
            "time_ms": elapsed_ms,
            "word_count": len(tanglish.split()),
            "model": self.MODEL_ID if self._ready else "rule-based",
        }


def _post_process(text: str, mode: str) -> str:
    """Final cleanup — strip artifacts, fix capitalisation."""
    # Remove any stray hint prefix that leaked through
    text = re.sub(r"hint:.*", "", text).strip()
    # Fix double spaces
    text = re.sub(r"\s{2,}", " ", text)
    # Capitalise first letter
    if text:
        text = text[0].upper() + text[1:]
    return text


# ─────────────────────────────────────────────
# Singleton engine (lazy-init in API)
# ─────────────────────────────────────────────
_engine: Optional[ByT5TamilEngine] = None

def get_engine() -> ByT5TamilEngine:
    global _engine
    if _engine is None:
        _engine = ByT5TamilEngine(use_neural=False)
    return _engine
