"""
Hindi ↔ Hinglish Rule-Based Transliteration Engine
====================================================
Architecture mirrors the Tamil ByT5 engine:
  Layer 1 — Unicode normalizer + script detector
  Layer 2 — Rule-based G2P (Hindi → Hinglish)
           — Dictionary + rule-based phoneme parser (Hinglish → Hindi)
  Layer 3 — ByT5 neural correction (disabled; enable after fine-tuning)
  Layer 4 — Post-processor

Completely offline — no API key required.
"""

import re
import unicodedata
import time
import logging
from functools import lru_cache
from typing import Optional

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# LAYER 1 — Normalizer + Script Detector
# ─────────────────────────────────────────────

DEVANAGARI_START = 0x0900
DEVANAGARI_END   = 0x097F
_DEVANAGARI_RE   = re.compile(r"[\u0900-\u097F]")

HALANT   = "\u094D"  # ् virama
ANUSVARA = "\u0902"  # ं
VISARGA  = "\u0903"  # ः


def normalize_hindi(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\u200c", "").replace("\u200d", "")
    text = re.sub(r"[ \t]+", " ", text).strip()
    return text


def detect_script(text: str) -> str:
    return "hindi" if _DEVANAGARI_RE.search(text) else "hinglish"


def is_devanagari(ch: str) -> bool:
    return DEVANAGARI_START <= ord(ch) <= DEVANAGARI_END


# ─────────────────────────────────────────────
# LAYER 2A — Hindi → Hinglish  (Devanagari G2P)
# ─────────────────────────────────────────────

VOWELS: dict[str, str] = {
    "अ": "a",  "आ": "aa", "इ": "i",  "ई": "ee",
    "उ": "u",  "ऊ": "oo", "ऋ": "ri",
    "ए": "e",  "ऐ": "ai", "ओ": "o",  "औ": "au",
}

MATRAS: dict[str, str] = {
    "\u093E": "aa",  # ा
    "\u093F": "i",   # ि
    "\u0940": "ee",  # ी
    "\u0941": "u",   # ु
    "\u0942": "oo",  # ू
    "\u0943": "ri",  # ृ
    "\u0947": "e",   # े
    "\u0948": "ai",  # ै
    "\u094B": "o",   # ो
    "\u094C": "au",  # ौ
    ANUSVARA: "n",   # ं
    VISARGA:  "h",   # ः
    HALANT:   "",    # ् no inherent vowel
}

CONSONANTS: dict[str, str] = {
    "क": "k",  "ख": "kh", "ग": "g",  "घ": "gh", "ङ": "ng",
    "च": "ch", "छ": "chh","ज": "j",  "झ": "jh", "ञ": "ny",
    "ट": "t",  "ठ": "th", "ड": "d",  "ढ": "dh", "ण": "n",
    "त": "t",  "थ": "th", "द": "d",  "ध": "dh", "न": "n",
    "प": "p",  "फ": "ph", "ब": "b",  "भ": "bh", "म": "m",
    "य": "y",  "र": "r",  "ल": "l",  "व": "v",
    "श": "sh", "ष": "sh", "स": "s",  "ह": "h",
    "ळ": "l",
    "क़": "q",  "ख़": "kh", "ग़": "g",  "ज़": "z",
    "ड़": "r",  "ढ़": "rh", "फ़": "f",  "य़": "y",
}

CONJUNCTS: dict[str, str] = {
    "क्ष": "ksh", "त्र": "tr", "ज्ञ": "gya",
    "श्र": "shr", "द्व": "dw",
}


def hindi_to_hinglish_word(word: str) -> str:
    result: list[str] = []
    i = 0
    n = len(word)

    while i < n:
        ch = word[i]

        # Conjunct clusters
        matched = False
        for conjunct, roman in CONJUNCTS.items():
            if word[i:i + len(conjunct)] == conjunct:
                result.append(roman)
                i += len(conjunct)
                if i < n and word[i] in MATRAS:
                    result.append(MATRAS[word[i]])
                    i += 1
                matched = True
                break
        if matched:
            continue

        # Independent vowel
        if ch in VOWELS:
            result.append(VOWELS[ch])
            i += 1
            continue

        # Consonant
        if ch in CONSONANTS:
            base     = CONSONANTS[ch]
            next_ch  = word[i + 1] if i + 1 < n else ""
            next2_ch = word[i + 2] if i + 2 < n else ""

            if next_ch == HALANT:
                # No inherent vowel
                result.append(base)
                i += 2
            elif next_ch in MATRAS:
                result.append(base + MATRAS[next_ch])
                i += 2
            else:
                # Inherent 'a' — suppress only at word end (safest rule;
                # avoids over-deletion that turns "namaste" into "namste")
                at_end = (i + 1 >= n) or not is_devanagari(next_ch)
                result.append(base if at_end else base + "a")
                i += 1
            continue

        # Matra (standalone)
        if ch in MATRAS:
            result.append(MATRAS[ch])
            i += 1
            continue

        result.append(ch)
        i += 1

    return "".join(result)


def hindi_to_hinglish(text: str) -> str:
    text = normalize_hindi(text)
    parts = []
    for tok in text.split(" "):
        pre, core, post = _split_deva_token(tok)
        parts.append(pre + (hindi_to_hinglish_word(core) if core else "") + post)
    result = " ".join(parts)
    return (result[0].upper() + result[1:]) if result else result


def _split_deva_token(tok: str):
    i = 0
    while i < len(tok) and not is_devanagari(tok[i]):
        i += 1
    j = len(tok)
    while j > i and not is_devanagari(tok[j - 1]):
        j -= 1
    return tok[:i], tok[i:j], tok[j:]


# ─────────────────────────────────────────────
# LAYER 2B — Hinglish → Hindi  (Dict + Parser)
# ─────────────────────────────────────────────

# Common words dictionary — covers high-frequency cases the rule parser
# cannot handle reliably (schwa ambiguity, anusvara, special spellings)
HINGLISH_WORD_DICT: dict[str, str] = {
    # Pronouns
    "main": "मैं", "mein": "मैं", "mai": "मैं",
    "tum": "तुम", "aap": "आप", "woh": "वो", "wo": "वो",
    "hum": "हम", "ye": "ये", "yeh": "यह", "voh": "वह",
    "mujhe": "मुझे", "tumhe": "तुम्हें", "unhe": "उन्हें",
    "mera": "मेरा", "meri": "मेरी", "mere": "मेरे",
    "tera": "तेरा", "teri": "तेरी", "tere": "तेरे",
    "apna": "अपना", "apni": "अपनी", "apne": "अपने",
    # Verbs
    "hai": "है", "hain": "हैं", "tha": "था", "thi": "थी",
    "ho": "हो", "hun": "हूं", "hoon": "हूं",
    "kar": "कर", "karo": "करो", "karna": "करना",
    "ja": "जा", "jao": "जाओ", "jana": "जाना",
    "aa": "आ", "aao": "आओ", "aana": "आना",
    "de": "दे", "do": "दो", "dena": "देना",
    "le": "ले", "lo": "लो", "lena": "लेना",
    "aaunga": "आऊंगा", "aaungi": "आऊंगी",
    "karunga": "करूंगा", "karungi": "करूंगी",
    "jaunga": "जाऊंगा",
    # Common words
    "kya": "क्या", "kyo": "क्यों", "kyun": "क्यों",
    "nahi": "नहीं", "nahin": "नहीं", "na": "ना",
    "haan": "हाँ", "han": "हाँ", "ha": "हाँ",
    "accha": "अच्छा", "achha": "अच्छा", "acha": "अच्छा",
    "theek": "ठीक", "thik": "ठीक", "theek": "ठीक",
    "bahut": "बहुत", "bht": "बहुत",
    "bilkul": "बिल्कुल",
    "shukriya": "शुक्रिया", "dhanyavaad": "धन्यवाद",
    "namaste": "नमस्ते", "namaskar": "नमस्कार",
    "baat": "बात", "baten": "बातें",
    "dost": "दोस्त", "doston": "दोस्तों",
    "ghar": "घर", "naam": "नाम",
    "kab": "कब", "kahan": "कहाँ", "kaise": "कैसे",
    "kaun": "कौन", "kitna": "कितना", "kitni": "कितनी",
    "mein": "में",    # preposition "in/inside" (different from मैं)
    "se": "से", "ko": "को", "ka": "का", "ki": "की", "ke": "के",
    "par": "पर", "pe": "पे",
    "aur": "और", "ya": "या", "lekin": "लेकिन",
    "agar": "अगर", "toh": "तो", "to": "तो",
    "phir": "फिर", "fir": "फिर",
    "ab": "अब", "abhi": "अभी", "kal": "कल",
    "aaj": "आज", "parso": "परसों",
    "sab": "सब", "koi": "कोई", "kuch": "कुछ",
    "wala": "वाला", "wali": "वाली", "wale": "वाले",
    "karni": "करनी",
    # Common verb forms with clusters the parser struggles with
    "tumse": "तुमसे", "humse": "हमसे", "usse": "उससे",
    "isse": "इससे", "jisse": "जिससे", "kisse": "किससे",
    "tumko": "तुमको", "humko": "हमको",
    "samajh": "समझ", "samjha": "समझा",
    "sunna": "सुनना", "dekhna": "देखना",
    "padhna": "पढ़ना", "likhna": "लिखना",
    "khana": "खाना", "peena": "पीना",
    "sona": "सोना", "uthna": "उठना",
    "milna": "मिलना", "bolna": "बोलना",
    "rehna": "रहना", "chalana": "चलाना",
    "chahiye": "चाहिए", "chahie": "चाहिए",
    "zyada": "ज़्यादा", "thoda": "थोड़ा", "thodi": "थोड़ी",
    "sirf": "सिर्फ", "sach": "सच",
    "jhooth": "झूठ", "jhut": "झूठ",
    "pyaar": "प्यार", "ishq": "इश्क़",
    "zindagi": "ज़िंदगी", "duniya": "दुनिया",
}

# Vowel → dependent matra (after consonant). "a"→"" means inherent vowel.
VOWEL_TO_MATRA: dict[str, str] = {
    "aa": "\u093E",  # ा
    "ee": "\u0940",  # ी
    "ii": "\u0940",  # ी
    "oo": "\u0942",  # ू
    "uu": "\u0942",  # ू
    "ri": "\u0943",  # ृ
    "ai": "\u0948",  # ै
    "ae": "\u0948",  # ै
    "au": "\u094C",  # ौ
    "aw": "\u094C",  # ौ
    "i":  "\u093F",  # ि
    "u":  "\u0941",  # ु
    "e":  "\u0947",  # े
    "o":  "\u094B",  # ो
    "a":  "",        # inherent — no matra written
}

VOWEL_TO_INDEPENDENT: dict[str, str] = {
    "aa": "आ", "ee": "ई", "ii": "ई",
    "oo": "ऊ", "uu": "ऊ", "ri": "ऋ",
    "ai": "ऐ", "ae": "ऐ", "au": "औ", "aw": "औ",
    "a":  "अ", "i":  "इ", "u":  "उ",
    "e":  "ए", "o":  "ओ",
}

# Hinglish consonant sequences → Devanagari (LONGEST FIRST — order critical)
HINGLISH_CONSONANTS: list[tuple[str, str]] = [
    # 3-char
    ("ksh", "क्ष"), ("gya", "ज्ञ"), ("shr", "श्र"), ("chh", "छ"),
    # Doubled aspirates / geminates
    ("cch", "च्छ"), ("ddh", "द्ध"), ("tth", "त्थ"), ("ssh", "स्श"),
    ("kkh", "क्ख"),
    # 2-char
    ("ch",  "च"),  ("kh",  "ख"),  ("gh",  "घ"),  ("jh",  "झ"),
    ("th",  "थ"),  ("dh",  "ध"),  ("ph",  "फ"),  ("bh",  "भ"),
    ("sh",  "श"),  ("ng",  "ङ"),  ("ny",  "ञ"),  ("rh",  "ढ़"),
    ("tr",  "त्र"),("dw",  "द्व"),("sw",  "स्व"),
    # Geminates (doubled consonants)
    ("kk",  "क्क"), ("gg",  "ग्ग"), ("cc",  "च्च"), ("jj",  "ज्ज"),
    ("tt",  "त्त"), ("dd",  "द्द"), ("nn",  "न्न"), ("pp",  "प्प"),
    ("bb",  "ब्ब"), ("mm",  "म्म"), ("ll",  "ल्ल"), ("ss",  "स्स"),
    ("rr",  "र्र"), ("vv",  "व्व"),
    # 1-char
    ("k",   "क"),  ("g",   "ग"),  ("c",   "क"),  ("j",   "ज"),
    ("t",   "त"),  ("d",   "द"),  ("n",   "न"),  ("p",   "प"),
    ("f",   "फ़"), ("b",   "ब"),  ("m",   "म"),  ("y",   "य"),
    ("r",   "र"),  ("l",   "ल"),  ("v",   "व"),  ("w",   "व"),
    ("s",   "स"),  ("h",   "ह"),  ("q",   "क़"), ("z",   "ज़"),
    ("x",   "क्स"),
]

_VOWEL_PATTERNS: list[str] = sorted(VOWEL_TO_MATRA.keys(), key=len, reverse=True)


def _match_vowel(text: str, pos: int) -> tuple[str, int]:
    lower = text.lower()
    for v in _VOWEL_PATTERNS:
        if lower[pos:pos + len(v)] == v:
            return v, pos + len(v)
    return "", pos


def _match_consonant(text: str, pos: int) -> tuple[str, int]:
    lower = text.lower()
    for roman, deva in HINGLISH_CONSONANTS:
        if lower[pos:pos + len(roman)] == roman:
            return deva, pos + len(roman)
    return "", pos


def hinglish_to_hindi_word(word: str) -> str:
    """Convert a single Hinglish word to Devanagari using greedy phoneme parsing."""
    # Dictionary lookup first
    lookup = HINGLISH_WORD_DICT.get(word.lower())
    if lookup:
        return lookup

    result: list[str] = []
    i = 0
    n = len(word)

    while i < n:
        ch = word[i].lower()

        # Punctuation / digits
        if not ch.isalpha():
            result.append(word[i])
            i += 1
            continue

        # Consonant
        deva_cons, next_i = _match_consonant(word, i)
        if deva_cons:
            i = next_i
            vowel_str, next_i2 = _match_vowel(word, i)

            if vowel_str == "a":
                # Inherent 'a': write consonant, advance past 'a'
                result.append(deva_cons)
                i = next_i2
            elif vowel_str:
                matra = VOWEL_TO_MATRA[vowel_str]
                result.append(deva_cons + matra)
                i = next_i2
            else:
                # No vowel — add HALANT if another consonant follows, else bare consonant
                _, peek_i = _match_consonant(word, i)
                if peek_i > i:
                    result.append(deva_cons + HALANT)
                else:
                    result.append(deva_cons)
            continue

        # Standalone vowel
        vowel_str, next_i = _match_vowel(word, i)
        if vowel_str:
            result.append(VOWEL_TO_INDEPENDENT.get(vowel_str, word[i]))
            i = next_i
            continue

        result.append(word[i])
        i += 1

    return "".join(result)


def hinglish_to_hindi(text: str) -> str:
    text = normalize_hindi(text)
    parts = []
    for tok in text.split(" "):
        pre, core, post = _split_latin_token(tok)
        parts.append(pre + (hinglish_to_hindi_word(core) if core else "") + post)
    return " ".join(parts)


def _split_latin_token(tok: str):
    i = 0
    while i < len(tok) and not tok[i].isalpha():
        i += 1
    j = len(tok)
    while j > i and not tok[j - 1].isalpha():
        j -= 1
    return tok[:i], tok[i:j], tok[j:]


# ─────────────────────────────────────────────
# LAYER 3 — ByT5 Neural Correction (placeholder)
# ─────────────────────────────────────────────

class HindiTransliterationEngine:
    """
    Hybrid engine mirroring ByT5TamilEngine.
    Rule-based + dictionary handles all conversions (use_neural=False by default).
    Set use_neural=True after fine-tuning a ByT5 checkpoint on Hindi data.
    """

    MODEL_ID = "google/byt5-small"

    def __init__(self, use_neural: bool = False, device: str = "cpu"):
        self.use_neural = use_neural
        self.device     = device
        self.model      = None
        self.tokenizer  = None
        self._ready     = False
        if use_neural:
            self._try_load_model()

    def _try_load_model(self):
        try:
            from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
            import torch
            logger.info(f"Loading ByT5 for Hindi: {self.MODEL_ID}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_ID)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                self.MODEL_ID, dtype=torch.float32
            ).to(self.device)
            self.model.eval()
            self._ready = True
            logger.info("ByT5 Hindi model loaded.")
        except Exception as e:
            logger.warning(f"ByT5 load failed ({e}). Rules only.")

    @lru_cache(maxsize=2048)
    def _rule_cached(self, text: str, direction: str) -> str:
        if direction == "hindi_to_hinglish":
            return hindi_to_hinglish(text)
        return hinglish_to_hindi(text)

    def _byt5_correct(self, source: str, rule_hint: str, direction: str) -> str:
        if not self._ready:
            return rule_hint
        import torch
        prompt = f"{direction}: {source}  hint: {rule_hint}"
        inputs = self.tokenizer(
            prompt, return_tensors="pt", max_length=512, truncation=True
        ).to(self.device)
        with torch.no_grad():
            out = self.model.generate(
                **inputs, max_new_tokens=256, num_beams=4,
                early_stopping=True, no_repeat_ngram_size=3, length_penalty=1.0,
            )
        return self.tokenizer.decode(out[0], skip_special_tokens=True).strip()

    def convert(self, text: str) -> dict:
        t0        = time.perf_counter()
        text      = normalize_hindi(text)
        script    = detect_script(text)
        direction = "hindi_to_hinglish" if script == "hindi" else "hinglish_to_hindi"

        rule_output = self._rule_cached(text, direction)

        if self._ready and self.use_neural:
            try:
                output = self._byt5_correct(text, rule_output, direction)
                layer  = "byt5+rules"
            except Exception as e:
                logger.error(f"ByT5 error: {e}")
                output, layer = rule_output, "rules-only (byt5 fallback)"
        else:
            output, layer = rule_output, "rules-only"

        output     = _post_process(output, direction)
        elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)

        return {
            "output":       output,
            "direction":    direction,
            "input_script": script,
            "layer":        layer,
            "time_ms":      elapsed_ms,
            "word_count":   len(output.split()),
            "model":        self.MODEL_ID if self._ready else "rule-based",
        }


def _post_process(text: str, direction: str) -> str:
    text = re.sub(r"\s{2,}", " ", text).strip()
    if direction == "hindi_to_hinglish" and text:
        text = text[0].upper() + text[1:]
    return text


# ─────────────────────────────────────────────
# Singleton
# ─────────────────────────────────────────────
_engine: Optional[HindiTransliterationEngine] = None


def get_hindi_engine() -> HindiTransliterationEngine:
    global _engine
    if _engine is None:
        _engine = HindiTransliterationEngine(use_neural=False)
    return _engine
