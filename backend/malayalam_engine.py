"""
Malayalam ↔ Manglish Rule-Based Transliteration Engine
=======================================================
Architecture mirrors the Tamil ByT5 and Hindi engines:
  Layer 1 — Unicode normalizer + script detector
  Layer 2 — Rule-based G2P  (Malayalam → Manglish)
           — Dictionary + rule-based phoneme parser (Manglish → Malayalam)
  Layer 3 — ByT5 neural correction (disabled; enable after fine-tuning)
  Layer 4 — Post-processor

Malayalam specifics handled:
  - Chillu letters (ൺ ൻ ർ ൽ ൾ) — final consonants with no inherent vowel
  - Chandrakkala (്) — virama suppressing inherent 'a'
  - ഴ (zh) — unique retroflex approximant
  - ള (l) — retroflex lateral
  - റ (r) — trill vs ര (flap)
  - ഞ (nj) — unique palatal nasal romanization in Malayalam
  - Anusvara ം and Visarga ഃ

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

MALAYALAM_START = 0x0D00
MALAYALAM_END   = 0x0D7F
_MALAYALAM_RE   = re.compile(r"[\u0D00-\u0D7F]")

CHANDRAKKALA = "\u0D4D"  # ് virama / chandrakkala
ANUSVARA     = "\u0D02"  # ം
VISARGA      = "\u0D03"  # ഃ

# Chillu letters: standalone final consonants (no inherent 'a')
CHILLU = {
    "\u0D7A": "n",   # ൺ (retroflex n)
    "\u0D7B": "n",   # ൻ (dental n)
    "\u0D7C": "r",   # ർ (r)
    "\u0D7D": "l",   # ൽ (dental l)
    "\u0D7E": "l",   # ൾ (retroflex l)
    "\u0D7F": "k",   # ൿ (k)
}


def normalize_malayalam(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\u200c", "").replace("\u200d", "")
    text = re.sub(r"[ \t]+", " ", text).strip()
    return text


def detect_script(text: str) -> str:
    return "malayalam" if _MALAYALAM_RE.search(text) else "manglish"


def is_malayalam(ch: str) -> bool:
    return MALAYALAM_START <= ord(ch) <= MALAYALAM_END


# ─────────────────────────────────────────────
# LAYER 2A — Malayalam → Manglish  (G2P)
# ─────────────────────────────────────────────

# Independent vowels
VOWELS: dict[str, str] = {
    "അ": "a",  "ആ": "aa", "ഇ": "i",  "ഈ": "ee",
    "ഉ": "u",  "ഊ": "oo", "ഋ": "ru",
    "എ": "e",  "ഏ": "e",  "ഐ": "ai",
    "ഒ": "o",  "ഓ": "o",  "ഔ": "au",
}

# Dependent vowel signs (matras)
MATRAS: dict[str, str] = {
    "\u0D3E": "aa",  # ാ
    "\u0D3F": "i",   # ി
    "\u0D40": "ee",  # ീ
    "\u0D41": "u",   # ു
    "\u0D42": "oo",  # ൂ
    "\u0D43": "ru",  # ൃ
    "\u0D46": "e",   # െ
    "\u0D47": "e",   # േ (long e, same in Manglish)
    "\u0D48": "ai",  # ൈ
    "\u0D4A": "o",   # ൊ
    "\u0D4B": "o",   # ോ (long o, same in Manglish)
    "\u0D4C": "au",  # ൌ
    "\u0D57": "au",  # ൗ (alternate au)
    ANUSVARA: "m",   # ം
    VISARGA:  "h",   # ഃ
    CHANDRAKKALA: "",  # ് (no vowel)
}

# Consonants
CONSONANTS: dict[str, str] = {
    # Velars
    "ക": "k",  "ഖ": "kh", "ഗ": "g",  "ഘ": "gh", "ങ": "ng",
    # Palatals
    "ച": "ch", "ഛ": "chh","ജ": "j",  "ഝ": "jh", "ഞ": "nj",
    # Retroflexes
    "ട": "t",  "ഠ": "tt", "ഡ": "d",  "ഢ": "dd", "ണ": "n",
    # Dentals
    "ത": "th", "ഥ": "thh","ദ": "d",  "ധ": "dh", "ന": "n",
    # Labials
    "പ": "p",  "ഫ": "ph", "ബ": "b",  "ഭ": "bh", "മ": "m",
    # Approximants / liquids
    "യ": "y",  "ര": "r",  "ല": "l",  "വ": "v",
    # Sibilants
    "ശ": "sh", "ഷ": "sh", "സ": "s",  "ഹ": "h",
    # Unique Malayalam consonants
    "ള": "l",  "ഴ": "zh", "റ": "r",
}

# Common conjunct clusters → single romanisation (check before single chars)
CONJUNCTS: dict[str, str] = {
    "ക്ഷ": "ksh",
    "ത്ര": "tr",
    "ജ്ഞ": "jnj",
    "ശ്ര": "shr",
    "ദ്ഭ": "dbh",
    "ദ്ധ": "ddh",
    "ദ്ദ": "dd",
    "ന്ത": "nth",
    "ന്ദ": "nd",
    "ന്ന": "nn",
    "ണ്ണ": "nn",
    "മ്മ": "mm",
    "ക്ക": "kk",
    "ത്ത": "tt",
    "ല്ല": "ll",
    "ള്ള": "ll",
    "ങ്ങ": "ng",   # double ṅ → single ng in Manglish
    "ഞ്ഞ": "nj",   # double ñ → single nj in Manglish
    "പ്പ": "pp",
    "ബ്ബ": "bb",
    "ര്ര": "rr",
    "ററ": "rr",
    "ര്‍": "r",
}


def _is_vowel(ch: str) -> bool:     return ch in VOWELS
def _is_consonant(ch: str) -> bool: return ch in CONSONANTS
def _is_matra(ch: str) -> bool:     return ch in MATRAS
def _is_chillu(ch: str) -> bool:    return ch in CHILLU


def malayalam_to_manglish_word(word: str) -> str:
    """Convert a single Malayalam word to Manglish."""
    result: list[str] = []
    i = 0
    n = len(word)

    while i < n:
        ch = word[i]

        # ── Chillu letter (standalone final consonant — no inherent 'a') ──
        if _is_chillu(ch):
            result.append(CHILLU[ch])
            i += 1
            continue

        # ── Conjunct cluster lookup ────────────────────────────────────────
        matched = False
        for conjunct, roman in CONJUNCTS.items():
            clen = len(conjunct)
            if word[i:i + clen] == conjunct:
                result.append(roman)
                i += clen
                if i < n and _is_matra(word[i]):
                    # Explicit vowel after conjunct
                    result.append(MATRAS[word[i]])
                    i += 1
                elif i < n and word[i] == CHANDRAKKALA:
                    # Explicit no-vowel marker after conjunct
                    i += 1
                else:
                    # No explicit matra or virama — inherent 'a' is preserved in Malayalam
                    result.append("a")
                matched = True
                break
        if matched:
            continue

        # ── Independent vowel ──────────────────────────────────────────────
        if _is_vowel(ch):
            result.append(VOWELS[ch])
            i += 1
            continue

        # ── Anusvara / Visarga (standalone) ───────────────────────────────
        if ch in (ANUSVARA, VISARGA):
            result.append(MATRAS[ch])
            i += 1
            continue

        # ── Consonant ──────────────────────────────────────────────────────
        if _is_consonant(ch):
            base    = CONSONANTS[ch]
            next_ch = word[i + 1] if i + 1 < n else ""

            if next_ch == CHANDRAKKALA:
                # Explicit virama: no vowel
                result.append(base)
                i += 2
            elif next_ch in (ANUSVARA, VISARGA):
                # Anusvara/Visarga after consonant: consonant keeps inherent 'a'
                # e.g. ര+ം = ra+m = "ram", not "rm"
                result.append(base + "a" + MATRAS[next_ch])
                i += 2
            elif _is_matra(next_ch):
                # Explicit vowel matra — check if anusvara follows matra
                vowel_sound = MATRAS[next_ch]
                peek = word[i + 2] if i + 2 < n else ""
                if peek in (ANUSVARA, VISARGA):
                    result.append(base + vowel_sound + MATRAS[peek])
                    i += 3
                else:
                    result.append(base + vowel_sound)
                    i += 2
            else:
                # Inherent 'a' — Malayalam preserves it; no schwa deletion rule
                result.append(base + "a")
                i += 1
            continue

        # ── Pass-through ───────────────────────────────────────────────────
        result.append(ch)
        i += 1

    return "".join(result)


def malayalam_to_manglish(text: str) -> str:
    """Full pipeline: normalize → split → convert each token → reassemble."""
    text = normalize_malayalam(text)
    parts = []
    for tok in text.split(" "):
        pre, core, post = _split_malayalam_token(tok)
        parts.append(pre + (malayalam_to_manglish_word(core) if core else "") + post)
    result = " ".join(parts)
    return (result[0].upper() + result[1:]) if result else result


def _split_malayalam_token(tok: str):
    i = 0
    while i < len(tok) and not is_malayalam(tok[i]):
        i += 1
    j = len(tok)
    while j > i and not is_malayalam(tok[j - 1]):
        j -= 1
    return tok[:i], tok[i:j], tok[j:]


# ─────────────────────────────────────────────
# LAYER 2B — Manglish → Malayalam  (Dict + Parser)
# ─────────────────────────────────────────────

# High-frequency words: dictionary lookup beats rule-based for these
MANGLISH_WORD_DICT: dict[str, str] = {
    # Pronouns
    "njan":     "ഞാൻ",  "njaan":   "ഞാൻ",  "naan":    "ഞാൻ",
    "nee":      "നീ",   "ningal":  "നിങ്ങൾ","ningale": "നിങ്ങളെ",
    "avan":     "അവൻ",  "aval":    "അവൾ",   "avaru":   "അവർ",
    "avar":     "അവർ",  "nammal":  "നമ്മൾ", "nammalku":"നമുക്ക്",
    "njangal":  "ഞങ്ങൾ","njangalku":"ഞങ്ങൾക്ക്",
    # Possessives
    "ente":     "എന്റെ","ninte":   "നിന്റെ","avante":  "അവന്റെ",
    "avalude":  "അവളുടെ","averude": "അവരുടെ","nammalude":"നമ്മുടെ",
    # Question words
    "enthu":    "എന്ത്","entha":   "എന്താ", "enthanu": "എന്താണ്",
    "evide":    "എവിടെ","eppo":    "എപ്പോ", "engane":  "എങ്ങനെ",
    "enganey":  "എങ്ങനെ","aaru":   "ആരു",   "aar":     "ആര്",
    "evidunnu": "എവിടുന്ന്","eppozhanu":"എപ്പോഴാണ്",
    # Affirmations / negations
    "athe":     "അതെ",  "alla":    "അല്ല",  "alle":    "അല്ലേ",
    "illa":     "ഇല്ല", "ille":    "ഇല്ലേ", "undo":    "ഉണ്ടോ",
    "undu":     "ഉണ്ട്","illallo": "ഇല്ലല്ലോ",
    # Greetings / courtesies
    "namaskaram":"നമസ്കാരം","nanni":  "നന്ദി", "shari":  "ശരി",
    "sharike":  "ശരിക്കേ","sherikkum":"ശരിക്കും","kollam": "കൊള്ളാം",
    "okkay":    "ഒക്കേ", "okay":    "ഒക്കേ",
    # Common adjectives
    "nalla":    "നല്ല",  "nallathu":"നല്ലത്","chetha":  "ചേത",
    "valiya":   "വലിയ",  "cheriya": "ചെറിയ", "puthiya": "പുതിയ",
    "pazhaya":  "പഴയ",   "nallavan":"നല്ലവൻ","nallaval":"നല്ലവൾ",
    # Common nouns
    "veedu":    "വീട്",  "veetil":  "വീട്ടിൽ","panam":  "പണം",
    "vazhi":    "വഴി",   "vandi":   "വണ്ടി",  "kaari":  "കാർ",
    "vishayam": "വിഷയം","sahayam":  "സഹായം", "sthalam":"സ്ഥലം",
    "kazhcha":  "കാഴ്ച","nanma":   "നന്മ",
    # Time words
    "innu":     "ഇന്ന്", "annu":    "അന്ന്",  "nale":   "നാളെ",
    "munnallu": "മുന്നാൾ","pinne":   "പിന്നെ", "ippo":   "ഇപ്പോ",
    "apo":      "അപ്പോ", "appol":   "അപ്പോൾ",
    # Common verbs / verb forms
    "parayunnu":"പറയുന്നു","parayam": "പറയാം",  "para":   "പറ",
    "parayuu":  "പറയൂ",  "parayaan":"പറയാൻ",
    "varunu":   "വരുന്നു","varum":   "വരും",    "vannu":  "വന്നു",
    "pokunnu":  "പോകുന്നു","pokum":   "പോകും",  "poyi":   "പോയി",
    "cheyyunnu":"ചെയ്യുന്നു","cheyyam":"ചെയ്യാം","cheyth": "ചെയ്ത്",
    "nokkam":   "നോക്കാം","nokkunnu":"നോക്കുന്നു",
    "ariyunnu": "അറിയുന്നു","ariyam": "അറിയാം",  "ariyilla":"അറിയില്ല",
    "kaanum":   "കാണും",  "kaanuka": "കാണുക",   "kandu":  "കണ്ടു",
    "masilayi": "മനസ്സിലായി","manasilaakam":"മനസ്സിലാകാം",
    "kodukkam": "കൊടുക്കാം","edukkanam":"എടുക്കണം",
    # Postpositions / particles
    "aal":      "ആൽ",    "ku":      "ക്ക്",   "il":     "ഇൽ",
    "nte":      "ന്റെ",  "ude":     "ഉടെ",
    # Common verb forms
    "aanu":     "ആണ്",   "aanel":   "ആണേൽ",  "aano":   "ആണോ",
    "sukham":   "സുഖം",  "sukhamaano":"സുഖമാണോ","sukhamaanu":"സുഖമാണ്",
    "aananu":   "ആണന്ന്","aallu":   "ആള്ള്",
    "kandittu": "കണ്ടിട്ട്","kittunno":"കിട്ടുന്നോ",
    "theernnu": "തീർന്ന്","thodangi": "തുടങ്ങി",
    "cheythu":  "ചെയ്തു","parayathe":"പറയാതെ",
    # Conjunctions
    "um":       "ഉം",    "athava":  "അഥവാ",   "ennal":  "എന്നാൽ",
    "ennalum":  "എന്നാലും","pinne":  "പിന്നെ", "karanam":"കാരണം",
    "athukond": "അതുകൊണ്ട്",
}

# Manglish consonant sequences → Malayalam (LONGEST FIRST)
MANGLISH_CONSONANTS: list[tuple[str, str]] = [
    # 3-char
    ("ksh", "ക്ഷ"), ("shr", "ശ്ര"), ("jnj", "ജ്ഞ"),
    ("nth", "ന്ത"), ("ndh", "ന്ധ"),
    # 2-char (check zh before z to avoid wrong match)
    ("zh",  "ഴ"),   ("ch",  "ച"),   ("kh",  "ഖ"),
    ("gh",  "ഘ"),   ("jh",  "ഝ"),   ("th",  "ത"),
    ("dh",  "ധ"),   ("ph",  "ഫ"),   ("bh",  "ഭ"),
    ("sh",  "ശ"),   ("ng",  "ങ"),   ("nj",  "ഞ"),
    ("tt",  "ഠ"),   ("dd",  "ഢ"),   ("nn",  "ന്ന"),
    ("mm",  "മ്മ"), ("ll",  "ല്ല"), ("kk",  "ക്ക"),
    ("tr",  "ത്ര"), ("rr",  "ർ"),
    # 1-char
    ("k",   "ക"),   ("g",   "ഗ"),   ("c",   "ക"),
    ("j",   "ജ"),   ("t",   "ട"),   ("d",   "ഡ"),
    ("n",   "ന"),   ("p",   "പ"),   ("f",   "ഫ"),
    ("b",   "ബ"),   ("m",   "മ"),   ("y",   "യ"),
    ("r",   "ര"),   ("l",   "ല"),   ("v",   "വ"),
    ("w",   "വ"),   ("s",   "സ"),   ("h",   "ഹ"),
    ("z",   "ഴ"),   ("x",   "ക്സ"),
]

# Vowel sound → dependent matra (after consonant). "a" → "" means inherent.
VOWEL_TO_MATRA: dict[str, str] = {
    "aa": "\u0D3E",  # ാ
    "ee": "\u0D40",  # ീ
    "ii": "\u0D40",  # ീ
    "oo": "\u0D42",  # ൂ
    "uu": "\u0D42",  # ൂ
    "ru": "\u0D43",  # ൃ
    "ai": "\u0D48",  # ൈ
    "ae": "\u0D48",  # ൈ
    "au": "\u0D4C",  # ൌ
    "aw": "\u0D4C",  # ൌ
    "i":  "\u0D3F",  # ി
    "u":  "\u0D41",  # ു
    "e":  "\u0D46",  # െ
    "o":  "\u0D4A",  # ൊ
    "a":  "",        # inherent — no matra written
}

# Vowel sound → independent vowel character (word-initial / standalone)
VOWEL_TO_INDEPENDENT: dict[str, str] = {
    "aa": "ആ",  "ee": "ഈ",  "ii": "ഈ",
    "oo": "ഊ",  "uu": "ഊ",  "ru": "ഋ",
    "ai": "ഐ",  "ae": "ഐ",  "au": "ഔ",  "aw": "ഔ",
    "a":  "അ",  "i":  "ഇ",  "u":  "ഉ",
    "e":  "എ",  "o":  "ഒ",
}

_VOWEL_PATTERNS: list[str] = sorted(VOWEL_TO_MATRA.keys(), key=len, reverse=True)


def _match_vowel(text: str, pos: int) -> tuple[str, int]:
    lower = text.lower()
    for v in _VOWEL_PATTERNS:
        if lower[pos:pos + len(v)] == v:
            return v, pos + len(v)
    return "", pos


def _match_consonant(text: str, pos: int) -> tuple[str, int]:
    lower = text.lower()
    for roman, mal in MANGLISH_CONSONANTS:
        if lower[pos:pos + len(roman)] == roman:
            return mal, pos + len(roman)
    return "", pos


def manglish_to_malayalam_word(word: str) -> str:
    """Convert a single Manglish word to Malayalam script."""
    # Dictionary lookup first
    lookup = MANGLISH_WORD_DICT.get(word.lower())
    if lookup:
        return lookup

    result: list[str] = []
    i = 0
    n = len(word)

    while i < n:
        ch = word[i].lower()

        if not ch.isalpha():
            result.append(word[i])
            i += 1
            continue

        # Consonant
        mal_cons, next_i = _match_consonant(word, i)
        if mal_cons:
            i = next_i
            vowel_str, next_i2 = _match_vowel(word, i)

            if vowel_str == "a":
                result.append(mal_cons)
                i = next_i2
            elif vowel_str:
                matra = VOWEL_TO_MATRA[vowel_str]
                result.append(mal_cons + matra)
                i = next_i2
            else:
                # No vowel — halant if another consonant follows, else bare
                _, peek_i = _match_consonant(word, i)
                if peek_i > i:
                    result.append(mal_cons + CHANDRAKKALA)
                else:
                    result.append(mal_cons)
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


def manglish_to_malayalam(text: str) -> str:
    text = normalize_malayalam(text)
    parts = []
    for tok in text.split(" "):
        pre, core, post = _split_latin_token(tok)
        parts.append(pre + (manglish_to_malayalam_word(core) if core else "") + post)
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

class MalayalamTransliterationEngine:
    """
    Hybrid engine mirroring ByT5TamilEngine and HindiTransliterationEngine.
    Rule-based + dictionary handles all conversions (use_neural=False by default).
    Set use_neural=True after fine-tuning a ByT5 checkpoint on Malayalam data.
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
            logger.info(f"Loading ByT5 for Malayalam: {self.MODEL_ID}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_ID)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                self.MODEL_ID, dtype=torch.float32
            ).to(self.device)
            self.model.eval()
            self._ready = True
            logger.info("ByT5 Malayalam model loaded.")
        except Exception as e:
            logger.warning(f"ByT5 load failed ({e}). Rules only.")

    @lru_cache(maxsize=2048)
    def _rule_cached(self, text: str, direction: str) -> str:
        if direction == "malayalam_to_manglish":
            return malayalam_to_manglish(text)
        return manglish_to_malayalam(text)

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
        text      = normalize_malayalam(text)
        script    = detect_script(text)
        direction = "malayalam_to_manglish" if script == "malayalam" else "manglish_to_malayalam"

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
    if direction == "malayalam_to_manglish":
        # Malayalam suffix convention: word-final "oom" → "um"
        # (e.g. "pokoom" → "pokum", "sharikkoom" → "sharikkum")
        text = re.sub(r"oom\b", "um", text)
        if text:
            text = text[0].upper() + text[1:]
    return text


# ─────────────────────────────────────────────
# Singleton
# ─────────────────────────────────────────────
_engine: Optional[MalayalamTransliterationEngine] = None


def get_malayalam_engine() -> MalayalamTransliterationEngine:
    global _engine
    if _engine is None:
        _engine = MalayalamTransliterationEngine(use_neural=False)
    return _engine
