"""
Unified translate() / translate_pair() for Syntora — fully bidirectional
and cross-language.

translate_pair(text, target) always returns (script_form, roman_form):
  - script_form: text in the TARGET language's native script
  - roman_form : romanised pronunciation guide

Cross-language pipeline:
  source_native → romanise (source engine) → target_native (target engine)

This means any of the four Indic scripts can appear in any of the four tabs:
  • same-language : Tamil lyrics → Tamil script + Tanglish
  • cross-language: Tamil lyrics in Hindi tab → Hindi script + Tanglish roman
                    Malayalam lyrics in Telugu tab → Telugu script + Manglish roman
                    … etc. for all 12 cross-language pairs.
"""
from __future__ import annotations
from functools import lru_cache

from byt5_engine      import (get_engine           as _get_tamil_engine,
                               is_tamil_char, tanglish_to_tamil)
from hindi_engine     import (get_hindi_engine     as _get_hindi_engine,
                               is_devanagari,
                               hindi_to_hinglish, hinglish_to_hindi)
from malayalam_engine import (get_malayalam_engine as _get_malayalam_engine,
                               is_malayalam,
                               malayalam_to_manglish, manglish_to_malayalam)
from telugu_engine    import (get_telugu_engine    as _get_telugu_engine,
                               is_telugu,
                               telugu_to_tenglish, tenglish_to_telugu)


# ── Script detection ──────────────────────────────────────────────────────────

def _detect_script(text: str) -> str:
    """Return dominant script: 'tamil' | 'hindi' | 'malayalam' | 'telugu' | 'latin'."""
    tamil = hindi = mal = tel = latin = 0
    for ch in text:
        if   is_tamil_char(ch): tamil += 1
        elif is_devanagari(ch): hindi += 1
        elif is_malayalam(ch):  mal   += 1
        elif is_telugu(ch):     tel   += 1
        elif ch.isalpha() and ord(ch) < 128: latin += 1

    total = tamil + hindi + mal + tel + latin
    if total == 0:
        return "latin"
    if latin / total > 0.9:
        return "latin"

    counts = {"tamil": tamil, "hindi": hindi, "malayalam": mal, "telugu": tel}
    return max(counts, key=counts.get)


# ── Romanise any source script to ASCII ──────────────────────────────────────

def _to_roman(text: str) -> tuple[str, str]:
    """
    Return (roman, source_script) for any input.

    roman        — ASCII romanised form of the text
    source_script — 'tamil' | 'hindi' | 'malayalam' | 'telugu' | 'latin'
    """
    script = _detect_script(text)
    if script == "tamil":
        roman = _get_tamil_engine().convert(text, mode="spoken").get("tanglish", text)
    elif script == "hindi":
        roman = hindi_to_hinglish(text)
    elif script == "malayalam":
        roman = malayalam_to_manglish(text)
    elif script == "telugu":
        roman = telugu_to_tenglish(text)
    else:
        roman = text  # already romanised
    return roman, script


# ── Main API ──────────────────────────────────────────────────────────────────

@lru_cache(maxsize=4096)
def translate_pair(text: str, target: str) -> tuple[str, str]:
    """
    Return (native_script, roman_form) for *target* language.

    target: 'tanglish' | 'hinglish' | 'manglish' | 'tenglish'

    Works for ANY input language:
      • same-language → uses the engine directly (highest fidelity)
      • cross-language → romanised bridge (source → roman → target script)

    The roman_form is always the romanised representation of the *source* text,
    serving as a pronunciation guide regardless of the tab selected.
    """
    text = text.strip()
    if not text:
        return "", ""

    roman, src = _to_roman(text)

    if target == "tanglish":
        if src == "tamil":
            # Same language: return original Tamil + its romanisation
            return text, roman
        # Cross-language: roman → Tamil script
        script = tanglish_to_tamil(roman)
        return script, roman

    if target == "hinglish":
        if src == "hindi":
            # Same language: return original Hindi + its romanisation
            return text, roman
        # Cross-language: roman → Hindi script
        script = hinglish_to_hindi(roman)
        return script, roman

    if target == "manglish":
        if src == "malayalam":
            return text, roman
        script = manglish_to_malayalam(roman)
        return script, roman

    if target == "tenglish":
        if src == "telugu":
            return text, roman
        script = tenglish_to_telugu(roman)
        return script, roman

    # Fallback (shouldn't be reached with valid targets)
    return text, roman


@lru_cache(maxsize=2048)
def translate(text: str, target: str = "auto") -> str:
    """
    Single-string transliteration.
    target: 'tanglish' | 'hinglish' | 'manglish' | 'tenglish' | 'auto'

    For explicit targets the appropriate engine handles both directions.
    For 'auto' the dominant script is detected and romanised.
    """
    text = text.strip()
    if not text:
        return ""

    has_tamil     = any(is_tamil_char(ch) for ch in text)
    has_hindi     = any(is_devanagari(ch) for ch in text)
    has_malayalam = any(is_malayalam(ch)  for ch in text)
    has_telugu    = any(is_telugu(ch)     for ch in text)
    has_latin     = any(ch.isalpha() and ord(ch) < 128 for ch in text)

    if target == "tanglish":
        if has_tamil:
            return _get_tamil_engine().convert(text, mode="spoken").get("tanglish", text)
        if has_latin:
            return tanglish_to_tamil(text)
        return text

    if target == "hinglish":
        if has_hindi or has_latin:
            return _get_hindi_engine().convert(text).get("output", text)
        return text

    if target == "manglish":
        if has_malayalam or has_latin:
            return _get_malayalam_engine().convert(text).get("output", text)
        return text

    if target == "tenglish":
        if has_telugu or has_latin:
            return _get_telugu_engine().convert(text).get("output", text)
        return text

    # auto: detect dominant script → romanise
    script = _detect_script(text)
    if script == "latin":
        return text
    if script == "tamil":
        return _get_tamil_engine().convert(text, mode="spoken").get("tanglish", text)
    if script == "hindi":
        return _get_hindi_engine().convert(text).get("output", text)
    if script == "telugu":
        return _get_telugu_engine().convert(text).get("output", text)
    return _get_malayalam_engine().convert(text).get("output", text)
