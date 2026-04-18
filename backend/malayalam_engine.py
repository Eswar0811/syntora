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
    # 4-consonant / longer clusters first
    "ക്ഷ്ണ": "kshna",
    "ന്ത്ര": "nthr",
    # 3-consonant clusters
    "ക്ഷ": "ksh",
    "ത്ര": "tr",
    "ജ്ഞ": "jnj",
    "ശ്ര": "shr",
    "ദ്ഭ": "dbh",
    "ദ്ധ": "ddh",
    "ഷ്ട്ര": "shtr",
    # 2-consonant clusters (most common)
    "ദ്ദ": "dd",
    "ണ്ട": "nd",   # retroflex nd — very common (unda, pandu…)
    "ന്ത": "nth",
    "ന്ദ": "nd",
    "ന്ന": "nn",
    "ണ്ണ": "nn",
    "ട്ട": "tt",   # retroflex double-t (attu, kutti…)
    "ത്ത": "tth",
    "ച്ച": "chch",
    "ഗ്ഗ": "gg",
    "ജ്ജ": "jj",
    "മ്മ": "mm",
    "ക്ക": "kk",
    "ല്ല": "ll",
    "ള്ള": "ll",
    "ങ്ങ": "ng",
    "ഞ്ഞ": "nj",
    "പ്പ": "pp",
    "ബ്ബ": "bb",
    "ക്ര": "kr",
    "പ്ര": "pr",
    "ഗ്ര": "gr",
    "ഭ്ര": "bhr",
    "ഷ്ട": "shd",  # as in kashtam, ishtam
    "ഷ്ണ": "shna",
    "ക്ത": "kth",
    "സ്ഥ": "sth",
    "ന്മ": "nm",
    "ര്ര": "rr",
    "ററ": "rr",
    "ര്‍": "r",
    # Additional common clusters
    "മ്പ": "mb",   # ambalam, sampram
    "ഞ്ച": "nch",  # panchayat, anchupaise
    "ങ്ക": "nk",   # thankam, sankar
    "ദ്ധ": "ddh",  # Buddha, siddha
    "ദ്ര": "dr",   # drama, dravam
    "ബ്ര": "br",   # brahmanam, brahmi
    "ദ്വ": "dv",   # dvani, dvaaram
    "ണ്ഡ": "nd",   # pandu, banda (retroflex ND)
    "ന്ധ": "ndh",  # gandham, bandhura
    "ഷ്ഠ": "shth", # nishtha, vrishtha
    "സ്ത": "sta",
    "ഹ്മ": "hm",
    "ട്ഡ": "dd",
}


def _is_vowel(ch: str) -> bool:     return ch in VOWELS
def _is_consonant(ch: str) -> bool: return ch in CONSONANTS
def _is_matra(ch: str) -> bool:     return ch in MATRAS
def _is_chillu(ch: str) -> bool:    return ch in CHILLU


# Word-final consonant+virama → chillu form (standalone final consonant)
_FINAL_TO_CHILLU: dict[str, str] = {
    "ന" + CHANDRAKKALA: "ൻ",  # dental n
    "ണ" + CHANDRAKKALA: "ൺ",  # retroflex n
    "ര" + CHANDRAKKALA: "ർ",  # r
    "ല" + CHANDRAKKALA: "ൽ",  # dental l
    "ള" + CHANDRAKKALA: "ൾ",  # retroflex l
}


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
                if i < n and word[i] in (ANUSVARA, VISARGA):
                    # Anusvara/Visarga after conjunct: inherent 'a' + nasal/aspirate
                    result.append("a" + MATRAS[word[i]])
                    i += 1
                elif i < n and _is_matra(word[i]):
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

    # ── Song / lyric vocabulary ──────────────────────────────────────────
    # Love / emotion
    "pranayam":  "പ്രണയം",  "ishttam":    "ഇഷ്ടം",    "sneham":    "സ്നേഹം",
    "manassu":   "മനസ്സ്",  "hridayam":   "ഹൃദയം",    "jeevitham": "ജീവിതം",
    "uyir":      "ഉയിർ",    "aathmavu":   "ആത്മാവ്",  "swapnam":   "സ്വപ്നം",
    "swapnangal":"സ്വപ്നങ്ങൾ","ormmakal":  "ഓർമ്മകൾ", "ormma":     "ഓർമ്മ",
    "vedana":    "വേദന",    "santhosham": "സന്തോഷം",  "dukham":    "ദുഃഖം",
    "aanandham": "ആനന്ദം",  "kashtam":    "കഷ്ടം",    "virahavyakulam":"വിരഹവ്യാകുലം",
    "virahavyakula":"വിരഹവ്യാകുല",
    # Nature / sky
    "mazha":     "മഴ",      "mazhayil":  "മഴയിൽ",    "veyil":     "വെയിൽ",
    "nilavu":    "നിലാവ്",  "nilavil":   "നിലാവിൽ",  "nakshatram":"നക്ഷത്രം",
    "nakshatrangal":"നക്ഷത്രങ്ങൾ","tharaka":"താരക",    "kadal":     "കടൽ",
    "kadalinil": "കടലിൽ",   "tharakam":  "താരകം",    "theeram":   "തീരം",
    "vaanam":    "വാനം",    "maram":     "മരം",       "pookkal":   "പൂക്കൾ",
    "poov":      "പൂവ്",    "poovinte":  "പൂവിന്റെ", "thaarakal": "താരകൾ",
    "akasham":   "ആകാശം",   "bhumi":     "ഭൂമി",
    # Common song phrases
    "ente jeevane": "എന്റെ ജീവനേ",
    "ente pranayame":"എന്റെ പ്രണയമേ",
    "ente manassin":"എന്റെ മനസ്സിൻ",
    "ninte ormakal":"നിന്റെ ഓർമ്മകൾ",
    "oru":       "ഒരു",     "oraal":     "ഒരാൾ",     "orukal":    "ഒരുകൽ",
    "varumo":    "വരുമോ",   "nee varumo":"നീ വരുമോ",  "ariyathe":  "അറിയാതെ",
    "ariyatha":  "അറിയാത",  "kaanunnu":  "കാണുന്നു", "kaanumbol": "കാണുമ്പോൾ",
    "kaanumpo":  "കാണുമ്പോ",
    # Movement / places
    "vazhi":     "വഴി",     "vazhikal":  "വഴികൾ",    "idathu":    "ഇടത്ത്",
    "ullil":     "ഉള്ളിൽ",  "manasil":   "മനസ്സിൽ",  "hridayathin":"ഹൃദയത്തിൻ",
    "kannukal":  "കണ്ണുകൾ", "kannil":    "കണ്ണിൽ",   "kannu":     "കണ്ണ്",
    "kayyil":    "കൈയ്യിൽ", "kai":       "കൈ",        "kayyum":    "കൈയ്യും",
    # Common film song words
    "ponnu":     "പൊന്ന്",  "ponnin":    "പൊന്നിൻ",  "thenkassi":  "തെൻകാശി",
    "thennal":   "തെന്നൽ",  "thenil":    "തേനിൽ",    "then":      "തേൻ",
    "poonila":   "പൂനിലാ",  "poomazha":  "പൂമഴ",     "puzhayum":  "പുഴയും",
    "puzhayil":  "പുഴയിൽ",  "puzha":     "പുഴ",       "malare":    "മലരേ",
    "malar":     "മലർ",     "malarinu":  "മലരിനു",
    # Adjectives (song lyrics)
    "sundaram":  "സുന്ദരം",  "sundari":   "സുന്ദരി",  "sundaran":  "സുന്ദരൻ",
    "manohari":  "മനോഹരി",   "manoharam": "മനോഹരം",   "thilakan":  "തിലകൻ",
    "roopam":    "രൂപം",     "sooryam":   "സൂര്യം",
    # Musical filler
    "aaa":       "ആ",        "ooo":       "ഓ",         "hmm":       "ഹ്മ്മ്",
    "la":        "ല",        "lo":        "ലോ",        "na":        "ന",
    "naa":       "നാ",       "ee":        "ഈ",
    # Body / senses
    "chundil":   "ചുണ്ടിൽ",  "chund":     "ചുണ്ട്",   "thoovam":   "തൂവൽ",
    "thoovalin": "തൂവലിൻ",
    # Time / waiting
    "naalayi":   "നാളായി",   "nerathu":   "നേരത്ത്",  "thamasam":  "താമസം",
    "veendum":   "വീണ്ടും",  "veendumum": "വീണ്ടുമും","ellam":     "എല്ലാം",
    "ellam um":  "എല്ലാം ഉം","ellarum":   "എല്ലാരും",
    # Common suffixes / particles encountered in songs
    "aayi":      "ആയി",     "aayittu":   "ആയിട്ട്",  "aayirunnu": "ആയിരുന്നു",
    "aakum":     "ആകും",    "aakatte":   "ആകട്ടെ",   "aakumbo":   "ആകുമ്പോ",
    "pol":       "പോൽ",     "pole":      "പോലെ",      "poleya":    "പോലെയ",
    "ennilum":   "എന്നിലും","nilkkunnu":  "നിൽക്കുന്നു",
    "nilkku":    "നിൽക്ക്", "nikkam":    "നിൽക്കാം",
    # Very common connectors
    "athin":     "അതിൻ",    "athinnu":   "അതിന്",    "athu":      "അത്",
    "ithu":      "ഇത്",     "ethu":      "ഏത്",       "enthu":     "എന്ത്",

    # ── Family / relationships ───────────────────────────────────────────────
    "amma":      "അമ്മ",    "amme":      "അമ്മേ",     "ammakku":   "അമ്മക്ക്",
    "achan":     "അച്ഛൻ",  "achane":    "അച്ഛനേ",    "achanku":   "അച്ഛനു",
    "chechi":    "ചേച്ചി",  "chechikku": "ചേച്ചിക്ക്","chechiyude": "ചേച്ചിയുടെ",
    "chettan":   "ചേട്ടൻ",  "chettanu":  "ചേട്ടനു",   "chettande": "ചേട്ടന്റെ",
    "mol":       "മോൾ",     "mole":      "മോളേ",       "mon":       "മോൻ",
    "mone":      "മോനേ",    "molude":    "മോളുടെ",    "monude":    "മോന്റെ",
    "ikka":      "ഇക്ക",    "umma":      "ഉമ്മ",       "uppa":      "ഉപ്പ",
    "ammumma":   "അമ്മുമ്മ","appupan":   "അപ്പൂപ്പൻ", "muthassi":  "മുത്തശ്ശി",
    "muthassan": "മുത്തശ്ശൻ","valiyamma": "വലിയമ്മ",   "valiyachan":"വലിയച്ഛൻ",
    "kutti":     "കുട്ടി",  "kuttiye":   "കുട്ടിയെ",  "kuttikkal": "കുട്ടികൾ",
    "makkal":    "മക്കൾ",   "makan":     "മകൻ",        "makal":     "മകൾ",
    "aniyathi":  "അനിയത്തി","aniyettan":  "അനിയേട്ടൻ",
    "eliyamma":  "ഏലിയമ്മ", "koochamma":  "കൂച്ചമ്മ",
    "swontham":  "സ്വന്തം",  "swonthamanu":"സ്വന്തമാണ്",

    # ── Food / drink ─────────────────────────────────────────────────────────
    "choru":     "ചോറ്",    "chaorum":   "ചോറും",     "kanji":     "കഞ്ഞി",
    "curry":     "കറി",     "chaaya":    "ചായ",        "kaapi":     "കാപ്പി",
    "appam":     "അപ്പം",   "puttu":     "പുട്ട്",     "idli":      "ഇഡ്ഡലി",
    "dosa":      "ദോശ",     "aviyal":    "അവിയൽ",      "sambar":    "സാമ്പാർ",
    "thoran":    "തോരൻ",    "payasam":   "പായസം",     "manga":     "മാങ്ങ",
    "pazham":    "പഴം",     "chakka":    "ചക്ക",       "thenga":    "തേങ്ങ",
    "ulli":      "ഉള്ളി",   "meenu":     "മീൻ",        "irachi":    "ഇറച്ചി",
    "mutta":     "മുട്ട",   "paal":      "പാൽ",        "velichenna":"വെളിച്ചെണ്ണ",
    "vellam":    "വെള്ളം",  "vellathin": "വെള്ളത്തിൽ",
    "naadan":    "നാടൻ",    "nellikkha": "നെല്ലിക്ക",

    # ── Common verbs (expanded) ───────────────────────────────────────────────
    "padikkuka": "പഠിക്കുക","padikkunnu": "പഠിക്കുന്നു","padikkum":  "പഠിക്കും",
    "padichchu": "പഠിച്ചു", "padichu":    "പഠിച്ചു",   "padichcha": "പഠിച്ച",
    "ezhuthuka": "എഴുതുക",  "ezhuthunnu": "എഴുതുന്നു", "ezhuthum":  "എഴുതും",
    "ezhuthi":   "എഴുതി",   "ezhuthiya":  "എഴുതിയ",
    "vaayikkuka":"വായിക്കുക","vaayikkunnu":"വായിക്കുന്നു","vaayikkum":"വായിക്കും",
    "vaayichu":  "വായിച്ചു",
    "oduka":     "ഓടുക",    "odunnu":     "ഓടുന്നു",   "odum":      "ഓടും",
    "odippoyi":  "ഓടിപ്പോയി","odi":        "ഓടി",
    "uranguka":  "ഉറങ്ങുക", "urangunnu":  "ഉറങ്ങുന്നു","urangum":   "ഉറങ്ങും",
    "urangi":    "ഉറങ്ങി",   "urangiyilla":"ഉറങ്ങിയില്ല",
    "irikka":    "ഇരിക്ക",  "irikkuka":   "ഇരിക്കുക",  "irikkunnu": "ഇരിക്കുന്നു",
    "irikkum":   "ഇരിക്കും", "irunnu":     "ഇരുന്നു",   "irunna":    "ഇരുന്ന",
    "nilkkuka":  "നിൽക്കുക","nilkkunnu":  "നിൽക്കുന്നു","nilkkum":  "നിൽക്കും",
    "ninnu":     "നിന്നു",   "ninna":      "നിന്ന",
    "nadakkuka": "നടക്കുക",  "nadakkunnu": "നടക്കുന്നു","nadakkum":  "നടക്കും",
    "nadannu":   "നടന്നു",   "nadanna":    "നടന്ന",
    "kazhikkuka":"കഴിക്കുക","kazhikkunnu":"കഴിക്കുന്നു","kazhikkum":"കഴിക്കും",
    "kazhichu":  "കഴിച്ചു",  "kazhicha":   "കഴിച്ച",
    "chirikkunnu":"ചിരിക്കുന്നു","chirikkum":"ചിരിക്കും","chirichu": "ചിരിച്ചു",
    "chiriku":   "ചിരിക്ക്", "chirikka":   "ചിരിക്ക",
    "karayuka":  "കരയുക",   "karayunnu":  "കരയുന്നു",  "karayum":   "കരയും",
    "karanju":   "കരഞ്ഞ്",  "karanjilla": "കരഞ്ഞില്ല",
    "paaduka":   "പാടുക",   "paadunnu":   "പാടുന്നു",  "paadum":    "പാടും",
    "paadi":     "പാടി",     "paadiya":    "പാടിയ",
    "aaduka":    "ആടുക",    "aadunnu":    "ആടുന്നു",   "aadum":     "ആടും",
    "aadi":      "ആടി",
    "thulluka":  "തുള്ളുക",  "thullunnu":  "തുള്ളുന്നു","thullum":   "തുള്ളും",
    "thurannu":  "തുറന്നു",  "thurakkuka": "തുറക്കുക",  "thurakkunnu":"തുറക്കുന്നു",
    "thurakkum": "തുറക്കും",
    "adakkuka":  "അടക്കുക",  "adakkunnu":  "അടക്കുന്നു","adakkum":   "അടക്കും",
    "adachu":    "അടച്ചു",
    "ayakkuka":  "അയക്കുക",  "ayakkunnu":  "അയക്കുന്നു","ayakkum":   "അയക്കും",
    "ayachu":    "അയച്ചു",
    "sahikkuka": "സഹിക്കുക","sahikkunnu": "സഹിക്കുന്നു","sahikkum":  "സഹിക്കും",
    "sahichu":   "സഹിച്ചു",  "sahichilla": "സഹിച്ചില്ല",
    "vaanguka":  "വാങ്ങുക",  "vaangukkunnu":"വാങ്ങുന്നു","vaangum":  "വാങ്ങും",
    "vaangi":    "വാങ്ങി",   "vaangiya":   "വാങ്ങിയ",
    "vilkkuka":  "വിൽക്കുക","vilkkunnu":  "വിൽക്കുന്നു","vilkkum":  "വിൽക്കും",
    "thoduka":   "തൊടുക",   "thodunnu":   "തൊടുന്നു",  "thodum":    "തൊടും",
    "tottilla":  "തൊട്ടില്ല",
    "thodangu":  "തുടങ്ങ്",  "thodangi":   "തുടങ്ങി",   "thodangum": "തുടങ്ങും",
    "vaazhuka":  "വാഴുക",   "vaazhunnu":  "വാഴുന്നു",  "vaazhum":   "വാഴും",
    "thodarunnu":"തുടരുന്നു","thodarium":  "തുടരും",    "thodarnnu": "തുടർന്നു",

    # ── Permission / necessity ───────────────────────────────────────────────
    "venda":     "വേണ്ട",   "venom":      "വേണം",       "venamo":    "വേണോ",
    "vene":      "വേണേ",    "vendatilla": "വേണ്ടതില്ല", "vendilla":  "വേണ്ടില്ല",
    "vendathe":  "വേണ്ടാതെ","paatilla":   "പാടില്ല",    "paatundo":  "പാടുണ്ടോ",
    "sheriyanu": "ശരിയാണ്",  "sheriya":    "ശരിയ",
    "pattum":    "പറ്റും",   "pattilla":   "പറ്റില്ല",   "pattumo":   "പറ്റുമോ",
    "varanam":   "വരണം",     "pokkanam":   "പോകണം",      "parayanam": "പറയണം",
    "irikkanam": "ഇരിക്കണം","nilkkanam":  "നിൽക്കണം",  "padikkanam":"പഠിക്കണം",
    "nokkanam":  "നോക്കണം",  "cheyyaanam": "ചെയ്യണം",
    "pokatte":   "പോകട്ടെ",  "varatte":    "വരട്ടെ",     "cheyyatte": "ചെയ്യട്ടെ",
    "irikkatte": "ഇരിക്കട്ടെ",

    # ── Conversational connectors ─────────────────────────────────────────────
    "angane":    "അങ്ങനെ",  "anganeya":   "അങ്ങനേ",     "anganeyaanu":"അങ്ങനെയാണ്",
    "ingane":    "ഇങ്ങനെ",  "inganeya":   "ഇങ്ങനേ",
    "pakshe":    "പക്ഷേ",   "paksham":    "പക്ഷം",
    "engil":     "എങ്കിൽ",  "enkil":      "എങ്കിൽ",     "enkilum":   "എങ്കിലും",
    "athinal":   "അതിനാൽ",  "athukondu":  "അതുകൊണ്ട്",
    "karanam":   "കാരണം",   "karanathal": "കാരണത്താൽ",
    "koode":     "കൂടെ",    "koottam":    "കൂട്ടം",     "koottathil":"കൂട്ടത്തിൽ",
    "koodutal":  "കൂടുതൽ",  "kurav":      "കുറവ്",       "kurach":    "കുറച്ч്",
    "orupadu":   "ഒരുപാട്",  "ithreyum":   "ഇത്രയും",    "athreyum":  "അത്രയും",
    "thirichu":  "തിരിച്ചു", "thirichupoya":"തിരിച്ചുപോയ",
    "nere":      "നേരെ",    "nerathu":    "നേരത്ത്",    "nerathe":   "നേരത്തേ",
    "mundil":    "മുന്നിൽ",  "pindil":     "പിന്നിൽ",    "maelil":    "മേലിൽ",
    "meele":     "മേലേ",     "keezhe":     "കീഴേ",        "keezhill":  "കീഴിൽ",
    "aduthu":    "അടുത്ത്",  "aduthuulla": "അടുത്തുള്ള", "dooram":    "ദൂരം",
    "doorathe":  "ദൂരതേ",    "doorathu":   "ദൂരത്ത്",    "arike":     "അരികേ",
    "arukil":    "അരുകിൽ",   "munnottu":   "മുന്നോട്ട്", "pinnoku":   "പിന്നോക്ക്",

    # ── Nature / places (expanded) ────────────────────────────────────────────
    "velicham":  "വെളിച്ചം","velichathin": "വെളിച്ചത്തിൽ","irul":    "ഇരുൾ",
    "irulil":    "ഇരുളിൽ",   "mazhakalam": "മഴക്കാലം",
    "kaatu":     "കാറ്റ്",   "kaattil":    "കാറ്റിൽ",    "kaattu":    "കാറ്റ്",
    "vaayu":     "വായു",     "agni":        "അഗ്നി",
    "nadhi":     "നദി",      "nadhiyil":   "നദിയിൽ",
    "mala":      "മല",       "malayil":    "മലയിൽ",      "malanil":   "മലനിൽ",
    "vayal":     "വയൽ",      "vayalil":    "വയലിൽ",
    "thottam":   "തോട്ടം",   "thottil":    "തോട്ടിൽ",
    "chola":     "ചോല",      "karayil":    "കരയിൽ",
    "theertham": "തീർഥം",    "theerthathin":"തീർഥത്തിൽ",
    "akam":      "അകം",      "akale":      "അകലേ",        "akalath":   "അകലത്ത്",

    # ── Birds / creatures (song vocab) ───────────────────────────────────────
    "kili":      "കിളി",     "kiliye":      "കിളിയേ",     "kilikal":   "കിളികൾ",
    "vaanambaadi":"വാനമ്പാടി","mayil":       "മയിൽ",      "kokku":     "കൊക്ക്",
    "kuruvikkal":"കുരുവിക്കൾ","pakshi":     "പക്ഷി",      "pakshikal": "പക്ഷികൾ",

    # ── Flowers (song vocab) ──────────────────────────────────────────────────
    "poo":       "പൂ",       "poovukal":    "പൂക്കൾ",     "mullappoo": "മുല്ലപ്പൂ",
    "thamaara":  "താമര",    "thaamarayil": "താമരയിൽ",
    "parijatham":"പാരിജാതം", "champa":      "ചമ്പ",

    # ── Sky / time (song vocab) ───────────────────────────────────────────────
    "thingal":   "തിങ്കൾ",  "thingalin":   "തിങ്കളിൽ",  "vennila":   "വെണ്ണില",
    "vennilav":  "വെണ്ണിലാ", "rathrikal":   "രാത്രികൾ",  "rathril":   "രാത്രിൽ",
    "iravu":     "ഇരവ്",    "pakalinil":   "പകലിൽ",

    # ── Music / arts ─────────────────────────────────────────────────────────
    "kavitha":   "കവിത",     "kavithe":     "കവിതേ",     "kavithakal":"കവിതകൾ",
    "paattu":    "പാട്ട്",   "paattukal":   "പാട്ടുകൾ",  "thalam":    "താളം",
    "raagam":    "രാഗം",     "gaanam":      "ഗാനം",       "gaaname":   "ഗാനമേ",
    "geetham":   "ഗീതം",     "sangeetham":  "സംഗീതം",    "thaalam":   "താളം",
    "nadam":     "നാദം",     "nadame":      "നാദമേ",

    # ── Emotions (expanded) ───────────────────────────────────────────────────
    "moham":     "മോഹം",    "mohichu":      "മോഹിച്ചു",
    "azhaku":    "അഴക്",    "azhakulla":    "അഴകുള്ള",   "azhake":    "അഴകേ",
    "viraham":   "വിരഹം",   "virahame":     "വിരഹമേ",
    "paavam":    "പാവം",    "paavame":      "പാവമേ",
    "aarogyam":  "ആരോഗ്യം","rogam":         "രോഗം",
    "sambhavam": "സംഭവം",   "kaaryam":      "കാര്യം",    "kaaryakal": "കാര്യങ്ങൾ",
    "kalyanam":  "കല്യാണം", "vivaham":      "വിവാഹം",
    "janmam":    "ജന്മം",   "janmantaram":  "ജന്മാന്തരം",
    "maranam":   "മരണം",    "vaazhvu":      "വാഴ്വ്",    "vaazhvil":  "വാഴ്വിൽ",
    "aayussu":   "ആയുസ്സ്",  "ullam":        "ഉള്ളം",     "ullathu":   "ഉള്ളത്",
    "bandham":   "ബന്ധം",   "bandhangal":   "ബന്ധങ്ങൾ",
    "aasha":     "ആശ",       "aasakku":      "ആശക്ക്",    "aasakal":   "ആശകൾ",
    "vishwaasam":"വിശ്വാസം","vishwaasathin": "വിശ്വാസത്തിൽ",
    "sathyam":   "സത്യം",   "satyamaanu":   "സത്യമാണ്",
    "shanthi":   "ശാന്തി",   "shaantham":    "ശാന്തം",
    "vegam":     "വേഗം",     "vegathil":     "വേഗത്തിൽ",
    "vijayam":   "വിജയം",   "vijayichu":    "വിജയിച്ചു",
    "abhilasham":"അഭിലാഷം",  "aasayam":      "ആശയം",
    "samsaaram": "സംസാരം",  "chodyam":      "ചോദ്യം",    "chodichal": "ചോദിച്ചാൽ",
    "padippu":   "പഠിപ്പ്",  "thozhil":      "തൊഴിൽ",
    "koottukaar":"കൂട്ടുകാർ","koottukaran":  "കൂട്ടുകാരൻ","koottukari":"കൂട്ടുകാരി",
    "mazhavillu":"മഴവില്ല്",

    # ── Colors ────────────────────────────────────────────────────────────────
    "velluttha": "വെളുത്ത",  "karutha":      "കറുത്ത",    "chaara":    "ചാര",
    "paccha":    "പച്ച",     "chuvanna":      "ചുവന്ന",    "manja":     "മഞ്ഞ",
    "neela":     "നീല",      "neelakasham":   "നീലാകാശം",
    "thangam":   "തങ്കം",    "thanka":        "തങ്ക",

    # ── Common social phrases ─────────────────────────────────────────────────
    "uyirinte":  "ഉയിരിന്റെ","uyiril":       "ഉയിരിൽ",
    "neeyum":    "നീയും",    "njaanum":       "ഞാനും",     "neeyum_njaanum":"നീ ഞാനും",
    "ente_koode":"എന്റെ കൂടെ","ninte_koode":  "നിന്റെ കൂടെ",
    "mangalam":  "മംഗളം",   "mangalya":      "മംഗല്യ",
    "swathanthryam":"സ്വാതന്ത്ര്യം",

    # ── Additional common conversational words ───────────────────────────────
    # Verbs — present / past / future / infinitive
    "pokuka":    "പോകുക",   "pokunnu":   "പോകുന്നു", "pokum":     "പോകും",
    "poyi":      "പോയി",    "poyittu":   "പോയിട്ട്", "pona":      "പോന",
    "varuka":    "വരുക",    "varunnu":   "വരുന്നു",  "varum":     "വരും",
    "vanna":     "വന്ന",    "vannilla":  "വന്നില്ല",
    "cheyyuka":  "ചെയ്യുക", "cheyyunnu": "ചെയ്യുന്നു","cheyyum":  "ചെയ്യും",
    "cheythu":   "ചെയ്തു",  "cheytha":   "ചെയ്ത",    "cheythal":  "ചെയ്താൽ",
    "nokku":     "നോക്ക്",  "nokkuka":   "നോക്കുക",  "nokkunnu":  "നോക്കുന്നു",
    "parayuka":  "പറയുക",   "parayunnu": "പറയുന്നു", "parayum":   "പറയും",
    "kekkunnu":  "കേൾക്കുന്നു","kekkam": "കേൾക്കാം",
    "thinnu":    "തിന്നു",  "thinnuka":  "തിന്നുക",  "thinnam":   "തിന്നാം",
    "kudikku":   "കുടിക്ക്","kudikkunnu":"കുടിക്കുന്നു","kudikkum":"കുടിക്കും",
    "uyaruka":   "ഉയരുക",   "uyarth":    "ഉയർത്ത്",  "uyarunnu":  "ഉയരുന്നു",
    "marannu":   "മറന്നു",  "marakkuka": "മറക്കുക",  "marakkunnu":"മറക്കുന്നു",
    "orkkuka":   "ഓർക്കുക","orkkunnu":  "ഓർക്കുന്നു","orkkam":   "ഓർക്കാം",
    "thudangu":  "തുടങ്ങ്", "thudangi":  "തുടങ്ങി",  "thudangum": "തുടങ്ങും",
    "theernnu":  "തീർന്നു", "theeruka":  "തീരുക",    "theerum":   "തീരും",
    "kittu":     "കിട്ടു",  "kittuka":   "കിട്ടുക",  "kittunnu":  "കിട്ടുന്നു",
    "kittum":    "കിട്ടും",  "kittilla":  "കിട്ടില്ല",
    "pidikku":   "പിടിക്ക്","pidikkunnu":"പിടിക്കുന്നു","pidikkum":"പിടിക്കും",
    "thinnuka":  "തിന്നുക", "thinnunnu": "തിന്നുന്നു",
    "undakku":   "ഉണ്ടാക്ക്","undakkuka": "ഉണ്ടാക്കുക","undakkunnu":"ഉണ്ടാക്കുന്നു",
    # Nouns — places and objects
    "veedu":     "വീട്",    "veetil":    "വീട്ടിൽ",  "veetinte":  "വീട്ടിന്റെ",
    "palli":     "പള്ളി",   "palliyil":  "പള്ളിയിൽ",
    "naadu":     "നാട്",    "naattil":   "നാട്ടിൽ",  "naattinte": "നാട്ടിന്റെ",
    "kaadu":     "കാട്",    "kaattil":   "കാട്ടിൽ",
    "nagar":     "നഗർ",    "nagaram":   "നഗരം",
    "paattal":   "പാട്ടൽ",  "paattu":    "പാട്ട്",   "paattinte": "പാട്ടിന്റെ",
    "vazhi":     "വഴി",     "vazhikal":  "വഴികൾ",
    "kayyil":    "കൈയ്യിൽ", "kayyum":    "കൈയ്യും",  "kai":       "കൈ",
    "kannu":     "കണ്ണ്",   "kannukal":  "കണ്ണുകൾ",  "kannil":    "കണ്ണിൽ",
    "chevi":     "ചെവി",    "cheviyil":  "ചെവിയിൽ",
    "vayaru":    "വയറ്",    "kayy":      "കൈ",        "kaal":      "കാൽ",
    "kayyaal":   "കൈകൊണ്ട്","kaallaal":  "കൈകൊണ്ട്",
    "thalayil":  "തലയിൽ",  "thala":     "തല",
    "manasil":   "മനസ്സിൽ", "manassu":   "മനസ്സ്",
    "hridayam":  "ഹൃദയം",   "hridayathin":"ഹൃദയത്തിൻ",
    # Adjectives
    "valiya":    "വലിയ",    "cheriya":   "ചെറിയ",    "puthiya":   "പുതിയ",
    "pazhaya":   "പഴയ",     "nalla":     "നല്ല",     "chetha":    "ചേത",
    "sundaram":  "സുന്ദരം", "manohari":  "മനോഹരി",
    "vellutta":  "വെള്ളത്ത","karuttha":  "കരുത്ത",
    "muzhuvan":  "മുഴുവൻ",  "ellam":     "എല്ലാം",
    # Numbers and time
    "onnu":      "ഒന്ന്",   "rendu":     "രണ്ട്",    "moonnu":    "മൂന്ന്",
    "naalu":     "നാല്",    "anch":      "അഞ്ച്",    "aaru":      "ആറ്",
    "ezhu":      "ഏഴ്",     "ettu":      "എട്ട്",    "ombathu":   "ഒമ്പത്",
    "pathu":     "പത്ത്",
    "innu":      "ഇന്ന്",   "nale":      "നാളെ",     "munnale":   "മുന്നാൾ",
    "pinnale":   "പിന്നാൾ", "ippo":      "ഇപ്പോ",    "appol":     "അപ്പോൾ",
    "nerthe":    "നേർത്ത",  "rathri":    "രാത്രി",   "pakal":     "പകൽ",
    "ravile":    "രാവിലെ",  "uyarunna":  "ഉയരുന്ന",
    # Common particles and suffixes
    "alle":      "അല്ലേ",   "ano":       "ആണോ",       "undo":      "ഉണ്ടോ",
    "aano":      "ആണോ",     "aanu":      "ആണ്",       "aayirunnu": "ആയിരുന്നു",
    "ennu":      "എന്ന്",   "ennalum":   "എന്നാലും", "enthinu":   "എന്തിന്",
    "evideyum":  "എവിടെയും","eppozhum":  "എപ്പോഴും", "enganeyum": "എങ്ങനെയും",
    "veendum":   "വീണ്ടും", "koodi":     "കൂടി",     "koodathe":  "കൂടാതെ",
    "illatha":   "ഇല്ലാത",  "illathavan":"ഇല്ലാതവൻ",
    # Common song lyric words not yet covered
    "thaaram":   "താരം",    "tharake":   "താരകേ",    "nilave":    "നിലാവേ",
    "mazhaye":   "മഴയേ",    "veyile":    "വെയിലേ",
    "poomazha":  "പൂമഴ",    "poonila":   "പൂനിലാ",
    "thulassi":  "തുളസ്സി",  "thulasi":   "തുളസി",
    "ormmakal":  "ഓർമ്മകൾ", "ormmayil":  "ഓർമ്മയിൽ",
    "sneham":    "സ്നേഹം",   "snehame":   "സ്നേഹമേ",
    "ishttam":   "ഇഷ്ടം",   "ishtam":    "ഇഷ്ടം",   "ishtapetta":"ഇഷ്ടപ്പെട്ട",
    "pranayam":  "പ്രണയം",   "pranayame": "പ്രണയമേ",
    "jeevane":   "ജീവനേ",    "jeevan":    "ജീവൻ",     "jeevanam":  "ജീവനം",
    "ullil":     "ഉള്ളിൽ",   "ulle":      "ഉള്ളേ",    "ullath":    "ഉള്ളത്",

    # ── Permission / ability ──────────────────────────────────────────────────
    "cheyyaam":  "ചെയ്യാം",  "varaam":     "വരാം",      "pokaam":     "പോകാം",
    "parayaam":  "പറയാം",    "kaanam":     "കാണാം",     "kekkaam":    "കേൾക്കാം",
    "thinnaam":  "തിന്നാം",   "kudikkaam":  "കുടിക്കാം", "uranukaam":  "ഉറങ്ങാം",
    "thudangaam":"തുടങ്ങാം",  "thodaraam":  "തുടരാം",

    # ── Family (expanded) ─────────────────────────────────────────────────────
    "penninkutti":"പെൺകുട്ടി","ammayi":     "അമ്മായി",  "ettan":      "ഏട്ടൻ",
    "ettante":   "ഏട്ടന്റെ",  "ammachi":    "അമ്മച്ചി", "appachen":   "അപ്പച്ചൻ",
    "maash":     "മാഷ്",      "maashe":     "മാഷേ",      "teacher":    "ടീച്ചർ",
    "chettar":   "ചേട്ടൻ",   "mole":       "മോളേ",      "mone":       "മോനേ",

    # ── Song / lyric vocabulary (expanded) ────────────────────────────────────
    "tharangam": "തരംഗം",    "kadalaazham":"കടലാഴം",   "mazhayil":   "മഴയിൽ",
    "veyilil":   "വെയിലിൽ",  "nilalil":    "നിഴലിൽ",   "thennalil":  "തെന്നലിൽ",
    "pulariyil": "പുലരിയിൽ", "sandhyayil": "സന്ധ്യയിൽ","orikkal":    "ഒരിക്കൽ",
    "tharamam":  "താരമം",    "nakshatrame":"നക്ഷത്രമേ","vennilavu":  "വെണ്ണിലാവ്",
    "chandrikakal":"ചന്ദ്രികക്കൽ","mazhavillu":"മഴവില്ല്",
    "gaaname":   "ഗാനമേ",    "raagame":    "രാഗമേ",     "thaname":    "താനമേ",
    "sangeethamay":"സംഗീതമായ","kavithakal": "കവിതകൾ",   "nadamay":    "നാദമായ",
    "unniyarcha":"ഉണ്ണിയാർച്ച","vazhappoov": "വഴിപ്പൂവ്","kumkumam":   "കുംകുമം",
    "ponnonam":  "പൊന്നോണം", "vishu":      "വിഷു",       "onam":       "ഓണം",
    "thiruvathira":"തിരുവാതിര","kaikottikali":"കൈകൊട്ടിക്കളി",

    # ── Additional verb forms ─────────────────────────────────────────────────
    "paranjathu":"പറഞ്ഞത്",   "paranjappol":"പറഞ്ഞപ്പോൾ","paranjitu":  "പറഞ്ഞിട്ട്",
    "kandathu":  "കണ്ടത്",    "kandappol":  "കണ്ടപ്പോൾ", "varunnathu": "വരുന്നത്",
    "pokunnathu":"പോകുന്നത്", "cheyyunnathu":"ചെയ്യുന്നത്",
    "ariyunnathu":"അറിയുന്നത്","parayunnathu":"പറയുന്നത്",
    "vannappo":  "വന്നപ്പോ",   "poyappo":    "പോയപ്പോ",   "cheythapo":  "ചെയ്തപ്പോ",
    "parajjapo": "പറഞ്ഞപ്പോ",
    "cheyyatte": "ചെയ്യട്ടെ", "pokatte":    "പോകട്ടെ",   "varatte":    "വരട്ടെ",
    "nilkatte":  "നിൽക്കട്ടെ",

    # ── Common conversational connectors ─────────────────────────────────────
    "ennalum":   "എന്നാലും",  "enthayalum": "എന്തായാലും","evideya":    "എവിടേ",
    "eppozhaayi":"എപ്പോഴായി","naalayi":    "നാളായി",   "athrem":     "അത്രേ",
    "ithreyum":  "ഇത്രയും",   "athreyum":   "അത്രയും",   "orupadu":    "ഒരുപാട്",
    "sherikkum": "ശരിക്കും",  "kashtamanu": "കഷ്ടമാണ്",  "sughamanu":  "സുഖമാണ്",

    # ── Nature / weather (expanded) ───────────────────────────────────────────
    "mazhakaalam":"മഴക്കാലം",  "veeshum":    "വീശും",     "kaattu veeshu":"കാറ്റ് വീശ്",
    "mazha peyyum":"മഴ പെയ്യും","veyil veeshum":"വെയിൽ വീശ്",
    "irul chirittu":"ഇരുൾ ചിരിട്ട്","thaarakam chiriyu":"താരകം ചിരിയ്",
    "nilavele":  "നിലാവേ",    "mazhaye":    "മഴയേ",      "thendrale":  "തെന്നലേ",

    # ── Food / Kerala cuisine ─────────────────────────────────────────────────
    "kanji":     "കഞ്ഞി",     "koozh":      "കൂഴ്",      "sambhaaram": "സൽക്കാരം",
    "pazhampori":"പഴം പൊരി",  "unniyappam": "ഉണ്ണിയപ്പം","kinnathappam":"കിണ്ണത്തപ്പം",
    "naadan kanji":"നാടൻ കഞ്ഞി","vishu katta":"വിഷുക്കൈനീട്ടം",

    # ── Common social phrases ─────────────────────────────────────────────────
    "saarasamanu":"സാരമാണ്",  "saaramilla": "സാരമില്ല",  "chuma":      "ചുമ്മ",
    "vellam poley":"വെള്ളം പോലെ","thenga poley":"തേങ്ങ പോലെ",
    "njan poven": "ഞാൻ പോവേൻ","njan varen":  "ഞാൻ വരേൻ",
    "enthokke":  "എന്തൊക്കെ",  "ellareyum":  "എല്ലാരേം",

    # ── Numbers ───────────────────────────────────────────────────────────────
    "onnu":      "ഒന്ന്",      "randu":      "രണ്ട്",     "moonu":      "മൂന്ന്",
    "naalu":     "നാല്",       "anchu":      "അഞ്ച്",     "aaru":       "ആറ്",
    "ezhu":      "ഏഴ്",         "ettu":       "എട്ട്",    "onpathu":    "ഒൻപത്",
    "pathu":     "പത്ത്",       "nooru":      "നൂറ്",      "aayiram":    "ആയിരം",

    # ── Colors ────────────────────────────────────────────────────────────────
    "chuvappu":  "ചുവപ്പ്",   "pachachu":   "പച്ചഛ",     "neelam":     "നീലം",
    "velluppu":  "വെള്ളുപ്പ്","karuppu":    "കറുപ്പ്",   "manjal":     "മഞ്ഞൾ",
    "naringa":   "നാരങ്ങ",    "chuvanna":   "ചുവന്ന",     "pachachoru": "പച്ചഛൊ",
    "neeli":     "നീലി",       "thilakan":   "തിളക്കം",

    # ── Body parts (extended) ─────────────────────────────────────────────────
    "thal":      "തല",          "kayyil":     "കൈയ്യിൽ",   "kalil":      "കാലിൽ",
    "kaavikal":  "കാവിക",       "mooku":      "മൂക്ക്",    "vayu":       "വായ്",
    "palli":     "പള്ളി",       "shariram":   "ശരീരം",     "hridayam":   "ഹൃദയം",
    "vayar":     "വയർ",          "mudi":       "മുടി",       "thol":       "തൊലി",

    # ── Daily life (extended) ─────────────────────────────────────────────────
    "veetu":     "വീട്",        "kaanam":     "കാണം",       "vazhiyilude":"വഴിയിലൂടെ",
    "pasham":    "പണം",          "pally":      "പള്ളി",      "shalaa":     "ശാല",
    "vaayankada":"വായ്ക്കട",    "ezhuthu":    "എഴുത്ത്",  "padippu":    "പഠിപ്പ്",
    "thozhil":   "തൊഴിൽ",      "vaahakam":   "വാഹനം",      "kaar":       "കാർ",
    "buss":      "ബസ്",          "railway":    "റെയിൽവേ",   "koodi":      "കൂടി",

    # ── Emotions / states (extended) ─────────────────────────────────────────
    "aagraham":  "ആഗ്രഹം",     "bhayam":     "ഭയം",        "santhosham": "സന്തോഷം",
    "dukkham":   "ദുഃഖം",       "kopam":      "കോപം",       "snehippikkan":"സ്നേഹിക്കൻ",
    "preethi":   "പ്രീതി",     "visvaasamanu":"വിശ്വാസമാണ്","visvaasamilla":"വിശ്വാസമില്ല",
    "orunna":    "ഒരുന്ന",      "kaanikkunna": "കാണിക്കുന്ന","manassilaayi": "മനസ്സിലായി",

    # ── Additional song vocabulary ────────────────────────────────────────────
    "paadalaanu":"പാടലാണ്",    "thirike":    "തിരിക",      "vaayilude":  "വായിലൂടെ",
    "sneham":    "സ്നേഹം",     "ulle":       "ഉള്ളേ",      "premanam":   "പ്രേമനം",
    "raavile":   "രാവിലേ",     "maazhayil":  "മഴയിൽ",      "poovinil":   "പൂവിൽ",
    "kadutha":   "കടുത്ത",     "mriduvaya":  "മൃദുവായ",    "madhuramaya":"മധുരമായ",
    "thaarathumby":"താരാതുമ്പി","kurinji":   "കുരിഞ്ഞി",  "neela koovalam":"നീല കൂവളം",
    "aadithyanu":"ആദിത്യൻ",    "chandranu":  "ചന്ദ്രൻ",    "ninakku":    "നിനക്ക്",
    "ennikku":   "എനിക്ക്",    "namukku":    "നമുക്ക്",    "ellaa naalum":"എല്ലാ നാളും",
    "aarum illaa":"ആരുമില്ല",  "aarum ondumilla":"ആരുമൊന്ദുമില്ല",
    "marakkaan":  "മറക്കാൻ",  "marichu":    "മറിഞ്ഞ്",   "kondu varum":"കൊണ്ടു വരും",
    "koode vaa": "കൂടേ വാ",   "koode nillkku":"കൂടേ നിൽക്കൂ",
}

# Manglish consonant sequences → Malayalam (LONGEST FIRST — order is critical)
MANGLISH_CONSONANTS: list[tuple[str, str]] = [
    # 5-char
    ("kshna", "ക്ഷ്ണ"),
    # 4-char
    ("kksh", "ക്ഷ"),  ("nthr", "ന്ത്ര"), ("chch", "ച്ച"),  ("shna", "ഷ്ണ"),
    # 3-char  — longer patterns MUST come before their prefixes
    ("ksh",  "ക്ഷ"),  ("shr",  "ശ്ര"),   ("jnj",  "ജ്ഞ"),
    ("nth",  "ന്ത"),  ("ndh",  "ന്ധ"),   ("nch",  "ഞ്ച"),
    ("chh",  "ഛ"),    ("thr",  "ത്ര"),   ("ngh",  "ങ്ങ"),
    ("tth",  "ത്ത"),  ("ddh",  "ദ്ധ"),   ("shd",  "ഷ്ട"),
    ("kth",  "ക്ത"),  ("sth",  "സ്ഥ"),   ("sta",  "സ്ത"),
    # 2-char
    ("zh",   "ഴ"),    ("sh",   "ശ"),     ("ch",   "ച"),
    ("kh",   "ഖ"),    ("gh",   "ഘ"),     ("jh",   "ഝ"),
    ("th",   "ത"),    ("dh",   "ധ"),     ("ph",   "ഫ"),
    ("bh",   "ഭ"),    ("ng",   "ങ"),     ("nj",   "ഞ"),
    ("tt",   "ട്ട"), ("dd",   "ദ്ദ"),   ("nn",   "ന്ന"),
    ("nd",   "ന്ദ"),  ("nt",   "ന്ത"),   ("mb",   "മ്പ"),
    ("mm",   "മ്മ"),  ("ll",   "ല്ല"),   ("kk",   "ക്ക"),
    ("pp",   "പ്പ"),  ("bb",   "ബ്ബ"),   ("rr",   "ർ"),
    ("tr",   "ത്ര"),  ("lp",   "ൽ"),     ("kr",   "ക്ര"),
    ("pr",   "പ്ര"),  ("gr",   "ഗ്ര"),   ("dr",   "ദ്ര"),
    ("br",   "ബ്ര"),  ("dv",   "ദ്വ"),   ("sv",   "സ്വ"),
    ("nk",   "ങ്ക"),
    # 1-char
    ("k",    "ക"),   ("g",    "ഗ"),   ("c",    "ക"),
    ("j",    "ജ"),   ("t",    "ട"),   ("d",    "ദ"),
    ("n",    "ന"),   ("p",    "പ"),   ("f",    "ഫ"),
    ("b",    "ബ"),   ("m",    "മ"),   ("y",    "യ"),
    ("r",    "ര"),   ("l",    "ല"),   ("v",    "വ"),
    ("w",    "വ"),   ("s",    "സ"),   ("h",    "ഹ"),
    ("z",    "ഴ"),   ("x",    "ക്സ"),
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
                result.append(mal_cons)          # inherent 'a' — no matra
                i = next_i2
            elif vowel_str:
                matra = VOWEL_TO_MATRA[vowel_str]
                result.append(mal_cons + matra)
                i = next_i2
            else:
                # No vowel follows — add chandrakkala (virama)
                # so consonant cluster or word-final consonant renders correctly
                result.append(mal_cons + CHANDRAKKALA)
            continue

        # Standalone vowel
        vowel_str, next_i = _match_vowel(word, i)
        if vowel_str:
            result.append(VOWEL_TO_INDEPENDENT.get(vowel_str, word[i]))
            i = next_i
            continue

        result.append(word[i])
        i += 1

    joined = "".join(result)
    # Convert word-final consonant+virama to the correct chillu (standalone final consonant)
    for cv, chillu in _FINAL_TO_CHILLU.items():
        if joined.endswith(cv):
            joined = joined[:-len(cv)] + chillu
            break
    return joined


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
        # ത്ത emits "tth" — normalise to "tth" (keep) but collapse triple+ t to "tt"
        text = re.sub(r"t{3,}", "tt", text)
        # Word-final "-oom" → "-um" (pokoom → pokum, sharikkoom → sharikkum)
        text = re.sub(r"oom\b", "um", text)
        # Word-final "-aav" → "-av" (common suffix normalisation)
        text = re.sub(r"aav\b", "av", text)
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
