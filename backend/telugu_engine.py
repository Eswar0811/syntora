"""
Telugu ↔ Tenglish Rule-Based + ByT5 Transliteration Engine
===========================================================
Architecture mirrors the other Syntora engines:
  Layer 1 — Unicode normalizer + script detector
  Layer 2 — Rule-based G2P  (Telugu → Tenglish)
           — Dictionary + rule-based phoneme parser (Tenglish → Telugu)
  Layer 3 — ByT5 neural correction (correction layer over rule output)
  Layer 4 — Post-processor

Telugu specifics handled:
  - Ottulu (half-consonants formed with virama ్)
  - Anusvara ం (m/n nasal) and Visarga ః (h aspirate)
  - Unique Telugu: ళ (retroflex l), ఱ (trill r)
  - Geminate consonants (very common in Telugu)
  - Distinguishing ట/ఠ (retroflex t/th) from త/థ (dental th/thh)
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

TELUGU_START = 0x0C00
TELUGU_END   = 0x0C7F
_TELUGU_RE   = re.compile(r"[\u0C00-\u0C7F]")

VIRAMA   = "\u0C4D"  # ్  (halant / ottulu marker)
ANUSVARA = "\u0C02"  # ం
VISARGA  = "\u0C03"  # ః


def normalize_telugu(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\u200c", "").replace("\u200d", "")
    text = re.sub(r"[ \t]+", " ", text).strip()
    return text


def detect_script(text: str) -> str:
    return "telugu" if _TELUGU_RE.search(text) else "tenglish"


def is_telugu(ch: str) -> bool:
    return TELUGU_START <= ord(ch) <= TELUGU_END


# ─────────────────────────────────────────────
# LAYER 2A — Telugu → Tenglish  (G2P)
# ─────────────────────────────────────────────

VOWELS: dict[str, str] = {
    "అ": "a",  "ఆ": "aa", "ఇ": "i",  "ఈ": "ee",
    "ఉ": "u",  "ఊ": "oo", "ఋ": "ru",
    "ఎ": "e",  "ఏ": "ae", "ఐ": "ai",
    "ఒ": "o",  "ఓ": "oo", "ఔ": "au",
}

MATRAS: dict[str, str] = {
    "\u0C3E": "aa",   # ా
    "\u0C3F": "i",    # ి
    "\u0C40": "ee",   # ీ
    "\u0C41": "u",    # ు
    "\u0C42": "oo",   # ూ
    "\u0C43": "ru",   # ృ
    "\u0C46": "e",    # ె
    "\u0C47": "ae",   # ే
    "\u0C48": "ai",   # ై
    "\u0C4A": "o",    # ొ
    "\u0C4B": "oo",   # ో
    "\u0C4C": "au",   # ౌ
    VIRAMA:   "",     # ్ (no vowel)
    ANUSVARA: "m",    # ం
    VISARGA:  "h",    # ః
}

CONSONANTS: dict[str, str] = {
    # Velars
    "క": "k",   "ఖ": "kh",  "గ": "g",   "ఘ": "gh",  "ఙ": "ng",
    # Palatals
    "చ": "ch",  "ఛ": "chh", "జ": "j",   "ఝ": "jh",  "ఞ": "nj",
    # Retroflexes (ट row)
    "ట": "t",   "ఠ": "th",  "డ": "d",   "ఢ": "dh",  "ణ": "n",
    # Dentals (त row)
    "త": "th",  "థ": "thh", "ద": "d",   "ధ": "dh",  "న": "n",
    # Labials
    "ప": "p",   "ఫ": "ph",  "బ": "b",   "భ": "bh",  "మ": "m",
    # Approximants / liquids
    "య": "y",   "ర": "r",   "ల": "l",   "వ": "v",
    # Sibilants
    "శ": "sh",  "ష": "sh",  "స": "s",   "హ": "h",
    # Unique Telugu
    "ళ": "ll",  "ఱ": "r",
}

# Conjuncts (check before individual chars — longest first)
CONJUNCTS: dict[str, str] = {
    "క్ష":  "ksh",
    "జ్ఞ":  "gna",
    "శ్ర":  "shr",
    "ప్ర":  "pr",
    "త్ర":  "tr",
    "బ్ర":  "br",
    "గ్ర":  "gr",
    "క్ర":  "kr",
    "ద్ర":  "dr",
    "స్ర":  "sr",
    "వ్ర":  "vr",
    "న్న":  "nn",
    "ళ్ళ":  "ll",
    "క్క":  "kk",
    "గ్గ":  "gg",
    "చ్చ":  "chch",
    "జ్జ":  "jj",
    "ట్ట":  "tt",
    "డ్డ":  "dd",
    "త్త":  "tth",
    "ద్ద":  "ddh",
    "న్ద":  "nd",
    "న్త":  "nth",
    "ప్ప":  "pp",
    "బ్బ":  "bb",
    "మ్మ":  "mm",
    "ర్ర":  "rr",
    "ల్ల":  "ll",
    "వ్వ":  "vv",
    "స్స":  "ss",
    "ష్ఠ":  "sht",
    "ంగ":   "nga",
    "ంక":   "nka",
    "ంత":   "ntha",
    "ంద":   "ndha",
    "ంప":   "mpa",
}


def _is_vowel(ch: str) -> bool:     return ch in VOWELS
def _is_consonant(ch: str) -> bool: return ch in CONSONANTS
def _is_matra(ch: str) -> bool:     return ch in MATRAS


def telugu_to_tenglish_word(word: str) -> str:
    """Convert a single Telugu word to Tenglish."""
    result: list[str] = []
    i = 0
    n = len(word)

    while i < n:
        ch = word[i]

        # ── Conjunct cluster ──────────────────────────────────────────────
        matched = False
        for conjunct, roman in CONJUNCTS.items():
            clen = len(conjunct)
            if word[i:i + clen] == conjunct:
                result.append(roman)
                i += clen
                if i < n and word[i] in (ANUSVARA, VISARGA):
                    result.append("a" + MATRAS[word[i]])
                    i += 1
                elif i < n and _is_matra(word[i]):
                    result.append(MATRAS[word[i]])
                    i += 1
                elif i < n and word[i] == VIRAMA:
                    i += 1
                else:
                    result.append("a")
                matched = True
                break
        if matched:
            continue

        # ── Independent vowel ─────────────────────────────────────────────
        if _is_vowel(ch):
            result.append(VOWELS[ch])
            i += 1
            continue

        # ── Anusvara / Visarga standalone ─────────────────────────────────
        if ch in (ANUSVARA, VISARGA):
            result.append(MATRAS[ch])
            i += 1
            continue

        # ── Consonant ──────────────────────────────────────────────────────
        if _is_consonant(ch):
            base    = CONSONANTS[ch]
            next_ch = word[i + 1] if i + 1 < n else ""

            if next_ch == VIRAMA:
                # Explicit ottulu: no inherent vowel
                result.append(base)
                i += 2
            elif next_ch in (ANUSVARA, VISARGA):
                # Anusvara/Visarga after consonant: consonant + "a" + nasal/aspirate
                result.append(base + "a" + MATRAS[next_ch])
                i += 2
            elif _is_matra(next_ch):
                vowel_sound = MATRAS[next_ch]
                peek = word[i + 2] if i + 2 < n else ""
                if peek in (ANUSVARA, VISARGA):
                    result.append(base + vowel_sound + MATRAS[peek])
                    i += 3
                else:
                    result.append(base + vowel_sound)
                    i += 2
            else:
                # Inherent 'a'
                result.append(base + "a")
                i += 1
            continue

        # ── Pass-through ──────────────────────────────────────────────────
        result.append(ch)
        i += 1

    return "".join(result)


def telugu_to_tenglish(text: str) -> str:
    text = normalize_telugu(text)
    parts = []
    for tok in text.split(" "):
        pre, core, post = _split_telugu_token(tok)
        parts.append(pre + (telugu_to_tenglish_word(core) if core else "") + post)
    result = " ".join(parts)
    return (result[0].upper() + result[1:]) if result else result


def _split_telugu_token(tok: str):
    i = 0
    while i < len(tok) and not is_telugu(tok[i]):
        i += 1
    j = len(tok)
    while j > i and not is_telugu(tok[j - 1]):
        j -= 1
    return tok[:i], tok[i:j], tok[j:]


# ─────────────────────────────────────────────
# LAYER 2B — Tenglish → Telugu  (Dict + Parser)
# ─────────────────────────────────────────────

TENGLISH_WORD_DICT: dict[str, str] = {
    # Pronouns
    "nenu":      "నేను",   "na":        "నా",
    "nuvvu":     "నువ్వు", "nee":       "నీ",     "meeru":     "మీరు",
    "mee":       "మీ",     "atanu":     "అతను",   "aame":      "ఆమె",
    "vaallu":    "వాళ్ళు", "vaaru":     "వారు",   "memu":      "మేము",
    "mana":      "మన",     "manamu":    "మనము",   "vallu":     "వాళ్ళు",
    "okadu":     "ఒకడు",   "okate":     "ఒకతె",
    # Possessives
    "naa":       "నా",     "tanadi":    "తనది",
    "tana":      "తన",     "vaari":     "వారి",
    # Question words
    "em":        "ఏం",     "emiti":     "ఏమిటి",  "ekkada":    "ఎక్కడ",
    "enduku":    "ఎందుకు", "ela":       "ఎలా",    "epudu":     "ఎప్పుడు",
    "evaru":     "ఎవరు",   "enti":      "ఏంటి",
    # Verbs (common)
    "vachhaanu": "వచ్చాను","vachhadu":  "వచ్చాడు","vachhindi": "వచ్చింది",
    "vellanu":   "వెళ్ళాను","velladu":  "వెళ్ళాడు","vellindi": "వెళ్ళింది",
    "chesanu":   "చేశాను", "chesadu":   "చేశాడు",  "chesindi":  "చేసింది",
    "cheyyi":    "చెయ్యి", "cheyyandi": "చెయ్యండి",
    "vini":      "విని",   "vinanu":    "విన్నాను","vinadu":    "విన్నాడు",
    "chudu":     "చూడు",   "chusanu":   "చూశాను",  "chusadu":   "చూశాడు",
    "telusu":    "తెలుసు", "telidhu":   "తెలీదు",  "telu":      "తెలుసు",
    "undhi":     "ఉంది",   "unnanu":    "ఉన్నాను", "unnadu":    "ఉన్నాడు",
    "ledu":      "లేదు",   "ledhu":     "లేదు",
    "vastaanu":  "వస్తాను","vastadu":   "వస్తాడు",
    "pothanu":   "పోతాను", "pothadu":   "పోతాడు",
    "cheppanu":  "చెప్పాను","cheppadu": "చెప్పాడు","cheppu":    "చెప్పు",
    "artham":    "అర్థం",  "arthamai":  "అర్థమై",
    "iddham":    "ఇద్దాం", "podam":     "పోదాం",
    # Affirmations / negations
    "avunu":     "అవును",  "kadu":      "కాదు",   "kaadu":     "కాదు",
    "ille":      "ఇల్లే",  "ante":      "అంటే",   "aithe":     "అయితే",
    # Greetings / social
    "namaskaram": "నమస్కారం","dhanyavaadalu":"ధన్యవాదాలు",
    "bagunnara":  "బాగున్నారా","bagunnav": "బాగున్నావ్",
    "baagu":      "బాగు",   "bagundi":   "బాగుంది",
    # Time words
    "ippudu":    "ఇప్పుడు","appudu":    "అప్పుడు","mundhu":    "ముందు",
    "taruvata":  "తర్వాత", "intha":     "ఇంత",    "evroju":    "ఏ రోజు",
    "ninna":     "నిన్న",  "reyyi":     "రేపు",    "roju":      "రోజు",
    "rojula":    "రోజులా",
    # ── Song / film vocabulary ───────────────────────────────────────────
    # Love & emotion
    "prema":     "ప్రేమ",  "premanu":   "ప్రేమను","premalo":   "ప్రేమలో",
    "ishttam":   "ఇష్టం",  "ishtam":    "ఇష్టం",  "ishtapadu": "ఇష్టపడు",
    "manassu":   "మనసు",   "manassulo": "మనసులో", "manase":    "మనసే",
    "hrudayam":  "హృదయం",  "jeevitam":  "జీవితం", "aatma":     "ఆత్మ",
    "swapnam":   "స్వప్నం","swapnalu":  "స్వప్నాలు",
    "vedana":    "వేదన",   "anandham":  "ఆనందం",  "santosham": "సంతోషం",
    "dukham":    "దుఃఖం",  "aanandam":  "ఆనందం",
    "virahamu":  "విరహము", "kalalu":    "కలలు",   "kala":      "కల",
    "asha":      "ఆశ",     "nammakam":  "నమ్మకం",
    "badha":     "బాధ",    "baadha":    "బాధ",
    # Nature
    "akasam":    "ఆకాశం",  "meghalu":   "మేఘాలు", "megham":    "మేఘం",
    "vaana":     "వాన",    "vaanalu":   "వానలు",
    "surya":     "సూర్య",  "chandra":   "చంద్ర",  "chandrudu": "చంద్రుడు",
    "nallu":     "నల్లు",  "puvvu":     "పువ్వు",  "puvvulu":   "పువ్వులు",
    "kadali":    "కడలి",   "samudram":  "సముద్రం", "nadi":      "నది",
    "rojaalu":   "రోజాలు", "roja":      "రోజా",
    # Common film song words
    "priya":     "ప్రియ",  "priyuda":   "ప్రియుడా","priyatama": "ప్రియతమా",
    "chinuku":   "చినుకు", "oka":       "ఒక",      "jivamu":    "జీవము",
    "gundello":  "గుండెల్లో","gunde":   "గుండె",  "gundelo":   "గుండెలో",
    "kaanuka":   "కానుక",  "kshnam":    "క్షణం",  "kshanam":   "క్షణం",
    "toli":      "తొలి",   "toli prema":"తొలి ప్రేమ",
    "oohalu":    "ఊహలు",   "oohallo":   "ఊహల్లో", "ooha":      "ఊహ",
    "nuvvante":  "నువ్వంటే","nuvvuna":  "నువ్వున",
    "ninne":     "నిన్నే",  "ninnu":     "నిన్ను",  "ninnunu":   "నిన్నున్",
    "neekai":    "నీకై",    "niku":      "నీకు",   "neeku":     "నీకు",
    "naaku":     "నాకు",   "naatho":    "నాతో",   "neeto":     "నీతో",
    "vuntivi":   "వుంటివి","undali":    "ఉండాలి", "undipoyi":  "ఉండిపోయి",
    "thirigaa":  "తిరిగా", "padaali":   "పడాలి",  "choodu":    "చూడు",
    "choodandi": "చూడండి",
    # Adjectives
    "andamaina":  "అందమైన", "andame":   "అందమే",  "andam":     "అందం",
    "manchidi":   "మంచిది", "manchiga": "మంచిగా", "manchi":    "మంచి",
    "chala":      "చాలా",   "chaala":   "చాలా",
    "adbudam":    "అద్భుతం","pellaina": "పెళ్ళైన",
    # Common song endings
    "le":        "లే",     "lo":        "లో",     "tho":       "తో",
    "ki":        "కి",     "ni":        "ని",     "nu":        "ను",
    "loo":       "లో",     "loki":      "లోకి",   "nundi":     "నుండి",
    "gurinchi":  "గురించి","valla":     "వల్ల",    "kosam":     "కోసం",
    "lona":      "లోన",    "lone":      "లోనే",
    # Music filler syllables
    "aa":        "ఆ",      "ee":        "ఈ",     "oo":        "ఓ",
    "naa naa":   "నా నా",  "la la":     "లా లా",
    # Common verbs in song context
    "pade":      "పాడే",   "paaduta":   "పాడుతా", "paadu":     "పాడు",
    "nintu":     "నిలబడు", "ilaa":      "ఇలా",
    "prati":     "ప్రతి",  "maraa":     "మరా",    "marachipovu":"మరచిపోవు",
    "marachipovanu":"మరచిపోవాను",
    "niluvumu":  "నిలువుము","viduvanu":  "విదువను",
    "pokunda":   "పోకుండా","undipoku":  "ఉండిపోకు",
    "vachheyyaa":"వచ్చేయా",

    # ── Pronoun case forms ────────────────────────────────────────────────────
    "vaadiki":     "వాడికి",      "aamediki":    "ఆమెకి",       "vaallaki":    "వాళ్ళకి",
    "maaku":       "మాకు",        "meeku":       "మీకు",         "manaki":      "మనకి",
    "nannu":       "నన్ను",       "vaadini":     "వాడిని",       "aamedini":    "ఆమెను",
    "vaallani":    "వాళ్ళని",     "neetho":      "నీతో",         "vaadito":     "వాడితో",
    "vaallato":    "వాళ్ళతో",     "manato":      "మనతో",         "meeto":       "మీతో",
    "naavalla":    "నావల్ల",      "neevalla":    "నీవల్ల",

    # ── Demonstratives / locatives ────────────────────────────────────────────
    "ikkada":      "ఇక్కడ",       "akkada":      "అక్కడ",        "ikkade":      "ఇక్కడే",
    "akkade":      "అక్కడే",      "idhi":        "ఇది",          "adhi":        "అది",
    "edhi":        "ఏది",         "ivvi":        "ఇవి",          "avvi":        "అవి",
    "edhaina":     "ఏదైనా",       "evaraina":    "ఎవరైనా",       "ivvaala":     "ఇవాళ",
    "monnana":     "మొన్న",       "konni":       "కొన్ని",       "anni":        "అన్ని",

    # ── Present continuous ────────────────────────────────────────────────────
    "chestunnaanu":  "చేస్తున్నాను",   "chestunnaadu":  "చేస్తున్నాడు",
    "chestundi":     "చేస్తుంది",      "chestunnaaru":  "చేస్తున్నారు",
    "velthunnaanu":  "వెళ్తున్నాను",   "velthunnaadu":  "వెళ్తున్నాడు",
    "velthundi":     "వెళ్తుంది",      "velthunnaaru":  "వెళ్తున్నారు",
    "vastunnaanu":   "వస్తున్నాను",    "vastunnaadu":   "వస్తున్నాడు",
    "vastundi":      "వస్తుంది",       "vastunnaaru":   "వస్తున్నారు",
    "chepthunnaanu": "చెప్తున్నాను",   "chepthunnaadu": "చెప్తున్నాడు",
    "vintunnaanu":   "వింటున్నాను",    "vintunnaadu":   "వింటున్నాడు",
    "chustunnaanu":  "చూస్తున్నాను",   "chustunnaadu":  "చూస్తున్నాడు",
    "chustundi":     "చూస్తుంది",      "chustunnaaru":  "చూస్తున్నారు",
    "tintunnaanu":   "తింటున్నాను",    "tintunnaadu":   "తింటున్నాడు",
    "taagutunnaanu": "తాగుతున్నాను",   "taagutunnaadu": "తాగుతున్నాడు",
    "padukuntunnaanu":"పడుకుంటున్నాను","padukuntunnaadu":"పడుకుంటున్నాడు",
    "raastunnaanu":  "రాస్తున్నాను",   "nadustunnaanu": "నడుస్తున్నాను",
    "paadutunnaanu": "పాడుతున్నాను",   "aadutunnaanu":  "ఆడుతున్నాను",
    "nerchukuntunnaanu":"నేర్చుకుంటున్నాను",

    # ── Future tense ──────────────────────────────────────────────────────────
    "chestanu":    "చేస్తాను",    "chestadu":    "చేస్తాడు",    "chestaaru":   "చేస్తారు",
    "chestaam":    "చేస్తాం",
    "veltanu":     "వెళ్తాను",    "veltadu":     "వెళ్తాడు",    "veltaaru":    "వెళ్తారు",
    "veltaam":     "వెళ్తాం",
    "vastanu":     "వస్తాను",     "vastadu":     "వస్తాడు",     "vastaaru":    "వస్తారు",
    "vastaam":     "వస్తాం",
    "cheptanu":    "చెప్తాను",    "cheptadu":    "చెప్తాడు",
    "vintanu":     "వింటాను",     "vintadu":     "వింటాడు",
    "chustanu":    "చూస్తాను",    "chustadu":    "చూస్తాడు",
    "tintanu":     "తింటాను",     "tintadu":     "తింటాడు",
    "taagutanu":   "తాగుతాను",    "taagutadu":   "తాగుతాడు",
    "padukuntanu": "పడుకుంటాను",  "padukuntadu": "పడుకుంటాడు",
    "raastanu":    "రాస్తాను",    "paadutanu":   "పాడుతాను",
    "istanu":      "ఇస్తాను",     "istadu":      "ఇస్తాడు",
    "teestanu":    "తీస్తాను",    "teestadu":    "తీస్తాడు",
    "aadutanu":    "ఆడుతాను",     "nadustanu":   "నడుస్తాను",

    # ── Negations ────────────────────────────────────────────────────────────
    "cheyyadhu":   "చేయలేదు",    "vellaledhu":  "వెళ్ళలేదు",   "raledhu":     "రాలేదు",
    "cheyyanu":    "చేయను",       "vellaanu":    "వెళ్ళను",     "raanu":       "రాను",
    "ivvanu":      "ఇవ్వను",      "vaddhu":      "వద్దు",        "vaddhe":      "వద్దే",
    "paravaledu":  "పర్వాలేదు",   "paravale":    "పర్వాలే",     "leru":        "లేరు",
    "ledha":       "లేదా",        "kakunda":     "కాకుండా",     "lekuntha":    "లేకుంటే",
    "aipoindhi":   "అయిపోయింది",  "ayyindhi":    "అయ్యింది",    "kooda":       "కూడా",
    "kuda":        "కూడా",

    # ── Verb forms (oblique / infinitive) ─────────────────────────────────────
    "cheyyaali":   "చేయాలి",     "vellali":     "వెళ్ళాలి",    "raavaali":    "రావాలి",
    "choodali":    "చూడాలి",     "vinali":      "వినాలి",       "cheppali":    "చెప్పాలి",
    "isthe":       "ఇస్తే",       "cheste":      "చేస్తే",       "velte":       "వెళ్తే",
    "vaste":       "వస్తే",       "chuste":      "చూస్తే",       "vinte":       "వింటే",
    "tinu":        "తిను",        "tinandi":     "తినండి",       "taagu":       "తాగు",
    "taagandi":    "తాగండి",      "paduko":      "పడుకో",
    "tinnanu":     "తిన్నాను",    "tinnadu":     "తిన్నాడు",    "tinnindi":    "తిన్నింది",
    "taagaanu":    "తాగాను",      "taagaadu":    "తాగాడు",       "taagindi":    "తాగింది",
    "raasaanu":    "రాశాను",      "raasaadu":    "రాశాడు",       "raasindi":    "రాసింది",
    "nadichanu":   "నడిచాను",     "nadichadu":   "నడిచాడు",      "nadichindi":  "నడిచింది",
    "paadaanu":    "పాడాను",      "paadaadu":    "పాడాడు",       "paadindi":    "పాడింది",
    "aadaanu":     "ఆడాను",       "aadaadu":     "ఆడాడు",        "aadindi":     "ఆడింది",

    # ── Family / relationships ─────────────────────────────────────────────────
    "amma":        "అమ్మ",        "nanna":       "నాన్న",        "anna":        "అన్న",
    "akka":        "అక్క",        "thammudu":    "తమ్ముడు",      "chelli":      "చెల్లి",
    "attha":       "అత్త",        "mamayya":     "మామయ్య",       "pinni":       "పిన్ని",
    "babai":       "బాబాయ్",      "tata":        "తాత",           "ammamma":     "అమ్మమ్మ",
    "thathayya":   "తాతయ్య",      "naanamma":    "నానమ్మ",
    "pillalu":     "పిల్లలు",     "pillaadu":    "పిల్లాడు",    "pillamaa":    "పిల్లమ్మ",
    "kodalu":      "కోడలు",       "alludu":      "అల్లుడు",      "maradalu":    "మరదలు",
    "baava":       "బావ",          "vadina":      "వదిన",
    "bandhuvulu":  "బంధువులు",    "mitrulu":     "మిత్రులు",
    "snehithudu":  "స్నేహితుడు",  "snehithi":    "స్నేహితి",     "sneham":      "స్నేహం",
    "premiku":     "ప్రేమికుడు",  "premika":     "ప్రేమికురాలు",

    # ── Food / drink ──────────────────────────────────────────────────────────
    "annam":       "అన్నం",       "koora":       "కూర",           "chaaru":      "చారు",
    "pappu":       "పప్పు",       "avakaya":     "అవకాయ",         "neyyi":       "నెయ్యి",
    "palu":        "పాలు",        "neellu":      "నీళ్ళు",        "chaaya":      "చాయ్",
    "idli":        "ఇడ్లీ",       "dosa":        "దోశ",            "pesarattu":   "పెసరట్టు",
    "upma":        "ఉప్మా",        "poori":       "పూరీ",           "vada":        "వడ",
    "biryani":     "బిర్యాని",    "pulihora":    "పులిహోర",       "muddapappu":  "ముద్దపప్పు",
    "kobbari":     "కొబ్బరి",     "rotti":       "రొట్టె",        "mirchi":      "మిర్చి",
    "allam":       "అల్లం",        "kandi":       "కందిపప్పు",     "jilebi":      "జిలేబీ",
    "laddu":       "లడ్డూ",       "halwa":       "హల్వా",          "gulab jamun": "గులాబ్ జామున్",

    # ── Body parts ────────────────────────────────────────────────────────────
    "tala":        "తల",           "mukham":      "ముఖం",          "kannu":       "కన్ను",
    "kannulu":     "కన్నులు",      "chevi":       "చెవి",           "mukkhu":      "ముక్కు",
    "naalaaka":    "నాలుక",        "kaalu":       "కాలు",           "kaallu":      "కాళ్ళు",
    "cheyyilu":    "చేతులు",       "velu":        "వేలు",           "vellu":       "వేళ్ళు",
    "oopiri":      "ఊపిరి",        "shariram":    "శరీరం",

    # ── Places / transport ────────────────────────────────────────────────────
    "illu":        "ఇల్లు",        "intlo":       "ఇంట్లో",        "veedhi":      "వీధి",
    "ooru":        "ఊరు",          "pattanam":    "పట్టణం",
    "cinema":      "సినిమా",        "school":      "స్కూల్",        "college":     "కాలేజ్",
    "office":      "ఆఫీస్",        "hospital":    "హాస్పటల్",      "market":      "మార్కెట్",
    "dukaanam":    "దుకాణం",        "gudi":        "గుడి",           "railu":       "రైలు",
    "basu":        "బస్సు",         "auto":        "ఆటో",

    # ── Adjectives / adverbs ──────────────────────────────────────────────────
    "pedda":       "పెద్ద",        "chinna":      "చిన్న",         "kotta":       "కొత్త",
    "paatha":      "పాత",           "tella":       "తెల్ల",         "nalla":       "నల్ల",
    "erru":        "ఎర్ర",          "pacha":       "పచ్చ",          "pachi":       "పచ్చి",
    "theeyani":    "తీయని",         "kaaramaina":  "కారమైన",        "manchiga":    "మంచిగా",
    "santhoshamga":"సంతోషంగా",      "dooramga":    "దూరంగా",        "daggara":     "దగ్గర",
    "konchemu":    "కొంచెం",        "malli":       "మళ్ళీ",          "inka":        "ఇంక",
    "inkaa":       "ఇంకా",          "sarele":      "సరేలే",          "sare":        "సరే",
    "okke":        "ఒక్కే",         "anthe":       "అంతే",           "ayyo":        "అయ్యో",
    "kadha":       "కదా",            "sari":        "సరి",

    # ── Common expressions ────────────────────────────────────────────────────
    "yemandi":     "ఏమండి",        "yemantivi":   "ఏమంటివి",       "eemiti":      "ఏమిటి",
    "sontham":     "సొంతం",         "sonthamga":   "సొంతంగా",       "malli malli": "మళ్ళీ మళ్ళీ",
    "kadha":       "కదా",            "chestha":     "చేస్తా",         "veluthaanu":  "వెళ్తాను",

    # ── Song / Tollywood (expanded) ───────────────────────────────────────────
    "vennela":     "వెన్నెల",       "vennelalo":   "వెన్నెలలో",     "vennelai":    "వెన్నెలై",
    "veena":       "వీణ",           "ragam":       "రాగం",           "taalam":      "తాళం",
    "gaanam":      "గానం",           "nadam":       "నాదం",           "sangeetham":  "సంగీతం",
    "kavitha":     "కవిత",           "pallavi":     "పల్లవి",         "charanam":    "చరణం",
    "paataalu":    "పాటలు",          "paata":       "పాట",
    "nidra":       "నిద్ర",          "kalalo":      "కలలో",
    "choostu":     "చూస్తూ",         "vintu":       "వింటూ",          "aaduta":      "ఆడుతూ",
    "praana":      "ప్రాణ",          "praanam":     "ప్రాణం",         "praanama":    "ప్రాణమా",
    "praanale":    "ప్రాణాలే",       "jeevanam":    "జీవనం",
    "cheliya":     "చెలియ",          "sakhiya":     "సఖియ",
    "raadha":      "రాధ",            "krishna":     "కృష్ణ",          "geetha":      "గీత",
    "thulasi":     "తులసి",          "akasanlo":    "ఆకాశంలో",
    "varsha":      "వర్ష",           "varsham":     "వర్షం",
    "nidhi":       "నిధి",           "nidhigaa":    "నిధిగా",
    "poovulaai":   "పువ్వులై",       "rojaale":     "రోజాలే",
    "virahave":    "విరహవే",          "badhaga":     "బాధగా",          "badhapadu":   "బాధపడు",
    "ashalu":      "ఆశలు",           "manasaa":     "మనసా",
    "gunde ninda": "గుండె నిండ",
    "oka prema":   "ఒక ప్రేమ",       "naa prema":   "నా ప్రేమ",       "nee prema":   "నీ ప్రేమ",
    "ishtamga":    "ఇష్టంగా",        "ishtapaddanu":"ఇష్టపడ్డాను",
    "nuvvu leka":  "నువ్వు లేక",     "nee valla":   "నీ వల్ల",        "naa valla":   "నా వల్ల",
    "nee gurinchi":"నీ గురించి",      "naa gurinchi":"నా గురించి",
    "navvu":       "నవ్వు",          "navvulu":     "నవ్వులు",        "navvaga":     "నవ్వగా",
    "madhuram":    "మధురం",           "madhuranga":  "మధురంగా",
    "alaa":        "అలా",            "chithram":    "చిత్రం",
    "hero":        "హీరో",            "heroine":     "హీరోయిన్",
    "nuvvu nenu":  "నువ్వు నేను",    "nee kannulu": "నీ కన్నులు",
    "nee navvu":   "నీ నవ్వు",       "priya sakhi": "ప్రియ సఖి",
    "toli saari":  "తొలి సారి",      "chivari saari":"చివరి సారి",
    "nee sontham": "నీ సొంతం",       "naa sontham": "నా సొంతం",
    "nee thodu":   "నీ తోడు",        "naa thodu":   "నా తోడు",
    "oopiri lo":   "ఊపిరిలో",        "gunde lo":    "గుండెలో",
    "sangeethamlo":"సంగీతంలో",        "raagamlo":    "రాగంలో",

    # ── Postpositions / case markers ──────────────────────────────────────────
    "meeda":       "మీద",            "kinda":       "కింద",           "venaka":      "వెనక",
    "paiku":       "పైకి",           "varaku":      "వరకు",           "daaka":       "దాకా",
    "saari":       "సారి",           "mundhu nundi":"ముందు నుండి",

    # ── Numbers ───────────────────────────────────────────────────────────────
    "okati":       "ఒకటి",          "rendu":       "రెండు",          "mooḍu":       "మూడు",
    "nalugu":      "నాలుగు",        "aidu":        "ఐదు",            "aaru":        "ఆరు",
    "edu":         "ఏడు",            "enimidi":     "ఎనిమిది",        "tommidi":     "తొమ్మిది",
    "padi":        "పది",            "nooru":       "నూరు",           "veyyi":       "వేయి",

    # ── Colors ────────────────────────────────────────────────────────────────
    "yerru":       "ఎర్రు",          "pacha ranga": "పచ్చ రంగు",      "neeli":       "నీలి",
    "tella":       "తెల్ల",          "nalla":       "నల్ల",            "peetha":      "పీత",
    "gola ranga":  "గోల రంగు",       "jamoon":      "జమ్ముని",        "vanadaa":     "వనడా",

    # ── Time (extended) ───────────────────────────────────────────────────────
    "thella varama":"తెల్లవారం",     "madhyaanam":  "మధ్యాహ్నం",      "saayanthram": "సాయంత్రం",
    "raatri":      "రాత్రి",         "ardhaaraatri":"అర్ధరాత్రి",     "veluthundi":  "వెలుతుంది",
    "nela":        "నెల",             "samvatsaram": "సంవత్సరం",       "vaaram":      "వారం",
    "nimisham":    "నిమిషం",          "ganta":       "గంట",             "gantalu":     "గంటలు",

    # ── Common daily life words ────────────────────────────────────────────────
    "neellu":      "నీళ్ళు",         "tindi":       "తిండి",           "nidra":       "నిద్ర",
    "vellu":       "వెళ్ళు",          "raa":         "రా",               "choodandi":   "చూడండి",
    "maatlaadu":   "మాట్లాడు",        "aagu":        "ఆగు",             "padu":        "పడు",
    "levu":        "లేవు",            "paadu":       "పాడు",             "vinu":        "వినూ",
    "nadipinchu":  "నడిపించు",        "nerpu":       "నేర్పు",           "nerchuko":    "నేర్చుకో",
    "cheppindi":   "చెప్పింది",       "arthamaiindi":"అర్థమైంది",       "sari petthu": "సరి పెట్టు",
    "manchidi":    "మంచిది",          "cheddudi":    "చెడ్డది",          "nija":        "నిజ",
    "nijam":       "నిజం",            "abbaddham":   "అబద్ధం",

    # ── Song vocabulary (extra Tollywood) ─────────────────────────────────────
    "jabilamma":   "జాబిలమ్మ",       "jabili":      "జాబిలి",          "munjuru":     "మున్జురు",
    "mugdhama":    "ముగ్ధమా",        "nijamaa":     "నిజమా",            "kanugontini":  "కనుగొంటిని",
    "aakasamlo":   "ఆకాశంలో",        "nakshatram":  "నక్షత్రం",        "nakshatralu":  "నక్షత్రాలు",
    "paavurama":   "పావురమా",         "kokila":      "కోకిల",            "vaayuvu":     "వాయువు",
    "aanattu":     "ఆనట్టు",          "yedutaa":     "ఎదుటా",            "dooramloo":   "దూరంలో",
    "kaavaali":    "కావాలి",          "kaadhu":      "కాదు",             "andham":      "అందం",
    "sundaramga":  "సుందరంగా",        "priyathamaa": "ప్రియతమా",
    "nee kaosam":  "నీ కోసం",        "naa kaosam":  "నా కోసం",
    "chivari varaku":"చివరి వరకు",   "manasantha":  "మనసంతా",
    "nindu":       "నిండు",           "ninduga":     "నిండుగా",
}

# Tenglish consonant sequences → Telugu (LONGEST FIRST)
TENGLISH_CONSONANTS: list[tuple[str, str]] = [
    # 4-char
    ("chch", "చ్చ"),
    # 3-char
    ("ksh", "క్ష"),  ("shr", "శ్ర"),  ("chh", "ఛ"),
    ("thh", "థ"),    ("nth", "న్థ"),   ("ndh", "న్ధ"),
    ("gna", "జ్ఞ"),
    # 2-char (longest first within 2-char)
    ("ng",  "ఙ"),    ("nj",  "ఞ"),    ("ch",  "చ"),
    ("kh",  "ఖ"),    ("gh",  "ఘ"),    ("jh",  "ఝ"),
    ("th",  "త"),    ("dh",  "ధ"),    ("ph",  "ఫ"),
    ("bh",  "భ"),    ("sh",  "శ"),
    ("ll",  "ళ్ళ"),  ("nn",  "న్న"),  ("kk",  "క్క"),
    ("pp",  "ప్ప"),  ("tt",  "ట్ట"),  ("dd",  "డ్డ"),
    ("mm",  "మ్మ"),  ("rr",  "ర్ర"),  ("gg",  "గ్గ"),
    ("jj",  "జ్జ"),  ("bb",  "బ్బ"),  ("ss",  "స్స"),
    ("vv",  "వ్వ"),
    # 1-char
    ("k",   "క"),    ("g",   "గ"),    ("c",   "చ"),
    ("j",   "జ"),    ("t",   "ట"),    ("d",   "డ"),
    ("n",   "న"),    ("p",   "ప"),    ("f",   "ఫ"),
    ("b",   "బ"),    ("m",   "మ"),    ("y",   "య"),
    ("r",   "ర"),    ("l",   "ల"),    ("v",   "వ"),
    ("w",   "వ"),    ("s",   "స"),    ("h",   "హ"),
    ("z",   "జ"),    ("x",   "క్స"),
]

VOWEL_TO_MATRA: dict[str, str] = {
    "aa": "\u0C3E",  # ా
    "ee": "\u0C40",  # ీ
    "ii": "\u0C40",  # ీ
    "oo": "\u0C42",  # ూ
    "uu": "\u0C42",  # ూ
    "ru": "\u0C43",  # ృ
    "ae": "\u0C47",  # ే
    "ai": "\u0C48",  # ై
    "au": "\u0C4C",  # ౌ
    "i":  "\u0C3F",  # ి
    "u":  "\u0C41",  # ు
    "e":  "\u0C46",  # ె
    "o":  "\u0C4A",  # ొ
    "a":  "",        # inherent
}

VOWEL_TO_INDEPENDENT: dict[str, str] = {
    "aa": "ఆ",  "ee": "ఈ",  "ii": "ఈ",
    "oo": "ఊ",  "uu": "ఊ",  "ru": "ఋ",
    "ae": "ఏ",  "ai": "ఐ",  "au": "ఔ",
    "a":  "అ",  "i":  "ఇ",  "u":  "ఉ",
    "e":  "ఎ",  "o":  "ఒ",
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
    for roman, tel in TENGLISH_CONSONANTS:
        if lower[pos:pos + len(roman)] == roman:
            return tel, pos + len(roman)
    return "", pos


def tenglish_to_telugu_word(word: str) -> str:
    lookup = TENGLISH_WORD_DICT.get(word.lower())
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

        tel_cons, next_i = _match_consonant(word, i)
        if tel_cons:
            i = next_i
            vowel_str, next_i2 = _match_vowel(word, i)

            if vowel_str == "a":
                result.append(tel_cons)
                i = next_i2
            elif vowel_str:
                matra = VOWEL_TO_MATRA[vowel_str]
                result.append(tel_cons + matra)
                i = next_i2
            else:
                result.append(tel_cons + VIRAMA)
            continue

        vowel_str, next_i = _match_vowel(word, i)
        if vowel_str:
            result.append(VOWEL_TO_INDEPENDENT.get(vowel_str, word[i]))
            i = next_i
            continue

        result.append(word[i])
        i += 1

    return "".join(result)


def tenglish_to_telugu(text: str) -> str:
    text = normalize_telugu(text)
    parts = []
    for tok in text.split(" "):
        pre, core, post = _split_latin_token(tok)
        parts.append(pre + (tenglish_to_telugu_word(core) if core else "") + post)
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
# LAYER 3 — ByT5 Neural Correction
# ─────────────────────────────────────────────

class TeluguTransliterationEngine:
    """
    Hybrid Telugu ↔ Tenglish engine.
    Rule-based layer always runs first; ByT5 corrects edge cases
    when a checkpoint is available.
    """

    MODEL_ID = "google/byt5-small"

    def __init__(self, use_neural: bool = True, device: str = "cpu"):
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
            logger.info("Loading ByT5 for Telugu: %s", self.MODEL_ID)
            self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_ID)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                self.MODEL_ID, torch_dtype=torch.float32
            ).to(self.device)
            self.model.eval()
            self._ready = True
            logger.info("ByT5 Telugu model loaded.")
        except Exception as e:
            logger.warning("ByT5 load failed (%s). Rules only.", e)

    @lru_cache(maxsize=2048)
    def _rule_cached(self, text: str, direction: str) -> str:
        if direction == "telugu_to_tenglish":
            return telugu_to_tenglish(text)
        return tenglish_to_telugu(text)

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
        decoded = self.tokenizer.decode(out[0], skip_special_tokens=True).strip()
        # Only accept output if it doesn't look like garbage
        if decoded and len(decoded) > 1:
            return decoded
        return rule_hint

    def convert(self, text: str) -> dict:
        t0        = time.perf_counter()
        text      = normalize_telugu(text)
        script    = detect_script(text)
        direction = "telugu_to_tenglish" if script == "telugu" else "tenglish_to_telugu"

        rule_output = self._rule_cached(text, direction)

        if self._ready and self.use_neural:
            try:
                output = self._byt5_correct(text, rule_output, direction)
                layer  = "byt5+rules"
            except Exception as e:
                logger.error("ByT5 error: %s", e)
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
    if direction == "telugu_to_tenglish":
        if text:
            text = text[0].upper() + text[1:]
    return text


# ─────────────────────────────────────────────
# Singleton
# ─────────────────────────────────────────────
_engine: Optional[TeluguTransliterationEngine] = None


def get_telugu_engine() -> TeluguTransliterationEngine:
    global _engine
    if _engine is None:
        _engine = TeluguTransliterationEngine(use_neural=True)
    return _engine
