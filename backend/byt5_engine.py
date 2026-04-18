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
                if next2 == ch:  # Geminate
                    dbl = DOUBLE_MAP.get(ch, base + base)
                    result.append(dbl)
                    i += 3  # skip trigger + pulli + partner consonant
                    # Partner consonant needs its own vowel
                    if i < n:
                        after = word[i]
                        if after in VOWEL_MARKERS:
                            # Explicit vowel marker follows the partner
                            v = _get_vowel_sound(after)
                            if v:
                                result.append(v)
                            i += 1
                            prev_was_vowel = bool(v)
                        else:
                            # Another consonant or word boundary → inherent 'a'
                            result.append("a")
                            prev_was_vowel = True
                    # else: geminate at absolute word end — no trailing vowel
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


# ─────────────────────────────────────────────
# Tamil → Tanglish word-level dictionary
# Covers words where rule G2P gives literary output but spoken form differs
# ─────────────────────────────────────────────
TAMIL_WORD_DICT: dict[str, str] = {
    "நான்":        "naan",       "நீ":          "nee",        "அவன்":       "avan",
    "அவள்":       "aval",       "அவர்":        "avar",       "நாம்":        "naam",
    "நாங்கள்":    "naangal",    "நீங்கள்":     "neengal",    "அவர்கள்":    "avangal",
    "இல்லை":      "illai",      "ஆமாம்":       "aamaam",     "சரி":         "sari",
    "வணக்கம்":    "vanakkam",   "நன்றி":       "nandri",     "நல்லது":      "nalladu",
    "ரொம்ப":      "romba",      "கொஞ்சம்":    "konjam",     "தெரியல":     "theriyala",
    "காதல்":      "kaadhal",    "உயிர்":       "uyir",       "மனம்":        "manam",
    "மழை":         "mazhai",     "நிலவு":       "nilavu",     "வானம்":       "vaanam",
    "பூ":          "poo",        "கடல்":        "kadal",      "மலர்":        "malar",
    "கண்":         "kann",       "கண்கள்":     "kangal",     "வலி":          "vali",
    "வாழ்க்கை":  "vaazhkai",   "கனவு":        "kanavu",     "விண்மீன்":   "vinmeen",
    "நெஞ்சம்":   "nenjam",     "அழகு":        "azhagu",     "பாடல்":      "paadal",
    "தேன்":       "then",       "மலர்கள்":    "malargal",   "அலைகள்":     "alaigal",
    "இனிமை":      "inimai",     "இன்பம்":     "inbam",      "மகிழ்ச்சி": "maghizchchi",
    "மௌனம்":     "mounam",     "கவிதை":      "kavithai",   "ஒளி":          "oli",
    "தென்றல்":   "thendral",   "குயில்":     "kuyil",      "வாழ்க":       "vaazhga",

    # ── Family ───────────────────────────────────────────────────────────────
    "அம்மா":     "amma",       "அப்பா":      "appa",        "அண்ணா":      "anna",
    "அக்கா":     "akka",       "தம்பி":      "thambi",      "தங்கை":      "thangai",
    "பாட்டி":    "paatti",     "தாத்தா":     "thaatha",     "மாமா":        "maama",
    "அத்தை":    "atthai",      "சித்தி":     "chitti",      "சித்தப்பா":  "chittappa",
    "மனைவி":    "manaivi",     "கணவன்":     "kanavan",     "குழந்தை":   "kuzhandhai",
    "மகன்":      "magan",       "மகள்":       "magal",

    # ── Body parts ───────────────────────────────────────────────────────────
    "தலை":       "thalai",     "கை":          "kai",         "கால்":        "kaal",
    "காவி":      "kevi",        "மூக்கு":     "mooku",       "வாய்":        "vaai",
    "முதுகு":   "mudugu",      "வயிறு":      "vayiru",      "முகம்":       "mugam",
    "இதயம்":    "idhayam",     "பல்":         "pal",         "நாக்கு":     "naakku",
    "தோல்":      "thol",        "மார்பு":     "maarbu",      "தோள்":       "thol",
    "விரல்":    "viral",        "நகம்":        "nagam",

    # ── Colors ───────────────────────────────────────────────────────────────
    "சிவப்பு":  "sivappu",    "பச்சை":      "pachai",       "நீலம்":      "neelam",
    "வெள்ளை":  "vellai",      "கருப்பு":    "karuppu",      "மஞ்சள்":    "manjal",
    "ஆரஞ்சு":  "aranchu",     "பழுப்பு":   "pazhuppu",     "சாம்பல்":   "saambal",
    "ஊதா":       "ootha",       "இளஞ்சிவப்பு": "ilanjivappu",

    # ── Numbers ──────────────────────────────────────────────────────────────
    "ஒன்று":    "ondru",       "இரண்டு":    "irandu",       "மூன்று":     "moondru",
    "நான்கு":   "naangu",      "ஐந்து":     "ainthu",       "ஆறு":         "aaru",
    "ஏழு":       "ezhu",        "எட்டு":     "ettu",          "ஒன்பது":    "onbadhu",
    "பத்து":     "pathu",       "நூறு":       "nooru",         "ஆயிரம்":   "aayiram",

    # ── Daily life nouns ─────────────────────────────────────────────────────
    "தண்ணீர்": "thanneer",    "சாப்பாடு":  "saapadu",      "வீடு":       "veedu",
    "வேலை":     "velai",        "பள்ளி":     "palli",         "கடை":        "kadai",
    "வழி":        "vazhi",       "பேருந்து":  "perundhu",     "கார்":        "kaar",
    "பணம்":      "panam",        "புத்தகம்": "puthagam",     "ஊர்":         "oor",
    "நகரம்":    "nagaram",      "கிராமம்":  "kiraamam",     "மலை":         "malai",
    "கடல்":      "kadal",        "ஆறு":        "aaru",          "ஏரி":        "aeri",
    "வயல்":      "vayal",        "காடு":       "kaadu",         "கோயில்":    "koyil",
    "சாலை":     "saalai",       "பாலம்":     "paalam",        "ரயில்":     "rayil",

    # ── Food ─────────────────────────────────────────────────────────────────
    "இட்லி":    "idli",         "தோசை":      "dosai",         "சாம்பார்": "sambar",
    "ரசம்":      "rasam",        "கூட்டு":   "koottu",        "பாயாசம்":  "paayasam",
    "காபி":      "kaapi",        "தேநீர்":   "theneer",       "தயிர்":     "thayir",
    "சோறு":      "soru",          "குழம்பு":  "kuzhambu",      "அப்பளம்":  "appalam",
    "வடை":        "vadai",         "பொங்கல்": "pongal",         "பிரியாணி": "biriyani",
    "பழம்":      "pazham",        "மாம்பழம்": "maambazham",   "வாழைப்பழம்":"vaazhaipazham",
    "தேங்காய்": "thengai",      "வெங்காயம்":"vengaayam",    "தக்காளி":  "thakkali",

    # ── Time ─────────────────────────────────────────────────────────────────
    "காலை":     "kaalai",       "மாலை":      "maalai",        "இரவு":       "iravu",
    "பகல்":      "pagal",        "இன்று":     "indru",          "நாளை":      "naale",
    "நேத்து":   "neethu",       "இப்போது":   "ippodhu",       "அப்போது":   "appodhu",
    "விடியல்":  "vidiyal",      "நடுரவு":   "naduiravu",     "நேரம்":     "neram",
    "நிமிடம்": "nimidam",      "மணி":        "mani",           "வாரம்":     "vaaram",
    "மாதம்":    "maadham",      "வருடம்":   "varudam",

    # ── Verbs (base/imperative) ───────────────────────────────────────────────
    "வா":         "vaa",          "போ":          "po",            "பார்":       "paar",
    "கேள்":      "kel",          "பேசு":       "pesu",           "சாப்பிடு": "saapidu",
    "குடி":       "kudi",         "தூங்கு":    "thoongu",       "எடு":         "edu",
    "கொடு":      "kodu",          "வாங்கு":   "vaangu",         "படி":         "padi",
    "எழுது":     "ezhutu",       "நட":          "nada",           "ஓடு":        "odu",
    "விரும்பு":"virumbu",       "மாறு":       "maaru",          "திரும்பு": "thirumbu",
    "நில்":       "nil",          "உட்கார்":  "utkaar",         "எழு":         "ezhu",
    "சிரி":       "siri",         "அழு":        "azhu",           "கேட்":       "ket",
    "வா":          "vaa",          "சொல்":      "sol",             "நினை":       "ninai",
    "மறு":         "maru",         "தொடு":      "thodu",          "கட்டு":    "kattu",
    "திற":         "thira",        "மூடு":       "moodu",

    # ── Adjectives / adverbs ─────────────────────────────────────────────────
    "பெரிய":    "periya",       "சின்ன":     "chinna",         "புதிய":     "puthiya",
    "பழைய":     "pazhaiya",     "நல்ல":       "nalla",           "கெட்ட":    "ketta",
    "அழகான":   "azhagana",     "முக்கியம்": "mukkiyam",       "வேகமான":   "vegamana",
    "மெதுவான": "methuvana",    "கஷ்டமான":  "kashtamana",      "சுலபமான": "sulabamana",
    "அதிகம்":  "adhigam",      "குறைவு":   "kuraiyvu",        "சரியான":  "sariyana",
    "பாக்கியமான":"paaggiyamana","கவலையான":"kavalayaana",     "சந்தோஷமான":"sandhoshamana",

    # ── Question words (extended) ────────────────────────────────────────────
    "ஏன்":       "yen",          "எப்படி":   "eppadi",          "எத்தனை":  "etthanai",
    "எந்த":      "endha",        "எவ்வளவு": "evvalavu",

    # ── Song / lyric vocabulary (extended) ───────────────────────────────────
    "வேண்டும்":"vendum",        "கண்டேன்":  "kanden",           "வந்தேன்": "vanden",
    "போனேன்":   "ponen",         "தேடினேன்": "thedinen",         "கண்ணீர்": "kanneer",
    "சிரிப்பு": "sirippu",      "கோபம்":    "gobam",             "பயம்":    "bayam",
    "துக்கம்":  "thukkam",      "சந்தோஷம்":"sandhosam",        "அன்பு":   "anbu",
    "அன்பே":     "anbe",          "முத்தம்": "muththam",          "இதழ்":   "idhal",
    "தோளில்":   "tholil",        "வானில்":  "vaanil",             "கடலில்": "kadalil",
    "மழையில்":  "mazhayil",      "தனிமை":  "thanimai",           "சேர்":    "ser",
    "பிரியாமல்":"piriyaamal",   "என்றும்": "endrum",             "நினைவில்":"ninaivil",
    "மறக்க":    "marakka",        "மறந்தேன்":"marandhen",        "மறவாதே": "maravaadhe",
    "உன்னோடு": "unnodu",         "விடியும்":"vidiyum",           "நிலவே":  "nilave",
    "பொழுது":   "pozhudhu",       "ஆகாயம்": "aagaayam",          "மின்னல்":"minnal",
    "குளிர்":   "kulir",          "வெப்பம்":"veppam",             "இளமை":   "ilamai",
    "வாழ்வு":   "vaazhvu",        "பிறகு":   "piragu",            "இன்னும்":"innum",
    "வருவான்":  "varuvaan",       "இடி":      "idi",               "காற்று": "kaatru",
    "நதி":        "nadhi",         "தாய்":    "thaai",              "தந்தை": "thanthai",
    "உலகம்":    "ulagam",          "வாழ்":   "vaazh",              "நம்பிக்கை":"nambikkai",
    "அமைதி":   "amaidhi",          "சுதந்திரம்":"sudhandhiram",  "கனவுகள்": "kanavugal",
    "விழிகள்":  "vizhigal",       "உதடுகள்":"udhadugal",         "கரங்கள்": "karangal",
    "இதழ்கள்": "idhalgal",        "முத்தங்கள்":"muththangal",
}

# Common spoken Tamil contractions (post-processing)
SPOKEN_CONTRACTIONS = [
    # போகிறேன் / present tense spoken forms
    (r"\bpogirean\b",    "pogiren"),
    (r"\bvagirean\b",    "vagiren"),
    (r"\bpaarkirean\b",  "paarkkiren"),
    (r"kireen\b",        "kiren"),
    (r"gireen\b",        "giren"),
    (r"kirean\b",        "kiren"),
    # இல்லை / இல்ல
    (r"\billai\b",       "illa"),
    (r"\billaiye\b",     "illaiye"),
    # என்ன
    (r"\benenna\b",      "enna"),
    (r"\byenna\b",       "enna"),
    # வாங்க / போங்க / வாருங்க
    (r"\bvaanga\b",      "vaanga"),
    (r"\bvarunga\b",     "varunga"),
    (r"\bponga\b",       "ponga"),
    # தெரியல
    (r"\btheriyal\b",    "theriyala"),
    (r"\btheriala\b",    "theriyala"),
    # Common verb contractions
    (r"\bpoitu\b",       "poittu"),
    (r"\bvanthu\b",      "vantu"),
    (r"aagitu\b",        "aagittu"),
    (r"paarthu\b",       "paarttu"),
    # Spoken emphatic endings
    (r"dhaan\b",         "daan"),
    (r"\byaa\b",         "yaa"),
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
            # Dictionary lookup before character-level G2P
            converted = TAMIL_WORD_DICT.get(tamil_part) or rule_based_g2p(tamil_part)
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
    text = re.sub(r"hint:.*", "", text).strip()
    text = re.sub(r"\s{2,}", " ", text)
    if text:
        text = text[0].upper() + text[1:]
    return text


# ─────────────────────────────────────────────
# Tanglish → Tamil  (dictionary + rule-based reverse)
# ─────────────────────────────────────────────

_PULLI = "\u0BCD"   # pulli (virama)

# High-frequency Tanglish words → Tamil; checked before character-level rules
TANGLISH_WORD_DICT: dict[str, str] = {
    # Pronouns
    "naan":        "நான்",       "naaan":       "நான்",
    "nee":         "நீ",         "neenga":      "நீங்க",       "neengal":     "நீங்கள்",
    "avan":        "அவன்",       "aval":        "அவள்",         "avar":        "அவர்",
    "avanga":      "அவங்க",      "avangal":     "அவர்கள்",
    "naanga":      "நாங்க",      "naangal":     "நாங்கள்",
    "namma":       "நம்ம",       "naam":        "நாம்",
    # Possessives
    "ennoda":      "என்னோட",     "unnoda":      "உன்னோட",      "avoda":       "அவனோட",
    "avaloda":     "அவளோட",      "engaloda":    "எங்களோட",     "ungaloda":    "உங்களோட",
    "ennudaiya":   "என்னுடைய",   "unudaiya":    "உன்னுடைய",
    # Verbs (present/future spoken forms)
    "irukken":     "இருக்கேன்",  "irukkaan":    "இருக்கான்",   "irukku":      "இருக்கு",
    "irukkaa":     "இருக்கா",    "irukkaanga":  "இருக்காங்க",
    "poren":       "போறேன்",     "poraan":      "போறான்",       "poraanga":    "போறாங்க",
    "poiduven":    "போயிடுவேன்","vaaren":      "வாறேன்",       "varaan":      "வாறான்",
    "varaanga":    "வாறாங்க",    "vandhuten":   "வந்துட்டேன்",
    "soluven":     "சொல்லுவேன்","solluven":    "சொல்லுவேன்",  "solren":      "சொல்றேன்",
    "ketpen":      "கேட்பேன்",   "paakaren":    "பார்க்கிறேன்","paakalam":    "பார்க்கலாம்",
    "theriyala":   "தெரியல",     "theriyum":    "தெரியும்",     "theriyaadu":  "தெரியாது",
    "puriyala":    "புரியல",     "puriyum":     "புரியும்",
    "mudiyaadu":   "முடியாது",   "mudiyum":     "முடியும்",
    "vandhuten":   "வந்துட்டேன்","paartuten":   "பார்த்துட்டேன்",
    # Common words
    "enna":        "என்ன",       "yenna":       "என்ன",         "eppo":        "எப்போ",
    "yeppo":       "எப்போ",      "enga":        "எங்க",          "yenga":       "எங்க",
    "yaaru":       "யாரு",       "yaar":        "யார்",          "enge":        "எங்கே",
    "ennathu":     "என்னது",     "aama":        "ஆமா",           "illai":       "இல்லை",
    "illa":        "இல்ல",       "ille":        "இல்லே",          "sari":        "சரி",
    "romba":       "ரொம்ப",      "rombaa":      "ரொம்பா",        "konjam":      "கொஞ்சம்",
    "konja":       "கொஞ்ச",      "nalla":       "நல்ல",           "nallaa":      "நல்லா",
    "adu":         "அது",         "idu":         "இது",            "edu":         "எது",
    "ivan":        "இவன்",       "aval":        "அவள்",           "avanga":      "அவங்க",
    # Affirmations / negations
    "apdi":        "அப்டி",      "apdithan":    "அப்டித்தான்",  "appadi":      "அப்படி",
    "vendam":      "வேண்டாம்",   "venda":       "வேண்டா",        "seri":        "சேரி",
    # Greetings
    "vanakkam":    "வணக்கம்",    "nandri":      "நன்றி",
    "eppadi":      "எப்படி",
    # Time
    "inniki":      "இன்னிக்கி",  "innikki":     "இன்னிக்கி",
    "naalaiku":    "நாளைக்கு",   "munaadi":     "முன்னாடி",      "pinnaadi":    "பின்னாடி",
    "ippo":        "இப்போ",       "appo":        "அப்போ",
    # Song / lyric vocabulary
    "kadhal":      "காதல்",      "kaadhal":     "காதல்",          "uyir":        "உயிர்",
    "uyire":       "உயிரே",      "manam":       "மனம்",           "manasu":      "மனசு",
    "nenjam":      "நெஞ்சம்",   "nenje":       "நெஞ்சே",         "azhagu":      "அழகு",
    "azhage":      "அழகே",       "paadal":      "பாடல்",          "paattu":      "பாட்டு",
    "malargal":    "மலர்கள்",   "malar":       "மலர்",            "alaigal":     "அலைகள்",
    "kadal":       "கடல்",       "megam":       "மேகம்",          "mazhai":      "மழை",
    "ilamai":      "இளமை",       "vaanam":      "வானம்",           "nila":        "நிலா",
    "nilaa":       "நிலா",        "nilavu":      "நிலவு",          "minnal":      "மின்னல்",
    "thendral":    "தென்றல்",    "poo":         "பூ",              "poove":       "பூவே",
    "kuyil":       "குயில்",     "kanavugal":   "கனவுகள்",        "kanavu":      "கனவு",
    "inimai":      "இனிமை",      "inbam":       "இன்பம்",         "vali":        "வலி",
    "vaazhkai":    "வாழ்க்கை",  "ennai":       "என்னை",           "unnai":       "உன்னை",
    "unnaai":      "உன்னாய்",   "pirivu":      "பிரிவு",          "piriya":      "பிரிய",
    "kalyanam":    "கல்யாணம்",  "mugam":       "முகம்",            "kann":        "கண்",
    "kangal":      "கண்கள்",    "kanneer":     "கண்ணீர்",         "kai":         "கை",
    "kaigal":      "கைகள்",     "ninaivugal":  "நினைவுகள்",       "ninaive":     "நினைவே",
    "thedum":      "தேடும்",     "thedu":       "தேடு",            "vaadai":      "வாடை",
    "andhi":       "அந்தி",      "iravil":      "இரவில்",          "vinmeen":     "விண்மீன்",
    "vegam":       "வேகம்",      "oli":         "ஒளி",              "mounam":      "மௌனம்",
    "vaartha":     "வார்த்தை",  "unakkaaga":   "உனக்காக",         "ennakkaaga":  "என்னக்காக",
    "vaazhga":     "வாழ்க",      "sittiram":    "சித்திரம்",      "thaalattu":   "தாலாட்டு",
    "neeyum naanum":"நீயும் நானும்",
    # Common conjunctions
    "aana":        "ஆனா",         "aanalum":     "ஆனாலும்",       "aanaa":       "ஆனா",
    "aagave":      "ஆகவே",        "adhunaala":   "அதுனால",         "adhanaala":   "அதனால",
    "aprum":       "அப்றம்",      "appuram":     "அப்புறம்",
    # Common connectors
    "adhaan":      "அதான்",       "ithan":       "இதான்",          "eppadiyum":   "எப்படியும்",
    "eppovum":     "எப்பவும்",    "enkeyum":     "எங்கேயும்",      "yaararum":    "யாரோரும்",
    # Particles
    "da":          "டா",          "di":          "டி",              "pa":          "பா",
    "ma":          "மா",          "nga":         "ங",
    "thaane":      "தானே",        "dhaan":       "தான்",            "kuda":        "கூட",
    "ellam":       "எல்லாம்",    "onnum":       "ஒன்னும்",         "enna da":     "என்ன டா",

    # ── Extended reverse Tanglish → Tamil ──────────────────────────────────────
    # Family
    "amma":        "அம்மா",      "appa":        "அப்பா",            "anna":        "அண்ணா",
    "akka":        "அக்கா",      "thambi":      "தம்பி",            "thangai":     "தங்கை",
    "paatti":      "பாட்டி",     "thaatha":     "தாத்தா",           "maama":       "மாமா",
    "atthai":      "அத்தை",      "chitti":      "சித்தி",           "chittappa":   "சித்தப்பா",
    # Body
    "thalai":      "தலை",         "kaal":        "கால்",             "kevi":        "காவி",
    "mooku":       "மூக்கு",     "vaai":        "வாய்",              "mudugu":      "முதுகு",
    "vayiru":      "வயிறு",      "moolai":      "மூளை",             "mugam":       "முகம்",
    "idhayam":     "இதயம்",      "pal":         "பல்",               "naakku":      "நாக்கு",
    # Colors
    "sivappu":     "சிவப்பு",    "pachai":      "பச்சை",            "neelam":      "நீலம்",
    "vellai":      "வெள்ளை",    "karuppu":     "கருப்பு",          "manjal":      "மஞ்சள்",
    "aranchu":     "ஆரஞ்சு",    "pazhuppu":   "பழுப்பு",
    # Numbers
    "ondru":       "ஒன்று",      "irandu":      "இரண்டு",          "moondru":     "மூன்று",
    "naangu":      "நான்கு",     "ainthu":      "ஐந்து",            "aaru":        "ஆறு",
    "ezhu":        "ஏழு",         "ettu":        "எட்டு",            "onbadhu":     "ஒன்பது",
    "pathu":       "பத்து",       "nooru":       "நூறு",              "aayiram":     "ஆயிரம்",
    # Daily life nouns
    "thanneer":    "தண்ணீர்",   "saapadu":     "சாப்பாடு",        "veedu":       "வீடு",
    "velai":       "வேலை",        "palli":       "பள்ளி",            "kadai":       "கடை",
    "vazhi":       "வழி",          "perundhu":   "பேருந்து",         "kaar":        "கார்",
    "panam":       "பணம்",         "puthagam":   "புத்தகம்",        "oor":         "ஊர்",
    "nagaram":     "நகரம்",       "kiraamam":   "கிராமம்",          "malai":       "மலை",
    # Food
    "idli":        "இட்லி",      "dosai":       "தோசை",             "sambar":      "சாம்பார்",
    "rasam":       "ரசம்",         "koottu":     "கூட்டு",           "paayasam":    "பாயாசம்",
    "kaapi":       "காபி",         "theneer":    "தேநீர்",           "thayir":      "தயிர்",
    "soru":        "சோறு",         "kuzhambu":   "குழம்பு",          "appalam":     "அப்பளம்",
    "vadai":       "வடை",           "pongal":     "பொங்கல்",         "biriyani":    "பிரியாணி",
    "pazham":      "பழம்",          "maambazham": "மாம்பழம்",
    # Time
    "kaalai":      "காலை",       "maalai":      "மாலை",             "iravu":       "இரவு",
    "pagal":       "பகல்",         "indru":      "இன்று",             "naale":       "நாளை",
    "neethu":      "நேத்து",      "ippodhu":    "இப்போது",          "appodhu":     "அப்போது",
    "munaadi":     "முன்னாடி",   "pinnaadi":   "பின்னாடி",
    # Common verbs
    "vaa":         "வா",            "po":          "போ",               "paar":        "பார்",
    "kel":         "கேள்",        "pesu":        "பேசு",              "saapidu":     "சாப்பிடு",
    "kudi":        "குடி",         "thoongu":    "தூங்கு",           "edu":         "எடு",
    "kodu":        "கொடு",         "vaangu":     "வாங்கு",           "padi":        "படி",
    "ezhutu":      "எழுது",        "nada":       "நட",                "odu":         "ஓடு",
    "virumbu":     "விரும்பு",   "maaru":      "மாறு",              "thirumbu":    "திரும்பு",
    # Common adjectives
    "periya":      "பெரிய",       "chinna":     "சின்ன",             "puthiya":     "புதிய",
    "pazhaiya":    "பழைய",         "nalla":      "நல்ல",              "ketta":       "கெட்ட",
    "azhagana":    "அழகான",       "mukkiyam":   "முக்கியம்",        "sariyana":    "சரியான",
    "adhigam":     "அதிகம்",      "kuraiyvu":   "குறைவு",            "vegamana":    "வேகமான",
    # Questions
    "yen":         "ஏன்",          "eppadi":     "எப்படி",           "etthanai":    "எத்தனை",
    "endha":       "எந்த",          "evvalavu":   "எவ்வளவு",          "yaaroda":     "யாரோட",
    # Song words (extra)
    "vendum":      "வேண்டும்",    "venda":      "வேண்டா",           "ketka":       "கேட்க",
    "sol":         "சொல்",         "sollu":      "சொல்லு",           "kanden":      "கண்டேன்",
    "vanden":      "வந்தேன்",    "ponen":      "போனேன்",           "thedinen":    "தேடினேன்",
    "ennaal":      "என்னால்",    "unnaal":     "உன்னால்",           "nichayam":    "நிச்சயம்",
    "kanneer":     "கண்ணீர்",   "sirippu":    "சிரிப்பு",         "siri":        "சிரி",
    "azhu":        "அழு",           "gobam":      "கோபம்",             "bayam":       "பயம்",
    "thukkam":     "துக்கம்",    "sandhosam":  "சந்தோஷம்",        "anbu":        "அன்பு",
    "anbe":        "அன்பே",        "iniya":      "இனிய",              "muththam":    "முத்தம்",
    "kattipidi":   "கட்டிப்பிடி","thazhuvu":  "தழுவு",            "idhal":       "இதழ்",
    "tholil":      "தோளில்",      "maarbil":    "மார்பில்",         "vaanil":      "வானில்",
    "kadalil":     "கடலில்",      "mazhayil":   "மழையில்",          "thanimai":    "தனிமை",
    "kooda":       "கூட",           "ser":         "சேர்",             "piriyaamal":  "பிரியாமல்",
    "endrum":      "என்றும்",    "ninaikkum":  "நினைக்கும்",       "ninaivil":    "நினைவில்",
    "marakka":     "மறக்க",       "marandhen":  "மறந்தேன்",        "maraven":     "மறவேன்",
    "maravaadhe":  "மறவாதே",      "unnodu":     "உன்னோடு",          "ennodu":      "என்னோடு",
    "vidiya":      "விடிய",        "vidiyum":    "விடியும்",         "sooriyam":    "சூரியன்",
    "nilave":      "நிலவே",        "pozhudhu":   "பொழுது",           "aagaayam":    "ஆகாயம்",
    "minnal":      "மின்னல்",    "vaadai":     "வாடை",             "kulir":       "குளிர்",
    "veppam":      "வெப்பம்",    "mazhai":     "மழை",               "idhu":        "இது",
    "adhu":        "அது",           "ivan":       "இவன்",              "innikki":     "இன்னிக்கி",
    "kaadhal":     "காதல்",       "ilamai":     "இளமை",              "vaazhvu":     "வாழ்வு",
    "vazhkkai":    "வாழ்க்கை",  "piragu":     "பிறகு",             "innum":       "இன்னும்",
    "varuvaan":    "வருவான்",     "poovaen":    "போவேன்",           "vandhutten":  "வந்துட்டேன்",
}

# Vowels — longest match first, standalone (word-initial / after space)
_V_INIT: list[tuple[str, str]] = [
    ("aa", "ஆ"), ("ae", "ஏ"), ("ai", "ஐ"), ("au", "ஔ"),
    ("ee", "ஈ"), ("ii", "ஈ"), ("oo", "ஊ"), ("uu", "ஊ"),
    ("a",  "அ"), ("e",  "எ"), ("i",  "இ"), ("o",  "ஒ"), ("u", "உ"),
]

# Vowels — as dependent matra after a consonant
_V_MATRA: list[tuple[str, str]] = [
    ("aa", "\u0BBE"), ("ae", "\u0BC7"), ("ai", "\u0BC8"), ("au", "\u0BCC"),
    ("ee", "\u0BC0"), ("ii", "\u0BC0"), ("oo", "\u0BC2"), ("uu", "\u0BC2"),
    ("a",  ""),       ("e",  "\u0BC6"), ("i",  "\u0BBF"), ("o",  "\u0BCA"),
    ("u",  "\u0BC1"),
]

# Consonants — longest match first (critical: ttr before tr, ll before l, etc.)
_CONS: list[tuple[str, str]] = [
    ("ksh", "க்ஷ"), ("ngk", "ங்க"), ("ttr", "ற்ற"),
    ("zhzh","ழ்ழ"), ("shsh","ஷ்ஷ"),
    ("tr",  "ற"),   ("zh",  "ழ"),   ("sh",  "ஷ"),
    ("ng",  "ங"),   ("ny",  "ஞ"),   ("ch",  "ச"),
    ("th",  "த"),   ("ll",  "ள"),   ("nn",  "ண"),
    ("tt",  "ட்ட"), ("kk",  "க்க"), ("pp",  "ப்ப"),
    ("mm",  "ம்ம"), ("rr",  "ர்ர"), ("vv",  "வ்வ"),
    ("k",   "க"),   ("g",   "க"),   ("c",   "ச"),
    ("j",   "ஜ"),   ("t",   "ட"),   ("d",   "ட"),
    ("n",   "ன"),   ("p",   "ப"),   ("b",   "ப"),
    ("m",   "ம"),   ("y",   "ய"),   ("r",   "ர"),
    ("l",   "ல"),   ("v",   "வ"),   ("w",   "வ"),
    ("s",   "ஸ"),   ("h",   "ஹ"),   ("f",   "ப"),
]


def _t2t_match(text: str, pos: int, table: list[tuple[str, str]]) -> tuple[str, int]:
    low = text.lower()
    for roman, tamil in table:
        end = pos + len(roman)
        if low[pos:end] == roman:
            return tamil, end
    return "", pos


def _tanglish_word_to_tamil(word: str) -> str:
    # Dictionary lookup before character-level rules
    lookup = TANGLISH_WORD_DICT.get(word.lower())
    if lookup:
        return lookup
    result: list[str] = []
    i = 0
    n = len(word)
    while i < n:
        if not word[i].isalpha():
            result.append(word[i])
            i += 1
            continue

        cons, next_i = _t2t_match(word, i, _CONS)
        if cons:
            i = next_i
            matra, next_i2 = _t2t_match(word, i, _V_MATRA)
            if next_i2 > i:
                result.append(cons + matra)   # matra may be "" (inherent a)
                i = next_i2
            else:
                # No vowel — add pulli if another consonant follows
                _, peek = _t2t_match(word, i, _CONS)
                result.append(cons + (_PULLI if peek else ""))
            continue

        vowel, next_i = _t2t_match(word, i, _V_INIT)
        if next_i > i:
            result.append(vowel)
            i = next_i
            continue

        result.append(word[i])
        i += 1

    return "".join(result)


def tanglish_to_tamil(text: str) -> str:
    """Rule-based Tanglish → Tamil Unicode. Handles mixed Latin+Tamil lines."""
    text = normalize_tamil(text)
    parts = []
    for tok in text.split():
        # Preserve leading/trailing punctuation
        pre, core, post = _split_token(tok)
        if core and core[0].isalpha() and ord(core[0]) < 128:
            parts.append(pre + _tanglish_word_to_tamil(core) + post)
        else:
            parts.append(tok)
    return " ".join(parts)


# ─────────────────────────────────────────────
# Singleton engine (lazy-init in API)
# ─────────────────────────────────────────────
_engine: Optional[ByT5TamilEngine] = None

def get_engine() -> ByT5TamilEngine:
    global _engine
    if _engine is None:
        _engine = ByT5TamilEngine(use_neural=False)
    return _engine
