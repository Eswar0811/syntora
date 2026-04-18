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

HALANT        = "\u094D"  # ् virama
ANUSVARA      = "\u0902"  # ं  (bindhu)
CHANDRABINDU  = "\u0901"  # ँ  (chandrabindu — nasal, same romanization as anusvara)
VISARGA       = "\u0903"  # ः


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
    ANUSVARA:     "n",   # ं
    CHANDRABINDU: "n",   # ँ
    VISARGA:      "h",   # ः
    HALANT:       "",    # ् no inherent vowel
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
            # Check for nukta-modified form (e.g. ज + ़ = ज़ → z)
            NUKTA   = "\u093C"
            next_ch = word[i + 1] if i + 1 < n else ""
            if next_ch == NUKTA:
                combined = ch + NUKTA
                if combined in CONSONANTS:
                    base = CONSONANTS[combined]
                    i += 1          # skip the nukta — consume it here
                    next_ch = word[i + 1] if i + 1 < n else ""
                else:
                    base = CONSONANTS[ch]
            else:
                base = CONSONANTS[ch]

            next2_ch = word[i + 2] if i + 2 < n else ""

            if next_ch == HALANT:
                result.append(base)
                i += 2
            elif next_ch in (ANUSVARA, CHANDRABINDU):
                # Anusvara / chandrabindu after consonant: inherent 'a' + nasal
                result.append(base + "a" + MATRAS[next_ch])
                i += 2
            elif next_ch in MATRAS:
                result.append(base + MATRAS[next_ch])
                i += 2
            else:
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

    # ── Song / Bollywood vocabulary ──────────────────────────────────────────
    # Love & longing
    "mohabbat":  "मोहब्बत",  "mohabbatein": "मोहब्बतें",
    "pyar":      "प्यार",    "chahat":      "चाहत",
    "dil":       "दिल",      "dilo":        "दिलों",
    "dilse":     "दिल से",   "dil se":      "दिल से",
    "dil mein":  "दिल में",  "dil hai":     "दिल है",
    "dil ko":    "दिल को",   "dil ka":      "दिल का",
    "armaan":    "अरमान",    "armaano":     "अरमानों",
    "armaanon":  "अरमानों",  "khwaab":      "ख़्वाब",
    "khwab":     "ख़्वाब",   "khwabon":     "ख़्वाबों",
    "sapna":     "सपना",     "sapne":       "सपने",
    "sapno":     "सपनों",    "arzoo":       "आरज़ू",
    "tamanna":   "तमन्ना",   "tamannaa":    "तमन्ना",
    "intezaar":  "इंतज़ार",  "intezar":     "इंतज़ार",
    "wait":      "वेट",      "yaad":        "याद",
    "yaaden":    "यादें",    "yaadon":      "यादों",
    "yaad aana": "याद आना",  "yaad hai":    "याद है",
    "wafa":      "वफ़ा",     "bewafa":      "बेवफ़ा",
    "judai":     "जुदाई",    "judaai":      "जुदाई",
    "bichadna":  "बिछड़ना",  "milna":       "मिलना",
    "tanha":     "तन्हा",    "akela":       "अकेला",
    "akeli":     "अकेली",    "akele":       "अकेले",
    # Heart / soul / body
    "rooh":      "रूह",      "jaan":        "जान",
    "jaana":     "जाना",     "mere jaan":   "मेरी जान",
    "meri jaan": "मेरी जान","jaaneman":    "जानेमन",
    "jaanewala": "जानेवाला", "aankhein":    "आँखें",
    "aankhen":   "आँखें",   "aankhon":     "आँखों",
    "aansu":     "आँसू",    "aansoon":     "आँसुओं",
    "aankhon mein":"आँखों में","honth":     "होंठ",
    "hont":      "होंठ",    "haath":       "हाथ",
    "hath":      "हाथ",     "haathon":     "हाथों",
    # Nature / sky
    "raat":      "रात",      "raaton":      "रातों",
    "raat mein": "रात में",  "din":         "दिन",
    "subah":     "सुबह",     "shaam":       "शाम",
    "sham":      "शाम",      "dopahar":     "दोपहर",
    "baarish":   "बारिश",    "barish":      "बारिश",
    "baadal":    "बादल",     "bijli":       "बिजली",
    "aakash":    "आकाश",     "aasman":      "आसमान",
    "sitara":    "सितारा",   "sitare":      "सितारे",
    "sitaron":   "सितारों",  "chaand":      "चाँद",
    "chand":     "चाँद",    "chandni":      "चाँदनी",
    "suraj":     "सूरज",     "dhoop":       "धूप",
    "hawa":      "हवा",      "hawao":       "हवाओं",
    "pani":      "पानी",     "darya":       "दरिया",
    "nadiya":    "नदिया",    "kinara":      "किनारा",
    "kinaaron":  "किनारों",  "mausam":      "मौसम",
    "bahaar":    "बहार",     "bahar":       "बहार",
    "phool":     "फूल",      "phoolon":     "फूलों",
    "gulshan":   "गुलशन",   "baagho":       "बाग़ों",
    # Emotion / feelings
    "dard":      "दर्द",     "dardo":       "दर्दों",
    "khushi":    "ख़ुशी",    "khushiyan":   "ख़ुशियाँ",
    "gham":      "ग़म",      "gamo":        "ग़मों",
    "dukh":      "दुख",      "sukh":        "सुख",
    "chain":     "चैन",      "aaram":       "आराम",
    "sookoon":   "सुकून",    "sukoon":      "सुकून",
    "ehsaas":    "एहसास",    "ehsas":       "एहसास",
    "jazbaat":   "जज़्बात",  "dhadkan":     "धड़कन",
    "dhadkane":  "धड़कनें",  "rulaana":     "रुलाना",
    "rona":      "रोना",     "hasna":       "हँसना",
    "hansi":     "हँसी",     "muskaan":     "मुस्कान",
    "muskuraan": "मुस्कुरान",
    # Common song phrases
    "tere bina": "तेरे बिना","tere baad":   "तेरे बाद",
    "tere liye": "तेरे लिए","tere saath":   "तेरे साथ",
    "mere paas": "मेरे पास","mere liye":    "मेरे लिए",
    "mere saath":"मेरे साथ","mere bin":     "मेरे बिन",
    "tujhse":    "तुझसे",   "tujhe":        "तुझे",
    "mujhse":    "मुझसे",   "mujhko":       "मुझको",
    "hume":      "हमें",    "humein":       "हमें",
    "tumhara":   "तुम्हारा","tumhari":      "तुम्हारी",
    "tumhare":   "तुम्हारे","tujhko":       "तुझको",
    "kyon":      "क्यों",   "kyonki":       "क्योंकि",
    "isliye":    "इसलिए",   "isiliye":      "इसीलिए",
    # Verbs (conjugated — hard for rule engine)
    "bhool":     "भूल",     "bhoola":       "भूला",
    "bhooli":    "भूली",    "bhoole":       "भूले",
    "bhoolna":   "भूलना",   "bhool gaye":   "भूल गए",
    "chhod":     "छोड़",    "chhoda":       "छोड़ा",
    "chhodna":   "छोड़ना",  "chhod diya":   "छोड़ दिया",
    "dekh":      "देख",     "dekha":        "देखा",
    "dekhi":     "देखी",    "dekho":        "देखो",
    "dekhna":    "देखना",   "dekhe":        "देखे",
    "sun":       "सुन",     "suno":         "सुनो",
    "suna":      "सुना",    "suni":         "सुनी",
    "sunna":     "सुनना",   "sunle":        "सुन ले",
    "ruk":       "रुक",     "ruko":         "रुको",
    "rukna":     "रुकना",   "bol":          "बोल",
    "bolo":      "बोलो",    "bola":         "बोला",
    "bolna":     "बोलना",   "keh":          "कह",
    "kaho":      "कहो",     "kaha":         "कहा",
    "kehna":     "कहना",    "kehdo":        "कह दो",
    "chalna":    "चलना",    "chal":         "चल",
    "chalo":     "चलो",     "chale":        "चले",
    "aaja":      "आजा",     "aajao":        "आ जाओ",
    "aana":      "आना",     "aaye":         "आए",
    "jaana":     "जाना",    "jaoge":        "जाओगे",
    "jayenge":   "जाएंगे",  "jaaoge":       "जाओगे",
    "dena":      "देना",    "dedo":         "दे दो",
    "lena":      "लेना",    "le lo":        "ले लो",
    "reh":       "रह",      "raho":         "रहो",
    "rehna":     "रहना",    "rahe":         "रहे",
    "mil":       "मिल",     "milo":         "मिलो",
    "soch":      "सोच",     "socha":        "सोचा",
    "sochna":    "सोचना",   "maan":         "मान",
    "maano":     "मानो",    "maana":        "माना",
    "paana":     "पाना",    "paya":         "पाया",
    "paaye":     "पाए",     "khona":        "खोना",
    "khoya":     "खोया",    "khoyi":        "खोई",
    "toot":      "टूट",     "toota":        "टूटा",
    "tutna":     "टूटना",   "jee":          "जी",
    "jeena":     "जीना",    "jiye":         "जिए",
    # Particles & connectors
    "wahan":     "वहाँ",    "yahan":        "यहाँ",
    "idhar":     "इधर",     "udhar":        "उधर",
    "pehle":     "पहले",    "baad":         "बाद",
    "saath":     "साथ",     "bina":         "बिना",
    "liye":      "लिए",     "wajah":        "वजह",
    "wajha":     "वजह",     "bas":          "बस",
    "sirf":      "सिर्फ़",  "zaroor":       "ज़रूर",
    "shayad":    "शायद",    "kabhi":        "कभी",
    "kabhi na":  "कभी ना",  "hamesha":      "हमेशा",
    "kaafi":     "काफ़ी",   "bohat":        "बहुत",
    "itna":      "इतना",    "itni":         "इतनी",
    "aisa":      "ऐसा",     "aisi":         "ऐसी",
    "waisa":     "वैसा",    "jaisa":        "जैसा",
    "jaisi":     "जैसी",    "jaise":        "जैसे",
    # Commonly sung words
    "o":         "ओ",       "oh":           "ओह",
    "aa re":     "आ रे",    "o re":         "ओ रे",
    "arre":      "अरे",     "arrey":        "अरे",
    "hai re":    "है रे",   "re":           "रे",
    "na na":     "ना ना",   "la la":        "ला ला",
    "haye":      "हाय",     "hai na":       "है ना",
    "hai kya":   "है क्या",
    # Nouns (song context)
    "pal":       "पल",      "palon":        "पलों",
    "waqt":      "वक़्त",   "samay":        "समय",
    "door":      "दूर",     "paas":         "पास",
    "raah":      "राह",     "raste":        "रास्ते",
    "manzil":    "मंज़िल",  "safar":        "सफ़र",
    "sафar":     "सफ़र",    "musafir":      "मुसाफ़िर",
    "mohabbton": "मोहब्बतों","pyaron":      "प्यारों",
    "baahon":    "बाहों",   "baahen":       "बाहें",
    "teri baahon":"तेरी बाहों","gore":      "गोरे",
    "gehra":     "गहरा",    "gehri":        "गहरी",
    "tujhme":    "तुझमें",  "mujhme":       "मुझमें",
    "dono":      "दोनों",   "donon":        "दोनों",
    "sanam":     "सनम",     "jaanu":        "जानू",

    # ── Verb conjugations the rule engine mis-handles ─────────────────────
    # -ta / -ti / -te  (present habitual, masculine/feminine/plural)
    "karta":    "करता",   "karti":    "करती",   "karte":    "करते",
    "bolta":    "बोलता",  "bolti":    "बोलती",  "bolte":    "बोलते",
    "chahta":   "चाहता",  "chahti":   "चाहती",  "chahte":   "चाहते",
    "sunta":    "सुनता",  "sunti":    "सुनती",  "sunte":    "सुनते",
    "deta":     "देता",   "deti":     "देती",   "dete":     "देते",
    "leta":     "लेता",   "leti":     "लेती",   "lete":     "लेते",
    "rehta":    "रहता",   "rehti":    "रहती",   "rehte":    "रहते",
    "jata":     "जाता",   "jati":     "जाती",   "jate":     "जाते",
    "aata":     "आता",    "aati":     "आती",    "aate":     "आते",
    "dekhhta":  "देखता",  "dekhta":   "देखता",  "dekhti":   "देखती",
    "hota":     "होता",   "hoti":     "होती",   "hote":     "होते",
    "milta":    "मिलता",  "milti":    "मिलती",  "milte":    "मिलते",
    "rota":     "रोता",   "roti":     "रोती",   "rote":     "रोते",
    "manta":    "मानता",  "manti":    "मानती",  "mante":    "मानते",
    "samjhta":  "समझता",  "samjhti":  "समझती",  "samjhte":  "समझते",
    # -a / -i / -e  (simple past masculine/feminine/plural)
    "gaya":     "गया",    "gayi":     "गई",     "gaye":     "गए",
    "aaya":     "आया",    "aayi":     "आई",     "aaye":     "आए",
    "hua":      "हुआ",    "hui":      "हुई",    "hue":      "हुए",
    "tha":      "था",     "thi":      "थी",     "the":      "थे",
    "diya":     "दिया",   "di":       "दी",     "diye":     "दिए",
    "liya":     "लिया",   "li":       "ली",     "liye":     "लिए",
    "kiya":     "किया",   "ki":       "की",     "kiye":     "किए",
    "mila":     "मिला",   "mili":     "मिली",   "mile":     "मिले",
    "raha":     "रहा",    "rahi":     "रही",    "rahe":     "रहे",
    "khoya":    "खोया",   "khoyi":    "खोई",    "khoye":    "खोए",
    "bola":     "बोला",   "boli":     "बोली",   "bole":     "बोले",
    "suna":     "सुना",   "suni":     "सुनी",   "sune":     "सुने",
    "dekha":    "देखा",   "dekhi":    "देखी",   "dekhe":    "देखे",
    "toda":     "तोड़ा",  "todi":     "तोड़ी",  "tode":     "तोड़े",
    "tod":      "तोड़",   "toot":     "टूट",    "toota":    "टूटा",
    "paya":     "पाया",   "payi":     "पाई",    "paye":     "पाए",
    "chaha":    "चाहा",   "chahi":    "चाही",   "chahe":    "चाहे",
    "soocha":   "सोचा",   "soochi":   "सोची",   "sooche":   "सोचे",
    "bhoola":   "भूला",   "bhooli":   "भूली",   "bhoole":   "भूले",
    "maana":    "माना",   "maani":    "मानी",   "maane":    "माने",
    "jaana":    "जाना",   "jaani":    "जानी",   "jaane":    "जाने",
    "pehchana": "पहचाना", "choda":    "छोड़ा",  "chodi":    "छोड़ी",
    # -ega / -egi / -enge  (future)
    "karega":   "करेगा",  "karegi":   "करेगी",  "karenge":  "करेंगे",
    "jayega":   "जाएगा",  "jayegi":   "जाएगी",  "jayenge":  "जाएंगे",
    "aayega":   "आएगा",   "aayegi":   "आएगी",   "aayenge":  "आएंगे",
    "hoga":     "होगा",   "hogi":     "होगी",   "honge":    "होंगे",
    "milega":   "मिलेगा", "milegi":   "मिलेगी", "milenge":  "मिलेंगे",
    "rahega":   "रहेगा",  "rahegi":   "रहेगी",  "rahenge":  "रहेंगे",
    "bolega":   "बोलेगा", "bolegi":   "बोलेगी", "bolenge":  "बोलेंगे",
    "dekhega":  "देखेगा", "dekhegi":  "देखेगी", "dekhenge": "देखेंगे",
    # Compound / common past
    "kar diya": "कर दिया","kar di":   "कर दी",  "kar do":   "कर दो",
    "kar le":   "कर ले",  "kar lo":   "कर लो",  "kar liya": "कर लिया",
    "ho gaya":  "हो गया", "ho gayi":  "हो गई",  "ho gaye":  "हो गए",
    "aa gaya":  "आ गया",  "aa gayi":  "आ गई",   "aa gaye":  "आ गए",
    "ja raha":  "जा रहा", "ja rahi":  "जा रही",
    "kar raha": "कर रहा", "kar rahi": "कर रही",
    # Often-wrong specific words
    "saari":    "सारी",   "sari":     "साड़ी",  "saara":    "सारा",
    "laoon":    "लाऊं",   "laaoon":   "लाऊं",   "laoonga":  "लाऊंगा",
    "tujhpe":   "तुझपे",  "mujhpe":   "मुझपे",  "tumpe":    "तुमपे",
    "vaaron":   "वारूँ",  "vaarun":   "वारूँ",
    "jhoom":    "झूम",    "jhoome":   "झूमें",  "jhoomna":  "झूमना",
    "ghoom":    "घूम",    "ghoome":   "घूमें",  "ghoomna":  "घूमना",
    "saans":    "साँस",   "saansen":  "साँसें",
    "teri saans":"तेरी साँसों में",
    "raaton":   "रातों",  "raato":    "रातों",
    "palon":    "पलों",   "pal":      "पल",
    "khwabon":  "ख़्वाबों","armaano":  "अरमानों",
    "aankhon mein":"आँखों में",
    "dil mein": "दिल में","dil se":   "दिल से",
    "mann":     "मन",     "mano":     "मनों",   "mann mein":"मन में",
    "raat ko":  "रात को", "din ko":   "दिन को",
    "jab":      "जब",     "tab":      "तब",     "jab bhi":  "जब भी",
    "jaha":     "जहाँ",   "jahan":    "जहाँ",   "jab se":   "जब से",
    "har":      "हर",     "har pal":  "हर पल",  "har roz":  "हर रोज़",
    "har dam":  "हर दम",  "har waqt": "हर वक़्त",
    "o re":     "ओ रे",   "o sanam":  "ओ सनम",
    "sun le":   "सुन ले", "maan le":  "मान ले",
    "chal":     "चल",     "chal de":  "चल दे",
    "ruk ja":   "रुक जा", "ruk jao":  "रुक जाओ",
    # Words the rule engine halant-splits incorrectly
    "vaada":    "वादा",   "waada":    "वादा",   "wada":     "वादा",
    "shikwa":   "शिकवा",  "shikwah":  "शिकवा",
    "sakte":    "सकते",   "sakta":    "सकता",   "sakti":    "सकती",
    "nahi sakte":"नहीं सकते","nahi sakta":"नहीं सकता",
    "rakhna":   "रखना",   "rakha":    "रखा",    "rakh":     "रख",
    "nikalna":  "निकलना", "nikla":    "निकला",  "nikli":    "निकली",
    "sambhal":  "संभाल",  "sambhalna":"संभालना","sambhala":  "संभाला",
    "muskura":  "मुस्कुरा","muskurana":"मुस्कुराना",
    "tadap":    "तड़प",   "tadpana":  "तड़पना", "tadpa":    "तड़पा",
    "theharna": "ठहरना",  "thehre":   "ठहरे",   "thehra":   "ठहरा",
    "bikhar":   "बिखर",   "bikharna": "बिखरना", "bikhre":   "बिखरे",
    "sawaal":   "सवाल",   "jawaab":   "जवाब",   "jawab":    "जवाब",
    "khwahish": "ख्वाहिश","armaan":   "अरमान",
    "kasam":    "कसम",    "kasme":    "क़समें",
    "tasveer":  "तस्वीर", "tasviren": "तस्वीरें",
    "zulfein":  "जुल्फ़ें","zulf":     "जुल्फ़",
    "aahon":    "आहों",   "aah":      "आह",
    "sitam":    "सितम",   "zulm":     "ज़ुल्म",
    "woh lamhe":"वो लम्हे","lamhe":   "लम्हे",  "lamha":    "लम्हा",
    "teri yaad":"तेरी याद","meri yaad":"मेरी याद",
    "doori":    "दूरी",   "nazar":    "नज़र",   "nazron":   "नज़रों",
    "aankhon":  "आँखों",  "aankhen":  "आँखें",
    "zara":     "ज़रा",   "zara si":  "ज़रा सी",
    "beqarar":  "बेक़रार","bekaraar":  "बेक़रार",
    "dhadke":   "धड़के",  "dhadkane": "धड़कनें",
    "panghat":  "पनघट",   "palkon":   "पलकों",
    "palkein":  "पलकें",  "pal bhar": "पल भर",

    # ── Family / relationships ───────────────────────────────────────────────
    "maa":      "माँ",    "baap":     "बाप",    "papa":     "पापा",
    "bhai":     "भाई",    "behen":    "बहन",    "beta":     "बेटा",
    "beti":     "बेटी",   "dadi":     "दादी",   "dada":     "दादा",
    "nani":     "नानी",   "nana":     "नाना",   "chacha":   "चाचा",
    "chachi":   "चाची",   "mama":     "मामा",   "mami":     "मामी",
    "bhabhi":   "भाभी",   "devar":    "देवर",   "saas":     "सास",
    "sasur":    "ससुर",   "dulhan":   "दुल्हन", "dulha":    "दूल्हा",
    "dost":     "दोस्त",  "yaar":     "यार",    "saathi":   "साथी",

    # ── Food ──────────────────────────────────────────────────────────────────
    "roti":     "रोटी",   "sabzi":    "सब्ज़ी", "dal":      "दाल",
    "chawal":   "चावल",   "mithai":   "मिठाई",  "chai":     "चाय",
    "doodh":    "दूध",    "namak":    "नमक",    "mirch":    "मिर्च",
    "masala":   "मसाला",  "ghee":     "घी",     "biryani":  "बिरयानी",
    "samosa":   "समोसा",  "jalebi":   "जलेबी",  "halwa":    "हलवा",
    "lassi":    "लस्सी",  "sharbat":  "शरबत",

    # ── Numbers ───────────────────────────────────────────────────────────────
    "ek":       "एक",     "do":       "दो",      "teen":     "तीन",
    "chaar":    "चार",    "paanch":   "पाँच",   "chhe":     "छह",
    "saat":     "सात",    "aath":     "आठ",     "nau":      "नौ",
    "das":      "दस",     "bees":     "बीस",    "sau":      "सौ",
    "hazaar":   "हज़ार",

    # ── Body / health ─────────────────────────────────────────────────────────
    "sir":      "सिर",    "baal":     "बाल",    "pair":     "पैर",
    "haath":    "हाथ",    "pet":      "पेट",    "seena":    "सीना",
    "chehara":  "चेहरा",  "dimag":    "दिमाग",  "jism":     "जिस्म",
    "badan":    "बदन",

    # ── Common verbs / gerunds ────────────────────────────────────────────────
    "bata":     "बता",    "batao":    "बताओ",   "batana":   "बताना",
    "samjhao":  "समझाओ",  "karke":    "करके",   "jaake":    "जाकर",
    "aakar":    "आकर",    "dekhke":   "देखकर",  "likhna":   "लिखना",
    "padna":    "पढ़ना",  "nachna":   "नाचना",  "gaana":    "गाना",
    "khelna":   "खेलना",  "larna":    "लड़ना",   "jeetna":   "जीतना",
    "rona":     "रोना",   "hasna":    "हंसना",  "haarna":   "हारना",
    "uthna":    "उठना",   "baithna":  "बैठना",  "sona":     "सोना",
    "bolna":    "बोलना",  "sunna":    "सुनना",  "dekhna":   "देखना",
    "milna":    "मिलना",  "chalna":   "चलना",   "daudna":   "दौड़ना",

    # ── Phrases / social connectors ────────────────────────────────────────────
    "pata":     "पता",    "pata nahi":"पता नहीं","pata hai": "पता है",
    "lag raha": "लग रहा", "lagta hai":"लगता है","lagti hai":"लगती है",
    "yahin":    "यहीं",   "wahin":    "वहीं",   "wahi":     "वही",
    "yehi":     "यही",    "isi":      "इसी",    "usi":      "उसी",
    "matlab":   "मतलब",   "seedha":   "सीधा",   "waise":    "वैसे",
    "waise bhi":"वैसे भी","mushkil":  "मुश्किल","aasaan":   "आसान",
    "sundar":   "सुंदर",  "pyaara":   "प्यारा", "pyaari":   "प्यारी",
    "theek hai":"ठीक है", "shuruaat": "शुरुआत", "khatam":   "खत्म",
    "galat":    "गलत",    "sahi":     "सही",    "sachchi":  "सच्ची",

    # ── Song / poetry (Urdu-influenced) ───────────────────────────────────────
    "roshni":   "रोशनी",  "noor":     "नूर",    "roshan":   "रोशन",
    "surma":    "सुरमा",  "kajal":    "काजल",   "bindi":    "बिंदी",
    "mehendi":  "मेहंदी", "dupatta":  "दुपट्टा","chunri":   "चुनरी",
    "nasib":    "नसीब",   "taqdeer":  "तक़दीर",  "qismat":   "क़िस्मत",
    "dua":      "दुआ",    "duaen":    "दुआएं",  "dawaa":    "दवा",
    "ghazal":   "ग़ज़ल",  "sher":     "शेर",     "nazm":     "नज़्म",
    "taraana":  "तराना",  "fariyaad": "फ़रियाद",
    "khwaishein":"ख़्वाहिशें","aahat":  "आहट",   "mehfil":   "महफ़िल",
    "raat bhar":"रात भर", "subah tak":"सुबह तक",
    "teri aankhon":"तेरी आँखों","meri aankhon":"मेरी आँखों",
    "tere dil": "तेरे दिल","mere dil": "मेरे दिल",
    "teri yaaden":"तेरी यादें","meri duniya":"मेरी दुनिया",
    "teri muskaan":"तेरी मुस्कान","meri jaan": "मेरी जान",
    "teri kasam":"तेरी क़सम","bewajah":   "बेवजह",  "wajah":    "वजह",
    "paigam":   "पैगाम",  "sandesh":  "संदेश",  "khata":    "खता",
    "maafi":    "माफ़ी",  "माफ करो": "maaf karo","shikayat": "शिकायत",
    "shikwa":   "शिकवा",  "gilaah":   "गिला",    "ghubaar":  "ग़ुबार",
    "aasmaan":  "आसमान",  "zameen":   "ज़मीन",   "jahaan":   "जहान",
    "patthar":  "पत्थर",  "rang":     "रंग",     "rangon":   "रंगों",
    "sitaron":  "सितारों","khwabon":  "ख़्वाबों",

    # ── Numbers ───────────────────────────────────────────────────────────────
    "ek":       "एक",     "do":       "दो",      "teen":     "तीन",
    "chaar":    "चार",    "paanch":   "पाँच",   "chhah":    "छह",
    "saat":     "सात",    "aath":     "आठ",     "nau":      "नौ",
    "das":      "दस",     "gyaarah":  "ग्यारह", "baarah":   "बारह",
    "bis":      "बीस",    "pachaas":  "पचास",   "sau":      "सौ",
    "hazaar":   "हज़ार",  "laakh":    "लाख",

    # ── Colors ────────────────────────────────────────────────────────────────
    "laal":     "लाल",    "peela":    "पीला",    "neela":    "नीला",
    "hara":     "हरा",    "safed":    "सफ़ेद",   "kaala":    "काला",
    "narangi":  "नारंगी", "gulaabi":  "गुलाबी",  "baingani": "बैंगनी",
    "bhura":    "भूरा",   "grey":     "ग्रे",    "sunhara":  "सुनहरा",

    # ── Body parts (extended) ─────────────────────────────────────────────────
    "aankhein": "आँखें",  "kaan":     "कान",     "naak":     "नाक",
    "muh":      "मुँह",   "dant":     "दाँत",   "baal":     "बाल",
    "gardhan":  "गर्दन",  "kandha":   "कंधा",   "peetha":   "पीठ",
    "pet":      "पेट",    "haath":    "हाथ",     "pair":     "पैर",
    "ungli":    "उंगली",  "nakh":     "नख",      "ghutna":   "घुटना",

    # ── Daily life (extended) ─────────────────────────────────────────────────
    "ghar":     "घर",     "kamra":    "कमरा",   "darwaza":  "दरवाज़ा",
    "khidki":   "खिड़की", "chhath":   "छत",      "seedhi":   "सीढ़ी",
    "school":   "स्कूल",  "kitaab":   "किताब",   "kalam":    "क़लम",
    "machine":  "मशीन",   "phone":    "फ़ोन",    "computer": "कंप्यूटर",
    "sadak":    "सड़क",   "gaadi":    "गाड़ी",   "bus":      "बस",
    "train":    "ट्रेन",  "havai jahaz":"हवाई जहाज़",

    # ── Verbs (infinitive / gerunds, extended) ───────────────────────────────
    "bhaagna":  "भागना",  "koodna":   "कूदना",   "uthana":   "उठाना",
    "rakhna":   "रखना",   "dhundna":  "ढूंढना",  "paana":    "पाना",
    "banana":   "बनाना",  "todna":    "तोड़ना",  "kholna":   "खोलना",
    "band karna":"बंद करना","saaf karna":"साफ़ करना","bharna":  "भरना",
    "khaana":   "खाना",   "peena":    "पीना",    "pukarna":  "पुकारना",
    "bulana":   "बुलाना", "dekhna":   "देखना",   "dikhaana": "दिखाना",

    # ── Time (extended) ───────────────────────────────────────────────────────
    "subha":    "सुबह",   "dopahar":  "दोपहर",   "shaam":    "शाम",
    "raat":     "रात",    "aaj":      "आज",      "kal bhi":  "कल भी",
    "har din":  "हर दिन", "puri raat":"पूरी रात","din bhar":  "दिन भर",
    "thodi der":"थोड़ी देर","kuch pal": "कुछ पल","ek din":   "एक दिन",
    "mahina":   "महीना",  "saal":     "साल",     "ghanta":   "घंटा",
    "minute":   "मिनट",   "second":   "सेकंड",

    # ── Emotions / states (extended) ─────────────────────────────────────────
    "thaka":    "थका",    "thaki":    "थकी",     "thake":    "थके",
    "bhookha":  "भूखा",   "pyaasa":   "प्यासा",  "neend":    "नींद",
    "khwaab":   "ख़्वाब", "sapne mein":"सपने में","jaagna":   "जागना",
    "hosh":     "होश",    "pagal":    "पागल",    "bekarar":  "बेक़रार",
    "majboor":  "मजबूर",  "bebas":    "बेबस",    "aazad":    "आज़ाद",
    "khushi se":"ख़ुशी से","dukh mein":"दुख में",
    "dil dhadakta":"दिल धड़कता","rooh kaanpti":"रूह काँपती",
    "neend udana":"नींद उड़ाना","aankhon mein aansu":"आँखों में आँसू",
    "hothon pe": "होठों पे","haathon mein":"हाथों में",
    "waqt guzarna":"वक़्त गुज़रना","pal bitana":"पल बिताना",

    # ── Common conjunctions / transitions ─────────────────────────────────────
    "jabse":    "जबसे",   "tabse":    "तबसे",    "jitna":    "जितना",
    "utna":     "उतना",   "jaisa":    "जैसा",    "waisa":    "वैसा",
    "phir bhi": "फिर भी", "tab bhi":  "तब भी",   "fir se":   "फिर से",
    "aur bhi":  "और भी",  "naaki":    "नाकि",    "warna":    "वरना",
    "tabhi":    "तभी",    "yahi":     "यही",     "wahi sab": "वही सब",
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
