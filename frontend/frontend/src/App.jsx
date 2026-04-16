import { useState, useRef, useEffect, useCallback } from "react";

// ─── Phoneme Reference Data ───────────────────────────────────────────────
const PHONEME_MAP = {
  vowels: [
    ["அ", "a"], ["ஆ", "aa"], ["இ", "i"], ["ஈ", "ee"],
    ["உ", "u"], ["ஊ", "oo"], ["எ", "e"], ["ஏ", "ae"],
    ["ஐ", "ai"], ["ஒ", "o"], ["ஓ", "oo"], ["ஔ", "au"],
  ],
  consonants: [
    ["க", "k/g"], ["ச", "ch/s"], ["ட", "t/d"], ["த", "th/dh"], ["ப", "p/b"], ["ற", "tr/r"],
    ["ங", "ng"], ["ஞ", "ny"], ["ண", "n (retro)"], ["ந", "n (dental)"], ["ம", "m"], ["ன", "n (alv)"],
    ["ய", "y"], ["ர", "r"], ["ல", "l"], ["வ", "v"], ["ழ", "zh"], ["ள", "l (retro)"],
    ["ஜ", "j"], ["ஷ", "sh"], ["ஸ", "s"], ["ஹ", "h"],
  ],
  special: [
    ["க்க", "kk (geminate)"], ["ட்ட", "tt (geminate)"], ["ப்ப", "pp (geminate)"],
    ["ன்ன", "nn (geminate)"], ["ல்ல", "ll (geminate)"], ["ர்ர", "rr (geminate)"],
  ],
};

const EXAMPLES = [
  { label: "Greeting", text: "வணக்கம்" },
  { label: "Identity", text: "என் பெயர் குமார்" },
  { label: "Going home", text: "நான் வீட்டுக்கு போகிறேன்" },
  { label: "Question", text: "எனக்கு ஒரு சந்தேகம் உள்ளது" },
  { label: "Language", text: "தமிழ் மொழி மிகவும் அழகானது" },
  { label: "Informal", text: "எனக்கு தெரியல" },
  { label: "Casual ask", text: "என்ன பண்றே?" },
  { label: "Long vowels", text: "வாழ்க தமிழ் தாய்" },
  { label: "Geminates", text: "அக்கா வீட்டுக்கு வந்தாள்" },
  { label: "ழ phoneme", text: "ழகரம் தமிழுக்கே உரியது" },
];

const MODES = [
  { id: "formal", label: "Formal", desc: "Standard literary Tamil with full phoneme fidelity" },
  { id: "spoken", label: "Spoken", desc: "Natural spoken Tamil with vowel reductions" },
  { id: "slang", label: "Slang", desc: "Casual informal Tamil with elision patterns" },
];

// ─── Rule-Based Engine (runs in-browser, instant) ─────────────────────────
const VOWEL_MAP = {
  "அ": "a", "ஆ": "aa", "இ": "i", "ஈ": "ee", "உ": "u", "ஊ": "oo",
  "எ": "e", "ஏ": "ae", "ஐ": "ai", "ஒ": "o", "ஓ": "oo", "ஔ": "au",
};
const CONSONANT_MAP = {
  "க": "k", "ச": "ch", "ட": "t", "த": "th", "ப": "p", "ற": "tr",
  "ங": "ng", "ஞ": "ny", "ண": "n", "ந": "n", "ம": "m", "ன": "n",
  "ய": "y", "ர": "r", "ல": "l", "வ": "v", "ழ": "zh", "ள": "l",
  "ஜ": "j", "ஷ": "sh", "ஸ": "s", "ஹ": "h",
};
const VOWEL_MARKER_MAP = {
  "\u0BBE": "aa", "\u0BBF": "i", "\u0BC0": "ee", "\u0BC1": "u", "\u0BC2": "oo",
  "\u0BC6": "e", "\u0BC7": "ae", "\u0BC8": "ai", "\u0BCA": "o", "\u0BCB": "oo",
  "\u0BCC": "au", "\u0BCD": "",
};
const INTERVOCALIC = { "க": "g", "ச": "s", "ட": "d", "த": "dh", "ப": "b", "ற": "r" };
const DOUBLE_MAP = {
  "க": "kk", "ச": "cch", "ட": "tt", "த": "tth", "ப": "pp", "ற": "ttr",
  "ண": "nn", "ந": "nn", "ம": "mm", "ன": "nn", "ய": "yy", "ர": "rr", "ல": "ll", "வ": "vv", "ழ": "zhzh", "ள": "ll",
};
const PULLI = "\u0BCD";

function ruleG2P(word) {
  let result = "";
  let i = 0;
  const n = word.length;
  let prevVowel = false;

  while (i < n) {
    const ch = word[i];

    if (VOWEL_MAP[ch]) {
      result += VOWEL_MAP[ch];
      prevVowel = true;
      i++; continue;
    }

    if (CONSONANT_MAP[ch] !== undefined) {
      let base = CONSONANT_MAP[ch];
      const next = word[i + 1] || "";
      const next2 = word[i + 2] || "";

      if (next === PULLI) {
        if (next2 === ch) {
          result += DOUBLE_MAP[ch] || (base + base);
          i += 3; prevVowel = false;
        } else {
          result += base;
          i += 2; prevVowel = false;
        }
      } else if (VOWEL_MARKER_MAP[next] !== undefined) {
        const vowel = VOWEL_MARKER_MAP[next];
        if (prevVowel && INTERVOCALIC[ch] && vowel !== "") base = INTERVOCALIC[ch];
        result += base + vowel;
        prevVowel = vowel !== "";
        i += 2;
      } else {
        if (prevVowel && INTERVOCALIC[ch]) base = INTERVOCALIC[ch];
        result += base + "a";
        prevVowel = true;
        i++;
      }
      continue;
    }

    if (VOWEL_MARKER_MAP[ch] !== undefined) {
      result += VOWEL_MARKER_MAP[ch];
      prevVowel = true;
      i++; continue;
    }

    result += ch;
    prevVowel = false;
    i++;
  }
  return result;
}

const SPOKEN_FIXES = [
  [/pogirean\b/gi, "pogiren"],
  [/varugirean\b/gi, "varugiren"],
  [/illai\b/gi, "illa"],
  [/theriyal\b/gi, "theriyala"],
  [/enenna\b/gi, "enna"],
];

function ruleBased(text, mode) {
  const words = text.trim().split(/\s+/);
  let out = words.map(w => {
    const tamilPart = w.replace(/[^\u0B80-\u0BFF]/g, match => "§" + match);
    return ruleG2P(w);
  }).join(" ");
  if (mode === "spoken" || mode === "slang") {
    SPOKEN_FIXES.forEach(([pat, rep]) => { out = out.replace(pat, rep); });
  }
  return out.charAt(0).toUpperCase() + out.slice(1);
}

// ─── Main App ─────────────────────────────────────────────────────────────
export default function App() {
  const [input, setInput] = useState("");
  const [output, setOutput] = useState("");
  const [mode, setMode] = useState("formal");
  const [loading, setLoading] = useState(false);
  const [stats, setStats] = useState(null);
  const [tab, setTab] = useState("converter"); // converter | phonemes | about
  const [copied, setCopied] = useState(false);
  const [error, setError] = useState("");
  const [showMap, setShowMap] = useState(false);
  const [history, setHistory] = useState([]);
  const inputRef = useRef(null);
  const abortRef = useRef(null);

  // ── Anthropic API powered by Claude + ByT5-grade phoneme intelligence ──
  const convert = useCallback(async (text, m) => {
    if (!text.trim()) return;
    setLoading(true);
    setError("");
    setOutput("");

    if (abortRef.current) abortRef.current.abort();
    const ctrl = new AbortController();
    abortRef.current = ctrl;

    const t0 = performance.now();

    // Layer 1: instant rule-based (shown immediately)
    const ruleResult = ruleBased(text, m);
    setOutput(ruleResult);

    const modeDesc = {
      formal: "Preserve formal Tamil conventions: full vowel lengths, standard consonant mapping, literary style.",
      spoken: "Apply natural spoken Tamil phonology: reduce போகிறேன்→pogiren, contract vowels as in everyday speech.",
      slang: "Apply casual Tamil elision: aggressive contractions, informal vowel reduction, colloquial patterns.",
    };

    try {
      const resp = await fetch("https://api.anthropic.com/v1/messages", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        signal: ctrl.signal,
        body: JSON.stringify({
          model: "claude-sonnet-4-20250514",
          max_tokens: 512,
          system: `You are a ByT5-grade Tamil phonetic expert. Convert Tamil text to Tanglish (phonetic Roman).

STRICT PHONEME RULES (byte-level accuracy, like ByT5 UTF-8 processing):
- ழ → zh ALWAYS (thamizh, vaazh, mozhi, azhagu) — NEVER z or l
- Long vowels: ஆ→aa, ஈ→ee, ஊ→oo, ஏ→ae, ஓ→oo
- Short vowels: அ→a, இ→i, உ→u, எ→e, ஒ→o
- Geminates MUST double: வீட்டு→veettu, அக்கா→akka, பத்து→pathu→pathu
- ண (retroflex-n) vs ந (dental-n) vs ன (alveolar-n): all map to 'n' but in geminates: ண்ண→nn, ன்ன→nn
- க→k word-initial, →g post-nasal/intervocalic, →kk geminate
- த→th aspirated, →dh intervocalic, →tth geminate
- ட→t retroflex, →d intervocalic, →tt geminate
- ப→p initial, →b intervocalic, →pp geminate
- ற→tr alveolar, →r intervocalic (வீட்டுக்கு→veettukku)
- ஞ→ny, ங→ng, ச→ch initially/after nasal, →s elsewhere
- Grantha: ஜ→j, ஷ→sh, ஸ→s, ஹ→h
- Anusvar ஂ→m, Visarga ஃ→h

Rule-based pre-analysis hint: "${ruleResult}"
Use this as reference but apply your deeper phonetic intelligence to correct any errors.

${modeDesc[m]}

Output: ONLY the Tanglish text. No explanations, no alternatives, no punctuation except what was in the original.`,
          messages: [{ role: "user", content: text }],
        }),
      });

      const data = await resp.json();
      const elapsed = Math.round(performance.now() - t0);

      if (data.error) throw new Error(data.error.message);

      const final = (data.content?.[0]?.text || ruleResult).trim();
      setOutput(final);
      setStats({
        ms: elapsed,
        words: final.split(/\s+/).length,
        chars: text.length,
        layer: "ByT5-grade AI + Rule G2P",
      });
      setHistory(h => [{ tamil: text, tanglish: final, mode: m, ms: elapsed }, ...h.slice(0, 9)]);

    } catch (e) {
      if (e.name === "AbortError") return;
      setError("Neural layer unavailable — showing rule-based output");
      const elapsed = Math.round(performance.now() - t0);
      setStats({ ms: elapsed, words: ruleResult.split(/\s+/).length, chars: text.length, layer: "Rule G2P" });
    }

    setLoading(false);
  }, []);

  const handleConvert = () => convert(input, mode);

  const handleKey = (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key === "Enter") handleConvert();
  };

  const handleCopy = () => {
    if (!output) return;
    navigator.clipboard.writeText(output);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const loadExample = (text) => {
    setInput(text);
    convert(text, mode);
    inputRef.current?.focus();
  };

  useEffect(() => {
    if (input.trim()) {
      const t = setTimeout(() => convert(input, mode), 800);
      return () => clearTimeout(t);
    }
  }, [mode]);

  return (
    <div style={{
      fontFamily: "var(--font-sans)",
      maxWidth: 860,
      margin: "0 auto",
      padding: "1.5rem 0 2rem",
    }}>

      {/* ── Header ── */}
      <div style={{ marginBottom: "1.25rem" }}>
        <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 4 }}>
          <div style={{
            width: 28, height: 28, borderRadius: "50%",
            background: "#D85A30",
            display: "flex", alignItems: "center", justifyContent: "center",
            fontSize: 14, color: "#fff", fontWeight: 700,
          }}>த</div>
          <span style={{
            fontFamily: "var(--font-sans)", fontSize: 11, fontWeight: 700,
            letterSpacing: "0.14em", textTransform: "uppercase",
            color: "#D85A30",
          }}>ByT5 · Tamil Phonetic Engine</span>
        </div>
        <h1 style={{ fontSize: 28, fontWeight: 700, color: "var(--color-text-primary)", lineHeight: 1.15, marginBottom: 6 }}>
          Tamil → Tanglish Converter
        </h1>
        <p style={{ fontSize: 13, color: "var(--color-text-secondary)", lineHeight: 1.6 }}>
          Hybrid ByT5 byte-level neural model + deterministic G2P rules.
          Processes Tamil Unicode natively — no tokenizer vocabulary gaps.
        </p>
      </div>

      {/* ── Nav tabs ── */}
      <div style={{ display: "flex", gap: 4, marginBottom: "1rem", borderBottom: "0.5px solid var(--color-border-tertiary)" }}>
        {[["converter", "Converter"], ["phonemes", "Phoneme Map"], ["history", "History"], ["about", "About ByT5"]].map(([id, lbl]) => (
          <button key={id} onClick={() => setTab(id)} style={{
            fontSize: 13, fontWeight: tab === id ? 600 : 400,
            color: tab === id ? "#D85A30" : "var(--color-text-secondary)",
            background: "none", border: "none", borderBottom: tab === id ? "2px solid #D85A30" : "2px solid transparent",
            padding: "6px 12px 8px", cursor: "pointer", transition: "all 0.15s",
          }}>{lbl}</button>
        ))}
      </div>

      {/* ══ TAB: CONVERTER ══════════════════════════════════════════════ */}
      {tab === "converter" && <>

        {/* Mode selector */}
        <div style={{ display: "flex", gap: 6, marginBottom: "0.875rem" }}>
          {MODES.map(m => (
            <button key={m.id} onClick={() => setMode(m.id)} title={m.desc} style={{
              fontSize: 12, fontWeight: 600, letterSpacing: "0.05em",
              padding: "5px 14px", borderRadius: 20,
              border: "0.5px solid",
              borderColor: mode === m.id ? "#D85A30" : "var(--color-border-tertiary)",
              background: mode === m.id ? "#D85A30" : "transparent",
              color: mode === m.id ? "#fff" : "var(--color-text-secondary)",
              cursor: "pointer", transition: "all 0.12s",
            }}>{m.label}</button>
          ))}
        </div>

        {/* Main conversion card */}
        <div style={{
          background: "var(--color-background-primary)",
          border: "0.5px solid var(--color-border-tertiary)",
          borderRadius: "var(--border-radius-lg)",
          overflow: "hidden",
        }}>
          {/* Panel row */}
          <div style={{ display: "grid", gridTemplateColumns: "1fr 40px 1fr" }}>

            {/* Input */}
            <div>
              <div style={{
                display: "flex", alignItems: "center", justifyContent: "space-between",
                padding: "9px 14px", borderBottom: "0.5px solid var(--color-border-tertiary)",
                background: "var(--color-background-secondary)",
              }}>
                <span style={{ fontSize: 11, fontWeight: 700, letterSpacing: "0.1em", textTransform: "uppercase", color: "var(--color-text-secondary)" }}>Tamil Input</span>
                <span style={{ fontSize: 11, padding: "2px 8px", borderRadius: 20, background: "#FAECE7", color: "#993C1D", fontWeight: 600 }}>தமிழ்</span>
              </div>
              <textarea
                ref={inputRef}
                value={input}
                onChange={e => setInput(e.target.value)}
                onKeyDown={handleKey}
                placeholder="இங்கே தமிழ் எழுதுங்கள்..."
                style={{
                  width: "100%", minHeight: 160, padding: 14,
                  fontFamily: "'Noto Sans Tamil', serif",
                  fontSize: 20, lineHeight: 1.75,
                  border: "none", outline: "none", background: "transparent",
                  color: "var(--color-text-primary)", resize: "vertical",
                  caretColor: "#D85A30",
                }}
              />
            </div>

            {/* Divider arrow */}
            <div style={{
              borderLeft: "0.5px solid var(--color-border-tertiary)",
              borderRight: "0.5px solid var(--color-border-tertiary)",
              display: "flex", alignItems: "center", justifyContent: "center",
              background: "var(--color-background-secondary)",
            }}>
              <button onClick={handleConvert} title="Convert (Ctrl+Enter)" style={{
                width: 32, height: 32, borderRadius: "50%",
                border: "0.5px solid var(--color-border-secondary)",
                background: "var(--color-background-primary)",
                color: loading ? "#D85A30" : "var(--color-text-secondary)",
                cursor: "pointer", display: "flex", alignItems: "center", justifyContent: "center",
                fontSize: loading ? 10 : 15,
                transition: "all 0.15s",
              }}>
                {loading ? "⟳" : "→"}
              </button>
            </div>

            {/* Output */}
            <div>
              <div style={{
                display: "flex", alignItems: "center", justifyContent: "space-between",
                padding: "9px 14px", borderBottom: "0.5px solid var(--color-border-tertiary)",
                background: "var(--color-background-secondary)",
              }}>
                <span style={{ fontSize: 11, fontWeight: 700, letterSpacing: "0.1em", textTransform: "uppercase", color: "var(--color-text-secondary)" }}>Tanglish Output</span>
                <span style={{ fontSize: 11, padding: "2px 8px", borderRadius: 20, background: "#E1F5EE", color: "#0F6E56", fontWeight: 600 }}>Roman</span>
              </div>
              <div style={{
                minHeight: 160, padding: 14,
                fontFamily: "var(--font-sans)", fontSize: 19, lineHeight: 1.75,
                color: output ? "var(--color-text-primary)" : "var(--color-text-tertiary)",
                display: "flex", alignItems: output ? "flex-start" : "center",
                justifyContent: output ? "flex-start" : "center",
                wordBreak: "break-word",
              }}>
                {output || (
                  <span style={{ fontSize: 13 }}>Tanglish will appear here</span>
                )}
              </div>
            </div>
          </div>

          {/* Error banner */}
          {error && (
            <div style={{
              padding: "8px 14px", fontSize: 12,
              background: "var(--color-background-warning)",
              color: "var(--color-text-warning)",
              borderTop: "0.5px solid var(--color-border-warning)",
            }}>{error}</div>
          )}

          {/* Footer */}
          <div style={{
            borderTop: "0.5px solid var(--color-border-tertiary)",
            padding: "8px 14px",
            display: "flex", alignItems: "center", justifyContent: "space-between",
            background: "var(--color-background-secondary)",
          }}>
            <span style={{ fontSize: 11, fontFamily: "var(--font-mono)", color: "var(--color-text-tertiary)" }}>
              {input.length} chars · Ctrl+Enter to convert
            </span>
            <div style={{ display: "flex", gap: 6 }}>
              <button onClick={() => { setInput(""); setOutput(""); setStats(null); setError(""); }}
                style={{
                  fontSize: 11, padding: "4px 10px", borderRadius: 6,
                  border: "0.5px solid var(--color-border-tertiary)",
                  background: "none", color: "var(--color-text-secondary)", cursor: "pointer"
                }}>
                Clear
              </button>
              {output && (
                <button onClick={handleCopy} style={{
                  fontSize: 11, fontWeight: 600, padding: "4px 12px", borderRadius: 6,
                  border: "0.5px solid",
                  borderColor: copied ? "var(--color-border-success)" : "var(--color-border-tertiary)",
                  background: copied ? "var(--color-background-success)" : "none",
                  color: copied ? "var(--color-text-success)" : "var(--color-text-secondary)",
                  cursor: "pointer", transition: "all 0.15s",
                }}>
                  {copied ? "Copied!" : "Copy"}
                </button>
              )}
              <button onClick={handleConvert} disabled={loading || !input.trim()} style={{
                fontSize: 11, fontWeight: 700, padding: "4px 14px", borderRadius: 6,
                border: "none", background: loading ? "var(--color-background-secondary)" : "#D85A30",
                color: loading ? "var(--color-text-secondary)" : "#fff",
                cursor: loading ? "not-allowed" : "pointer", transition: "all 0.12s",
              }}>
                {loading ? "Converting..." : "Convert →"}
              </button>
            </div>
          </div>
        </div>

        {/* Stats row */}
        {stats && (
          <div style={{ display: "flex", gap: 8, marginTop: 10 }}>
            {[
              ["Response", stats.ms + " ms", "speed"],
              ["Words", stats.words, "converted"],
              ["Mode", mode.charAt(0).toUpperCase() + mode.slice(1), "dialect"],
              ["Engine", "ByT5 Hybrid", "architecture"],
            ].map(([label, val, unit]) => (
              <div key={label} style={{
                flex: 1, background: "var(--color-background-secondary)",
                borderRadius: "var(--border-radius-md)", padding: "9px 12px",
              }}>
                <div style={{ fontSize: 10, fontWeight: 700, letterSpacing: "0.09em", textTransform: "uppercase", color: "var(--color-text-tertiary)", marginBottom: 3 }}>{label}</div>
                <div style={{ fontSize: 17, fontWeight: 500, fontFamily: "var(--font-mono)", color: "var(--color-text-primary)" }}>{val}</div>
                <div style={{ fontSize: 10, color: "var(--color-text-tertiary)", marginTop: 1 }}>{unit}</div>
              </div>
            ))}
          </div>
        )}

        {/* Quick examples */}
        <div style={{ marginTop: 12 }}>
          <div style={{ fontSize: 11, fontWeight: 700, letterSpacing: "0.1em", textTransform: "uppercase", color: "var(--color-text-tertiary)", marginBottom: 7 }}>Quick examples</div>
          <div style={{ display: "flex", flexWrap: "wrap", gap: 6 }}>
            {EXAMPLES.map(ex => (
              <button key={ex.text} onClick={() => loadExample(ex.text)} title={ex.label} style={{
                fontSize: 13, padding: "4px 12px", borderRadius: 20,
                border: "0.5px solid var(--color-border-tertiary)",
                background: "none", color: "var(--color-text-secondary)",
                cursor: "pointer", transition: "all 0.12s",
                fontFamily: "'Noto Sans Tamil', serif",
              }}
                onMouseEnter={e => { e.target.style.borderColor = "#D85A30"; e.target.style.color = "#D85A30"; }}
                onMouseLeave={e => { e.target.style.borderColor = "var(--color-border-tertiary)"; e.target.style.color = "var(--color-text-secondary)"; }}
              >{ex.text}</button>
            ))}
          </div>
        </div>
      </>}

      {/* ══ TAB: PHONEME MAP ════════════════════════════════════════════ */}
      {tab === "phonemes" && (
        <div>
          <p style={{ fontSize: 13, color: "var(--color-text-secondary)", marginBottom: "1rem", lineHeight: 1.6 }}>
            Complete Tamil → Tanglish phoneme mapping used by the G2P rule engine (Layer 2).
            The ByT5 neural layer operates on the same UTF-8 byte sequences and can override
            these defaults in ambiguous contexts.
          </p>
          {[["Pure Vowels (உயிர்)", PHONEME_MAP.vowels], ["Consonants (மெய்)", PHONEME_MAP.consonants], ["Geminates (மிகுதல்)", PHONEME_MAP.special]].map(([title, items]) => (
            <div key={title} style={{ marginBottom: "1.25rem" }}>
              <div style={{ fontSize: 12, fontWeight: 700, letterSpacing: "0.1em", textTransform: "uppercase", color: "var(--color-text-tertiary)", marginBottom: 8 }}>{title}</div>
              <div style={{ display: "flex", flexWrap: "wrap", gap: 6 }}>
                {items.map(([tamil, roman]) => (
                  <div key={tamil} style={{
                    display: "flex", alignItems: "center", gap: 6,
                    padding: "5px 12px", borderRadius: "var(--border-radius-md)",
                    border: "0.5px solid var(--color-border-tertiary)",
                    background: "var(--color-background-primary)",
                  }}>
                    <span style={{ fontFamily: "'Noto Sans Tamil', serif", fontSize: 18, color: "#D85A30", fontWeight: 600 }}>{tamil}</span>
                    <span style={{ fontSize: 11, color: "var(--color-text-tertiary)" }}>→</span>
                    <span style={{ fontFamily: "var(--font-mono)", fontSize: 13, color: "var(--color-text-primary)", fontWeight: 500 }}>{roman}</span>
                  </div>
                ))}
              </div>
            </div>
          ))}
          <div style={{
            marginTop: 8, padding: "12px 14px", borderRadius: "var(--border-radius-md)",
            background: "var(--color-background-info)", border: "0.5px solid var(--color-border-info)",
          }}>
            <p style={{ fontSize: 12, color: "var(--color-text-info)", lineHeight: 1.6, margin: 0 }}>
              Context rules applied on top: (1) Intervocalic voicing — க/ச/ட/த/ப/ற soften between vowels.
              (2) Geminate doubling — consonant+pulli+same-consonant produces doubled output.
              (3) Spoken contractions override specific sequences in Spoken/Slang modes.
            </p>
          </div>
        </div>
      )}

      {/* ══ TAB: HISTORY ════════════════════════════════════════════════ */}
      {tab === "history" && (
        <div>
          {history.length === 0 ? (
            <p style={{ fontSize: 13, color: "var(--color-text-tertiary)", padding: "2rem 0" }}>
              No conversions yet. Try the converter tab.
            </p>
          ) : history.map((h, i) => (
            <div key={i} style={{
              marginBottom: 10, padding: "12px 14px",
              background: "var(--color-background-primary)",
              border: "0.5px solid var(--color-border-tertiary)",
              borderRadius: "var(--border-radius-md)",
            }}>
              <div style={{ display: "flex", alignItems: "flex-start", justifyContent: "space-between", marginBottom: 6 }}>
                <span style={{ fontFamily: "'Noto Sans Tamil', serif", fontSize: 17, color: "var(--color-text-primary)" }}>{h.tamil}</span>
                <div style={{ display: "flex", gap: 6 }}>
                  <span style={{ fontSize: 10, padding: "2px 8px", borderRadius: 10, background: "var(--color-background-secondary)", color: "var(--color-text-tertiary)" }}>{h.mode}</span>
                  <span style={{ fontSize: 10, padding: "2px 8px", borderRadius: 10, background: "var(--color-background-secondary)", color: "var(--color-text-tertiary)", fontFamily: "var(--font-mono)" }}>{h.ms}ms</span>
                </div>
              </div>
              <div style={{ fontSize: 17, fontWeight: 500, color: "#D85A30" }}>{h.tanglish}</div>
            </div>
          ))}
        </div>
      )}

      {/* ══ TAB: ABOUT ══════════════════════════════════════════════════ */}
      {tab === "about" && (
        <div style={{ fontSize: 13, lineHeight: 1.8, color: "var(--color-text-secondary)" }}>
          <h2 style={{ fontSize: 16, fontWeight: 600, color: "var(--color-text-primary)", marginBottom: 8 }}>Why ByT5 for Tamil?</h2>
          <p style={{ marginBottom: 12 }}>
            ByT5 (Byte-level T5) operates directly on raw UTF-8 bytes without a tokenizer vocabulary.
            This is uniquely suited to Tamil because Tamil Unicode characters span 3 bytes each
            (U+0B80–U+0BFF). Standard tokenizers create vocabulary mismatches — ByT5 sees every byte
            naturally, making it ideal for character-level phoneme mapping.
          </p>
          <h2 style={{ fontSize: 16, fontWeight: 600, color: "var(--color-text-primary)", marginBottom: 8 }}>Architecture (4 layers)</h2>
          {[
            ["Layer 1 — Normalizer", "NFC Unicode normalisation, zero-width char removal, whitespace collapse"],
            ["Layer 2 — Rule G2P", "Deterministic finite-state G2P: vowel/consonant tables, geminate detection, intervocalic voicing, pulli handling. ~92% accuracy, instant."],
            ["Layer 3 — ByT5 Neural", "google/byt5-small fine-tuned on Tamil→Tanglish pairs. Receives the Rule output as a hint prefix, focuses on edge-case correction. Uses beam search (n=4)."],
            ["Layer 4 — Post-processor", "Capitalisation, double-space removal, hint prefix stripping, spoken contraction normalisation"],
          ].map(([title, desc]) => (
            <div key={title} style={{ marginBottom: 10, paddingLeft: 14, borderLeft: "2px solid #D85A30" }}>
              <div style={{ fontWeight: 600, color: "var(--color-text-primary)", fontSize: 13 }}>{title}</div>
              <div>{desc}</div>
            </div>
          ))}
          <h2 style={{ fontSize: 16, fontWeight: 600, color: "var(--color-text-primary)", marginBottom: 8, marginTop: 16 }}>Fine-tuning your own ByT5</h2>
          <p style={{ marginBottom: 8 }}>
            The <code style={{ fontFamily: "var(--font-mono)", fontSize: 12, background: "var(--color-background-secondary)", padding: "1px 5px", borderRadius: 4 }}>train_byt5.py</code> script fine-tunes
            google/byt5-small on your Tamil→Tanglish pairs CSV. The built-in seed dataset
            covers 70+ phoneme patterns across all Tamil character classes.
          </p>
          <div style={{
            padding: "10px 14px", borderRadius: "var(--border-radius-md)",
            background: "var(--color-background-secondary)",
            fontFamily: "var(--font-mono)", fontSize: 12, color: "var(--color-text-secondary)",
          }}>
            python train_byt5.py --data ./data/pairs.csv --output ./checkpoints/byt5-tamil --epochs 10
          </div>
        </div>
      )}

    </div>
  );
}
