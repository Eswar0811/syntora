import { useState } from 'react'
import './App.css'

const API = 'http://localhost:8000'

const INPUT_LANGS = [
  { id: 'tamil',     label: 'Tamil',     native: 'தமிழ்',    hint: 'வணக்கம், நீங்கள் எப்படி இருக்கீங்க?' },
  { id: 'hindi',     label: 'Hindi',     native: 'हिन्दी',   hint: 'नमस्ते, आप कैसे हैं?  or  namaste, aap kaise hain?' },
  { id: 'malayalam', label: 'Malayalam', native: 'മലയാളം',  hint: 'നമസ്കാരം, സുഖമാണോ?  or  namaskaram, sukhamaano?' },
]

const MONO_TARGETS = [
  { id: 'hindi',     label: 'Hindi',     native: 'हिन्दी',  script: 'Devanagari' },
  { id: 'malayalam', label: 'Malayalam', native: 'മലയാളം', script: 'Malayalam' },
  { id: 'tamil',     label: 'Tamil',     native: 'தமிழ்',   script: 'Tamil' },
]

const BI_TARGETS = [
  { id: 'tanglish', label: 'Tanglish', sub: 'Tamil · English',     lang: 'tamil' },
  { id: 'hinglish', label: 'Hinglish', sub: 'Hindi · English',     lang: 'hindi' },
  { id: 'manglish', label: 'Manglish', sub: 'Malayalam · English', lang: 'malayalam' },
]

function supported(inputLang, outputMode, target) {
  if (outputMode === 'bilingual') {
    if (target === 'tanglish') return inputLang === 'tamil'
    if (target === 'hinglish') return inputLang === 'hindi'
    if (target === 'manglish') return inputLang === 'malayalam'
  } else {
    if (target === 'tamil')     return false
    if (target === 'hindi')     return inputLang === 'tamil' || inputLang === 'hindi'
    if (target === 'malayalam') return inputLang === 'tamil' || inputLang === 'malayalam'
  }
  return false
}

async function callAPI(inputLang, outputMode, target, text, mode) {
  if (outputMode === 'bilingual') {
    if (target === 'tanglish') {
      const r = await fetch(`${API}/convert`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text, mode }),
      })
      if (!r.ok) throw new Error((await r.json()).detail || 'Conversion failed')
      const d = await r.json()
      return { output: d.tanglish, meta: d }
    }
    if (target === 'hinglish') {
      const r = await fetch(`${API}/hindi/convert`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text }),
      })
      if (!r.ok) throw new Error(((await r.json().catch(() => ({}))).detail) || 'Conversion failed')
      const d = await r.json()
      return { output: d.output, meta: d }
    }
    if (target === 'manglish') {
      const r = await fetch(`${API}/malayalam/convert`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text }),
      })
      if (!r.ok) throw new Error(((await r.json().catch(() => ({}))).detail) || 'Conversion failed')
      const d = await r.json()
      return { output: d.output, meta: d }
    }
  } else {
    const r = await fetch(`${API}/song/convert`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text }),
    })
    if (!r.ok) throw new Error(((await r.json().catch(() => ({}))).detail) || 'Conversion failed')
    const d = await r.json()
    if (target === 'hindi')     return { output: d.hindi,     meta: { ...d, layer: d.hindi_layer } }
    if (target === 'malayalam') return { output: d.malayalam, meta: { ...d, layer: d.malayalam_layer } }
  }
  throw new Error('Unsupported conversion')
}

function autoSelect(lang) {
  if (lang === 'tamil')     return { outputMode: 'monolingual', monoTarget: 'hindi',   biTarget: 'tanglish' }
  if (lang === 'hindi')     return { outputMode: 'bilingual',   monoTarget: 'hindi',   biTarget: 'hinglish' }
  if (lang === 'malayalam') return { outputMode: 'bilingual',   monoTarget: 'malayalam', biTarget: 'manglish' }
  return { outputMode: 'monolingual', monoTarget: 'hindi', biTarget: 'tanglish' }
}

export default function App() {
  const [inputLang,  setInputLang]  = useState('tamil')
  const [outputMode, setOutputMode] = useState('monolingual')
  const [monoTarget, setMonoTarget] = useState('hindi')
  const [biTarget,   setBiTarget]   = useState('tanglish')
  const [mode,       setMode]       = useState('formal')
  const [text,       setText]       = useState('')
  const [result,     setResult]     = useState(null)
  const [loading,    setLoading]    = useState(false)
  const [error,      setError]      = useState('')

  const target = outputMode === 'bilingual' ? biTarget : monoTarget
  const canConvert = !loading && text.trim().length > 0 && supported(inputLang, outputMode, target)

  function switchInput(lang) {
    setInputLang(lang)
    setResult(null)
    setError('')
    const sel = autoSelect(lang)
    setOutputMode(sel.outputMode)
    setMonoTarget(sel.monoTarget)
    setBiTarget(sel.biTarget)
  }

  async function handleConvert() {
    if (!canConvert) return
    setLoading(true)
    setError('')
    setResult(null)
    try {
      setResult(await callAPI(inputLang, outputMode, target, text.trim(), mode))
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }

  const maxLen = (inputLang === 'tamil' && outputMode === 'monolingual') ? 5000 : 2000
  const outLabel = outputMode === 'bilingual'
    ? BI_TARGETS.find(t => t.id === biTarget)?.label
    : MONO_TARGETS.find(t => t.id === monoTarget)?.label + ' Script'

  return (
    <div className="app">
      {/* ── Header ── */}
      <header className="site-header">
        <div className="brand">
          <div className={`brand-icon ${inputLang}-icon`}>
            {inputLang === 'tamil' ? 'த' : inputLang === 'hindi' ? 'ह' : 'മ'}
          </div>
          <div className="brand-text">
            <span className="brand-name">TamTan</span>
            <span className="brand-sub">Indic Transliterator</span>
          </div>
        </div>
      </header>

      <main className="main">
        {/* ── Converter Card ── */}
        <div className="converter-card">

          {/* Input language strip */}
          <div className="section-row">
            <span className="row-label">Input</span>
            <div className="lang-strip">
              {INPUT_LANGS.map(l => (
                <button
                  key={l.id}
                  className={`lang-chip ${l.id}-chip ${inputLang === l.id ? 'active' : ''}`}
                  onClick={() => switchInput(l.id)}
                >
                  <span className="chip-native">{l.native}</span>
                  <span className="chip-label">{l.label}</span>
                </button>
              ))}
            </div>
          </div>

          {/* Textarea */}
          <div className="textarea-wrap">
            <textarea
              className={`main-textarea ${inputLang}-caret`}
              placeholder={INPUT_LANGS.find(l => l.id === inputLang)?.hint}
              value={text}
              onChange={e => setText(e.target.value)}
              onKeyDown={e => { if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) handleConvert() }}
              rows={5}
              maxLength={maxLen}
            />
            <div className="char-count">{text.length}<span>/{maxLen}</span></div>
          </div>

          {/* Convert-to section */}
          <div className="section-row convert-to-row">
            <span className="row-label">Convert to</span>

            <div className="target-grid">
              {/* Monolingual targets */}
              {MONO_TARGETS.map(t => {
                const avail = supported(inputLang, 'monolingual', t.id)
                return (
                  <button
                    key={t.id}
                    className={`target-chip ${t.id}-chip mono-chip
                      ${outputMode === 'monolingual' && monoTarget === t.id ? 'active' : ''}
                      ${!avail ? 'disabled' : ''}`}
                    onClick={() => {
                      if (!avail) return
                      setOutputMode('monolingual')
                      setMonoTarget(t.id)
                      setResult(null)
                    }}
                    disabled={!avail}
                    title={!avail ? 'Not supported for this input language' : t.script}
                  >
                    <span className="chip-native">{t.native}</span>
                    <span className="chip-label">{t.label}</span>
                  </button>
                )
              })}

              {/* Bilingual toggle */}
              <button
                className={`target-chip bilingual-chip ${outputMode === 'bilingual' ? 'active' : ''}`}
                onClick={() => { setOutputMode('bilingual'); setResult(null) }}
              >
                <span className="chip-native">AB</span>
                <span className="chip-label">Bilingual</span>
              </button>
            </div>

            {/* Bilingual sub-targets */}
            {outputMode === 'bilingual' && (
              <div className="bi-strip">
                {BI_TARGETS.map(t => {
                  const avail = supported(inputLang, 'bilingual', t.id)
                  return (
                    <button
                      key={t.id}
                      className={`bi-chip ${t.lang}-chip ${biTarget === t.id ? 'active' : ''} ${!avail ? 'disabled' : ''}`}
                      onClick={() => { if (avail) { setBiTarget(t.id); setResult(null) } }}
                      disabled={!avail}
                    >
                      <span className="bi-name">{t.label}</span>
                      <span className="bi-sub">{t.sub}</span>
                    </button>
                  )
                })}
              </div>
            )}
          </div>

          {/* Mode selector – only Tamil→Tanglish */}
          {inputLang === 'tamil' && outputMode === 'bilingual' && biTarget === 'tanglish' && (
            <div className="section-row mode-row">
              <span className="row-label">Style</span>
              <div className="mode-strip">
                {['formal', 'spoken', 'slang'].map(m => (
                  <button
                    key={m}
                    className={`mode-chip ${mode === m ? 'active' : ''}`}
                    onClick={() => setMode(m)}
                  >
                    {m.charAt(0).toUpperCase() + m.slice(1)}
                  </button>
                ))}
              </div>
            </div>
          )}

          {/* Convert button */}
          <div className="action-row">
            <button
              className={`go-btn ${inputLang}-btn ${!canConvert ? 'go-disabled' : ''}`}
              onClick={handleConvert}
              disabled={!canConvert}
            >
              {loading
                ? <><span className="spinner" /> Converting…</>
                : <><span>Convert</span><kbd>⌘↵</kbd></>
              }
            </button>
          </div>
        </div>

        {/* ── Error ── */}
        {error && (
          <div className="result-card error-card">
            <span className="err-icon">⚠</span>
            <span>{error}</span>
          </div>
        )}

        {/* ── Result ── */}
        {result && (
          <div className={`result-card output-card ${inputLang}-card`}>
            <div className="result-header">
              <span className={`result-tag ${inputLang}-tag`}>{outLabel}</span>
            </div>
            <div className="result-text">{result.output}</div>
            <div className="result-meta">
              {result.meta?.layer && <span className="meta-badge">{result.meta.layer}</span>}
              {result.meta?.time_ms !== undefined && (
                <span className="meta-badge time-badge">{result.meta.time_ms} ms</span>
              )}
              {result.meta?.word_count !== undefined && (
                <span className="meta-badge">{result.meta.word_count} words</span>
              )}
              {result.meta?.direction && (
                <span className="meta-badge dir-badge">{result.meta.direction.replace(/_/g, ' ')}</span>
              )}
              {result.meta?.model && (
                <span className="meta-badge model-badge">{result.meta.model}</span>
              )}
            </div>
          </div>
        )}
      </main>
    </div>
  )
}
