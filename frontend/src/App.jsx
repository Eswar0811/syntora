import { useState, useEffect, useRef } from 'react'
import './App.css'

const API = '/api'

const LANG_DISPLAY = [
  { key: 'tamil',     romanKey: 'tanglish', label: 'Tamil',     native: 'தமிழ்',   cls: 'tamil' },
  { key: 'hindi',     romanKey: 'hinglish', label: 'Hindi',     native: 'हिन्दी',  cls: 'hindi' },
  { key: 'malayalam', romanKey: 'manglish', label: 'Malayalam', native: 'മലയാളം',  cls: 'malayalam' },
  { key: 'telugu',    romanKey: 'tenglish', label: 'Telugu',    native: 'తెలుగు',  cls: 'telugu' },
]

const DETECTED_TO_LANG = { ta: 'tamil', hi: 'hindi', ml: 'malayalam', te: 'telugu' }

const SOURCE_LABEL = {
  curated: { icon: '★', text: 'Curated lyrics' },
  whisper: { icon: '🎙', text: 'Audio transcription (Whisper)' },
}

const POLL_PLAYING = 1000
const POLL_IDLE    = 3000
const POLL_ERROR   = 6000

function LiveLyrics({ authStatus, authError, redirectUri, onConnect, onDisconnect }) {
  const authed = authStatus === 'authed'
  const [playing,  setPlaying]  = useState(null)
  const [pollErr,  setPollErr]  = useState('')

  const timerRef    = useRef(null)
  const inflightRef = useRef(false)
  const abortRef    = useRef(null)

  useEffect(() => {
    if (!authed) {
      clearTimeout(timerRef.current)
      abortRef.current?.abort()
      setPlaying(null)
      setPollErr('')
      return
    }

    async function doPoll() {
      if (inflightRef.current) return
      inflightRef.current = true
      const controller = new AbortController()
      abortRef.current = controller

      try {
        const r = await fetch(`${API}/spotify/current`, { signal: controller.signal })
        if (r.status === 401) { onDisconnect(); return }
        if (r.status === 429) { timerRef.current = setTimeout(doPoll, POLL_ERROR * 2); return }
        if (!r.ok) throw new Error(`Server error ${r.status}`)

        const data = await r.json()
        setPollErr('')
        setPlaying(prev => {
          if (!data.is_playing) return null
          if (!prev || prev.track_id !== data.track_id) return data
          if (prev.original === data.original) return prev
          return { ...prev, ...data }
        })
        timerRef.current = setTimeout(doPoll, data.is_playing ? POLL_PLAYING : POLL_IDLE)
      } catch (e) {
        if (e.name === 'AbortError') return
        setPollErr(e.message)
        timerRef.current = setTimeout(doPoll, POLL_ERROR)
      } finally {
        inflightRef.current = false
      }
    }

    doPoll()
    return () => { clearTimeout(timerRef.current); abortRef.current?.abort() }
  }, [authed, onDisconnect])

  if (!authed) {
    return (
      <div className="lyrics-panel lyrics-center">
        <div className="spotify-connect-card">
          <img src="/syntora-logo.png" alt="Syntora" className="connect-card-logo" />
          <h2 className="spotify-title">Your Song, Your Vibe!</h2>
          <p className="spotify-desc">
            Connect Spotify to any song and Syntora instantly transliterates it
            into Tamil, Hindi, Malayalam and Telugu — script and romanised.
          </p>
          {authStatus === 'error' && (
            <div className="spotify-err">
              {authError === 'access_denied' && (
                <p>You denied access. Click Connect and approve the permissions to continue.</p>
              )}
              {authError === 'exchange_failed' && (
                <p>
                  Token exchange failed — your <strong>Client ID / Secret</strong> may be wrong, or{' '}
                  <code>{redirectUri}</code> is not registered in your{' '}
                  <a href="https://developer.spotify.com/dashboard" target="_blank" rel="noreferrer">
                    Spotify Developer Dashboard
                  </a>. Add it as a Redirect URI and try again.
                </p>
              )}
              {authError === 'cannot_reach_server' && (
                <p>Cannot reach the backend server. Make sure it is running on port 8000.</p>
              )}
              {(!authError || !['access_denied', 'exchange_failed', 'cannot_reach_server'].includes(authError)) && (
                <p>
                  Authentication failed ({authError || 'unknown'}). Make sure{' '}
                  <code>{redirectUri}</code> is added as a Redirect URI in your{' '}
                  <a href="https://developer.spotify.com/dashboard" target="_blank" rel="noreferrer">
                    Spotify Developer Dashboard
                  </a>, then try again.
                </p>
              )}
            </div>
          )}
          <button className="spotify-btn" onClick={onConnect} disabled={authStatus === 'exchanging'}>
            {authStatus === 'exchanging'
              ? <><span className="spinner" /> Connecting…</>
              : 'Connect Spotify'}
          </button>
        </div>
      </div>
    )
  }

  return (
    <div className="lyrics-panel">
      <div className="lyrics-toolbar">
        <span className="lyrics-connected-badge">● Connected</span>
        <button className="lyrics-disconnect-btn" onClick={onDisconnect}>Disconnect</button>
      </div>

      {pollErr && (
        <div className="result-card error-card" style={{ marginBottom: '1rem' }}>
          <span className="err-icon">⚠</span>
          <span>{pollErr} — retrying…</span>
        </div>
      )}

      {!playing ? (
        <div className="lyrics-idle">
          <span className="lyrics-idle-icon">⏸</span>
          <p>Nothing playing on Spotify right now.</p>
          <p className="lyrics-idle-sub">Start a song and it will appear here.</p>
        </div>
      ) : (
        <NowPlaying key={playing.track_id || playing.song} data={playing} />
      )}
    </div>
  )
}

function NowPlaying({ data }) {
  const initLang = DETECTED_TO_LANG[data.detected_language] || 'tamil'
  const [activeLang, setActiveLang] = useState(initLang)

  const mins = Math.floor(data.progress_seconds / 60)
  const secs = String(Math.floor(data.progress_seconds % 60)).padStart(2, '0')
  const src  = SOURCE_LABEL[data.source] ?? SOURCE_LABEL.curated
  const hasLyrics = Boolean(data.original)

  const lc     = LANG_DISPLAY.find(l => l.key === activeLang) || LANG_DISPLAY[0]
  const script = data[lc.key]      || ''
  const roman  = data[lc.romanKey] || ''

  return (
    <div className="lyrics-now-playing">
      <div className="lyrics-song-info">
        <div className="lyrics-song-row">
          <span className="lyrics-song-name">{data.song}</span>
          <span className="lyrics-progress">{mins}:{secs}</span>
        </div>
        <div className="lyrics-meta-row">
          <span className="lyrics-artist">{data.artist}</span>
          <span className={`source-badge source-${data.source}`}>
            {src.icon} {src.text}
            {data.detected_label ? ` · ${data.detected_label}` : ''}
          </span>
        </div>
      </div>

      <div className="lyrics-lang-tabs">
        {LANG_DISPLAY.map(l => (
          <button
            key={l.key}
            className={`lyrics-ltab ${l.cls}-ltab ${activeLang === l.key ? 'active' : ''}`}
            onClick={() => setActiveLang(l.key)}
          >
            <span className="ltab-native">{l.native}</span>
            <span className="ltab-label">{l.label}</span>
          </button>
        ))}
      </div>

      {hasLyrics ? (
        (script || roman) ? (
          <div className={`lyrics-display ${lc.cls}-display`}>
            <div className="lyrics-display-script">{script || roman}</div>
            {roman && roman !== script && (
              <div className="lyrics-display-roman">{roman}</div>
            )}
          </div>
        ) : (
          <div className="lyrics-waiting-block">
            <div className="lyrics-waiting-icon">◌</div>
            <div className="lyrics-waiting-text">Processing…</div>
          </div>
        )
      ) : (
        <div className="lyrics-waiting-block">
          <div className="lyrics-waiting-icon">◌</div>
          <div className="lyrics-waiting-text">
            {data.source === 'whisper'
              ? 'Transcribing audio…'
              : data.no_preview
                ? 'No audio preview available for this track'
                : 'Waiting for lyrics…'}
          </div>
        </div>
      )}
    </div>
  )
}

export default function App() {
  const [authStatus,  setAuthStatus]  = useState('idle')
  const [authError,   setAuthError]   = useState('')
  const [redirectUri, setRedirectUri] = useState('http://127.0.0.1:8000/callback')

  useEffect(() => {
    const params  = new URLSearchParams(window.location.search)
    const spotify = params.get('spotify')
    if (!spotify) return
    const reason = params.get('reason') || ''
    window.history.replaceState({}, '', '/')
    if (spotify === 'connected') {
      setAuthStatus('authed')
    } else {
      setAuthError(reason)
      setAuthStatus('error')
    }
  }, [])

  function handleConnect() {
    setAuthStatus('exchanging')
    fetch(`${API}/spotify/auth-url`)
      .then(r => { if (!r.ok) throw new Error('server_error'); return r.json() })
      .then(d => {
        if (d.redirect_uri) setRedirectUri(d.redirect_uri)
        window.location.href = d.url
      })
      .catch(() => { setAuthError('cannot_reach_server'); setAuthStatus('error') })
  }

  async function handleDisconnect() {
    fetch(`${API}/spotify/logout`, { method: 'POST' }).catch(() => {})
    setAuthStatus('idle')
    setAuthError('')
  }

  return (
    <div className="app">
      <header className="site-header">
        <div className="brand">
          <img src="/syntora-logo.png" alt="Syntora" className="syntora-logo" />
          <div className="brand-text">
            <span className="brand-name">Syntora</span>
            <span className="brand-sub">Your Song, Your Vibe!</span>
          </div>
        </div>
      </header>

      <LiveLyrics
        authStatus={authStatus}
        authError={authError}
        redirectUri={redirectUri}
        onConnect={handleConnect}
        onDisconnect={handleDisconnect}
      />
      <div className="syntora-watermark">Syntora</div>
    </div>
  )
}
