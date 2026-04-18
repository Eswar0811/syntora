# Syntora

Real-time Spotify lyrics transliterated into Tamil, Hindi, Malayalam, and Telugu — live, as you listen.

![Syntora](frontend-next/public/syntora-logo.png)

## What it does

Connect your Spotify account and Syntora instantly shows your currently playing song transliterated into all four major Indic languages — in both native script and romanised form. Switch between languages with a single tap.

- **Tamil** — தமிழ் + Tanglish
- **Hindi** — हिन्दी + Hinglish
- **Malayalam** — മലയാളം + Manglish
- **Telugu** — తెలుగు + Tenglish

## Live Demo

**[syntora-app.vercel.app](https://syntora-app.vercel.app)**

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | Next.js 15, TypeScript, Tailwind CSS |
| Backend | FastAPI (Python 3.11), Uvicorn |
| Transliteration | Rule-based engines (Tamil, Hindi, Malayalam, Telugu) + optional ByT5 neural layer |
| Lyrics | Curated DB → lrclib synced lyrics → Whisper audio fallback |
| Auth | Spotify OAuth 2.0 |
| Deployment | Vercel (frontend) + Render (backend) |

## Project Structure

```
syntora/
├── backend/                  # FastAPI backend
│   ├── main.py               # App entry point, routes, lifespan
│   ├── byt5_engine.py        # Tamil → Tanglish (rule-based + optional ByT5)
│   ├── hindi_engine.py       # Hindi ↔ Hinglish
│   ├── malayalam_engine.py   # Malayalam ↔ Manglish
│   ├── telugu_engine.py      # Telugu ↔ Tenglish
│   ├── song_engine.py        # Song-optimised transliteration
│   ├── lyrics.py             # lrclib lyrics fetcher
│   ├── song_lyrics_db.py     # Curated lyrics database
│   ├── spotify.py            # Spotify API client
│   ├── audio_engine.py       # Whisper audio transcription fallback
│   ├── translator.py         # Unified translation layer
│   ├── requirements.txt      # Production deps (no ML/Whisper)
│   ├── requirements-neural.txt  # Full deps with torch + transformers
│   └── render.yaml           # Render deployment config
│
├── frontend-next/            # Next.js frontend (production)
│   ├── app/                  # Next.js App Router
│   ├── components/           # React components
│   │   ├── MainApp.tsx       # Root app shell + header
│   │   ├── ConnectCard.tsx   # Spotify connect screen
│   │   ├── LiveLyrics.tsx    # Live lyrics container
│   │   ├── NowPlaying.tsx    # Now playing card
│   │   ├── LangTabs.tsx      # Language selector tabs
│   │   ├── LyricsDisplay.tsx # Script + romanised display
│   │   └── SourceBadge.tsx   # Lyrics source indicator
│   ├── lib/                  # Hooks, types, constants, API
│   └── vercel.json           # Vercel deployment config
│
└── frontend/                 # Legacy Vite/React frontend
```

## Local Development

### Prerequisites

- Python 3.11+
- Node.js 18+
- A Spotify app — create one at [developer.spotify.com/dashboard](https://developer.spotify.com/dashboard)

### 1. Clone the repo

```bash
git clone https://github.com/Eswar0811/syntora.git
cd syntora
```

### 2. Backend setup

```bash
cd backend
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Copy the env example and fill in your Spotify credentials:

```bash
cp .env.example .env
```

```env
SPOTIFY_CLIENT_ID=your_client_id
SPOTIFY_CLIENT_SECRET=your_client_secret
SPOTIFY_REDIRECT_URI=http://127.0.0.1:8000/callback
FRONTEND_URL=http://localhost:3000
ALLOWED_ORIGINS=http://localhost:3000
```

Start the backend:

```bash
uvicorn main:app --reload --port 8000
```

### 3. Frontend setup

```bash
cd frontend-next
npm install
cp .env.local.example .env.local
```

```env
BACKEND_URL=http://localhost:8000
```

Start the dev server:

```bash
npm run dev
```

Open [http://localhost:3000](http://localhost:3000).

### 4. Spotify OAuth (local)

In your [Spotify Developer Dashboard](https://developer.spotify.com/dashboard), add this as a Redirect URI:

```
http://127.0.0.1:8000/callback
```

## Deployment

### Backend — Render

1. Connect your GitHub repo to [render.com](https://render.com)
2. Set **Root Directory** → `backend`
3. **Build Command** → `pip install -r requirements.txt`
4. **Start Command** → `uvicorn main:app --host 0.0.0.0 --port $PORT --workers 1`
5. Add environment variables in the Render dashboard:

| Key | Value |
|---|---|
| `SPOTIFY_CLIENT_ID` | From Spotify Dashboard |
| `SPOTIFY_CLIENT_SECRET` | From Spotify Dashboard |
| `SPOTIFY_REDIRECT_URI` | `https://your-backend.onrender.com/callback` |
| `FRONTEND_URL` | `https://your-app.vercel.app` |
| `ALLOWED_ORIGINS` | `https://your-app.vercel.app` |

### Frontend — Vercel

1. Import repo on [vercel.com](https://vercel.com)
2. Set **Root Directory** → `frontend-next`
3. Add environment variable:

| Key | Value |
|---|---|
| `BACKEND_URL` | `https://your-backend.onrender.com` |

### Spotify Redirect URI (production)

In your Spotify Developer Dashboard → Edit Settings → Redirect URIs, add:

```
https://your-backend.onrender.com/callback
```

## Neural Mode (optional)

By default, Syntora runs fully rule-based transliteration — fast, zero RAM overhead, works on free hosting tiers.

To enable the ByT5 neural correction layer (requires ≥2GB RAM):

```bash
pip install -r requirements-neural.txt
```

The engine automatically detects torch and activates the neural layer on startup.

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Health check |
| `POST` | `/transliterate` | Tamil → Tanglish |
| `POST` | `/hindi` | Hindi ↔ Hinglish |
| `POST` | `/malayalam` | Malayalam ↔ Manglish |
| `POST` | `/telugu` | Telugu ↔ Tenglish |
| `GET` | `/spotify/auth-url` | Get Spotify OAuth URL |
| `GET` | `/spotify/current` | Get currently playing track with transliteration |
| `POST` | `/spotify/logout` | Clear Spotify session |
| `GET` | `/callback` | Spotify OAuth callback |

## License

MIT
