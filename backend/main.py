from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import logging

from byt5_engine import get_engine, normalize_tamil
from hindi_engine import get_hindi_engine
from malayalam_engine import get_malayalam_engine
from song_engine import get_song_engine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="TamTan API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled error on {request.url}: {exc}")
    return JSONResponse(status_code=500, content={"detail": str(exc)})


class ConvertRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=2000)
    mode: str = Field("formal")


class HindiRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=2000)


@app.on_event("startup")
async def startup_event():
    logger.info("Pre-loading ByT5 engine...")
    get_engine()
    logger.info("Engine ready.")


@app.post("/convert")
async def convert(req: ConvertRequest):
    if req.mode not in ("formal", "spoken", "slang"):
        raise HTTPException(400, "mode must be formal|spoken|slang")
    engine = get_engine()
    result = engine.convert(req.text, mode=req.mode)
    result["input_cleaned"] = normalize_tamil(req.text)
    return JSONResponse(content=result)


@app.post("/hindi/convert")
async def hindi_convert(req: HindiRequest):
    """
    Bidirectional Hindi ↔ Hinglish transliteration.
    Fully offline — rule-based engine, no API key required.
    Auto-detects script direction.
    """
    engine = get_hindi_engine()
    result = engine.convert(req.text)
    return JSONResponse(content=result)


class MalayalamRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=2000)


@app.post("/malayalam/convert")
async def malayalam_convert(req: MalayalamRequest):
    """
    Bidirectional Malayalam ↔ Manglish transliteration.
    Fully offline — rule-based engine, no API key required.
    Auto-detects script direction.
    """
    engine = get_malayalam_engine()
    result = engine.convert(req.text)
    return JSONResponse(content=result)


class SongRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000)


@app.post("/song/convert")
async def song_convert(req: SongRequest):
    engine = get_song_engine()
    result = engine.convert(req.text)
    return JSONResponse(content=result)


@app.get("/health")
async def health():
    tamil_engine = get_engine()
    hindi_engine = get_hindi_engine()
    mal_engine   = get_malayalam_engine()
    song_engine  = get_song_engine()
    return {
        "status": "ok",
        "tamil_neural_ready":     tamil_engine._ready,
        "hindi_neural_ready":     hindi_engine._ready,
        "malayalam_neural_ready": mal_engine._ready,
        "song_neural_ready":      song_engine._ready,
        "tamil_model":     tamil_engine.MODEL_ID,
        "hindi_model":     hindi_engine.MODEL_ID  if hindi_engine._ready else "rule-based",
        "malayalam_model": mal_engine.MODEL_ID    if mal_engine._ready   else "rule-based",
        "song_model":      song_engine.MODEL_ID   if song_engine._ready  else "rule-based",
    }
