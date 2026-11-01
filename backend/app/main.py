from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from app.api import auth, audio, analysis, ws_audio
from app.core.database import connect_db, disconnect_db
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="EchoShield API",
    description="Real-Time AI Scammer Interceptor Backend",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


_db_initialized = False


@app.middleware("http")
async def ensure_db_middleware(request, call_next):
    global _db_initialized
    if not _db_initialized:
        try:
            await connect_db()
        except Exception:
            pass
        _db_initialized = True
    return await call_next(request)


@app.on_event("startup")
async def startup_event():
    global _db_initialized
    try:
        await connect_db()
        _db_initialized = True
    except Exception as e:
        logger.warning(f"Database connection failed: {e}. Continuing in mock mode.")
    logger.info("EchoShield backend started.")


@app.on_event("shutdown")
async def shutdown_event():
    await disconnect_db()
    logger.info("EchoShield backend stopped.")


# Mount endpoints at root
app.include_router(auth.router, prefix="/auth", tags=["Authentication"])
app.include_router(audio.router, tags=["Audio Processing"])
app.include_router(analysis.router, tags=["Analysis"])
app.include_router(ws_audio.router, tags=["WebSocket Streaming"])

# Mount endpoints under /api to support both Vercel rewrites and direct API calls
app.include_router(auth.router, prefix="/api/auth", tags=["Authentication (API)"])
app.include_router(audio.router, prefix="/api", tags=["Audio Processing (API)"])
app.include_router(analysis.router, prefix="/api", tags=["Analysis (API)"])


@app.get("/health", tags=["System"])
@app.get("/api/health", tags=["System"])
async def health_check():
    return {"status": "ok", "service": "EchoShield", "version": "1.0.0"}


# ── Serve web dashboard ──────────────────────────────────────────────────────
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.abspath(os.path.join(_BASE_DIR, "..", ".."))
_CANDIDATE_PATHS = [
    os.path.join(_ROOT_DIR, "public", "index.html"),
    os.path.join(_ROOT_DIR, "web-dashboard", "index.html"),
    os.path.join(_BASE_DIR, "..", "..", "web-dashboard", "index.html"),
    os.path.join(_BASE_DIR, "..", "..", "public", "index.html"),
]


@app.get("/", tags=["Dashboard"])
async def serve_dashboard():
    """Serve the web dashboard."""
    for path in _CANDIDATE_PATHS:
        if os.path.exists(path):
            return FileResponse(path, media_type="text/html")
    return {"message": "EchoShield API is running. Visit /docs for API documentation."}

