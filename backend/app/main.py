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
    redirect_slashes=False,
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
async def ensure_db_and_routing_middleware(request, call_next):
    global _db_initialized
    if not _db_initialized:
        try:
            await connect_db()
        except Exception:
            pass
        _db_initialized = True

    # 1. Resolve path from query parameter (injected by Vercel rewrites: ?path=...)
    path_param = request.query_params.get("path")
    if path_param:
        target_path = "/" + path_param.lstrip("/")
        request.scope["path"] = target_path

        # Clean 'path' query parameter from request scope to avoid polluting handler arguments
        try:
            from urllib.parse import parse_qs, urlencode
            raw_qs = request.scope.get("query_string", b"").decode("latin-1")
            if "path=" in raw_qs:
                params = parse_qs(raw_qs, keep_blank_values=True)
                params.pop("path", None)
                request.scope["query_string"] = urlencode(params, doseq=True).encode("latin-1")
        except Exception:
            pass
    else:
        # 2. Normalize Vercel serverless rewritten paths from proxy headers
        matched_path = (
            request.headers.get("x-matched-path")
            or request.headers.get("x-forwarded-uri")
            or request.headers.get("x-invoke-path")
            or request.headers.get("x-original-uri")
        )
        if matched_path:
            clean_path = matched_path.split("?")[0]
            curr_path = request.scope.get("path", "")
            if curr_path in ["/api/index.py", "/api", "/api/"]:
                request.scope["path"] = clean_path
            elif clean_path in ["/analyze-audio", "/stream-audio", "/detect-voice", "/analyze", "/call-logs", "/stats", "/health", "/docs", "/openapi.json"]:
                request.scope["path"] = clean_path
        elif request.scope.get("path", "").endswith("index.py"):
            request.scope["path"] = request.scope["path"].replace("/index.py", "")

    return await call_next(request)


@app.on_event("startup")
async def startup_event():
    global _db_initialized
    # Initialize default deepfake ML model path if trained model checkpoint exists
    ml_ckpt = os.path.join(os.path.dirname(__file__), "ml", "voice_classifier.pkl")
    if os.path.exists(ml_ckpt) and not os.environ.get("DEEPFAKE_MODEL_PATH"):
        os.environ["DEEPFAKE_MODEL_PATH"] = ml_ckpt
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
@app.get("/api", tags=["System"])
@app.get("/api/index.py", tags=["System"])
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

