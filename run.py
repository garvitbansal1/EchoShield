#!/usr/bin/env python3
"""Entry point to run EchoShield fullstack application locally."""
import os
import sys

# Configure UTF-8 stdout if supported
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

root_dir = os.path.dirname(os.path.abspath(__file__))
backend_dir = os.path.join(root_dir, "backend")

for p in [root_dir, backend_dir]:
    if p not in sys.path:
        sys.path.insert(0, p)

import uvicorn
from dotenv import load_dotenv

load_dotenv()
load_dotenv(os.path.join(backend_dir, ".env"))

if __name__ == "__main__":
    print("=" * 65)
    print("[*] EchoShield - Real-Time AI Scammer Interceptor")
    print("[*] Dashboard running at: http://localhost:8000")
    print("[*] Swagger API docs at:  http://localhost:8000/docs")
    print("=" * 65)
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info",
    )
