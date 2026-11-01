import os
import sys

# Ensure backend and root paths are available in sys.path for Vercel Serverless
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
backend_dir = os.path.join(root_dir, "backend")

for path in [root_dir, backend_dir]:
    if path not in sys.path:
        sys.path.insert(0, path)

from app.main import app

# Export app and handler for Vercel
handler = app

