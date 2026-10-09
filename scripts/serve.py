"""Serve the built site locally, exactly as Cloudflare will: python3 scripts/serve.py [port]"""
import functools
import http.server
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "build" / "site"
port = int(sys.argv[1]) if len(sys.argv) > 1 else 4200
handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(ROOT))
print(f"Serving {ROOT} on http://localhost:{port}")
http.server.ThreadingHTTPServer(("", port), handler).serve_forever()
