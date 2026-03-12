#!/usr/bin/env python3
"""Simple HTTP server for the SAM-Audio web app.

Serves the web app and ONNX models with correct CORS and MIME headers.
Requires: SharedArrayBuffer headers for ONNX Runtime Web WASM threads.

Usage:
    python web/serve.py [--port 8080] [--models-dir export/onnx_models_web]
"""

import argparse
import functools
import http.server
import os
from pathlib import Path


class CORSHandler(http.server.SimpleHTTPRequestHandler):
    """HTTP handler with CORS and SharedArrayBuffer headers."""

    def __init__(self, *args, models_dir=None, **kwargs):
        self.models_dir = models_dir
        super().__init__(*args, **kwargs)

    def end_headers(self):
        # Required for SharedArrayBuffer (ONNX Runtime WASM threads)
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
        self.send_header("Access-Control-Allow-Origin", "*")
        super().end_headers()

    def translate_path(self, path):
        """Route /models/ to the models directory."""
        if path.startswith("/models/"):
            rel = path[len("/models/") :]
            return os.path.join(self.models_dir, rel)
        # Serve web/ directory for everything else
        web_dir = os.path.join(os.path.dirname(__file__))
        return os.path.join(web_dir, path.lstrip("/"))


def main():
    parser = argparse.ArgumentParser(description="Serve SAM-Audio web app")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument(
        "--models-dir",
        type=str,
        default="export/onnx_models_web",
        help="Directory containing quantized ONNX models",
    )
    args = parser.parse_args()

    models_dir = str(Path(args.models_dir).resolve())
    handler = functools.partial(CORSHandler, models_dir=models_dir)

    server = http.server.HTTPServer(("0.0.0.0", args.port), handler)
    print(f"Serving SAM-Audio web app on http://localhost:{args.port}")
    print(f"Models directory: {models_dir}")
    print("Press Ctrl+C to stop")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
