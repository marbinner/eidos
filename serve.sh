#!/bin/bash
# Serve the browser version on http://localhost:8765 (forward the port, then open it).
#   ./serve.sh [port]
cd "$(dirname "$0")"
# (the newest Python there is: tools/serve.py needs 3.7 or later)
exec "$(command -v python3.12 || command -v python3.11 || command -v python3)" tools/serve.py web "${1:-8765}"
