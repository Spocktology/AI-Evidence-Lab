#!/usr/bin/env bash
# Start AI Evidence Lab locally: ./run.sh   then open http://localhost:8000
set -e
cd "$(dirname "$0")"
if [ ! -d .venv ]; then python3 -m venv .venv; fi
. .venv/bin/activate
pip install -q -r requirements.txt
[ -f .env ] || { cp .env.example .env; echo "Created .env - put your ANTHROPIC_API_KEY in it, then re-run."; exit 1; }
command -v exiftool >/dev/null || echo "WARNING: exiftool not found - metadata will be limited (see README)."
command -v ffmpeg  >/dev/null || echo "WARNING: ffmpeg not found - video analysis disabled (see README)."
exec uvicorn app.main:app --host 127.0.0.1 --port "${PORT:-8000}"
