#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"

# Activate virtualenv if present
if [[ -d ".venv" ]]; then
  # shellcheck disable=SC1091
  source ".venv/bin/activate"
fi

# Quick port check to avoid silent bind failures
python - <<'PY' "$HOST" "$PORT"
import socket, sys
host, port = sys.argv[1], int(sys.argv[2])
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
try:
    s.bind((host, port))
except OSError as e:
    print(f"[ERROR] Port {port} on {host} is in use: {e}")
    sys.exit(1)
finally:
    s.close()
PY

python -m uvicorn src.core.main_api:app --host "$HOST" --port "$PORT" --reload
