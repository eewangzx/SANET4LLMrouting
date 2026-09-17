#!/usr/bin/env bash
# Linux/macOS CPU environment. Usage: bash scripts/setup_environment.sh [python3] [--with-sanet]
set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${1:-python3}"
include_sanet="${2:-}"
if [[ -n "$include_sanet" && "$include_sanet" != "--with-sanet" ]]; then
  echo "Usage: bash scripts/setup_environment.sh [python3] [--with-sanet]" >&2
  exit 2
fi
cd "$project_root"
"$python_bin" -c 'import sys; assert sys.version_info >= (3, 11), "Python 3.11+ required; 3.12 recommended"'
"$python_bin" -m venv .venv
env_python="$project_root/.venv/bin/python"
"$env_python" -m pip install --upgrade pip setuptools wheel
if [[ "$(uname -s)" == "Linux" ]]; then
  "$env_python" -m pip install --index-url https://download.pytorch.org/whl/cpu 'torch>=2.2'
else
  "$env_python" -m pip install 'torch>=2.2'
fi
"$env_python" -m pip install -r requirements.txt
if [[ "$include_sanet" == "--with-sanet" ]]; then
  "$env_python" -m pip install -r requirements-sanet.txt
fi
"$env_python" -m pip install --no-deps -e .
"$env_python" scripts/check_environment.py
echo "Ready. Activate with: source .venv/bin/activate"
echo "Verify with: python -m pytest -q"

