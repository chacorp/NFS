#!/usr/bin/env bash
# uv-based env setup (cuda 12.1 / python 3.10 stack, see pyproject.toml + uv.lock).
set -euo pipefail

if ! command -v uv >/dev/null 2>&1; then
    echo "uv not found. Install it first: curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

uv sync

echo "done. run scripts with: uv run python train.py ..."
