#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

mamba run -n shared-control python -m pip install -e . --no-deps
mamba run -n shared-control python -m pytest -q

