# Agent Rules

## Python Environment
- Default to the local virtual environment named `.venv-hil-pusht` at repo root.
- If local venv dependency installation is blocked, use the `shared-control` mamba environment.
- Do not install dependencies globally.
- Do not run tests or scripts with the system Python.

## Setup Workflow
1. Create/update the venv:
   - `bash scripts/setup_venv.sh`
2. Activate it:
   - `source .venv-hil-pusht/bin/activate`
3. Verify interpreter:
   - `python -c "import sys; print(sys.prefix)"`
   - Expected output includes `.venv-hil-pusht`.

## Mamba Fallback Workflow
1. Install this repo into `shared-control`:
   - `mamba run -n shared-control python -m pip install -e . --no-deps`
2. Run tests in `shared-control`:
   - `mamba run -n shared-control python -m pytest -q`
3. One-command helper:
   - `bash scripts/test_shared_control.sh`

## Command Conventions
- Run package and test commands from repo root.
- Preferred local-venv commands:
  - `python -m pip install -e .[dev]`
  - `python -m pytest -q`
- Fallback mamba commands:
  - `mamba run -n shared-control python -m pip install -e . --no-deps`
  - `mamba run -n shared-control python -m pytest -q`
- Make targets:
  - `make setup-venv`
  - `make test-venv`
  - `make test-mamba`
