SHELL := /bin/bash

.PHONY: setup-venv install-venv test-venv test-mamba

setup-venv:
	bash scripts/setup_venv.sh

install-venv:
	source .venv-hil-pusht/bin/activate && python -m pip install -e .[dev]

test-venv:
	source .venv-hil-pusht/bin/activate && python -m pytest -q

test-mamba:
	bash scripts/test_shared_control.sh

