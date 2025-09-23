.PHONY: venv test

venv:
python -m venv .venv
. .venv/bin/activate && pip install --upgrade pip && pip install -r requirements-dev.txt
. .venv/bin/activate && pip install -e .

test:
. .venv/bin/activate && pytest -q
