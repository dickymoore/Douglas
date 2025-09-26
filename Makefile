.PHONY: venv test ci-offline

venv:
	python -m venv .venv
	. .venv/bin/activate && \
		pip install --upgrade pip && \
		pip install -r requirements-dev.txt && \
		pip install -e .

test:
        . .venv/bin/activate && pytest -q

ci-offline:
	./scripts/ci_offline.sh
