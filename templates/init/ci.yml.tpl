name: CI
# Basic lint/test workflow for the $language project scaffolded by Douglas.

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt || true
          pip install pytest
      - name: Lint
        run: |
          pip install ruff black isort
          ruff check .
          black --check .
          isort --check-only .
      - name: Tests
        run: pytest
