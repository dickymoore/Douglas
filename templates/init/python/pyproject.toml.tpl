[build-system]
requires = ["setuptools>=61.0"]
build-backend = "setuptools.build_meta"

[project]
name = "${package_name}"
version = "0.1.0"
description = "Scaffolded with Douglas."
authors = [{ name = "${license_holder}" }]
readme = "README.md"
requires-python = ">=3.11"
dependencies = []

[project.optional-dependencies]
dev = ["pytest>=7.0"]
