import sys

from setuptools import setup

MIN_PYTHON = (3, 11)


if sys.version_info < MIN_PYTHON:
    detected = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    required = ".".join(str(part) for part in MIN_PYTHON)
    raise SystemExit(
        "inbox-vault requires Python "
        f"{required} or later (detected {detected}). "
        "Create the virtualenv with `python3.11 -m venv .venv` and retry."
    )


setup()
