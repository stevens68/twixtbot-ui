"""Aggregate test suite for IDE runners that execute one module at a time."""

import pathlib


ROOT = pathlib.Path(__file__).resolve().parents[1]
TESTS = pathlib.Path(__file__).resolve().parent


def load_tests(loader, tests, pattern):
    return loader.discover(
        start_dir=str(TESTS),
        pattern='test_*.py',
        top_level_dir=str(ROOT))