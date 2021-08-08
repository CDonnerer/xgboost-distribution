"""Test suite to check that the examples will run
"""
import os
import sys
from subprocess import check_call

import pytest

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
EXAMPLES_DIR = os.path.join(os.path.dirname(TESTS_DIR), "examples")

EXAMPLES = [
    "basic_walkthrough.py --no-plot",
]


@pytest.mark.parametrize("example", EXAMPLES)
def test_example(example):
    cmd = example.split()
    filename, args = cmd[0], cmd[1:]
    filepath = os.path.join(EXAMPLES_DIR, filename)

    check_call([sys.executable, filepath] + args)
