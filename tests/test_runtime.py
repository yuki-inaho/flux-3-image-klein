"""Check that installed dependencies expose the real public entry points."""

from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('command', [
    ['scripts/cli.py', '--help'],
    ['scripts/app.py', '--help'],
    ['-m', 'flux3.pup', '--help'],
    ['scripts/download_weights.py', '--help'],
])
def test_entry_point(command):
    result = subprocess.run([sys.executable, *command], cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr
    assert 'usage:' in result.stdout
