"""Diffusers real inference E2E; run from repo root with --run-e2e."""

import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_cli_help():
    result = subprocess.run([sys.executable, str(ROOT / 'media/image.py'), '--help'], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


@pytest.mark.e2e
@pytest.mark.parametrize('model', ['distilled', 'base'])
def test_generate_and_edit(request, model):
    from PIL import Image, ImageChops, ImageStat

    import torch
    assert torch.cuda.is_available(), 'E2E requires CUDA'
    artifacts = Path(request.config.getoption('--e2e-artifacts')).resolve() / 'diffusers'
    artifacts.mkdir(parents=True, exist_ok=True)
    size = request.config.getoption('--e2e-size')
    steps = request.config.getoption('--e2e-steps') or (50 if model == 'base' else 16)
    previous = None
    for task, prompt in [('generate', 'A red bicycle leaning against a white wall, afternoon sunlight.'),
                         ('edit', 'Change the red bicycle to blue. Keep the wall and lighting unchanged.')]:
        target = artifacts / f'{model}-{task}.png'
        target.unlink(missing_ok=True)
        target.with_suffix('.json').unlink(missing_ok=True)
        command = [sys.executable, str(ROOT / 'media/image.py'), prompt, '--model', model, '--out', str(target),
                   '--width', str(size), '--height', str(size), '--steps', str(steps)]
        if previous:
            command += ['--image', str(previous)]
        log = target.with_suffix('.log')
        with log.open('w') as output:
            result = subprocess.run(command, cwd=ROOT, stdout=output, stderr=subprocess.STDOUT,
                                    timeout=request.config.getoption('--e2e-timeout'))
        assert result.returncode == 0, f'Inference failed; see {log}'
        with Image.open(target) as image:
            image.load()
            assert image.size == (size, size)
            assert max(ImageStat.Stat(image.convert('RGB')).stddev) > 1
            if previous:
                with Image.open(previous) as source:
                    assert ImageChops.difference(source.convert('RGB'), image.convert('RGB')).getbbox()
        record = json.loads(target.with_suffix('.json').read_text())
        assert record['model'] == model
        assert record['input_prompt'] == prompt
        assert record['prompt']
        assert record['references'] == ([str(previous)] if previous else [])
        assert record['steps'] == steps
        previous = target
