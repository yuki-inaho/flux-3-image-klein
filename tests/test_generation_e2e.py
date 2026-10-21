"""Real subprocess tests: no mocks, full model defaults, persistent output artifacts."""

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
PROMPT = 'A red bicycle leaning against a white brick wall, afternoon sunlight.'
pytestmark = pytest.mark.e2e


def run(config, command, log):
    with log.open('w') as output:
        result = subprocess.run([sys.executable, *command], cwd=ROOT, stdout=output,
                                stderr=subprocess.STDOUT, timeout=config.getoption('--e2e-timeout'))
    assert result.returncode == 0, f'Command failed ({result.returncode}); see {log}'


@pytest.fixture(scope='session')
def caption(request):
    import torch

    assert torch.cuda.is_available(), 'E2E requires a working CUDA GPU'
    weights = Path(os.environ.get('WEIGHTS_DIR', ROOT / 'weights')).resolve()
    artifacts = Path(request.config.getoption('--e2e-artifacts')).resolve()
    artifacts.mkdir(parents=True, exist_ok=True)
    cached = request.config.getoption('--e2e-caption-dir')
    if cached:
        target = Path(cached).resolve() / 'caption.txt'
        assert target.is_file(), f'Missing cached caption: {target}'
        text = target.read_text().strip()
        assert len(text.split()) > 10
        return text, artifacts, weights
    assert (weights / 'pup' / 'config.json').is_file(), 'Download weights/pup first; see README'
    target = artifacts / 'caption.txt'
    target.unlink(missing_ok=True)
    # Separate process releases the upsampler GPU memory before either image model loads.
    run(request.config, ['-m', 'flux3.pup', '--prompt', PROMPT, '--out', str(target)],
        artifacts / 'pup.log')
    assert target.is_file(), 'Upsampler did not save a caption'
    text = target.read_text().strip()
    assert len(text.split()) > 10, 'Upsampler caption is unexpectedly short'
    assert text != PROMPT, 'Upsampler returned the unchanged prompt'
    return text, artifacts, weights


def test_prompt_upsampler(request, caption):
    if request.config.getoption('--e2e-caption-dir'):
        pytest.skip('Using a caption from a previous real GPU upsampler run')
    assert caption[0]


@pytest.mark.parametrize('model,steps,guidance', [('distilled', 16, None), ('base', 50, 4.0)])
def test_image_generation(request, caption, model, steps, guidance):
    from PIL import Image, ImageStat

    text, artifacts, weights = caption
    size = request.config.getoption('--e2e-size')
    steps = request.config.getoption('--e2e-steps') or steps
    quant = request.config.getoption('--e2e-quant')
    name = 'flux-3-image-klein' + ('-base' if model == 'base' else '')
    checkpoint = weights / name
    if quant != 'bf16':
        checkpoint /= 'variants/' + ('fp8r' if quant == 'fp8' else 'nvfp4')
    assert (checkpoint / 'model.safetensors').is_file(), f'Missing model weights: {checkpoint}'
    target = artifacts / f'{model}-{quant}.png'
    record_path = target.with_suffix('.json')
    target.unlink(missing_ok=True)
    record_path.unlink(missing_ok=True)
    command = ['scripts/cli.py', text, '--model', model, '--no-pup', '--width', str(size),
               '--height', str(size), '--steps', str(steps), '--seed', '7', '--out', str(target)]
    if quant != 'bf16':
        command.append('--' + quant)
    run(request.config, command, artifacts / f'{model}-{quant}.log')
    with Image.open(target) as image:
        image.load()
        assert image.format == 'PNG'
        assert image.size == (size, size)
        assert image.mode == 'RGB'
        assert max(ImageStat.Stat(image).stddev) > 1, 'Generated image is blank or nearly uniform'
        assert image.getexif().get(305) == f'AI generated;{model}'
    record = json.loads(record_path.read_text())
    assert record['model'] == model
    assert record['seed'] == 7
    assert (record['width'], record['height']) == (size, size)
    assert record['steps'] == steps
    assert record['guidance'] == guidance
    assert record['input_prompt'] == text
    assert record['upsampled'] is False  # caption already expanded by the separate real upsampler
    cached = request.config.getoption('--e2e-caption-dir')
    if cached:
        edit_caption = Path(cached).resolve() / 'distilled-edit-caption.txt'
    else:
        edit_caption = artifacts / f'{model}-edit-caption.txt'
        edit_caption.unlink(missing_ok=True)
        run(request.config, ['-m', 'flux3.pup', '--prompt', 'Change the red bicycle to blue.',
                             '--image', str(target), '--out', str(edit_caption)],
            artifacts / f'{model}-edit-pup.log')
    assert edit_caption.is_file() and edit_caption.read_text().strip()
    edited = artifacts / f'{model}-{quant}-edited.png'
    edited.unlink(missing_ok=True)
    edited.with_suffix('.json').unlink(missing_ok=True)
    command = ['scripts/cli.py', edit_caption.read_text().strip(), '--model', model,
               '--image', str(target), '--no-pup', '--width', str(size), '--height', str(size),
               '--steps', str(steps),
               '--seed', '8', '--out', str(edited)]
    if quant != 'bf16':
        command.append('--' + quant)
    run(request.config, command, artifacts / f'{model}-{quant}-edit.log')
    from PIL import ImageChops

    with Image.open(target) as source, Image.open(edited) as image:
        assert image.size == (size, size)
        assert max(ImageStat.Stat(image).stddev) > 1
        assert ImageChops.difference(source.convert('RGB'), image.convert('RGB')).getbbox()
    edited_record = json.loads(edited.with_suffix('.json').read_text())
    assert edited_record['references'] == [str(target)]
    assert edited_record['seed'] == 8
    assert edited_record['steps'] == steps
