import pytest


def pytest_addoption(parser):
    parser.addoption('--run-e2e', action='store_true', help='Run real GPU inference (requires local weights)')
    parser.addoption('--e2e-quant', choices=['bf16', 'fp8', 'nvfp4'], default='fp8')
    parser.addoption('--e2e-artifacts', default='artifacts/e2e')
    parser.addoption('--e2e-timeout', type=int, default=1800)
    parser.addoption('--e2e-size', type=int, default=256)
    parser.addoption('--e2e-steps', type=int, default=4, help='0 uses model default steps')
    parser.addoption('--e2e-caption-dir', type=str, help='Reuse captions from a previous real upsampler run')


def pytest_collection_modifyitems(config, items):
    if not config.getoption('--run-e2e'):
        for item in items:
            if 'e2e' in item.keywords:
                item.add_marker(pytest.mark.skip(reason='Real inference: pass --run-e2e'))
