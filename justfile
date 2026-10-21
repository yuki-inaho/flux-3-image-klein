set positional-arguments

# List available commands.
default:
    @just --list

# Install the locked uv environment and test dependencies.
setup:
    uv python install 3.12
    uv sync --managed-python --locked --group test

# Generate and edit with both FP8 models at 256x256 / 4 steps, with memory monitoring.
test-minimal size="256" steps="4":
    #!/usr/bin/env bash
    set -euo pipefail
    uv run --locked --group test python scripts/monitor_e2e.py \
      --log artifacts/e2e-minimal/resources.csv \
      -- .venv/bin/python -m pytest tests -q --run-e2e \
      --e2e-caption-dir tests/fixtures/captions \
      --e2e-artifacts artifacts/e2e-minimal --e2e-size "$1" --e2e-steps "$2"

# Generate a sample at 800x608 by default (distilled: 16 steps, base: 50 steps).
generate width="800" height="608" model="distilled":
    #!/usr/bin/env bash
    set -euo pipefail
    out="artifacts/generated/${3}-${1}x${2}.png"
    caption="$(cat tests/fixtures/captions/caption.txt)"
    uv run --locked --group test python scripts/monitor_e2e.py \
      --log "${out%.png}-resources.csv" \
      -- .venv/bin/python scripts/cli.py "$caption" --model "$3" --fp8 --no-pup \
      --width "$1" --height "$2" --seed 7 --out "$out"
