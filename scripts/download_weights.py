"""Download both image models, the prompt upsampler, and safety classifier."""

import argparse
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=['all', 'base', 'distilled', 'pup'], default='all')
    parser.add_argument('--quant', choices=['bf16', 'fp8', 'nvfp4'], default='fp8')
    parser.add_argument('--weights-dir', type=Path, default=Path(os.environ.get('WEIGHTS_DIR', 'weights')))
    args = parser.parse_args()
    from huggingface_hub import snapshot_download

    names = {'base': 'flux-3-image-klein-base', 'distilled': 'flux-3-image-klein'}
    for model, name in names.items():
        if args.model not in ('all', model):
            continue
        # Keep the shared encoder and autoencoder, but download only the selected DiT.
        patterns = ['text_encoder/*', 'ae.safetensors', '*.json', '*.md']
        patterns += ['model.safetensors'] if args.quant == 'bf16' else [
            f"variants/{'fp8r' if args.quant == 'fp8' else 'nvfp4'}/*"
        ]
        snapshot_download('black-forest-labs/' + name, local_dir=args.weights_dir / name,
                          allow_patterns=patterns)
    if args.model in ('all', 'pup'):
        snapshot_download('black-forest-labs/flux-3-image-klein-pup', local_dir=args.weights_dir / 'pup')
    if args.model != 'pup':
        snapshot_download('Falconsai/nsfw_image_detection', local_dir=args.weights_dir / 'nsfw-classifier')


if __name__ == '__main__':
    main()
