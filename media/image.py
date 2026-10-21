"""FLUX 3 image generation/editing with Diffusers and component CPU offload."""

import argparse
import json
from pathlib import Path
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('prompt')
    parser.add_argument('--model', choices=['base', 'distilled'], default='distilled')
    parser.add_argument('--image', type=Path, action='append', default=[])
    parser.add_argument('--out', type=Path, default=Path('artifacts/diffusers/image.png'))
    parser.add_argument('--width', type=int, default=512)
    parser.add_argument('--height', type=int, default=512)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--steps', type=int, help='Default: distilled 16, base 50')
    args = parser.parse_args()
    if len(args.image) > 10:
        parser.error('At most 10 reference images are supported')
    if any(v < 64 or v > 4096 or v % 16 for v in (args.width, args.height)):
        parser.error('Width and height must be multiples of 16 between 64 and 4096')
    if args.steps is not None and args.steps < 1:
        parser.error('Steps must be positive')

    import torch
    from diffusers import ComponentsManager, ModularPipeline
    from PIL import Image

    if not torch.cuda.is_available():
        raise SystemExit('CUDA GPU is required')
    model_id = 'black-forest-labs/flux-3-image-klein' + ('-base' if args.model == 'base' else '')
    started = time.monotonic()
    manager = ComponentsManager()
    pipe = ModularPipeline.from_pretrained(model_id, components_manager=manager)
    pipe.load_components(dtype={'vae': torch.float32, 'default': torch.bfloat16})
    manager.enable_auto_cpu_offload(device='cuda')
    if args.model == 'base':
        pipe.guider = pipe.guider.new(guidance_scale=4.0)
    references = []
    for path in args.image:
        with Image.open(path) as source:
            references.append(source.convert('RGB'))
    steps = args.steps or (50 if args.model == 'base' else 16)
    inputs = dict(prompt=args.prompt, width=args.width, height=args.height,
                  num_inference_steps=steps, generator=torch.Generator().manual_seed(args.seed),
                  output=['images', 'prompt'])
    if references:
        inputs['image'] = references
    result = pipe(**inputs)
    image = result['images'][0]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    tags = Image.Exif()
    tags[305], tags[271] = f'AI generated;{args.model}', 'Black Forest Labs'
    image.save(args.out, exif=tags)
    record = dict(backend='diffusers', model=args.model, model_id=model_id, input_prompt=args.prompt,
                  prompt=result['prompt'], references=[str(p) for p in args.image],
                  width=image.width, height=image.height, seed=args.seed, steps=steps,
                  guidance=4.0 if args.model == 'base' else None,
                  seconds=time.monotonic() - started, gpu=torch.cuda.get_device_name(),
                  torch_version=torch.__version__)
    args.out.with_suffix('.json').write_text(json.dumps(record, indent=2, ensure_ascii=False) + '\n')
    print(f'wrote {args.out}')


if __name__ == '__main__':
    main()
