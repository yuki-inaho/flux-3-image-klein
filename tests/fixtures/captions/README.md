# Captions for the minimal GPU smoke test

These are actual outputs of the local FLUX 3 Prompt Upsampler, recorded on 2026-10-21.
`caption.txt` expands a request for a red bicycle against a white brick wall.
`distilled-edit-caption.txt` expands a request to change the bicycle to blue, with a generated image as reference.
The image caption can be used with either Base or distilled. It is an input to real image inference, not a mock.

Pass `--e2e-caption-dir tests/fixtures/captions` to reuse these inputs without loading the 9B upsampler.
The upsampler test is explicitly skipped in that mode. Omit the option to test live upsampling as well.
