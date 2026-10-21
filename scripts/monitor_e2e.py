"""Run a sequential smoke test, record memory, and stop on resource limits."""

import argparse
import csv
import os
from pathlib import Path
import signal
import subprocess
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--log', type=Path, required=True)
    parser.add_argument('--ram-gib', type=float, default=64)
    parser.add_argument('--vram-mib', type=int, default=30000)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not command:
        parser.error('A command is required after --')
    env = dict(os.environ, OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2',
               TOKENIZERS_PARALLELISM='false')
    args.log.parent.mkdir(parents=True, exist_ok=True)
    process = subprocess.Popen(command, env=env, start_new_session=True)
    def stop():
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
    try:
        with args.log.open('w', newline='') as output:
            writer = csv.writer(output)
            writer.writerow(['unix_time', 'cgroup_ram_bytes', 'gpu_memory_mib'])
            while process.poll() is None:
                ram = int(Path(os.sep, 'sys', 'fs', 'cgroup', 'memory.current').read_text())
                result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
                                        capture_output=True, text=True, check=True, timeout=5)
                vram = max(int(line.strip()) for line in result.stdout.splitlines())
                writer.writerow([time.time(), ram, vram])
                output.flush()
                if ram > args.ram_gib * 1024**3 or vram > args.vram_mib:
                    print(f'Resource limit exceeded: RAM={ram / 1024**3:.1f} GiB, VRAM={vram} MiB', flush=True)
                    stop()
                    raise SystemExit(75)
                time.sleep(2)
        raise SystemExit(process.returncode)
    finally:
        stop()


if __name__ == '__main__':
    main()
