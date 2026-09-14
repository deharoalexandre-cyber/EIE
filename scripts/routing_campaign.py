"""Explicit isolated forward campaign. Does not start Next or replace a server.

All outputs, including failures, are retained. No automatic overwrite/resume.
The first workload also runs with tracing off to test numerical neutrality.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--binary', type=Path, required=True)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--runtime-dir', type=Path, required=True)
    p.add_argument('--workloads', type=Path, default=Path(__file__).resolve().parents[1] / 'tests/routing_workloads_v1.json')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--slots', type=int, default=8)
    p.add_argument('--gpu-layers', type=int, default=20)
    p.add_argument('--profile', choices=['glm-cpu', 'glm-gpu'], default='glm-gpu')
    p.add_argument('--model-receipt', type=Path, required=True, help='Existing hashed model provenance; copied, not silently reverified')
    args = p.parse_args()
    binary, model, runtime = args.binary.resolve(strict=True), args.model.resolve(strict=True), args.runtime_dir.resolve(strict=True)
    workloads = json.loads(args.workloads.read_text(encoding='utf-8'))
    ids = [item['id'] for item in workloads['items']]
    assert len(set(ids)) == len(ids) and all(i.replace('-', '').isalnum() for i in ids)
    assert {i['domain'] for i in workloads['items']} == {'code', 'writing', 'math', 'conversation', 'documents'}
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    provenance_bytes = args.model_receipt.read_bytes()
    json.loads(provenance_bytes) # Check the supplied receipt before starting.
    (output / 'model-provenance.json').write_bytes(provenance_bytes)
    libraries = {f.name: sha(f) for f in sorted(runtime.glob('*.dll'))}
    manifest = {'schema': 'eie.routing-campaign/v1', 'workloads': workloads,
                'workloads_sha256': sha(args.workloads), 'binary_sha256': sha(binary),
                'runtime_libraries_sha256': libraries,
                'model_first_shard': model.name, 'model_size_bytes': model.stat().st_size,
                'model_provenance_sha256': sha(args.model_receipt), 'model_hashes_rechecked': False,
                'profile': args.profile, 'slots': args.slots, 'gpu_layers': args.gpu_layers,
                'sampling': 'greedy argmax; no stochastic sampling',
                'conditions': 'fresh process and empty EWS/KV caches per item; OS cache not flushed',
                'created_unix': time.time(), 'status': 'preregistered'}
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    env = os.environ.copy()
    env['PATH'] = str(runtime) + os.pathsep + env.get('PATH', '')
    for index, item in enumerate(workloads['items']):
        prompt = output / (item['id'] + '.prompt.txt')
        prompt.write_text(item['prompt'], encoding='utf-8')
        for trace in ([False, True] if index == 0 else [True]):
            prefix = output / (item['id'] + ('' if trace else '-untraced'))
            command = [str(binary), str(model), str(prompt), str(prefix), str(args.slots),
                       str(workloads['predict']), args.profile, str(args.gpu_layers), str(int(trace))]
            print(f"START {prefix.name}", flush=True)
            started = time.perf_counter()
            with prefix.with_suffix('.stdout.log').open('wb') as out, prefix.with_suffix('.stderr.log').open('wb') as err:
                result = subprocess.run(command, stdout=out, stderr=err, env=env,
                                        creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
            record = {'item': item, 'trace': trace, 'returncode': result.returncode,
                      'wall_seconds': time.perf_counter() - started}
            prefix.with_suffix('.run.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
            if result.returncode:
                raise RuntimeError(f'{item["id"]} failed; evidence retained, campaign stopped')
            if index == 0 and trace:
                reference = output / (item['id'] + '-untraced.logits.bin')
                candidate = prefix.with_suffix('.logits.bin')
                equal = sha(reference) == sha(candidate)
                (output / 'trace-neutrality.json').write_text(json.dumps({
                    'reference_sha256': sha(reference), 'candidate_sha256': sha(candidate), 'bit_identical': equal,
                    'scope': 'first short workload only; not proof for every input'}, indent=2), encoding='utf-8')
                if not equal:
                    raise RuntimeError('Tracing changed logits; stop before interpreting histograms')
            print(f"DONE {prefix.name}: {record['wall_seconds']:.2f}s", flush=True)
    (output / 'complete.json').write_text(json.dumps({'status': 'complete', 'items': len(ids)}), encoding='utf-8')


if __name__ == '__main__':
    main()
