"""Package explicit synthetic-corpus campaign artifacts; never reads Next state."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile
from analyze_routing import analyze


def digest(data):
    return hashlib.sha256(data).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('campaign', type=Path)
    p.add_argument('backend_smoke', type=Path)
    p.add_argument('destination', type=Path)
    args = p.parse_args()
    campaign = args.campaign.resolve(strict=True)
    smoke = args.backend_smoke.resolve(strict=True)
    analyzed = analyze(campaign) # Reject inconsistent, partial or missing receipts.
    manifest = json.loads((campaign / 'manifest.json').read_text())
    assert json.loads((campaign / 'complete.json').read_text())['status'] == 'complete'
    neutral = json.loads((campaign / 'trace-neutrality.json').read_text())
    first = manifest['workloads']['items'][0]['id']
    reference = (campaign / (first + '-untraced.logits.bin')).read_bytes()
    candidate = (campaign / (first + '.logits.bin')).read_bytes()
    assert reference == candidate and neutral['bit_identical']
    assert digest(reference) == neutral['reference_sha256'] == neutral['candidate_sha256']
    smoke_result = next(json.loads(l) for l in (smoke / 'stdout.log').read_text().splitlines() if l.startswith('{'))
    assert smoke_result['result'] == 'pass'
    prefix = args.destination.resolve()
    outputs = [prefix.with_suffix(s) for s in ('.zip', '.json', '.svg')]
    if any(path.exists() for path in outputs):
        raise FileExistsError('Refusing to replace a published receipt')
    prefix.parent.mkdir(parents=True, exist_ok=True)
    files = {}
    for name in ['manifest.json', 'model-provenance.json', 'complete.json', 'trace-neutrality.json',
                 'analysis.json', 'concentration.svg']:
        files[name] = (campaign / name).read_bytes()
    for index, item in enumerate(manifest['workloads']['items']):
        files[item['id'] + '.prompt.txt'] = (campaign / (item['id'] + '.prompt.txt')).read_bytes()
        names = [item['id']] + ([item['id'] + '-untraced'] if index == 0 else [])
        for name in names:
            for suffix in ['.json', '.stdout.log', '.stderr.log', '.run.json']:
                files[name + suffix] = (campaign / (name + suffix)).read_bytes()
    # Publish the complete numerical off/on control, not just its hashes.
    files[first + '-untraced.logits.bin'] = reference
    files[first + '.logits.bin'] = candidate
    files['backend-smoke.stdout.log'] = (smoke / 'stdout.log').read_bytes()
    files['backend-smoke.stderr.log'] = (smoke / 'stderr.log').read_bytes()
    all_logits = {f.name: {'sha256': digest(f.read_bytes()), 'bytes': f.stat().st_size}
                  for f in campaign.glob('*.logits.bin')}
    files['all-logits-manifest.json'] = json.dumps(all_logits, indent=2).encode()
    file_manifest = {name: {'sha256': digest(data), 'bytes': len(data)} for name, data in files.items()}
    files['files.sha256.json'] = json.dumps(file_manifest, indent=2).encode()
    with zipfile.ZipFile(outputs[0], 'x', zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for name, data in sorted(files.items()):
            info = zipfile.ZipInfo(name, (2026, 9, 14, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, data)
    rows = []
    for c in analyzed['curves']:
        if c['split'] != 'test' or c['phase'] != 'decode':
            continue
        held, oracle = c['calibration_ranked'], c['in_sample_oracle']
        rows.append({'domain': c['domain'], 'measured_8_slot_hit_rate': held['actual_ews_hit_rate'],
                     'held_out_coverage_64': held['selection_coverage'][64],
                     'uniform_64_slots_expert_bytes': held['expert_weight_bytes'][64],
                     'in_sample_90pct_slots_per_layer': next(i for i, v in enumerate(oracle['selection_coverage']) if v >= .9),
                     'held_out_90pct_slots_per_layer': next(i for i, v in enumerate(held['selection_coverage']) if v >= .9)})
    summary = {'schema': 'eie.glm-routing-pilot/v1', 'date': '2026-09-14',
               'status': 'local pilot passed; not a hotset deployment qualification',
               'manifest': manifest, 'decode_test_results': rows, 'trace_neutrality': neutral,
               'logit_floats_compared': len(reference) // 4,
               'backend_lifecycle': {k: v for k, v in smoke_result.items() if k != 'first_trace'},
               'archive': {'file': outputs[0].name, 'bytes': outputs[0].stat().st_size,
                           'sha256': digest(outputs[0].read_bytes())},
               'limitations': ['one short calibration and test prompt per domain', 'raw completion, not Next conversation',
                               '16 predictions per item; no 1k/4k/16k/64k sweep',
                               'first-control timing pair is not an overhead benchmark; concurrent compilation',
                               'selection counts are not probability mass', 'LRU curve is not the EWS top-k-protected policy',
                               'model receipt retained; 199.7 GB shard hashes not rechecked in this campaign',
                               'in-sample ranking is an oracle; unseen-expert ties are resolved by expert ID',
                               'no Kimi/Mixtral/Gemma family classification or multi-tier latency claim']}
    outputs[1].write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    outputs[2].write_bytes((campaign / 'concentration.svg').read_bytes())
    print(json.dumps(summary['archive']))


if __name__ == '__main__':
    main()
