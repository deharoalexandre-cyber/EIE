"""Validate routing receipts and compute hard-selection concentration / LRU curves.

Not router probability mass. LRU curves describe an empty-cache serial LRU,
not EWS's top-k-protected victim policy or a measured latency prediction.
"""
import argparse
import hashlib
import json
from pathlib import Path
from xml.sax.saxutils import escape


def validate(report):
    h = report['routing']
    if h.get('schema') != 'eie.routing/v1' or not h['enabled'] or h['outcome'] != 'complete':
        raise ValueError('Need a complete enabled routing receipt')
    n, k = h['expert_count'], h['top_k']
    if not 0 < k <= n or not h['layers']:
        raise ValueError('Invalid dimensions or no routed layers')
    total_hits = total_misses = total_callbacks = 0
    seen_layers = set()
    for layer in h['layers']:
        if layer['layer'] in seen_layers or layer['expert_weight_bytes'] <= 0:
            raise ValueError('Duplicate layer or missing weight size')
        seen_layers.add(layer['layer'])
        for phase in ('prefill', 'decode'):
            p = layer[phase]
            experts = p['experts']
            ids = [e['expert'] for e in experts]
            if len(set(ids)) != len(ids) or any(not 0 <= i < n for i in ids):
                raise ValueError('Duplicate / physical / invalid expert IDs')
            if len(p['reuse_distance']) != n:
                raise ValueError('Wrong stack-distance dimension')
            counters = [p['callbacks'], p['first_accesses'], *p['reuse_distance']]
            counters += [e[key] for e in experts for key in ('selected', 'hits', 'misses')]
            if any(type(v) is not int or v < 0 for v in counters):
                raise ValueError('Invalid counter')
            hits, misses = sum(e['hits'] for e in experts), sum(e['misses'] for e in experts)
            selected = sum(e['selected'] for e in experts)
            if any(e['selected'] != e['hits'] + e['misses'] for e in experts):
                raise ValueError('Expert counters do not conserve selections')
            if selected != k * p['callbacks'] or selected != p['first_accesses'] + sum(p['reuse_distance']):
                raise ValueError('Incomplete callback or reuse histogram')
            expected = report['prompt_tokens'] if phase == 'prefill' else report['predict'] - 1
            if p['callbacks'] != expected:
                raise ValueError('Phase callbacks do not match standalone forward evaluations')
            total_hits += hits; total_misses += misses; total_callbacks += p['callbacks']
    stats = report['ews']
    if (total_hits, total_misses, total_callbacks) != (stats['hits'], stats['misses'], stats['callbacks']):
        raise ValueError('Histogram and actual EWS totals differ')
    return h


def phase_curves(hist, phase, ranked_by=None):
    n = hist['expert_count']
    by_layer = {layer['layer']: layer for layer in hist['layers']}
    if ranked_by is not None and (set(by_layer) != {l['layer'] for l in ranked_by['layers']} or
                                  ranked_by['expert_count'] != n):
        raise ValueError('Calibration/test layouts differ')
    counts, ranking = {}, {}
    for layer in hist['layers']:
        lid = layer['layer']
        counts[lid] = {e['expert']: e['selected'] for e in layer[phase]['experts']}
        source = layer if ranked_by is None else next(l for l in ranked_by['layers'] if l['layer'] == lid)
        if source['expert_weight_bytes'] != layer['expert_weight_bytes']:
            raise ValueError('Calibration/test slab size differs')
        ranks = {e['expert']: e['selected'] for e in source[phase]['experts']}
        ranking[lid] = sorted(range(n), key=lambda i: (-ranks.get(i, 0), i))
    total = sum(sum(c.values()) for c in counts.values())
    selected = cumulative_lru = 0
    coverage, lru, weight_bytes = [0.0], [0.0], [0]
    for size in range(1, n + 1):
        selected += sum(counts[lid].get(ranking[lid][size - 1], 0) for lid in counts)
        cumulative_lru += sum(l[phase]['reuse_distance'][size - 1] for l in hist['layers'])
        coverage.append(selected / total if total else None)
        lru.append(cumulative_lru / total if total else None)
        weight_bytes.append(size * sum(l['expert_weight_bytes'] for l in hist['layers']))
    actual_hits = sum(sum(e['hits'] for e in l[phase]['experts']) for l in hist['layers'])
    return {'selections': total, 'actual_ews_hit_rate': actual_hits / total if total else None,
            'first_accesses': sum(l[phase]['first_accesses'] for l in hist['layers']),
            'slots_per_layer': list(range(n + 1)), 'expert_weight_bytes': weight_bytes,
            'selection_coverage': coverage, 'serial_lru_hit_rate': lru,
            'ranking': {str(lid): ids for lid, ids in ranking.items()}}


def analyze(directory):
    manifest = json.loads((directory / 'manifest.json').read_text())
    data, evidence = {}, {}
    for item in manifest['workloads']['items']:
        path = directory / (item['id'] + '.json')
        report = json.loads(path.read_text())
        hist = validate(report)
        data[item['id']] = hist
        evidence[item['id']] = hashlib.sha256(path.read_bytes()).hexdigest()
    outputs = []
    for item in manifest['workloads']['items']:
        hist = data[item['id']]
        calibration = next(i for i in manifest['workloads']['items']
                           if i['domain'] == item['domain'] and i['split'] == 'calibration')
        for phase in ('prefill', 'decode'):
            own = phase_curves(hist, phase)
            held = phase_curves(hist, phase, data[calibration['id']])
            outputs.append({'id': item['id'], 'domain': item['domain'], 'split': item['split'], 'phase': phase,
                            'in_sample_oracle': own, 'calibration_ranked': held})
    return {'schema': 'eie.routing-analysis/v1', 'receipt_sha256': evidence,
            'scope': manifest['workloads']['scope'],
            'metric': 'hard top-k selections, not router probability mass',
            'lru_scope': 'empty request-start serial LRU; stack carries from prefill to decode; not exact EWS policy',
            'curves': outputs}


def plot(analysis, path):
    colors = {'code': '#0072B2', 'writing': '#D55E00', 'math': '#009E73',
              'conversation': '#CC79A7', 'documents': '#8B6508'}
    parts = ['<svg xmlns="http://www.w3.org/2000/svg" width="1040" height="620" viewBox="0 0 1040 620">',
             '<rect width="1040" height="620" fill="white"/>',
             '<g font-family="sans-serif" fill="#202020">',
             '<text x="70" y="35" font-size="22">GLM routing: short-context pilot</text>',
             '<text x="70" y="62" font-size="14">Held-out hard-selection coverage, experts ranked on calibration (decode)</text>']
    left, top, width, height = 80, 95, 720, 410
    for i in range(6):
        y = top + height * (1 - i / 5)
        parts.append(f'<path d="M{left},{y}h{width}" stroke="#ddd"/><text x="35" y="{y+5}">{i*20}%</text>')
    curves = [c for c in analysis['curves'] if c['split'] == 'test' and c['phase'] == 'decode']
    n = max(c['calibration_ranked']['slots_per_layer'][-1] for c in curves)
    for x in (0, 32, 64, 128, 192, n):
        px = left + width * x / n
        parts.append(f'<text x="{px-10}" y="530">{x}</text>')
    for index, curve in enumerate(curves):
        values = curve['calibration_ranked']['selection_coverage']
        if any(v is None for v in values):
            raise ValueError('Cannot plot phase without selections')
        points = ' '.join(f'{left+width*i/n:.2f},{top+height*(1-v):.2f}' for i, v in enumerate(values))
        color = colors[curve['domain']]
        parts.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2.5"/>')
        parts.append(f'<text x="825" y="{125+index*30}" fill="{color}">{escape(curve["domain"])}</text>')
    parts += ['<text x="275" y="565">Experts retained per routed layer (uniform capacity)</text>',
              '<text x="70" y="597" font-size="13">One calibration + one test prompt/domain. 16 predictions. No long-context or Kimi conclusion.</text>', '</g></svg>']
    path.write_text('\n'.join(parts), encoding='utf-8')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('directory', type=Path)
    args = p.parse_args()
    result = analyze(args.directory)
    output = args.directory / 'analysis.json'
    if output.exists():
        raise FileExistsError(output)
    output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    plot(result, args.directory / 'concentration.svg')
    for row in result['curves']:
        if row['split'] == 'test' and row['phase'] == 'decode':
            c = row['calibration_ranked']
            print(row['domain'], 'measured hits', round(c['actual_ews_hit_rate'], 4),
                  'held-out coverage@64', round(c['selection_coverage'][64], 4))


if __name__ == '__main__':
    main()
