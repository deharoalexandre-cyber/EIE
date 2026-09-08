"""Exercise real HTTP chat/embeddings using a separately launched test server."""
import argparse
import json
import math
import os
from pathlib import Path
import socket
import subprocess
import time
import urllib.request
import urllib.error

p = argparse.ArgumentParser()
p.add_argument('--eie', type=Path, required=True)
p.add_argument('--out', type=Path, required=True)
p.add_argument('--model', required=True)
p.add_argument('--embedder', required=True)
a = p.parse_args()
a.out.mkdir(parents=True, exist_ok=False)
with socket.socket() as sock:
    sock.bind(('127.0.0.1', 0))
    port = sock.getsockname()[1]
config = a.out / 'eie.yaml'
config.write_text(f"host: 127.0.0.1\nport: {port}\nauto_discover: false\ntype_k: f16\ntype_v: f16\nn_ctx: 512\nflash_attn: true\nmodels:\n  resident: {a.model}\n  nomic-embed-text: {a.embedder}\npreload: [resident, nomic-embed-text]\n", encoding='utf-8')
env = dict(os.environ)
env['PATH'] = str(a.eie / 'build-ews/bin/Release') + os.pathsep + env.get('PATH', '')
env['GGML_CUDA_DISABLE_GRAPHS'] = '1'
def request(path, body=None):
    req = urllib.request.Request(f'http://127.0.0.1:{port}{path}',
        data=None if body is None else json.dumps(body).encode(),
        headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=180) as response:
        return response.read().decode('utf-8')
def chat(stream=False, stop=None):
    body = {'model': 'resident', 'messages': [{'role': 'user', 'content': 'Write one short sentence about a blue bicycle.'}],
            'temperature': 0, 'max_tokens': 32, 'one_shot': True, 'stream': stream}
    if stop is not None: body['stop'] = stop
    raw = request('/v1/chat/completions', body)
    if not stream:
        v = json.loads(raw)
        return {'text': v['choices'][0]['message']['content'], 'finish': v['choices'][0]['finish_reason'], 'usage': v['usage']}
    chunks = [line[6:] for line in raw.splitlines() if line.startswith('data: ')]
    assert chunks[-1] == '[DONE]', raw
    values = [json.loads(line) for line in chunks[:-1]]
    return {'text': ''.join(v['choices'][0]['delta'].get('content', '') for v in values),
            'finish': values[-1]['choices'][0]['finish_reason'], 'usage': values[-1]['usage']}
report = {'pass': False, 'model_backend': 'real', 'port': port}
with (a.out / 'server.log').open('wb') as log:
    process = subprocess.Popen([str(a.eie / 'build-ews/Release/eie-server.exe'), '--config', str(config)],
        env=env, stdout=log, stderr=subprocess.STDOUT, creationflags=subprocess.CREATE_NO_WINDOW)
    try:
        deadline = time.monotonic() + 180
        while time.monotonic() < deadline:
            if process.poll() is not None: raise RuntimeError('EIE exited at boot')
            try:
                catalog = json.loads(request('/v1/models'))
                assert {m['id'] for m in catalog['data']} == {'resident', 'nomic-embed-text'}, catalog
                break
            except OSError: time.sleep(.5)
        else: raise TimeoutError('EIE startup')
        print('REAL HTTP: two models loaded', flush=True)
        assert json.loads(request('/health'))['models'] == 2
        assert 'eie_models_loaded 2\n' in request('/metrics')
        normal, streamed = chat(), chat(True)
        assert normal == streamed and normal['text'], (normal, streamed)
        assert normal['usage']['prompt_tokens'] > 0
        assert normal['usage']['total_tokens'] == normal['usage']['prompt_tokens'] + normal['usage']['completion_tokens']
        stop = normal['text'].split()[0]
        stopped, stopped_sse = chat(stop=stop), chat(True, stop)
        assert stopped == stopped_sse and stopped['text'] == '' and stopped['finish'] == 'stop'
        vectors = json.loads(request('/v1/embeddings', {'model': 'nomic-embed-text', 'input': ['search_query: blue bicycle', 'search_document: A blue bicycle rests near a tree.']}))
        dims = [len(item['embedding']) for item in vectors['data']]
        assert dims == [768, 768], dims
        assert all(math.isfinite(v) for item in vectors['data'] for v in item['embedding'])
        assert all(sum(v*v for v in item['embedding']) > 0 for item in vectors['data'])
        for body, code in [
            ({'model': 'missing', 'strict_model': True, 'prompt': 'test'}, 404),
            ({'model': 'resident', 'prompt': 'test', 'max_tokens': 512, 'truncate_prompt': False}, 400),
        ]:
            try: request('/v1/chat/completions', body); raise AssertionError('Expected HTTP error')
            except urllib.error.HTTPError as exc: assert exc.code == code
        assert chat() == normal, 'Chat failed to recover'
        assert json.loads(request('/health'))['models'] == 2
        report.update(reference=normal, stopped=stopped, embedding_dimensions=dims,
                      health=json.loads(request('/health')), metrics=request('/metrics'))
        report['pass'] = True
        print('REAL HTTP PASS: chat/SSE/stop/usage/embeddings/errors/recovery', flush=True)
    except Exception as e:
        report['error'] = repr(e)
        raise
    finally:
        if process.poll() is None: process.terminate()
        process.wait(timeout=30)
        (a.out / 'report.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
