"""Real EIE HTTP routes, deterministic fake backend. No model or GPU."""
import concurrent.futures
import http.client
import json
import socket
import subprocess
import sys
import time


def run(binary):
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    proc = subprocess.Popen([binary, str(port)], stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL,
                            creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    checks = 0

    def request(path, body=None):
        conn = http.client.HTTPConnection('127.0.0.1', port, timeout=5)
        try:
            conn.request('POST' if body is not None else 'GET', path,
                         json.dumps(body) if body is not None else None,
                         {'Content-Type': 'application/json'})
            response = conn.getresponse()
            return response.status, response.read().decode('utf-8')
        finally:
            conn.close()

    def chat(extra, stream=False):
        body = {'model': 'fixture', 'prompt': 'test', 'max_tokens': 8,
                **extra, 'stream': stream}
        status, raw = request('/v1/chat/completions', body)
        assert status == 200, raw
        if not stream:
            obj = json.loads(raw)
            return obj['choices'][0]['message']['content'], obj['choices'][0]['finish_reason'], obj['usage']
        events = [line[6:] for line in raw.splitlines() if line.startswith('data: ')]
        assert events and events[-1] == '[DONE]', raw
        chunks = [json.loads(line) for line in events[:-1]]
        text = ''.join(item['choices'][0]['delta'].get('content', '') for item in chunks)
        last = chunks[-1]
        return text, last['choices'][0]['finish_reason'], last['usage']

    try:
        for _ in range(100):
            if proc.poll() is not None: raise RuntimeError('Fixture exited before listening')
            try:
                if request('/health')[0] == 200: break
            except OSError: pass
            time.sleep(.02)
        else: raise RuntimeError('Fixture did not listen')

        assert json.loads(request('/health')[1])['models'] == 3
        checks += 1
        for extra, expected, reason, tokens in [
            ({}, 'caf\u00e9 END tail', 'length', 4),
            ({'stop': 'END'}, 'caf\u00e9 ', 'stop', 4),
            ({'stop': ['END', 'absent']}, 'caf\u00e9 ', 'stop', 4),
            ({'stop': 'END', 'max_tokens': 3}, 'caf\u00e9 E', 'length', 3),
            ({'stop': 'END', 'max_tokens': 1}, 'caf\ufffd', 'length', 1),
            ({'stop': 'caf'}, '', 'stop', 1),
            ({'stop': ''}, 'caf\u00e9 END tail', 'length', 4),
            ({'stop': 'END', 'one_shot': True}, 'caf\u00e9 ', 'stop', 4),
        ]:
            normal, streamed = chat(extra), chat(extra, True)
            assert normal == streamed, (extra, normal, streamed)
            text, finish, usage = normal
            assert (text, finish) == (expected, reason), normal
            assert usage['prompt_tokens'] == 11 and usage['completion_tokens'] == tokens
            assert usage['total_tokens'] == 11 + tokens
            assert usage['prompt_tokens_details']['cached_tokens'] == (0 if extra.get('one_shot') else 3)
            checks += 1

        status, raw = request('/v1/chat/completions', {'model': 'fixture', 'prompt': 'error'})
        assert status == 500 and json.loads(raw)['error']['message'] == 'injected test error'
        status, raw = request('/v1/chat/completions', {'model': 'fixture', 'prompt': 'error', 'stream': True})
        assert status == 200 and 'injected test error' in raw and '[DONE]' not in raw
        assert chat({'stop': 'END'})[0] == 'caf\u00e9 '
        checks += 1

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            calls = [pool.submit(chat, {'stop': 'END'}, i % 2 == 0) for i in range(40)]
            calls += [pool.submit(request, '/metrics') for _ in range(20)]
            for call in calls: call.result()
        assert json.loads(request('/health')[1])['models'] == 3
        assert 'eie_models_loaded 3\n' in request('/metrics')[1]
        assert len(json.loads(request('/v1/models')[1])['data']) == 3
        checks += 1
        print(json.dumps({'result': 'pass', 'http_scenarios': checks,
                          'concurrent_requests': 60, 'backend': 'fake', 'gpu_used': False}))
    finally:
        proc.terminate()
        proc.wait(timeout=10)


if __name__ == '__main__':
    run(sys.argv[1])
