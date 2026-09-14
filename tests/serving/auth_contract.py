"""Exercise authentication on actual HTTP routes; backend is a model-free fixture."""
import http.client
import json
import socket
import subprocess
import sys
import time


def run(binary):
    token = 'public-fixture-token-not-a-secret'
    checks = 0
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    proc = subprocess.Popen([binary, str(port), token], stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL,
                            creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))

    def request(path, method='GET', headers=(), body=None):
        conn = http.client.HTTPConnection('127.0.0.1', port, timeout=10)
        try:
            conn.putrequest(method, path)
            for key, value in headers:
                conn.putheader(key, value)
            data = json.dumps(body).encode() if body is not None else b''
            conn.putheader('Content-Length', str(len(data)))
            conn.endheaders()
            if data:
                time.sleep(.003) # Body deliberately arrives after the headers.
                conn.send(data)
            try:
                response = conn.getresponse()
            except OSError as error:
                raise AssertionError((path, method, headers, proc.poll())) from error
            return response.status, dict(response.getheaders()), response.read().decode()
        finally:
            conn.close()

    good = [('Authorization', 'Bearer ' + token)]
    try:
        for _ in range(200):
            if proc.poll() is not None:
                raise RuntimeError('Fixture exited')
            try:
                if request('/health', headers=good)[0] == 200:
                    break
            except OSError:
                pass
            time.sleep(.02)
        else:
            raise RuntimeError('Fixture did not start')
        routes = ['/health', '/v1/models', '/metrics', '/v1/admin/models/discover',
                  '/v1/admin/vram/status', '/v1/admin/ews/status', '/v1/admin/ews/routing',
                  '/v1/admin/scheduling/status', '/v1/admin/health/deep', '/absent']
        posts = ['/v1/chat/completions', '/v1/embeddings', '/v1/batch/execute',
                 '/v1/chain/execute', '/v1/admin/config/reload']
        bad = [[], [('Authorization', 'Bearer wrong')], [('Authorization', 'Basic ' + token)],
               [('Authorization', 'Bearer')], [('Authorization', 'Bearer ')],
               [('Authorization', 'Bearer ' + token + 'x')],
               [('Authorization', 'Bearer  ' + token)], good + good,
               good + [('authorization', 'Bearer wrong')]]
        for path in routes + posts:
            for headers in bad:
                status, response_headers, raw = request(path, 'POST' if path in posts else 'GET',
                                                       headers, {'stream': True} if path in posts else None)
                assert status == 401, (path, status, raw)
                assert response_headers['WWW-Authenticate'] == 'Bearer'
                assert json.loads(raw)['error']['type'] == 'authentication_error'
                assert token not in raw and 'wrong' not in raw
                checks += 1
        for headers in [good, [('authorization', 'bEaReR ' + token)]]:
            assert request('/health', headers=headers)[0] == 200
            for stream in [False, True]:
                status, _, raw = request('/v1/chat/completions', 'POST', headers,
                                         {'model': 'fixture', 'prompt': 'test', 'stream': stream})
                assert status == 200, raw
                assert '[DONE]' in raw if stream else json.loads(raw)['choices']
                checks += 1
        print(json.dumps({'result': 'pass', 'auth_checks': checks, 'backend': 'fake'}))
    finally:
        proc.terminate()
        proc.wait(timeout=10)


if __name__ == '__main__':
    run(sys.argv[1])
