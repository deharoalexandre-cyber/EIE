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

    def chat_response(body, stream=False):
        status, raw = request('/v1/chat/completions', {'model': 'fixture', **body,
                                                        'stream': stream})
        assert status == 200, raw
        if not stream:
            return json.loads(raw)
        events = [line[6:] for line in raw.splitlines() if line.startswith('data: ')]
        assert events and events[-1] == '[DONE]', raw
        return [json.loads(line) for line in events[:-1]]

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

        status, raw = request('/v1/chat/completions', {'model': 'fixture', 'prompt': 'overflow'})
        error = json.loads(raw)['error']
        assert status == 400 and error['code'] == 'context_length_exceeded'
        assert error['type'] == 'invalid_request_error'
        status, raw = request('/v1/chat/completions', {'model': 'fixture', 'prompt': 'overflow', 'stream': True})
        assert status == 200 and 'context_length_exceeded' in raw and '[DONE]' not in raw
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

        # Tool definitions must reach the native chat formatter. The fake backend
        # returns a structured call only if it received the schema and choice.
        weather = {'type': 'function', 'function': {
            'name': 'lookup_weather', 'description': 'Look up the weather',
            'parameters': {'type': 'object', 'properties': {
                'city': {'type': 'string'}}, 'required': ['city']}}}
        tool_body = {'messages': [{'role': 'user', 'content': 'Weather in Paris?'}],
                     'tools': [weather], 'tool_choice': 'required', 'max_tokens': 32}
        response = chat_response(tool_body)
        choice = response['choices'][0]
        assert choice['finish_reason'] == 'tool_calls', response
        assert choice['message']['role'] == 'assistant', response
        assert choice['message'].get('content') in (None, ''), response
        assert choice['message']['reasoning_content'] == 'Checking weather.', response
        calls = choice['message']['tool_calls']
        assert len(calls) == 1 and calls[0]['id'] == 'call_fixture_weather', response
        assert calls[0]['type'] == 'function', response
        assert calls[0]['function']['name'] == 'lookup_weather', response
        assert json.loads(calls[0]['function']['arguments']) == {'city': 'Paris'}, response
        assert response['usage']['prompt_tokens'] == 11, response
        checks += 1

        chunks = chat_response(tool_body, True)
        deltas = [chunk['choices'][0]['delta'] for chunk in chunks]
        streamed_calls = [call for delta in deltas for call in delta.get('tool_calls', [])]
        assert streamed_calls, chunks
        assert chunks[-1]['choices'][0]['finish_reason'] == 'tool_calls', chunks
        assert ''.join(delta.get('reasoning_content', '') for delta in deltas) == 'Checking weather.', chunks
        assert chunks[-1]['usage']['completion_tokens'] > 0, chunks
        assert not ''.join(delta.get('content') or '' for delta in deltas), chunks
        assert streamed_calls[0]['index'] == 0, chunks
        assert streamed_calls[0]['id'] == calls[0]['id'], chunks
        assert streamed_calls[0]['type'] == 'function', chunks
        assert streamed_calls[0]['function']['name'] == 'lookup_weather', chunks
        assert json.loads(''.join(c['function'].get('arguments', '')
                                  for c in streamed_calls)) == {'city': 'Paris'}, chunks
        checks += 1

        parallel = dict(tool_body, parallel_tool_calls=True)
        parallel_calls = chat_response(parallel)['choices'][0]['message']['tool_calls']
        assert [call['id'] for call in parallel_calls] == [
            'call_fixture_weather', 'call_fixture_weather_2'], parallel_calls
        assert [json.loads(call['function']['arguments']) for call in parallel_calls] == [
            {'city': 'Paris'}, {'city': 'Lyon'}], parallel_calls
        parallel_chunks = chat_response(parallel, True)
        parallel_deltas = [call for chunk in parallel_chunks
                           for call in chunk['choices'][0]['delta'].get('tool_calls', [])]
        assert [call['index'] for call in parallel_deltas] == [0, 1], parallel_chunks
        assert [call['id'] for call in parallel_deltas] == [
            'call_fixture_weather', 'call_fixture_weather_2'], parallel_chunks
        assert parallel_chunks[-1]['choices'][0]['finish_reason'] == 'tool_calls', parallel_chunks
        checks += 1

        # A named choice is normalized to one available function; "none" must
        # suppress calls. An assistant call and tool reply must survive the next
        # turn so the model can complete the exchange.
        named_body = dict(tool_body, tools=[weather, {
            'type': 'function', 'function': {'name': 'other',
                                             'parameters': {'type': 'object'}}}],
            tool_choice={'type': 'function',
                         'function': {'name': 'lookup_weather'}})
        assert chat_response(named_body)['choices'][0]['finish_reason'] == 'tool_calls'
        assert chat_response(dict(tool_body, tool_choice='auto'))['choices'][0][
            'finish_reason'] == 'tool_calls'
        assert chat_response(dict(tool_body, reasoning_effort='none',
                                  chat_template_kwargs={'enable_thinking': True}))[
            'choices'][0]['finish_reason'] == 'tool_calls'
        none_choice = chat_response(dict(tool_body, tool_choice='none'))['choices'][0]
        assert none_choice['finish_reason'] != 'tool_calls'
        assert not none_choice['message'].get('tool_calls')
        none_chunks = chat_response(dict(tool_body, tool_choice='none'), True)
        assert not any(chunk['choices'][0]['delta'].get('tool_calls')
                       for chunk in none_chunks), none_chunks
        assert ''.join(chunk['choices'][0]['delta'].get('content', '')
                       for chunk in none_chunks) == none_choice['message']['content']
        followup = chat_response({'messages': [
            tool_body['messages'][0], choice['message'],
            {'role': 'tool', 'tool_call_id': calls[0]['id'],
             'content': '{"temperature_c":18}'}],
            'tools': [weather], 'tool_choice': 'auto'})
        final = followup['choices'][0]
        assert final['finish_reason'] == 'stop', followup
        assert final['message']['content'] == 'It is 18 C in Paris.', followup
        assert not final['message'].get('tool_calls'), followup
        checks += 1

        # Generic-template fallback still accepts the OpenAI null assistant
        # content used beside tool calls.
        null_history = chat_response({'messages': [
            {'role': 'user', 'content': 'Hello'},
            {'role': 'assistant', 'content': None},
            {'role': 'user', 'content': 'Continue'}]})
        assert null_history['choices'][0]['message']['content'], null_history
        checks += 1

        # Bad tool requests must fail rather than silently becoming plain chat.
        for invalid in [
            dict(tool_body, tools={'type': 'function'}),
            dict(tool_body, tools=[{'type': 'function', 'function': {}}]),
            dict(tool_body, tool_choice='impossible'),
            dict(tool_body, tool_choice={'type': 'function',
                                         'function': {'name': 'missing'}}),
            {'messages': tool_body['messages'], 'tool_choice': 'required'},
        ]:
            status, raw = request('/v1/chat/completions', {'model': 'fixture', **invalid})
            assert status == 400 and 'error' in json.loads(raw), (invalid, status, raw)
        checks += 1

        print(json.dumps({'result': 'pass', 'http_scenarios': checks,
                          'concurrent_requests': 60, 'backend': 'fake', 'gpu_used': False}))
        for route in ['/v1/batch/execute', '/v1/chain/execute']:
            for action in ['retry_once', 'replace_with']:
                status, raw = request(route, {'group': action, 'prompt': 'recover-' + route + action})
                result = json.loads(raw)
                assert status == 200 and result['status'] == 'complete', raw
                assert result['completed'] == 1 and len(result['attempts']) == 2, raw
                assert result['attempts'][0]['error'] == 'first call failed', raw
                assert result['attempts'][1]['ok'] and result['responses'][0]['ok'], raw
                assert result['responses'][0]['model'] == ('fixture' if action == 'retry_once' else 'idle')
        chat({'ews_trace': True})
        h = json.loads(request('/v1/admin/ews/routing')[1])['fixture']
        assert h['enabled'] and h['layers'][0]['prefill']['experts'][-1]['expert'] == 287
        seq = h['request_sequence']
        chat({})
        h = json.loads(request('/v1/admin/ews/routing')[1])['fixture']
        assert not h['enabled'] and not h['layers'] and h['request_sequence'] == seq + 1
        request('/v1/chat/completions', {'model': 'idle', 'prompt': 'test'})
        assert json.loads(request('/v1/admin/ews/routing')[1])['idle']['enabled']
        print(json.dumps({'result': 'pass', 'http_recovery_scenarios': 4, 'trace_request_flag_and_reset': True}))
    finally:
        proc.terminate()
        proc.wait(timeout=10)


if __name__ == '__main__':
    run(sys.argv[1])
