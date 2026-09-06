// Real 26B EWS cancellation regression, using only Node's standard library.
// Usage: node tests/prefill_cancel.cjs <evidence-dir> <baseline-exe> <candidate-exe>
// Starts/stops only its own servers. Fails closed on competing inference processes.
const fs = require('node:fs');
const path = require('node:path');
const http = require('node:http');
const net = require('node:net');
const { spawn, spawnSync } = require('node:child_process');
const assert = require('node:assert/strict');
const [outArg, baseline, candidate] = process.argv.slice(2);
assert(outArg && baseline && candidate, 'Supply evidence directory and both executables');
const root = path.resolve(__dirname, '..');
const out = path.resolve(outArg);
assert(out.startsWith(root + path.sep), 'Evidence must stay inside EIE');
fs.mkdirSync(out, { recursive: true });
const port = 18080;
const report = { started: new Date().toISOString(), runs: [], guard_checks: 0 };
let server, guardTimer, guardFailure;
const now = () => performance.now();
const sleep = ms => new Promise(resolve => setTimeout(resolve, ms));
const save = () => fs.writeFileSync(path.join(out, 'report.json'), JSON.stringify(report, null, 2));
function guard() {
  const scan = spawnSync('powershell.exe', ['-NoProfile', '-Command',
    "$ErrorActionPreference='Stop'; @(Get-Process | Where-Object { $_.ProcessName -match '^(elyne.*|eie-server.*|llama-server.*|python.*|node)$' } | Select-Object Id,ProcessName,Path) | ConvertTo-Json -Compress"],
    { encoding: 'utf8', windowsHide: true, timeout: 5000 });
  assert.equal(scan.status, 0, 'Cannot monitor competing processes: ' + scan.stderr);
  const items = scan.stdout.trim() ? JSON.parse(scan.stdout) : [];
  const competing = (Array.isArray(items) ? items : [items]).filter(p =>
    p.Id !== server?.pid && p.Id !== process.pid &&
    !(p.ProcessName === 'node' && p.Path?.includes('\\OpenAI\\Codex\\')));
  report.guard_checks++;
  assert.equal(competing.length, 0, 'Competing process appeared: ' + JSON.stringify(competing));
}
function check() {
  if (guardFailure) throw guardFailure;
  if (server && server.exitCode !== null) throw Error('Test server exited: ' + server.exitCode);
}
function get(route) {
  return new Promise((resolve, reject) => {
    const req = http.get({ host: '127.0.0.1', port, path: route, agent: false }, res => {
      let body = '';
      res.on('data', b => body += b);
      res.on('end', () => { try { resolve(JSON.parse(body)); } catch (e) { reject(e); } });
      res.on('error', reject);
    });
    req.setTimeout(3000, () => req.destroy(Error('GET timeout')));
    req.on('error', reject);
  });
}
function payload(content, options = {}) {
  return { model: 'gemma-26b-ews', strict_model: true, truncate_prompt: false,
    temperature: 0, max_tokens: 4, stream: true, one_shot: true,
    messages: [{ role: 'user', content }], ...options };
}
function complete(body) {
  const started = now();
  return new Promise((resolve, reject) => {
    const data = JSON.stringify(body);
    const result = { header_ms: null, first_token_ms: null, total_ms: null, text: '' };
    const req = http.request({ host: '127.0.0.1', port, path: '/v1/chat/completions',
      method: 'POST', agent: false, headers: { 'Content-Type': 'application/json', 'Content-Length': Buffer.byteLength(data) } }, res => {
      result.header_ms = now() - started;
      let pending = '', raw = '';
      res.on('data', b => {
        raw += b; pending += b;
        let split;
        while ((split = pending.indexOf('\n')) >= 0) {
          const line = pending.slice(0, split); pending = pending.slice(split + 1);
          if (!line.startsWith('data: {')) continue;
          const event = JSON.parse(line.slice(6));
          if (event.error) { req.destroy(Error(JSON.stringify(event.error))); return; }
          const piece = event.choices?.[0]?.delta?.content;
          if (piece !== undefined) {
            result.first_token_ms ??= now() - started;
            result.text += piece;
          }
        }
      });
      res.on('error', reject);
      res.on('end', () => {
        try {
          assert.equal(res.statusCode, 200);
          if (body.stream) assert(raw.includes('data: [DONE]'), raw);
          else result.text = JSON.parse(raw).choices[0].message.content;
          result.total_ms = now() - started;
          check(); resolve(result);
        } catch (e) { reject(e); }
      });
    });
    req.setTimeout(180000, () => req.destroy(Error('Completion timeout')));
    req.on('error', reject); req.end(data);
  });
}
async function abandoned(body, mode = 'fin', duringDecode = false) {
  const data = JSON.stringify(body);
  const socket = net.createConnection({ host: '127.0.0.1', port });
  let raw = '', failure;
  socket.on('data', b => raw += b);
  socket.on('error', e => { failure = e; });
  const started = now();
  try {
    await new Promise((resolve, reject) => { socket.once('connect', resolve); socket.once('error', reject); });
    socket.write(`POST /v1/chat/completions HTTP/1.1\r\nHost: 127.0.0.1:${port}\r\nContent-Type: application/json\r\nContent-Length: ${Buffer.byteLength(data)}\r\nConnection: close\r\n\r\n${data}`);
    if (duringDecode) {
      while (!raw.includes('"content":')) {
        check(); if (failure) throw failure;
        assert(now() - started < 30000, 'No decode within 30s');
        await sleep(25);
      }
    } else {
      while (now() - started < 5000) { check(); await sleep(25); }
      assert(raw.includes('HTTP/1.1 200'), 'No HTTP headers before cancellation');
      assert(!raw.includes('data: '), 'Long request already generated before cancellation');
    }
    const cancelled = now();
    if (mode === 'rst') socket.resetAndDestroy();
    else socket.destroy(); // Normal TCP close; request has already been sent and drained.
    // Dispatch immediately, before waiting for server-side cancellation.
    const next = await complete(payload('Dis OK.', { max_tokens: 1 }));
    return { mode, phase: duringDecode ? 'decode' : 'prefill',
      cancel_after_ms: cancelled - started,
      body_bytes_before_cancel: Buffer.byteLength(raw.split('\r\n\r\n').slice(1).join('\r\n\r\n')),
      next, stats: await get('/v1/admin/ews/status') };
  } finally { socket.destroy(); }
}
async function run(label, exe) {
  guard();
  await new Promise((resolve, reject) => {
    const probe = net.createServer(); probe.once('error', reject);
    probe.listen(port, '127.0.0.1', () => probe.close(resolve));
  });
  const cfg = path.join(out, 'test.yaml');
  fs.writeFileSync(cfg, 'host: 127.0.0.1\nport: 18080\nauto_discover: false\ntype_k: f16\ntype_v: f16\nn_ctx: 4096\nflash_attn: true\nmodels:\n  gemma-26b-ews: C:/Users/User/models/google_gemma-4-26B-A4B-it-Q4_0.gguf\news_slots:\n  gemma-26b-ews: 16\npreload: [gemma-26b-ews]\n');
  const log = fs.openSync(path.join(out, label + '.log'), 'w');
  server = spawn(path.resolve(exe), ['--config', cfg], { cwd: root, windowsHide: true,
    env: { ...process.env, PATH: path.join(root, 'build-ews/bin/Release') + ';' + process.env.PATH,
      GGML_CUDA_DISABLE_GRAPHS: '1' }, stdio: ['ignore', log, log] });
  const entry = { label, exe, pid: server.pid, cases: [] };
  report.runs.push(entry); save();
  guardTimer = setInterval(() => {
    try { guard(); } catch (e) {
      guardFailure = e;
      if (server?.exitCode === null) server.kill(); // Only the owned verification process.
      clearInterval(guardTimer);
    }
  }, 1000);
  try {
    const started = now();
    while (true) {
      check();
      try { const catalog = await get('/v1/models'); assert.equal(catalog.data[0].id, 'gemma-26b-ews'); break; }
      catch (e) { if (now() - started > 120000) throw e; await sleep(250); }
    }
    entry.startup_ms = now() - started;
    entry.idle_short = await complete(payload('Dis OK.', { max_tokens: 1 }));
    console.log(label + ' idle: ' + JSON.stringify(entry.idle_short));
    const long = payload('Lis ces mots puis reponds OK : ' + 'alpha beta gamma delta '.repeat(120));
    entry.cases.push(await abandoned(long)); save();
    console.log(label + ' prefill FIN: ' + JSON.stringify(entry.cases.at(-1)));
    if (label === 'candidate') {
      const seed = payload('Combien font deux plus deux ?', { one_shot: false, max_tokens: 8 });
      const reference = await complete(seed);
      entry.cases.push(await abandoned(long, 'rst'));
      const afterOneShot = await complete(seed);
      assert.equal(afterOneShot.text, reference.text, 'One-shot cancellation changed persistent output');
      entry.cases.push(await abandoned({ ...long, one_shot: false }, 'fin'));
      const afterPersistent = await complete(seed);
      assert.equal(afterPersistent.text, reference.text, 'Persistent cancellation corrupted context');
      entry.cases.push(await abandoned(payload('Compte de 1 a 100.', { max_tokens: 256, one_shot: false }), 'rst', true));
      const afterDecode = await complete(seed);
      assert.equal(afterDecode.text, reference.text, 'Decode cancellation corrupted context');
      entry.recovery = { reference, afterOneShot, afterPersistent, afterDecode };
      entry.non_stream = await complete({ ...seed, stream: false });
      assert.equal(entry.non_stream.text, reference.text);
      for (const c of entry.cases) {
        assert(c.next.first_token_ms !== null && c.next.first_token_ms < 15000, JSON.stringify(c));
      }
      entry.health = await get('/health');
      console.log(label + ' recovery checks passed');
    }
    save();
  } finally {
    clearInterval(guardTimer);
    if (server.exitCode === null) {
      const exited = new Promise(resolve => server.once('exit', resolve));
      server.kill(); await exited;
    }
    fs.closeSync(log); server = null;
  }
}
(async () => {
  try {
    await run('baseline', baseline);
    await run('candidate', candidate);
    const before = report.runs[0].cases[0].next.first_token_ms;
    const after = report.runs[1].cases[0].next.first_token_ms;
    assert(before > 30000, 'Baseline did not reproduce a long blocked prefill');
    assert(after < before / 3, 'No meaningful cancellation improvement');
    report.passed = true;
  } catch (e) { report.passed = false; report.error = e.stack; process.exitCode = 1; }
  finally { report.finished = new Date().toISOString(); save(); console.log(JSON.stringify(report)); }
})();
