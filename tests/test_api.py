#!/usr/bin/env python3
"""EIE API tests — run against a running server. Exits non-zero on failure."""
import requests, sys

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8080"

SKIP = object()

def loaded_models():
    r = requests.get(f"{BASE}/v1/models")
    assert r.status_code == 200
    return [m["id"] for m in r.json().get("data", [])]

def test_health():
    r = requests.get(f"{BASE}/health")
    assert r.status_code == 200, f"status {r.status_code}"
    print(f"  health: {r.json()}")

def test_models():
    models = loaded_models()
    print(f"  models: {models}")

def test_chat():
    models = loaded_models()
    if not models:
        print("  chat: no model loaded")
        return SKIP
    r = requests.post(f"{BASE}/v1/chat/completions", json={
        "model": models[0],
        "messages": [{"role": "user", "content": "Hello"}],
        "max_tokens": 32
    })
    assert r.status_code == 200, f"status {r.status_code}: {r.text[:200]}"
    content = r.json()["choices"][0]["message"]["content"]
    assert isinstance(content, str)
    print(f"  chat[{models[0]}]: {content[:80]!r}")

def test_batch():
    r = requests.post(f"{BASE}/v1/batch/execute", json={
        "group": "core",
        "messages": [{"role": "user", "content": "Test"}]
    })
    if r.status_code == 404:
        print("  batch: no 'core' group configured")
        return SKIP
    assert r.status_code == 200, f"status {r.status_code}: {r.text[:200]}"
    body = r.json()
    assert "responses" in body and "status" in body
    print(f"  batch: {body['status']} ({body['completed']}/{body['required']})")

def test_metrics():
    r = requests.get(f"{BASE}/metrics")
    assert r.status_code == 200, f"status {r.status_code}"
    print(f"  metrics: {len(r.text)} bytes")

if __name__ == "__main__":
    print(f"Testing EIE at {BASE}")
    failures = 0
    for t in [test_health, test_models, test_chat, test_batch, test_metrics]:
        try:
            result = t()
            print(f"  {'SKIP' if result is SKIP else 'PASS'}: {t.__name__}")
        except Exception as e:
            failures += 1
            print(f"  FAIL: {t.__name__}: {e}")
    sys.exit(1 if failures else 0)
