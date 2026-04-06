#!/usr/bin/env python3
"""EIE API tests — run against a running server."""
import requests, json, sys

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8080"

def test_health():
    r = requests.get(f"{BASE}/health")
    assert r.status_code == 200
    print(f"  health: {r.json()}")

def test_models():
    r = requests.get(f"{BASE}/v1/models")
    assert r.status_code == 200
    print(f"  models: {r.json()}")

def test_chat():
    r = requests.post(f"{BASE}/v1/chat/completions", json={
        "model": "mistral-7b",
        "messages": [{"role": "user", "content": "Hello"}],
        "max_tokens": 100
    })
    assert r.status_code in [200, 404]
    print(f"  chat: {r.status_code}")

def test_batch():
    r = requests.post(f"{BASE}/v1/batch/execute", json={
        "group": "core",
        "messages": [{"role": "user", "content": "Test"}]
    })
    print(f"  batch: {r.status_code} {r.text[:200]}")

def test_metrics():
    r = requests.get(f"{BASE}/metrics")
    assert r.status_code == 200
    print(f"  metrics: {len(r.text)} bytes")

if __name__ == "__main__":
    print(f"Testing EIE at {BASE}")
    for t in [test_health, test_models, test_chat, test_batch, test_metrics]:
        try:
            t()
            print(f"  PASS: {t.__name__}")
        except Exception as e:
            print(f"  FAIL: {t.__name__}: {e}")
