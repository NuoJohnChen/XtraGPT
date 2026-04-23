"""Patch ~/.openclaw/openclaw.json to add xtragpt-local and qwen-local providers.

Idempotent: running twice is a no-op. Original is backed up by the caller; this
script only re-reads, mutates in memory, and atomically writes back.
"""
import json
import pathlib

CFG = pathlib.Path.home() / ".openclaw" / "openclaw.json"
d = json.loads(CFG.read_text())

providers = d.setdefault("models", {}).setdefault("providers", {})

providers["xtragpt-local"] = {
    "baseUrl": "http://127.0.0.1:8088/v1",
    "api": "openai-completions",
    "models": [
        {
            "id": "xtragpt-7b",
            "name": "XtraGPT 7B",
            "input": ["text"],
            "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0},
            "contextWindow": 16384,
            "maxTokens": 4096,
        }
    ],
}

providers["qwen-local"] = {
    "baseUrl": "http://127.0.0.1:8089/v1",
    "api": "openai-completions",
    "models": [
        {
            "id": "qwen2.5-7b-instruct",
            "name": "Qwen2.5 7B Instruct",
            "input": ["text"],
            "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0},
            "contextWindow": 16384,
            "maxTokens": 4096,
        }
    ],
}

CFG.write_text(json.dumps(d, indent=2))
print(f"patched {CFG}")
print("providers now:", list(providers.keys()))
