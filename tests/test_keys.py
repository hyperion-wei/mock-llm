"""LLM API Key 记录功能测试。"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from mock_llm.server import create_app


@pytest.fixture()
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.delenv("MOCK_AUTO", raising=False)
    monkeypatch.delenv("MOCK_LLM_KEY", raising=False)
    scenario_dir = tmp_path / "scenarios"
    scenario_dir.mkdir()
    (scenario_dir / "dummy.yaml").write_text(
        "name: dummy\nsteps:\n  - content: x\n", encoding="utf-8"
    )
    app = create_app(scenario_dir, "mock-agent")
    with TestClient(app) as c:
        yield c


def _chat(client: TestClient, headers: dict | None = None):
    return client.post(
        "/v1/chat/completions",
        json={"model": "mock-agent", "messages": [{"role": "user", "content": "hi"}]},
        headers=headers or {},
    )


def test_bearer_key_recorded(client: TestClient):
    _chat(client, {"Authorization": "Bearer sk-test-123"})
    data = client.get("/__inspect/keys").json()
    assert data["total_unique"] == 1
    entry = data["keys"][0]
    assert entry["key"] == "sk-test-123"
    assert entry["sources"] == ["Authorization: Bearer"]
    assert entry["count"] == 1
    assert entry["first_seen"] and entry["last_seen"]


def test_duplicate_key_counts_up(client: TestClient):
    _chat(client, {"Authorization": "Bearer sk-same"})
    _chat(client, {"Authorization": "Bearer sk-same"})
    data = client.get("/__inspect/keys").json()
    assert data["total_unique"] == 1
    assert data["keys"][0]["count"] == 2


def test_x_api_key_source(client: TestClient):
    _chat(client, {"x-api-key": "sk-anthropic-1"})
    entry = client.get("/__inspect/keys").json()["keys"][0]
    assert entry["key"] == "sk-anthropic-1"
    assert entry["sources"] == ["x-api-key"]


def test_api_key_header_source(client: TestClient):
    _chat(client, {"api-key": "sk-azure-1"})
    entry = client.get("/__inspect/keys").json()["keys"][0]
    assert entry["key"] == "sk-azure-1"
    assert entry["sources"] == ["api-key"]


def test_bearer_without_prefix(client: TestClient):
    _chat(client, {"Authorization": "raw-key-value"})
    entry = client.get("/__inspect/keys").json()["keys"][0]
    assert entry["key"] == "raw-key-value"
    assert entry["sources"] == ["Authorization"]


def test_no_key_not_recorded(client: TestClient):
    _chat(client)
    assert client.get("/__inspect/keys").json()["total_unique"] == 0
    recent = client.get("/__inspect/requests").json()["recent"]
    assert recent[-1]["llm_key"] is None


def test_requests_log_contains_llm_key(client: TestClient):
    _chat(client, {"Authorization": "Bearer sk-log-1"})
    recent = client.get("/__inspect/requests").json()["recent"]
    assert recent[-1]["llm_key"] == "sk-log-1"


def test_keys_clear(client: TestClient):
    _chat(client, {"Authorization": "Bearer sk-will-clear"})
    assert client.post("/__inspect/keys/clear").json() == {"ok": True}
    assert client.get("/__inspect/keys").json()["total_unique"] == 0


def test_key_recorded_even_when_auth_fails(client: TestClient, monkeypatch: pytest.MonkeyPatch):
    """设置了 MOCK_LLM_KEY 后，错误 key 被拒绝 401，但仍被记录（排查 401 用）。"""
    monkeypatch.setenv("MOCK_LLM_KEY", "sk-expected")
    resp = _chat(client, {"Authorization": "Bearer sk-wrong-key"})
    assert resp.status_code == 401
    data = client.get("/__inspect/keys", headers={"Authorization": "Bearer sk-expected"}).json()
    assert data["keys"][0]["key"] == "sk-wrong-key"


def test_inspect_protected_when_key_set(client: TestClient, monkeypatch: pytest.MonkeyPatch):
    """公网部署保护：设置 MOCK_LLM_KEY 后 /__inspect/* 需要 Bearer 鉴权。"""
    monkeypatch.setenv("MOCK_LLM_KEY", "sk-guard")
    assert client.get("/__inspect/keys").status_code == 401
    assert client.get("/__inspect/requests").status_code == 401
    assert client.get("/__inspect/scenarios").status_code == 401
    assert client.post("/__inspect/reload").status_code == 401
    ok = client.get("/__inspect/keys", headers={"Authorization": "Bearer sk-guard"})
    assert ok.status_code == 200


def test_inspect_open_without_key(client: TestClient):
    """纯本机使用（未设 MOCK_LLM_KEY）时 inspect 端点保持开放。"""
    assert client.get("/__inspect/keys").status_code == 200
