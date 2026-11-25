"""端到端测试：模拟一个 OpenAI 工具调用 agent 的完整多轮循环。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from mock_llm.scenario import ScenarioError, parse_scenario
from mock_llm.server import create_app

RUN_COMMAND_TOOL = {
    "type": "function",
    "function": {
        "name": "run_command",
        "description": "执行 shell 命令",
        "parameters": {
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"],
        },
    },
}

READ_FILE_TOOL = {
    "type": "function",
    "function": {
        "name": "read_file",
        "description": "读取文件",
        "parameters": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
    },
}


@pytest.fixture()
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    """场景编排模式（MOCK_AUTO=1）：自然语言消息按预写场景下发。"""
    monkeypatch.setenv("MOCK_AUTO", "1")
    src = Path(__file__).resolve().parent.parent / "scenarios"
    dst = tmp_path / "scenarios"
    dst.mkdir()
    for f in src.glob("*.yaml"):
        (dst / f.name).write_text(f.read_text(encoding="utf-8"), encoding="utf-8")
    app = create_app(dst, "mock-agent")
    with TestClient(app) as c:
        yield c


def chat(client: TestClient, messages: list[dict], tools: list[dict], stream: bool = False) -> dict:
    resp = client.post(
        "/v1/chat/completions",
        json={"model": "mock-agent", "messages": messages, "tools": tools, "stream": stream},
    )
    assert resp.status_code == 200, resp.text
    return resp.json()


def test_models_list(client: TestClient):
    data = client.get("/v1/models").json()
    assert data["data"][0]["id"] == "mock-agent"


def test_round0_returns_tool_call(client: TestClient):
    """第 1 轮：无 tool 消息 -> 返回预排第 1 步 run_command(echo)。"""
    body = chat(client, [{"role": "user", "content": "执行命令"}], [RUN_COMMAND_TOOL])
    choice = body["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    calls = choice["message"]["tool_calls"]
    assert len(calls) == 1
    call = calls[0]
    assert call["function"]["name"] == "run_command"
    assert json.loads(call["function"]["arguments"]) == {"command": "echo hello-from-mock"}
    assert call["id"].startswith("call_")
    assert "index" not in call  # 非流式响应无 index 字段
    assert choice["message"]["content"] is None


def test_round_progression(client: TestClient):
    """模拟 agent 完整循环：轮1 echo -> 轮2 pwd -> 轮3 收尾文本停止；继续追加 tool 消息则走 default。"""
    messages: list[dict] = [{"role": "user", "content": "执行命令"}]

    for turn in range(3):
        body = chat(client, messages, [RUN_COMMAND_TOOL])
        choice = body["choices"][0]
        msg = choice["message"]
        calls = msg.get("tool_calls") or []

        if turn == 0:
            assert choice["finish_reason"] == "tool_calls"
            assert json.loads(calls[0]["function"]["arguments"])["command"] == "echo hello-from-mock"
        elif turn == 1:
            assert json.loads(calls[0]["function"]["arguments"])["command"] == "pwd"
        else:
            assert choice["finish_reason"] == "stop"
            assert calls == []
            # 模板变量替换为最后一条 tool 结果
            assert "pwd-result-marker" in msg["content"]
            break  # agent 收到最终文本后正常停止

        # agent 执行工具并把结果回传
        messages.append(msg)
        for call in calls:
            result = json.dumps({"stdout": f"{call['function']['name']}-pwd-result-marker", "code": 0})
            messages.append(
                {"role": "tool", "tool_call_id": call["id"], "content": result}
            )

    # default 兑底：预排步骤耗尽后 agent 仍回传了 tool 结果
    messages.append(
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_extra",
                    "type": "function",
                    "function": {"name": "run_command", "arguments": "{}"},
                }
            ],
        }
    )
    messages.append({"role": "tool", "tool_call_id": "call_extra", "content": "extra"})
    body = chat(client, messages, [RUN_COMMAND_TOOL])
    assert body["choices"][0]["finish_reason"] == "stop"
    assert "预排命令均已执行完毕" in body["choices"][0]["message"]["content"]


def test_stream_tool_call_reassembles(client: TestClient):
    """流式：首块声明 id/name，后续 arguments 分片可拼回完整 JSON，最后 [DONE]。"""
    messages = [
        {"role": "user", "content": "执行命令"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"id": "call_prev", "type": "function", "function": {"name": "run_command", "arguments": "{}"}}
            ],
        },
        {"role": "tool", "tool_call_id": "call_prev", "content": "ok"},
    ]
    with client.stream(
        "POST",
        "/v1/chat/completions",
        json={"model": "mock-agent", "messages": messages, "tools": [RUN_COMMAND_TOOL], "stream": True},
    ) as resp:
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")
        raw = "".join(resp.iter_text())

    events = [line.removeprefix("data: ") for line in raw.splitlines() if line.startswith("data: ")]
    assert events[-1] == "[DONE]"
    chunks = [json.loads(e) for e in events[:-1]]

    # 首块带完整 tool_call 元信息
    first = chunks[0]["choices"][0]["delta"]
    assert first["tool_calls"][0]["function"]["name"] == "run_command"
    assert first["tool_calls"][0]["id"].startswith("call_")

    # 分片拼回完整 arguments
    joined = "".join(
        c["choices"][0]["delta"].get("tool_calls", [{}])[0].get("function", {}).get("arguments", "")
        for c in chunks
    )
    assert json.loads(joined) == {"command": "pwd"}

    # 收尾块 finish_reason
    assert chunks[-1]["choices"][0]["finish_reason"] == "tool_calls"


def test_scenario_matched_by_tools(client: TestClient):
    """请求带 read_file 工具 -> 自动匹配 read-file-demo 场景。"""
    body = chat(client, [{"role": "user", "content": "读文件"}], [READ_FILE_TOOL])
    call = body["choices"][0]["message"]["tool_calls"][0]
    assert call["function"]["name"] == "read_file"
    assert json.loads(call["function"]["arguments"]) == {"path": "/etc/hostname"}


def test_parallel_tool_calls(client: TestClient):
    """read-file-demo 第 2 步同时下发两个 tool_calls。"""
    messages = [
        {"role": "user", "content": "读文件"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "read_file", "arguments": '{"path": "/etc/hostname"}'},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "myhost"},
    ]
    body = chat(client, messages, [READ_FILE_TOOL])
    calls = body["choices"][0]["message"]["tool_calls"]
    assert len(calls) == 2
    assert [c["function"]["name"] for c in calls] == ["read_file", "read_file"]


def test_no_matching_scenario_returns_hint(client: TestClient):
    """工具清单不匹配任何场景 -> 200 + 提示文本（调试可见）。"""
    body = chat(client, [{"role": "user", "content": "hi"}], [])
    choice = body["choices"][0]
    assert choice["finish_reason"] == "stop"
    assert "没有场景匹配工具清单" in choice["message"]["content"]


def test_inspect_endpoints(client: TestClient):
    chat(client, [{"role": "user", "content": "执行命令"}], [RUN_COMMAND_TOOL])
    reqs = client.get("/__inspect/requests").json()
    assert reqs["total"] >= 1
    assert reqs["recent"][-1]["tools"] == ["run_command"]
    assert reqs["recent"][-1]["scenario"] == "exec-commands"

    scen = client.get("/__inspect/scenarios").json()
    assert any("exec-commands" in s for s in scen["scenarios"])

    reload = client.post("/__inspect/reload").json()
    assert reload["ok"] is True


# ---------- 场景解析单测 ----------


def test_parse_scenario_rejects_empty_step():
    with pytest.raises(ScenarioError):
        parse_scenario({"name": "bad", "steps": [{}]})


def test_parse_scenario_accepts_string_arguments():
    s = parse_scenario(
        {
            "name": "ok",
            "steps": [{"tool_calls": [{"name": "t", "arguments": '{"a": 1}'}]}],
        }
    )
    assert s.steps[0].tool_calls[0].arguments == {"a": 1}
