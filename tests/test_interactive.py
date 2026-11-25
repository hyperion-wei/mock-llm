"""交互模式测试：消息格式路由（默认模式，不设 MOCK_AUTO）。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from mock_llm.server import create_app

PWSH_TOOL = {
    "type": "function",
    "function": {
        "name": "pwsh",
        "description": "执行 PowerShell 命令",
        "parameters": {
            "type": "object",
            "properties": {"command": {"type": "string", "description": "要执行的命令"}},
            "required": ["command"],
        },
    },
}

WRITE_TOOL = {
    "type": "function",
    "function": {
        "name": "write",
        "description": "写文件",
        "parameters": {
            "type": "object",
            "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
            "required": ["path", "content"],
        },
    },
}


@pytest.fixture()
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    """交互模式（默认）：交互格式优先，未命中格式返回工具清单。"""
    monkeypatch.delenv("MOCK_AUTO", raising=False)
    scenario_dir = tmp_path / "scenarios"
    scenario_dir.mkdir()
    (scenario_dir / "dummy.yaml").write_text(
        "name: dummy\nsteps:\n  - content: should-not-trigger\n", encoding="utf-8"
    )
    app = create_app(scenario_dir, "mock-agent")
    with TestClient(app) as c:
        yield c


def chat_text(client: TestClient, messages: list[dict]) -> str:
    body = client.post(
        "/v1/chat/completions",
        json={"model": "mock-agent", "messages": messages, "tools": [PWSH_TOOL, WRITE_TOOL]},
    )
    assert body.status_code == 200, body.text
    return body.json()["choices"][0]["message"]["content"]


def test_default_message_returns_tool_list(client: TestClient):
    """初始对话（不按格式）-> 返回工具清单。"""
    content = chat_text(client, [{"role": "user", "content": "你好，我们开始"}])
    assert "工具 2 个" in content
    assert "- pwsh" in content and "- write" in content
    assert "工具名 {参数JSON}" in content


def test_tools_command(client: TestClient):
    assert "pwsh" in chat_text(client, [{"role": "user", "content": "/tools"}])


def test_tool_detail_by_bare_name(client: TestClient):
    """发送恰好一个工具名 -> 返回参数 schema 与必填参数示例。"""
    content = chat_text(client, [{"role": "user", "content": "pwsh"}])
    assert "工具 `pwsh`" in content
    assert '"command"' in content
    assert "调用示例" in content
    assert 'pwsh {"command"' in content
    assert "/call" in content  # 强制下发前缀提示


def test_tool_detail_by_command(client: TestClient):
    content = chat_text(client, [{"role": "user", "content": "/tool write"}])
    assert "工具 `write`" in content
    assert '"content"' in content


def test_tool_detail_unknown(client: TestClient):
    content = chat_text(client, [{"role": "user", "content": "/tool nope"}])
    assert "不在本次请求的工具清单中" in content


def test_call_dispatches_tool_call(client: TestClient):
    """工具名 + JSON -> 下发标准 tool_calls。"""
    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "mock-agent",
            "messages": [{"role": "user", "content": 'pwsh {"command": "Get-Date"}'}],
            "tools": [PWSH_TOOL],
        },
    )
    choice = resp.json()["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    call = choice["message"]["tool_calls"][0]
    assert call["function"]["name"] == "pwsh"
    assert json.loads(call["function"]["arguments"]) == {"command": "Get-Date"}


def test_call_replay_returns_result_note(client: TestClient):
    """工具执行回传后，重发同一格式 -> 返回结果确认文本（闭环展示执行结果）。"""
    call_msg = 'pwsh {"command": "Get-Date"}'
    messages = [
        {"role": "user", "content": call_msg},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"id": "c1", "type": "function", "function": {"name": "pwsh", "arguments": '{"command": "Get-Date"}'}}
            ],
        },
        {"role": "tool", "tool_call_id": "c1", "content": "2026-11-25 result"},
    ]
    content = chat_text(client, messages)
    assert "已执行" in content
    assert "2026-11-25 result" in content


def test_call_missing_required_blocked(client: TestClient):
    """缺必填参数 -> 拦截并给出完整可复制的示例，不下发。"""
    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "mock-agent",
            "messages": [{"role": "user", "content": 'write {"path": "a.txt"}'}],
            "tools": [PWSH_TOOL, WRITE_TOOL],
        },
    )
    choice = resp.json()["choices"][0]
    assert choice["finish_reason"] == "stop"
    content = choice["message"]["content"]
    assert "缺少必填参数" in content and "content" in content
    assert "完整调用示例" in content
    assert 'write {"path"' in content and '"content"' in content  # 示例含全部必填字段


def test_call_force_prefix_bypasses_check(client: TestClient):
    """月/call 前缀跳过必填校验强制下发。"""
    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "mock-agent",
            "messages": [{"role": "user", "content": '/call write {"path": "a.txt"}'}],
            "tools": [PWSH_TOOL, WRITE_TOOL],
        },
    )
    choice = resp.json()["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    call = choice["message"]["tool_calls"][0]
    assert call["function"]["name"] == "write"
    assert json.loads(call["function"]["arguments"]) == {"path": "a.txt"}


def test_call_invalid_json(client: TestClient):
    content = chat_text(client, [{"role": "user", "content": "pwsh {bad json}"}])
    assert "JSON 解析失败" in content


def test_call_with_trailing_injection(client: TestClient):
    """客户端在消息后追加注入内容（IDE 上下文等）-> 首行路由 + raw_decode 仍能命中。"""
    msg = 'pwsh {"command": "Get-Date"}\n\n<ide-context>open files: main.py</ide-context>'
    resp = client.post(
        "/v1/chat/completions",
        json={"model": "mock-agent", "messages": [{"role": "user", "content": msg}], "tools": [PWSH_TOOL]},
    )
    choice = resp.json()["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    assert json.loads(choice["message"]["tool_calls"][0]["function"]["arguments"]) == {"command": "Get-Date"}


def test_bare_tool_name_with_trailing_injection(client: TestClient):
    """首行为工具名、尾部被追加内容 -> 仍返回参数详情。"""
    msg = "pwsh\n(客户端注入的环境信息，不参与路由)"
    assert "工具 `pwsh`" in chat_text(client, [{"role": "user", "content": msg}])


def test_call_truncated_json_reports_error(client: TestClient):
    """JSON 被截断时给出明确报错而非静默走默认。"""
    content = chat_text(client, [{"role": "user", "content": 'pwsh {"command": "Ge'}])
    assert "JSON 解析失败" in content


def test_exec_disabled_by_default(client: TestClient):
    """默认禁用 /exec（公网防 RCE）：返回禁用提示，不执行任何命令。"""
    content = chat_text(client, [{"role": "user", "content": '/exec pwsh {"command": "echo should-not-run"}'}])
    assert "/exec 已禁用" in content
    assert "MOCK_EXEC_ENABLED=1" in content
    assert "should-not-run" not in content


def test_exec_runs_locally(client: TestClient, monkeypatch: pytest.MonkeyPatch):
    """显式开启后 /exec -> mock 本地执行命令字段并返回结果。"""
    monkeypatch.setenv("MOCK_EXEC_ENABLED", "1")
    content = chat_text(client, [{"role": "user", "content": '/exec pwsh {"command": "echo hi-from-mock"}'}])
    assert "[mock-exec]" in content
    assert "exit=0" in content
    assert "hi-from-mock" in content


def test_exec_no_command_key(client: TestClient, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("MOCK_EXEC_ENABLED", "1")
    content = chat_text(client, [{"role": "user", "content": '/exec write {"path": "a.txt"}'}])
    assert "没有命令字段" in content


def test_help(client: TestClient):
    assert "/exec" in chat_text(client, [{"role": "user", "content": "/help"}])


def test_stream_interactive_call(client: TestClient):
    """交互 tool_calls 的流式响应同样按 SSE 分片输出。"""
    with client.stream(
        "POST",
        "/v1/chat/completions",
        json={
            "model": "mock-agent",
            "stream": True,
            "messages": [{"role": "user", "content": 'pwsh {"command": "Get-Date"}'}],
            "tools": [PWSH_TOOL],
        },
    ) as resp:
        raw = "".join(resp.iter_text())
    events = [line.removeprefix("data: ") for line in raw.splitlines() if line.startswith("data: ")]
    assert events[-1] == "[DONE]"
    joined = "".join(
        json.loads(e)["choices"][0]["delta"].get("tool_calls", [{}])[0].get("function", {}).get("arguments", "")
        for e in events[:-1]
    )
    assert json.loads(joined) == {"command": "Get-Date"}
