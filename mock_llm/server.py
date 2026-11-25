"""OpenAI 兼容 mock 服务器。

端点：
- GET  /v1/models                模型列表
- POST /v1/chat/completions      对话补全（支持 stream / tool_calls / 模板变量）
- GET  /__inspect/scenarios      查看已加载场景与步骤序列
- GET  /__inspect/requests       查看最近请求的处理摘要
- POST /__inspect/reload         热重载场景目录

每个请求都会：
1. 解析请求中的 tools 清单并打印日志（先看清有哪些工具）；
2. 按 requires_tools 匹配场景；
3. 统计 role=="tool" 消息数得到已完成轮数，取出预写的下一步骤；
4. 以 OpenAI 格式返回 tool_calls / 文本（流式或非流式）。
"""

from __future__ import annotations

import asyncio
import json
import os
import time
import uuid
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

try:
    from .interactive import format_tool_list, route_interactive
    from .scenario import (
        Scenario,
        Step,
        count_tool_rounds,
        extract_last_tool_result,
        load_scenario_dir,
        match_scenario,
    )
except ImportError:  # 允许直接运行本文件：python mock_llm/server.py
    from interactive import format_tool_list, route_interactive
    from scenario import (
        Scenario,
        Step,
        count_tool_rounds,
        extract_last_tool_result,
        load_scenario_dir,
        match_scenario,
    )

STREAM_CHUNK_SIZE = 24  # 流式时 arguments JSON 的分片大小（字符）
MAX_REQUEST_LOG = 200


class AppState:
    """全局状态：场景列表 + 请求日志。"""

    def __init__(self, scenario_dir: Path, model_name: str) -> None:
        self.scenario_dir = scenario_dir
        self.model_name = model_name
        self.scenarios: list[Scenario] = load_scenario_dir(scenario_dir)
        self.request_log: deque[dict] = deque(maxlen=MAX_REQUEST_LOG)
        self.request_seq = 0
        self.last_tool_defs: dict[str, dict] = {}  # 最近一次请求看到的工具定义（写场景用）
        self.last_messages: list[dict] = []  # 最近一次请求的 messages（诊断路由问题用）

    def reload(self) -> None:
        self.scenarios = load_scenario_dir(self.scenario_dir)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _new_id() -> str:
    return "chatcmpl-" + uuid.uuid4().hex[:24]


def _log(line: str) -> None:
    print(f"[mock-llm] {line}", flush=True)


def _requested_tools(body: dict) -> list[str]:
    """宽松提取请求声明的工具名（支持 OpenAI function 格式）。"""
    names = []
    for t in body.get("tools") or []:
        if not isinstance(t, dict):
            continue
        fn = t.get("function")
        if isinstance(fn, dict) and fn.get("name"):
            names.append(str(fn["name"]))
        elif t.get("name"):  # 兼容旧版 tools=[{type:"function", name:...}] 平铺格式
            names.append(str(t["name"]))
    return names


def _tool_definitions(body: dict) -> dict[str, dict]:
    """提取请求声明的工具定义：{工具名: {description, parameters(JSON Schema)}}。"""
    defs: dict[str, dict] = {}
    for t in body.get("tools") or []:
        if not isinstance(t, dict):
            continue
        fn = t.get("function")
        if isinstance(fn, dict) and fn.get("name"):
            defs[str(fn["name"])] = {
                "description": fn.get("description", ""),
                "parameters": fn.get("parameters") or {},
            }
        elif t.get("name"):
            defs[str(t["name"])] = {
                "description": t.get("description", ""),
                "parameters": t.get("parameters") or {},
            }
    return defs


def _warn_argument_mismatch(step: Step, tool_defs: dict[str, dict]) -> None:
    """场景引用的工具在请求中存在、但 arguments 与声明的 schema 不符时打 WARN。"""
    for call in step.tool_calls:
        schema = (tool_defs.get(call.name) or {}).get("parameters") or {}
        props = set(schema.get("properties") or {})
        if not props:  # schema 缺失或无属性定义，无法校验
            continue
        extra = sorted(set(call.arguments) - props)
        missing = sorted(set(schema.get("required") or []) - set(call.arguments))
        if extra or missing:
            parts = []
            if extra:
                parts.append(f"schema 外的参数 {extra}")
            if missing:
                parts.append(f"缺少必填参数 {missing}")
            _log(f"WARN 工具 {call.name} 参数与声明不符: {'; '.join(parts)}（GET /__inspect/tools 可查真实 schema）")


def _validate_api_key(request: Request) -> bool:
    """校验 Authorization: Bearer <key>。未设置 MOCK_LLM_KEY 环境变量时跳过校验。"""
    expected = os.environ.get("MOCK_LLM_KEY")
    if not expected:
        return True
    auth = request.headers.get("authorization", "")
    return auth == f"Bearer {expected}"


def _build_step_response(
    state: AppState,
    body: dict,
    scenario: Scenario | None,
    step: Step,
    step_index: int | str,
) -> tuple[str, list[dict], str]:
    """构造 tool_calls 载荷与 finish_reason，并打印本轮处理日志。"""
    tool_calls = []
    for i, call in enumerate(step.tool_calls):
        tool_calls.append(
            {
                "id": f"call_{uuid.uuid4().hex[:24]}",
                "type": "function",
                "index": i,
                "function": {
                    "name": call.name,
                    "arguments": json.dumps(call.arguments, ensure_ascii=False),
                },
            }
        )
    finish_reason = "tool_calls" if tool_calls else "stop"

    action = step.describe()
    _log(
        f"req#{state.request_seq} model={body.get('model', state.model_name)} "
        f"tools={_requested_tools(body)} tool_rounds={count_tool_rounds(body.get('messages') or [])} "
        f"scenario={scenario.name if scenario else None} step={step_index} -> {action}"
    )
    return finish_reason, tool_calls, action


def _record(
    state: AppState,
    request_id: str,
    body: dict,
    scenario: Scenario | None,
    step_index: int | str,
    finish_reason: str,
    action: str,
) -> None:
    state.request_log.append(
        {
            "seq": state.request_seq,
            "id": request_id,
            "time": _now_iso(),
            "model": body.get("model", state.model_name),
            "stream": bool(body.get("stream")),
            "tools": _requested_tools(body),
            "tool_rounds": count_tool_rounds(body.get("messages") or []),
            "scenario": scenario.name if scenario else None,
            "step": step_index,
            "finish_reason": finish_reason,
            "action": action,
        }
    )


def _create_app(state: AppState) -> FastAPI:
    app = FastAPI(title="mock-llm", version="0.1.0")
    app.state.mock = state

    # ---------- /v1/models ----------

    @app.get("/v1/models")
    def list_models() -> dict:
        return {
            "object": "list",
            "data": [
                {"id": state.model_name, "object": "model", "created": 0, "owned_by": "mock-llm"},
            ],
        }

    # ---------- /v1/chat/completions ----------

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        if not _validate_api_key(request):
            return JSONResponse(
                status_code=401,
                content={"error": {"message": "invalid api key (设置环境变量 MOCK_LLM_KEY 启用校验)"}},
            )
        try:
            body = await request.json()
        except Exception:
            return JSONResponse(status_code=400, content={"error": {"message": "invalid json body"}})
        if not isinstance(body, dict):
            return JSONResponse(status_code=400, content={"error": {"message": "body must be a json object"}})

        state.request_seq += 1
        request_id = _new_id()
        created = int(time.time())
        model = body.get("model", state.model_name)
        stream = bool(body.get("stream"))
        messages = body.get("messages") or []
        rounds = count_tool_rounds(messages)
        requested_tools = _requested_tools(body)
        tool_defs = _tool_definitions(body)
        state.last_tool_defs = tool_defs
        state.last_messages = messages

        if os.environ.get("MOCK_AUTO") == "1":
            # 自动模式：跳过交互路由，按预写场景逐轮下发
            scenario = match_scenario(state.scenarios, requested_tools)
            if scenario is None:
                _log(f"req#{state.request_seq} tools={requested_tools} -> 无匹配场景，返回提示文本")
                text = f"[mock-llm] 没有场景匹配工具清单 {requested_tools}，请在 scenarios/ 目录添加场景或调整 requires_tools。"
                _record(state, request_id, body, None, "-", "stop", text)
                return _text_response(request_id, created, model, text, stream)
        else:
            # 交互模式（默认）：按消息格式手动驱动，见 mock_llm/interactive.py
            interactive = route_interactive(messages, tool_defs)
            if interactive is not None:
                label = interactive["label"]
                if interactive["kind"] == "text":
                    content = interactive["content"]
                    _log(f"req#{state.request_seq} interactive[{label}] -> 文本 {len(content)} 字符")
                    _record(state, request_id, body, None, f"interactive:{label}", "stop", f"text[{label}]")
                    return _text_response(request_id, created, model, content, stream)
                step = Step(tool_calls=interactive["calls"])
                _warn_argument_mismatch(step, tool_defs)
                finish_reason, tool_calls, action = _build_step_response(
                    state, body, None, step, f"interactive:{label}"
                )
                _record(state, request_id, body, None, f"interactive:{label}", finish_reason, action)
                if stream:
                    return _stream_response(request_id, created, model, finish_reason, tool_calls, step, messages)
                return _json_response(request_id, created, model, finish_reason, tool_calls, step, messages)

            _log(f"req#{state.request_seq} interactive[tools_list] -> 默认返回工具清单")
            _record(state, request_id, body, None, "interactive:tools_list", "stop", "text[tools_list]")
            return _text_response(request_id, created, model, format_tool_list(tool_defs), stream)

        step_index: int | str = rounds if rounds < len(scenario.steps) else "default"
        step = scenario.pick_step(rounds)
        _warn_argument_mismatch(step, tool_defs)
        finish_reason, tool_calls, action = _build_step_response(state, body, scenario, step, step_index)
        _record(state, request_id, body, scenario, step_index, finish_reason, action)

        if stream:
            return _stream_response(request_id, created, model, finish_reason, tool_calls, step, messages)
        return _json_response(request_id, created, model, finish_reason, tool_calls, step, messages)

    # ---------- 调试端点 ----------

    @app.get("/__inspect/scenarios")
    def inspect_scenarios() -> dict:
        return {
            "model": state.model_name,
            "scenario_dir": str(state.scenario_dir),
            "scenarios": [s.describe() for s in state.scenarios],
        }

    @app.get("/__inspect/requests")
    def inspect_requests() -> dict:
        return {"total": state.request_seq, "recent": list(state.request_log)}

    @app.get("/__inspect/tools")
    def inspect_tools() -> dict:
        """最近一次请求中 agent 声明的工具定义。写场景的 arguments 前先看这里核对参数名。"""
        return {"count": len(state.last_tool_defs), "tools": state.last_tool_defs}

    @app.get("/__inspect/last_request")
    def inspect_last_request() -> dict:
        """最近一次请求的 messages 原文（每条内容截断 800 字符），诊断交互路由未命中时用。"""
        trimmed = []
        for m in state.last_messages[-20:]:
            if not isinstance(m, dict):
                trimmed.append(m)
                continue
            content = m.get("content")
            if isinstance(content, str) and len(content) > 800:
                content = content[:800] + f"...(共 {len(m['content'])} 字符)"
            trimmed.append({k: content if k == "content" else v for k, v in m.items()})
        return {"count": len(state.last_messages), "messages": trimmed}

    @app.post("/__inspect/reload")
    def inspect_reload() -> dict:
        try:
            state.reload()
        except Exception as e:  # 重载失败保留旧场景
            return {"ok": False, "error": str(e), "loaded": len(state.scenarios)}
        _log(f"场景已重载: {len(state.scenarios)} 个 <- {state.scenario_dir}")
        return {"ok": True, "loaded": len(state.scenarios)}

    @app.head("/")
    @app.get("/")
    def root() -> Response:
        return Response(status_code=200)

    return app


# ---------- 响应构造 ----------


def _base_chunk(request_id: str, created: int, model: str) -> dict:
    return {"id": request_id, "object": "chat.completion.chunk", "created": created, "model": model}


def _json_response(
    request_id: str,
    created: int,
    model: str,
    finish_reason: str,
    tool_calls: list[dict],
    step: Step,
    messages: list[dict],
) -> dict:
    message: dict
    if tool_calls:
        # 非流式 tool_calls 响应里去掉 index 字段（OpenAI 非流式无 index）
        calls = [{k: v for k, v in c.items() if k != "index"} for c in tool_calls]
        message = {"role": "assistant", "content": None, "tool_calls": calls}
    else:
        content = (step.content or "").replace("{last_tool_result}", extract_last_tool_result(messages))
        message = {"role": "assistant", "content": content}
    return {
        "id": request_id,
        "object": "chat.completion",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        },
        "mock": True,
    }


def _stream_response(
    request_id: str,
    created: int,
    model: str,
    finish_reason: str,
    tool_calls: list[dict],
    step: Step,
    messages: list[dict],
) -> StreamingResponse:
    async def generate():
        if tool_calls:
            # 首块：完整声明每个 tool_call（id/name），arguments 从空串开始增量下发
            first_calls = []
            for call in tool_calls:
                first_calls.append(
                    {
                        "index": call["index"],
                        "id": call["id"],
                        "type": "function",
                        "function": {"name": call["function"]["name"], "arguments": ""},
                    }
                )
            chunk = _base_chunk(request_id, created, model)
            chunk["choices"] = [
                {
                    "index": 0,
                    "delta": {"role": "assistant", "content": None, "tool_calls": first_calls},
                    "finish_reason": None,
                }
            ]
            yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"

            # arguments 分片（每个工具独立按 index 增量）
            for call in tool_calls:
                args = call["function"]["arguments"]
                for i in range(0, len(args), STREAM_CHUNK_SIZE):
                    piece = args[i : i + STREAM_CHUNK_SIZE]
                    chunk = _base_chunk(request_id, created, model)
                    chunk["choices"] = [
                        {
                            "index": 0,
                            "delta": {
                                "tool_calls": [
                                    {"index": call["index"], "function": {"arguments": piece}}
                                ]
                            },
                            "finish_reason": None,
                        }
                    ]
                    yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
                    await asyncio.sleep(0)
        else:
            content = (step.content or "").replace("{last_tool_result}", extract_last_tool_result(messages))
            for i in range(0, len(content), STREAM_CHUNK_SIZE):
                piece = content[i : i + STREAM_CHUNK_SIZE]
                chunk = _base_chunk(request_id, created, model)
                chunk["choices"] = [
                    {"index": 0, "delta": {"role": "assistant", "content": piece}, "finish_reason": None}
                ]
                yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
                await asyncio.sleep(0)

        # 收尾块
        chunk = _base_chunk(request_id, created, model)
        chunk["choices"] = [{"index": 0, "delta": {}, "finish_reason": finish_reason}]
        yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


def _text_response(
    request_id: str, created: int, model: str, text: str, stream: bool
):
    if stream:
        return _stream_response(request_id, created, model, "stop", [], Step(content=text), [])
    return _json_response(request_id, created, model, "stop", [], Step(content=text), [])


def create_app(scenario_dir: Path, model_name: str = "mock-agent") -> FastAPI:
    return _create_app(AppState(scenario_dir, model_name))


if __name__ == "__main__":
    # 直接运行本文件：默认用项目根目录下的 scenarios/
    import uvicorn

    _scenario_dir = Path(__file__).resolve().parent.parent / "scenarios"
    _app = create_app(_scenario_dir)
    print(f"[mock-llm] 模型: mock-agent  场景目录: {_scenario_dir}")
    print("[mock-llm] http://127.0.0.1:8789/v1/chat/completions")
    uvicorn.run(_app, host="127.0.0.1", port=8789, log_level="warning")
