"""聊天式交互模式：用特定格式的消息手动驱动 mock，无需预写场景。

对请求中最后一条 user 消息做路由：
- 任意其他内容 / /tools              -> 返回工具清单（名称 + 描述）
- /tool <名称>（或消息恰为工具名）   -> 返回该工具的参数 schema 与调用示例
- <名称> <JSON参数>                  -> 下发标准 tool_calls，由客户端真实执行；
                                        结果回传后再次命中同一格式时返回结果确认文本
- /exec <名称> <JSON参数>            -> mock 本地执行参数中的命令字段并返回结果
- /help                              -> 指令说明
"""

from __future__ import annotations

import json
import os
import re
import subprocess

try:
    from .scenario import ToolCall, extract_last_tool_result
except ImportError:  # 允许随 server.py 直接运行
    from scenario import ToolCall, extract_last_tool_result

# /exec 时在工具参数里识别命令字段的候选键名
COMMAND_PARAM_KEYS = ("command", "cmd", "script", "shell", "code", "expression")
EXEC_TIMEOUT = 30
EXEC_OUTPUT_LIMIT = 4000

# 首行路由：工具名开头 + JSON（不锚定末尾，兼容客户端在消息后追加注入内容）
_LINE_CALL_RE = re.compile(r"^([^\s{]+)\s*(\{.*)$", re.DOTALL)

HELP_TEXT = """[mock-llm] 交互模式可用指令：
- 直接发任意内容              查看工具清单
- /tools                      查看工具清单
- /tool <工具名>              查看工具参数 schema 与调用示例
- <工具名> <JSON参数>         下发 tool_calls，由客户端真实执行并回传结果
- /call <工具名> <JSON参数>   跳过必填校验强制下发
- /exec <工具名> <JSON参数>   服务端本地执行（默认禁用，服务端设 MOCK_EXEC_ENABLED=1 开启）
- /help                       显示本说明"""


def _last_user_text(messages: list[dict]) -> str | None:
    """取最后一条 user 消息的文本（兼容多模态数组格式）。"""
    for message in reversed(messages):
        if isinstance(message, dict) and message.get("role") == "user":
            content = message.get("content")
            if isinstance(content, str):
                return content.strip()
            if isinstance(content, list):
                parts = [
                    p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text"
                ]
                return "\n".join(p for p in parts if p).strip() or None
            return None
    return None


def _executed_since_last_user(messages: list[dict]) -> bool:
    """最后一条 user 消息之后是否已有 tool 结果回传（防止同一指令被重复下发）。"""
    last_user_idx = -1
    for i, message in enumerate(messages):
        if isinstance(message, dict) and message.get("role") == "user":
            last_user_idx = i
    return any(
        isinstance(m, dict) and m.get("role") == "tool" for m in messages[last_user_idx + 1 :]
    )


def format_tool_list(tool_defs: dict[str, dict]) -> str:
    if not tool_defs:
        return "[mock-llm] 本次请求未携带工具定义（tools 为空）。"
    lines = [f"[mock-llm] 当前请求携带工具 {len(tool_defs)} 个："]
    for name, d in tool_defs.items():
        desc = (d.get("description") or "").strip().splitlines()[0][:60]
        lines.append(f"- {name}" + (f" — {desc}" if desc else ""))
    lines.append("")
    lines.append("发送 `工具名` 查看参数；发送 `工具名 {参数JSON}` 下发调用；/exec 前缀由 mock 本地执行；/help 查看全部指令。")
    return "\n".join(lines)


def _placeholder(spec: dict) -> object:
    """按参数类型生成示例占位值；字符串类型用它的 description 截断填充。"""
    if spec.get("type") == "string":
        desc = (spec.get("description") or "").strip()
        return (desc[:30] + "…") if len(desc) > 30 else (desc or "...")
    return {"number": 0, "integer": 0, "boolean": False}.get(spec.get("type"), None)


def _example_call(name: str, params: dict) -> str:
    """生成仅含必填参数的调用示例行；无必填时取前两个可选字段示意。"""
    props = params.get("properties") or {}
    required = params.get("required") or []
    keys = required or list(props)[:2]
    example = {k: _placeholder(props.get(k) or {}) for k in keys if k in props or not required}
    return f"{name} {json.dumps(example, ensure_ascii=False)}"


def format_tool_detail(name: str, tool_defs: dict[str, dict]) -> str:
    d = tool_defs.get(name)
    if d is None:
        return f"[mock-llm] 工具 `{name}` 不在本次请求的工具清单中，发送 /tools 查看可用工具。"
    lines = [f"[mock-llm] 工具 `{name}`"]
    if d.get("description"):
        lines.append(f"说明: {d['description']}")
    params = d.get("parameters") or {}
    lines.append("参数 schema:")
    lines.append(json.dumps(params, ensure_ascii=False, indent=2))
    props = params.get("properties") or {}
    required = params.get("required") or []
    lines.append("调用示例（仅含必填参数，复制发送即可下发）:")
    lines.append(_example_call(name, params))
    optional = [k for k in props if k not in required]
    if optional:
        lines.append(f"可选参数: {', '.join(optional)}")
    lines.append(f"缺必填参数会被拦截；要跳过校验强制下发，改用: /call {name} {{...}}")
    return "\n".join(lines)


def local_exec(name: str, args: dict) -> str:
    """在 mock 所在机器上执行参数中的命令字段。仅当 MOCK_EXEC_ENABLED=1 时可达。"""
    command = next(
        (args[k] for k in COMMAND_PARAM_KEYS if isinstance(args.get(k), str)), None
    )
    if command is None:
        return (
            f"[mock-exec] 工具 `{name}` 的参数里没有命令字段（支持键: {', '.join(COMMAND_PARAM_KEYS)}），"
            f"请去掉 /exec 前缀改为 `<工具名> <JSON>` 下发给客户端执行。"
        )
    run_list = ["powershell", "-NoProfile", "-Command", command] if os.name == "nt" else ["sh", "-c", command]
    try:
        proc = subprocess.run(run_list, capture_output=True, text=True, timeout=EXEC_TIMEOUT)
    except subprocess.TimeoutExpired:
        return f"[mock-exec] 工具 `{name}` 本地执行超时（{EXEC_TIMEOUT}s）。"
    output = (
        f"--- stdout ---\n{proc.stdout.strip() or '(空)'}\n"
        f"--- stderr ---\n{proc.stderr.strip() or '(空)'}"
    )
    if len(output) > EXEC_OUTPUT_LIMIT:
        output = output[:EXEC_OUTPUT_LIMIT] + "\n...(截断)"
    return f"[mock-exec] `{name}` exit={proc.returncode}\n{output}"


def _make_call_response(
    name: str, args: dict, tool_defs: dict[str, dict], messages: list[dict], *, force: bool
) -> dict:
    """构造 tool_calls 下发响应；非强制模式下缺必填参数时拦截并给出完整示例。"""
    if _executed_since_last_user(messages):
        return {
            "kind": "text",
            "label": "executed",
            "content": (
                f"[mock-llm] 工具 `{name}` 已执行，客户端回传结果：\n"
                f"{extract_last_tool_result(messages) or '(空)'}\n"
                f"继续发送 `工具名 {{JSON}}` 执行下一步，或 /tools 查看清单。"
            ),
        }
    if not force:
        schema = (tool_defs.get(name) or {}).get("parameters") or {}
        missing = [k for k in schema.get("required") or [] if k not in args]
        if missing:
            return {
                "kind": "text",
                "label": "call_blocked",
                "content": (
                    f"[mock-llm] 调用 `{name}` 缺少必填参数: {missing}（客户端会拒绝）\n"
                    f"完整调用示例（复制发送即可）：\n{_example_call(name, schema)}\n"
                    f"要跳过校验强制下发，改用 /call 前缀。"
                ),
            }
    return {"kind": "tool_calls", "label": "call_forced" if force else "call", "calls": [ToolCall(name=name, arguments=args)]}


def _extract_json_obj(raw: str) -> tuple[dict | None, str | None]:
    """从文本中提取第一个完整 JSON 对象（尾部附加内容自动忽略）。"""
    idx = raw.find("{")
    if idx < 0:
        return None, "未找到 JSON 对象"
    try:
        obj, _end = json.JSONDecoder().raw_decode(raw[idx:])
    except json.JSONDecodeError as e:
        return None, f"JSON 解析失败: {e}（消息可能被截断）"
    if not isinstance(obj, dict):
        return None, "参数必须是 JSON 对象"
    return obj, None


def route_interactive(messages: list[dict], tool_defs: dict[str, dict]) -> dict | None:
    """命中交互格式返回响应描述；未命中返回 None。

    返回结构：
    - {"kind": "text", "label": str, "content": str}
    - {"kind": "tool_calls", "label": str, "calls": [ToolCall, ...]}

    路由只看最后一条 user 消息的**首行**，兼容客户端在消息尾部追加注入内容。
    """
    text = _last_user_text(messages)
    if not text:
        return None
    first_line = text.splitlines()[0].strip()

    if first_line == "/help":
        return {"kind": "text", "label": "help", "content": HELP_TEXT}
    if first_line in ("/tools", "/list"):
        return {"kind": "text", "label": "tools_list", "content": format_tool_list(tool_defs)}
    m = re.match(r"^/tool\s+(\S+)$", first_line)
    if m:
        return {
            "kind": "text",
            "label": "tool_detail",
            "content": format_tool_detail(m.group(1), tool_defs),
        }

    # /exec 服务端本地执行命令字段 —— 默认禁用（公网部署防 RCE），设 MOCK_EXEC_ENABLED=1 才开启
    if first_line.startswith("/exec "):
        if os.environ.get("MOCK_EXEC_ENABLED") != "1":
            return {
                "kind": "text",
                "label": "exec_disabled",
                "content": "[mock-llm] /exec 已禁用：服务端本地执行属高危能力，公网部署默认关闭。如需开启，在服务端设置环境变量 MOCK_EXEC_ENABLED=1 并重启；当前可改用 `<工具名> {JSON}` 交由客户端执行。",
            }
        parts = first_line.split(maxsplit=2)
        if len(parts) < 3:
            return {
                "kind": "text",
                "label": "exec_error",
                "content": '[mock-llm] 用法: /exec <工具名> <JSON参数>，例如 /exec pwsh {"command": "Get-Date"}',
            }
        name, raw = parts[1], parts[2]
        args, err = _extract_json_obj(raw)
        if err or args is None:
            return {"kind": "text", "label": "exec_error", "content": f"[mock-llm] 工具 `{name}` {err}"}
        return {"kind": "text", "label": "exec", "content": local_exec(name, args)}

    # /call 前缀：跳过必填校验强制下发
    if first_line.startswith("/call "):
        parts = first_line.split(maxsplit=2)
        if len(parts) < 3:
            return {
                "kind": "text",
                "label": "call_forced_error",
                "content": '[mock-llm] 用法: /call <工具名> <JSON参数>，例如 /call pwsh {"command": "Get-Date"}',
            }
        name, raw = parts[1], parts[2]
        args, err = _extract_json_obj(raw)
        if err or args is None:
            return {"kind": "text", "label": "call_error", "content": f"[mock-llm] 工具 `{name}` {err}"}
        return _make_call_response(name, args, tool_defs, messages, force=True)

    # 首行恰好是一个工具名 -> 查看参数
    if first_line in tool_defs:
        return {"kind": "text", "label": "tool_detail", "content": format_tool_detail(first_line, tool_defs)}

    # 工具名 + JSON -> 校验必填后下发 tool_calls；若已执行回传则返回结果确认
    m = _LINE_CALL_RE.match(first_line)
    if m and m.group(1) in tool_defs:
        args, err = _extract_json_obj(m.group(2))
        if err or args is None:
            return {"kind": "text", "label": "call_error", "content": f"[mock-llm] 工具 `{m.group(1)}` {err}"}
        return _make_call_response(m.group(1), args, tool_defs, messages, force=False)

    return None
