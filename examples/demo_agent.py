"""最小 agent 演示：把 mock LLM 下发的命令真的执行掉，验证完整调试回路。

依赖场景编排模式（MOCK_AUTO=1），启动服务器：
    $env:MOCK_AUTO='1'; python run.py      # PowerShell
    MOCK_AUTO=1 python run.py              # bash

运行：
    python examples/demo_agent.py
"""

from __future__ import annotations

import json
import subprocess
import sys

import httpx

BASE = "http://127.0.0.1:8789/v1"

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "run_command",
            "description": "在受控环境执行 shell 命令并返回 stdout/stderr/退出码",
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"],
            },
        },
    }
]


def run_tool(name: str, arguments: dict) -> str:
    """工具执行器：这里替换成你的真实工具逻辑即可。"""
    if name == "run_command":
        proc = subprocess.run(
            arguments["command"], shell=True, capture_output=True, text=True, timeout=15
        )
        return json.dumps(
            {"stdout": proc.stdout.strip(), "stderr": proc.stderr.strip(), "code": proc.returncode},
            ensure_ascii=False,
        )
    return json.dumps({"error": f"unknown tool: {name}"}, ensure_ascii=False)


def main() -> int:
    messages: list[dict] = [{"role": "user", "content": "请执行预设的命令序列并汇总结果"}]

    for turn in range(1, 7):
        resp = httpx.post(
            f"{BASE}/chat/completions",
            json={"model": "mock-agent", "messages": messages, "tools": TOOLS},
            timeout=30,
        )
        resp.raise_for_status()
        choice = resp.json()["choices"][0]
        message = choice["message"]
        calls = message.get("tool_calls") or []
        print(f"--- 第 {turn} 轮 | finish_reason={choice['finish_reason']}")
        if not calls:
            print("最终回复:", message["content"])
            return 0

        messages.append(message)
        for call in calls:
            name = call["function"]["name"]
            args = json.loads(call["function"]["arguments"])
            result = run_tool(name, args)
            print(f"    执行 {name}({args}) -> {result[:120]}")
            messages.append(
                {"role": "tool", "tool_call_id": call["id"], "content": result}
            )

    print("达到最大轮数仍未结束，检查场景步骤是否覆盖全部轮次")
    return 1


if __name__ == "__main__":
    sys.exit(main())
