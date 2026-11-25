"""场景编排：定义 mock LLM 每一轮返回什么（tool_calls 或文本）。

核心思路：
- agent 的 messages 是累积的，统计 role=="tool" 的消息数量即可知道
  agent 已完成几轮工具调用，无需服务端维护会话状态。
- scenario.steps[已完成轮数] 即当前应返回的步骤，步骤耗尽后走 default。
- requires_tools 声明本场景需要 agent 暴露哪些工具；请求的 tools 清单
  覆盖这些工具时场景才启用，从而实现"先看清请求里有哪些工具再决定怎么调"。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class ToolCall:
    """一步里要下发的单个工具调用。arguments 为 dict，发送时序列化成 JSON 字符串。"""

    name: str
    arguments: dict = field(default_factory=dict)


@dataclass
class Step:
    """场景中的一个步骤。tool_calls 与 content 至少配置一个，同时配置时两者都返回。"""

    tool_calls: list[ToolCall] = field(default_factory=list)
    content: str | None = None

    def describe(self) -> str:
        if self.tool_calls and self.content:
            calls = ", ".join(f"{c.name}({json.dumps(c.arguments, ensure_ascii=False)})" for c in self.tool_calls)
            return f"tool_calls[{calls}] + content"
        if self.tool_calls:
            calls = ", ".join(c.name for c in self.tool_calls)
            return f"tool_calls[{calls}]"
        if self.content:
            preview = self.content if len(self.content) <= 40 else self.content[:37] + "..."
            return f"content: {preview}"
        return "(empty)"


@dataclass
class Scenario:
    """一个完整场景：预先写好的步骤序列 + 兜底行为。"""

    name: str
    steps: list[Step]
    default: Step = field(default_factory=Step)
    description: str = ""
    requires_tools: list[str] = field(default_factory=list)
    source: str = ""

    def matches(self, requested_tools: list[str]) -> bool:
        """请求携带的工具清单是否满足本场景的前置要求。"""
        if not self.requires_tools:
            return True
        available = set(requested_tools)
        return all(t in available for t in self.requires_tools)

    def pick_step(self, completed_tool_rounds: int) -> Step:
        """按已完成轮数选步骤；超出预写序列时走 default。"""
        if 0 <= completed_tool_rounds < len(self.steps):
            return self.steps[completed_tool_rounds]
        return self.default

    def describe(self) -> str:
        lines = [f"scenario[{self.name}] source={self.source}"]
        if self.requires_tools:
            lines.append(f"  requires_tools: {self.requires_tools}")
        for i, step in enumerate(self.steps):
            lines.append(f"  step[{i}] -> {step.describe()}")
        lines.append(f"  default -> {self.default.describe()}")
        return "\n".join(lines)


class ScenarioError(ValueError):
    """场景文件格式错误。"""


def _parse_tool_calls(raw: object) -> list[ToolCall]:
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise ScenarioError("tool_calls 必须是列表")
    calls = []
    for i, item in enumerate(raw):
        if not isinstance(item, dict) or "name" not in item:
            raise ScenarioError(f"tool_calls[{i}] 缺少 name 字段")
        arguments = item.get("arguments", {})
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments) if arguments.strip() else {}
            except json.JSONDecodeError as e:
                raise ScenarioError(f"tool_calls[{i}].arguments 不是合法 JSON: {e}") from e
        if not isinstance(arguments, dict):
            raise ScenarioError(f"tool_calls[{i}].arguments 必须是对象")
        calls.append(ToolCall(name=str(item["name"]), arguments=arguments))
    return calls


def _parse_step(raw: object, where: str) -> Step:
    if raw is None:
        return Step()
    if not isinstance(raw, dict):
        raise ScenarioError(f"{where} 必须是映射")
    step = Step(tool_calls=_parse_tool_calls(raw.get("tool_calls")))
    content = raw.get("content")
    if content is not None and not isinstance(content, str):
        raise ScenarioError(f"{where}.content 必须是字符串")
    step.content = content
    if not step.tool_calls and not step.content:
        raise ScenarioError(f"{where} 至少需要 tool_calls 或 content 之一")
    return step


def parse_scenario(data: object, source: str = "<memory>") -> Scenario:
    if not isinstance(data, dict):
        raise ScenarioError(f"{source}: 根节点必须是映射")
    if "name" not in data:
        raise ScenarioError(f"{source}: 缺少 name 字段")
    raw_steps = data.get("steps")
    if not isinstance(raw_steps, list) or not raw_steps:
        raise ScenarioError(f"{source}: steps 必须是非空列表")
    steps = [_parse_step(s, f"{source}.steps[{i}]") for i, s in enumerate(raw_steps)]
    requires = data.get("requires_tools", [])
    if not isinstance(requires, list) or not all(isinstance(t, str) for t in requires):
        raise ScenarioError(f"{source}: requires_tools 必须是字符串列表")
    return Scenario(
        name=str(data["name"]),
        description=str(data.get("description", "")),
        requires_tools=requires,
        steps=steps,
        default=_parse_step(data.get("default", {"content": "任务完成"}), f"{source}.default"),
        source=source,
    )


def load_scenario_file(path: Path) -> Scenario:
    try:
        text = path.read_text(encoding="utf-8")
        data = yaml.safe_load(text)
    except OSError as e:
        raise ScenarioError(f"读取场景文件失败 {path}: {e}") from e
    except yaml.YAMLError as e:
        raise ScenarioError(f"场景文件 YAML 解析失败 {path}: {e}") from e
    return parse_scenario(data, source=path.name)


def load_scenario_dir(dir_path: Path) -> list[Scenario]:
    """加载目录下全部 *.yaml / *.yml，按文件名排序。解析失败的单个文件报错中断。"""
    if not dir_path.is_dir():
        raise ScenarioError(f"场景目录不存在: {dir_path}")
    scenarios: list[Scenario] = []
    for path in sorted(dir_path.glob("*.y*ml")):
        scenarios.append(load_scenario_file(path))
    if not scenarios:
        raise ScenarioError(f"场景目录中没有 yaml 场景文件: {dir_path}")
    return scenarios


def match_scenario(scenarios: list[Scenario], requested_tools: list[str]) -> Scenario | None:
    """优先返回第一个 requires_tools 满足的场景；否则回落到第一个无条件场景。"""
    conditional = [s for s in scenarios if s.requires_tools and s.matches(requested_tools)]
    if conditional:
        return conditional[0]
    unconditional = [s for s in scenarios if not s.requires_tools]
    return unconditional[0] if unconditional else None


def extract_last_tool_result(messages: list[dict]) -> str:
    """取最后一条 role=="tool" 消息的文本内容，供步骤模板变量 {last_tool_result} 使用。"""
    for message in reversed(messages):
        if not isinstance(message, dict) or message.get("role") != "tool":
            continue
        content = message.get("content")
        if isinstance(content, str):
            return content[:500]
        if isinstance(content, list):  # 多模态数组格式，取 text 片段拼接
            parts = [
                p.get("text", "")
                for p in content
                if isinstance(p, dict) and p.get("type") == "text"
            ]
            return "\n".join(p for p in parts if p)[:500]
        return json.dumps(content, ensure_ascii=False)[:500] if content is not None else ""
    return ""


def count_tool_rounds(messages: list[dict]) -> int:
    """已完成工具轮数 = role=="tool" 消息条数。"""
    return sum(1 for m in messages if isinstance(m, dict) and m.get("role") == "tool")
