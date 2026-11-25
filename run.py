"""mock-llm 启动入口。

用法：
    python run.py                          # 默认 127.0.0.1:8789，场景目录 scenarios/
    python run.py --port 9000 --model my-mock
    python run.py --scenarios path/to/scenarios

可选环境变量 MOCK_LLM_KEY：设置后要求请求携带 Authorization: Bearer <key>。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import uvicorn

from mock_llm.server import create_app


def main() -> None:
    parser = argparse.ArgumentParser(description="OpenAI 兼容 mock LLM，按场景编排 tool_calls 调试 agent")
    parser.add_argument(
        "--scenarios",
        default=None,
        help="场景目录（含 *.yaml 场景文件），默认用项目自带的 <项目根>/scenarios",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8789)
    parser.add_argument("--model", default="mock-agent", help="对外暴露的模型名")
    args = parser.parse_args()

    if args.scenarios:
        scenario_dir = Path(args.scenarios).resolve()  # 显式传入的相对路径按当前目录解析
    else:
        scenario_dir = Path(__file__).resolve().parent / "scenarios"  # 默认跟随项目根目录，不受启动目录影响
    app = create_app(scenario_dir, args.model)
    print(f"[mock-llm] 模型: {args.model}  场景目录: {scenario_dir}")
    print(f"[mock-llm] http://{args.host}:{args.port}/v1/chat/completions")
    print(f"[mock-llm] 调试面板: http://{args.host}:{args.port}/__inspect/scenarios")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
