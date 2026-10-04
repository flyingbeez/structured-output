#!/usr/bin/env python3
"""示例 4：约束解码原理 —— 每一步都知道"下一个 token 能选什么"。

运行：python examples/04_constrained_decoding.py

vLLM 的 guided decoding、Outlines、llama.cpp 的 GBNF、OpenAI 的 Structured Outputs，
底层都是同一件事：**在采样前把会破坏 JSON 语法的 token 概率置零**。
这个示例用一个小词表暴力演示这个 mask 是怎么算出来的。
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from schemaout import allowed_tokens, is_complete, is_viable_prefix, parse_status  # noqa: E402

LINE = "=" * 84
VOCAB = ["{", "}", "[", "]", ",", ":", '"', "true", "false", "null",
         "0", "1", "2", "name", "age", "city", " "]
TARGET_STEPS = ['{"', "name", '":', " ", "1", "}"]


def main() -> None:
    print(f"\n{LINE}\n约束解码：一步一步看 mask 怎么收紧\n{LINE}")
    print(f"词表（{len(VOCAB)} 个 token）：{' '.join(repr(t) for t in VOCAB)}\n")

    prefix = ""
    for step, token in enumerate(TARGET_STEPS, 1):
        options = allowed_tokens(prefix, VOCAB)
        blocked = [t for t in VOCAB if t not in options]
        print(f"第 {step} 步  当前前缀 = {prefix!r}")
        print(f"         允许：{' '.join(repr(t) for t in options)}")
        print(f"         屏蔽：{' '.join(repr(t) for t in blocked) or '（无）'}")
        print(f"         状态：{parse_status(prefix)}")
        prefix += token
        print()

    print(f"最终：{prefix!r}  完整={is_complete(prefix)}  可行前缀={is_viable_prefix(prefix)}")

    print(f"\n{LINE}\n几种典型前缀的可行性\n{LINE}")
    for text in ["", "{", '{"name"', '{"name":', '{"name": 1', '{"name": 1}',
                 '{"name": 1}}', "tru", "nope", "{]", "-", "12."]:
        print(f"  {text!r:<18} → {parse_status(text)}")

    print(f"\n{LINE}\n为什么这件事重要\n{LINE}")
    print("  1. **事前约束 vs 事后修复**：约束解码从源头保证语法合法，")
    print("     我的 extract.py 是事后修复。生产环境两者叠加最优。")
    print("  2. 约束解码只能保证**语法**。{\"age\": \"未知\"} 语法完全合法，")
    print("     但不符合 {\"age\": {\"type\": \"integer\"}} —— 那需要把 schema 也编译成状态机")
    print("     （Outlines 的做法），或者像我的 strategies.py 那样做校验 + 反馈修复。")
    print("  3. 代价：约束越严，模型的'表达空间'越小，有时会牺牲推理质量。")
    print("     所以业界现在倾向于'语法约束 + 语义校验'两层，而不是把 schema 全压进采样。")


if __name__ == "__main__":
    main()
