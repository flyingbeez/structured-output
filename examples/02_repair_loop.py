#!/usr/bin/env python3
"""示例 2：修复循环 —— 把校验错误精确回灌给模型。

运行：python examples/02_repair_loop.py

这个示例把"修复循环里每一轮到底发了什么"打印出来。
面试时如果被问"你怎么让模型改对"，能具体说出回灌的文案，比说"我加了重试"强得多。
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from schemaout import SAMPLES, SimulatedLLM, extract_json, validate  # noqa: E402
from schemaout.strategies import build_repair_prompt, build_user_prompt  # noqa: E402

LINE = "=" * 84


def demo(sample_id: str) -> None:
    sample = next(s for s in SAMPLES if s.id == sample_id)
    llm = SimulatedLLM(SAMPLES)

    print(f"\n{LINE}")
    print(f"样本 {sample.id}｜坏法 = {sample.defect}｜任务：{sample.instruction}")
    print(LINE)

    user = build_user_prompt(sample)
    previous = ""

    for round_no in range(1, 4):
        completion = llm.complete(system="你是抽取助手", user=user, schema=sample.schema,
                                  mode="text", meta={"sample_id": sample.id, "round": round_no})
        print(f"\n--- 第 {round_no} 轮 ---")
        print(f"模型输出：{completion.text[:150]}")

        extraction = extract_json(completion.text)
        if not extraction.ok:
            error_text = extraction.error
            print(f"解析结果：❌ {error_text}")
        else:
            check = validate(extraction.value, sample.schema)
            if check.ok:
                print(f"解析结果：✅ 通过校验，共用了 {round_no} 轮")
                return
            error_text = check.error_text()
            print(f"解析结果：❌ 校验失败\n{error_text}")

        previous = completion.text
        user = build_repair_prompt(sample, previous, error_text)
        print("\n[回灌给模型的修复提示]")
        for line in user.splitlines():
            if "校验" in line or line.startswith("-"):
                print(f"   {line}")

    print("\n结论：3 轮仍未通过。")


def main() -> None:
    print(f"\n{LINE}\n修复循环：类型错误（可以修）\n{LINE}")
    demo("o4")  # bad_enum → 反馈后能修好
    print(f"\n\n{LINE}\n修复循环：被截断（修不了）\n{LINE}")
    demo("o5")  # truncated → 反馈也没用，因为原因是 max_tokens

    print(f"\n{LINE}\n两个必须讲清楚的取舍\n{LINE}")
    print("  1. 修复提示里必须带**具体的错误路径和期望值**（$.currency: 取值必须是 [...] 之一）")
    print("     —— 只说'格式错误'，模型只能瞎猜。")
    print("  2. 修复是有成本的：token 约 ×1.3，延迟翻倍。所以只对'校验失败'的样本重试，")
    print("     而不是所有样本都跑两遍。")
    print("  3. 截断这类问题重试多少次都没用 —— 该改的是 max_tokens，不是提示词。")


if __name__ == "__main__":
    main()
