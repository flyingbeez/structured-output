#!/usr/bin/env python3
"""示例 3：六种策略的对照评测 —— 本仓库的核心产出。

运行：python examples/03_strategy_benchmark.py
     python examples/03_strategy_benchmark.py --live      # 对真实模型跑同一套流程

离线模式下模型是**确定性模拟器**：20 条样本的坏法是手工标注的，跑多少次结果都一样。
所以这里的数字可复现，但它证明的是"我的解析/校验/修复逻辑是否正确处理了这些坏法"，
**不能证明真实模型的手滑分布就是这样**。要后者请用 --live（需要 API Key）。

输出三张表：
  1. 策略对照表：成功率 / 平均尝试次数 / token 成本
  2. 按坏法分解：哪一种坏法被哪一种策略救回来了
  3. 逐样本明细：失败样本具体错在哪
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from schemaout import (  # noqa: E402
    SAMPLES,
    SimulatedLLM,
    benchmark,
    defect_histogram_of,
    render_defect_breakdown,
    render_detail,
    render_table,
)
from schemaout.bench import render_histogram  # noqa: E402

LINE = "=" * 92


def main() -> None:
    live = "--live" in sys.argv

    if live:
        from schemaout import OpenAICompatLLM
        print("【真实模型模式】" + "=" * 60)
        llm = OpenAICompatLLM.from_env()
    else:
        llm = SimulatedLLM(SAMPLES)

    print(f"\n{LINE}")
    print(f"策略对照评测 —— 20 条抽取样本 × 6 种策略")
    print(f"坏法分布：{render_histogram()}")
    print(f"模型：{'真实模型 ' + llm.model if live else '确定性模拟器（离线，结果可复现）'}")
    print(LINE)

    reports = benchmark(llm, SAMPLES)

    print("\n## 一、策略对照表\n")
    print(render_table(reports))

    print("\n## 二、按坏法分解（✅ 全对 / 🔧 部分对 / ❌ 全错）\n")
    print(render_defect_breakdown(reports, SAMPLES))

    print("\n## 三、逐样本明细 —— 修复循环\n")
    repair = next(r for r in reports if r.strategy.key == "repair_loop")
    print(render_detail(repair, SAMPLES))

    print(f"\n{LINE}")
    print("## 四、读表要点\n")
    print("  1. `naive` → `extract` 的跃升全部来自客户端解析层：不花一分钱请求成本，")
    print("     只是把'模型说的 JSON'从文本里抠出来。这是性价比最高的一步。")
    print("  2. `json_mode` 与 `extract` 打平 —— 两者解决的是**同一类问题（语法）**，")
    print("     所以是替代关系而不是叠加关系。已经有好解析层时，JSON 模式边际收益接近 0。")
    print("  3. `tool_call` 略胜，它多解决的是**语义**里的'多余字段' —— ")
    print("     因为 schema 直接约束了模型能产生哪些字段。")
    print("  4. `repair_loop` 最高，它解决的是**语义错误**（类型错/枚举错），")
    print("     代价是 token ×1.3 左右。语法能靠约束，语义只能靠反馈。")
    print("  5. 谁都修不了'被截断' —— 那是 max_tokens 的问题，属于参数配置，不是格式问题。")

    hist = defect_histogram_of(SAMPLES)
    print(f"\n  样本坏法明细：{hist}")
    print(LINE)


if __name__ == "__main__":
    main()
