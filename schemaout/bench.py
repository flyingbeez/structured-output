"""基准测试：把六种策略跑在同一批样本上，输出可贴进 README 的对照表。

除了总体成功率，还做**按坏法分类的分解** —— 这才是有信息量的部分：
你能看到"哪一种坏法被哪一种策略救回来了"，从而知道该给自己加什么层。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

from .llm import BaseLLM
from .samples import (
    SEMANTIC_DEFECTS,
    STRUCTURAL_DEFECTS,
    UNFIXABLE_DEFECTS,
    Sample,
    defect_histogram,
)
from .strategies import STRATEGIES, RunResult, Strategy, run

__all__ = ["StrategyReport", "benchmark", "render_table", "render_defect_breakdown", "render_detail"]


@dataclass
class StrategyReport:
    strategy: Strategy
    results: list[RunResult] = field(default_factory=list)

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def passed(self) -> int:
        return sum(1 for r in self.results if r.ok)

    @property
    def success_rate(self) -> float:
        return self.passed / self.total if self.total else 0.0

    @property
    def mean_attempts(self) -> float:
        return sum(r.attempts for r in self.results) / self.total if self.total else 0.0

    @property
    def total_tokens(self) -> int:
        return sum(r.tokens for r in self.results)

    @property
    def tokens_per_sample(self) -> float:
        return self.total_tokens / self.total if self.total else 0.0

    def by_defect(self, samples: Sequence[Sample]) -> dict[str, tuple[int, int]]:
        """返回 {坏法: (通过数, 总数)}。"""
        lookup = {s.id: s for s in samples}
        buckets: dict[str, list[int]] = {}
        for result in self.results:
            sample = lookup.get(result.sample_id)
            if sample is None:
                continue
            bucket = buckets.setdefault(sample.defect, [0, 0])
            bucket[1] += 1
            if result.ok:
                bucket[0] += 1
        return {k: (v[0], v[1]) for k, v in buckets.items()}


def benchmark(llm: BaseLLM, samples: Sequence[Sample],
              strategies: Sequence[Strategy] = STRATEGIES) -> list[StrategyReport]:
    reports: list[StrategyReport] = []
    for strategy in strategies:
        report = StrategyReport(strategy=strategy)
        for sample in samples:
            report.results.append(run(strategy, llm, sample))
        reports.append(report)
    return reports


# --------------------------------------------------------------------------- #
# 渲染
# --------------------------------------------------------------------------- #

def render_table(reports: Sequence[StrategyReport]) -> str:
    lines = [
        "| 策略 | 成功率 | 通过/总数 | 平均尝试次数 | 每样本 token | 相对基线 token |",
        "|---|---|---|---|---|---|",
    ]
    baseline = reports[0].tokens_per_sample if reports else 0.0
    for report in reports:
        ratio = (report.tokens_per_sample / baseline) if baseline else 0.0
        lines.append(
            f"| `{report.strategy.key}` | **{report.success_rate * 100:.1f}%** | "
            f"{report.passed}/{report.total} | {report.mean_attempts:.2f} | "
            f"{report.tokens_per_sample:.0f} | {ratio:.2f}× |"
        )
    return "\n".join(lines)


_DEFECT_ORDER = [
    "clean", "fenced", "prose", "single_quotes", "trailing_comma",
    "bare_keys", "python_literals", "wrong_type", "bad_enum", "extra_field", "truncated",
]

_DEFECT_LABEL = {
    "clean": "干净输出",
    "fenced": "```json 围栏",
    "prose": "前后客套话",
    "single_quotes": "单引号",
    "trailing_comma": "尾随逗号",
    "bare_keys": "裸键名",
    "python_literals": "Python 字面量",
    "wrong_type": "类型错误",
    "bad_enum": "枚举值错误",
    "extra_field": "多余字段",
    "truncated": "被截断",
}


def _defect_kind(defect: str) -> str:
    if defect in STRUCTURAL_DEFECTS:
        return "结构性"
    if defect in SEMANTIC_DEFECTS:
        return "语义性"
    if defect in UNFIXABLE_DEFECTS:
        return "修不了"
    return "—"


def render_defect_breakdown(reports: Sequence[StrategyReport], samples: Sequence[Sample]) -> str:
    keys = [r.strategy.key for r in reports]
    header = "| 坏法 | 类别 | 样本数 | " + " | ".join(f"`{k}`" for k in keys) + " |"
    divider = "|---|---|---|" + "---|" * len(keys)
    lines = [header, divider]

    for defect in _DEFECT_ORDER:
        total = sum(1 for s in samples if s.defect == defect)
        if total == 0:
            continue
        cells = []
        for report in reports:
            passed, count = report.by_defect(samples).get(defect, (0, 0))
            mark = "✅" if passed == count else ("🔧" if passed else "❌")
            cells.append(f"{mark} {passed}/{count}")
        lines.append(
            f"| {_DEFECT_LABEL.get(defect, defect)} | {_defect_kind(defect)} | {total} | "
            + " | ".join(cells) + " |"
        )
    return "\n".join(lines)


def render_detail(report: StrategyReport, samples: Sequence[Sample]) -> str:
    """单个策略的逐样本明细（排查用）。"""
    lookup = {s.id: s for s in samples}
    lines = [f"# 策略 {report.strategy.key} —— {report.strategy.label}", ""]
    for result in report.results:
        sample = lookup.get(result.sample_id)
        defect = sample.defect if sample else "?"
        status = "OK " if result.ok else "FAIL"
        detail = result.parse_error or ("; ".join(result.schema_errors[:2]) or "")
        lines.append(
            f"{status} {result.sample_id:<4} 坏法={defect:<16} 尝试={result.attempts} "
            f"token={result.tokens:<5} {detail[:90]}"
        )
    lines.append("")
    lines.append(f"成功率 {report.passed}/{report.total} = {report.success_rate * 100:.1f}%")
    return "\n".join(lines)


def render_histogram() -> str:
    hist = defect_histogram()
    return " ".join(f"{_DEFECT_LABEL.get(k, k)}={v}" for k, v in sorted(hist.items()))


def summary_dict(reports: Sequence[StrategyReport]) -> list[dict[str, Any]]:
    return [
        {
            "strategy": r.strategy.key,
            "label": r.strategy.label,
            "success_rate": round(r.success_rate, 4),
            "passed": r.passed,
            "total": r.total,
            "mean_attempts": round(r.mean_attempts, 3),
            "tokens_per_sample": round(r.tokens_per_sample, 1),
        }
        for r in reports
    ]
