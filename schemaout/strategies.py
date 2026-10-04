"""六种策略的对照实验。

这个模块的全部价值在于**把"用哪种做法"变成一个可以测量的选择**，
而不是靠"大家都说 json mode 更稳"。

六种策略其实是三个正交开关的组合：

    mode    : text / json / tool        —— 请求侧要不要加格式约束
    parser  : naive / extract           —— 客户端要不要做解析修复
    rounds  : 1 / 3                     —— 要不要把校验错误回灌给模型

| 策略 key | mode | parser | rounds | 一句话 |
|---|---|---|---|---|
| `naive` | text | naive | 1 | 什么都不做，直接 json.loads |
| `json_mode` | json | naive | 1 | 只开 response_format |
| `extract` | text | extract | 1 | 只加一个强解析层 |
| `json_extract` | json | extract | 1 | 两者都加 |
| `tool_call` | tool | extract | 1 | 用强制 function call 约束格式 |
| `repair_loop` | text | extract | 3 | 解析 + 校验 + 把错误回灌重试 |
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Sequence

from .extract import extract_json
from .llm import BaseLLM, Completion
from .samples import Sample
from .schema import validate

__all__ = ["Strategy", "STRATEGIES", "RunResult", "run", "run_many", "get_strategy",
           "SYSTEM_PROMPT", "build_user_prompt", "build_repair_prompt"]


SYSTEM_PROMPT = (
    "你是一个信息抽取助手。请严格按照用户给出的 JSON Schema 输出结果，"
    "只输出 JSON 本身，不要添加任何解释、客套话或 Markdown 代码块。"
)

REPAIR_SYSTEM_PROMPT = (
    "你是一个 JSON 修复助手。用户会给你一段有问题的输出和具体的校验错误，"
    "请只输出修正后的 JSON，不要解释。"
)


def build_user_prompt(sample: Sample) -> str:
    schema_text = json.dumps(sample.schema, ensure_ascii=False, indent=2)
    return (
        f"任务：{sample.instruction}\n\n"
        f"请输出符合下面 JSON Schema 的 JSON：\n```json\n{schema_text}\n```\n\n"
        f"只输出 JSON。"
    )


def build_repair_prompt(sample: Sample, previous: str, error_text: str) -> str:
    schema_text = json.dumps(sample.schema, ensure_ascii=False, indent=2)
    return (
        f"任务：{sample.instruction}\n\n"
        f"你上一次的输出是：\n{previous}\n\n"
        f"它没有通过校验，具体错误如下：\n{error_text}\n\n"
        f"请输出修正后的 JSON，必须符合这个 Schema：\n```json\n{schema_text}\n```\n\n"
        f"只输出 JSON。"
    )


@dataclass(frozen=True)
class Strategy:
    key: str
    label: str
    mode: str          # text | json | tool
    parser: str        # naive | extract
    rounds: int = 1
    note: str = ""

    @property
    def uses_extractor(self) -> bool:
        return self.parser == "extract"

    @property
    def repairs(self) -> bool:
        return self.rounds > 1


STRATEGIES: tuple[Strategy, ...] = (
    Strategy("naive", "什么都不做（direct json.loads）", "text", "naive", 1,
             "基线：模型直接说，客户端直接解析"),
    Strategy("json_mode", "只开 JSON 模式", "json", "naive", 1,
             "response_format=json_object，服务端保证语法合法"),
    Strategy("extract", "只加强解析层", "text", "extract", 1,
             "不加任何请求侧约束，纯客户端修复"),
    Strategy("json_extract", "JSON 模式 + 强解析层", "json", "extract", 1,
             "两者都上，看是否有叠加收益"),
    Strategy("tool_call", "强制 function call", "tool", "extract", 1,
             "把 schema 塞进 tools 并用 tool_choice 强制调用"),
    Strategy("repair_loop", "解析 + 校验 + 反馈修复（最多 3 轮）", "text", "extract", 3,
             "把校验错误回灌给模型重试"),
)


def get_strategy(key: str) -> Strategy:
    for item in STRATEGIES:
        if item.key == key:
            return item
    raise KeyError(f"未知策略 {key!r}，可选：{[s.key for s in STRATEGIES]}")


@dataclass
class RunResult:
    sample_id: str
    strategy: str
    ok: bool
    attempts: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    method: str = ""
    parse_error: str = ""
    schema_errors: list[str] = field(default_factory=list)
    rounds_used: int = 0

    @property
    def tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def to_dict(self) -> dict:
        return {
            "sample_id": self.sample_id,
            "strategy": self.strategy,
            "ok": self.ok,
            "attempts": self.attempts,
            "tokens": self.tokens,
            "method": self.method,
            "parse_error": self.parse_error,
            "schema_errors": self.schema_errors,
        }


def _parse(strategy: Strategy, text: str) -> tuple[bool, Any, str, str]:
    """返回 (是否解析成功, 值, 方法, 解析错误)。"""
    if strategy.uses_extractor:
        result = extract_json(text)
        if result.ok:
            return True, result.value, result.method, ""
        return False, None, "failed", result.error

    # naive：直接 json.loads，不做任何修复
    try:
        return True, json.loads(text.strip()), "json.loads", ""
    except (json.JSONDecodeError, TypeError) as exc:
        return False, None, "failed", f"JSON 语法错误：{exc}"


def run(strategy: Strategy, llm: BaseLLM, sample: Sample) -> RunResult:
    """对一个样本跑一种策略。"""
    result = RunResult(sample_id=sample.id, strategy=strategy.key, ok=False)
    user = build_user_prompt(sample)
    system = SYSTEM_PROMPT
    last_error = ""
    previous_output = ""

    for round_no in range(1, strategy.rounds + 1):
        result.attempts += 1
        result.rounds_used = round_no

        completion: Completion = llm.complete(
            system=system, user=user, schema=sample.schema, mode=strategy.mode,
            meta={"sample_id": sample.id, "round": round_no},
        )
        result.prompt_tokens += completion.prompt_tokens
        result.completion_tokens += completion.completion_tokens
        previous_output = completion.text

        parsed_ok, value, method, parse_error = _parse(strategy, completion.text)
        result.method = method

        if not parsed_ok:
            last_error = parse_error
            result.parse_error = parse_error
            result.schema_errors = []
        else:
            check = validate(value, sample.schema)
            if check.ok:
                result.ok = True
                result.parse_error = ""
                result.schema_errors = []
                return result
            result.parse_error = ""
            result.schema_errors = [str(e) for e in check.errors]
            last_error = check.error_text()

        if round_no >= strategy.rounds:
            break

        # 准备下一轮：把错误精确回灌
        user = build_repair_prompt(sample, previous_output, last_error)
        system = REPAIR_SYSTEM_PROMPT

    return result


def run_many(strategy: Strategy, llm: BaseLLM, samples: Sequence[Sample]) -> list[RunResult]:
    return [run(strategy, llm, sample) for sample in samples]
