"""JSON 前缀可行性判定 —— "约束解码"（constrained decoding）的最小原理演示。

## 为什么需要这个

让模型输出合法 JSON 有两条路：

1. **事后修复**（本仓库 `extract.py` 干的活）：模型想说什么就说什么，我再把 JSON 抠出来。
2. **事前约束**（vLLM 的 guided decoding、Outlines、llama.cpp 的 GBNF、OpenAI 的
   Structured Outputs）：在**每一个 token 采样的瞬间**，把"会让 JSON 变非法"的 token
   概率直接置零。

第二条路的关键是要能回答一个问题：

    "当前已经生成的这段文本，还有可能被补成一个合法的 JSON 吗？"

如果能，就保留所有候选 token；如果不能，说明上一个 token 选错了，整条分支要砍掉。

## 三种状态

| 状态 | 含义 | 处理 |
|---|---|---|
| `complete` | 已经是一个完整的 JSON 值 | 可以结束 |
| `viable` | 还没结束，但存在合法补全 | 继续生成 |
| `invalid` | 已经不可能补成合法 JSON | 砍掉这条分支 |

## 实现方式

不是写一个显式状态机（那个很容易写错），而是让递归下降解析器在
"输入用完但语法还没结束"时抛 `_Incomplete`，在"语法真的错了"时抛 `_Invalid`。
两者的区分就是这个模块的全部价值。

**局限**：只做**语法**层面。`{"age": "abc"}` 语法完全合法，
但不符合 `{"age": {"type": "integer"}}` —— 那就是 `schema.py` 的事了。
真实系统里这两层是叠加的（语法约束 + 类型约束），后者需要 FSM 化的 schema 编译。
"""

from __future__ import annotations

import json
from typing import Iterable, Sequence

__all__ = [
    "ParseStatus",
    "parse_status",
    "is_complete",
    "is_viable_prefix",
    "allowed_tokens",
    "constrained_next",
]


class ParseStatus:
    COMPLETE = "complete"
    VIABLE = "viable"
    INVALID = "invalid"


class _Incomplete(Exception):
    """输入用完，但语法上还可以继续 —— 这是"可行前缀"。"""


class _Invalid(Exception):
    """语法已经错了，补什么都不可能合法。"""


_WS = " \t\n\r"
_NUMBER_CHARS = set("0123456789+-.eE")


class _Parser:
    def __init__(self, text: str) -> None:
        self.text = text
        self.pos = 0

    # -- 基础 ------------------------------------------------------------ #

    def _peek(self) -> str:
        return self.text[self.pos] if self.pos < len(self.text) else ""

    def _skip_ws(self) -> None:
        while self.pos < len(self.text) and self.text[self.pos] in _WS:
            self.pos += 1

    def _expect_literal(self, word: str) -> None:
        for expected in word:
            if self.pos >= len(self.text):
                raise _Incomplete()
            actual = self.text[self.pos]
            if actual != expected:
                raise _Invalid(f"期望 {expected!r}，实际 {actual!r}")
            self.pos += 1

    # -- 各类型 ---------------------------------------------------------- #

    def parse_value(self) -> None:
        self._skip_ws()
        ch = self._peek()
        if ch == "":
            raise _Incomplete()
        if ch == "{":
            self.parse_object()
        elif ch == "[":
            self.parse_array()
        elif ch == '"':
            self.parse_string()
        elif ch == "t":
            self._expect_literal("true")
        elif ch == "f":
            self._expect_literal("false")
        elif ch == "n":
            self._expect_literal("null")
        elif ch == "-" or ch.isdigit():
            self.parse_number()
        else:
            raise _Invalid(f"值不能以 {ch!r} 开头")

    def parse_string(self) -> None:
        if self._peek() != '"':
            raise _Invalid("字符串必须以双引号开头")
        self.pos += 1
        while True:
            if self.pos >= len(self.text):
                raise _Incomplete("字符串未闭合")
            ch = self.text[self.pos]
            if ch == "\\":
                if self.pos + 1 >= len(self.text):
                    raise _Incomplete("转义符后缺少字符")
                self.pos += 2
                continue
            if ch == '"':
                self.pos += 1
                return
            self.pos += 1

    def parse_number(self) -> None:
        start = self.pos
        if self._peek() == "-":
            self.pos += 1
        digits = 0
        while self.pos < len(self.text) and self.text[self.pos].isdigit():
            self.pos += 1
            digits += 1
        if digits == 0:
            # 只有 '-' 或 '-' 后面跟着非数字：前者还能补，后者直接非法
            if self.pos >= len(self.text):
                raise _Incomplete("数字还没有整数部分")
            raise _Invalid(f"数字必须以数字开头，实际 {self.text[self.pos]!r}")
        if self._peek() == ".":
            self.pos += 1
            if self.pos >= len(self.text):
                raise _Incomplete("小数点后还没有数字")
            if not self.text[self.pos].isdigit():
                raise _Invalid("小数点后必须是数字")
            while self.pos < len(self.text) and self.text[self.pos].isdigit():
                self.pos += 1
        if self._peek() in ("e", "E"):
            self.pos += 1
            if self._peek() in ("+", "-"):
                self.pos += 1
            if self.pos >= len(self.text):
                raise _Incomplete("指数部分还没有数字")
            if not self.text[self.pos].isdigit():
                raise _Invalid("指数部分必须是数字")
            while self.pos < len(self.text) and self.text[self.pos].isdigit():
                self.pos += 1
        if self.pos == start:
            raise _Invalid("数字格式非法")

    def parse_array(self) -> None:
        self.pos += 1  # '['
        self._skip_ws()
        if self._peek() == "":
            raise _Incomplete("数组未闭合")
        if self._peek() == "]":
            self.pos += 1
            return
        while True:
            self.parse_value()
            self._skip_ws()
            ch = self._peek()
            if ch == "":
                raise _Incomplete("数组未闭合")
            if ch == ",":
                self.pos += 1
                continue
            if ch == "]":
                self.pos += 1
                return
            raise _Invalid(f"数组里期望 ',' 或 ']'，实际 {ch!r}")

    def parse_object(self) -> None:
        self.pos += 1  # '{'
        self._skip_ws()
        if self._peek() == "":
            raise _Incomplete("对象未闭合")
        if self._peek() == "}":
            self.pos += 1
            return
        while True:
            self._skip_ws()
            if self._peek() == "":
                raise _Incomplete("对象未闭合")
            self.parse_string()
            self._skip_ws()
            if self._peek() == "":
                raise _Incomplete("键后面缺少 ':'")
            if self._peek() != ":":
                raise _Invalid(f"键后面期望 ':'，实际 {self._peek()!r}")
            self.pos += 1
            self.parse_value()
            self._skip_ws()
            ch = self._peek()
            if ch == "":
                raise _Incomplete("对象未闭合")
            if ch == ",":
                self.pos += 1
                continue
            if ch == "}":
                self.pos += 1
                return
            raise _Invalid(f"对象里期望 ',' 或 '}}'，实际 {ch!r}")


def parse_status(text: str) -> str:
    """返回 ``complete`` / ``viable`` / ``invalid``。"""
    if text is None:
        return ParseStatus.INVALID
    parser = _Parser(text)
    try:
        parser.parse_value()
    except _Incomplete:
        return ParseStatus.VIABLE
    except _Invalid:
        return ParseStatus.INVALID
    except RecursionError:
        return ParseStatus.INVALID
    parser._skip_ws()
    return ParseStatus.COMPLETE if parser.pos >= len(parser.text) else ParseStatus.INVALID


def is_complete(text: str) -> bool:
    """是不是一个完整的 JSON 值（且没有多余尾巴）。"""
    return parse_status(text) == ParseStatus.COMPLETE


def is_viable_prefix(text: str) -> bool:
    """这段文本还有可能被补成一个合法 JSON 吗？"""
    return parse_status(text) in (ParseStatus.COMPLETE, ParseStatus.VIABLE)


def allowed_tokens(prefix: str, vocabulary: Sequence[str]) -> list[str]:
    """约束解码的核心操作：对候选 token 做掩码。

    真实实现里，这一步是在 logits 上做 mask（O(1) 的状态机转移）；
    这里用"暴力试一遍"来演示同一个语义 —— 因为可读性优先。
    """
    return [token for token in vocabulary if is_viable_prefix(prefix + token)]


def constrained_next(prefix: str, vocabulary: Sequence[str]) -> list[str]:
    """``allowed_tokens`` 的别名，语义更贴近"下一个 token 该选什么"。"""
    return allowed_tokens(prefix, vocabulary)


DEFAULT_VOCABULARY: tuple[str, ...] = (
    "{", "}", "[", "]", ",", ":", '"',
    "true", "false", "null",
    "0", "1", "2",
    "name", "age", "city",
)

__all__.append("DEFAULT_VOCABULARY")


def pretty_complete(text: str, vocabulary: Iterable[str] = DEFAULT_VOCABULARY) -> str:
    """给示例用：把"下一步允许的 token"渲染成一行。"""
    allowed = allowed_tokens(text, tuple(vocabulary))
    return " ".join(allowed) if allowed else "（无可用 token —— 这条分支必须砍掉）"


def _self_check() -> None:  # pragma: no cover - 手工自检
    cases = [("{", ParseStatus.VIABLE), ("{}", ParseStatus.COMPLETE),
             ("{]", ParseStatus.INVALID), ('{"a": 1', ParseStatus.VIABLE),
             ("[1, 2", ParseStatus.VIABLE), ("tru", ParseStatus.VIABLE),
             ("nope", ParseStatus.INVALID), (json.dumps({"a": [1, 2]}), ParseStatus.COMPLETE)]
    for text, expected in cases:
        actual = parse_status(text)
        assert actual == expected, f"{text!r}: 期望 {expected}，实际 {actual}"
    print("grammar self-check OK")


if __name__ == "__main__":  # pragma: no cover
    _self_check()
