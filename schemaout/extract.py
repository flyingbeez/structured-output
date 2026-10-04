"""从"模型自由发挥的文本"里把 JSON 捞出来。

这是结构化输出里最脏、也最实用的一层。真实模型返回的 JSON 有十几种坏法：

    1. 外面包了 ```json ... ``` 代码围栏
    2. 前后有中文客套话（"当然可以！以下是结果：… 希望对你有帮助"）
    3. 用单引号代替双引号
    4. 尾随逗号 {"a": 1,}
    5. 键名不加引号 {a: 1}
    6. Python 字面量 True / False / None
    7. 中文/全角引号 “a”: “b”
    8. 字符串里有未转义的换行
    9. 被 max_tokens 截断，括号没闭合
   10. 同一段文本里有多个 JSON，给了错的那个
   11. 数字带千分位或单位（"1,234"、"3天"）
   12. 嵌套结构里混了注释 //

**这个模块不做越权的事**：它只负责"把文本变成 Python 对象"，
值对不对由 `schema.py` 判断。两件事分开，错误信息才能各司其职。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

__all__ = ["ExtractionResult", "extract_json", "extract_all_json", "repair_text"]


@dataclass
class ExtractionResult:
    ok: bool
    value: Any = None
    method: str = ""          # 用了哪条路径成功（便于统计各路径命中率）
    text: str = ""            # 实际被解析的文本
    notes: list[str] = field(default_factory=list)
    error: str = ""

    def to_dict(self) -> dict:
        return {"ok": self.ok, "method": self.method, "notes": self.notes, "error": self.error}


# --------------------------------------------------------------------------- #
# 文本级修复
# --------------------------------------------------------------------------- #

_FENCE_RE = re.compile(r"```[a-zA-Z0-9_+-]*\s*\n?(.*?)\n?```", re.DOTALL)
_FULLWIDTH = {"“": '"', "”": '"', "‘": "'", "’": "'", "：": ":", "，": ",", "【": "[", "】": "]"}


def strip_fences(text: str) -> tuple[str, bool]:
    """去掉代码围栏；返回 (新文本, 是否改动过)。"""
    match = _FENCE_RE.search(text)
    if match:
        return match.group(1).strip(), True
    # 只有开头有围栏、结尾被截断的情况
    if text.lstrip().startswith("```"):
        body = re.sub(r"^\s*```[a-zA-Z0-9_+-]*\s*", "", text)
        return body.strip(), True
    return text, False


def repair_text(text: str) -> tuple[str, list[str]]:
    """对文本做一系列保守的"语法修复"，返回 (修复后文本, 修改说明)。

    只做**不会改变语义**的修复：
      * 全角引号/冒号/逗号 → 半角
      * 单引号字符串 → 双引号字符串（仅当文本里没有双引号时）
      * 裸键名加引号
      * 去尾随逗号
      * Python 字面量 → JSON 字面量
      * 去 // 与 /* */ 注释
    """
    notes: list[str] = []
    out = text

    # 1) 全角标点
    replaced = "".join(_FULLWIDTH.get(ch, ch) for ch in out)
    if replaced != out:
        out = replaced
        notes.append("全角标点→半角")

    # 2) 去掉注释（字符串内的 // 不处理，这是已知局限）
    without_comments = re.sub(r"//[^\n]*", "", out)
    without_comments = re.sub(r"/\*.*?\*/", "", without_comments, flags=re.DOTALL)
    if without_comments != out:
        out = without_comments
        notes.append("去除注释")

    # 3) Python 字面量
    literal_fixed = re.sub(r"\bTrue\b", "true", out)
    literal_fixed = re.sub(r"\bFalse\b", "false", literal_fixed)
    literal_fixed = re.sub(r"\bNone\b", "null", literal_fixed)
    if literal_fixed != out:
        out = literal_fixed
        notes.append("Python 字面量→JSON")

    # 4) 单引号字符串 → 双引号（只在没有双引号时做，避免把内部引号搞坏）
    if "'" in out and '"' not in out:
        out = out.replace("'", '"')
        notes.append("单引号→双引号")

    # 5) 裸键名加引号：  {a: 1, b: 2}
    bare = re.sub(r'([{,]\s*)([A-Za-z_][A-Za-z0-9_\-]*)(\s*:)', r'\1"\2"\3', out)
    if bare != out:
        out = bare
        notes.append("裸键名加引号")

    # 6) 尾随逗号
    trailing = re.sub(r",(\s*[}\]])", r"\1", out)
    if trailing != out:
        out = trailing
        notes.append("去除尾随逗号")

    # 7) 字符串里的裸换行（把未转义换行转成 \n）—— 只处理 "..." 内部的换行
    def _escape_newlines(match: re.Match[str]) -> str:
        return '"' + match.group(1).replace("\n", "\\n").replace("\r", "") + '"'

    newline_fixed = re.sub(r'"([^"\\]*(?:\\.[^"\\]*)*)"', _escape_newlines, out, flags=re.DOTALL)
    if newline_fixed != out:
        out = newline_fixed
        notes.append("转义字符串内换行")

    return out, notes


def close_truncated(text: str) -> tuple[str, bool]:
    """尝试闭合被截断的 JSON：补齐未闭合的字符串和括号。

    这是"尽力而为"，不保证语义正确 —— 它的价值是让**部分**被截断的输出还能用，
    并且能让校验器给出更准确的错误（而不是一律报"JSON 语法错误"）。
    """
    stack: list[str] = []
    in_string = False
    escaped = False
    for ch in text:
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch in "{[":
            stack.append(ch)
        elif ch in "}]":
            if stack:
                stack.pop()

    if not stack and not in_string:
        return text, False

    out = text
    changed = False
    if in_string:
        out += '"'
        changed = True
    # 去掉末尾悬空的逗号/冒号再闭合
    out = re.sub(r"[,:]\s*$", "", out)
    while stack:
        opener = stack.pop()
        out += "}" if opener == "{" else "]"
        changed = True
    return out, changed


# --------------------------------------------------------------------------- #
# 候选片段提取
# --------------------------------------------------------------------------- #

def _iter_balanced_spans(text: str):
    """扫描出所有"括号平衡"的 {...} / [...] 片段（正确跳过字符串内部）。"""
    stack: list[int] = []
    in_string = False
    escaped = False
    for index, ch in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch in "{[":
            stack.append(index)
        elif ch in "}]":
            if stack:
                start = stack.pop()
                if not stack:  # 最外层闭合
                    yield text[start: index + 1]


def extract_all_json(text: str) -> list[Any]:
    """把一段文本里所有能解析出来的 JSON 值都找出来（按出现顺序）。"""
    found: list[Any] = []
    if not text:
        return found

    candidates: list[str] = []
    stripped, _ = strip_fences(text)
    candidates.append(text.strip())
    candidates.append(stripped)
    candidates.extend(_iter_balanced_spans(stripped))
    candidates.extend(_iter_balanced_spans(text))

    for candidate in candidates:
        if not candidate:
            continue
        for variant in (candidate, *[repair_text(candidate)[0]]):
            try:
                value = json.loads(variant)
            except (json.JSONDecodeError, TypeError):
                continue
            if value not in found:
                found.append(value)
    return found


def extract_json(text: str) -> ExtractionResult:
    """从模型输出里提取一个 JSON 值。按"从严格到宽容"的顺序尝试。"""
    if text is None:
        return ExtractionResult(ok=False, error="输入为 None")
    if isinstance(text, (dict, list)):
        return ExtractionResult(ok=True, value=text, method="already_parsed")

    raw = text.strip()
    if not raw:
        return ExtractionResult(ok=False, error="输入为空")

    notes: list[str] = []

    # 路径 1：直接就是合法 JSON
    try:
        return ExtractionResult(ok=True, value=json.loads(raw), method="direct", text=raw, notes=notes)
    except json.JSONDecodeError as exc:
        first_error = f"{exc.msg}（位置 {exc.pos}）"

    # 路径 2：去围栏后直接解析
    inner, fenced = strip_fences(raw)
    if fenced:
        notes.append("去掉代码围栏")
        try:
            return ExtractionResult(ok=True, value=json.loads(inner), method="strip_fence",
                                    text=inner, notes=notes)
        except json.JSONDecodeError:
            pass

    # 路径 3：从平衡括号片段里挑（优先挑最长的对象）
    spans = sorted(set(_iter_balanced_spans(inner)), key=len, reverse=True)
    for span in spans:
        try:
            value = json.loads(span)
        except json.JSONDecodeError:
            continue
        if isinstance(value, (dict, list)):
            notes.append("从文本中截取平衡括号片段")
            return ExtractionResult(ok=True, value=value, method="balanced_span", text=span, notes=notes)

    # 路径 4：文本级修复后再试
    repaired, repair_notes = repair_text(inner)
    if repair_notes:
        try:
            value = json.loads(repaired)
            notes.extend(repair_notes)
            return ExtractionResult(ok=True, value=value, method="text_repair",
                                    text=repaired, notes=notes)
        except json.JSONDecodeError:
            pass

    # 路径 5：修复 + 平衡片段
    for span in sorted(set(_iter_balanced_spans(repaired)), key=len, reverse=True):
        fixed_span, span_notes = repair_text(span)
        try:
            value = json.loads(fixed_span)
        except json.JSONDecodeError:
            continue
        if isinstance(value, (dict, list)):
            notes.extend(repair_notes + span_notes)
            notes.append("从修复后的文本中截取片段")
            return ExtractionResult(ok=True, value=value, method="repair_span",
                                    text=fixed_span, notes=notes)

    # 路径 6：闭合截断的 JSON
    closed, changed = close_truncated(repaired)
    if changed:
        try:
            value = json.loads(closed)
            notes.extend(repair_notes)
            notes.append("补齐被截断的括号/引号（内容可能不完整）")
            return ExtractionResult(ok=True, value=value, method="close_truncated",
                                    text=closed, notes=notes)
        except json.JSONDecodeError:
            pass

    return ExtractionResult(
        ok=False,
        method="failed",
        text=repaired or inner,
        notes=notes,
        error=f"无法解析出 JSON：{first_error}",
    )
