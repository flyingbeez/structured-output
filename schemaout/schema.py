"""轻量 JSON Schema 校验器（常用子集），错误带 JSON 路径。

为什么自己写而不装 jsonschema？
  1. 这个仓库是零依赖的，clone 即可运行；
  2. 更重要的是：只有自己实现一遍，才知道"约束解码"里那个"约束"到底约束了什么。

支持的子集：
  type / enum / const / properties / required / additionalProperties
  items / minItems / maxItems / uniqueItems
  minimum / maximum / exclusiveMinimum / exclusiveMaximum / multipleOf
  minLength / maxLength / pattern
  anyOf / allOf / oneOf

不支持（在 README 的局限里写明）：$ref / $defs、format 语义校验、patternProperties、
dependentRequired、if-then-else。
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from typing import Any

__all__ = ["SchemaError", "ValidationResult", "validate", "json_path", "is_valid"]


@dataclass
class SchemaError:
    """一条校验错误。path 形如 ``$.items[0].price``。"""

    path: str
    message: str
    keyword: str = ""

    def __str__(self) -> str:  # pragma: no cover - 展示用
        return f"{self.path}: {self.message}"

    def to_dict(self) -> dict:
        return {"path": self.path, "message": self.message, "keyword": self.keyword}


@dataclass
class ValidationResult:
    ok: bool
    errors: list[SchemaError] = field(default_factory=list)
    value: Any = None

    def error_text(self) -> str:
        """给模型看的错误说明 —— 精炼、带路径、可操作。"""
        if self.ok:
            return ""
        return "\n".join(f"- {e.path}: {e.message}" for e in self.errors)


def json_path(parts: list[Any]) -> str:
    out = "$"
    for part in parts:
        out += f"[{part}]" if isinstance(part, int) else f".{part}"
    return out


# --------------------------------------------------------------------------- #
# 类型判定
# --------------------------------------------------------------------------- #

_TYPE_MAP = {
    "object": dict,
    "array": list,
    "string": str,
    "boolean": bool,
    "null": type(None),
}


def _type_of(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    return type(value).__name__


def _matches_type(value: Any, expected: str) -> bool:
    actual = _type_of(value)
    if expected == "number":
        # JSON Schema 里 integer 也是 number
        return actual in ("number", "integer")
    if expected == "integer":
        return actual == "integer" or (actual == "number" and float(value).is_integer())
    return actual == expected


# --------------------------------------------------------------------------- #
# 主校验器
# --------------------------------------------------------------------------- #

def _validate(value: Any, schema: Any, path: list[Any], errors: list[SchemaError],
              *, coerce: bool) -> Any:
    if schema is True or schema == {}:
        return value
    if schema is False:
        errors.append(SchemaError(json_path(path), "该位置不允许出现任何值", "false"))
        return value
    if not isinstance(schema, dict):
        return value

    # --- 组合关键字 ---------------------------------------------------- #
    if "allOf" in schema:
        for sub in schema["allOf"]:
            value = _validate(value, sub, path, errors, coerce=coerce)

    if "anyOf" in schema:
        matched = False
        branch_errors: list[SchemaError] = []
        for sub in schema["anyOf"]:
            trial: list[SchemaError] = []
            candidate = _validate(value, sub, path, trial, coerce=coerce)
            if not trial:
                matched = True
                value = candidate
                break
            branch_errors.extend(trial)
        if not matched:
            types = " / ".join(str(s.get("type", "?")) for s in schema["anyOf"] if isinstance(s, dict))
            errors.append(SchemaError(json_path(path), f"不匹配任何一种允许的形式（{types}）", "anyOf"))

    if "oneOf" in schema:
        hits = 0
        winner = value
        for sub in schema["oneOf"]:
            trial = []
            candidate = _validate(value, sub, path, trial, coerce=coerce)
            if not trial:
                hits += 1
                winner = candidate
        if hits != 1:
            errors.append(SchemaError(json_path(path), f"必须恰好匹配一种形式，实际匹配 {hits} 种", "oneOf"))
        else:
            value = winner

    # --- type ---------------------------------------------------------- #
    expected = schema.get("type")
    if expected is not None:
        expected_list = expected if isinstance(expected, list) else [expected]
        if not any(_matches_type(value, e) for e in expected_list):
            if coerce:
                value = _coerce(value, expected_list[0])
            if not any(_matches_type(value, e) for e in expected_list):
                errors.append(SchemaError(
                    json_path(path),
                    f"类型应为 {expected_list[0]}，实际是 {_type_of(value)}（值：{_short(value)}）",
                    "type",
                ))
                return value

    # --- const / enum --------------------------------------------------- #
    if "const" in schema and value != schema["const"]:
        errors.append(SchemaError(json_path(path), f"必须等于 {schema['const']!r}，实际 {value!r}", "const"))

    if "enum" in schema and value not in schema["enum"]:
        errors.append(SchemaError(
            json_path(path),
            f"取值必须是 {schema['enum']} 之一，实际 {value!r}",
            "enum",
        ))

    # --- 数值 ----------------------------------------------------------- #
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            errors.append(SchemaError(json_path(path), f"不能小于 {schema['minimum']}，实际 {value}", "minimum"))
        if "maximum" in schema and value > schema["maximum"]:
            errors.append(SchemaError(json_path(path), f"不能大于 {schema['maximum']}，实际 {value}", "maximum"))
        if "exclusiveMinimum" in schema and value <= schema["exclusiveMinimum"]:
            errors.append(SchemaError(json_path(path), f"必须大于 {schema['exclusiveMinimum']}，实际 {value}",
                                      "exclusiveMinimum"))
        if "exclusiveMaximum" in schema and value >= schema["exclusiveMaximum"]:
            errors.append(SchemaError(json_path(path), f"必须小于 {schema['exclusiveMaximum']}，实际 {value}",
                                      "exclusiveMaximum"))
        if "multipleOf" in schema:
            quotient = value / schema["multipleOf"]
            if abs(quotient - round(quotient)) > 1e-9:
                errors.append(SchemaError(json_path(path), f"必须是 {schema['multipleOf']} 的整数倍，实际 {value}",
                                          "multipleOf"))

    # --- 字符串 --------------------------------------------------------- #
    if isinstance(value, str):
        if "minLength" in schema and len(value) < schema["minLength"]:
            errors.append(SchemaError(json_path(path), f"长度不能小于 {schema['minLength']}，实际 {len(value)}",
                                      "minLength"))
        if "maxLength" in schema and len(value) > schema["maxLength"]:
            errors.append(SchemaError(json_path(path), f"长度不能大于 {schema['maxLength']}，实际 {len(value)}",
                                      "maxLength"))
        if schema.get("pattern"):
            try:
                if not re.search(schema["pattern"], value):
                    errors.append(SchemaError(json_path(path),
                                              f"不匹配模式 {schema['pattern']!r}，实际 {value!r}", "pattern"))
            except re.error as exc:  # schema 自己写错了
                errors.append(SchemaError(json_path(path), f"schema 里的 pattern 非法：{exc}", "pattern"))

    # --- 数组 ----------------------------------------------------------- #
    if isinstance(value, list):
        if "minItems" in schema and len(value) < schema["minItems"]:
            errors.append(SchemaError(json_path(path), f"元素个数不能少于 {schema['minItems']}，实际 {len(value)}",
                                      "minItems"))
        if "maxItems" in schema and len(value) > schema["maxItems"]:
            errors.append(SchemaError(json_path(path), f"元素个数不能多于 {schema['maxItems']}，实际 {len(value)}",
                                      "maxItems"))
        if schema.get("uniqueItems"):
            seen: list[Any] = []
            for item in value:
                key = json.dumps(item, sort_keys=True, ensure_ascii=False, default=str)
                if key in seen:
                    errors.append(SchemaError(json_path(path), f"元素必须唯一，{_short(item)} 重复出现",
                                              "uniqueItems"))
                    break
                seen.append(key)
        if "items" in schema:
            value = [_validate(item, schema["items"], path + [i], errors, coerce=coerce)
                     for i, item in enumerate(value)]

    # --- 对象 ----------------------------------------------------------- #
    if isinstance(value, dict):
        properties = schema.get("properties", {})
        required = schema.get("required", [])

        for name in required:
            if name not in value:
                errors.append(SchemaError(json_path(path + [name]), "缺少必填字段", "required"))

        cleaned: dict[str, Any] = {}
        for key, item in value.items():
            if key in properties:
                cleaned[key] = _validate(item, properties[key], path + [key], errors, coerce=coerce)
            else:
                extra = schema.get("additionalProperties", True)
                if extra is False:
                    errors.append(SchemaError(json_path(path + [key]), "不允许出现这个额外字段",
                                              "additionalProperties"))
                else:
                    cleaned[key] = item

        # default 填充（只补缺失的，不覆盖模型给的值）
        for name, sub in properties.items():
            if name not in cleaned and isinstance(sub, dict) and "default" in sub:
                cleaned[name] = sub["default"]

        value = cleaned

    return value


def _coerce(value: Any, wanted: str) -> Any:
    """保守纠偏：只在无歧义时转换。

    **刻意不做"数字 → 字符串"的转换。** 模型的常见手滑是 `"5"` 写成 5 的反面
    （把数字写成字符串），所以 `"5" → 5` 值得救。反过来把 5 变成 "5" 却是在
    掩盖类型错误 —— 一个本该是字符串的字段收到了整数，说明模型理解错了，
    应该报错而不是悄悄转换。
    """
    if wanted == "integer" and isinstance(value, str):
        try:
            return int(value.strip())
        except ValueError:
            return value
    if wanted == "number" and isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return value
    if wanted == "boolean" and isinstance(value, str):
        low = value.strip().lower()
        if low in ("true", "yes", "1"):
            return True
        if low in ("false", "no", "0"):
            return False
    if wanted == "integer" and isinstance(value, float) and value.is_integer():
        return int(value)
    return value


def _short(value: Any, limit: int = 40) -> str:
    text = json.dumps(value, ensure_ascii=False, default=str) if not isinstance(value, str) else value
    return text if len(text) <= limit else text[: limit - 1] + "…"


def validate(instance: Any, schema: Any, *, coerce: bool = True) -> ValidationResult:
    """校验一个已解析的 Python 对象是否符合 schema。"""
    errors: list[SchemaError] = []
    try:
        cleaned = _validate(instance, schema, [], errors, coerce=coerce)
    except RecursionError:
        errors.append(SchemaError("$", "schema 嵌套过深", "recursion"))
        cleaned = instance
    return ValidationResult(ok=not errors, errors=errors, value=cleaned)


def is_valid(instance: Any, schema: Any, *, coerce: bool = True) -> bool:
    return validate(instance, schema, coerce=coerce).ok


def schema_is_sane(schema: dict) -> list[str]:
    """粗查 schema 自己有没有写错（面试常问：schema 本身也可能是 bug 源）。"""
    problems: list[str] = []

    def walk(node: Any, path: str) -> None:
        if not isinstance(node, dict):
            return
        if "type" in node:
            allowed = {"object", "array", "string", "number", "integer", "boolean", "null"}
            values = node["type"] if isinstance(node["type"], list) else [node["type"]]
            for item in values:
                if item not in allowed:
                    problems.append(f"{path}: 未知 type {item!r}")
        if node.get("type") == "object" and "properties" not in node and "additionalProperties" not in node:
            problems.append(f"{path}: object 既没写 properties 也没写 additionalProperties（等于不约束）")
        if "required" in node and "properties" in node:
            for name in node["required"]:
                if name not in node["properties"]:
                    problems.append(f"{path}: required 里的 {name!r} 不在 properties 里")
        if "minimum" in node and "maximum" in node and node["minimum"] > node["maximum"]:
            problems.append(f"{path}: minimum > maximum，永远无法满足")
        if isinstance(node.get("enum"), list) and len(node["enum"]) == 0:
            problems.append(f"{path}: enum 为空，永远无法满足")
        for key in ("anyOf", "allOf", "oneOf"):
            for i, sub in enumerate(node.get(key, []) or []):
                walk(sub, f"{path}.{key}[{i}]")
        if isinstance(node.get("properties"), dict):
            for name, sub in node["properties"].items():
                walk(sub, f"{path}.{name}")
        if isinstance(node.get("items"), dict):
            walk(node["items"], f"{path}[]")

    walk(schema, "$")
    return problems


def _unused_math_guard() -> float:  # pragma: no cover
    return math.nan
