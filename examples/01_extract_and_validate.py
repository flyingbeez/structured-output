#!/usr/bin/env python3
"""示例 1：解析 + 校验 —— 模型输出的十一种坏法，逐个演示怎么兜。

运行：python examples/01_extract_and_validate.py
"""

from __future__ import annotations

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from schemaout import ORDER_SCHEMA, extract_json, validate  # noqa: E402

LINE = "=" * 84

CASES: list[tuple[str, str]] = [
    ("干净输出", '{"order_id": "ORD-1024", "amount": 199.0, "currency": "CNY"}'),
    ("```json 围栏", '```json\n{"order_id": "ORD-1024", "amount": 199.0, "currency": "CNY"}\n```'),
    ("前后客套话", '好的，我来提取一下。\n{"order_id": "ORD-1024", "amount": 199.0, "currency": "CNY"}\n希望有帮助！'),
    ("单引号", "{'order_id': 'ORD-1024', 'amount': 199.0, 'currency': 'CNY'}"),
    ("尾随逗号", '{"order_id": "ORD-1024", "amount": 199.0, "currency": "CNY",}'),
    ("裸键名", '{order_id: "ORD-1024", amount: 199.0, currency: "CNY"}'),
    ("Python 字面量+多余字段", '{"order_id": "ORD-1024", "amount": 199.0, "currency": "CNY", "ok": True}'),
    ("全角标点", '{“order_id”: “ORD-1024”, “amount”: 199.0, “currency”: “CNY”}'),
    ("被截断", '{"order_id": "ORD-1024", "amount": 19'),
    ("无法解析", "这个问题我需要再想想，暂时给不出答案。"),
    ("语法合法但值不对", '{"order_id": "ORD-1024", "amount": 199.0, "currency": "RMB"}'),
]


def main() -> None:
    print(f"\n{LINE}\n解析 + 校验：逐条演示\n{LINE}")
    print(f"{'情形':<20} {'解析':<6} {'路径':<18} {'校验':<6} 说明")
    print("-" * 84)

    for label, text in CASES:
        extraction = extract_json(text)
        if extraction.ok:
            check = validate(extraction.value, ORDER_SCHEMA)
            status = "OK" if check.ok else "FAIL"
            note = "" if check.ok else check.error_text().replace("\n", " | ")[:46]
            print(f"{label:<20} {'✅':<5} {extraction.method:<18} {status:<6} {note}")
        else:
            print(f"{label:<20} {'❌':<5} {'-':<18} {'-':<6} {extraction.error[:46]}")

    print(f"\n{LINE}\n要点\n{LINE}")
    print("  1. 解析和校验是两件事。'语法合法但值不对'（RMB 不在 enum 里）必须由 schema 兜住。")
    print("  2. 解析失败时不要抛异常炸掉流程 —— 把错误文本回灌给模型，它通常能自己改对。")
    print("  3. 被截断的输出能被'补括号'救回一部分，但内容不完整，最终应由校验器拦下。")

    print(f"\n{LINE}\n附：给模型看的 JSON Schema\n{LINE}")
    print(json.dumps(ORDER_SCHEMA, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
