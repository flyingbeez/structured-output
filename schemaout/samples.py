"""评测样本：20 条抽取任务 + 4 个 schema + 12 种"模型手滑"。

为什么样本要**手工标注坏法**而不是随机生成？
因为评测要能回答"哪一类坏法被哪种策略救回来了"。
随机生成的坏法混在一起，你只知道总体成功率，不知道怎么改进。

20 条样本的坏法分布（刻意设计，让对照表有解释力）：

    干净         8 条
    结构性坏法   8 条   fenced×2, prose×2, single_quotes, trailing_comma, bare_keys, python_literals
    语义性坏法   3 条   wrong_type, bad_enum, extra_field
    修不了       1 条   truncated（等价于被 max_tokens 截断）
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Sequence

__all__ = ["Sample", "SAMPLES", "STRUCTURAL_DEFECTS", "SEMANTIC_DEFECTS", "UNFIXABLE_DEFECTS",
           "apply_defect", "schemas_used"]


# --------------------------------------------------------------------------- #
# 四个 schema
# --------------------------------------------------------------------------- #

PERSON_SCHEMA: dict = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "name": {"type": "string", "minLength": 1, "description": "姓名"},
        "age": {"type": "integer", "minimum": 0, "maximum": 150, "description": "年龄"},
        "city": {"type": "string", "description": "所在城市"},
    },
    "required": ["name", "age", "city"],
}

ORDER_SCHEMA: dict = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "order_id": {"type": "string", "pattern": r"^ORD-\d{4}$", "description": "订单号"},
        "amount": {"type": "number", "exclusiveMinimum": 0, "description": "金额"},
        "currency": {"type": "string", "enum": ["CNY", "USD", "EUR"], "description": "币种"},
    },
    "required": ["order_id", "amount", "currency"],
}

SENTIMENT_SCHEMA: dict = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "label": {"type": "string", "enum": ["positive", "negative", "neutral"]},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "reason": {"type": "string", "maxLength": 80},
    },
    "required": ["label", "confidence", "reason"],
}

COMPANY_SCHEMA: dict = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "company": {"type": "string", "minLength": 1},
        "is_public": {"type": "boolean", "description": "是否上市"},
        "employees": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "name": {"type": "string"},
                    "role": {"type": "string"},
                },
                "required": ["name", "role"],
            },
        },
    },
    "required": ["company", "is_public", "employees"],
}


# --------------------------------------------------------------------------- #
# 坏法分类
# --------------------------------------------------------------------------- #

STRUCTURAL_DEFECTS = frozenset({
    "fenced", "prose", "single_quotes", "trailing_comma", "bare_keys", "python_literals",
})
"""纯语法层面的坏法 —— 一个足够强的解析层就能救回来。"""

SEMANTIC_DEFECTS = frozenset({"wrong_type", "bad_enum", "extra_field"})
"""语法合法但值不对 —— 必须靠 schema 校验 + 反馈修复。"""

UNFIXABLE_DEFECTS = frozenset({"truncated"})
"""重试也修不了（真实原因通常是 max_tokens 太小）。"""

ALL_DEFECTS = STRUCTURAL_DEFECTS | SEMANTIC_DEFECTS | UNFIXABLE_DEFECTS | {"clean"}


@dataclass
class Sample:
    id: str
    instruction: str
    schema: dict
    gold: Any
    defect: str = "clean"
    defect_field: str = ""
    defect_value: Any = None
    tag: str = ""

    def __post_init__(self) -> None:
        if self.defect not in ALL_DEFECTS:
            raise ValueError(f"未知坏法 {self.defect!r}")

    @property
    def clean_text(self) -> str:
        return json.dumps(self.gold, ensure_ascii=False)


# --------------------------------------------------------------------------- #
# 坏法注入
# --------------------------------------------------------------------------- #

def apply_defect(sample: Sample) -> str:
    """把"模型的正确输出"变成一个带有指定坏法的文本。"""
    js = sample.clean_text
    defect = sample.defect

    if defect == "clean":
        return js

    if defect == "fenced":
        return f"```json\n{js}\n```"

    if defect == "prose":
        return f"好的，我来帮你提取一下。\n\n{js}\n\n以上就是提取结果，希望对你有所帮助！"

    if defect == "single_quotes":
        return js.replace('"', "'")

    if defect == "trailing_comma":
        return js[:-1] + ",}" if js.endswith("}") else js + ","

    if defect == "bare_keys":
        return re.sub(r'"([A-Za-z_][A-Za-z0-9_]*)"(\s*:)', r"\1\2", js)

    if defect == "python_literals":
        text = re.sub(r"\btrue\b", "True", js)
        text = re.sub(r"\bfalse\b", "False", text)
        text = re.sub(r"\bnull\b", "None", text)
        return text

    if defect == "truncated":
        cut = max(1, int(len(js) * 0.6))
        return js[:cut]

    # --- 语义性坏法：改的是"值"，语法仍然合法 ---------------- #
    if defect in SEMANTIC_DEFECTS:
        mutated = json.loads(js)
        if defect == "wrong_type":
            # 刻意用一个**无法被纠偏**的值。
            # 如果写成 "23"（数字的字符串形式），schema.py 的 coerce 会把它悄悄改成 23，
            # 那这条样本就测不出"语义错误需要反馈修复"了 —— 这是一个真实的陷阱：
            # 过度宽容的纠偏会掩盖模型质量问题。
            mutated[sample.defect_field] = "未知"
        elif defect == "bad_enum":
            mutated[sample.defect_field] = sample.defect_value
        elif defect == "extra_field":
            mutated[sample.defect_field] = sample.defect_value
        return json.dumps(mutated, ensure_ascii=False)

    raise ValueError(f"未处理的坏法：{defect}")


# --------------------------------------------------------------------------- #
# 20 条样本
# --------------------------------------------------------------------------- #

SAMPLES: list[Sample] = [
    # ---- 人员抽取（5 条） ------------------------------------------------ #
    Sample("p1", "从「张三，28岁，现居杭州。」中抽取人员信息。", PERSON_SCHEMA,
           {"name": "张三", "age": 28, "city": "杭州"}),
    Sample("p2", "从「李四，35岁，现居北京。」中抽取人员信息。", PERSON_SCHEMA,
           {"name": "李四", "age": 35, "city": "北京"}, defect="fenced"),
    Sample("p3", "从「王五，41岁，现居上海。」中抽取人员信息。", PERSON_SCHEMA,
           {"name": "王五", "age": 41, "city": "上海"}, defect="prose"),
    Sample("p4", "从「赵六，23岁，现居深圳。」中抽取人员信息。", PERSON_SCHEMA,
           {"name": "赵六", "age": 23, "city": "深圳"},
           defect="wrong_type", defect_field="age"),
    Sample("p5", "从「钱七，30岁，现居成都。」中抽取人员信息。", PERSON_SCHEMA,
           {"name": "钱七", "age": 30, "city": "成都"},
           defect="extra_field", defect_field="note", defect_value="这是一条多余的解释"),

    # ---- 订单抽取（5 条） ------------------------------------------------ #
    Sample("o1", "从「订单 ORD-1024，金额 199.00 元人民币。」中抽取订单信息。", ORDER_SCHEMA,
           {"order_id": "ORD-1024", "amount": 199.0, "currency": "CNY"}),
    Sample("o2", "从「订单 ORD-2048，金额 59.9 美元。」中抽取订单信息。", ORDER_SCHEMA,
           {"order_id": "ORD-2048", "amount": 59.9, "currency": "USD"}, defect="single_quotes"),
    Sample("o3", "从「订单 ORD-3072，金额 1280 欧元。」中抽取订单信息。", ORDER_SCHEMA,
           {"order_id": "ORD-3072", "amount": 1280.0, "currency": "EUR"}, defect="trailing_comma"),
    Sample("o4", "从「订单 ORD-4096，金额 88 元。」中抽取订单信息。", ORDER_SCHEMA,
           {"order_id": "ORD-4096", "amount": 88.0, "currency": "CNY"},
           defect="bad_enum", defect_field="currency", defect_value="RMB"),
    Sample("o5", "从「订单 ORD-5120，金额 3000 元。」中抽取订单信息。", ORDER_SCHEMA,
           {"order_id": "ORD-5120", "amount": 3000.0, "currency": "CNY"}, defect="truncated"),

    # ---- 情感分类（5 条） ------------------------------------------------ #
    Sample("c1", "判断情感：「这个产品太好用了，强烈推荐！」", SENTIMENT_SCHEMA,
           {"label": "positive", "confidence": 0.95, "reason": "出现强烈正面评价"}),
    Sample("c2", "判断情感：「客服态度很差，再也不买了。」", SENTIMENT_SCHEMA,
           {"label": "negative", "confidence": 0.9, "reason": "明确负面表述"},
           defect="fenced"),
    Sample("c3", "判断情感：「东西还行吧，没什么特别的。」", SENTIMENT_SCHEMA,
           {"label": "neutral", "confidence": 0.6, "reason": "表述中立"}, defect="bare_keys"),
    Sample("c4", "判断情感：「发货很快，但是包装有破损。」", SENTIMENT_SCHEMA,
           {"label": "neutral", "confidence": 0.55, "reason": "褒贬并存"}),
    Sample("c5", "判断情感：「性价比非常高，会回购。」", SENTIMENT_SCHEMA,
           {"label": "positive", "confidence": 0.88, "reason": "正面购买意愿"}),

    # ---- 嵌套结构（5 条） ------------------------------------------------ #
    Sample("d1", "从「字节跳动有员工张一鸣，职位创始人。」中抽取公司信息。", COMPANY_SCHEMA,
           {"company": "字节跳动", "is_public": False,
            "employees": [{"name": "张一鸣", "role": "创始人"}]}),
    Sample("d2", "从「腾讯有员工马化腾，职位董事会主席。」中抽取公司信息。", COMPANY_SCHEMA,
           {"company": "腾讯", "is_public": True,
            "employees": [{"name": "马化腾", "role": "董事会主席"}]},
           defect="prose"),
    Sample("d3", "从「阿里巴巴有员工马云，职位创始人。」中抽取公司信息。", COMPANY_SCHEMA,
           {"company": "阿里巴巴", "is_public": True,
            "employees": [{"name": "马云", "role": "创始人"}]},
           defect="python_literals"),
    Sample("d4", "从「美团有员工王兴，职位 CEO。」中抽取公司信息。", COMPANY_SCHEMA,
           {"company": "美团", "is_public": True,
            "employees": [{"name": "王兴", "role": "CEO"}]}),
    Sample("d5", "从「小米有员工雷军，职位董事长。」中抽取公司信息。", COMPANY_SCHEMA,
           {"company": "小米", "is_public": True,
            "employees": [{"name": "雷军", "role": "董事长"}]}),
]


def schemas_used() -> dict[str, dict]:
    """返回样本用到的 schema 集合（给报告与测试用）。"""
    return {
        "person": PERSON_SCHEMA,
        "order": ORDER_SCHEMA,
        "sentiment": SENTIMENT_SCHEMA,
        "company": COMPANY_SCHEMA,
    }


def defect_histogram(samples: "Sequence[Sample] | None" = None) -> dict[str, int]:
    """统计各坏法的样本数。不传参就统计内置的 20 条样本。"""
    items = SAMPLES if samples is None else samples
    counts: dict[str, int] = {}
    for sample in items:
        counts[sample.defect] = counts.get(sample.defect, 0) + 1
    return counts
