"""schemaout —— 让大模型稳定输出符合 Schema 的 JSON。

三个正交的问题，分三个模块解决：

    extract.py    文本 → Python 对象（模型爱加围栏、客套话、单引号、尾逗号）
    schema.py     Python 对象 → 是否合法（带 JSON 路径的错误）
    strategies.py 要不要加请求侧约束 / 要不要重试 / 重试几次
    grammar.py    约束解码的原理：判断"这段前缀还能补成合法 JSON 吗"
    bench.py      把上面这些组合成 6 种策略，在同一批样本上量出成功率与成本

用法::

    from schemaout import extract_json, validate, SimulatedLLM, benchmark, SAMPLES

    llm = SimulatedLLM(SAMPLES)
    reports = benchmark(llm, SAMPLES)
    print(reports[0].success_rate)
"""

from .bench import (
    StrategyReport,
    benchmark,
    render_defect_breakdown,
    render_detail,
    render_table,
    summary_dict,
)
from .samples import defect_histogram as defect_histogram_of
from .extract import ExtractionResult, extract_all_json, extract_json, repair_text
from .grammar import (
    DEFAULT_VOCABULARY,
    ParseStatus,
    allowed_tokens,
    is_complete,
    is_viable_prefix,
    parse_status,
)
from .llm import BaseLLM, Completion, LLMError, OpenAICompatLLM, SimulatedLLM
from .samples import (
    COMPANY_SCHEMA,
    ORDER_SCHEMA,
    PERSON_SCHEMA,
    SAMPLES,
    SENTIMENT_SCHEMA,
    Sample,
    apply_defect,
    schemas_used,
)
from .schema import SchemaError, ValidationResult, is_valid, schema_is_sane, validate
from .strategies import STRATEGIES, RunResult, Strategy, get_strategy, run, run_many

__version__ = "0.1.0"

__all__ = [
    # 解析
    "extract_json",
    "extract_all_json",
    "repair_text",
    "ExtractionResult",
    # 校验
    "validate",
    "is_valid",
    "schema_is_sane",
    "SchemaError",
    "ValidationResult",
    # 语法（约束解码原理）
    "parse_status",
    "is_complete",
    "is_viable_prefix",
    "allowed_tokens",
    "ParseStatus",
    "DEFAULT_VOCABULARY",
    # 模型
    "BaseLLM",
    "Completion",
    "LLMError",
    "SimulatedLLM",
    "OpenAICompatLLM",
    # 样本与策略
    "Sample",
    "SAMPLES",
    "apply_defect",
    "schemas_used",
    "PERSON_SCHEMA",
    "ORDER_SCHEMA",
    "SENTIMENT_SCHEMA",
    "COMPANY_SCHEMA",
    "Strategy",
    "STRATEGIES",
    "get_strategy",
    "RunResult",
    "run",
    "run_many",
    # 评测
    "benchmark",
    "StrategyReport",
    "render_table",
    "render_defect_breakdown",
    "render_detail",
    "summary_dict",
    "defect_histogram_of",
    "__version__",
]
