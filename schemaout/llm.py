"""模型客户端：离线模拟器 + 真实 OpenAI 兼容接口。

**离线模拟器是这个仓库能被信任的关键。**
结构化输出的评测如果依赖真实 LLM，就变成"每次跑数字都不一样、还要花钱、还可能因为
限流跑不完"。所以这里把"模型会怎么手滑"编码成一个确定性的模拟器：
每条样本的坏法是**手工标注**的，跑多少次结果都一样。

**但必须诚实说明**：模拟器只能证明"我的解析与修复逻辑是否正确处理了这些坏法"，
不能证明"真实模型的手滑分布就是这样"。所以同一个评测脚本支持 `--live`，
在有 API Key 时对真实模型跑同一套流程 —— 结论要以后者为准。
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Sequence

__all__ = ["Completion", "BaseLLM", "SimulatedLLM", "OpenAICompatLLM", "LLMError"]


class LLMError(RuntimeError):
    """模型调用失败。"""


@dataclass
class Completion:
    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    model: str = ""
    mode: str = ""

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class BaseLLM:
    """接口：只要能按 (system, user, schema, mode) 产出文本就能接进评测。"""

    model = "base"

    def complete(self, *, system: str, user: str, schema: dict | None = None,
                 mode: str = "text", meta: dict | None = None) -> Completion:  # pragma: no cover
        raise NotImplementedError


# --------------------------------------------------------------------------- #
# 离线模拟器
# --------------------------------------------------------------------------- #

def _estimate_tokens(text: str) -> int:
    """粗略的 token 估算：中文约 1 字 1 token，英文约 4 字符 1 token。

    只用于**策略之间的相对比较**，不是真实计费口径。
    """
    if not text:
        return 0
    cjk = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
    other = len(text) - cjk
    return max(1, cjk + other // 4)


class SimulatedLLM(BaseLLM):
    """确定性模拟模型：按样本的坏法标注产出文本。

    行为规则（与真实模型的常见表现对齐）：

    | 坏法 | text 模式 | json 模式 | tool 模式 |
    |---|---|---|---|
    | 结构性（围栏/客套话/单引号/尾逗号/裸键名/Python 字面量） | 原样输出 | 修好 | 修好 |
    | 语义性（类型错/枚举错） | 原样输出 | 原样输出 | 原样输出 |
    | 多余字段 | 原样输出 | 原样输出 | 修好（schema 里 additionalProperties=false 的约束生效） |
    | 截断 | 原样输出 | 原样输出 | 原样输出（max_tokens 问题，格式约束救不了） |
    | 修复轮（round > 1） | 修好 | 修好 | 修好 |

    也就是说：**格式约束能治好"语法"，治不好"语义"；只有反馈修复能治语义。**
    """

    model = "simulated-llm"

    def __init__(self, samples: Sequence[Any]) -> None:
        self._by_id = {s.id: s for s in samples}
        self.calls: list[dict] = []

    def complete(self, *, system: str, user: str, schema: dict | None = None,
                 mode: str = "text", meta: dict | None = None) -> Completion:
        meta = meta or {}
        sample_id = meta.get("sample_id")
        round_no = int(meta.get("round", 1))
        sample = self._by_id.get(sample_id)
        if sample is None:
            raise LLMError(f"模拟器不认识样本 {sample_id!r}")

        from .samples import STRUCTURAL_DEFECTS, UNFIXABLE_DEFECTS, apply_defect

        defect = sample.defect
        if round_no > 1:
            # 收到精确的校验错误后，除"物理上修不了"的截断之外都给出修正版
            effective = "truncated" if defect in UNFIXABLE_DEFECTS else "clean"
        elif mode in ("json", "tool") and defect in STRUCTURAL_DEFECTS:
            effective = "clean"
        elif mode == "tool" and defect == "extra_field":
            effective = "clean"
        else:
            effective = defect

        probe = type(sample)(id=sample.id, instruction=sample.instruction, schema=sample.schema,
                             gold=sample.gold, defect=effective,
                             defect_field=sample.defect_field, defect_value=sample.defect_value)
        text = apply_defect(probe)

        prompt_tokens = _estimate_tokens(system) + _estimate_tokens(user)
        completion_tokens = _estimate_tokens(text)
        self.calls.append({
            "sample_id": sample_id, "round": round_no, "mode": mode,
            "prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
        })
        return Completion(text=text, prompt_tokens=prompt_tokens,
                          completion_tokens=completion_tokens, model=self.model, mode=mode)


# --------------------------------------------------------------------------- #
# 真实接口
# --------------------------------------------------------------------------- #

class OpenAICompatLLM(BaseLLM):
    """调用任何 OpenAI 兼容的 /chat/completions，仅用标准库。

    三种模式对应的请求差异（这就是"结构化输出的三种做法"）：

    * ``mode="text"``：什么都不加，靠提示词约束。
    * ``mode="json"``：加 ``response_format={"type": "json_object"}``。
      服务端保证返回**语法合法**的 JSON，但不保证符合你的 schema。
    * ``mode="tool"``：把 schema 塞进一个 function 的 parameters，
      并用 ``tool_choice`` 强制调用它。返回值在 ``tool_calls[0].function.arguments``
      里 —— 这是**在 Structured Outputs 普及之前最可靠的格式约束手段**。
    """

    def __init__(self, api_key: str, *, model: str = "deepseek-chat",
                 base_url: str = "https://api.deepseek.com/v1", timeout: float = 90.0,
                 max_retries: int = 2, temperature: float = 0.0) -> None:
        if not api_key:
            raise LLMError("api_key 为空")
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self.temperature = temperature

    @classmethod
    def from_env(cls) -> "OpenAICompatLLM":
        key = os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("LLM_API_KEY")
        if not key:
            raise LLMError(
                "没有找到 API Key（DEEPSEEK_API_KEY / LLM_API_KEY）。"
                "离线评测请用 SimulatedLLM。"
            )
        return cls(
            key,
            model=os.environ.get("LLM_MODEL", "deepseek-chat"),
            base_url=os.environ.get("LLM_BASE_URL", "https://api.deepseek.com/v1"),
        )

    def _payload(self, system: str, user: str, schema: dict | None, mode: str) -> dict:
        body: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
            "temperature": self.temperature,
        }
        if mode == "json":
            body["response_format"] = {"type": "json_object"}
        elif mode == "tool":
            if not schema:
                raise LLMError("tool 模式必须提供 schema")
            body["tools"] = [{
                "type": "function",
                "function": {"name": "emit_result", "description": "输出结构化结果", "parameters": schema},
            }]
            body["tool_choice"] = {"type": "function", "function": {"name": "emit_result"}}
        return body

    def complete(self, *, system: str, user: str, schema: dict | None = None,
                 mode: str = "text", meta: dict | None = None) -> Completion:
        payload = json.dumps(self._payload(system, user, schema, mode), ensure_ascii=False).encode("utf-8")
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"}

        last: Exception | None = None
        for attempt in range(self.max_retries + 1):
            request = urllib.request.Request(f"{self.base_url}/chat/completions",
                                             data=payload, headers=headers, method="POST")
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    raw = json.loads(response.read().decode("utf-8"))
                return self._parse(raw, mode)
            except urllib.error.HTTPError as exc:
                detail = ""
                try:
                    detail = exc.read().decode("utf-8", "replace")[:300]
                except Exception:  # noqa: BLE001
                    pass
                last = LLMError(f"HTTP {exc.code}: {detail}")
                if exc.code in (400, 401, 403, 404, 422):
                    raise last from exc
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
                last = LLMError(f"{type(exc).__name__}: {exc}")
            if attempt < self.max_retries:
                time.sleep(0.8 * (2 ** attempt))
        raise last or LLMError("未知错误")

    @staticmethod
    def _parse(raw: dict, mode: str) -> Completion:
        choices = raw.get("choices") or []
        if not choices:
            raise LLMError(f"响应缺少 choices：{json.dumps(raw, ensure_ascii=False)[:200]}")
        message = choices[0].get("message") or {}
        usage = raw.get("usage") or {}

        text = message.get("content") or ""
        if mode == "tool":
            calls = message.get("tool_calls") or []
            if not calls:
                raise LLMError("tool 模式下调用了零个工具")
            text = (calls[0].get("function") or {}).get("arguments") or ""

        return Completion(
            text=text,
            prompt_tokens=int(usage.get("prompt_tokens") or 0),
            completion_tokens=int(usage.get("completion_tokens") or 0),
            model=raw.get("model", ""),
            mode=mode,
        )
