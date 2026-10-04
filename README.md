# structured-output

> 让大模型稳定吐出**符合 Schema 的 JSON** —— 解析修复 + 校验 + 反馈重试 + **策略对照评测**。
> 零第三方依赖，不需要 API Key 就能跑出可复现的数字。

[![tests](https://github.com/flyingbeez/structured-output/actions/workflows/test.yml/badge.svg)](https://github.com/flyingbeez/structured-output/actions/workflows/test.yml)
![python](https://img.shields.io/badge/python-3.10%2B-blue)
![deps](https://img.shields.io/badge/dependencies-0-brightgreen)
![tests](https://img.shields.io/badge/tests-132%20passed-success)

---

## 它回答什么问题

结构化输出有一堆"众所周知"的建议：开 JSON 模式、用 function calling、上 Pydantic、
加校验重试……但很少有人量化过：**这些手段各自值多少？哪些是重复的？代价是什么？**

这个仓库把 20 条抽取样本 × 6 种策略跑成一张对照表，结论和直觉不太一样：

| 策略 | 成功率 | 通过/总数 | 平均尝试次数 | 每样本 token | 相对基线 token |
|---|---|---|---|---|---|
| `naive`（直接 `json.loads`） | **40.0%** | 8/20 | 1.00 | 245 | 1.00× |
| `json_mode`（`response_format=json_object`） | **80.0%** | 16/20 | 1.00 | 242 | 0.99× |
| `extract`（只加强解析层） | **80.0%** | 16/20 | 1.00 | 245 | 1.00× |
| `json_extract`（两者都上） | **80.0%** | 16/20 | 1.00 | 242 | 0.99× |
| `tool_call`（强制 function call） | **85.0%** | 17/20 | 1.00 | 241 | 0.99× |
| `repair_loop`（校验 + 反馈重试 3 轮） | **95.0%** | 19/20 | 1.25 | 313 | 1.28× |

### 三个反直觉的结论

**1. 最大的收益来自客户端解析层，而且免费。**
`naive → extract` 提升 40 个百分点，**不增加任何请求成本**（平均尝试次数仍是 1.00）。
你只是把"模型说的 JSON"从文本里抠出来而已。

**2. JSON 模式和强解析层是替代关系，不是叠加关系。**
两者都恰好停在 80%，`json_extract` 也是 80% —— 因为它们解决的是**同一类问题：语法**。
如果你已经有一个足够强的解析层，`response_format` 的边际收益接近零。
（反过来说，如果你只有一个 `json.loads`，JSON 模式的收益就非常大：40% → 80%。）

**3. 格式约束治不了"值不对"，只有反馈修复能治。**
按坏法拆开看就清楚了：

| 坏法 | 类别 | 样本数 | `naive` | `json_mode` | `extract` | `json_extract` | `tool_call` | `repair_loop` |
|---|---|---|---|---|---|---|---|---|
| 干净输出 | — | 8 | ✅ 8/8 | ✅ 8/8 | ✅ 8/8 | ✅ 8/8 | ✅ 8/8 | ✅ 8/8 |
| ` ```json ` 围栏 | 结构性 | 2 | ❌ 0/2 | ✅ 2/2 | ✅ 2/2 | ✅ 2/2 | ✅ 2/2 | ✅ 2/2 |
| 前后客套话 | 结构性 | 2 | ❌ 0/2 | ✅ 2/2 | ✅ 2/2 | ✅ 2/2 | ✅ 2/2 | ✅ 2/2 |
| 单引号 | 结构性 | 1 | ❌ 0/1 | ✅ 1/1 | ✅ 1/1 | ✅ 1/1 | ✅ 1/1 | ✅ 1/1 |
| 尾随逗号 | 结构性 | 1 | ❌ 0/1 | ✅ 1/1 | ✅ 1/1 | ✅ 1/1 | ✅ 1/1 | ✅ 1/1 |
| 裸键名 | 结构性 | 1 | ❌ 0/1 | ✅ 1/1 | ✅ 1/1 | ✅ 1/1 | ✅ 1/1 | ✅ 1/1 |
| Python 字面量 | 结构性 | 1 | ❌ 0/1 | ✅ 1/1 | ✅ 1/1 | ✅ 1/1 | ✅ 1/1 | ✅ 1/1 |
| 类型错误 | 语义性 | 1 | ❌ 0/1 | ❌ 0/1 | ❌ 0/1 | ❌ 0/1 | ❌ 0/1 | ✅ 1/1 |
| 枚举值错误 | 语义性 | 1 | ❌ 0/1 | ❌ 0/1 | ❌ 0/1 | ❌ 0/1 | ❌ 0/1 | ✅ 1/1 |
| 多余字段 | 语义性 | 1 | ❌ 0/1 | ❌ 0/1 | ❌ 0/1 | ❌ 0/1 | ✅ 1/1 | ✅ 1/1 |
| 被截断 | 修不了 | 1 | ❌ 0/1 | ❌ 0/1 | ❌ 0/1 | ❌ 0/1 | ❌ 0/1 | ❌ 0/1 |

**一句话总结**：**语法问题靠解析层或格式约束，语义问题只能靠校验 + 反馈。**

## 数据可复现性（重要，请如实看待）

这份表里模型是**确定性模拟器**，20 条样本的坏法是**手工标注**的：

- ✅ 它证明的是：**我的解析 / 校验 / 修复逻辑是否正确处理了这 11 类坏法**；
- ❌ 它**不能**证明：真实模型的手滑分布就是这样。

真实分布会随模型、提示词、温度变化。所以同一个评测脚本支持 `--live`：

```bash
export DEEPSEEK_API_KEY=sk-xxx
python examples/03_strategy_benchmark.py --live
```

跑同一批样本、同一套流程，**结论要以后者为准**。我把它做成模拟器优先，
是因为"每次数字都不一样、还要花钱、还可能跑不完"的评测根本没法当回归门禁用。

另外，`docs/02` 里记录了一个真实踩到的坑：一开始我把"类型错误"样本设成 `"age": "23"`，
结果校验器的类型纠偏把 `"23"` 悄悄变成了 `23`，这条样本就测不出语义错误了 ——
**过度宽容的纠偏会掩盖模型质量问题**。现在这条样本用的是无法纠偏的 `"未知"`。

## 四个正交的能力

```
模型返回的文本
      │
      ▼
┌──────────────────┐
│ extract.py       │  文本 → Python 对象
│ 六条降级路径      │  围栏 / 客套话 / 单引号 / 尾逗号 / 裸键名 /
│                  │  Python 字面量 / 全角标点 / 截断补全
└────────┬─────────┘
         ▼
┌──────────────────┐
│ schema.py        │  Python 对象 → 是否合法
│ 带 JSON 路径的    │  type / enum / const / pattern / 数值范围 /
│ 错误             │  minItems / uniqueItems / required /
│                  │  additionalProperties / anyOf / oneOf / allOf
└────────┬─────────┘
         │ 失败
         ▼
┌──────────────────┐
│ strategies.py    │  把 error_text 精确回灌给模型 → 重试
│ repair_loop      │  （错误带 $.path，模型能直接定位）
└────────┬─────────┘
         ▼
┌──────────────────┐
│ grammar.py       │  约束解码的原理：判断"这段前缀还能补成合法 JSON 吗"
│ 三种状态         │  complete / viable / invalid + token 掩码
└──────────────────┘
```

## 快速开始

```bash
git clone https://github.com/flyingbeez/structured-output.git
cd structured-output

python run_tests.py                              # 132 个测试，约 0.03 秒
python examples/01_extract_and_validate.py       # 11 种坏法逐条演示
python examples/02_repair_loop.py                # 修复循环里到底发了什么
python examples/03_strategy_benchmark.py         # ← 上面那张对照表
python examples/04_constrained_decoding.py       # 约束解码的 token 掩码
```

**零依赖、不需要 API Key、不需要联网。**

## 三行接入

```python
from schemaout import extract_json, validate

result = extract_json(llm_output)          # 模型爱加什么都行
check = validate(result.value, my_schema)  # 值对不对由 schema 说了算
if not check.ok:
    print(check.error_text())              # 带 $.path 的错误，可直接回灌给模型
```

## 技术选型：我考虑过但放弃的方案

**1. 直接用 `jsonschema` 库 → 放弃，自己写子集**

零依赖是这个仓库能被 clone 即跑的前提。更重要的是：只有自己实现一遍，
才知道"约束解码"里被约束的到底是什么，也才会注意到 `additionalProperties`、
`uniqueItems` 这些关键字在 Agent 场景里的实际作用。

**2. 用 Pydantic 做校验 → 放弃（本项目），但生产建议用**

Pydantic 的模型定义更简洁、生态更好。这里没用它的原因是：
（a）不想引入依赖；（b）我需要**精确控制"纠偏"的边界**。
Pydantic 的 `coerce` 很便利，但`"23" → 23` 这种静默纠偏在评测里会掩盖问题
（我在 samples.py 的注释里记了这个坑）。生产环境两者可以并存：
Pydantic 管结构，自己的评测层管指标。

**3. 让模拟器随机生成坏法 → 放弃，改成手工标注**

随机的坏法混在一起，你只知道总体成功率，不知道**哪一类坏法没被兜住**。
手工标注才能做出上面那张"按坏法分解"的表，而那才是有信息量的部分。

**4. 默认 `coerce=True` 里把整数转成字符串 → 放弃**

模型的常见手滑是把数字写成字符串（`"5"` 而不是 `5`），所以 `"5" → 5` 值得救。
反过来把 `5` 变成 `"5"` 是在**掩盖类型错误** —— 一个本该是字符串的字段收到整数，
说明模型理解错了，应该报错。`tests/test_schema.py::test_number_is_not_silently_stringified` 盯着这个。

## 已知局限（主动写出来）

1. **Schema 只支持常用子集**：没有 `$ref` / `$defs`、`patternProperties`、
   `dependentRequired`、`if-then-else`，`format` 只当注释不校验。
2. **模拟器的坏法分布是设计的，不是测出来的**。它适合做解析逻辑的回归测试，
   **不适合引用为"真实模型有 40% 概率输出坏 JSON"**。
3. **修复循环只做一次校验反馈**，没有做更聪明的策略（比如按错误类型选择不同的修复提示、
   或换一个更小的模型做修复）。3 轮是硬编码的。
4. **`grammar.py` 只校验语法**，不校验 schema。`{"age": "未知"}` 在它眼里是完全合法的。
   要把 schema 也压进采样（Outlines 那种做法）需要把 JSON Schema 编译成状态机，
   这是另一个量级的工作。
5. **解析器的"文本修复"可能改坏语义**：比如单引号→双引号转换在字符串内部本身含引号时会出错。
   代码里用"只在没有双引号时才转换"来规避，但这不是完备的解法。
6. **没有做并发**：20 条样本 × 6 策略是串行跑的。真实评测规模大了需要并发 + 缓存。

## 目录结构

```
schemaout/
├── extract.py      文本 → Python 对象（六条降级路径 + 文本级修复 + 截断补全）
├── schema.py       JSON Schema 子集校验器，错误带 $.path
├── strategies.py   六种策略（mode × parser × rounds 的三个正交开关）
├── bench.py        对照评测 + 按坏法分解 + 明细渲染
├── grammar.py      约束解码原理：前缀可行性判定 + token 掩码
├── samples.py      20 条样本 + 4 个 schema + 11 种坏法注入
└── llm.py          确定性模拟器 + 真实 OpenAI 兼容客户端
examples/           4 个可运行示例
tests/              132 个测试
docs/               原理讲解 / 代码导读 / 面试问答
```

## 延伸阅读

- [docs/01-结构化输出为什么难.md](docs/01-结构化输出为什么难.md) —— 11 类坏法与四种解法
- [docs/02-代码导读.md](docs/02-代码导读.md) —— 逐模块导读与关键取舍
- [docs/03-面试问答.md](docs/03-面试问答.md) —— 围绕本项目的 18 个面试问题与答法
- 姊妹项目：[mini-react-agent](https://github.com/flyingbeez/mini-react-agent)（Agent 循环）、
  [tool-forge](https://github.com/flyingbeez/tool-forge)（工具层）

## License

MIT
