"""策略与评测测试 —— 本仓库的核心断言。

重点不是"某个策略成功率是多少"（那会随样本改动），而是**策略之间的相对关系**：
客户端解析层带来最大跃升；json_mode 与强解析层是替代关系；
反馈修复能治语义错误但治不了截断。
"""

from __future__ import annotations

import json
import unittest

from schemaout.bench import benchmark, render_defect_breakdown, render_table, summary_dict
from schemaout.llm import Completion, SimulatedLLM
from schemaout.samples import (
    SAMPLES,
    STRUCTURAL_DEFECTS,
    UNFIXABLE_DEFECTS,
    apply_defect,
    defect_histogram,
)
from schemaout.strategies import STRATEGIES, get_strategy, run, run_many

SAMPLE_BY_ID = {s.id: s for s in SAMPLES}


def fresh_llm() -> SimulatedLLM:
    return SimulatedLLM(SAMPLES)


class TestSampleSet(unittest.TestCase):
    def test_twenty_samples(self):
        self.assertEqual(len(SAMPLES), 20)

    def test_ids_unique(self):
        self.assertEqual(len({s.id for s in SAMPLES}), 20)

    def test_defect_mix(self):
        hist = defect_histogram()
        self.assertEqual(hist["clean"], 8)
        self.assertEqual(sum(hist.get(d, 0) for d in STRUCTURAL_DEFECTS), 8)
        self.assertEqual(hist.get("wrong_type", 0) + hist.get("bad_enum", 0)
                         + hist.get("extra_field", 0), 3)
        self.assertEqual(sum(hist.get(d, 0) for d in UNFIXABLE_DEFECTS), 1)

    def test_clean_sample_parses_and_validates(self):
        sample = SAMPLE_BY_ID["p1"]
        self.assertEqual(json.loads(apply_defect(sample)), sample.gold)

    def test_every_defect_changes_the_text(self):
        for sample in SAMPLES:
            if sample.defect == "clean":
                continue
            with self.subTest(sample=sample.id):
                self.assertNotEqual(apply_defect(sample), sample.clean_text)

    def test_truncated_output_is_shorter(self):
        sample = SAMPLE_BY_ID["o5"]
        self.assertLess(len(apply_defect(sample)), len(sample.clean_text))


class TestSimulatedLLM(unittest.TestCase):
    def test_deterministic(self):
        a = fresh_llm().complete(system="s", user="u", mode="text", meta={"sample_id": "p2"})
        b = fresh_llm().complete(system="s", user="u", mode="text", meta={"sample_id": "p2"})
        self.assertEqual(a.text, b.text)

    def test_text_mode_keeps_structural_defect(self):
        completion = fresh_llm().complete(system="s", user="u", mode="text",
                                          meta={"sample_id": "p2"})
        self.assertTrue(completion.text.startswith("```"), "text 模式应保留围栏")

    def test_json_mode_fixes_structural_defect(self):
        completion = fresh_llm().complete(system="s", user="u", mode="json",
                                          meta={"sample_id": "p2"})
        self.assertFalse(completion.text.startswith("```"))
        json.loads(completion.text)

    def test_json_mode_keeps_semantic_defect(self):
        completion = fresh_llm().complete(system="s", user="u", mode="json",
                                          meta={"sample_id": "o4"})
        self.assertIn("RMB", completion.text, "格式约束治不了语义错误")

    def test_tool_mode_fixes_extra_field(self):
        completion = fresh_llm().complete(system="s", user="u", mode="tool",
                                          meta={"sample_id": "p5"})
        self.assertNotIn("note", completion.text)

    def test_repair_round_fixes_everything_but_truncation(self):
        for sample_id in ("p2", "o4", "p5"):
            with self.subTest(sample=sample_id):
                completion = fresh_llm().complete(system="s", user="u", mode="text",
                                                  meta={"sample_id": sample_id, "round": 2})
                self.assertEqual(json.loads(completion.text), SAMPLE_BY_ID[sample_id].gold)
        truncated = fresh_llm().complete(system="s", user="u", mode="text",
                                        meta={"sample_id": "o5", "round": 2})
        self.assertNotEqual(truncated.text, SAMPLE_BY_ID["o5"].clean_text)

    def test_unknown_sample_raises(self):
        with self.assertRaises(Exception):
            fresh_llm().complete(system="s", user="u", meta={"sample_id": "zzz"})

    def test_records_calls(self):
        llm = fresh_llm()
        llm.complete(system="s", user="u", meta={"sample_id": "p1"})
        self.assertEqual(len(llm.calls), 1)

    def test_token_estimate_is_positive(self):
        completion = fresh_llm().complete(system="system", user="用户提问",
                                          meta={"sample_id": "p1"})
        self.assertGreater(completion.prompt_tokens, 0)
        self.assertGreater(completion.completion_tokens, 0)


class TestSingleRuns(unittest.TestCase):
    def test_clean_sample_passes_everywhere(self):
        for strategy in STRATEGIES:
            with self.subTest(strategy=strategy.key):
                result = run(strategy, fresh_llm(), SAMPLE_BY_ID["p1"])
                self.assertTrue(result.ok)
                self.assertEqual(result.attempts, 1)

    def test_naive_fails_on_fenced(self):
        result = run(get_strategy("naive"), fresh_llm(), SAMPLE_BY_ID["p2"])
        self.assertFalse(result.ok)
        self.assertIn("语法错误", result.parse_error)

    def test_naive_fails_on_semantic(self):
        result = run(get_strategy("naive"), fresh_llm(), SAMPLE_BY_ID["o4"])
        self.assertFalse(result.ok)
        self.assertTrue(result.schema_errors)

    def test_extract_fixes_fenced(self):
        result = run(get_strategy("extract"), fresh_llm(), SAMPLE_BY_ID["p2"])
        self.assertTrue(result.ok)

    def test_extract_cannot_fix_semantic(self):
        result = run(get_strategy("extract"), fresh_llm(), SAMPLE_BY_ID["o4"])
        self.assertFalse(result.ok)
        self.assertTrue(any("enum" in e or "取值" in e for e in result.schema_errors))

    def test_repair_loop_uses_extra_rounds(self):
        result = run(get_strategy("repair_loop"), fresh_llm(), SAMPLE_BY_ID["o4"])
        self.assertTrue(result.ok)
        self.assertEqual(result.attempts, 2)

    def test_repair_loop_gives_up_on_truncation(self):
        result = run(get_strategy("repair_loop"), fresh_llm(), SAMPLE_BY_ID["o5"])
        self.assertFalse(result.ok)
        self.assertEqual(result.attempts, 3)

    def test_truncation_defeats_every_strategy(self):
        for strategy in STRATEGIES:
            with self.subTest(strategy=strategy.key):
                result = run(strategy, fresh_llm(), SAMPLE_BY_ID["o5"])
                self.assertFalse(result.ok, "截断是 max_tokens 问题，格式手段都救不了")

    def test_run_many(self):
        results = run_many(get_strategy("naive"), fresh_llm(), SAMPLES)
        self.assertEqual(len(results), 20)

    def test_result_to_dict(self):
        payload = run(get_strategy("naive"), fresh_llm(), SAMPLE_BY_ID["p1"]).to_dict()
        self.assertTrue(payload["ok"])
        self.assertIn("tokens", payload)

    def test_unknown_strategy(self):
        with self.assertRaises(KeyError):
            get_strategy("nope")


class TestBenchmarkRelations(unittest.TestCase):
    """核心断言：策略之间的相对关系要比绝对数字稳定得多。"""

    @classmethod
    def setUpClass(cls):
        cls.reports = {r.strategy.key: r for r in benchmark(fresh_llm(), SAMPLES)}

    def test_baseline_is_worst(self):
        self.assertLess(self.reports["naive"].success_rate, 0.6)

    def test_parser_is_the_biggest_single_win(self):
        naive = self.reports["naive"].success_rate
        extract = self.reports["extract"].success_rate
        self.assertGreaterEqual(extract - naive, 0.3,
                                "客户端解析层的收益应该是最显著的")
        self.assertEqual(self.reports["extract"].mean_attempts, 1.0,
                         "解析层不需要额外请求，尝试次数不应增加")

    def test_json_mode_and_extractor_are_substitutes(self):
        self.assertAlmostEqual(self.reports["json_mode"].success_rate,
                               self.reports["extract"].success_rate, places=6)
        self.assertAlmostEqual(self.reports["json_extract"].success_rate,
                               self.reports["extract"].success_rate, places=6)

    def test_tool_call_beats_plain_extract(self):
        self.assertGreater(self.reports["tool_call"].success_rate,
                           self.reports["extract"].success_rate)

    def test_repair_loop_is_best(self):
        best = max(r.success_rate for r in self.reports.values())
        self.assertEqual(self.reports["repair_loop"].success_rate, best)
        self.assertGreaterEqual(self.reports["repair_loop"].success_rate, 0.9)

    def test_repair_loop_costs_more(self):
        self.assertGreater(self.reports["repair_loop"].mean_attempts, 1.0)
        self.assertGreater(self.reports["repair_loop"].tokens_per_sample,
                           self.reports["naive"].tokens_per_sample)

    def test_no_strategy_is_perfect(self):
        self.assertLess(self.reports["repair_loop"].success_rate, 1.0,
                        "总有一条修不了的样本，这才是诚实的结论")

    def test_breakdown_covers_structural_and_semantic(self):
        bucket = self.reports["extract"].by_defect(SAMPLES)
        self.assertEqual(bucket["fenced"][0], bucket["fenced"][1])
        self.assertEqual(bucket["bad_enum"][0], 0)

    def test_render_table_contains_all_strategies(self):
        table = render_table(list(self.reports.values()))
        for key in self.reports:
            self.assertIn(f"`{key}`", table)

    def test_render_breakdown_contains_defect_labels(self):
        text = render_defect_breakdown(list(self.reports.values()), SAMPLES)
        self.assertIn("被截断", text)
        self.assertIn("结构性", text)

    def test_summary_serialisable(self):
        payload = summary_dict(list(self.reports.values()))
        json.dumps(payload)
        self.assertEqual(len(payload), len(STRATEGIES))

    def test_benchmark_is_deterministic(self):
        again = {r.strategy.key: r for r in benchmark(fresh_llm(), SAMPLES)}
        for key in self.reports:
            self.assertEqual(self.reports[key].passed, again[key].passed, key)


if __name__ == "__main__":
    unittest.main()
