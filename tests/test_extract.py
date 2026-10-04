"""解析层测试：模型输出的十二种坏法都要能兜住。"""

from __future__ import annotations

import json
import unittest

from schemaout.extract import (
    ExtractionResult,
    close_truncated,
    extract_all_json,
    extract_json,
    repair_text,
    strip_fences,
)


class TestStripFences(unittest.TestCase):
    def test_json_fence(self):
        text, changed = strip_fences('```json\n{"a": 1}\n```')
        self.assertTrue(changed)
        self.assertEqual(text, '{"a": 1}')

    def test_plain_fence(self):
        text, changed = strip_fences('```\n{"a": 1}\n```')
        self.assertTrue(changed)
        self.assertEqual(text, '{"a": 1}')

    def test_unclosed_fence(self):
        text, changed = strip_fences('```json\n{"a": 1}')
        self.assertTrue(changed)
        self.assertIn('{"a": 1}', text)

    def test_no_fence(self):
        text, changed = strip_fences('{"a": 1}')
        self.assertFalse(changed)
        self.assertEqual(text, '{"a": 1}')


class TestRepairText(unittest.TestCase):
    def test_fullwidth_punctuation(self):
        fixed, notes = repair_text('{“a”：1}')
        self.assertIn('"a"', fixed)
        self.assertIn("全角标点", " ".join(notes))

    def test_python_literals(self):
        fixed, notes = repair_text('{"a": True, "b": False, "c": None}')
        self.assertEqual(json.loads(fixed), {"a": True, "b": False, "c": None})
        self.assertIn("Python 字面量→JSON", notes)

    def test_single_quotes(self):
        fixed, notes = repair_text("{'a': 'b'}")
        self.assertEqual(json.loads(fixed), {"a": "b"})
        self.assertIn("单引号→双引号", notes)

    def test_bare_keys(self):
        fixed, notes = repair_text('{a: 1, b: 2}')
        self.assertEqual(json.loads(fixed), {"a": 1, "b": 2})
        self.assertIn("裸键名加引号", notes)

    def test_trailing_comma(self):
        fixed, notes = repair_text('{"a": 1,}')
        self.assertEqual(json.loads(fixed), {"a": 1})
        self.assertEqual(json.loads(repair_text('{"a": [1, 2,]}')[0]), {"a": [1, 2]})

    def test_comments_removed(self):
        fixed, _ = repair_text('{"a": 1} // 备注')
        self.assertEqual(json.loads(fixed), {"a": 1})

    def test_clean_text_untouched(self):
        fixed, notes = repair_text('{"a": 1}')
        self.assertEqual(fixed, '{"a": 1}')
        self.assertEqual(notes, [])


class TestCloseTruncated(unittest.TestCase):
    def test_missing_brace(self):
        fixed, changed = close_truncated('{"a": 1, "b": {"c": 2}')
        self.assertTrue(changed)
        self.assertTrue(fixed.endswith("}}"))

    def test_unclosed_string(self):
        fixed, changed = close_truncated('{"a": "hello')
        self.assertTrue(changed)
        self.assertTrue(json.loads(fixed)["a"].startswith("hello"))

    def test_dangling_comma_removed(self):
        fixed, changed = close_truncated('{"a": 1,')
        self.assertTrue(changed)
        self.assertEqual(json.loads(fixed), {"a": 1})

    def test_complete_untouched(self):
        fixed, changed = close_truncated('{"a": 1}')
        self.assertFalse(changed)
        self.assertEqual(fixed, '{"a": 1}')


class TestExtractJson(unittest.TestCase):
    def test_direct(self):
        result = extract_json('{"a": 1}')
        self.assertTrue(result.ok)
        self.assertEqual(result.value, {"a": 1})
        self.assertEqual(result.method, "direct")

    def test_fenced(self):
        result = extract_json('```json\n{"a": 1}\n```')
        self.assertTrue(result.ok)
        self.assertEqual(result.value, {"a": 1})
        self.assertIn(result.method, ("strip_fence", "balanced_span"))

    def test_prose_wrapper(self):
        result = extract_json('好的，结果如下：\n{"a": 1}\n希望对你有帮助！')
        self.assertTrue(result.ok)
        self.assertEqual(result.value, {"a": 1})

    def test_single_quotes(self):
        result = extract_json("{'a': 'b'}")
        self.assertTrue(result.ok)
        self.assertEqual(result.value, {"a": "b"})

    def test_trailing_comma(self):
        result = extract_json('{"a": [1, 2,], "b": 3,}')
        self.assertTrue(result.ok)
        self.assertEqual(result.value, {"a": [1, 2], "b": 3})

    def test_bare_keys(self):
        result = extract_json('{a: 1, b: "x"}')
        self.assertTrue(result.ok)
        self.assertEqual(result.value, {"a": 1, "b": "x"})

    def test_python_literals(self):
        result = extract_json('{"a": True, "b": None}')
        self.assertTrue(result.ok)
        self.assertEqual(result.value, {"a": True, "b": None})

    def test_array_root(self):
        result = extract_json("[1, 2, 3]")
        self.assertTrue(result.ok)
        self.assertEqual(result.value, [1, 2, 3])

    def test_index_before_json(self):
        result = extract_json('第 3 步的结果是 {"a": 1}，请继续。')
        self.assertTrue(result.ok)
        self.assertEqual(result.value, {"a": 1})

    def test_brace_inside_string_not_confused(self):
        # 字符串里的花括号不能被当成结构括号
        result = extract_json('{"a": "}{", "b": 2}')
        self.assertTrue(result.ok)
        self.assertEqual(result.value, {"a": "}{", "b": 2})

    def test_dict_passthrough(self):
        result = extract_json({"a": 1})
        self.assertTrue(result.ok)
        self.assertEqual(result.method, "already_parsed")

    def test_empty_and_none(self):
        self.assertFalse(extract_json("").ok)
        self.assertFalse(extract_json(None).ok)
        self.assertFalse(extract_json("   ").ok)

    def test_garbage(self):
        result = extract_json("我觉得这个问题很难，容我想想。")
        self.assertFalse(result.ok)
        self.assertTrue(result.error)

    def test_truncated_object(self):
        result = extract_json('{"name": "张三", "age": 28, "city": "杭')
        self.assertTrue(result.ok)
        self.assertEqual(result.value["name"], "张三")

    def test_result_is_dataclass(self):
        self.assertIsInstance(extract_json('{"a":1}'), ExtractionResult)

    def test_stats_are_serialisable(self):
        payload = extract_json('{"a":1}').to_dict()
        self.assertTrue(payload["ok"])
        self.assertIn("method", payload)


class TestExtractAll(unittest.TestCase):
    def test_multiple_values(self):
        values = extract_all_json('{"a": 1} 以及 [2, 3]')
        self.assertIn({"a": 1}, values)
        self.assertIn([2, 3], values)

    def test_empty(self):
        self.assertEqual(extract_all_json("没有 JSON"), [])


if __name__ == "__main__":
    unittest.main()
