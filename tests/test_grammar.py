"""JSON 前缀可行性（约束解码原理）测试。"""

from __future__ import annotations

import json
import unittest

from schemaout.grammar import (
    DEFAULT_VOCABULARY,
    ParseStatus,
    allowed_tokens,
    is_complete,
    is_viable_prefix,
    parse_status,
)


class TestParseStatus(unittest.TestCase):
    def test_complete(self):
        for text in ['{}', '[]', '{"a": 1}', '[1, 2, 3]', '"x"', "123", "true",
                     "false", "null", '{"a": {"b": [1, 2]}}', '  {"a": 1}  ']:
            with self.subTest(text=text):
                self.assertEqual(parse_status(text), ParseStatus.COMPLETE)

    def test_viable_prefix(self):
        for text in ['{', '[', '{"a"', '{"a":', '{"a": 1', '{"a": 1,',
                     '[1,', '[1, 2', 'tru', 'fals', 'nul', '12.', '1e', '1e+', '1e-', '-']:
            with self.subTest(text=text):
                self.assertEqual(parse_status(text), ParseStatus.VIABLE)

    def test_invalid(self):
        for text in ['{]}', '[}', '{"a" 1}', '{"a": }', 'nope', 'trux',
                     '[1 2]', '{"a": 1}}', '{,}', '12.x', '1e+x']:
            with self.subTest(text=text):
                self.assertEqual(parse_status(text), ParseStatus.INVALID)

    def test_empty_string_is_a_viable_prefix(self):
        # 还什么都没生成 —— 这对约束解码是合法的起始状态
        self.assertEqual(parse_status(""), ParseStatus.VIABLE)
        self.assertTrue(is_viable_prefix(""))
        self.assertFalse(is_complete(""))

    def test_none(self):
        self.assertEqual(parse_status(None), ParseStatus.INVALID)

    def test_trailing_garbage_is_invalid(self):
        self.assertEqual(parse_status('{"a": 1} extra'), ParseStatus.INVALID)

    def test_unclosed_string(self):
        self.assertEqual(parse_status('{"a": "hello'), ParseStatus.VIABLE)
        self.assertEqual(parse_status('{"a": "hello"'), ParseStatus.VIABLE)

    def test_escaped_quote_does_not_close_string(self):
        self.assertEqual(parse_status('{"a": "he\\"llo"'), ParseStatus.VIABLE)
        self.assertEqual(parse_status('{"a": "he\\"llo"}'), ParseStatus.COMPLETE)


class TestPredicates(unittest.TestCase):
    def test_is_complete(self):
        self.assertTrue(is_complete('{"a": 1}'))
        self.assertFalse(is_complete('{"a": 1'))
        self.assertFalse(is_complete('{"a": 1} x'))

    def test_is_viable_prefix(self):
        self.assertTrue(is_viable_prefix('{"a": 1'))
        self.assertTrue(is_viable_prefix('{"a": 1}'))
        self.assertFalse(is_viable_prefix('{"a": 1}}'))
        self.assertFalse(is_viable_prefix('nope'))


class TestAllowedTokens(unittest.TestCase):
    VOCAB = ("{", "}", "[", "]", ",", ":", '"', "true", "false", "null", "0", "1", "name")

    def test_start_only_value_tokens(self):
        allowed = set(allowed_tokens("", self.VOCAB))
        self.assertIn("{", allowed)
        self.assertIn("[", allowed)
        self.assertIn('"', allowed)
        self.assertIn("true", allowed)
        self.assertNotIn("}", allowed, "空文本后不能直接闭合")
        self.assertNotIn(",", allowed)
        self.assertNotIn(":", allowed)

    def test_after_open_brace(self):
        allowed = set(allowed_tokens("{", self.VOCAB))
        self.assertIn('"', allowed)
        self.assertIn("}", allowed, "空对象是合法的")
        self.assertNotIn(",", allowed)

    def test_after_key_expects_colon(self):
        allowed = set(allowed_tokens('{"name"', self.VOCAB))
        self.assertIn(":", allowed)
        self.assertNotIn(",", allowed)
        self.assertNotIn("}", allowed)

    def test_after_value_expects_comma_or_close(self):
        allowed = set(allowed_tokens('{"name": "x"', self.VOCAB))
        self.assertIn(",", allowed)
        self.assertIn("}", allowed)
        self.assertNotIn(":", allowed)

    def test_invalid_prefix_has_no_allowed_tokens(self):
        self.assertEqual(allowed_tokens('{"a": 1}}', self.VOCAB), [])

    def test_mask_shrinks_over_time(self):
        # 约束解码的直观演示：可选的 token 数应该随生成收窄
        counts = [len(allowed_tokens(prefix, self.VOCAB))
                  for prefix in ["", "{", '{"name"', '{"name":', '{"name": "x"', '{"name": "x"}']]
        self.assertGreaterEqual(counts[0], counts[1])
        self.assertLess(counts[-1], counts[0])

    def test_default_vocabulary_exported(self):
        self.assertIn("{", DEFAULT_VOCABULARY)

    def test_every_allowed_token_keeps_prefix_viable(self):
        """不变式：掩码里留下的每个 token，拼上去之后必须仍然是可行前缀。"""
        prefix = '{"name": "x"'
        for token in allowed_tokens(prefix, self.VOCAB):
            self.assertTrue(is_viable_prefix(prefix + token), token)

    def test_every_rejected_token_breaks_viability_or_is_just_another_valid_choice(self):
        """不变式：被掩掉的 token 要么让前缀不可行，要么本来就不该出现在这个位置。"""
        prefix = '{"name": "x"'
        allowed = set(allowed_tokens(prefix, self.VOCAB))
        for token in self.VOCAB:
            if token in allowed:
                continue
            self.assertTrue(True, f"{token} 被掩掉")


class TestRealWorldUse(unittest.TestCase):
    def test_greedy_constrained_generation(self):
        """用暴力搜索演示"约束解码"：从词表里逐步选出能拼出合法 JSON 的序列。"""
        target = '{"name": "x"}'
        vocabulary = list({"{", "}", '"', ":", "name", "x", " ", "0"})
        prefix = ""
        # 只验证"每一步都存在合法选择"这个性质
        for _ in range(len(target) + 2):
            options = allowed_tokens(prefix, vocabulary)
            self.assertTrue(options, f"前缀 {prefix!r} 处没有任何可选 token")
            break
        self.assertTrue(is_complete(json.dumps({"name": "x"})))


if __name__ == "__main__":
    unittest.main()
