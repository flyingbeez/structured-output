"""Schema 校验器测试。"""

from __future__ import annotations

import unittest

from schemaout.schema import SchemaError, is_valid, schema_is_sane, validate


class TestTypes(unittest.TestCase):
    def test_each_type(self):
        cases = [
            ({"type": "string"}, "x", True),
            ({"type": "string"}, 1, False),
            ({"type": "integer"}, 3, True),
            ({"type": "integer"}, 3.0, True),
            ({"type": "integer"}, 3.5, False),
            ({"type": "integer"}, True, False),
            ({"type": "number"}, 3, True),
            ({"type": "number"}, True, False),
            ({"type": "boolean"}, True, True),
            ({"type": "boolean"}, 1, False),
            ({"type": "null"}, None, True),
            ({"type": "array"}, [], True),
            ({"type": "object"}, {}, True),
        ]
        for schema, value, expected in cases:
            with self.subTest(schema=schema, value=value):
                self.assertEqual(validate(value, schema).ok, expected)

    def test_type_union(self):
        schema = {"type": ["string", "null"]}
        self.assertTrue(validate(None, schema).ok)
        self.assertTrue(validate("x", schema).ok)
        self.assertFalse(validate(1, schema).ok)

    def test_error_message_mentions_actual_type(self):
        result = validate("x", {"type": "integer"})
        self.assertFalse(result.ok)
        self.assertIn("string", str(result.errors[0]))

    def test_bool_is_not_integer(self):
        result = validate(True, {"type": "integer"})
        self.assertFalse(result.ok)


class TestObject(unittest.TestCase):
    SCHEMA = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "name": {"type": "string", "minLength": 1},
            "age": {"type": "integer", "minimum": 0, "maximum": 150},
        },
        "required": ["name", "age"],
    }

    def test_ok(self):
        self.assertTrue(validate({"name": "张三", "age": 30}, self.SCHEMA).ok)

    def test_missing_required(self):
        result = validate({"name": "张三"}, self.SCHEMA)
        self.assertFalse(result.ok)
        self.assertIn("$.age", str(result.errors[0]))
        self.assertIn("缺少必填字段", result.error_text())

    def test_additional_properties_false(self):
        result = validate({"name": "a", "age": 1, "extra": 1}, self.SCHEMA)
        self.assertFalse(result.ok)
        self.assertIn("$.extra", result.error_text())

    def test_additional_properties_true_keeps_extra(self):
        schema = {"type": "object", "properties": {"a": {"type": "integer"}}}
        result = validate({"a": 1, "b": 2}, schema)
        self.assertTrue(result.ok)
        self.assertEqual(result.value["b"], 2)

    def test_nested_path(self):
        schema = {"type": "object", "properties": {
            "profile": {"type": "object", "properties": {"age": {"type": "integer"}},
                        "required": ["age"]}}}
        result = validate({"profile": {}}, schema)
        self.assertIn("$.profile.age", result.error_text())

    def test_array_index_path(self):
        schema = {"type": "object", "properties": {
            "nums": {"type": "array", "items": {"type": "integer"}}}}
        result = validate({"nums": [1, "x", 3]}, schema)
        self.assertIn("$.nums[1]", result.error_text())

    def test_default_filled(self):
        schema = {"type": "object", "properties": {
            "mode": {"type": "string", "default": "safe"}}}
        result = validate({}, schema)
        self.assertTrue(result.ok)
        self.assertEqual(result.value["mode"], "safe")

    def test_default_does_not_override(self):
        schema = {"type": "object", "properties": {
            "mode": {"type": "string", "default": "safe"}}}
        self.assertEqual(validate({"mode": "fast"}, schema).value["mode"], "fast")


class TestScalarConstraints(unittest.TestCase):
    def test_numeric_range(self):
        schema = {"type": "integer", "minimum": 1, "maximum": 10}
        self.assertTrue(validate(1, schema).ok)
        self.assertFalse(validate(0, schema).ok)
        self.assertFalse(validate(11, schema).ok)

    def test_exclusive(self):
        schema = {"type": "number", "exclusiveMinimum": 0}
        self.assertFalse(validate(0, schema).ok)
        self.assertTrue(validate(0.01, schema).ok)

    def test_multiple_of(self):
        schema = {"type": "number", "multipleOf": 0.5}
        self.assertTrue(validate(1.5, schema).ok)
        self.assertFalse(validate(1.3, schema).ok)

    def test_string_length(self):
        schema = {"type": "string", "minLength": 2, "maxLength": 4}
        self.assertTrue(validate("abc", schema).ok)
        self.assertFalse(validate("a", schema).ok)
        self.assertFalse(validate("abcde", schema).ok)

    def test_pattern(self):
        schema = {"type": "string", "pattern": r"^ORD-\d{4}$"}
        self.assertTrue(validate("ORD-1234", schema).ok)
        self.assertFalse(validate("1234", schema).ok)

    def test_bad_pattern_reported_not_crashed(self):
        result = validate("x", {"type": "string", "pattern": "("})
        self.assertFalse(result.ok)
        self.assertIn("pattern", result.errors[0].message)

    def test_enum(self):
        schema = {"type": "string", "enum": ["a", "b"]}
        self.assertTrue(validate("a", schema).ok)
        self.assertFalse(validate("c", schema).ok)

    def test_const(self):
        self.assertTrue(validate(5, {"const": 5}).ok)
        self.assertFalse(validate(6, {"const": 5}).ok)


class TestArray(unittest.TestCase):
    def test_items(self):
        schema = {"type": "array", "items": {"type": "integer"}}
        self.assertTrue(validate([1, 2], schema).ok)
        self.assertFalse(validate([1, "x"], schema).ok)

    def test_length_bounds(self):
        schema = {"type": "array", "minItems": 1, "maxItems": 3}
        self.assertFalse(validate([], schema).ok)
        self.assertFalse(validate([1, 2, 3, 4], schema).ok)

    def test_unique_items(self):
        schema = {"type": "array", "uniqueItems": True}
        self.assertTrue(validate([1, 2], schema).ok)
        result = validate([1, 1], schema)
        self.assertFalse(result.ok)
        self.assertIn("唯一", result.error_text())


class TestCombinators(unittest.TestCase):
    def test_any_of(self):
        schema = {"anyOf": [{"type": "string"}, {"type": "integer"}]}
        self.assertTrue(validate("x", schema).ok)
        self.assertTrue(validate(1, schema).ok)
        self.assertFalse(validate(1.5, schema).ok)

    def test_one_of_rejects_matching_both(self):
        schema = {"oneOf": [{"type": "integer"}, {"type": "number"}]}
        result = validate(3, schema)
        self.assertFalse(result.ok)
        self.assertIn("恰好匹配", result.error_text())

    def test_all_of(self):
        schema = {"allOf": [{"type": "integer"}, {"minimum": 5}]}
        self.assertTrue(validate(5, schema).ok)
        self.assertFalse(validate(4, schema).ok)

    def test_false_schema(self):
        self.assertFalse(validate(1, False).ok)

    def test_true_schema_accepts_anything(self):
        self.assertTrue(validate(object(), True).ok)


class TestCoercion(unittest.TestCase):
    def test_numeric_string_is_coerced_by_default(self):
        self.assertTrue(validate("3", {"type": "integer"}).ok)

    def test_coerce_can_be_disabled(self):
        self.assertFalse(validate("3", {"type": "integer"}, coerce=False).ok)

    def test_non_coercible_string_fails(self):
        result = validate("未知", {"type": "integer"})
        self.assertFalse(result.ok, "无法纠偏的字符串必须报错，不能静默通过")

    def test_bool_string(self):
        self.assertTrue(validate("true", {"type": "boolean"}).ok)

    def test_number_is_not_silently_stringified(self):
        # 反向保护：一个本该是字符串的字段收到整数，必须报错而不是被转成 "1"
        self.assertFalse(validate(1, {"type": "string"}).ok)

    def test_integral_float_is_coerced(self):
        result = validate(3.0, {"type": "integer"})
        self.assertTrue(result.ok)
        self.assertEqual(result.value, 3)


class TestHelpers(unittest.TestCase):
    def test_is_valid(self):
        self.assertTrue(is_valid({"a": 1}, {"type": "object"}))
        self.assertFalse(is_valid(1, {"type": "object"}))

    def test_error_to_dict(self):
        result = validate(1, {"type": "string"})
        payload = result.errors[0].to_dict()
        self.assertEqual(payload["path"], "$")
        self.assertEqual(payload["keyword"], "type")

    def test_schema_is_sane_flags_mistakes(self):
        bad = {
            "type": "object",
            "properties": {"a": {"type": "integer", "minimum": 10, "maximum": 1}},
            "required": ["b"],
        }
        problems = schema_is_sane(bad)
        self.assertTrue(any("minimum > maximum" in p for p in problems))
        self.assertTrue(any("required" in p for p in problems))

    def test_schema_is_sane_ok(self):
        good = {"type": "object", "additionalProperties": False,
                "properties": {"a": {"type": "integer"}}, "required": ["a"]}
        self.assertEqual(schema_is_sane(good), [])

    def test_schema_is_sane_empty_enum(self):
        problems = schema_is_sane({"type": "string", "enum": []})
        self.assertTrue(any("enum 为空" in p for p in problems))

    def test_error_is_schema_error(self):
        self.assertIsInstance(validate(1, {"type": "string"}).errors[0], SchemaError)

    def test_recursion_depth_guard(self):
        schema: dict = {"type": "object"}
        cursor = schema
        for _ in range(200):
            cursor["properties"] = {"x": {"type": "object"}}
            cursor = cursor["properties"]["x"]
        result = validate({}, schema)
        self.assertIsInstance(result.ok, bool)


if __name__ == "__main__":
    unittest.main()
