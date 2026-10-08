"""Unit tests for recursive schema canonicalizer and provider payload normalizers."""

import json
import unittest
from agent_prompt_cache_bench.normalizer import (
    canonical_json_bytes,
    canonical_json_dumps,
    canonicalize_schema,
    normalize_provider_payload,
    normalize_tool_spec,
)


class TestSchemaCanonicalizer(unittest.TestCase):
    def test_basic_key_sorting(self):
        schema = {
            "type": "object",
            "description": "A test schema",
            "title": "Test",
        }
        canonical = canonicalize_schema(schema)
        keys = list(canonical.keys())
        self.assertEqual(keys, ["description", "title", "type"])

    def test_properties_and_nested_properties_sorting(self):
        schema = {
            "type": "object",
            "properties": {
                "zeta": {"type": "string", "description": "Last letter"},
                "alpha": {
                    "type": "object",
                    "properties": {
                        "y": {"type": "integer"},
                        "x": {"type": "integer"},
                    },
                },
                "beta": {"type": "boolean"},
            },
        }
        canonical = canonicalize_schema(schema)
        # Check top-level properties keys
        prop_keys = list(canonical["properties"].keys())
        self.assertEqual(prop_keys, ["alpha", "beta", "zeta"])
        # Check nested properties keys
        nested_alpha_keys = list(canonical["properties"]["alpha"]["properties"].keys())
        self.assertEqual(nested_alpha_keys, ["x", "y"])

    def test_defs_and_definitions_sorting(self):
        schema = {
            "type": "object",
            "$defs": {
                "User": {"properties": {"name": {"type": "string"}, "age": {"type": "integer"}}},
                "Account": {"properties": {"id": {"type": "string"}}},
            },
            "definitions": {
                "Role": {"type": "string"},
                "Admin": {"type": "object"},
            },
        }
        canonical = canonicalize_schema(schema)
        self.assertEqual(list(canonical["$defs"].keys()), ["Account", "User"])
        self.assertEqual(list(canonical["definitions"].keys()), ["Admin", "Role"])
        # Check inner User properties
        self.assertEqual(list(canonical["$defs"]["User"]["properties"].keys()), ["age", "name"])

    def test_required_array_sorting_and_deduplication(self):
        schema = {
            "type": "object",
            "required": ["zeta", "beta", "alpha", "beta"],
        }
        canonical = canonicalize_schema(schema)
        self.assertEqual(canonical["required"], ["alpha", "beta", "zeta"])

    def test_logical_composition_allof_anyof_oneof(self):
        # Two schemas with allOf branches in different order
        schema1 = {
            "allOf": [
                {"properties": {"b": {"type": "string"}}},
                {"properties": {"a": {"type": "string"}}},
            ]
        }
        schema2 = {
            "allOf": [
                {"properties": {"a": {"type": "string"}}},
                {"properties": {"b": {"type": "string"}}},
            ]
        }
        canon1 = canonicalize_schema(schema1)
        canon2 = canonicalize_schema(schema2)
        # Both canonicalized schemas should be byte identical
        self.assertEqual(canonical_json_dumps(canon1), canonical_json_dumps(canon2))

    def test_items_and_prefix_items(self):
        schema = {
            "type": "array",
            "items": {
                "properties": {
                    "z": {"type": "string"},
                    "a": {"type": "string"},
                }
            },
            "prefixItems": [
                {"properties": {"y": {"type": "number"}, "x": {"type": "number"}}},
                {"type": "string"},
            ],
        }
        canonical = canonicalize_schema(schema)
        self.assertEqual(list(canonical["items"]["properties"].keys()), ["a", "z"])
        self.assertEqual(list(canonical["prefixItems"][0]["properties"].keys()), ["x", "y"])

    def test_enum_order_preserved(self):
        # Enum order has domain semantics; ensure it is preserved
        schema = {
            "type": "string",
            "enum": ["east", "south", "west", "north"],
        }
        canonical = canonicalize_schema(schema)
        self.assertEqual(canonical["enum"], ["east", "south", "west", "north"])

    def test_idempotency(self):
        schema = {
            "properties": {
                "foo": {"type": "string"},
                "bar": {"type": "number"},
            },
            "required": ["foo", "bar"],
        }
        once = canonicalize_schema(schema)
        twice = canonicalize_schema(once)
        self.assertEqual(canonical_json_dumps(once), canonical_json_dumps(twice))


class TestToolSpecNormalization(unittest.TestCase):
    def test_anthropic_tool_normalization(self):
        tool = {
            "description": "Searches files",
            "name": "search",
            "input_schema": {
                "required": ["query"],
                "properties": {
                    "path": {"type": "string"},
                    "query": {"type": "string"},
                },
                "type": "object",
            },
            "cache_control": {"type": "ephemeral"},
        }
        canon = normalize_tool_spec(tool, provider="anthropic")
        self.assertEqual(list(canon.keys()), ["cache_control", "description", "input_schema", "name"])
        self.assertEqual(list(canon["input_schema"]["properties"].keys()), ["path", "query"])

    def test_openai_tool_normalization(self):
        tool = {
            "type": "function",
            "function": {
                "description": "Calculates math expression",
                "name": "calculator",
                "parameters": {
                    "properties": {
                        "expr": {"type": "string"},
                        "base": {"type": "integer"},
                    },
                    "required": ["expr"],
                    "type": "object",
                },
            },
        }
        canon = normalize_tool_spec(tool, provider="openai")
        self.assertEqual(list(canon["function"].keys()), ["description", "name", "parameters"])
        self.assertEqual(list(canon["function"]["parameters"]["properties"].keys()), ["base", "expr"])

    def test_gemini_tool_normalization(self):
        tool = {
            "function_declarations": [
                {
                    "description": "Run shell",
                    "parameters": {"properties": {"cmd": {"type": "string"}}},
                    "name": "bash",
                },
                {
                    "description": "Read file",
                    "parameters": {"properties": {"file": {"type": "string"}}},
                    "name": "cat",
                },
            ]
        }
        canon = normalize_tool_spec(tool, provider="gemini")
        decls = canon["function_declarations"]
        # Should be sorted by name: 'bash' then 'cat'
        self.assertEqual(decls[0]["name"], "bash")
        self.assertEqual(decls[1]["name"], "cat")


class TestProviderPayloadNormalization(unittest.TestCase):
    def test_tool_reordering_produces_identical_payload(self):
        # Tool A and Tool B in different array orders
        tool_a = {
            "name": "alpha",
            "description": "First tool",
            "input_schema": {"type": "object", "properties": {"x": {"type": "string"}}},
        }
        tool_b = {
            "name": "beta",
            "description": "Second tool",
            "input_schema": {"type": "object", "properties": {"y": {"type": "string"}}},
        }

        payload1 = {
            "model": "claude-3-7-sonnet",
            "tools": [tool_b, tool_a],
            "messages": [{"role": "user", "content": "hello"}],
        }
        payload2 = {
            "messages": [{"role": "user", "content": "hello"}],
            "model": "claude-3-7-sonnet",
            "tools": [tool_a, tool_b],
        }

        canon1 = normalize_provider_payload(payload1, provider="anthropic")
        canon2 = normalize_provider_payload(payload2, provider="anthropic")

        # Serialized canonical JSON bytes should be 100% byte identical
        self.assertEqual(canonical_json_bytes(canon1), canonical_json_bytes(canon2))

    def test_chronological_messages_order_preserved(self):
        payload = {
            "messages": [
                {"role": "user", "content": "turn 1"},
                {"role": "assistant", "content": "reply 1"},
                {"role": "user", "content": "turn 2"},
            ]
        }
        canon = normalize_provider_payload(payload)
        contents = [m["content"] for m in canon["messages"]]
        self.assertEqual(contents, ["turn 1", "reply 1", "turn 2"])


if __name__ == "__main__":
    unittest.main()
