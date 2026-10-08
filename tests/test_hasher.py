"""Unit tests for wire-level SHA-256 calculation and drift detector."""

import random
import unittest
from agent_prompt_cache_bench.bench import permute_dict_keys
from agent_prompt_cache_bench.hasher import (
    compute_sha256,
    detect_drift,
    hash_payload,
    raw_json_bytes,
)


class TestWireHasherAndDrift(unittest.TestCase):
    def test_sha256_computation(self):
        h = compute_sha256(b"hello world")
        self.assertEqual(
            h,
            "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9",
        )

    def test_key_permutation_reproduces_cache_drift(self):
        # Two semantically identical payloads with different key orders
        payload1 = {
            "model": "claude-3-7-sonnet",
            "tools": [
                {
                    "name": "calc",
                    "description": "Math",
                    "input_schema": {
                        "type": "object",
                        "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
                        "required": ["a", "b"],
                    },
                }
            ],
        }

        # Create permuted version
        rng = random.Random(123)
        payload2 = permute_dict_keys(payload1, rng)

        # Raw hashes should differ (reproducing the cache miss bug)
        raw_hash1 = hash_payload(payload1, canonicalize=False)
        raw_hash2 = hash_payload(payload2, canonicalize=False)
        self.assertNotEqual(raw_hash1, raw_hash2)

        # Canonical hashes MUST be identical (verifying the solution)
        canon_hash1 = hash_payload(payload1, canonicalize=True, provider="anthropic")
        canon_hash2 = hash_payload(payload2, canonicalize=True, provider="anthropic")
        self.assertEqual(canon_hash1, canon_hash2)

    def test_detect_drift_flags_avoidable_cache_penetration(self):
        payload_a = {
            "tools": [{"name": "foo", "input_schema": {"properties": {"x": {"type": "string"}, "y": {"type": "string"}}}}]
        }
        payload_b = {
            "tools": [{"name": "foo", "input_schema": {"properties": {"y": {"type": "string"}, "x": {"type": "string"}}}}]
        }

        report = detect_drift(payload_a, payload_b, provider="anthropic")
        self.assertFalse(report.is_raw_identical)
        self.assertTrue(report.is_canonical_identical)
        self.assertTrue(report.drift_detected)
        self.assertFalse(report.semantic_drift)
        self.assertIn("Cache Penetration Warning", report.diff_summary)

    def test_detect_drift_flags_real_semantic_difference(self):
        payload_a = {"tools": [{"name": "foo", "description": "Version 1"}]}
        payload_b = {"tools": [{"name": "foo", "description": "Version 2"}]}

        report = detect_drift(payload_a, payload_b)
        self.assertFalse(report.is_raw_identical)
        self.assertFalse(report.is_canonical_identical)
        self.assertFalse(report.drift_detected)
        self.assertTrue(report.semantic_drift)


if __name__ == "__main__":
    unittest.main()
