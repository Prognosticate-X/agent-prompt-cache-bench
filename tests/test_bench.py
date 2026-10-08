"""Unit tests for benchmark simulation and CLI commands."""

import io
import json
import os
import sys
import tempfile
import unittest
from unittest.mock import patch

from agent_prompt_cache_bench.bench import (
    BenchmarkRunner,
    format_benchmark_table,
    run_benchmark,
)
from agent_prompt_cache_bench.cli import main


class TestBenchmarkEngine(unittest.TestCase):
    def test_run_benchmark_anthropic(self):
        report = run_benchmark(
            provider="anthropic",
            num_sessions=3,
            turns_per_session=5,
            num_tools=4,
            seed=42,
        )
        # Total turns = 3 * 5 = 15
        self.assertEqual(report.total_turns, 15)
        # Canonicalization should produce only 1 unique prefix hash
        self.assertEqual(report.canonical_unique_prefix_hashes, 1)
        # Canonical hit rate should be high (1 miss on turn 1 warm-up, then 14 hits)
        self.assertEqual(report.canonical_cache_hits, 14)
        self.assertEqual(report.canonical_cache_misses, 1)
        self.assertAlmostEqual(report.canonical_hit_rate, 14 / 15, places=3)
        # Baseline should have many unique prefix hashes due to permutations
        self.assertGreater(report.baseline_unique_prefix_hashes, 1)
        self.assertGreater(report.baseline_cache_misses, report.canonical_cache_misses)
        # Savings
        self.assertGreater(report.cost_savings_pct, 0.0)
        self.assertGreater(report.latency_reduction_pct, 0.0)

    def test_run_benchmark_openai_and_gemini(self):
        for provider in ("openai", "gemini"):
            report = run_benchmark(
                provider=provider,
                num_sessions=2,
                turns_per_session=4,
                num_tools=3,
                seed=99,
            )
            self.assertEqual(report.canonical_unique_prefix_hashes, 1)
            self.assertGreater(report.cost_savings_pct, 0.0)

    def test_format_table(self):
        report = run_benchmark(
            provider="anthropic",
            num_sessions=2,
            turns_per_session=3,
            num_tools=2,
        )
        table = format_benchmark_table(report)
        self.assertIn("AGENT PROMPT CACHE BENCHMARK", table)
        self.assertIn("Prefix Wire Hashes Generated", table)
        self.assertIn("Prompt Cache Hit Rate", table)


class TestCLI(unittest.TestCase):
    def test_cli_bench(self):
        stdout_capture = io.StringIO()
        with patch("sys.stdout", stdout_capture):
            code = main(["bench", "--provider", "anthropic", "--sessions", "2", "--turns", "2", "--tools", "2"])
        self.assertEqual(code, 0)
        output = stdout_capture.getvalue()
        self.assertIn("AGENT PROMPT CACHE BENCHMARK", output)

    def test_cli_bench_json(self):
        stdout_capture = io.StringIO()
        with patch("sys.stdout", stdout_capture):
            code = main(["bench", "--json", "--sessions", "2", "--turns", "2", "--tools", "2"])
        self.assertEqual(code, 0)
        data = json.loads(stdout_capture.getvalue())
        self.assertIn("canonicalized", data)
        self.assertIn("baseline", data)
        self.assertIn("improvements", data)

    def test_cli_normalize(self):
        with tempfile.NamedTemporaryFile("w+", delete=False, suffix=".json") as f:
            f.write(json.dumps({"z": 1, "a": 2, "properties": {"m": {}, "b": {}}}))
            temp_path = f.name

        try:
            stdout_capture = io.StringIO()
            with patch("sys.stdout", stdout_capture):
                code = main(["normalize", temp_path])
            self.assertEqual(code, 0)
            norm = json.loads(stdout_capture.getvalue())
            self.assertEqual(list(norm.keys()), ["a", "properties", "z"])
            self.assertEqual(list(norm["properties"].keys()), ["b", "m"])
        finally:
            os.remove(temp_path)

    def test_cli_verify_drift(self):
        with tempfile.NamedTemporaryFile("w+", delete=False, suffix=".json") as f1:
            f1.write(json.dumps({"type": "object", "properties": {"b": {"type": "string"}, "a": {"type": "string"}}}))
            p1 = f1.name
        with tempfile.NamedTemporaryFile("w+", delete=False, suffix=".json") as f2:
            f2.write(json.dumps({"properties": {"a": {"type": "string"}, "b": {"type": "string"}}, "type": "object"}))
            p2 = f2.name

        try:
            stdout_capture = io.StringIO()
            with patch("sys.stdout", stdout_capture):
                code = main(["verify", p1, p2])
            self.assertEqual(code, 0)
            output = stdout_capture.getvalue()
            self.assertIn("PASS WITH CANONICALIZATION", output)
        finally:
            os.remove(p1)
            os.remove(p2)


if __name__ == "__main__":
    unittest.main()
