"""
agent-prompt-cache-bench: Wire-Level Prompt Cache Benchmark & Schema Canonicalizer.
"""

from agent_prompt_cache_bench.bench import (
    BenchmarkReport,
    BenchmarkRunner,
    format_benchmark_table,
    get_synthetic_agent_tools,
    run_benchmark,
)
from agent_prompt_cache_bench.hasher import (
    DriftReport,
    compute_sha256,
    detect_drift,
    hash_payload,
    raw_json_bytes,
)
from agent_prompt_cache_bench.normalizer import (
    canonical_json_bytes,
    canonical_json_dumps,
    canonicalize_schema,
    normalize_provider_payload,
    normalize_tool_spec,
)

__version__ = "0.1.0"

__all__ = [
    "canonicalize_schema",
    "normalize_tool_spec",
    "normalize_provider_payload",
    "canonical_json_dumps",
    "canonical_json_bytes",
    "compute_sha256",
    "hash_payload",
    "detect_drift",
    "DriftReport",
    "raw_json_bytes",
    "BenchmarkReport",
    "BenchmarkRunner",
    "run_benchmark",
    "format_benchmark_table",
    "get_synthetic_agent_tools",
]
