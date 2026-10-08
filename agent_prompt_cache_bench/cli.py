"""
Command-Line Interface for agent-prompt-cache-bench.

Provides commands:
- bench: Run multi-session prompt cache stability and token savings benchmark.
- normalize: Output canonical, deterministic JSON schema or tool payload.
- verify: Compare two JSON payloads for wire-level prompt cache consistency.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import List, Optional

from agent_prompt_cache_bench.bench import (
    format_benchmark_table,
    run_benchmark,
)
from agent_prompt_cache_bench.hasher import detect_drift
from agent_prompt_cache_bench.normalizer import (
    canonical_json_dumps,
    canonicalize_schema,
    normalize_provider_payload,
)


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="agent-cache-bench",
        description="LLM Prompt Caching Wire-Level Benchmark & Schema Canonicalizer.",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # --- bench command ---
    bench_parser = subparsers.add_parser(
        "bench",
        help="Run multi-turn agent prompt cache hit stability and token savings benchmark.",
    )
    bench_parser.add_argument(
        "--provider",
        choices=["anthropic", "openai", "gemini"],
        default="anthropic",
        help="Target LLM provider pricing & cache profile (default: anthropic)",
    )
    bench_parser.add_argument(
        "--sessions",
        type=int,
        default=5,
        help="Number of agent sessions / worker restarts (default: 5)",
    )
    bench_parser.add_argument(
        "--turns",
        type=int,
        default=10,
        help="Number of turns per session (default: 10)",
    )
    bench_parser.add_argument(
        "--tools",
        type=int,
        default=8,
        help="Number of agent tools with nested schemas to register (default: 8)",
    )
    bench_parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for repeatable key permutation (default: 42)",
    )
    bench_parser.add_argument(
        "--json",
        action="store_true",
        help="Output raw JSON results instead of human-readable table",
    )

    # --- normalize command ---
    norm_parser = subparsers.add_parser(
        "normalize",
        help="Canonicalize a JSON schema or provider payload to deterministic wire format.",
    )
    norm_parser.add_argument(
        "file",
        help="Path to JSON file to canonicalize, or '-' for stdin",
    )
    norm_parser.add_argument(
        "--provider",
        choices=["auto", "anthropic", "openai", "gemini"],
        default="auto",
        help="Provider format parser (default: auto)",
    )
    norm_parser.add_argument(
        "--indent",
        type=int,
        default=2,
        help="Indentation spaces for output JSON (default: 2, 0 for compact wire format)",
    )
    norm_parser.add_argument(
        "-o", "--output",
        help="Path to write normalized JSON (default: stdout)",
    )

    # --- verify command ---
    verify_parser = subparsers.add_parser(
        "verify",
        help="Compare two JSON files to check wire-level prompt cache consistency.",
    )
    verify_parser.add_argument(
        "file_a",
        help="Path to first JSON file",
    )
    verify_parser.add_argument(
        "file_b",
        help="Path to second JSON file",
    )
    verify_parser.add_argument(
        "--provider",
        choices=["auto", "anthropic", "openai", "gemini"],
        default="auto",
        help="Provider schema mode (default: auto)",
    )
    verify_parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Display detailed raw hash diff and diagnostics",
    )

    return parser


def handle_bench(args: argparse.Namespace) -> int:
    report = run_benchmark(
        provider=args.provider,
        num_sessions=args.sessions,
        turns_per_session=args.turns,
        num_tools=args.tools,
        seed=args.seed,
    )
    if args.json:
        print(json.dumps(report.to_dict(), indent=2, ensure_ascii=False))
    else:
        print(format_benchmark_table(report))
        print()
        print("  Key Takeaways:")
        print(f"  * Baseline unnormalized wire payloads suffered from {report.baseline_unique_prefix_hashes} distinct prefix hashes, causing prompt cache penetration.")
        print(f"  * Canonicalized wire payloads achieved {report.canonical_hit_rate * 100:.1f}% cache hit rate and saved ~{report.cost_savings_pct:.1f}% on API costs.")
        print()
    return 0


def handle_normalize(args: argparse.Namespace) -> int:
    try:
        if args.file == "-":
            raw_text = sys.stdin.read()
        else:
            with open(args.file, "r", encoding="utf-8") as f:
                raw_text = f.read()
        data = json.loads(raw_text)
    except Exception as e:
        sys.stderr.write(f"Error reading JSON from '{args.file}': {e}\n")
        return 1

    if isinstance(data, dict) and any(k in data for k in ("messages", "contents", "tools")):
        normalized = normalize_provider_payload(data, provider=args.provider)
    else:
        normalized = canonicalize_schema(data)

    if args.indent > 0:
        output_str = json.dumps(normalized, indent=args.indent, ensure_ascii=False) + "\n"
    else:
        output_str = canonical_json_dumps(normalized)

    if args.output:
        try:
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(output_str)
        except Exception as e:
            sys.stderr.write(f"Error writing output to '{args.output}': {e}\n")
            return 1
    else:
        sys.stdout.write(output_str)

    return 0


def handle_verify(args: argparse.Namespace) -> int:
    try:
        with open(args.file_a, "r", encoding="utf-8") as f:
            data_a = json.loads(f.read())
        with open(args.file_b, "r", encoding="utf-8") as f:
            data_b = json.loads(f.read())
    except Exception as e:
        sys.stderr.write(f"Error loading comparison files: {e}\n")
        return 2

    report = detect_drift(data_a, data_b, provider=args.provider)

    print("=" * 72)
    print("  WIRE-LEVEL PROMPT CACHE CONSISTENCY VERIFICATION")
    print("=" * 72)
    print(f"  File A: {args.file_a}")
    print(f"  File B: {args.file_b}")
    print("-" * 72)
    print(f"  Raw SHA-256 (File A):       {report.raw_hash_a[:16]}...{report.raw_hash_a[-8:]}")
    print(f"  Raw SHA-256 (File B):       {report.raw_hash_b[:16]}...{report.raw_hash_b[-8:]}")
    print(f"  Canonical SHA-256 (A & B):  {report.canonical_hash_a[:16]}...{report.canonical_hash_a[-8:]}")
    print("-" * 72)

    if report.is_raw_identical:
        print("  Status: [PASS] Perfect wire-level byte match.")
        print("  Prompt cache hit guaranteed without normalization.")
        return 0
    elif report.drift_detected:
        print("  Status: [PASS WITH CANONICALIZATION] Cache Drift Prevented!")
        print("  Explanation: Raw byte hashes differ due to dictionary key permutation")
        print("               or tool list order variance. Canonicalization restores 100%")
        print("               wire-level equivalence and guarantees prompt cache hit.")
        if args.verbose:
            print("\n  Diff Summary:")
            print(report.diff_summary)
        return 0
    else:
        print("  Status: [FAIL] Semantic mismatch detected.")
        print("  Explanation: The two files contain differing schema definitions or contents.")
        if args.verbose:
            print("\n  Diff Summary:")
            print(report.diff_summary)
        return 1


def main(argv: Optional[List[str]] = None) -> int:
    parser = create_parser()
    args = parser.parse_args(argv)

    if not args.command:
        parser.print_help()
        return 0

    if args.command == "bench":
        return handle_bench(args)
    elif args.command == "normalize":
        return handle_normalize(args)
    elif args.command == "verify":
        return handle_verify(args)

    return 0


if __name__ == "__main__":
    sys.exit(main())
