"""
Prompt Cache Benchmark Engine.

Simulates multi-turn AI agent sessions with dynamic tool loading, process restarts,
and multi-worker dictionary key permutations to benchmark:
1. Cache Penetration Rate (0% baseline vs 100% canonicalized)
2. Wire Hash Churn & Prefix Stability
3. Simulated Token Savings & Billing Impact
4. Simulated TTFT (Time To First Token) Latency Improvements
"""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from agent_prompt_cache_bench.hasher import (
    compute_sha256,
    hash_payload,
    raw_json_bytes,
)
from agent_prompt_cache_bench.normalizer import (
    canonical_json_bytes,
    canonical_json_dumps,
    canonicalize_schema,
    normalize_provider_payload,
)


@dataclass
class ProviderPricing:
    """Pricing and latency characteristics for LLM prompt caching."""

    name: str
    base_input_per_m: float  # USD per 1M input tokens
    cache_write_per_m: float  # USD per 1M tokens written to cache
    cache_read_per_m: float  # USD per 1M cached tokens read
    ttft_miss_ms: float  # Estimated Time To First Token on cache miss
    ttft_hit_ms: float  # Estimated Time To First Token on cache hit
    min_cache_tokens: int = 1024


PROVIDER_PROFILES: Dict[str, ProviderPricing] = {
    "anthropic": ProviderPricing(
        name="Anthropic (Claude 3.5 / 3.7 Sonnet)",
        base_input_per_m=3.00,
        cache_write_per_m=3.75,
        cache_read_per_m=0.30,  # 90% discount
        ttft_miss_ms=1850.0,
        ttft_hit_ms=280.0,
        min_cache_tokens=1024,
    ),
    "openai": ProviderPricing(
        name="OpenAI (GPT-4o)",
        base_input_per_m=2.50,
        cache_write_per_m=2.50,
        cache_read_per_m=1.25,  # 50% discount
        ttft_miss_ms=1450.0,
        ttft_hit_ms=380.0,
        min_cache_tokens=1024,
    ),
    "gemini": ProviderPricing(
        name="Google Gemini (1.5 Pro)",
        base_input_per_m=3.50,
        cache_write_per_m=3.50,
        cache_read_per_m=0.875,  # 75% discount
        ttft_miss_ms=1600.0,
        ttft_hit_ms=320.0,
        min_cache_tokens=1024,
    ),
}


def get_synthetic_agent_tools(provider: str = "anthropic") -> List[Dict[str, Any]]:
    """Returns a realistic suite of rich agent tools with complex nested schemas."""
    raw_tools = [
        {
            "name": "read_file",
            "description": "Reads contents from a local file path with optional slice bounds.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Absolute file path."},
                    "offset": {"type": "integer", "description": "Byte offset to start reading from.", "default": 0},
                    "limit": {"type": "integer", "description": "Maximum bytes to read."},
                },
                "required": ["path"],
                "additionalProperties": False,
            },
        },
        {
            "name": "write_file",
            "description": "Writes or overwrites content into a file on the local filesystem.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Destination file path."},
                    "content": {"type": "string", "description": "Raw string content to write."},
                    "mode": {"type": "string", "enum": ["overwrite", "append"], "default": "overwrite"},
                    "create_parents": {"type": "boolean", "default": True},
                },
                "required": ["path", "content"],
                "additionalProperties": False,
            },
        },
        {
            "name": "execute_command",
            "description": "Executes a shell command in a persistent terminal environment.",
            "parameters": {
                "type": "object",
                "$defs": {
                    "EnvVar": {
                        "type": "object",
                        "properties": {
                            "key": {"type": "string"},
                            "val": {"type": "string"},
                        },
                        "required": ["key", "val"],
                    }
                },
                "properties": {
                    "command": {"type": "string", "description": "Command string to run."},
                    "timeout_seconds": {"type": "integer", "default": 60},
                    "working_dir": {"type": "string", "description": "CWD path."},
                    "env": {
                        "type": "array",
                        "items": {"$ref": "#/$defs/EnvVar"},
                    },
                },
                "required": ["command"],
            },
        },
        {
            "name": "search_codebase",
            "description": "Performs fast ripgrep regex searching across source files.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Regex or literal pattern."},
                    "case_sensitive": {"type": "boolean", "default": False},
                    "includes": {"type": "array", "items": {"type": "string"}},
                    "excludes": {"type": "array", "items": {"type": "string"}},
                    "max_matches": {"type": "integer", "default": 100},
                },
                "required": ["query"],
            },
        },
        {
            "name": "git_commit",
            "description": "Stages and commits files to the local git repository.",
            "parameters": {
                "type": "object",
                "properties": {
                    "message": {"type": "string", "description": "Commit message header."},
                    "files": {"type": "array", "items": {"type": "string"}},
                    "amend": {"type": "boolean", "default": False},
                    "signoff": {"type": "boolean", "default": True},
                },
                "required": ["message"],
            },
        },
        {
            "name": "create_pull_request",
            "description": "Creates a GitHub Pull Request with rich body and reviewers.",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "head_branch": {"type": "string"},
                    "base_branch": {"type": "string", "default": "main"},
                    "draft": {"type": "boolean", "default": False},
                    "body": {"type": "string"},
                    "reviewers": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["title", "head_branch"],
            },
        },
        {
            "name": "database_query",
            "description": "Executes SQL query against agent transactional database.",
            "parameters": {
                "type": "object",
                "properties": {
                    "sql": {"type": "string"},
                    "readonly": {"type": "boolean", "default": True},
                    "timeout_ms": {"type": "integer", "default": 5000},
                    "params": {
                        "type": "object",
                        "additionalProperties": True,
                    },
                },
                "required": ["sql"],
            },
        },
        {
            "name": "http_request",
            "description": "Sends an HTTP request with custom headers, query params, and body.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string"},
                    "method": {"type": "string", "enum": ["GET", "POST", "PUT", "DELETE", "PATCH"]},
                    "headers": {"type": "object", "additionalProperties": {"type": "string"}},
                    "body": {"type": "string"},
                    "timeout_seconds": {"type": "number", "default": 30.0},
                },
                "required": ["url", "method"],
            },
        },
    ]

    # Convert to provider wire format
    p = provider.lower()
    tools: List[Dict[str, Any]] = []

    for t in raw_tools:
        if p == "anthropic":
            tools.append({
                "name": t["name"],
                "description": t["description"],
                "input_schema": t["parameters"],
            })
        elif p == "openai":
            tools.append({
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t["description"],
                    "parameters": t["parameters"],
                },
            })
        elif p == "gemini":
            tools.append({
                "name": t["name"],
                "description": t["description"],
                "parameters": t["parameters"],
            })
        else:
            tools.append(t)

    return tools


def permute_dict_keys(obj: Any, rng: random.Random) -> Any:
    """Recursively permutes dictionary key order and tool list order.

    Simulates non-deterministic dictionary insertion order across worker processes,
    runtime restarts, and dynamic MCP plugin registrations.
    """
    if isinstance(obj, dict):
        keys = list(obj.keys())
        rng.shuffle(keys)
        new_dict = {}
        for k in keys:
            new_dict[k] = permute_dict_keys(obj[k], rng)
        return new_dict
    elif isinstance(obj, list):
        # We only shuffle tool lists if it looks like a list of tools
        # Keep message lists chronological!
        if obj and isinstance(obj[0], dict) and ("name" in obj[0] or "function" in obj[0]):
            shuffled = list(obj)
            rng.shuffle(shuffled)
            return [permute_dict_keys(item, rng) for item in shuffled]
        return [permute_dict_keys(item, rng) for item in obj]
    return obj


@dataclass
class BenchmarkReport:
    """Summary metrics comparing baseline raw wire payloads against canonicalized payloads."""

    provider: str
    num_sessions: int
    turns_per_session: int
    total_turns: int
    num_tools: int

    # Cache hits & misses
    baseline_cache_hits: int
    baseline_cache_misses: int
    baseline_hit_rate: float
    canonical_cache_hits: int
    canonical_cache_misses: int
    canonical_hit_rate: float

    # Hash stability
    baseline_unique_prefix_hashes: int
    canonical_unique_prefix_hashes: int

    # Token and cost calculations
    baseline_billed_tokens: int
    canonical_billed_tokens: int
    token_savings_pct: float
    baseline_cost_usd: float
    canonical_cost_usd: float
    cost_savings_pct: float

    # Latency estimation
    baseline_total_latency_s: float
    canonical_total_latency_s: float
    latency_reduction_pct: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "provider": self.provider,
            "num_sessions": self.num_sessions,
            "turns_per_session": self.turns_per_session,
            "total_turns": self.total_turns,
            "num_tools": self.num_tools,
            "baseline": {
                "cache_hits": self.baseline_cache_hits,
                "cache_misses": self.baseline_cache_misses,
                "hit_rate_pct": round(self.baseline_hit_rate * 100, 2),
                "unique_prefix_hashes": self.baseline_unique_prefix_hashes,
                "billed_tokens": self.baseline_billed_tokens,
                "cost_usd": round(self.baseline_cost_usd, 4),
                "total_ttft_latency_s": round(self.baseline_total_latency_s, 2),
            },
            "canonicalized": {
                "cache_hits": self.canonical_cache_hits,
                "cache_misses": self.canonical_cache_misses,
                "hit_rate_pct": round(self.canonical_hit_rate * 100, 2),
                "unique_prefix_hashes": self.canonical_unique_prefix_hashes,
                "billed_tokens": self.canonical_billed_tokens,
                "cost_usd": round(self.canonical_cost_usd, 4),
                "total_ttft_latency_s": round(self.canonical_total_latency_s, 2),
            },
            "improvements": {
                "hit_rate_gain_pct": round((self.canonical_hit_rate - self.baseline_hit_rate) * 100, 2),
                "token_savings_pct": round(self.token_savings_pct, 2),
                "cost_savings_pct": round(self.cost_savings_pct, 2),
                "latency_reduction_pct": round(self.latency_reduction_pct, 2),
            },
        }


class BenchmarkRunner:
    """Orchestrates multi-session agent benchmark simulations."""

    def __init__(
        self,
        provider: str = "anthropic",
        num_sessions: int = 5,
        turns_per_session: int = 10,
        num_tools: int = 8,
        seed: int = 42,
    ):
        self.provider = provider.lower()
        self.num_sessions = num_sessions
        self.turns_per_session = turns_per_session
        self.num_tools = num_tools
        self.seed = seed
        self.pricing = PROVIDER_PROFILES.get(self.provider, PROVIDER_PROFILES["anthropic"])

    def run(self) -> BenchmarkReport:
        rng = random.Random(self.seed)
        all_tools = get_synthetic_agent_tools(self.provider)[: self.num_tools]

        system_prompt = (
            "You are an expert autonomous software engineering agent. "
            "You have access to a suite of system tools to inspect, modify, "
            "test, and commit code changes safely."
        )

        # In-memory simulated provider prompt caches:
        # Maps prefix_hash -> expiration_turn / state
        baseline_cache_registry: set[str] = set()
        canonical_cache_registry: set[str] = set()

        baseline_hits = 0
        baseline_misses = 0
        canonical_hits = 0
        canonical_misses = 0

        baseline_prefix_hashes: set[str] = set()
        canonical_prefix_hashes: set[str] = set()

        baseline_billed_tokens = 0
        canonical_billed_tokens = 0

        baseline_latency_ms = 0.0
        canonical_latency_ms = 0.0

        total_turns = self.num_sessions * self.turns_per_session

        for session_idx in range(self.num_sessions):
            # Each session represents an agent instance, worker thread, or restart.
            # Conversation history accumulates across turns within the session.
            messages: List[Dict[str, Any]] = []

            for turn_idx in range(self.turns_per_session):
                # Add new user request or tool result
                user_content = f"Turn {turn_idx + 1}: Investigate issue #{1000 + turn_idx} and run tests."
                messages.append({"role": "user", "content": user_content})

                # Construct raw payload for this turn
                raw_payload = {
                    "model": "claude-3-7-sonnet" if self.provider == "anthropic" else "gpt-4o",
                    "system": system_prompt,
                    "tools": copy.deepcopy(all_tools),
                    "messages": copy.deepcopy(messages),
                }

                # --- 1. BASELINE (Simulate unnormalized runtime dictionary / worker variance) ---
                # Permute tool registration order and schema key insertion orders
                permuted_payload = permute_dict_keys(raw_payload, rng)
                # Provider computes prefix hash on tools + system prompt
                prefix_obj_baseline = {
                    "system": permuted_payload["system"],
                    "tools": permuted_payload["tools"],
                }
                baseline_wire_bytes = raw_json_bytes(prefix_obj_baseline)
                baseline_prefix_hash = compute_sha256(baseline_wire_bytes)
                baseline_prefix_hashes.add(baseline_prefix_hash)

                # Estimate prefix tokens (approx 4 chars per token)
                prefix_tokens = max(len(baseline_wire_bytes) // 4, self.pricing.min_cache_tokens)
                turn_msg_tokens = len(user_content) // 4 + 30

                if baseline_prefix_hash in baseline_cache_registry:
                    # Cache HIT (only occurs if hash accidentally collides)
                    baseline_hits += 1
                    # Cached prefix read + turn message input
                    baseline_billed_tokens += int(prefix_tokens * (self.pricing.cache_read_per_m / self.pricing.base_input_per_m)) + turn_msg_tokens
                    baseline_latency_ms += self.pricing.ttft_hit_ms
                else:
                    # Cache PENETRATION / MISS
                    baseline_misses += 1
                    # Register this new prefix to cache
                    baseline_cache_registry.add(baseline_prefix_hash)
                    baseline_billed_tokens += prefix_tokens + turn_msg_tokens
                    baseline_latency_ms += self.pricing.ttft_miss_ms

                # --- 2. CANONICALIZED (Normalized via agent_prompt_cache_bench) ---
                canonical_payload = normalize_provider_payload(permuted_payload, provider=self.provider)
                prefix_obj_canonical = {
                    "system": canonical_payload["system"],
                    "tools": canonical_payload["tools"],
                }
                canonical_wire_bytes = canonical_json_bytes(prefix_obj_canonical)
                canonical_prefix_hash = compute_sha256(canonical_wire_bytes)
                canonical_prefix_hashes.add(canonical_prefix_hash)

                canon_prefix_tokens = max(len(canonical_wire_bytes) // 4, self.pricing.min_cache_tokens)

                if canonical_prefix_hash in canonical_cache_registry:
                    # Cache HIT
                    canonical_hits += 1
                    canonical_billed_tokens += int(canon_prefix_tokens * (self.pricing.cache_read_per_m / self.pricing.base_input_per_m)) + turn_msg_tokens
                    canonical_latency_ms += self.pricing.ttft_hit_ms
                else:
                    # Cache MISS (only on the very first turn to warm up cache)
                    canonical_misses += 1
                    canonical_cache_registry.add(canonical_prefix_hash)
                    canonical_billed_tokens += canon_prefix_tokens + turn_msg_tokens
                    canonical_latency_ms += self.pricing.ttft_miss_ms

        # Compute aggregates
        b_hit_rate = baseline_hits / total_turns if total_turns > 0 else 0.0
        c_hit_rate = canonical_hits / total_turns if total_turns > 0 else 0.0

        # Cost in USD
        baseline_cost_usd = (baseline_billed_tokens / 1_000_000.0) * self.pricing.base_input_per_m
        canonical_cost_usd = (canonical_billed_tokens / 1_000_000.0) * self.pricing.base_input_per_m

        token_savings_pct = (
            ((baseline_billed_tokens - canonical_billed_tokens) / baseline_billed_tokens) * 100.0
            if baseline_billed_tokens > 0
            else 0.0
        )
        cost_savings_pct = (
            ((baseline_cost_usd - canonical_cost_usd) / baseline_cost_usd) * 100.0
            if baseline_cost_usd > 0
            else 0.0
        )
        latency_reduction_pct = (
            ((baseline_latency_ms - canonical_latency_ms) / baseline_latency_ms) * 100.0
            if baseline_latency_ms > 0
            else 0.0
        )

        return BenchmarkReport(
            provider=self.pricing.name,
            num_sessions=self.num_sessions,
            turns_per_session=self.turns_per_session,
            total_turns=total_turns,
            num_tools=self.num_tools,
            baseline_cache_hits=baseline_hits,
            baseline_cache_misses=baseline_misses,
            baseline_hit_rate=b_hit_rate,
            canonical_cache_hits=canonical_hits,
            canonical_cache_misses=canonical_misses,
            canonical_hit_rate=c_hit_rate,
            baseline_unique_prefix_hashes=len(baseline_prefix_hashes),
            canonical_unique_prefix_hashes=len(canonical_prefix_hashes),
            baseline_billed_tokens=baseline_billed_tokens,
            canonical_billed_tokens=canonical_billed_tokens,
            token_savings_pct=token_savings_pct,
            baseline_cost_usd=baseline_cost_usd,
            canonical_cost_usd=canonical_cost_usd,
            cost_savings_pct=cost_savings_pct,
            baseline_total_latency_s=baseline_latency_ms / 1000.0,
            canonical_total_latency_s=canonical_latency_ms / 1000.0,
            latency_reduction_pct=latency_reduction_pct,
        )


def format_benchmark_table(report: BenchmarkReport) -> str:
    """Formats benchmark results into an ASCII table."""
    border = "=" * 88
    divider = "-" * 88

    rows = [
        border,
        f"  AGENT PROMPT CACHE BENCHMARK: WIRE-LEVEL DETERMINISM & CACHE HIT STABILITY",
        f"  Provider: {report.provider}",
        f"  Sessions: {report.num_sessions} | Turns/Session: {report.turns_per_session} | Total Turns: {report.total_turns} | Tools: {report.num_tools}",
        border,
        f"  {'Metric':<34} {'Baseline (Raw)':<20} {'Canonicalized':<18} {'Improvement':<12}",
        divider,
        f"  {'Prefix Wire Hashes Generated':<34} {f'{report.baseline_unique_prefix_hashes} unique':<20} {f'{report.canonical_unique_prefix_hashes} unique':<18} {f'{report.baseline_unique_prefix_hashes}x stability':<12}",
        f"  {'Prompt Cache Hit Rate':<34} {f'{report.baseline_hit_rate * 100:.1f}% (penetration)':<20} {f'{report.canonical_hit_rate * 100:.1f}% (stable)':<18} {f'+{(report.canonical_hit_rate - report.baseline_hit_rate) * 100:.1f}%':<12}",
        f"  {'Cache Hits / Misses':<34} {f'{report.baseline_cache_hits} / {report.baseline_cache_misses}':<20} {f'{report.canonical_cache_hits} / {report.canonical_cache_misses}':<18} {f'{report.canonical_cache_hits} extra hits':<12}",
        f"  {'Effective Billed Tokens':<34} {f'{report.baseline_billed_tokens:,}':<20} {f'{report.canonical_billed_tokens:,}':<18} {f'-{report.token_savings_pct:.1f}%':<12}",
        f"  {'Estimated Cost (USD)':<34} {f'${report.baseline_cost_usd:.4f}':<20} {f'${report.canonical_cost_usd:.4f}':<18} {f'-{report.cost_savings_pct:.1f}%':<12}",
        f"  {'Simulated TTFT Latency':<34} {f'{report.baseline_total_latency_s:.2f}s':<20} {f'{report.canonical_total_latency_s:.2f}s':<18} {f'-{report.latency_reduction_pct:.1f}%':<12}",
        border,
    ]
    return "\n".join(rows)


def run_benchmark(
    provider: str = "anthropic",
    num_sessions: int = 5,
    turns_per_session: int = 10,
    num_tools: int = 8,
    seed: int = 42,
) -> BenchmarkReport:
    """Convenience function to run benchmark and return report."""
    runner = BenchmarkRunner(
        provider=provider,
        num_sessions=num_sessions,
        turns_per_session=turns_per_session,
        num_tools=num_tools,
        seed=seed,
    )
    return runner.run()
