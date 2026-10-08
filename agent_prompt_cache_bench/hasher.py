"""
Wire-level SHA-256 Hashing and Prompt Cache Drift Detection.

Provides byte-precise hash computations and token/prompt drift diagnostics
to prove whether two payloads maintain cache consistency at the provider boundary.
"""

from __future__ import annotations

import difflib
import hashlib
import json
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from agent_prompt_cache_bench.normalizer import (
    canonical_json_bytes,
    canonical_json_dumps,
    normalize_provider_payload,
)


def compute_sha256(data: bytes | str) -> str:
    """Computes SHA-256 hexadecimal digest for raw bytes or string."""
    if isinstance(data, str):
        data = data.encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def raw_json_bytes(payload: Any) -> bytes:
    """Serializes payload preserving current dictionary key insertion order."""
    return json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")


def hash_payload(
    payload: Dict[str, Any],
    canonicalize: bool = False,
    provider: str = "auto",
) -> str:
    """Computes SHA-256 hash of a payload.

    Args:
        payload: Dict representation of the request payload or tool spec.
        canonicalize: If True, recursively normalizes schemas, tools, and keys
            before hashing. If False, hashes current wire representation as-is.
        provider: Provider wire format ('openai', 'anthropic', 'gemini', 'auto').
    """
    if canonicalize:
        normalized = normalize_provider_payload(payload, provider=provider)
        data = canonical_json_bytes(normalized)
    else:
        data = raw_json_bytes(payload)
    return compute_sha256(data)


@dataclass
class DriftReport:
    """Diagnostic report comparing two JSON payloads at raw and canonical boundaries."""

    raw_hash_a: str
    raw_hash_b: str
    canonical_hash_a: str
    canonical_hash_b: str
    is_raw_identical: bool
    is_canonical_identical: bool
    drift_detected: bool  # True if raw hashes differ but canonical hashes match
    semantic_drift: bool  # True if canonical hashes differ (real content changes)
    raw_bytes_a: int
    raw_bytes_b: int
    diff_summary: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "raw_hash_a": self.raw_hash_a,
            "raw_hash_b": self.raw_hash_b,
            "canonical_hash_a": self.canonical_hash_a,
            "canonical_hash_b": self.canonical_hash_b,
            "is_raw_identical": self.is_raw_identical,
            "is_canonical_identical": self.is_canonical_identical,
            "drift_detected": self.drift_detected,
            "semantic_drift": self.semantic_drift,
            "raw_bytes_a": self.raw_bytes_a,
            "raw_bytes_b": self.raw_bytes_b,
            "diff_summary": self.diff_summary,
        }


def detect_drift(
    payload_a: Dict[str, Any],
    payload_b: Dict[str, Any],
    provider: str = "auto",
) -> DriftReport:
    """Compares two payloads to detect wire-level token/cache drift.

    Identifies if raw serialization differences would cause cache penetration
    even when the semantic contents are identical.
    """
    bytes_a = raw_json_bytes(payload_a)
    bytes_b = raw_json_bytes(payload_b)
    raw_hash_a = compute_sha256(bytes_a)
    raw_hash_b = compute_sha256(bytes_b)

    canon_a = normalize_provider_payload(payload_a, provider=provider)
    canon_b = normalize_provider_payload(payload_b, provider=provider)
    canon_bytes_a = canonical_json_bytes(canon_a)
    canon_bytes_b = canonical_json_bytes(canon_b)
    canonical_hash_a = compute_sha256(canon_bytes_a)
    canonical_hash_b = compute_sha256(canon_bytes_b)

    is_raw_identical = (raw_hash_a == raw_hash_b)
    is_canonical_identical = (canonical_hash_a == canonical_hash_b)
    drift_detected = (not is_raw_identical) and is_canonical_identical
    semantic_drift = not is_canonical_identical

    # Generate diff summary
    if is_raw_identical:
        diff_summary = "Byte-identical: 100% cache match at wire level."
    elif drift_detected:
        lines_a = json.dumps(payload_a, indent=2, ensure_ascii=False).splitlines()
        lines_b = json.dumps(payload_b, indent=2, ensure_ascii=False).splitlines()
        diff = list(difflib.unified_diff(
            lines_a, lines_b,
            fromfile="payload_a (raw)",
            tofile="payload_b (raw)",
            lineterm="",
            n=2,
        ))
        diff_summary = (
            "Cache Penetration Warning: Wire bytes differ due to key order/permutation!\n"
            + "\n".join(diff[:25])
            + ("\n... [diff truncated]" if len(diff) > 25 else "")
        )
    else:
        diff_summary = "Semantic mismatch: Payloads contain differing values or structures."

    return DriftReport(
        raw_hash_a=raw_hash_a,
        raw_hash_b=raw_hash_b,
        canonical_hash_a=canonical_hash_a,
        canonical_hash_b=canonical_hash_b,
        is_raw_identical=is_raw_identical,
        is_canonical_identical=is_canonical_identical,
        drift_detected=drift_detected,
        semantic_drift=semantic_drift,
        raw_bytes_a=len(bytes_a),
        raw_bytes_b=len(bytes_b),
        diff_summary=diff_summary,
    )
