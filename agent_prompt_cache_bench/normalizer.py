"""
Recursive JSON Schema and Provider Request Payload Normalizer.

Ensures wire-level determinism for LLM tool declarations and prompt caching.
Addresses cache penetration / prefix drift across multi-turn agent sessions.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Union


def canonical_json_dumps(obj: Any) -> str:
    """Serialize an object to a canonical, deterministic JSON string.

    Uses sorted keys, no whitespace after delimiters, and utf-8 compatible characters.
    """
    return json.dumps(
        obj,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )


def canonical_json_bytes(obj: Any) -> bytes:
    """Serialize an object to deterministic UTF-8 bytes."""
    return canonical_json_dumps(obj).encode("utf-8")


def canonicalize_schema(schema: Any) -> Any:
    """Recursively canonicalizes a JSON Schema to a deterministic representation.

    Handles:
    - Alphabetical sorting of object dictionary keys.
    - Nested maps in `$defs`, `definitions`, `properties`, `patternProperties`, `dependentSchemas`.
    - Logical composition arrays `allOf`, `anyOf`, `oneOf` (elements sorted deterministically by canonical json).
    - Schema array keywords `items` (single or tuple) and `prefixItems`.
    - Mandatory array normalization: `required` list (sorted and deduplicated).
    - Type list normalization (e.g. `['string', 'null']` sorted).
    - Preserves semantic order for `enum` and `examples`.
    """
    if not isinstance(schema, dict):
        if isinstance(schema, list):
            return [canonicalize_schema(item) for item in schema]
        return schema

    # Create a fresh dict with deterministic alphabetical key order
    canonical: Dict[str, Any] = {}

    # Sort all top-level keys in the schema object
    for key in sorted(schema.keys()):
        val = schema[key]

        if key in ("properties", "patternProperties", "$defs", "definitions", "dependentSchemas"):
            if isinstance(val, dict):
                # Sort sub-schema dictionary keys and recursively canonicalize each sub-schema
                canonical[key] = {
                    sub_key: canonicalize_schema(val[sub_key])
                    for sub_key in sorted(val.keys())
                }
            else:
                canonical[key] = canonicalize_schema(val)

        elif key in ("allOf", "anyOf", "oneOf"):
            if isinstance(val, list):
                # Canonicalize each branch, then sort deterministically by their canonical json
                canon_branches = [canonicalize_schema(branch) for branch in val]
                canon_branches.sort(key=canonical_json_dumps)
                canonical[key] = canon_branches
            else:
                canonical[key] = canonicalize_schema(val)

        elif key == "required":
            if isinstance(val, list):
                # In JSON Schema, required array items have no semantic ordering.
                # Sorting eliminates token drift between permuted required lists.
                canonical[key] = sorted(list(dict.fromkeys(val)))
            else:
                canonical[key] = val

        elif key == "type":
            if isinstance(val, list):
                canonical[key] = sorted(val)
            else:
                canonical[key] = val

        elif key in ("items", "prefixItems"):
            if isinstance(val, list):
                # Tuple validation: maintain list order, canonicalize each item
                canonical[key] = [canonicalize_schema(sub) for sub in val]
            elif isinstance(val, dict):
                canonical[key] = canonicalize_schema(val)
            else:
                canonical[key] = val

        elif key in ("additionalProperties", "not", "if", "then", "else", "contains", "propertyNames"):
            canonical[key] = canonicalize_schema(val)

        elif isinstance(val, dict):
            canonical[key] = canonicalize_schema(val)

        elif isinstance(val, list):
            # Preserves list order for items like enum, examples, etc.
            canonical[key] = [
                canonicalize_schema(item) if isinstance(item, (dict, list)) else item
                for item in val
            ]
        else:
            canonical[key] = val

    return canonical


def normalize_tool_spec(
    tool_spec: Dict[str, Any],
    provider: str = "auto",
) -> Dict[str, Any]:
    """Normalizes an individual tool specification into canonical structure.

    Supports OpenAI, Anthropic, Gemini, or auto-detected schemas.
    """
    if not isinstance(tool_spec, dict):
        return tool_spec

    p = provider.lower()
    if p == "auto":
        if "input_schema" in tool_spec:
            p = "anthropic"
        elif "function" in tool_spec or (tool_spec.get("type") == "function"):
            p = "openai"
        elif "function_declarations" in tool_spec:
            p = "gemini"
        elif "parameters" in tool_spec:
            p = "openai"

    if p == "anthropic":
        return _normalize_anthropic_tool(tool_spec)
    elif p == "openai":
        return _normalize_openai_tool(tool_spec)
    elif p == "gemini":
        return _normalize_gemini_tool(tool_spec)

    # Generic fallback: canonicalize all nested dictionaries and schemas
    return _generic_canonicalize_dict(tool_spec)


def _normalize_anthropic_tool(tool_spec: Dict[str, Any]) -> Dict[str, Any]:
    """Canonicalize Anthropic tool definition:

    {
        "name": ...,
        "description": ...,
        "input_schema": {...},
        "cache_control": {"type": "ephemeral"} (optional)
    }
    """
    canon: Dict[str, Any] = {}
    for key in sorted(tool_spec.keys()):
        val = tool_spec[key]
        if key == "input_schema":
            canon[key] = canonicalize_schema(val)
        elif key == "cache_control" and isinstance(val, dict):
            canon[key] = _generic_canonicalize_dict(val)
        elif isinstance(val, dict):
            canon[key] = canonicalize_schema(val)
        elif isinstance(val, list):
            canon[key] = [
                canonicalize_schema(item) if isinstance(item, (dict, list)) else item
                for item in val
            ]
        else:
            canon[key] = val
    return canon


def _normalize_openai_tool(tool_spec: Dict[str, Any]) -> Dict[str, Any]:
    """Canonicalize OpenAI tool definition:

    {
        "type": "function",
        "function": {
            "name": ...,
            "description": ...,
            "parameters": {...},
            "strict": True (optional)
        }
    }
    or legacy direct function definition.
    """
    canon: Dict[str, Any] = {}
    for key in sorted(tool_spec.keys()):
        val = tool_spec[key]
        if key == "function" and isinstance(val, dict):
            func_canon: Dict[str, Any] = {}
            for f_key in sorted(val.keys()):
                f_val = val[f_key]
                if f_key == "parameters":
                    func_canon[f_key] = canonicalize_schema(f_val)
                elif isinstance(f_val, dict):
                    func_canon[f_key] = canonicalize_schema(f_val)
                elif isinstance(f_val, list):
                    func_canon[f_key] = [
                        canonicalize_schema(item) if isinstance(item, (dict, list)) else item
                        for item in f_val
                    ]
                else:
                    func_canon[f_key] = f_val
            canon[key] = func_canon
        elif key == "parameters":
            canon[key] = canonicalize_schema(val)
        elif isinstance(val, dict):
            canon[key] = canonicalize_schema(val)
        elif isinstance(val, list):
            canon[key] = [
                canonicalize_schema(item) if isinstance(item, (dict, list)) else item
                for item in val
            ]
        else:
            canon[key] = val
    return canon


def _normalize_gemini_tool(tool_spec: Dict[str, Any]) -> Dict[str, Any]:
    """Canonicalize Gemini tool definition:

    {
        "function_declarations": [
            {"name": ..., "description": ..., "parameters": {...}}
        ]
    }
    """
    canon: Dict[str, Any] = {}
    for key in sorted(tool_spec.keys()):
        val = tool_spec[key]
        if key == "function_declarations" and isinstance(val, list):
            decls = []
            for decl in val:
                if isinstance(decl, dict):
                    d_canon: Dict[str, Any] = {}
                    for d_key in sorted(decl.keys()):
                        d_val = decl[d_key]
                        if d_key == "parameters":
                            d_canon[d_key] = canonicalize_schema(d_val)
                        else:
                            d_canon[d_key] = d_val
                    decls.append(d_canon)
                else:
                    decls.append(decl)
            # Sort declarations by function name for deterministic wire ordering
            decls.sort(key=lambda d: d.get("name", "") if isinstance(d, dict) else "")
            canon[key] = decls
        elif key == "parameters":
            canon[key] = canonicalize_schema(val)
        elif isinstance(val, dict):
            canon[key] = canonicalize_schema(val)
        else:
            canon[key] = val
    return canon


def _generic_canonicalize_dict(d: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively canonicalizes dictionary keys in sorted order."""
    result: Dict[str, Any] = {}
    for key in sorted(d.keys()):
        val = d[key]
        if isinstance(val, dict):
            if any(k in val for k in ("properties", "$defs", "definitions", "type")):
                result[key] = canonicalize_schema(val)
            else:
                result[key] = _generic_canonicalize_dict(val)
        elif isinstance(val, list):
            result[key] = [
                canonicalize_schema(item) if isinstance(item, dict) else item
                for item in val
            ]
        else:
            result[key] = val
    return result


def normalize_provider_payload(
    payload: Dict[str, Any],
    provider: str = "auto",
) -> Dict[str, Any]:
    """Normalizes an entire LLM request payload before wire transmission.

    Features:
    - Sorts top-level payload keys.
    - Deterministically orders tools list by tool name to prevent prefix cache misses.
    - Canonicalizes each tool definition / schema.
    - Preserves chronological order of messages / conversation history.
    - Canonicalizes message block fields and tool call arguments.
    """
    if not isinstance(payload, dict):
        return payload

    p = provider.lower()
    if p == "auto":
        if "contents" in payload:
            p = "gemini"
        elif "tools" in payload:
            tools = payload["tools"]
            if isinstance(tools, list) and len(tools) > 0:
                first = tools[0]
                if isinstance(first, dict):
                    if "input_schema" in first:
                        p = "anthropic"
                    elif "function" in first or first.get("type") == "function":
                        p = "openai"
                    elif "function_declarations" in first:
                        p = "gemini"

    canon: Dict[str, Any] = {}

    for key in sorted(payload.keys()):
        val = payload[key]

        if key == "tools":
            if isinstance(val, list):
                normalized_tools = []
                for tool in val:
                    if isinstance(tool, dict):
                        normalized_tools.append(normalize_tool_spec(tool, provider=p))
                    else:
                        normalized_tools.append(tool)

                def tool_sort_key(t: Any) -> str:
                    if not isinstance(t, dict):
                        return canonical_json_dumps(t)
                    if "name" in t:
                        return str(t["name"])
                    if "function" in t and isinstance(t["function"], dict):
                        return str(t["function"].get("name", ""))
                    if "function_declarations" in t and isinstance(t["function_declarations"], list):
                        decls = t["function_declarations"]
                        if decls and isinstance(decls[0], dict):
                            return str(decls[0].get("name", ""))
                    return canonical_json_dumps(t)

                normalized_tools.sort(key=tool_sort_key)
                canon[key] = normalized_tools
            else:
                canon[key] = val

        elif key == "messages":
            if isinstance(val, list):
                canon_messages = []
                for msg in val:
                    if isinstance(msg, dict):
                        c_msg: Dict[str, Any] = {}
                        for m_key in sorted(msg.keys()):
                            m_val = msg[m_key]
                            if m_key == "content" and isinstance(m_val, list):
                                c_blocks = []
                                for block in m_val:
                                    if isinstance(block, dict):
                                        b_dict: Dict[str, Any] = {}
                                        for bk in sorted(block.keys()):
                                            bv = block[bk]
                                            if bk in ("input", "args", "arguments"):
                                                if isinstance(bv, dict):
                                                    b_dict[bk] = _generic_canonicalize_dict(bv)
                                                else:
                                                    b_dict[bk] = bv
                                            else:
                                                b_dict[bk] = bv
                                        c_blocks.append(b_dict)
                                    else:
                                        c_blocks.append(block)
                                c_msg[m_key] = c_blocks
                            elif m_key == "tool_calls" and isinstance(m_val, list):
                                c_tool_calls = []
                                for tc in m_val:
                                    if isinstance(tc, dict):
                                        c_tool_calls.append(_generic_canonicalize_dict(tc))
                                    else:
                                        c_tool_calls.append(tc)
                                c_msg[m_key] = c_tool_calls
                            elif isinstance(m_val, dict):
                                c_msg[m_key] = _generic_canonicalize_dict(m_val)
                            else:
                                c_msg[m_key] = m_val
                        canon_messages.append(c_msg)
                    else:
                        canon_messages.append(msg)
                canon[key] = canon_messages
            else:
                canon[key] = val

        elif key == "contents":
            if isinstance(val, list):
                canon_contents = []
                for item in val:
                    if isinstance(item, dict):
                        canon_contents.append(_generic_canonicalize_dict(item))
                    else:
                        canon_contents.append(item)
                canon[key] = canon_contents
            else:
                canon[key] = val

        elif key == "system":
            if isinstance(val, list):
                canon_system = []
                for block in val:
                    if isinstance(block, dict):
                        canon_system.append(_generic_canonicalize_dict(block))
                    else:
                        canon_system.append(block)
                canon[key] = canon_system
            elif isinstance(val, dict):
                canon[key] = _generic_canonicalize_dict(val)
            else:
                canon[key] = val

        elif isinstance(val, dict):
            canon[key] = _generic_canonicalize_dict(val)
        elif isinstance(val, list):
            canon[key] = [
                _generic_canonicalize_dict(i) if isinstance(i, dict) else i
                for i in val
            ]
        else:
            canon[key] = val

    return canon
