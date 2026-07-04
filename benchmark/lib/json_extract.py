from __future__ import annotations

import json
import re
from typing import Any


_FENCE_RE = re.compile(r"```(?:json)?\s*([\s\S]*?)\s*```", re.IGNORECASE)


def _try_load(s: str) -> Any | None:
    try:
        return json.loads(s)
    except Exception:
        return None


def extract_first_json_value(text: str) -> Any:
    """
    Best-effort extraction of the first JSON value from a messy model output:
    - supports ```json fences
    - supports leading/trailing junk text
    - supports objects or arrays
    Returns a parsed JSON value (dict/list/str/number/bool/null) or {} as fallback.
    """
    if not text:
        return {}

    t = text.strip()

    # 1) Prefer fenced JSON blocks
    m = _FENCE_RE.search(t)
    if m:
        inner = m.group(1).strip()
        v = _try_load(inner)
        if v is not None:
            return v

    # 2) Try whole text
    v = _try_load(t)
    if v is not None:
        return v

    # 3) Scan for a balanced {...} or [...] region
    # Find earliest '{' or '[' and attempt to parse progressively.
    starts = [i for i in [t.find("{"), t.find("[")] if i != -1]
    if not starts:
        return {}
    start = min(starts)
    s = t[start:]

    # Simple bracket matching
    stack: list[str] = []
    in_str = False
    esc = False
    for i, ch in enumerate(s):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
            continue
        if ch in "{[":
            stack.append(ch)
        elif ch in "}]":
            if not stack:
                continue
            open_ch = stack.pop()
            if (open_ch == "{" and ch != "}") or (open_ch == "[" and ch != "]"):
                # mismatched; give up
                return {}
            if not stack:
                cand = s[: i + 1]
                vv = _try_load(cand)
                if vv is not None:
                    return vv
                # keep searching (rare)
    return {}


def coerce_json_object(value: Any) -> dict[str, Any]:
    """
    Coerce a parsed JSON value to a dict for downstream evaluators:
    - dict -> dict
    - list with one dict -> dict
    - list -> {"_raw_list": ...}
    - scalar -> {"_raw": ...}
    """
    if isinstance(value, dict):
        return value
    if isinstance(value, list):
        if len(value) == 1 and isinstance(value[0], dict):
            return value[0]
        return {"_raw_list": value}
    return {"_raw": value}


