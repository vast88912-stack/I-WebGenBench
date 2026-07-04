from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PromptExpectations:
    site_id: str
    expected_module_names: list[str]
    expected_module_count: int | None
    requires_canvas: bool
    requires_inspector: bool
    requires_game_like: bool


_BULLET_RE = re.compile(r"^\s*(?:-|\*|\d+\.|\d+\))\s+(.*\S)\s*$")


def _find_section_lines(lines: list[str], header_pat: re.Pattern[str]) -> tuple[int, int] | None:
    """
    Returns [start, end) indices for a section starting at a header line, ending at next blank+header-ish.
    Very lightweight / heuristic on purpose.
    """
    start = None
    for i, ln in enumerate(lines):
        if header_pat.search(ln):
            start = i + 1
            break
    if start is None:
        return None
    end = len(lines)
    for j in range(start, len(lines)):
        if lines[j].strip().startswith("###") or lines[j].strip().startswith("## "):
            end = j
            break
    return start, end


def parse_prompt_spec(site_id: str, text: str) -> PromptExpectations:
    """
    Parse a prompt spec (prompts/*.txt) into a minimal expectation set used for deterministic scoring.

    Important: This is intentionally heuristic and stable across runs.
    """
    t = text
    low = t.lower()
    lines = t.splitlines()

    # Module list: try "Module Plan" section first (top-level bullets only)
    module_names: list[str] = []
    sec = _find_section_lines(lines, re.compile(r"\bmodule plan\b", re.I))
    if sec:
        s, e = sec
        for ln in lines[s:e]:
            indent = len(ln) - len(ln.lstrip(" "))
            m = _BULLET_RE.match(ln)
            if not m:
                continue
            # keep top-level bullets; nested bullets are usually per-module details/checklists
            if indent > 2:
                continue
            item = m.group(1).strip()
            # stop if bullets end
            if not item:
                continue
            # keep only a short title (before ':' if present)
            title = item.split(":", 1)[0].strip()
            if not (1 < len(title) <= 60):
                continue
            # Only keep lines that actually look like module headings.
            if not re.search(r"\b(module|landing|dashboard)\b", title, flags=re.I):
                continue
            module_names.append(title)

    # Fallback: infer "Landing + N Modules"
    expected_count = None
    m = re.search(r"landing\s*\+\s*(\d+)\s*modules", low)
    if m:
        try:
            expected_count = int(m.group(1)) + 1
        except Exception:
            expected_count = None
    else:
        m2 = re.search(r"\b(\d+)\s*modules\b", low)
        if m2:
            try:
                n = int(m2.group(1))
                if 3 <= n <= 12:
                    expected_count = n
            except Exception:
                expected_count = None

    if module_names and expected_count is None:
        # Many specs list modules including landing; accept as expected count if reasonable.
        if 3 <= len(module_names) <= 12:
            expected_count = len(module_names)

    # Requirements: interpret "Acceptance Checklist" / MUST-like wording
    acceptance = _find_section_lines(lines, re.compile(r"\bacceptance checklist\b|\bacceptance\b", re.I))
    acc_text = "\n".join(lines[acceptance[0] : acceptance[1]]) if acceptance else t
    acc_low = acc_text.lower()

    def must_have(keyword: str) -> bool:
        if keyword.lower() not in low:
            return False
        # prefer "must/mandatory/required" near the keyword (or in acceptance section)
        return bool(re.search(rf"(must|mandatory|required).{{0,120}}{re.escape(keyword.lower())}", acc_low, flags=re.I)) or (
            keyword.lower() in acc_low and ("must" in acc_low or "required" in acc_low or "mandatory" in acc_low)
        )

    requires_canvas = must_have("canvas") or must_have("webgl")
    requires_inspector = ("inspector" in acc_low) or ("hyperparameters" in acc_low) or must_have("inspector")
    requires_game_like = ("challenge" in acc_low) or ("game" in acc_low) or ("score" in acc_low and "steps" in acc_low)

    return PromptExpectations(
        site_id=site_id,
        expected_module_names=module_names,
        expected_module_count=expected_count,
        requires_canvas=requires_canvas,
        requires_inspector=requires_inspector,
        requires_game_like=requires_game_like,
    )


def load_prompt_expectations(site_id: str, prompts_dir: Path) -> PromptExpectations:
    path = prompts_dir / f"{site_id}.txt"
    if not path.exists():
        return PromptExpectations(
            site_id=site_id,
            expected_module_names=[],
            expected_module_count=None,
            requires_canvas=False,
            requires_inspector=False,
            requires_game_like=False,
        )
    return parse_prompt_spec(site_id, path.read_text(encoding="utf-8"))


