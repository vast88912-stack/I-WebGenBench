from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from benchmark.generation.prompt_rulebook import PromptExpectations, load_prompt_expectations
from benchmark.lib.benchlib import ensure_dir, now_ms


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else 1.0 if x > 1.0 else x


def log_norm(x: float, lo: float, hi: float) -> float:
    """
    Map x in [lo, hi] to [0,1] in log space.
    """
    if x <= 0:
        return 0.0
    x = max(lo, min(hi, x))
    return clamp01((math.log(x) - math.log(lo)) / (math.log(hi) - math.log(lo)))


@dataclass(frozen=True)
class ModuleScore:
    module_id: str
    module_name: str
    score: float
    components: dict[str, float]
    dom_stats: dict[str, Any]
    metrics: dict[str, Any]
    errors: list[str]
    baseline: str | None


def score_module(probe: dict[str, Any]) -> ModuleScore:
    module_id = str(probe.get("module_id") or "")
    module_name = str(probe.get("module_name") or module_id)
    errors = list(probe.get("errors") or [])
    baseline = probe.get("baseline") or (probe.get("screens") or [None])[0]

    dom = probe.get("dom_stats_after") or probe.get("dom_stats") or {}
    metrics = probe.get("metrics") or {}
    health = probe.get("health") or {}

    # Calculate Rule Score (Max 5 points)
    # Base: 5 points.
    # Deductions: -5 for fatal (blank/crash), -2 for console errors
    has_text = bool(health.get("has_text")) if "has_text" in health else True
    root_children = int(health.get("root_child_count") or 0) if "root_child_count" in health else 1
    dark = float(metrics.get("dark_ratio") or 0.0)
    uniq = float(metrics.get("unique_color_count") or 0.0)
    
    fatal = (root_children == 0 and not has_text) or (dark >= 0.99) or (uniq <= 3)
    
    rule_score = 5.0
    if fatal:
        rule_score = 0.0
    elif errors:
        rule_score = max(0.0, rule_score - 2.0) # Penalty for console errors

    return ModuleScore(
        module_id=module_id,
        module_name=module_name,
        score=rule_score, # For module-level, we just return the rule score for now
        components={
            "rule_score": rule_score
        },
        dom_stats=dom,
        metrics=metrics,
        errors=errors,
        baseline=str(baseline) if baseline else None,
    )


def apply_prompt_adjustments(
    base_score: float,
    site_id: str,
    expectations: PromptExpectations,
    module_scores: list[ModuleScore],
    all_probes: list[dict[str, Any]],
) -> tuple[float, dict[str, Any]]:
    # In the new 100-point VLM-driven protocol, the VLM's "Topic" score 
    # handles missing modules, canvas checks, and semantic alignment.
    # Therefore, we no longer apply hard-coded percentage penalties here.
    # We return the base (Rule) score unmodified.
    return base_score, {"note": "Hard-coded penalties disabled in favor of VLM Topic scoring"}

def calculate_bsr_ir(run_dir: Path) -> dict[str, float]:
    # BSR: this run exists only for build-successful apps probed by Playwright.
    # Missing run directories are handled as zero by the caller.
    # IR follows the paper: at least one semantically mapped action triggers
    # a DOM mutation observed by MutationObserver.
    
    modules_dir = run_dir / "modules"
    module_probe_paths = sorted(modules_dir.glob("*/module_probe.json"))
    
    if not module_probe_paths:
        return {"bsr": 0.0, "ir": 0.0}
    
    # If we have probes, it built.
    bsr = 1.0 
    
    # Interaction Rate: at least one DOM mutation across any probed module.
    has_interaction = False
    for p in module_probe_paths:
        try:
            probe = load_json(p)
            interaction_probe = probe.get("interaction_probe") or {}
            if bool(interaction_probe.get("has_dom_mutation")):
                has_interaction = True
                break
            metrics = probe.get("metrics") or {}
            if bool(metrics.get("has_dom_mutation_after_actions")):
                has_interaction = True
                break
        except Exception:
            continue
            
    ir = 1.0 if has_interaction else 0.0
    return {"bsr": bsr, "ir": ir}


def load_vlm_components_from_runs(run_dir: Path, prefix: str) -> dict[str, Any]:
    """
    Load paper-aligned VLM judge runs and convert 0-1 rubric values to
    Visual(30), Interaction(40), Topic(15), and Clarity(10).
    """
    runs_path = run_dir / f"{prefix}_eval_runs.jsonl"
    if not runs_path.exists():
        return {"total": 0.0, "components": {}}

    buckets: dict[str, list[float]] = {
        "visual": [],
        "interaction": [],
        "topic": [],
        "clarity": [],
    }
    for line in runs_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
            raw = row.get("raw") if isinstance(row.get("raw"), dict) else {}
            rubric = raw.get("rubric") if isinstance(raw.get("rubric"), dict) else {}
            buckets["visual"].append(clamp01(float(rubric.get("visual_quality", 0.0) or 0.0)) * 30.0)
            buckets["interaction"].append(clamp01(float(rubric.get("interactivity", 0.0) or 0.0)) * 40.0)
            buckets["topic"].append(clamp01(float(rubric.get("spec_adherence", 0.0) or 0.0)) * 15.0)
            buckets["clarity"].append(clamp01(float(rubric.get("module_navigation", 0.0) or 0.0)) * 10.0)
        except Exception:
            continue

    components = {
        name: {"score": (sum(values) / len(values)) if values else 0.0}
        for name, values in buckets.items()
    }
    return {"total": sum(v["score"] for v in components.values()), "components": components}


def score_site_run(run_dir: Path, site_id: str, prompts_dir: Path, openai_site_prefix: str = "openai_gpt4v") -> dict[str, Any]:
    modules_dir = run_dir / "modules"
    module_probe_paths = sorted(modules_dir.glob("*/module_probe.json"))
    probes: list[dict[str, Any]] = []
    for p in module_probe_paths:
        try:
            probes.append(load_json(p))
        except Exception:
            continue

    module_scores = [score_module(pr) for pr in probes]
    module_scores.sort(key=lambda m: (0 if m.module_id == "_landing" else 1, m.module_name.lower()))

    # Calculate average Rule Score (out of 5)
    rule_score_avg = sum(m.score for m in module_scores) / len(module_scores) if module_scores else 0.0
    
    expectations = load_prompt_expectations(site_id, prompts_dir)
    rule_final, adj = apply_prompt_adjustments(rule_score_avg, site_id, expectations, module_scores, probes)

    # Extract BSR and IR
    compile_metrics = calculate_bsr_ir(run_dir)

    vlm = load_vlm_components_from_runs(run_dir, openai_site_prefix)
    vlm_total = float(vlm.get("total", 0.0) or 0.0)
    vlm_components = vlm.get("components") or {}

    # Final 100-point score
    final_100_score = rule_final + vlm_total

    out = {
        "site_id": site_id,
        "run_dir": str(run_dir),
        "ts_ms": now_ms(),
        "metrics": {
            "BSR": compile_metrics["bsr"],
            "IR": compile_metrics["ir"]
        },
        "score_breakdown": {
            "Rule (5)": rule_final,
            "Visual (30)": vlm_components.get("visual", {}).get("score", 0.0),
            "Interaction (40)": vlm_components.get("interaction", {}).get("score", 0.0),
            "Topic (15)": vlm_components.get("topic", {}).get("score", 0.0),
            "Clarity (10)": vlm_components.get("clarity", {}).get("score", 0.0),
        },
        "final_100_score": final_100_score,
        "prompt_expectations": asdict(expectations),
        "modules": [asdict(m) for m in module_scores],
    }
    return out


def write_table(site_scores: list[dict[str, Any]], out_path: Path, *, llm_missing_zero: bool) -> None:
    site_ids = [s["site_id"] for s in site_scores]
    header = ["Metric"] + site_ids + ["Overall Mean"]
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(["---"] * len(header)) + " |"]

    def row(name: str, key_path: list[str], is_percentage: bool = False) -> None:
        vals = []
        cells = [name]
        for s in site_scores:
            try:
                # Traverse nested dict if key_path has multiple elements
                v = s
                for k in key_path:
                    v = v.get(k, {}) if isinstance(v, dict) else 0.0
                v = float(v) if not isinstance(v, dict) else 0.0
            except Exception:
                v = 0.0
            vals.append(v)
            if is_percentage:
                cells.append(f"{v*100:.1f}%")
            else:
                cells.append(f"{v:.1f}")
                
        overall = (sum(vals) / len(vals)) if vals else 0.0
        if is_percentage:
            cells.append(f"{overall*100:.1f}%")
        else:
            cells.append(f"{overall:.1f}")
        lines.append("| " + " | ".join(cells) + " |")

    # Final 100-point score
    row("Final Score (/100)", ["final_100_score"])
    
    # Breakdown
    row("Rule (/5)", ["score_breakdown", "Rule (5)"])
    row("Visual (/30)", ["score_breakdown", "Visual (30)"])
    row("Interaction (/40)", ["score_breakdown", "Interaction (40)"])
    row("Topic (/15)", ["score_breakdown", "Topic (15)"])
    row("Clarity (/10)", ["score_breakdown", "Clarity (10)"])
    
    # Build and Probe Metrics
    row("BSR (Build Success)", ["metrics", "BSR"], is_percentage=True)
    row("IR (Interaction Rate)", ["metrics", "IR"], is_percentage=True)

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--websites", required=True, help="benchmark/inputs/websites/tsx_20.json")
    ap.add_argument("--results_root", required=True, help="benchmark/results root")
    ap.add_argument("--run_id", default="run_001")
    ap.add_argument("--prompts_dir", default=str(Path(__file__).parents[2] / "prompts"))
    ap.add_argument("--out_table", default=str(Path(__file__).parents[1] / "generated" / "codegen_score_table.md"))
    ap.add_argument("--only_site", default="", help="score only one site_id (exact match)")
    ap.add_argument("--openai_site_prefix", default="openai_gpt4v", help="prefix for <prefix>_site_result.json")
    ap.add_argument(
        "--llm_missing_zero",
        action="store_true",
        help="Treat missing LLM judge scores as 0.0%% instead of NA (useful when build/probe failed sites should count as 0).",
    )
    args = ap.parse_args()

    websites = load_json(Path(args.websites))["websites"]
    if args.only_site:
        websites = [w for w in websites if w.get("id") == args.only_site]
    results_root = Path(args.results_root)
    prompts_dir = Path(args.prompts_dir)
    out_table = Path(args.out_table)
    ensure_dir(out_table.parent)

    all_scores: list[dict[str, Any]] = []
    for w in websites:
        sid = w["id"]
        run_dir = results_root / "codegen_score_v1" / sid / args.run_id
        if not run_dir.exists():
            # still produce a placeholder score
            sc = {
                "site_id": sid,
                "run_dir": str(run_dir),
                "ts_ms": now_ms(),
                "base_score": 0.0,
                "final_score": 0.0,
                "error": "missing_run_dir",
            }
            all_scores.append(sc)
            continue
        sc = score_site_run(
            run_dir,
            site_id=sid,
            prompts_dir=prompts_dir,
            openai_site_prefix=args.openai_site_prefix,
        )
        # per-site output required by spec
        (run_dir / "site_score.json").write_text(json.dumps(sc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

        # Attach optional LLM judge scores if present
        openai_p = run_dir / f"{args.openai_site_prefix}_site_result.json"
        if openai_p.exists():
            try:
                r = load_json(openai_p)
                sc["openai_vlm_score_mean"] = float(r.get("score_mean", 0.0) or 0.0)
            except Exception:
                pass
        all_scores.append(sc)

    write_table(all_scores, out_table, llm_missing_zero=bool(args.llm_missing_zero))
    print("DONE. Table at:", out_table)


if __name__ == "__main__":
    main()


