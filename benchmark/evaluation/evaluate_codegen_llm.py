from __future__ import annotations

import argparse
import base64
import json
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field, ValidationError
from rich.console import Console
from tenacity import retry, stop_after_attempt, wait_exponential

from benchmark.lib.benchlib import mean_std, write_json
from benchmark.lib.json_extract import coerce_json_object, extract_first_json_value


console = Console()


class Rubric(BaseModel):
    spec_adherence: float = Field(ge=0, le=1)
    module_navigation: float = Field(ge=0, le=1)
    interactivity: float = Field(ge=0, le=1)
    visual_quality: float = Field(ge=0, le=1)
    stability: float = Field(ge=0, le=1)


class ModuleOut(BaseModel):
    module_id: str
    module_name: str
    score_0_1: float = Field(ge=0, le=1)
    notes: str = ""


class SiteEvalOut(BaseModel):
    site_id: str
    passed: bool
    score_0_1: float = Field(ge=0, le=1)
    confidence_0_1: float = Field(ge=0, le=1)
    failure_reasons_topk: list[str] = Field(default_factory=list, max_length=5)
    rubric: Rubric
    modules: list[ModuleOut] = Field(default_factory=list)
    explanation_short: str = ""
    recommended_fix: str = ""


def load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def b64_png(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def module_images(module_dir: Path, k: int) -> list[Path]:
    """
    Prefer a few key screenshots if present; otherwise take last-k by name.
    """
    screens_dir = module_dir / "screens"
    if not screens_dir.exists():
        return []
    prefer = ["step_000.png", "after_slider.png", "after_play_wait.png", "after_button_clicks.png", "after_pan.png", "after_zoom.png"]
    picked: list[Path] = []
    for name in prefer:
        p = screens_dir / name
        if p.exists():
            picked.append(p)
    if picked:
        return picked[:k] if k > 0 else []
    allp = sorted(screens_dir.glob("*.png"))
    return allp[-k:] if k > 0 else []


def build_content_parts(user_prompt_text: str, grouped_images: list[tuple[str, list[Path]]]) -> list[dict[str, Any]]:
    """
    Build a mixed text+image content list so the model can see module grouping separators.
    """
    parts: list[dict[str, Any]] = [{"type": "text", "text": user_prompt_text}]
    for header, imgs in grouped_images:
        parts.append({"type": "text", "text": header})
        for img in imgs:
            parts.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_png(img)}"}})
    return parts


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8))
def call_openai_json(client: OpenAI, model: str, system_prompt: str, content_parts: list[dict[str, Any]]) -> dict[str, Any]:
    if hasattr(client, "responses"):
        resp = client.responses.create(
            model=model,
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content_parts},
            ],
            response_format={"type": "json_object"},
        )
        parsed = extract_first_json_value(resp.output_text or "")
        return coerce_json_object(parsed)
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content_parts},
        ],
        response_format={"type": "json_object"},
        temperature=0,
    )
    text = resp.choices[0].message.content or "{}"
    parsed = extract_first_json_value(text)
    return coerce_json_object(parsed)


def normalize_min_schema(raw: Any, site_id: str) -> dict[str, Any]:
    raw = coerce_json_object(raw)
    if "site_id" not in raw:
        raw["site_id"] = site_id
    if "confidence_0_1" not in raw:
        raw["confidence_0_1"] = 0.5
    if "score_0_1" not in raw:
        raw["score_0_1"] = 1.0 if bool(raw.get("passed")) else 0.0
    # Ensure rubric exists before deriving passed.
    if "failure_reasons_topk" not in raw or not isinstance(raw.get("failure_reasons_topk"), list):
        raw["failure_reasons_topk"] = []
    if "rubric" not in raw or not isinstance(raw.get("rubric"), dict):
        s = float(raw.get("score_0_1", 0.0) or 0.0)
        raw["rubric"] = {
            "spec_adherence": s,
            "module_navigation": s,
            "interactivity": s,
            "visual_quality": s,
            "stability": s,
        }
    if "modules" not in raw or not isinstance(raw.get("modules"), list):
        raw["modules"] = []
    # Derive passed deterministically if missing.
    if "passed" not in raw:
        s = float(raw.get("score_0_1", 0.0) or 0.0)
        spec = float((raw.get("rubric") or {}).get("spec_adherence", s) or 0.0)
        raw["passed"] = bool(s >= 0.5 and spec >= 0.5)
    return raw


def eval_one_trial(
    client: OpenAI,
    model: str,
    system_prompt: str,
    user_template: str,
    site_id: str,
    run_dir: Path,
    prompt_spec_text: str,
    last_k_per_module: int,
) -> tuple[bool, dict[str, Any]]:
    site_probe_path = run_dir / "site_probe.json"
    site_probe = load_json(site_probe_path) if site_probe_path.exists() else {}

    modules_dir = run_dir / "modules"
    module_probe_paths = sorted(modules_dir.glob("*/module_probe.json")) if modules_dir.exists() else []
    module_probes: list[dict[str, Any]] = []
    grouped_images: list[tuple[str, list[Path]]] = []
    for p in module_probe_paths:
        try:
            pr = load_json(p)
        except Exception:
            continue
        module_probes.append(pr)
        module_dir = p.parent
        name = str(pr.get("module_name") or module_dir.name)
        mid = str(pr.get("module_id") or module_dir.name)
        imgs = module_images(module_dir, k=last_k_per_module)
        grouped_images.append((f"\n\n=== MODULE {mid}: {name} (images={len(imgs)}) ===\n", imgs))

    # Truncate module probes to keep prompt stable
    packed = json.dumps(module_probes, ensure_ascii=False)
    if len(packed) > 120_000:
        packed = packed[:120_000] + "...TRUNCATED..."

    user_prompt = (
        user_template.replace("{{SITE_ID}}", site_id)
        .replace("{{PROMPT_SPEC_TEXT}}", prompt_spec_text[:120_000])
        .replace("{{SITE_PROBE_JSON}}", json.dumps(site_probe, ensure_ascii=False)[:80_000])
        .replace("{{MODULE_PROBES_JSON}}", packed)
    )
    content_parts = build_content_parts(user_prompt, grouped_images)
    raw = call_openai_json(client, model=model, system_prompt=system_prompt, content_parts=content_parts)
    raw = normalize_min_schema(raw, site_id=site_id)
    try:
        parsed = SiteEvalOut.model_validate(raw)
        passed = bool(parsed.score_0_1 >= 0.5 and parsed.rubric.spec_adherence >= 0.5)
        raw["_normalized_passed"] = passed
        return passed, {"ok": True, "raw": raw}
    except ValidationError as e:
        return False, {"ok": False, "error": "schema_validation_error", "raw": raw, "detail": str(e)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True, help=".../benchmark/results/codegen_score_v1/<site>/<run_id>")
    ap.add_argument("--site_id", required=True)
    ap.add_argument("--prompts_dir", required=True, help="repo/prompts")
    ap.add_argument("--model", default="gpt-4.1")
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--last_k_per_module", type=int, default=4)
    ap.add_argument("--system_prompt", default=str(Path(__file__).parent / "prompts" / "system_codegen_eval_v1.txt"))
    ap.add_argument("--user_prompt", default=str(Path(__file__).parent / "prompts" / "user_codegen_eval_v1.md"))
    ap.add_argument("--out_prefix", default="openai_gpt4v")
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    load_dotenv(repo_root / ".env")
    load_dotenv()

    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise SystemExit("Missing OPENAI_API_KEY in environment.")

    client = OpenAI(api_key=api_key)
    run_dir = Path(args.run_dir)
    site_id = args.site_id
    prompts_dir = Path(args.prompts_dir)
    prompt_path = prompts_dir / f"{site_id}.txt"
    prompt_spec_text = prompt_path.read_text(encoding="utf-8") if prompt_path.exists() else ""

    system_prompt = load_text(Path(args.system_prompt))
    user_template = load_text(Path(args.user_prompt))

    runs_path = run_dir / f"{args.out_prefix}_eval_runs.jsonl"
    if runs_path.exists():
        runs_path.unlink()

    trial_passes: list[int] = []
    trial_scores: list[float] = []
    trial_conf: list[float] = []
    failure_reasons: list[str] = []

    for t in range(args.trials):
        console.print(f"[cyan]Codegen LLM eval[/cyan] {site_id} trial {t+1}/{args.trials}")
        passed, payload = eval_one_trial(
            client=client,
            model=args.model,
            system_prompt=system_prompt,
            user_template=user_template,
            site_id=site_id,
            run_dir=run_dir,
            prompt_spec_text=prompt_spec_text,
            last_k_per_module=args.last_k_per_module,
        )
        out = {"site_id": site_id, "trial": t, "passed": passed, **payload}
        with runs_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(out, ensure_ascii=False) + "\n")

        # Even if schema validation failed, keep scalar score/conf to avoid "0 due to schema brittleness".
        if isinstance(payload.get("raw"), dict):
            raw = payload["raw"]
            p = bool(raw.get("_normalized_passed", passed))
            trial_passes.append(1 if p else 0)
            trial_scores.append(float(raw.get("score_0_1", 0.0) or 0.0))
            trial_conf.append(float(raw.get("confidence_0_1", 0.0) or 0.0))
            fr = raw.get("failure_reasons_topk") or []
            if isinstance(fr, list):
                for r in fr:
                    if isinstance(r, str) and r:
                        failure_reasons.append(r)

    pass_rate = sum(trial_passes) / max(1, len(trial_passes))
    passed_final = pass_rate >= 0.5
    score_mu, score_sd = mean_std(trial_scores)
    conf_mu, conf_sd = mean_std(trial_conf)
    freq: dict[str, int] = {}
    for r in failure_reasons:
        freq[r] = freq.get(r, 0) + 1
    topk = [k for k, _ in sorted(freq.items(), key=lambda kv: (-kv[1], kv[0]))[:5]]

    site_result = {
        "site_id": site_id,
        "passed": passed_final,
        "consistency": pass_rate,
        "score_mean": score_mu,
        "score_std": score_sd,
        "confidence_mean": conf_mu,
        "confidence_std": conf_sd,
        "failure_reasons_topk": topk,
        "model": args.model,
        "trials": args.trials,
        "last_k_per_module": args.last_k_per_module,
    }
    write_json(run_dir / f"{args.out_prefix}_site_result.json", site_result)
    console.print(f"[bold green]Codegen LLM site[/bold green] score_mean={score_mu:.3f} passed={passed_final}")


if __name__ == "__main__":
    main()


