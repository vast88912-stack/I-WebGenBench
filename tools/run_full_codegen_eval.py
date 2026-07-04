#!/usr/bin/env python3
"""
End-to-end "codegen quality" evaluation for a websites suite:
  1) Filter to build-success sites (dist/index.html exists locally)
  2) Run module probe suite (Playwright) on filtered sites
  3) Run GPT judge (OpenAI VLM) in parallel per site (up to N workers)
  4) Aggregate scores with codegen_scorer; missing sites count as 0 (optional)

This uses existing benchmark logic (no re-implementation):
  - benchmark.runner.run_module_probe_suite
  - benchmark.evaluation.evaluate_codegen_llm
  - benchmark.evaluation.codegen_scorer

Notes:
  - You must serve repo root with http.server so start_url works.
  - This script does NOT start/stop the HTTP server for you.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


ROOT = Path(__file__).resolve().parents[1]


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def dump_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def url_to_local_path(start_url: str) -> Path:
    u = urlparse(start_url)
    if not u.path:
        return ROOT / "__invalid__"
    if u.scheme == "file":
        p = Path(u.path)
        if p.is_absolute():
            return p
        return (ROOT / p).resolve()
    rel = u.path.lstrip("/")
    return ROOT / Path(rel)


@dataclass(frozen=True)
class Site:
    site_id: str
    start_url: str
    local_index: Path


def filter_sites(websites_path: Path) -> tuple[list[Site], list[Site]]:
    obj = load_json(websites_path)
    websites = obj.get("websites") or []
    ok: list[Site] = []
    missing: list[Site] = []
    for w in websites:
        sid = str(w.get("id") or "")
        url = str(w.get("start_url") or "")
        if not sid or not url:
            continue
        local = url_to_local_path(url)
        s = Site(site_id=sid, start_url=url, local_index=local)
        (ok if local.exists() else missing).append(s)
    return ok, missing


def run_cmd(cmd: list[str], *, cwd: Path | None = None, env: dict[str, str] | None = None) -> int:
    p = subprocess.run(cmd, cwd=str(cwd) if cwd else None, env=env, text=True)
    return int(p.returncode)


def run_module_probe(websites_path: Path, results_root: Path, run_id: str, headless: bool) -> int:
    cmd = [
        sys.executable,
        "-m",
        "benchmark.runner.run_module_probe_suite",
        "--websites",
        str(websites_path),
        "--results_root",
        str(results_root),
        "--run_id",
        str(run_id),
    ]
    if headless:
        cmd.append("--headless")
    return run_cmd(cmd, cwd=ROOT)


def judge_one_site(
    *,
    site_id: str,
    run_id: str,
    results_root: Path,
    prompts_dir: Path,
    model: str,
    trials: int,
    last_k_per_module: int,
    out_prefix: str,
    retries: int,
    jitter_s: float,
) -> tuple[str, bool, str]:
    run_dir = results_root / "codegen_score_v1" / site_id / run_id
    log_path = run_dir / f".{out_prefix}.judge.log"
    run_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        "-m",
        "benchmark.evaluation.evaluate_codegen_llm",
        "--run_dir",
        str(run_dir),
        "--site_id",
        str(site_id),
        "--prompts_dir",
        str(prompts_dir),
        "--model",
        str(model),
        "--trials",
        str(int(trials)),
        "--last_k_per_module",
        str(int(last_k_per_module)),
        "--out_prefix",
        str(out_prefix),
    ]

    last_out = ""
    for attempt in range(1, retries + 1):
        # small jitter to avoid synchronized bursts
        if jitter_s > 0:
            time.sleep(random.random() * jitter_s)
        p = subprocess.run(
            cmd,
            cwd=str(ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        last_out = p.stdout or ""
        log_path.write_text(last_out, encoding="utf-8", errors="replace")
        if p.returncode == 0:
            return site_id, True, str(log_path.relative_to(ROOT))

        # Basic backoff for throttling-like errors
        if any(x in last_out for x in ["429", "rate limit", "RateLimit", "Too Many Requests"]):
            time.sleep(min(60.0, 2.5 * attempt))
            continue
        time.sleep(min(10.0, 1.0 * attempt))

    return site_id, False, str(log_path.relative_to(ROOT))


def run_gpt_judge_parallel(
    *,
    site_ids: list[str],
    run_id: str,
    results_root: Path,
    prompts_dir: Path,
    model: str,
    trials: int,
    last_k_per_module: int,
    out_prefix: str,
    workers: int,
    retries: int,
    jitter_s: float,
) -> tuple[list[str], list[str]]:
    ok: list[str] = []
    bad: list[str] = []
    workers = max(1, int(workers))

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = []
        for sid in site_ids:
            futs.append(
                ex.submit(
                    judge_one_site,
                    site_id=sid,
                    run_id=run_id,
                    results_root=results_root,
                    prompts_dir=prompts_dir,
                    model=model,
                    trials=trials,
                    last_k_per_module=last_k_per_module,
                    out_prefix=out_prefix,
                    retries=retries,
                    jitter_s=jitter_s,
                )
            )
        for fut in as_completed(futs):
            sid, passed, log_rel = fut.result()
            if passed:
                ok.append(sid)
            else:
                bad.append(sid)
            print(f"[judge] {sid}: {'OK' if passed else 'ERR'} (log {log_rel})")

    ok.sort()
    bad.sort()
    return ok, bad


def run_codegen_scorer(
    websites_path: Path,
    results_root: Path,
    run_id: str,
    prompts_dir: Path,
    out_table: Path,
    llm_missing_zero: bool,
    openai_site_prefix: str,
) -> int:
    cmd = [
        sys.executable,
        "-m",
        "benchmark.evaluation.codegen_scorer",
        "--websites",
        str(websites_path),
        "--results_root",
        str(results_root),
        "--run_id",
        str(run_id),
        "--prompts_dir",
        str(prompts_dir),
        "--out_table",
        str(out_table),
        "--openai_site_prefix",
        str(openai_site_prefix),
    ]
    if llm_missing_zero:
        cmd.append("--llm_missing_zero")
    return run_cmd(cmd, cwd=ROOT)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--websites", required=True, help="websites json (20 sites)")
    ap.add_argument("--results_root", default="benchmark/results", help="benchmark/results root")
    ap.add_argument("--run_id", required=True, help="run id under benchmark/results/codegen_score_v1/<site_id>/<run_id>")
    ap.add_argument("--prompts_dir", default="prompts", help="prompts directory")

    ap.add_argument("--headless", action="store_true", help="run Playwright headless")
    ap.add_argument("--skip_probe", action="store_true", help="skip module probe suite")
    ap.add_argument("--skip_judge", action="store_true", help="skip GPT judge")
    ap.add_argument("--skip_score", action="store_true", help="skip codegen_scorer")

    ap.add_argument("--openai_model", default=os.getenv("OPENAI_VLM_MODEL", "gpt-4.1"))
    ap.add_argument("--openai_trials", type=int, default=int(os.getenv("OPENAI_VLM_TRIALS", "3")))
    ap.add_argument("--openai_last_k", type=int, default=int(os.getenv("OPENAI_VLM_LAST_K_PER_MODULE", "4")))
    ap.add_argument("--openai_prefix", default="openai_gpt4v")
    ap.add_argument("--judge_workers", type=int, default=10, help="parallel workers for judge (e.g. 30)")
    ap.add_argument("--judge_retries", type=int, default=2)
    ap.add_argument("--judge_jitter_s", type=float, default=0.8)

    ap.add_argument("--llm_missing_zero", action="store_true", help="missing GPT judge scores treated as 0 in table")
    ap.add_argument("--out_table", default="", help="output markdown table path")
    args = ap.parse_args()

    websites_path = (ROOT / args.websites).resolve() if not Path(args.websites).is_absolute() else Path(args.websites)
    results_root = (ROOT / args.results_root).resolve()
    prompts_dir = (ROOT / args.prompts_dir).resolve()

    ok_sites, missing_sites = filter_sites(websites_path)
    ok_ids = [s.site_id for s in ok_sites]
    print(f"[sites] total={len(ok_sites)+len(missing_sites)} ok_dist={len(ok_sites)} missing_dist={len(missing_sites)}")
    if missing_sites:
        print("[sites] missing dist (will be 0-score in aggregation):")
        for s in missing_sites:
            print(f"  - {s.site_id}: {s.local_index.relative_to(ROOT)}")

    filtered_path = ROOT / "benchmark" / "generated" / "tmp_websites" / f"websites_filtered_{args.run_id}.json"
    suite_obj = load_json(websites_path)
    suite_obj["websites"] = [{"id": s.site_id, "start_url": s.start_url} for s in ok_sites]
    dump_json(filtered_path, suite_obj)
    print(f"[sites] filtered websites: {filtered_path.relative_to(ROOT)}")

    if not args.skip_probe:
        code = run_module_probe(filtered_path, results_root, args.run_id, headless=bool(args.headless))
        if code != 0:
            raise SystemExit(code)

    if not args.skip_judge:
        if not (os.getenv("OPENAI_API_KEY") or os.getenv("CODEGEN_API_KEY")):
            raise SystemExit("Missing OPENAI_API_KEY (required for GPT judge).")
        ok_j, bad_j = run_gpt_judge_parallel(
            site_ids=ok_ids,
            run_id=args.run_id,
            results_root=results_root,
            prompts_dir=prompts_dir,
            model=args.openai_model,
            trials=args.openai_trials,
            last_k_per_module=args.openai_last_k,
            out_prefix=args.openai_prefix,
            workers=args.judge_workers,
            retries=max(1, int(args.judge_retries)),
            jitter_s=float(args.judge_jitter_s),
        )
        print(f"[judge] ok={len(ok_j)} err={len(bad_j)}")

    if not args.skip_score:
        if args.out_table:
            out_table = Path(args.out_table)
            if not out_table.is_absolute():
                out_table = (ROOT / out_table).resolve()
        else:
            out_table = (ROOT / "benchmark" / "generated" / f"codegen_score_table_{args.run_id}.md").resolve()
        code = run_codegen_scorer(
            websites_path=websites_path,
            results_root=results_root,
            run_id=args.run_id,
            prompts_dir=prompts_dir,
            out_table=out_table,
            llm_missing_zero=bool(args.llm_missing_zero),
            openai_site_prefix=args.openai_prefix,
        )
        if code != 0:
            raise SystemExit(code)
        try:
            rel = out_table.relative_to(ROOT)
            print("[OK] table:", rel)
        except Exception:
            print("[OK] table:", out_table)


if __name__ == "__main__":
    main()
