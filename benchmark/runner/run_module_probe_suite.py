from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--websites", required=True, help="json listing websites with start_url")
    ap.add_argument("--results_root", required=True, help="benchmark/results root")
    ap.add_argument("--run_id", default="run_001")
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--viewport_width", type=int, default=1024)
    ap.add_argument("--viewport_height", type=int, default=768)
    ap.add_argument("--max_modules", type=int, default=12)
    ap.add_argument("--limit", type=int, default=0, help="only first N sites")
    ap.add_argument("--only_site", type=str, default="", help="run only one site_id (exact match)")
    ap.add_argument(
        "--ignore_module_regex",
        action="append",
        default=[],
        help="pass-through: skip modules whose text matches this regex (case-insensitive). Can repeat.",
    )
    args = ap.parse_args()

    websites = load_json(Path(args.websites))["websites"]
    if args.only_site:
        websites = [w for w in websites if w.get("id") == args.only_site]
    if args.limit and args.limit > 0:
        websites = websites[: args.limit]

    results_root = Path(args.results_root)
    results_root.mkdir(parents=True, exist_ok=True)

    for w in websites:
        sid = w["id"]
        start_url = w["start_url"]
        out_dir = results_root / "codegen_score_v1" / sid / args.run_id
        out_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            sys.executable,
            "-m",
            "benchmark.runner.run_module_probe",
            "--start_url",
            start_url,
            "--site_id",
            sid,
            "--out_dir",
            str(out_dir),
            "--viewport_width",
            str(args.viewport_width),
            "--viewport_height",
            str(args.viewport_height),
            "--max_modules",
            str(args.max_modules),
        ]
        for pat in args.ignore_module_regex or []:
            cmd += ["--ignore_module_regex", str(pat)]
        if args.headless:
            cmd.append("--headless")
        print("RUN ModuleProbe", sid, "->", out_dir)
        r = subprocess.run(cmd, check=False)
        print("DONE ModuleProbe", sid, "exit_code=", r.returncode)


if __name__ == "__main__":
    main()


