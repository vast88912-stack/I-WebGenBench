import argparse
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TSX_ROOT = ROOT / "outputs" / "tsx"


def run(cmd: list[str], *, cwd: Path) -> tuple[int, str]:
    p = subprocess.run(
        cmd,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return p.returncode, p.stdout


def is_project_dir(p: Path) -> bool:
    return (
        p.is_dir()
        and (p / "package.json").exists()
        and (p / "src" / "main.tsx").exists()
        and (p / "index.html").exists()
    )


def main():
    parser = argparse.ArgumentParser(description="Batch build all outputs/tsx/* Vite projects.")
    parser.add_argument("--path", default="", help="TSX root path (default: outputs/tsx).")
    parser.add_argument("--npm", default="npm", help="NPM executable (default: npm).")
    parser.add_argument("--limit", type=int, default=0, help="Build at most N projects (0 = all).")
    parser.add_argument("--install", action="store_true", help="Run npm install before build (recommended).")
    parser.add_argument("--workers", type=int, default=1, help="Number of projects to build in parallel.")
    parser.add_argument("--skip-existing", action="store_true", help="Skip projects that already have dist/index.html.")
    parser.add_argument("--only-slugs", default="", help="Comma-separated project directory names to build.")
    parser.add_argument(
        "--extra-deps",
        default="recharts@^2.12.0,canvas-confetti@^1.9.3,lucide-react@^0.344.0,framer-motion@^11.0.8,clsx@^2.1.0,tailwind-merge@^2.2.1",
        help="Comma-separated extra runtime dependencies to install when --install is used.",
    )
    args = parser.parse_args()

    tsx_root = Path(args.path).resolve() if args.path else (ROOT / "outputs" / "tsx")
    if not tsx_root.exists():
        raise SystemExit(f"Missing folder: {tsx_root}")

    projects = [p for p in sorted(tsx_root.iterdir()) if is_project_dir(p)]
    if args.only_slugs.strip():
        allow = {item.strip() for item in args.only_slugs.split(",") if item.strip()}
        projects = [p for p in projects if p.name in allow]
    if args.limit and args.limit > 0:
        projects = projects[: args.limit]
    if args.skip_existing:
        projects = [p for p in projects if not (p / "dist" / "index.html").exists()]

    if not projects:
        raise SystemExit(f"No TSX Vite projects found under: {tsx_root}")

    total = len(projects)
    ok = 0
    extra_deps = [item.strip() for item in args.extra_deps.split(",") if item.strip()]

    def build_one(index: int, proj: Path) -> tuple[bool, str]:
        prefix = f"[{index}/{total}] {proj.name}"
        lines = [prefix]
        if args.install:
            code, out = run([args.npm, "install"], cwd=proj)
            (proj / ".build.install.log").write_text(out, encoding="utf-8")
            if code != 0:
                lines.append(f"  [ERR] npm install failed (see {proj}/.build.install.log)")
                return False, "\n".join(lines)
            if extra_deps:
                code, out = run([args.npm, "install", *extra_deps], cwd=proj)
                (proj / ".build.install.extra.log").write_text(out, encoding="utf-8")
                if code != 0:
                    lines.append(f"  [ERR] extra deps install failed (see {proj}/.build.install.extra.log)")
                    return False, "\n".join(lines)

        code, out = run([args.npm, "run", "build"], cwd=proj)
        (proj / ".build.log").write_text(out, encoding="utf-8")
        if code != 0:
            lines.append(f"  [ERR] build failed (see {proj}/.build.log)")
            return False, "\n".join(lines)

        if (proj / "dist" / "index.html").exists():
            lines.append("  [OK] dist/index.html")
            return True, "\n".join(lines)
        else:
            lines.append("  [WARN] build finished but dist/index.html not found")
            return False, "\n".join(lines)

    if args.workers <= 1:
        for i, proj in enumerate(projects, start=1):
            built, output = build_one(i, proj)
            print(output)
            if built:
                ok += 1
    else:
        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
            futs = [ex.submit(build_one, i, proj) for i, proj in enumerate(projects, start=1)]
            for fut in as_completed(futs):
                built, output = fut.result()
                print(output)
                if built:
                    ok += 1

    print(f"Done: {ok}/{total} built.")


if __name__ == "__main__":
    main()
