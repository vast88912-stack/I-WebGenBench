import argparse
import glob
import hashlib
import os
import shutil
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

try:
    from google import genai
    from google.genai import types
except ImportError:
    print("Missing dependency: google-genai. Install with: pip install google-genai python-dotenv")
    sys.exit(1)


DEFAULT_SYSTEM_INSTRUCTION_TEMPLATE = """
You are an expert AI Researcher and Educational UX Designer specializing in Human-Computer Interaction (HCI) and Scientific Communication.
Your task is to conduct a deep reading of the provided academic paper (PDF) and design an immersive, interactive, educational Web Application (React + TypeScript) that serves as a living visualization of the paper's core contributions.

Core Mandates:
- Accuracy: deeply interpret the mathematical mechanisms, experimental workflows, and causal relationships described in the paper.
- Interactivity: design 3-10 Core Interactive Points, including adjustable parameters, dynamic visual elements, and real-time feedback loops that allow users to explore what-if scenarios.
- Structure: organize the application into 5 distinct modules: 1) Hero/Abstract visual hook, 2) Architecture/Methodology interactive diagram, 3) Core Simulation/Experiment primary playground, 4) Results/Analysis interactive charts, and 5) Conclusion synthesis.

Technical Constraints:
- Output a structured natural language specification focusing on UI components, state variables, and visual logic.
- Do not prescribe a backend or multi-file implementation.
- Ensure the specification is entirely in the requested output language.

Language requirement:
- Write the entire output in {output_language}.
- Even if the source paper is written in Chinese or another language, translate the scientific content and express the app specification in {output_language}.
- All module names, UI labels, chart titles, annotations, and explanatory text in the planned app must be in {output_language}.

Your output MUST strictly follow this text format (no markdown fences, just plain text with these exact headers):

Project: [Paper Title] (Interactive Edition)
Role: Staff Frontend + Research Educator.
Objective: Build a single-file React+TS app that interactively visualizes the core contributions, methodologies, and findings of this paper.
Design: Academic but modern and engaging; crisp light theme with appropriate accent colors for distinct theoretical concepts.
Core Interactive Points:
- [List 3-10 paper-grounded interactive points. For each, name the controllable parameter/user action, dynamic visual component, and expected behavioral response.]
Module Plan:
1) Hero/Abstract: [Detailed visual hook, key paper claim, and first interactive affordance.]
2) Architecture/Methodology: [Interactive diagram of the method/workflow/system.]
3) Core Simulation/Experiment: [Primary playground with state variables and expected responses.]
4) Results/Analysis: [Interactive charts, comparisons, ablations, or measured outcomes.]
5) Conclusion: [Synthesis, takeaway, and optional guided recap.]
Interactions:
- [List 3-5 specific global interactions.]
Technical Notes:
- Use standard HTML Canvas, SVG, or recharts for visual diagrams.
- Use framer-motion for smooth transitions between states.
- Keep state small, use pure functional logic to simulate the paper's algorithms. Do not use a backend.
""".strip()


DEFAULT_USER_PROMPT_TEMPLATE = """
Please read the uploaded paper and generate a structured task specification for building an interactive scientific web application.
Identify the paper's core scientific mechanisms, dynamic processes, controllable parameters, and expected behavioral responses.
The specification should define interaction requirements clearly without prescribing implementation details beyond a single-file React + TypeScript target.
Important: the final specification must be written entirely in {output_language}.
""".strip()


def configure_stdio() -> None:
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:
        pass


def load_env(env_file: str) -> None:
    if load_dotenv is None:
        return
    repo_root = Path(__file__).resolve().parent
    load_dotenv(repo_root / ".env")
    load_dotenv()
    if env_file:
        path = Path(env_file)
        if not path.is_absolute():
            candidate = repo_root / path
            path = candidate if candidate.exists() else path
        load_dotenv(path, override=True)


def get_api_key() -> str:
    api_key = (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or "").strip()
    if not api_key:
        raise SystemExit("Missing GEMINI_API_KEY or GOOGLE_API_KEY in environment.")
    return api_key


def build_system_instruction(output_language: str) -> str:
    return DEFAULT_SYSTEM_INSTRUCTION_TEMPLATE.format(output_language=output_language)


def build_user_prompt(output_language: str) -> str:
    return DEFAULT_USER_PROMPT_TEMPLATE.format(output_language=output_language)


def safe_stem(pdf_path: str, prefix: str = "") -> str:
    original_stem = Path(pdf_path).stem
    stem = f"{prefix}{original_stem}" if prefix else original_stem
    cleaned = "".join(c if ((c.isascii() and c.isalnum()) or c in ("_", "-")) else "_" for c in stem)
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    cleaned = cleaned.strip("_-")
    base = cleaned or "paper"
    digest = hashlib.sha1(original_stem.encode("utf-8")).hexdigest()[:8]
    return f"{base}_{digest}"


def safe_ascii_display_name(pdf_path: str) -> str:
    stem = safe_stem(pdf_path)
    suffix = Path(pdf_path).suffix or ".pdf"
    if not stem:
        stem = "paper"
    return f"{stem}{suffix}"


def generate_prompt_for_pdf(
    *,
    api_key: str,
    pdf_path: str,
    output_dir: str,
    model: str,
    name_prefix: str,
    output_language: str,
) -> str:
    out_name = safe_stem(pdf_path, name_prefix)
    output_path = os.path.join(output_dir, f"{out_name}.txt")
    if os.path.exists(output_path):
        print(f"skip {Path(pdf_path).name} -> {Path(output_path).name}")
        return output_path

    print(f"process {Path(pdf_path).name}")
    last_err: Exception | None = None
    for attempt in range(1, 4):
        client = genai.Client(api_key=api_key)
        uploaded = None
        temp_upload_path = None
        try:
            temp_dir = tempfile.mkdtemp(prefix="paperwebagent_pdf_")
            temp_upload_path = os.path.join(temp_dir, safe_ascii_display_name(pdf_path))
            shutil.copy2(pdf_path, temp_upload_path)
            uploaded = client.files.upload(file=temp_upload_path, config={"display_name": safe_ascii_display_name(pdf_path)})
            response = client.models.generate_content(
                model=model,
                contents=[uploaded, build_user_prompt(output_language)],
                config=types.GenerateContentConfig(
                    system_instruction=build_system_instruction(output_language),
                    temperature=0.3,
                    max_output_tokens=int(os.getenv("PROMPT_MAX_OUTPUT_TOKENS", "8192")),
                ),
            )
            text = (response.text or "").strip()
            if not text:
                raise RuntimeError("empty response")
            os.makedirs(output_dir, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(text)
            print(f"saved {Path(output_path).name}")
            return output_path
        except Exception as e:
            last_err = e
            if attempt < 3:
                print(f"retry {Path(pdf_path).name} attempt {attempt}/3: {e}")
                time.sleep(min(2 * attempt, 6))
            else:
                raise
        finally:
            if uploaded is not None:
                try:
                    client.files.delete(name=uploaded.name)
                except Exception:
                    pass
            if temp_upload_path:
                try:
                    shutil.rmtree(Path(temp_upload_path).parent, ignore_errors=True)
                except Exception:
                    pass
    raise RuntimeError(str(last_err) if last_err else "unknown error")


def main() -> None:
    configure_stdio()
    parser = argparse.ArgumentParser(description="Generate detailed TSX prompts from PDFs using Gemini Files API.")
    parser.add_argument("--pdf-dir", default=str(Path("Papers") / "Chemistry"), help="Directory containing source PDFs.")
    parser.add_argument("--out-dir", default=str(Path("prompts") / "chemistry"), help="Directory for generated prompt txt files.")
    parser.add_argument("--env-file", default="", help="Optional dotenv file to load.")
    parser.add_argument("--model", default=(os.getenv("GEMINI_MODEL") or "gemini-3.1-pro-preview"), help="Gemini model for PDF prompt generation.")
    parser.add_argument("--limit", type=int, default=0, help="Only process the first N PDFs after sorting.")
    parser.add_argument("--workers", type=int, default=4, help="Concurrent PDF prompt workers.")
    parser.add_argument("--name-prefix", default="Chemistry_", help="Prefix to prepend to output prompt basenames.")
    parser.add_argument("--output-language", default=os.getenv("PROMPT_OUTPUT_LANGUAGE", "English"), help="Language to use for the generated prompt specification and planned UI text.")
    parser.add_argument("--allow-failures", action="store_true", help="Do not exit non-zero if some PDFs fail; continue with generated prompts.")
    args = parser.parse_args()

    load_env(args.env_file)
    pdfs = sorted(glob.glob(os.path.join(args.pdf_dir, "*.pdf")))
    if args.limit > 0:
        pdfs = pdfs[: args.limit]
    if not pdfs:
        raise SystemExit(f"No PDFs found in {args.pdf_dir}")

    print(f"found {len(pdfs)} pdfs in {args.pdf_dir}")
    print(f"output dir: {args.out_dir}")
    print(f"model: {args.model}")
    print(f"workers: {args.workers}")
    print(f"output language: {args.output_language}")

    api_key = get_api_key()
    workers = max(1, int(args.workers))

    if workers == 1:
        for pdf in pdfs:
            generate_prompt_for_pdf(
                api_key=api_key,
                pdf_path=pdf,
                output_dir=args.out_dir,
                model=args.model,
                name_prefix=args.name_prefix,
                output_language=args.output_language,
            )
            time.sleep(0.2)
        return

    failures: list[tuple[str, str]] = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {
            ex.submit(
                generate_prompt_for_pdf,
                api_key=api_key,
                pdf_path=pdf,
                output_dir=args.out_dir,
                model=args.model,
                name_prefix=args.name_prefix,
                output_language=args.output_language,
            ): pdf
            for pdf in pdfs
        }
        done = 0
        for fut in as_completed(futures):
            done += 1
            pdf = futures[fut]
            try:
                fut.result()
            except Exception as e:
                failures.append((pdf, str(e)))
                print(f"error {Path(pdf).name}: {e}")
            if done % 5 == 0 or done == len(futures):
                print(f"completed {done}/{len(futures)}")

    if failures:
        print("failures:")
        for pdf, err in failures:
            print(f"- {Path(pdf).name}: {err}")
        if not args.allow_failures:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
