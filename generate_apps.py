import os
import glob
import time
import argparse
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Windows PowerShell consoles may default to legacy code pages (e.g. GBK/CP936),
# which can crash on non-ASCII log output. Best-effort switch to UTF-8.
try:
  if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
  if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
except Exception:
  pass

# Try to import dependencies
try:
  from dotenv import load_dotenv
except ImportError:
  load_dotenv = None


def _getenv_str(key: str) -> str:
  return (os.getenv(key) or "").strip()


def _getenv_float(key: str) -> float | None:
  raw = _getenv_str(key)
  if not raw:
    return None
  try:
    return float(raw)
  except ValueError:
    raise SystemExit(f"Invalid {key}={raw!r} (expected float)")


def _getenv_int(key: str) -> int | None:
  raw = _getenv_str(key)
  if not raw:
    return None
  try:
    return int(raw)
  except ValueError:
    raise SystemExit(f"Invalid {key}={raw!r} (expected int)")


def _getenv_json_dict(key: str) -> dict[str, Any]:
  raw = _getenv_str(key)
  if not raw:
    return {}
  try:
    v = json.loads(raw)
  except Exception as e:
    raise SystemExit(f"Invalid {key} JSON: {e}")
  if not isinstance(v, dict):
    raise SystemExit(f"Invalid {key}: expected a JSON object")
  return v


@dataclass(frozen=True)
class CodegenConfig:
  provider: str
  api_key: str
  base_url: str
  model: str
  temperature: float
  max_output_tokens: int
  extra_body: dict[str, Any]
  use_responses: bool


def load_codegen_config(args: argparse.Namespace) -> CodegenConfig:
  provider = (args.provider or _getenv_str("CODEGEN_PROVIDER") or "gemini").strip().lower()

  if provider == "gemini":
    api_key = _getenv_str("GEMINI_API_KEY") or _getenv_str("GOOGLE_API_KEY")
    model = (args.model or _getenv_str("GEMINI_MODEL") or "gemini-2.0-flash-exp").strip()
    max_output_tokens = _getenv_int("CODEGEN_MAX_OUTPUT_TOKENS") or 65536
    return CodegenConfig(
      provider=provider,
      api_key=api_key,
      base_url="",
      model=model,
      temperature=0.2,
      max_output_tokens=max_output_tokens,
      extra_body={},
      use_responses=False,
    )

  if provider in ("openai", "openai_compatible", "compatible"):
    # Support vendor-specific key env vars used in official docs.
    base_url = (args.base_url or _getenv_str("CODEGEN_BASE_URL") or _getenv_str("OPENAI_BASE_URL")).strip()
    base_url_l = base_url.lower()

    # Hardcoded API keys as fallback
    hardcoded_key = ""
    if "dashscope" in base_url_l:
      hardcoded_key = "YOUR_DASHSCOPE_API_KEY"  # DashScope (北京地域)
    elif "moonshot" in base_url_l:
      hardcoded_key = "YOUR_KIMI_API_KEY"  # Kimi/Moonshot

    vendor_key = ""
    if "moonshot" in base_url_l:
      vendor_key = _getenv_str("MOONSHOT_API_KEY")
    elif "dashscope" in base_url_l:
      vendor_key = _getenv_str("DASHSCOPE_API_KEY")
    elif "deepseek" in base_url_l:
      vendor_key = _getenv_str("DEEPSEEK_API_KEY")
    elif "minimax" in base_url_l:
      vendor_key = _getenv_str("MINIMAX_API_KEY") or _getenv_str("OPENAI_API_KEY")

    api_key = (
      args.api_key
      or _getenv_str("CODEGEN_API_KEY")
      or vendor_key
      or _getenv_str("OPENAI_API_KEY")
      or _getenv_str("MOONSHOT_API_KEY")
      or _getenv_str("DASHSCOPE_API_KEY")
      or _getenv_str("DEEPSEEK_API_KEY")
      or hardcoded_key
    ).strip()
    model = (args.model or _getenv_str("CODEGEN_MODEL")).strip()
    extra_body = _getenv_json_dict("CODEGEN_EXTRA_BODY")
    use_responses = (_getenv_str("CODEGEN_USE_RESPONSES").lower() in ("1", "true", "yes"))

    # Default temperature: low for codegen, except Kimi K2.x on Moonshot often enforces temperature=1.
    t_env = _getenv_float("CODEGEN_TEMPERATURE")
    if t_env is not None:
      temperature = t_env
    else:
      temperature = 1.0 if "kimi" in model.lower() else 0.2

    max_output_tokens = _getenv_int("CODEGEN_MAX_OUTPUT_TOKENS") or 65536
    if "deepseek" in base_url.lower() and "v4" not in model.lower():
      max_output_tokens = min(max_output_tokens, 5500)  # Conservative limit for older DeepSeek API's 8192 max

    return CodegenConfig(
      provider="openai_compatible",
      api_key=api_key,
      base_url=base_url,
      model=model,
      temperature=temperature,
      max_output_tokens=max_output_tokens,
      extra_body=extra_body,
      use_responses=use_responses,
    )

  # Zhipu AI (GLM) via zai-sdk
  if provider in ("zai", "zhipu", "glm"):
    api_key = (
      args.api_key
      or _getenv_str("ZHIPU_API_KEY")
      or _getenv_str("ZAI_API_KEY")
      or _getenv_str("CODEGEN_API_KEY")
    ).strip()
    model = (args.model or _getenv_str("CODEGEN_MODEL") or "glm-5").strip()
    # Zhipu supports thinking={"type":"enabled"} (per official example). Allow override via JSON.
    thinking = _getenv_json_dict("CODEGEN_THINKING") or {"type": "enabled"}

    t_env = _getenv_float("CODEGEN_TEMPERATURE")
    temperature = t_env if t_env is not None else 0.2
    max_output_tokens = _getenv_int("CODEGEN_MAX_OUTPUT_TOKENS") or 65536

    return CodegenConfig(
      provider="zai",
      api_key=api_key,
      base_url="",
      model=model,
      temperature=temperature,
      max_output_tokens=max_output_tokens,
      extra_body={"thinking": thinking},
      use_responses=False,
    )

  raise SystemExit(f"Unsupported --provider={provider}. Use gemini, openai_compatible, or zai")


def ensure_gemini_imports() -> tuple[object, object]:
  try:
    from google import genai  # type: ignore
    from google.genai import types  # type: ignore

    return genai, types
  except ImportError:
    raise SystemExit("Missing dependency: google-genai. Install: pip install google-genai python-dotenv")


def ensure_openai_imports() -> object:
  try:
    from openai import OpenAI  # type: ignore

    return OpenAI
  except ImportError:
    raise SystemExit("Missing dependency: openai. Install: pip install openai python-dotenv")


def ensure_zai_imports() -> object:
  try:
    from zai import ZhipuAiClient  # type: ignore

    return ZhipuAiClient
  except ImportError:
    raise SystemExit("Missing dependency: zai-sdk. Install: pip install zai-sdk")


def call_codegen_llm(*, cfg: CodegenConfig, prompt: str) -> str:
  system_instruction = (
    "Return ONLY the TSX code for src/App.tsx. "
    "Do NOT include markdown fences. "
    "The code MUST include a default export (export default ...) so it can be imported as App. "
    "Build an interactive educational research webpage, not a static landing page. "
    "The first viewport must already show meaningful scientific content and clear interactive affordances such as tabs, sliders, toggles, charts, animated diagrams, or clickable modules. "
    "Implement all five modules described in the specification with functional state management and user-driven data exploration. "
    "Do not return an empty shell, sparse hero page, or navigation-only layout. "
    "Use English for all user-facing UI text unless the prompt explicitly requires another language."
  )

  if cfg.provider == "gemini":
    if not cfg.api_key:
      raise SystemExit("Missing GEMINI_API_KEY (or GOOGLE_API_KEY) in environment.")
    import time as _time
    import os as _os
    genai, types = ensure_gemini_imports()
    # Clear proxy env vars so dead local proxies don't block direct HTTPS connections.
    _saved_proxies = {k: _os.environ.pop(k, None) for k in ("HTTPS_PROXY", "HTTP_PROXY", "https_proxy", "http_proxy")}
    client = genai.Client(api_key=cfg.api_key)
    # Use streaming to avoid proxy timeout on long generations (30-60s).
    # Retry up to 4 times with exponential backoff on SSL/network errors.
    last_exc: Exception | None = None
    for _attempt in range(4):
      if _attempt > 0:
        _delay = 10 * (2 ** (_attempt - 1))
        print(f"  [gemini] Retry {_attempt}/3 after SSL/network error, sleeping {_delay}s...")
        _time.sleep(_delay)
      try:
        chunks = []
        for chunk in client.models.generate_content_stream(
          model=cfg.model,
          contents=f"{system_instruction}\n\n{prompt}",
          config=types.GenerateContentConfig(temperature=cfg.temperature, max_output_tokens=cfg.max_output_tokens),
        ):
          if chunk.text:
            chunks.append(chunk.text)
        result = "".join(chunks).strip()
        for k, v in _saved_proxies.items():
          if v is not None:
            _os.environ[k] = v
        return result
      except Exception as _exc:
        last_exc = _exc
        _msg = str(_exc)
        if "SSL" in _msg or "EOF" in _msg or "reset by peer" in _msg.lower() or "connection" in _msg.lower():
          continue  # retry on network errors
        for k, v in _saved_proxies.items():
          if v is not None:
            _os.environ[k] = v
        raise
    for k, v in _saved_proxies.items():
      if v is not None:
        _os.environ[k] = v
    raise RuntimeError(f"LLM API call failed after 4 attempts: {last_exc}") from last_exc

  if cfg.provider == "zai":
    if not cfg.api_key:
      raise SystemExit("Missing ZHIPU_API_KEY (or ZAI_API_KEY / CODEGEN_API_KEY) in environment.")
    if not cfg.model:
      raise SystemExit("Missing CODEGEN_MODEL (e.g. glm-5).")

    ZhipuAiClient = ensure_zai_imports()
    client = ZhipuAiClient(api_key=cfg.api_key)

    thinking = (cfg.extra_body or {}).get("thinking") or None
    resp = client.chat.completions.create(
      model=cfg.model,
      messages=[
        {"role": "system", "content": system_instruction},
        {"role": "user", "content": prompt},
      ],
      thinking=thinking,
      max_tokens=cfg.max_output_tokens,
      temperature=cfg.temperature,
    )

    msg = resp.choices[0].message
    content = getattr(msg, "content", None)
    if content is None and isinstance(msg, dict):
      content = msg.get("content")
    return (content or "").strip()

  # OpenAI-compatible
  if not cfg.api_key:
    raise SystemExit("Missing CODEGEN_API_KEY (or OPENAI_API_KEY) in environment.")
  if not cfg.model:
    raise SystemExit("Missing CODEGEN_MODEL. Set it to the model id you want to use.")

  OpenAI = ensure_openai_imports()
  kwargs = {"api_key": cfg.api_key}
  if cfg.base_url:
    kwargs["base_url"] = cfg.base_url
  # Avoid indefinite hangs on flaky provider links.
  timeout_s = (_getenv_float("CODEGEN_TIMEOUT_SECONDS") or 120.0)
  if "deepseek" in (cfg.base_url or "").lower():
    timeout_s = max(timeout_s, 300.0)
  kwargs["timeout"] = timeout_s
  client = OpenAI(**kwargs)

  # Prefer Responses API if explicitly enabled AND the upstream supports it.
  if cfg.use_responses and hasattr(client, "responses"):
    try:
      resp = client.responses.create(
        model=cfg.model,
        input=[
          {"role": "system", "content": system_instruction},
          {"role": "user", "content": prompt},
        ],
        temperature=cfg.temperature,
        max_output_tokens=cfg.max_output_tokens,
        extra_body=(cfg.extra_body or None),
      )
      return (getattr(resp, "output_text", "") or "").strip()
    except Exception:
      # Fall back to Chat Completions for compatibility.
      pass

  chat_kwargs: dict[str, Any] = {
    "model": cfg.model,
    "messages": [
      {"role": "system", "content": system_instruction},
      {"role": "user", "content": prompt},
    ],
    "temperature": cfg.temperature,
    "extra_body": (cfg.extra_body or None),
    "stream": False,
  }
  # Newer GPT-5 chat endpoints reject max_tokens and require max_completion_tokens.
  if cfg.model.lower().startswith("gpt-5"):
    chat_kwargs["max_completion_tokens"] = cfg.max_output_tokens
    # Move reasoning_effort from extra_body to top-level param
    eb = cfg.extra_body or {}
    if "reasoning_effort" in eb:
      chat_kwargs["reasoning_effort"] = eb.pop("reasoning_effort")
      if not eb:
        chat_kwargs["extra_body"] = None
  else:
    chat_kwargs["max_tokens"] = cfg.max_output_tokens
  resp = client.chat.completions.create(**chat_kwargs)
  return (resp.choices[0].message.content or "").strip()


# Configuration
PROMPTS_DIR = "./prompts"
OUTPUT_BASE = "./outputs/tsx"

# Vite Boilerplate Templates
BOILERPLATE = {
    "package.json": """{
  "name": "paper-web-agent-app",
  "private": true,
  "version": "0.0.0",
  "type": "module",
  "scripts": {
    "dev": "vite",
    "build": "vite build",
    "preview": "vite preview"
  },
  "dependencies": {
    "lucide-react": "^0.344.0",
    "react": "^18.2.0",
    "react-dom": "^18.2.0",
    "framer-motion": "^11.0.8",
    "recharts": "^2.12.0",
    "canvas-confetti": "^1.9.3",
    "clsx": "^2.1.0",
    "tailwind-merge": "^2.2.1"
  },
  "devDependencies": {
    "@types/react": "^18.2.64",
    "@types/react-dom": "^18.2.21",
    "@vitejs/plugin-react": "^4.2.1",
    "autoprefixer": "^10.4.18",
    "postcss": "^8.4.35",
    "tailwindcss": "^3.4.1",
    "typescript": "^5.2.2",
    "vite": "^5.1.4"
  }
}""",
    "vite.config.ts": """import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  base: './', 
})""",
    "tsconfig.json": """{
  "compilerOptions": {
    "target": "ES2020",
    "useDefineForClassFields": true,
    "lib": ["ES2020", "DOM", "DOM.Iterable"],
    "module": "ESNext",
    "skipLibCheck": true,
    "moduleResolution": "bundler",
    "allowImportingTsExtensions": true,
    "resolveJsonModule": true,
    "isolatedModules": true,
    "noEmit": true,
    "jsx": "react-jsx",
    "strict": true,
    "noUnusedLocals": false,
    "noUnusedParameters": false,
    "noFallthroughCasesInSwitch": true
  },
  "include": ["src"],
  "references": [{ "path": "./tsconfig.node.json" }]
}""",
    "tsconfig.node.json": """{
  "compilerOptions": {
    "composite": true,
    "skipLibCheck": true,
    "module": "ESNext",
    "moduleResolution": "bundler",
    "allowSyntheticDefaultImports": true
  },
  "include": ["vite.config.ts"]
}""",
    "index.html": """<!doctype html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>Interactive Paper App</title>
  </head>
  <body class="bg-slate-950 text-slate-100">
    <div id="root"></div>
    <script type="module" src="/src/main.tsx"></script>
  </body>
</html>""",
    "postcss.config.js": """export default {
  plugins: {
    tailwindcss: {},
    autoprefixer: {},
  },
}""",
    "tailwind.config.js": """/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {},
  },
  plugins: [],
}""",
    "src/index.css": """@tailwind base;
@tailwind components;
@tailwind utilities;

body {
  margin: 0;
  padding: 0;
  background-color: #020617; /* Slate 950 */
  color: #f1f5f9; /* Slate 100 */
}""",
    "src/main.tsx": """import React from 'react'
import ReactDOM from 'react-dom/client'
import App from './App.tsx'
import './index.css'

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
)"""
}

def setup_project_structure(base_path):
    """Creates the Vite project boilerplate in the target directory."""
    if not os.path.exists(base_path):
        os.makedirs(base_path)
    
    # Create src directory
    os.makedirs(os.path.join(base_path, "src"), exist_ok=True)

    # Write boilerplate files
    for filename, content in BOILERPLATE.items():
        file_path = os.path.join(base_path, filename)
        if not os.path.exists(file_path):
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)

def clean_code_fence(code: str) -> str:
    """Removes markdown code fences if present."""
    text = code.strip()

    # Some providers prepend chain-of-thought in <think>...</think>.
    # Keep only the post-think payload when present.
    if "<think>" in text and "</think>" in text:
        text = text.split("</think>")[-1].strip()

    # Prefer explicit fenced code blocks when available.
    blocks = re.findall(r"```(?:tsx|typescript|ts|jsx|javascript)?\n(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if blocks:
        return max(blocks, key=len).strip()

    # If response mixes narration with code, trim to likely code start.
    for marker in ("import React", "import {", "function App", "const App"):
        idx = text.find(marker)
        if idx >= 0:
            return text[idx:].strip()

    # Fallback to original behavior.
    text = re.sub(r'^```(tsx|typescript|ts)?\n', '', text)
    text = re.sub(r'\n```$', '', text)
    return text.strip()


def ensure_export_default(code: str) -> tuple[str, bool]:
    """Best-effort fix to ensure a default export exists for App.

    Returns: (fixed_code, changed)
    """
    if "export default" in code:
        return code, False

    # Prefer transforming existing App function declaration.
    if re.search(r"\bexport\s+function\s+App\b", code):
        fixed = re.sub(r"\bexport\s+function\s+App\b", "export default function App", code, count=1)
        return fixed, True

    if re.search(r"\bfunction\s+App\b", code):
        fixed = re.sub(r"\bfunction\s+App\b", "export default function App", code, count=1)
        return fixed, True

    # If App is declared as const/class, append default export.
    if re.search(r"\bconst\s+App\b", code) or re.search(r"\bclass\s+App\b", code):
        return code.rstrip() + "\n\nexport default App\n", True

    return code, False

def generate_app(cfg: CodegenConfig, prompt_path: str, output_dir: str) -> None:
    filename = os.path.basename(prompt_path)
    name = os.path.splitext(filename)[0]
    app_slug = name.lower().replace("_", "-")
    target_dir = os.path.join(output_dir, app_slug)
    
    print(f"🚀 Generating app for: {name}")
    print(f"   Target: {target_dir}")

    # 1. Setup Vite boilerplate
    setup_project_structure(target_dir)

    # 2. Read Prompt
    with open(prompt_path, "r", encoding="utf-8") as f:
        prompt_content = f.read()

    # 3. Call LLM
    print(f"   ⏳ Calling {cfg.provider} model={cfg.model}... (this may take 30-60s)")
    max_attempts = _getenv_int("CODEGEN_RETRIES") or 3
    last_err: str | None = None

    for attempt in range(1, max_attempts + 1):
      try:
        prompt_for_attempt = prompt_content
        if attempt > 1:
          prompt_for_attempt = (
            "IMPORTANT: Your output must be valid TSX for src/App.tsx and MUST contain a default export (export default ...).\n\n"
            + prompt_content
          )
        text = call_codegen_llm(cfg=cfg, prompt=prompt_for_attempt)
      except Exception as e:
        last_err = str(e)
        print(f"   ❌ API Error (attempt {attempt}/{max_attempts}): {last_err}")
        if attempt < max_attempts:
          time.sleep(min(2.0 * attempt, 6.0))
          continue
        return

      if not (text or "").strip():
        last_err = "Empty response"
        print(f"   ❌ Error: Empty response from API (attempt {attempt}/{max_attempts}).")
        if attempt < max_attempts:
          time.sleep(min(1.0 * attempt, 4.0))
          continue
        return

      # 4. Extract and Save Code
      code = clean_code_fence(text)
      code, changed = ensure_export_default(code)
      if changed:
        print("   🩹 Auto-fixed missing export default.")
      if "export default" not in code:
        last_err = "Missing export default"
        print(f"   ❌ Error: Missing export default (attempt {attempt}/{max_attempts}).")
        if attempt < max_attempts:
          time.sleep(min(1.0 * attempt, 4.0))
          continue
        print("   ❌ Giving up (invalid TSX for App.tsx).")
        return

      app_tsx_path = os.path.join(target_dir, "src", "App.tsx")
      with open(app_tsx_path, "w", encoding="utf-8") as f:
        f.write(code)

      print("   ✅ Saved src/App.tsx")
      return

    if last_err:
      print(f"   ❌ Failed after {max_attempts} attempts: {last_err}")

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate TSX apps from prompts.")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of apps to generate")
    parser.add_argument("--offset", type=int, default=0, help="Skip first N prompts")
    parser.add_argument("--workers", type=int, default=1, help="Parallel workers (default 1). Use 20 for high parallelism.")
    parser.add_argument("--sleep", type=float, default=0.0, help="Sleep seconds between requests (sequential mode).")
    parser.add_argument(
      "--only",
      default="",
      help="Comma-separated prompt basenames to run (e.g. Algorithm_Dynamic_Programming,Physics_Optics_Lab).",
    )
    parser.add_argument("--prompts-dir", default="", help="Prompt directory (default ./prompts)")
    parser.add_argument("--out-base", default="", help="Output base directory (default ./outputs/tsx)")
    parser.add_argument("--env-file", default="", help="Optional dotenv file to load (e.g. .env.qwen)")
    parser.add_argument("--provider", default="", help="gemini | openai_compatible (or set CODEGEN_PROVIDER)")
    parser.add_argument("--model", default="", help="Model id (overrides GEMINI_MODEL or CODEGEN_MODEL)")
    parser.add_argument("--base-url", default="", help="OpenAI-compatible base_url (or set CODEGEN_BASE_URL/OPENAI_BASE_URL)")
    parser.add_argument("--api-key", default="", help="Override API key (or set env vars)")
    args = parser.parse_args()

    # Load env (repo root .env preferred). If --env-file is provided, it overrides.
    if load_dotenv is not None:
      repo_root = Path(__file__).resolve().parent
      load_dotenv(repo_root / ".env")
      load_dotenv()
      if args.env_file:
        env_path = Path(args.env_file)
        if not env_path.is_absolute():
          # Make relative env paths work regardless of current working directory.
          candidate = repo_root / env_path
          env_path = candidate if candidate.exists() else env_path
        load_dotenv(env_path, override=True)

    cfg = load_codegen_config(args)

    prompts_dir = (args.prompts_dir or os.getenv("PROMPTS_DIR") or PROMPTS_DIR).strip()
    output_base = (args.out_base or os.getenv("OUTPUT_BASE") or OUTPUT_BASE).strip()
    workers = max(1, int(args.workers or 1))
    sleep_s = float(args.sleep or 0.0)

    prompts = sorted(glob.glob(os.path.join(prompts_dir, "*.txt")))
    if not prompts:
        print(f"No prompts found in {prompts_dir}")
        return

    to_process = prompts[args.offset:]
    if args.limit > 0:
        to_process = to_process[: args.limit]

    if args.only:
      wanted = {p.strip() for p in args.only.split(",") if p.strip()}
      if wanted:
        filtered: list[str] = []
        for p in to_process:
          base = os.path.splitext(os.path.basename(p))[0]
          if base in wanted:
            filtered.append(p)
        to_process = filtered

    print(f"📋 Found {len(prompts)} prompts. Processing {len(to_process)}...")
    print("-" * 50)

    if workers == 1:
        for i, p in enumerate(to_process):
            print(f"[{i+1}/{len(to_process)}] Processing {os.path.basename(p)}")
            generate_app(cfg, p, output_base)
            print("-" * 50)
            if sleep_s > 0:
                time.sleep(sleep_s)
        return

    print(f"⚡ Parallel mode: workers={workers} (prompts={len(to_process)})")
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(generate_app, cfg, p, output_base) for p in to_process]
        done = 0
        for fut in as_completed(futs):
            done += 1
            try:
                fut.result()
            except Exception as e:
                print(f"   ❌ Worker error: {e}")
            if done % 5 == 0 or done == len(futs):
                print(f"...completed {done}/{len(futs)}")

if __name__ == "__main__":
    main()
