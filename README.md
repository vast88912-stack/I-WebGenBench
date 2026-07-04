# I-WebGenBench

Minimal code release for **I-WebGenBench: Evaluating Interactivity in LLM-Generated Scientific Web Applications**.

Paper: [paper/2606.00750v1.pdf](paper/2606.00750v1.pdf) | [arXiv](https://arxiv.org/abs/2606.00750)

## What Is Included

This repository keeps only the basic pipeline needed to reproduce the evaluation flow:

- generate prompt specifications from PDFs: `GeneratePrompt.py`
- generate React/TypeScript apps from prompt specifications: `generate_apps.py`
- build generated Vite apps: `tools/build_all_tsx.py`
- probe generated apps with Playwright: `benchmark/runner/run_module_probe*.py`
- aggregate BSR/IR and rule scores: `benchmark/evaluation/codegen_scorer.py`
- optionally run an OpenAI-based visual judge: `benchmark/evaluation/evaluate_codegen_llm.py`

Large generated outputs, datasets, API keys, local environments, and benchmark result folders are intentionally excluded.

## Install

```bash
pip install -r requirements.txt
playwright install chromium
```

## Environment

Copy `.env.example` to `.env` and fill the keys you need.

```bash
cp .env.example .env
```

For prompt generation, set:

```bash
GEMINI_API_KEY=...
```

For app generation, set:

```bash
CODEGEN_PROVIDER=openai_compatible
CODEGEN_API_KEY=...
CODEGEN_BASE_URL=...
CODEGEN_MODEL=...
```

For optional OpenAI judge evaluation, set:

```bash
OPENAI_API_KEY=...
```

## Generate Apps

Generate prompt specifications from paper PDFs:

```bash
python GeneratePrompt.py \
  --pdf-dir Papers \
  --out-dir prompts \
  --workers 4
```

Generate apps from prompt specifications:

```bash
python generate_apps.py \
  --prompts-dir prompts \
  --out-base outputs/tsx \
  --workers 4
```

Build generated apps:

```bash
python tools/build_all_tsx.py --path outputs/tsx --install
```

## Prepare `websites.json`

The evaluator expects a JSON file listing built app entry points. Example:

```json
{
  "websites": [
    {
      "id": "example_site",
      "start_url": "http://127.0.0.1:8000/outputs/tsx/example_site/dist/index.html"
    }
  ]
}
```

## Evaluation Protocol

The evaluation follows the protocol described in the paper:

1. **Build Success Rate (BSR)**: a generated app counts as build-successful if its Vite build produces `dist/index.html`.
2. **Interaction Probe / Interaction Rate (IR)**: for each build-successful app, Playwright loads the rendered page, enumerates visible interactive elements such as buttons, sliders, text inputs, and select menus, maps each element to a canonical semantic action, and uses `MutationObserver` to detect DOM changes over `childList`, `subtree`, and `attributes`. A page is counted as interactive if at least one probed action triggers a DOM mutation.
3. **VLM scoring**: optional judge over screenshots and probe logs, with four qualitative dimensions: Visual Aesthetics (30), Interaction Fidelity (40), Topic/Semantic Alignment (15), and Clarity/Educational Value (10).
4. **Rule/Stability score**: deterministic 5-point score from the probe, penalizing fatal rendering failures and runtime/probe errors.
5. **Final score**: `Visual 30 + Interaction 40 + Topic 15 + Clarity 10 + Rule 5`.

Start a static server from the repository root before probing:

```bash
python -m http.server 8000
```

Run the deterministic Interaction Probe:

```bash
python -m benchmark.runner.run_module_probe_suite \
  --websites benchmark/generated/model_websites/websites.json \
  --results_root benchmark/results \
  --run_id run_001 \
  --headless
```

Aggregate BSR, IR, Rule score, and any available VLM judge results:

```bash
python -m benchmark.evaluation.codegen_scorer \
  --websites benchmark/generated/model_websites/websites.json \
  --results_root benchmark/results \
  --run_id run_001 \
  --prompts_dir prompts \
  --out_table benchmark/generated/score_table.md
```

Optional wrapper for probe + OpenAI VLM judge + scorer:

```bash
python tools/run_full_codegen_eval.py \
  --websites benchmark/generated/model_websites/websites.json \
  --results_root benchmark/results \
  --run_id run_001 \
  --prompts_dir prompts \
  --headless \
  --out_table benchmark/generated/score_table.md
```

Add `--skip_judge` if you only want deterministic BSR/IR/Rule scoring.

## Metrics

- **BSR**: fraction of tasks whose generated app has a built `dist/index.html`.
- **IR**: fraction of tasks where at least one semantically mapped user action triggers a DOM mutation observed by `MutationObserver`.
- **Rule score**: deterministic 5-point page health score based on blank-page checks, color diversity, and runtime/probe errors.

## Prompt Alignment With The Paper

The paper describes a benchmark construction process where paper-derived specifications are curated from PDFs, reviewed by domain experts, and standardized as natural-language task prompts with interaction requirements.

The current `GeneratePrompt.py` is a practical release script for producing such specifications, but it is not an exact reconstruction of the full paper annotation pipeline:

- it uses Gemini to generate specifications directly from PDFs;
- it asks for a fixed educational React/TypeScript app format with a `Module Plan`;
- it is more prescriptive about UI structure, design style, and implementation constraints than the paper's high-level description;
- it does not include the paper's human expert review and filtering stage;
- it should be treated as an approximation for reproducing prompt generation, not as the final curated benchmark prompt set.

If you have the finalized benchmark prompts used in the paper, use those prompts directly for evaluation instead of regenerating them with `GeneratePrompt.py`.

## Citation

```bibtex
@article{dai2026iwebgenbench,
  title={I-WebGenBench: Evaluating Interactivity in LLM-Generated Scientific Web Applications},
  author={Dai, Dasen and Wu, Biao and Fang, Meng and Li, Shuoqi and Wang, Wenhao},
  journal={arXiv preprint arXiv:2606.00750},
  year={2026}
}
```

## License

MIT License.
