from __future__ import annotations

import argparse
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from playwright.sync_api import Page, sync_playwright
from rich.console import Console

from benchmark.lib.benchlib import (
    append_jsonl,
    dark_ratio,
    ensure_dir,
    image_diff_ratio,
    now_ms,
    unique_color_count,
    write_json,
)


console = Console()


def safe_sleep(ms: int) -> None:
    time.sleep(ms / 1000.0)


def screenshot(page: Page, path: Path) -> None:
    page.screenshot(path=str(path), full_page=True)


def body_inner_text(page: Page, limit: int = 8000) -> str:
    try:
        txt = page.evaluate("() => document.body ? (document.body.innerText || '') : ''") or ""
    except Exception:
        return ""
    txt = re.sub(r"\s+\n", "\n", txt).strip()
    return txt[:limit]


def extract_dom_stats(page: Page) -> dict[str, Any]:
    try:
        return page.evaluate(
            """() => {
              const q = (sel) => document.querySelectorAll(sel).length;
              const buttonTexts = Array.from(document.querySelectorAll('button'))
                .slice(0, 30)
                .map(b => (b.innerText||'').trim())
                .filter(Boolean);
              return {
                button_count: q('button'),
                link_count: q('a'),
                range_count: q('input[type="range"]'),
                input_count: q('input'),
                select_count: q('select'),
                textarea_count: q('textarea'),
                canvas_count: q('canvas'),
                svg_count: q('svg'),
                button_text_sample: buttonTexts,
              };
            }"""
        )
    except Exception:
        return {
            "button_count": 0,
            "link_count": 0,
            "range_count": 0,
            "input_count": 0,
            "select_count": 0,
            "textarea_count": 0,
            "canvas_count": 0,
            "svg_count": 0,
            "button_text_sample": [],
        }


def page_health(page: Page) -> dict[str, Any]:
    """
    Cheap health checks to catch blank/crashed pages without relying on model judgment.
    """
    try:
        return page.evaluate(
            """() => {
              const root = document.querySelector('#root') || document.body;
              const rootChildCount = root ? (root.children ? root.children.length : 0) : 0;
              const bodyText = document.body ? (document.body.innerText || '').trim() : '';
              const hasText = bodyText.length > 40;
              return { root_child_count: rootChildCount, has_text: hasText, body_text_len: bodyText.length };
            }"""
        )
    except Exception as e:
        return {"error": str(e)}


def enumerate_interactive_elements(page: Page, max_n: int = 40) -> list[dict[str, Any]]:
    """
    Paper-aligned element enumeration: collect visible interactive elements
    with HTML semantics and bounding boxes for robust audit/debugging.
    """
    try:
        items = page.evaluate(
            """(maxN) => {
              const selectors = [
                'button',
                'a[href]',
                'input',
                'select',
                'textarea',
                '[role="button"]',
                '[role="tab"]',
                '[role="slider"]',
                '[contenteditable="true"]'
              ];
              const nodes = [];
              const seen = new Set();
              for (const selector of selectors) {
                for (const el of document.querySelectorAll(selector)) {
                  if (seen.has(el)) continue;
                  seen.add(el);
                  const r = el.getBoundingClientRect();
                  const st = window.getComputedStyle(el);
                  if (!r || r.width < 2 || r.height < 2) continue;
                  if (st.display === 'none' || st.visibility === 'hidden' || st.opacity === '0') continue;
                  if (r.bottom < 0 || r.right < 0 || r.top > innerHeight || r.left > innerWidth) continue;
                  const tag = (el.tagName || '').toLowerCase();
                  const type = (el.getAttribute('type') || '').toLowerCase();
                  const role = (el.getAttribute('role') || '').toLowerCase();
                  const label = (
                    el.innerText ||
                    el.getAttribute('aria-label') ||
                    el.getAttribute('title') ||
                    el.getAttribute('placeholder') ||
                    el.getAttribute('name') ||
                    ''
                  ).replace(/\\s+/g, ' ').trim().slice(0, 80);
                  nodes.push({
                    index: nodes.length,
                    tag,
                    type,
                    role,
                    label,
                    bbox: {
                      x: Math.round(r.left),
                      y: Math.round(r.top),
                      width: Math.round(r.width),
                      height: Math.round(r.height),
                      cx: Math.round(r.left + r.width / 2),
                      cy: Math.round(r.top + r.height / 2)
                    }
                  });
                  if (nodes.length >= maxN) return nodes;
                }
              }
              return nodes;
            }""",
            max_n,
        )
        return list(items or [])
    except Exception:
        return []


def probe_semantic_action(page: Page, element_index: int, wait_ms: int = 350) -> dict[str, Any]:
    """
    Execute one canonical action and observe DOM mutations.

    This mirrors the paper's Interaction Probe: semantic action mapping plus
    MutationObserver over childList/subtree/attributes.
    """
    return page.evaluate(
        """async ({ elementIndex, waitMs }) => {
          const selectors = [
            'button',
            'a[href]',
            'input',
            'select',
            'textarea',
            '[role="button"]',
            '[role="tab"]',
            '[role="slider"]',
            '[contenteditable="true"]'
          ];
          const collect = () => {
            const nodes = [];
            const seen = new Set();
            for (const selector of selectors) {
              for (const el of document.querySelectorAll(selector)) {
                if (seen.has(el)) continue;
                seen.add(el);
                const r = el.getBoundingClientRect();
                const st = window.getComputedStyle(el);
                if (!r || r.width < 2 || r.height < 2) continue;
                if (st.display === 'none' || st.visibility === 'hidden' || st.opacity === '0') continue;
                if (r.bottom < 0 || r.right < 0 || r.top > innerHeight || r.left > innerWidth) continue;
                nodes.push(el);
              }
            }
            return nodes;
          };

          const nodes = collect();
          const el = nodes[elementIndex];
          if (!el) return { ok: false, reason: 'element_not_found', mutation_count: 0 };

          const before = document.documentElement.outerHTML.length;
          const records = [];
          const observer = new MutationObserver((mutations) => {
            for (const m of mutations) {
              records.push({
                type: m.type,
                target: (m.target && m.target.nodeName) || '',
                added: m.addedNodes ? m.addedNodes.length : 0,
                removed: m.removedNodes ? m.removedNodes.length : 0,
                attributeName: m.attributeName || ''
              });
            }
          });
          observer.observe(document.documentElement, {
            childList: true,
            subtree: true,
            attributes: true
          });

          const tag = (el.tagName || '').toLowerCase();
          const type = (el.getAttribute('type') || '').toLowerCase();
          const role = (el.getAttribute('role') || '').toLowerCase();
          let action = 'click';
          let error = '';
          try {
            if (tag === 'input' && type === 'range') {
              action = 'set_range_midpoint';
              const min = Number.parseFloat(el.min || '0');
              const max = Number.parseFloat(el.max || '100');
              const value = min + (max - min) * 0.65;
              el.value = String(value);
              el.dispatchEvent(new Event('input', { bubbles: true }));
              el.dispatchEvent(new Event('change', { bubbles: true }));
            } else if (tag === 'input' && (type === 'checkbox' || type === 'radio')) {
              action = 'toggle_checked';
              el.click();
            } else if (tag === 'input' || tag === 'textarea' || el.isContentEditable) {
              action = 'fill_text';
              if (el.isContentEditable) {
                el.textContent = '42';
              } else {
                el.value = '42';
              }
              el.dispatchEvent(new Event('input', { bubbles: true }));
              el.dispatchEvent(new Event('change', { bubbles: true }));
            } else if (tag === 'select') {
              action = 'select_next_option';
              if (el.options && el.options.length > 1) {
                el.selectedIndex = (el.selectedIndex + 1) % el.options.length;
              }
              el.dispatchEvent(new Event('input', { bubbles: true }));
              el.dispatchEvent(new Event('change', { bubbles: true }));
            } else if (role === 'slider') {
              action = 'keyboard_slider';
              el.focus();
              el.dispatchEvent(new KeyboardEvent('keydown', { key: 'ArrowRight', bubbles: true }));
              el.dispatchEvent(new KeyboardEvent('keyup', { key: 'ArrowRight', bubbles: true }));
            } else {
              action = 'click';
              el.click();
            }
          } catch (e) {
            error = String(e && e.message ? e.message : e);
          }

          await new Promise(resolve => setTimeout(resolve, waitMs));
          observer.disconnect();
          const after = document.documentElement.outerHTML.length;
          return {
            ok: !error,
            action,
            tag,
            type,
            role,
            mutation_count: records.length,
            mutated: records.length > 0,
            html_length_before: before,
            html_length_after: after,
            mutation_sample: records.slice(0, 12),
            error
          };
        }""",
        {"elementIndex": element_index, "waitMs": wait_ms},
    )


@dataclass(frozen=True)
class ModuleCandidate:
    text: str
    center_x: float
    center_y: float
    score: float


def discover_module_candidates(page: Page, max_n: int = 18) -> list[ModuleCandidate]:
    """
    Heuristic: find sidebar-like clickable items (buttons/links) on the left side.
    Returns ordered unique candidates by descending sidebar score.
    """
    res = page.evaluate(
        """() => {
          const vw = window.innerWidth || 1024;
          const vh = window.innerHeight || 768;
          const els = Array.from(document.querySelectorAll('button,[role="button"],a'));
          const out = [];

          const norm = (s) => (s || '').replace(/\\s+/g, ' ').trim();
          const isVisible = (el) => {
            const r = el.getBoundingClientRect();
            if (!r || r.width < 26 || r.height < 18) return false;
            const st = window.getComputedStyle(el);
            if (!st || st.display === 'none' || st.visibility === 'hidden' || st.opacity === '0') return false;
            if (r.bottom < 0 || r.right < 0 || r.top > vh || r.left > vw) return false;
            return true;
          };

          const sidebarScore = (el) => {
            const r = el.getBoundingClientRect();
            let s = 0;
            if (r.left < vw * 0.38) s += 2.0;
            if (r.top > vh * 0.08 && r.top < vh * 0.92) s += 0.4;
            // nav-ish ancestors
            let p = el;
            for (let k = 0; k < 7 && p; k++) {
              const tag = (p.tagName || '').toLowerCase();
              const cls = ((p.className || '') + ' ' + (p.id || '')).toLowerCase();
              if (tag === 'nav' || tag === 'aside') { s += 2.0; break; }
              if (cls.includes('sidebar') || cls.includes('sider') || cls.includes('nav')) { s += 1.2; break; }
              p = p.parentElement;
            }
            // punish header-ish items (top area)
            if (r.top < 80) s -= 0.6;
            return s;
          };

          for (const el of els) {
            if (!isVisible(el)) continue;
            const txt = norm(el.innerText || el.getAttribute('aria-label') || el.getAttribute('title') || '');
            if (!txt) continue;
            if (txt.length < 2 || txt.length > 42) continue;
            // avoid obviously non-module labels
            if (/github|twitter|license|privacy|terms/i.test(txt)) continue;
            const r = el.getBoundingClientRect();
            const s = sidebarScore(el);
            out.push({ text: txt, cx: r.left + r.width/2, cy: r.top + r.height/2, score: s });
          }
          out.sort((a,b) => (b.score - a.score) || (a.cy - b.cy));
          const seen = new Set();
          const uniq = [];
          for (const it of out) {
            const key = it.text.toLowerCase();
            if (seen.has(key)) continue;
            seen.add(key);
            uniq.push(it);
            if (uniq.length >= 40) break;
          }
          return { candidates: uniq, viewport: { w: vw, h: vh } };
        }"""
    )
    cands = []
    for it in (res.get("candidates") or [])[: max(1, max_n * 3)]:
        try:
            cands.append(ModuleCandidate(text=str(it["text"]), center_x=float(it["cx"]), center_y=float(it["cy"]), score=float(it["score"])))
        except Exception:
            continue
    # Keep top-N but with a minimum score filter; if too few, keep at least some.
    kept = [c for c in cands if c.score >= 0.8]
    if len(kept) < 4:
        kept = cands[:max_n]
    else:
        kept = kept[:max_n]
    return kept


def click_module_by_text(page: Page, text: str) -> dict[str, Any]:
    """
    Robust click: re-find the best matching element and click by mouse at its center.
    Returns debug info including chosen rect and match count.
    """
    page.evaluate("() => window.scrollTo(0, 0)")
    safe_sleep(100)
    info = page.evaluate(
        """(targetText) => {
          const vw = window.innerWidth || 1024;
          const vh = window.innerHeight || 768;
          const norm = (s) => (s || '').replace(/\\s+/g, ' ').trim();
          const isVisible = (el) => {
            const r = el.getBoundingClientRect();
            if (!r || r.width < 26 || r.height < 18) return false;
            const st = window.getComputedStyle(el);
            if (!st || st.display === 'none' || st.visibility === 'hidden' || st.opacity === '0') return false;
            if (r.bottom < 0 || r.right < 0 || r.top > vh || r.left > vw) return false;
            return true;
          };
          const sidebarScore = (el) => {
            const r = el.getBoundingClientRect();
            let s = 0;
            if (r.left < vw * 0.40) s += 2.0;
            let p = el;
            for (let k = 0; k < 7 && p; k++) {
              const tag = (p.tagName || '').toLowerCase();
              const cls = ((p.className || '') + ' ' + (p.id || '')).toLowerCase();
              if (tag === 'nav' || tag === 'aside') { s += 2.0; break; }
              if (cls.includes('sidebar') || cls.includes('sider') || cls.includes('nav')) { s += 1.2; break; }
              p = p.parentElement;
            }
            if (r.top < 80) s -= 0.6;
            return s;
          };
          const target = norm(targetText).toLowerCase();
          const els = Array.from(document.querySelectorAll('button,[role="button"],a'));
          const matches = [];
          for (const el of els) {
            if (!isVisible(el)) continue;
            const txt = norm(el.innerText || el.getAttribute('aria-label') || el.getAttribute('title') || '').toLowerCase();
            if (!txt) continue;
            if (txt === target) matches.push(el);
          }
          // Fallback: contains match
          if (matches.length === 0) {
            for (const el of els) {
              if (!isVisible(el)) continue;
              const txt = norm(el.innerText || el.getAttribute('aria-label') || el.getAttribute('title') || '').toLowerCase();
              if (!txt) continue;
              if (txt.includes(target)) matches.push(el);
            }
          }
          if (matches.length === 0) return { ok: false, reason: 'no_match', match_count: 0 };
          let best = matches[0];
          let bestScore = -1e9;
          for (const el of matches) {
            const s = sidebarScore(el);
            if (s > bestScore) { bestScore = s; best = el; }
          }
          const r = best.getBoundingClientRect();
          return {
            ok: true,
            match_count: matches.length,
            score: bestScore,
            rect: { left: r.left, top: r.top, width: r.width, height: r.height },
            cx: r.left + r.width/2,
            cy: r.top + r.height/2,
          };
        }""",
        text,
    )
    if not info.get("ok"):
        raise RuntimeError(f"Cannot locate module button by text='{text}' ({info})")
    page.mouse.click(float(info["cx"]), float(info["cy"]))
    safe_sleep(280)
    return info


def drag_pan(page: Page, dx: int = 180, dy: int = 120) -> None:
    target = page.locator("canvas").first
    box = target.bounding_box()
    if not box:
        target = page.locator("main").first
        box = target.bounding_box()
    if not box:
        return
    sx = box["x"] + box["width"] * 0.55
    sy = box["y"] + box["height"] * 0.55
    page.mouse.move(sx, sy)
    page.mouse.down()
    page.mouse.move(sx + dx, sy + dy, steps=14)
    page.mouse.up()
    safe_sleep(180)


def wheel_zoom(page: Page, delta_y: int = -650) -> None:
    target = page.locator("canvas").first
    box = target.bounding_box()
    if not box:
        target = page.locator("main").first
        box = target.bounding_box()
    if not box:
        return
    cx = box["x"] + box["width"] * 0.55
    cy = box["y"] + box["height"] * 0.55
    page.mouse.move(cx, cy)
    page.mouse.wheel(0, delta_y)
    safe_sleep(220)


def move_first_slider(page: Page) -> bool:
    slider = page.locator('input[type="range"]').first
    if slider.count() == 0:
        return False
    try:
        page.evaluate(
            """(el) => {
              const min = parseFloat(el.min || '0');
              const max = parseFloat(el.max || '1');
              const val = min + (max - min) * 0.65;
              el.value = String(val);
              el.dispatchEvent(new Event('input', { bubbles: true }));
              el.dispatchEvent(new Event('change', { bubbles: true }));
            }""",
            slider,
        )
        safe_sleep(220)
        return True
    except Exception:
        return False


def click_play_like(page: Page) -> str | None:
    for pat in [r"Descend", r"Start", r"Play", r"Run", r"Train", r"Step", r"Simulate", r"Reset"]:
        try:
            btn = page.get_by_role("button", name=re.compile(pat, re.I)).first
            if btn.count() > 0:
                btn.click(timeout=2500)
                safe_sleep(220)
                return pat
        except Exception:
            continue
    return None


def click_some_non_sidebar_buttons(page: Page, n: int = 3) -> list[str]:
    """
    Click up to n visible buttons, preferring ones not in left sidebar area.
    Uses JS to pick candidates by bounding box, then clicks by coordinate for robustness.
    """
    picked = page.evaluate(
        """(n) => {
          const vw = window.innerWidth || 1024;
          const vh = window.innerHeight || 768;
          const norm = (s) => (s || '').replace(/\\s+/g, ' ').trim();
          const buttons = Array.from(document.querySelectorAll('button'));
          const isVisible = (el) => {
            const r = el.getBoundingClientRect();
            if (!r || r.width < 28 || r.height < 18) return false;
            const st = window.getComputedStyle(el);
            if (!st || st.display === 'none' || st.visibility === 'hidden' || st.opacity === '0') return false;
            if (r.bottom < 0 || r.right < 0 || r.top > vh || r.left > vw) return false;
            return true;
          };
          const out = [];
          for (const b of buttons) {
            if (!isVisible(b)) continue;
            const txt = norm(b.innerText || b.getAttribute('aria-label') || '');
            if (!txt || txt.length < 2 || txt.length > 38) continue;
            const r = b.getBoundingClientRect();
            // prefer not-sidebar and not header
            const notSidebar = r.left > vw * 0.40;
            const notHeader = r.top > 70;
            const score = (notSidebar ? 2.0 : 0.0) + (notHeader ? 0.6 : 0.0) + Math.min(0.5, r.width / 400);
            out.push({ text: txt, cx: r.left + r.width/2, cy: r.top + r.height/2, score });
          }
          out.sort((a,b) => b.score - a.score);
          const seen = new Set();
          const picked = [];
          for (const it of out) {
            const key = it.text.toLowerCase();
            if (seen.has(key)) continue;
            seen.add(key);
            picked.push(it);
            if (picked.length >= n) break;
          }
          return picked;
        }""",
        n,
    )
    clicked: list[str] = []
    for it in picked or []:
        try:
            page.mouse.click(float(it["cx"]), float(it["cy"]))
            safe_sleep(180)
            clicked.append(str(it["text"]))
        except Exception:
            continue
    return clicked


def run_module_probe(
    page: Page,
    start_url: str,
    out_dir: Path,
    site_id: str,
    max_modules: int,
    ignore_module_regex: list[re.Pattern[str]] | None = None,
) -> None:
    ensure_dir(out_dir)
    modules_dir = out_dir / "modules"
    ensure_dir(modules_dir)
    progress_path = out_dir / "progress.jsonl"
    if progress_path.exists():
        progress_path.unlink()

    run_meta = {
        "site_id": site_id,
        "start_url": start_url,
        "ts_ms": now_ms(),
        "max_modules": max_modules,
    }
    write_json(out_dir / "run_meta.json", run_meta)

    t_site0 = time.time()
    append_jsonl(progress_path, {"ts_ms": now_ms(), "event": "site_start", "site_id": site_id, "url": start_url})
    console.print(f"[bold]goto[/bold] {start_url}")
    page.goto(start_url, wait_until="domcontentloaded")
    safe_sleep(650)

    # Landing shot
    landing_dir = modules_dir / "_landing"
    ensure_dir(landing_dir / "screens")
    landing_shot = landing_dir / "screens" / "step_000.png"
    screenshot(page, landing_shot)
    landing_stats = extract_dom_stats(page)
    landing_health = page_health(page)
    landing_elements = enumerate_interactive_elements(page, max_n=40)
    write_json(
        landing_dir / "module_probe.json",
        {
            "ok": True,
            "module_name": "Landing",
            "module_id": "_landing",
            "clicked": None,
            "screens": [str(landing_shot)],
            "dom_stats": landing_stats,
            "health": landing_health,
            "body_text": body_inner_text(page, limit=6000),
            "interactive_elements": landing_elements,
            "interaction_probe": {
                "protocol": "semantic_action_mapping_with_mutation_observer",
                "actions": [],
                "has_dom_mutation": False,
                "mutation_action_count": 0,
            },
            "metrics": {
                "dark_ratio": dark_ratio(landing_shot),
                "unique_color_count": unique_color_count(landing_shot),
                "dom_mutation_count_after_actions": 0,
                "has_dom_mutation_after_actions": False,
            },
            "diffs": [],
            "errors": [],
            "ts_ms": now_ms(),
        },
    )
    append_jsonl(
        progress_path,
        {"ts_ms": now_ms(), "event": "module_done", "module_id": "_landing", "module_name": "Landing", "ok": True},
    )

    candidates = discover_module_candidates(page, max_n=max_modules)
    cand_texts = [c.text for c in candidates]
    # De-duplicate obvious landing-like item if it appears in sidebar
    cand_texts = [t for t in cand_texts if t.lower() not in {"landing", "home"}]
    if ignore_module_regex:
        before = list(cand_texts)
        cand_texts = [t for t in cand_texts if not any(p.search(t) for p in ignore_module_regex)]
        removed = [t for t in before if t not in cand_texts]
        if removed:
            console.print(f"[yellow]ignored_modules[/yellow] n={len(removed)} -> {removed[:10]}")
            append_jsonl(progress_path, {"ts_ms": now_ms(), "event": "ignored_modules", "count": len(removed), "texts": removed})
    console.print(f"[bold]discovered_modules[/bold] n={len(cand_texts)} -> {cand_texts[:10]}")
    append_jsonl(progress_path, {"ts_ms": now_ms(), "event": "discovered_modules", "count": len(cand_texts), "texts": cand_texts})

    site_index: dict[str, Any] = {
        "site_id": site_id,
        "start_url": start_url,
        "ts_ms": now_ms(),
        "discovered_module_texts": cand_texts,
        "modules": [{"module_id": "_landing", "module_name": "Landing", "dir": str(landing_dir)}],
    }

    prev_baseline = landing_shot
    for idx, mod_text in enumerate(cand_texts[:max_modules]):
        t_mod0 = time.time()
        console.print(f"[bold]module_start[/bold] {idx+1}/{min(len(cand_texts), max_modules)}: {mod_text}")
        append_jsonl(progress_path, {"ts_ms": now_ms(), "event": "module_start", "i": idx, "module_name": mod_text})
        module_id = re.sub(r"[^a-zA-Z0-9]+", "_", mod_text.strip()).strip("_").lower()[:48] or f"m{idx:02d}"
        mod_dir = modules_dir / module_id
        ensure_dir(mod_dir / "screens")

        errors: list[str] = []
        clicked_info: dict[str, Any] | None = None
        screen_paths: list[Path] = []

        try:
            clicked_info = click_module_by_text(page, mod_text)
        except Exception as e:
            errors.append(f"click_module_failed: {e}")

        # Baseline after entering module (or after failed click, still capture)
        baseline = mod_dir / "screens" / "step_000.png"
        screenshot(page, baseline)
        screen_paths.append(baseline)

        # Paper-aligned Interaction Probe:
        # enumerate visible interactive elements, map each to a canonical action,
        # and use MutationObserver to detect childList/subtree/attribute changes.
        action_steps: list[dict[str, Any]] = []

        def step_shot(step_name: str) -> Path:
            p = mod_dir / "screens" / f"{step_name}.png"
            screenshot(page, p)
            screen_paths.append(p)
            return p

        # 1) DOM stats snapshot
        try:
            ds0 = extract_dom_stats(page)
        except Exception as e:
            ds0 = {}
            errors.append(f"dom_stats_failed: {e}")

        interactive_elements = enumerate_interactive_elements(page, max_n=40)
        mutation_action_count = 0
        mutation_total = 0
        try:
            for elem in interactive_elements[:16]:
                result = probe_semantic_action(page, int(elem.get("index", 0)))
                result["element"] = elem
                shot = step_shot(f"after_action_{len(action_steps):03d}")
                result["screenshot"] = str(shot)
                action_steps.append(result)
                if result.get("mutated"):
                    mutation_action_count += 1
                mutation_total += int(result.get("mutation_count") or 0)
        except Exception as e:
            errors.append(f"interaction_probe_failed: {e}")

        # Final stats/health
        ds1 = extract_dom_stats(page)
        health = page_health(page)

        # Metrics and diffs (baseline vs each subsequent)
        diffs: list[dict[str, Any]] = []
        for p in screen_paths[1:]:
            try:
                diffs.append({"from": str(baseline), "to": str(p), "diff_ratio": image_diff_ratio(baseline, p)})
            except Exception as e:
                diffs.append({"from": str(baseline), "to": str(p), "error": str(e), "diff_ratio": 0.0})
        max_diff = max((d.get("diff_ratio", 0.0) or 0.0) for d in diffs) if diffs else 0.0

        # Also record distinctness vs previous module baseline
        try:
            nav_diff = image_diff_ratio(prev_baseline, baseline)
        except Exception:
            nav_diff = 0.0
        prev_baseline = baseline

        module_obj = {
            "ok": len(errors) == 0,
            "module_name": mod_text,
            "module_id": module_id,
            "clicked": clicked_info,
            "screens": [str(p) for p in screen_paths],
            "baseline": str(baseline),
            "dom_stats_before": ds0,
            "dom_stats_after": ds1,
            "health": health,
            "body_text": body_inner_text(page, limit=6000),
            "interactive_elements": interactive_elements,
            "interaction_probe": {
                "protocol": "semantic_action_mapping_with_mutation_observer",
                "actions": action_steps,
                "has_dom_mutation": mutation_action_count > 0,
                "mutation_action_count": mutation_action_count,
                "mutation_record_count": mutation_total,
            },
            "metrics": {
                "dark_ratio": dark_ratio(baseline),
                "unique_color_count": unique_color_count(baseline),
                "max_diff_after_actions": max_diff,
                "nav_diff_vs_prev": nav_diff,
                "dom_mutation_count_after_actions": mutation_total,
                "has_dom_mutation_after_actions": mutation_action_count > 0,
            },
            "diffs": diffs,
            "actions": action_steps,
            "errors": errors,
            "ts_ms": now_ms(),
        }
        write_json(mod_dir / "module_probe.json", module_obj)
        site_index["modules"].append({"module_id": module_id, "module_name": mod_text, "dir": str(mod_dir)})
        append_jsonl(
            progress_path,
            {
                "ts_ms": now_ms(),
                "event": "module_done",
                "i": idx,
                "module_id": module_id,
                "module_name": mod_text,
                "ok": len(errors) == 0,
                "seconds": round(time.time() - t_mod0, 3),
                "screens": len(screen_paths),
                "errors": errors[:3],
            },
        )
        console.print(f"[bold]module_done[/bold] {mod_text} ok={len(errors)==0} t={time.time()-t_mod0:.1f}s screens={len(screen_paths)}")

    write_json(out_dir / "site_probe.json", site_index)
    append_jsonl(progress_path, {"ts_ms": now_ms(), "event": "site_done", "site_id": site_id, "seconds": round(time.time() - t_site0, 3)})
    console.print(f"[bold]site_done[/bold] {site_id} t={time.time()-t_site0:.1f}s")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start_url", required=True, help="e.g. http://127.0.0.1:8000/outputs/tsx/<site>/dist/index.html")
    ap.add_argument("--site_id", required=True, help="site id (folder name)")
    ap.add_argument("--out_dir", required=True, help="output run directory (benchmark/results/codegen_score_v1/<site>/<run_id>)")
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--viewport_width", type=int, default=1024)
    ap.add_argument("--viewport_height", type=int, default=768)
    ap.add_argument("--max_modules", type=int, default=12)
    ap.add_argument(
        "--ignore_module_regex",
        action="append",
        default=[],
        help="skip discovered module buttons whose text matches this regex (case-insensitive). Can repeat.",
    )
    ap.add_argument("--nav_timeout_ms", type=int, default=30_000)
    ap.add_argument("--action_timeout_ms", type=int, default=5_000)
    args = ap.parse_args()

    start_url = args.start_url
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=args.headless)
        context = browser.new_context(viewport={"width": args.viewport_width, "height": args.viewport_height})
        page = context.new_page()
        page.set_default_navigation_timeout(args.nav_timeout_ms)
        page.set_default_timeout(args.action_timeout_ms)
        # Reduce motion where possible (helps determinism)
        try:
            page.emulate_media(reduced_motion="reduce")
        except Exception:
            pass
        console.print(f"[bold]ModuleProbe[/bold] site={args.site_id} url={start_url}")
        # Apply ignore filters (deterministic; purely text-based)
        ignore_pats: list[re.Pattern[str]] = []
        for s in args.ignore_module_regex:
            try:
                ignore_pats.append(re.compile(str(s), flags=re.I))
            except Exception as e:
                console.print(f"[yellow]WARN[/yellow] bad ignore_module_regex='{s}': {e}")
        run_module_probe(
            page,
            start_url=start_url,
            out_dir=out_dir,
            site_id=args.site_id,
            max_modules=args.max_modules,
            ignore_module_regex=ignore_pats,
        )
        context.close()
        browser.close()


if __name__ == "__main__":
    main()


