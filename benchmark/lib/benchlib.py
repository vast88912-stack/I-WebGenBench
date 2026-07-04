from __future__ import annotations

import json
import math
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image


FLOAT_RE = re.compile(r"(-?\d+(?:\.\d+)?)")


def now_ms() -> int:
    return int(time.time() * 1000)


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def append_jsonl(path: Path, obj: Any) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def image_diff_ratio(img_a: Path, img_b: Path) -> float:
    """
    Returns the fraction of pixels whose absolute RGB difference exceeds a small threshold.
    """
    a = Image.open(img_a).convert("RGB")
    b = Image.open(img_b).convert("RGB")
    if a.size != b.size:
        b = b.resize(a.size)
    arr_a = np.asarray(a, dtype=np.int16)
    arr_b = np.asarray(b, dtype=np.int16)
    diff = np.abs(arr_a - arr_b).sum(axis=2)  # 0..(255*3)
    changed = diff > 25  # small noise threshold
    return float(changed.mean())


def dark_ratio(img: Path) -> float:
    im = Image.open(img).convert("RGB")
    arr = np.asarray(im, dtype=np.uint8)
    lum = (0.2126 * arr[:, :, 0] + 0.7152 * arr[:, :, 1] + 0.0722 * arr[:, :, 2])
    return float((lum < 25).mean())


def unique_color_count(img: Path, max_samples: int = 120000) -> int:
    im = Image.open(img).convert("RGB")
    arr = np.asarray(im, dtype=np.uint8).reshape(-1, 3)
    if arr.shape[0] > max_samples:
        idx = np.random.choice(arr.shape[0], size=max_samples, replace=False)
        arr = arr[idx]
    # pack to int
    packed = (arr[:, 0].astype(np.uint32) << 16) | (arr[:, 1].astype(np.uint32) << 8) | arr[:, 2].astype(np.uint32)
    return int(len(np.unique(packed)))


def parse_first_float(text: str) -> float | None:
    m = FLOAT_RE.search(text)
    return float(m.group(1)) if m else None


def mean_std(xs: Iterable[float]) -> tuple[float, float]:
    xs = list(xs)
    if not xs:
        return 0.0, 0.0
    mu = sum(xs) / len(xs)
    var = sum((x - mu) ** 2 for x in xs) / max(1, (len(xs) - 1))
    return mu, math.sqrt(var)



