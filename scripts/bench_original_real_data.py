#!/usr/bin/env python3
"""Benchmark original Python StarDist on bundled real image data.

This script writes an NPZ artifact that the Rust Burn benchmark can consume.
It intentionally lives under scripts/ so it does not become part of the library
API or the optional crate CLI surface.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import resource
import sys
import time

import numpy as np
import tifffile


ROOT = Path(__file__).resolve().parents[1]
STARDIST_SRC = ROOT / "stardist"
ASSETS = ROOT / "assets" / "data" / "images"


def _normalize(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32, copy=False)
    lo = float(np.min(x))
    hi = float(np.max(x))
    if hi <= lo:
        return np.zeros_like(x, dtype=np.float32)
    return ((x - lo) / (hi - lo)).astype(np.float32, copy=False)


def _rss_kib() -> int:
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


def _bench_2d(repeats: int) -> dict[str, object]:
    from stardist.models import StarDist2D

    model_dir = ROOT / "stardist" / "models" / "examples" / "2D_demo"
    model = StarDist2D(None, name="2D_demo", basedir=str(model_dir.parent))
    input_yx = _normalize(tifffile.imread(ASSETS / "img2d.tif"))
    input_nhwc = input_yx[None, :, :, None].astype(np.float32, copy=False)

    raw_prob_nhwc = raw_dist_nhwc = None
    started = time.perf_counter()
    for _ in range(repeats):
        raw_prob_nhwc, raw_dist_nhwc = model.keras_model.predict(input_nhwc, verbose=0)
    raw_inference_seconds = (time.perf_counter() - started) / repeats

    started = time.perf_counter()
    sparse_prob, sparse_dist, sparse_points = model.predict_sparse(
        input_yx,
        axes="YX",
        normalizer=None,
        n_tiles=None,
        show_tile_progress=False,
    )
    predict_sparse_seconds = time.perf_counter() - started

    started = time.perf_counter()
    labels, polys = model._instances_from_prediction(
        input_yx.shape,
        sparse_prob,
        sparse_dist,
        points=sparse_points,
        prob_thresh=None,
        nms_thresh=None,
        return_labels=True,
    )
    postprocess_seconds = time.perf_counter() - started

    started = time.perf_counter()
    model.predict_instances(
        input_yx,
        axes="YX",
        normalizer=None,
        sparse=False,
        n_tiles=None,
        show_tile_progress=False,
    )
    predict_instances_seconds = time.perf_counter() - started

    return {
        "input_nhwc": input_nhwc,
        "input_nchw": np.transpose(input_nhwc, (0, 3, 1, 2)).astype(np.float32),
        "raw_prob_nhwc": raw_prob_nhwc.astype(np.float32),
        "raw_prob_nchw": np.transpose(raw_prob_nhwc, (0, 3, 1, 2)).astype(np.float32),
        "raw_dist_nhwc": raw_dist_nhwc.astype(np.float32),
        "raw_dist_nchw": np.transpose(raw_dist_nhwc, (0, 3, 1, 2)).astype(np.float32),
        "sparse_prob": sparse_prob.astype(np.float32),
        "sparse_dist": sparse_dist.astype(np.float32),
        "sparse_points": sparse_points.astype(np.float32),
        "labels": labels.astype(np.uint32),
        "points": polys["points"].astype(np.float32),
        "prob": polys["prob"].astype(np.float32),
        "coord": polys["coord"].astype(np.float32),
        "metrics": {
            "backend": "python-stardist",
            "dimension": "2d",
            "repeats": repeats,
            "raw_inference_seconds": raw_inference_seconds,
            "predict_sparse_seconds": predict_sparse_seconds,
            "postprocess_seconds": postprocess_seconds,
            "predict_instances_seconds": predict_instances_seconds,
            "max_rss_kib": _rss_kib(),
            "pid": os.getpid(),
        },
    }


def _bench_3d(repeats: int) -> dict[str, object]:
    from stardist.models import StarDist3D

    model_dir = ROOT / "stardist" / "models" / "examples" / "3D_demo"
    model = StarDist3D(None, name="3D_demo", basedir=str(model_dir.parent))
    input_zyx = _normalize(tifffile.imread(ASSETS / "img3d.tif"))
    input_ndhwc = input_zyx[None, :, :, :, None].astype(np.float32, copy=False)

    raw_prob_ndhwc = raw_dist_ndhwc = None
    started = time.perf_counter()
    for _ in range(repeats):
        raw_prob_ndhwc, raw_dist_ndhwc = model.keras_model.predict(input_ndhwc, verbose=0)
    raw_inference_seconds = (time.perf_counter() - started) / repeats

    started = time.perf_counter()
    sparse_prob, sparse_dist, sparse_points = model.predict_sparse(
        input_zyx,
        axes="ZYX",
        normalizer=None,
        n_tiles=None,
        show_tile_progress=False,
    )
    predict_sparse_seconds = time.perf_counter() - started

    started = time.perf_counter()
    labels, polys = model._instances_from_prediction(
        input_zyx.shape,
        sparse_prob,
        sparse_dist,
        points=sparse_points,
        prob_thresh=None,
        nms_thresh=None,
        return_labels=True,
    )
    postprocess_seconds = time.perf_counter() - started

    started = time.perf_counter()
    model.predict_instances(
        input_zyx,
        axes="ZYX",
        normalizer=None,
        sparse=False,
        n_tiles=None,
        show_tile_progress=False,
    )
    predict_instances_seconds = time.perf_counter() - started

    return {
        "input_ndhwc": input_ndhwc,
        "input_ncdhw": np.transpose(input_ndhwc, (0, 4, 1, 2, 3)).astype(np.float32),
        "raw_prob_ndhwc": raw_prob_ndhwc.astype(np.float32),
        "raw_prob_ncdhw": np.transpose(raw_prob_ndhwc, (0, 4, 1, 2, 3)).astype(np.float32),
        "raw_dist_ndhwc": raw_dist_ndhwc.astype(np.float32),
        "raw_dist_ncdhw": np.transpose(raw_dist_ndhwc, (0, 4, 1, 2, 3)).astype(np.float32),
        "sparse_prob": sparse_prob.astype(np.float32),
        "sparse_dist": sparse_dist.astype(np.float32),
        "sparse_points": sparse_points.astype(np.float32),
        "labels": labels.astype(np.uint32),
        "points": polys["points"].astype(np.float32),
        "prob": polys["prob"].astype(np.float32),
        "dist": polys["dist"].astype(np.float32),
        "metrics": {
            "backend": "python-stardist",
            "dimension": "3d",
            "repeats": repeats,
            "raw_inference_seconds": raw_inference_seconds,
            "predict_sparse_seconds": predict_sparse_seconds,
            "postprocess_seconds": postprocess_seconds,
            "predict_instances_seconds": predict_instances_seconds,
            "max_rss_kib": _rss_kib(),
            "pid": os.getpid(),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dimension", choices=["2d", "3d"])
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()

    if args.repeats < 1:
        raise SystemExit("--repeats must be at least 1")
    if not STARDIST_SRC.exists():
        raise SystemExit(f"missing upstream checkout: {STARDIST_SRC}")
    sys.path.insert(0, str(STARDIST_SRC))

    result = _bench_2d(args.repeats) if args.dimension == "2d" else _bench_3d(args.repeats)
    metrics = result.pop("metrics")
    out = args.out or ROOT / ".tmp" / f"bench_original_real_{args.dimension}.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, **result)

    metrics_path = out.with_suffix(".json")
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"npz": str(out), "metrics": str(metrics_path), **metrics}, sort_keys=True))


if __name__ == "__main__":
    main()
