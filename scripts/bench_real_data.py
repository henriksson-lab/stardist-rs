#!/usr/bin/env python3
"""Run real-data StarDist benchmarks and collect JSON results.

This is an orchestration script for local translation diagnostics. It uses the
original Python StarDist script to create NPZ fixtures, then runs the Rust Burn
and Candle examples against the same inputs.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
TMP = ROOT / ".tmp"


def _run(
    cmd: list[str],
    *,
    allow_failure: bool = False,
    env: dict[str, str] | None = None,
) -> dict[str, Any]:
    run_env = os.environ.copy()
    if env is not None:
        run_env.update(env)
    result = subprocess.run(
        cmd,
        cwd=ROOT,
        env=run_env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode != 0 and not allow_failure:
        sys.stderr.write(result.stderr)
        raise SystemExit(result.returncode)
    return {
        "cmd": cmd,
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


def _json_from_stdout(stdout: str) -> dict[str, Any]:
    for start in (index for index, char in enumerate(stdout) if char == "{"):
        try:
            return json.loads(stdout[start:])
        except json.JSONDecodeError:
            continue
    raise ValueError("command stdout did not contain a JSON object")


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _run_python_baseline(
    dimension: str,
    repeats: int,
    *,
    python_gpu: bool,
) -> tuple[Path, dict[str, Any]]:
    npz = TMP / f"bench_original_real_{dimension}.npz"
    _run(
        [
            sys.executable,
            "scripts/bench_original_real_data.py",
            dimension,
            "--out",
            str(npz),
            "--repeats",
            str(repeats),
        ],
        env={} if python_gpu else {"CUDA_VISIBLE_DEVICES": ""},
    )
    return npz, _load_json(npz.with_suffix(".json"))


def _run_rust_example(
    backend: str,
    dimension: str,
    npz: Path,
    *,
    device: str | None = None,
    allow_failure: bool = False,
    env: dict[str, str] | None = None,
) -> dict[str, Any]:
    if backend == "burn":
        cmd = [
            "cargo",
            "run",
            "--release",
            "--features",
            "burn",
            "--example",
            "bench_burn_real_data",
            "--",
            dimension,
            str(npz),
        ]
    elif backend == "candle":
        features = "candle-cuda,hdf5" if device == "cuda" else "candle,hdf5"
        cmd = [
            "cargo",
            "run",
            "--release",
            "--features",
            features,
            "--example",
            "bench_candle_real_data",
            "--",
            dimension,
            str(npz),
            device or "cpu",
        ]
    else:
        raise ValueError(f"unknown backend {backend!r}")

    result = _run(cmd, allow_failure=allow_failure, env=env)
    if result["returncode"] != 0:
        out = {
            "backend": f"rust-{backend}",
            "dimension": dimension,
            "device": device,
            "status": "failed",
            "returncode": result["returncode"],
            "stderr_tail": result["stderr"][-4000:],
        }
        if backend == "candle" and device == "cuda":
            out.update(_candle_cuda_failure_hint(result["stderr"]))
        return out
    return _json_from_stdout(result["stdout"])


def _skip_result(
    backend: str,
    dimension: str,
    device: str,
    reason: str,
    *,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    out = {
        "backend": backend,
        "dimension": dimension,
        "device": device,
        "status": "skipped",
        "reason": reason,
    }
    if metadata:
        out.update(metadata)
    return out


def _cuda_toolkit_env(cuda_home: Path | None) -> tuple[dict[str, str], dict[str, Any]]:
    metadata: dict[str, Any] = {}
    if cuda_home is None:
        nvcc = shutil.which("nvcc")
        metadata["nvcc"] = nvcc
        return {}, metadata

    cuda_home = cuda_home.resolve()
    nvcc = cuda_home / "bin" / "nvcc"
    metadata["cuda_home"] = str(cuda_home)
    metadata["nvcc"] = str(nvcc)
    path_prefix = str(cuda_home / "bin")
    lib_prefix = str(cuda_home / "lib64")
    return {
        "CUDA_HOME": str(cuda_home),
        "CUDA_PATH": str(cuda_home),
        "PATH": path_prefix + os.pathsep + os.environ.get("PATH", ""),
        "LD_LIBRARY_PATH": lib_prefix + os.pathsep + os.environ.get("LD_LIBRARY_PATH", ""),
    }, metadata


def _run_tool_text(cmd: list[str], env: dict[str, str] | None = None) -> str | None:
    try:
        result = _run(cmd, allow_failure=True, env=env)
    except FileNotFoundError:
        return None
    if result["returncode"] != 0:
        return None
    return result["stdout"].strip()


def _compute_capability(
    env: dict[str, str] | None = None,
    override: str | None = None,
) -> str | None:
    if override:
        return override
    env_override = os.environ.get("CUDA_COMPUTE_CAP")
    if env_override:
        return env_override
    output = _run_tool_text(
        ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
        env=env,
    )
    if output is None:
        return None
    first = output.splitlines()[0].strip()
    return first or None


def _parse_compute_cap(value: str) -> tuple[int, int] | None:
    normalized = value.strip().lower().removeprefix("sm_").replace(".", "")
    if len(normalized) < 2 or not normalized[:2].isdigit():
        return None
    return int(normalized[0]), int(normalized[1])


def _cuda_preflight(
    cuda_home: Path | None,
    compute_capability: str | None,
) -> tuple[str | None, dict[str, Any], dict[str, str]]:
    env, metadata = _cuda_toolkit_env(cuda_home)
    nvcc = metadata.get("nvcc")
    if nvcc is None:
        return (
            "nvcc is not on PATH; Candle CUDA builds through cudarc require the CUDA toolkit",
            metadata,
            env,
        )
    if not Path(nvcc).exists() and shutil.which(str(nvcc)) is None:
        return (f"nvcc was not found at {nvcc}", metadata, env)

    nvcc_version = _run_tool_text([str(nvcc), "--version"], env=env)
    if nvcc_version:
        metadata["nvcc_version"] = nvcc_version.splitlines()[-1]

    compute_cap = _compute_capability(env=env, override=compute_capability)
    if compute_cap:
        metadata["compute_capability"] = compute_cap

    return None, metadata, env


def _candle_cuda_failure_hint(stderr: str) -> dict[str, Any]:
    if "moe_wmma" in stderr and ("nv_bfloat16" in stderr or "wmma::fragment" in stderr):
        return {
            "known_reason": (
                "candle-kernels failed compiling BF16 WMMA/MoE CUDA kernels; "
                "this is a Candle CUDA build compatibility issue, not a StarDist parity failure"
            )
        }
    return {}


def _ratio(rust: dict[str, Any], python: dict[str, Any], key: str) -> float | None:
    rust_value = rust.get(key)
    python_value = python.get(key)
    if not isinstance(rust_value, (int, float)) or not isinstance(python_value, (int, float)):
        return None
    if python_value == 0:
        return None
    return float(rust_value) / float(python_value)


def _annotate_ratios(entry: dict[str, Any], python: dict[str, Any]) -> dict[str, Any]:
    out = dict(entry)
    out["ratios_vs_python"] = {
        key: _ratio(entry, python, key)
        for key in (
            "raw_inference_seconds",
            "predict_sparse_seconds",
            "postprocess_seconds",
            "max_rss_kib",
        )
    }
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--out", type=Path, default=TMP / "bench_real_data_summary.json")
    parser.add_argument("--skip-burn", action="store_true")
    parser.add_argument("--skip-candle-cpu", action="store_true")
    parser.add_argument("--candle-cuda", action="store_true")
    parser.add_argument("--cuda-home", type=Path)
    parser.add_argument("--cuda-compute-cap")
    parser.add_argument("--python-gpu", action="store_true")
    parser.add_argument("--dimensions", choices=["2d", "3d", "all"], default="all")
    args = parser.parse_args()

    if args.repeats < 1:
        raise SystemExit("--repeats must be at least 1")

    dimensions = ["2d", "3d"] if args.dimensions == "all" else [args.dimensions]
    TMP.mkdir(parents=True, exist_ok=True)

    summary: dict[str, Any] = {"python": {}, "rust": []}
    for dimension in dimensions:
        npz, python_metrics = _run_python_baseline(
            dimension,
            args.repeats,
            python_gpu=args.python_gpu,
        )
        summary["python"][dimension] = python_metrics

        if not args.skip_burn:
            burn = _run_rust_example("burn", dimension, npz)
            summary["rust"].append(_annotate_ratios(burn, python_metrics))

        if not args.skip_candle_cpu:
            candle_cpu = _run_rust_example("candle", dimension, npz, device="cpu")
            summary["rust"].append(_annotate_ratios(candle_cpu, python_metrics))

        if dimension == "2d" and args.candle_cuda:
            cuda_skip_reason, cuda_metadata, cuda_env = _cuda_preflight(
                args.cuda_home,
                args.cuda_compute_cap,
            )
            if cuda_skip_reason is None:
                candle_cuda = _run_rust_example(
                    "candle",
                    dimension,
                    npz,
                    device="cuda",
                    allow_failure=True,
                    env=cuda_env,
                )
                candle_cuda.update(cuda_metadata)
                if candle_cuda.get("status") != "failed":
                    candle_cuda = _annotate_ratios(candle_cuda, python_metrics)
            else:
                candle_cuda = _skip_result(
                    "rust-candle",
                    dimension,
                    "cuda",
                    cuda_skip_reason,
                    metadata=cuda_metadata,
                )
            summary["rust"].append(candle_cuda)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"summary": str(args.out), **summary}, sort_keys=True))


if __name__ == "__main__":
    main()
