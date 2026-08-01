# PR 9: CUDA GEMM Broadcast Batch Collapse

Branch: `cuda-f32-gemm-broadcast-collapse`
Base: `main`
Head commit: `2b8b3ace`

Suggested title:

```text
Optimize CUDA broadcast matmul collapse
```

Upload:

```bash
cd /home/mahogny/github/claude/candle
git checkout cuda-f32-gemm-broadcast-collapse
git push -u origin cuda-f32-gemm-broadcast-collapse
gh pr create \
  --base main \
  --head <your-github-user>:cuda-f32-gemm-broadcast-collapse \
  --title "Optimize CUDA broadcast matmul collapse" \
  --body-file /data/henriksson/github/claude/stardist-rs/prs/09-cuda-gemm-broadcast-collapse.md
```

This branch is independent of PR 8 and can be opened in any order relative to
it. Local sm75 builds need PR 8, so retest on `validation/cuda-f32-gemm-broadcast-collapse-sm75`,
which is PR 8 plus this commit and is not submitted.

Suggested PR body:

## Summary

This adds a CUDA matmul layout optimization for broadcasted batch dimensions.

When the right-hand side is shared across batches and the tensor layout proves the operation is equivalent, Candle can collapse `(b, m, n, k)` into `(1, b * m, n, k)` and dispatch one larger GEMM instead of many smaller batch GEMMs.

## Scope

- Adds a reusable `batch_broadcast_matmul_config` helper.
- Applies the collapse to CUDA F32 and F64 matmul paths.
- Leaves F16/BF16 unchanged.
- Keeps the F32 reduced-precision path on the existing SGEMM route when reduced precision is requested.

## Motivation

Broadcasted RHS matmul appears in model and image-processing workloads where the same weight matrix is applied across many batches. Collapsing the batch dimension reduces kernel launch and batched GEMM overhead without changing semantics.

## Validation

Validated locally on Quadro RTX 5000 / sm75 / CUDA 12.8. The CUDA commands below
were run on `validation/cuda-f32-gemm-broadcast-collapse-sm75`, which is this
commit plus the separate sm75 build-compatibility fix (PR 8); that fix only
affects whether `candle-kernels` compiles on Turing locally and is not part of
this PR.

```bash
cargo fmt

CUDA_HOME=/usr/local/cuda-12.8 \
CUDA_PATH=/usr/local/cuda-12.8 \
NVCC=/usr/local/cuda-12.8/bin/nvcc \
CUDA_COMPUTE_CAP=75 \
LD_LIBRARY_PATH=/usr/local/cuda-12.8/targets/x86_64-linux/lib:$LD_LIBRARY_PATH \
cargo test -p candle-core --features cuda matmul_broadcast_rhs_batch_stride_zero

CUDA_HOME=/usr/local/cuda-12.8 \
CUDA_PATH=/usr/local/cuda-12.8 \
NVCC=/usr/local/cuda-12.8/bin/nvcc \
CUDA_COMPUTE_CAP=75 \
LD_LIBRARY_PATH=/usr/local/cuda-12.8/targets/x86_64-linux/lib:$LD_LIBRARY_PATH \
cargo check -p candle-core --features cuda

git diff --check main..cuda-f32-gemm-broadcast-collapse
```

## Benchmark

Shape: `[64, 256, 256] @ [256, 256]`

| Branch | Time |
| --- | ---: |
| Baseline | 3.648 ms |
| This PR | 2.946 ms |

Ratio: 1.24x faster.

## Review Notes

- The helper is intentionally general across floating point element types.
- The optimization is gated by layout checks; non-matching layouts continue through the existing path.
- The diff touches only `candle-core/src/cuda_backend/mod.rs` and is independent of my other CUDA PRs.
