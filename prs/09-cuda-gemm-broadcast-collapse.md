# PR 9: CUDA GEMM Broadcast Batch Collapse

Branch: `cuda-f32-gemm-broadcast-collapse`
Base: `cuda-sm75-compat`
Head commit: `8b5605fc`

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
  --base cuda-sm75-compat \
  --head <your-github-user>:cuda-f32-gemm-broadcast-collapse \
  --title "Optimize CUDA broadcast matmul collapse" \
  --body-file /data/henriksson/github/claude/stardist-rs/prs/09-cuda-gemm-broadcast-collapse.md
```

If PR 8 is merged first, rebase this branch onto upstream `main` and open it against `main`.

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

Validated locally with:

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

git diff --check cuda-sm75-compat..cuda-f32-gemm-broadcast-collapse
```

## Benchmark

Shape: `[64, 256, 256] @ [256, 256]`

| Branch | Time |
| --- | ---: |
| `cuda-sm75-compat` | 3.648 ms |
| This PR | 2.946 ms |

Ratio: 1.24x faster.

## Review Notes

- The helper is intentionally general across floating point element types.
- The optimization is gated by layout checks; non-matching layouts continue through the existing path.
