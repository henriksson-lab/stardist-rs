# PR 10: CUDA F32 Softmax Last-Dim 1024 Fast Path

Branch: `cuda-softmax-1024-f32`
Base: `cuda-sm75-compat`
Head commit: `ccae8ef6`

Suggested title:

```text
Add CUDA f32 softmax 1024 fast path
```

Upload:

```bash
cd /home/mahogny/github/claude/candle
git checkout cuda-softmax-1024-f32
git push -u origin cuda-softmax-1024-f32
gh pr create \
  --base cuda-sm75-compat \
  --head <your-github-user>:cuda-softmax-1024-f32 \
  --title "Add CUDA f32 softmax 1024 fast path" \
  --body-file /data/henriksson/github/claude/stardist-rs/prs/10-cuda-softmax-1024-f32.md
```

If PR 8 is merged first, rebase this branch onto upstream `main` and open it against `main`.

Suggested PR body:

## Summary

This adds a CUDA F32 fast path for softmax over a contiguous last dimension of size 1024.

The existing implementation remains the fallback for all other shapes, dtypes, and layouts.

## Motivation

Softmax with a 1024-wide last dimension is common enough to benefit from a specialized path. The fast path reduces overhead for the exact layout where a single CUDA block can handle one row efficiently.

## Scope

- Add a specialized CUDA F32 softmax kernel for last dimension 1024.
- Dispatch only when dtype, shape, and contiguity match.
- Preserve existing fallback behavior for every other case.

## Validation

Validated locally with:

```bash
cargo fmt

CUDA_HOME=/usr/local/cuda-12.8 \
CUDA_PATH=/usr/local/cuda-12.8 \
NVCC=/usr/local/cuda-12.8/bin/nvcc \
CUDA_COMPUTE_CAP=75 \
LD_LIBRARY_PATH=/usr/local/cuda-12.8/targets/x86_64-linux/lib:$LD_LIBRARY_PATH \
cargo test -p candle-core --features cuda softmax_last_dim_1024_f32

CUDA_HOME=/usr/local/cuda-12.8 \
CUDA_PATH=/usr/local/cuda-12.8 \
NVCC=/usr/local/cuda-12.8/bin/nvcc \
CUDA_COMPUTE_CAP=75 \
LD_LIBRARY_PATH=/usr/local/cuda-12.8/targets/x86_64-linux/lib:$LD_LIBRARY_PATH \
cargo check -p candle-core --features cuda

git diff --check cuda-sm75-compat..cuda-softmax-1024-f32
```

## Benchmark

Shape: `[8192, 1024]`

| Branch | Time |
| --- | ---: |
| `cuda-sm75-compat` | 6.194 ms |
| This PR | 4.789 ms |

Ratio: 1.29x faster.

## Review Notes

This is intentionally a narrow specialization. It should be easy to review and can be generalized later if Candle wants broader softmax tiling coverage.
