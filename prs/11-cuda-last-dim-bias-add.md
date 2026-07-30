# PR 11: CUDA F32 Last-Dim Bias Add Fast Path

Branch: `cuda-linear-bias-add-f32`
Base: `cuda-sm75-compat`
Head commit: `635f8f43`

Suggested title:

```text
Add CUDA f32 last-dim bias add fast path
```

Upload:

```bash
cd /home/mahogny/github/claude/candle
git checkout cuda-linear-bias-add-f32
git push -u origin cuda-linear-bias-add-f32
gh pr create \
  --base cuda-sm75-compat \
  --head <your-github-user>:cuda-linear-bias-add-f32 \
  --title "Add CUDA f32 last-dim bias add fast path" \
  --body-file /data/henriksson/github/claude/stardist-rs/prs/11-cuda-last-dim-bias-add.md
```

If PR 8 is merged first, rebase this branch onto upstream `main` and open it against `main`.

Suggested PR body:

## Summary

This adds a private CUDA helper for adding a 1D F32 bias across the last dimension of a contiguous tensor.

`candle-nn::Linear` uses the helper when its output and bias match that layout, with the existing generic addition path kept as fallback.

## Motivation

Linear layers commonly produce a contiguous output followed by a last-dimension bias add. The generic broadcast add is correct but carries extra overhead for this simple case.

## Scope

- Add a CUDA kernel for contiguous F32 last-dimension bias add.
- Add a private reusable helper, `bias_add_last_dim`.
- Use the helper from `Linear`.
- Keep explicit preflight checks so real CUDA errors are not swallowed by fallback behavior.

## Validation

Validated locally with:

```bash
cargo fmt
cargo test -p candle-nn bias_add

CUDA_HOME=/usr/local/cuda-12.8 \
CUDA_PATH=/usr/local/cuda-12.8 \
NVCC=/usr/local/cuda-12.8/bin/nvcc \
CUDA_COMPUTE_CAP=75 \
LD_LIBRARY_PATH=/usr/local/cuda-12.8/targets/x86_64-linux/lib:$LD_LIBRARY_PATH \
cargo test -p candle-nn --features cuda bias_add

CUDA_HOME=/usr/local/cuda-12.8 \
CUDA_PATH=/usr/local/cuda-12.8 \
NVCC=/usr/local/cuda-12.8/bin/nvcc \
CUDA_COMPUTE_CAP=75 \
LD_LIBRARY_PATH=/usr/local/cuda-12.8/targets/x86_64-linux/lib:$LD_LIBRARY_PATH \
cargo check -p candle-nn --features cuda

git diff --check cuda-sm75-compat..cuda-linear-bias-add-f32
```

## Benchmark

Linear forward shape: input `[16384, 64]`, weight `[1024, 64]`, bias `[1024]`

| Branch | Time |
| --- | ---: |
| `cuda-sm75-compat` | 10.689 ms |
| This PR | 10.138 ms |

Ratio: 1.05x faster.

## Review Notes

- This was generalized from a Linear-only special case into a private last-dimension bias helper.
- Direct helper tests cover 3D tensors in addition to Linear module tests.
