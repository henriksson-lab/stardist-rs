# PR 8: CUDA sm75/CUDA 12.8 Kernel Compatibility

Branch: `cuda-sm75-compat`
Base: `main`
Head commit: `005ea216`

Suggested title:

```text
Fix CUDA 12.8 half compatibility guards
```

Upload:

```bash
cd /home/mahogny/github/claude/candle
git checkout cuda-sm75-compat
git push -u origin cuda-sm75-compat
gh pr create \
  --base main \
  --head <your-github-user>:cuda-sm75-compat \
  --title "Fix CUDA 12.8 half compatibility guards" \
  --body-file /data/henriksson/github/claude/stardist-rs/prs/08-cuda-sm75-compat.md
```

Suggested PR body:

## Summary

This updates the CUDA kernel compatibility header so Candle builds on sm75 hardware with CUDA 12.8.

The change is intentionally small and isolated to `candle-kernels/src/compatibility.cuh`.

## Motivation

CUDA 12.8 exposes newer half-related APIs through headers that can trip builds on older architectures such as Turing/sm75. This patch keeps the compatibility guards explicit so users with still-supported NVIDIA cards can build Candle CUDA kernels.

## Scope

- Add compatibility guards for the affected CUDA half APIs.
- Keep the change local to the shared CUDA compatibility header.
- Do not change runtime behavior for newer architectures beyond the guarded compatibility path.

## Validation

Validated locally with:

```bash
CUDA_HOME=/usr/local/cuda-12.8 \
CUDA_PATH=/usr/local/cuda-12.8 \
NVCC=/usr/local/cuda-12.8/bin/nvcc \
CUDA_COMPUTE_CAP=75 \
LD_LIBRARY_PATH=/usr/local/cuda-12.8/targets/x86_64-linux/lib:$LD_LIBRARY_PATH \
cargo check -p candle-core --features cuda

git diff --check
```

## Review Notes

- This PR was audited twice locally.
- I only validated on sm75/CUDA 12.8 hardware. Review from someone with sm80+ hardware would be useful before merge.
