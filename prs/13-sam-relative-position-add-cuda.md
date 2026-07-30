# PR 13: SAM Relative-Position CUDA Add

Branch: `sam-relative-position-add-cuda`
Base: `sam-image-encoder-batched-pos-embed`
Head commit: not yet committed on local `stardist-integration`

Suggested title:

```text
Add CUDA fast path for SAM relative-position attention add
```

Upload:

```bash
cd /home/mahogny/github/claude/candle
git checkout -b sam-relative-position-add-cuda sam-image-encoder-batched-pos-embed
# Apply only the Add3Inplace CUDA kernel, cached PTX, and focused CUDA test.
# Do not include temporary ImageEncoderViTTiming instrumentation in this PR.
git push -u origin sam-relative-position-add-cuda
gh pr create \
  --base sam-image-encoder-batched-pos-embed \
  --head <your-github-user>:sam-relative-position-add-cuda \
  --title "Add CUDA fast path for SAM relative-position attention add" \
  --body-file /data/henriksson/github/claude/stardist-rs/prs/13-sam-relative-position-add-cuda.md
```

Suggested PR body:

## Summary

Add a CUDA F32 fast path for the SAM image encoder's decomposed
relative-position attention add.

The existing CPU path already uses a custom `Add3` op to compute:

```text
attn[b, qh, qw, kh, kw] + rel_h[b, qh, qw, kh] + rel_w[b, qh, qw, kw]
```

This PR adds the equivalent CUDA path as an in-place custom op for contiguous
F32 tensors, with the existing broadcast expression retained as fallback for
non-CUDA devices.

## Motivation

For SAM image encoder inference, the generic CUDA broadcast expression for the
relative-position add is correct but slow enough to show up as a measurable
downstream bottleneck. The operation has a simple, fixed indexing pattern, so a
dedicated kernel avoids extra broadcast materialization and cuts launch/work
overhead.

This is not Cellpose-specific. It applies to Candle's SAM image encoder
whenever relative-position attention is enabled on CUDA.

## Scope

- Add `Add3Inplace` behind `#[cfg(feature = "cuda")]`.
- Use it only when `attn.device().is_cuda()`.
- Keep the existing CPU `Add3` path unchanged.
- Keep the generic broadcast fallback for non-CPU, non-CUDA devices.
- Cache the NVRTC-compiled PTX with `OnceLock` so inference does not recompile
  the kernel inside every attention block.
- Add a focused CUDA test comparing the in-place kernel against the existing
  broadcast expression.
- Do not include temporary timing/profiling APIs in this PR.

## Validation

Validated locally with:

```bash
cargo fmt
cargo check -p candle-transformers

CUDA_ROOT=/usr/local/cuda-12.8 \
CUDA_COMPUTE_CAP=75 \
CUDA_HOME=/usr/local/cuda-12.8 \
NVCC=/usr/local/cuda-12.8/bin/nvcc \
LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:/usr/local/cuda-12.8/targets/x86_64-linux/lib:$LD_LIBRARY_PATH \
PATH=/usr/local/cuda-12.8/bin:$PATH \
cargo test -p candle-transformers --features cuda add3_inplace_cuda_matches_broadcast_expression -- --nocapture
```

Downstream validation in `cellpose-rs`, temporarily using the local Candle
checkout:

```bash
CUDA_ROOT=/usr/local/cuda-12.8 \
CUDA_COMPUTE_CAP=75 \
CUDA_HOME=/usr/local/cuda-12.8 \
NVCC=/usr/local/cuda-12.8/bin/nvcc \
LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:/usr/local/cuda-12.8/targets/x86_64-linux/lib:$LD_LIBRARY_PATH \
PATH=/usr/local/cuda-12.8/bin:$PATH \
CARGO_TARGET_DIR=/data/henriksson/github/ripp/cellpose-rs/target/cuda12 \
cargo check --features cuda --example encoder_benchmark
```

## Benchmark Context

On Quadro RTX 5000 / CUDA 12.8 / `sm75`, downstream `cellpose-rs` CP-SAM
encoder benchmark with batch size 8 measured:

| Implementation | Encoder mean | Rust/Python |
| --- | ---: | ---: |
| Python/PyTorch baseline | 0.9668s | 1.000x |
| Rust/Candle before this PR | 1.1884s | 1.229x |
| Rust/Candle with this PR | 1.0189s | 1.054x |

The relative-position timing probe moved from `0.879s` before the CUDA fast path
to `0.179s` after caching PTX compilation, compared with `0.166s` in the
Python/PyTorch baseline. Timing probes include synchronization and are included
only to localize the bottleneck.

Downstream full demo5 benchmark:

| Implementation | Total eval | Rust/Python | Peak RSS |
| --- | ---: | ---: | ---: |
| Python/PyTorch saved baseline | 5.089s | 1.000x | 2,057,444 KB |
| Rust/Candle with this PR | 5.495s | 1.080x | 1,502,204 KB |

## Review Notes

- This should be reviewed after PR 12 because batched SAM encoder execution must
  work before the downstream batch-8 benchmark is meaningful.
- The kernel is intentionally narrow: contiguous CUDA F32 tensors in the SAM
  relative-position layout.
- The PTX cache is required. Compiling with NVRTC inside every `cuda_fwd` call
  caused a full-inference regression.
- A future cleanup can replace runtime NVRTC with a generated Candle kernel
  module if maintainers prefer that style.
