# PR 12: SAM Image Encoder Batched Position Embeddings

Branch: `sam-image-encoder-batched-pos-embed`
Base: `main`
Head commit: `e51298c9` on local `stardist-integration`

Suggested title:

```text
Fix batched absolute positional embeddings in SAM image encoder
```

Upload:

```bash
cd /home/mahogny/github/claude/candle
git checkout -b sam-image-encoder-batched-pos-embed main
git cherry-pick e51298c9
git push -u origin sam-image-encoder-batched-pos-embed
gh pr create \
  --base main \
  --head <your-github-user>:sam-image-encoder-batched-pos-embed \
  --title "Fix batched absolute positional embeddings in SAM image encoder" \
  --body-file /data/henriksson/github/claude/stardist-rs/prs/12-sam-image-encoder-batched-pos-embed.md
```

Suggested PR body:

## Summary

Use `broadcast_add` when adding absolute positional embeddings in `ImageEncoderViT`.

The SAM image encoder stores absolute position embeddings with shape
`[1, h, w, embed_dim]`, while batched image inputs produce patch embeddings with
shape `[batch, h, w, embed_dim]`. Plain `+` only works when `batch == 1`; batched
inputs should broadcast the leading dimension.

## Motivation

Batched SAM image encoder inference currently fails with a shape mismatch when
absolute positional embeddings are enabled:

```text
shape mismatch in add, lhs: [2, 32, 32, 1024], rhs: [1, 32, 32, 1024]
```

This is a model correctness fix, not a Cellpose-specific optimization.

## Scope

- Change the absolute positional embedding add from `xs + pos_embed` to
  `xs.broadcast_add(pos_embed)`.
- Add CPU and CUDA regression tests using a tiny zero-initialized
  `ImageEncoderViT` with batch size 2.
- Leave timing/profiling instrumentation out of this PR.

## Validation

Validated locally with:

```bash
cargo fmt
cargo test -p candle-transformers image_encoder_abs_pos_supports_batched_inputs_cpu

CUDA_ROOT=/usr/local/cuda-12.8 \
CUDA_COMPUTE_CAP=75 \
CUDA_HOME=/usr/local/cuda-12.8 \
NVCC=/usr/local/cuda-12.8/bin/nvcc \
LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:/usr/local/cuda-12.8/targets/x86_64-linux/lib:$LD_LIBRARY_PATH \
PATH=/usr/local/cuda-12.8/bin:$PATH \
cargo test -p candle-transformers --features cuda image_encoder_abs_pos_supports_batched_inputs_cuda -- --nocapture
```

Downstream validation in `cellpose-rs`, temporarily using the local Candle
checkout:

```bash
cargo run --release --features cuda --example candle_batch_probe -- --use-gpu --batch 2
make bench-inference-rust BENCH_IMAGE_LIST=demodata/demo5_images.txt BENCH_RUNS=1 BENCH_RUST_EXTRA_ARGS=--no-timing
```

The downstream batch-2 encoder probe succeeds after this change. The demo5
Cellpose benchmark with default batch size 8 also succeeds with exact cell-count
parity against the saved Python baseline.

## Benchmark Context

On Quadro RTX 5000 / CUDA 12.8 / `sm75`, downstream `cellpose-rs` demo5
throughput with this fix measured:

| Implementation | Total eval | Rust/Python | Peak RSS |
| --- | ---: | ---: | ---: |
| Python/PyTorch saved baseline | 5.154s | 1.000x | 2,057,088 KB |
| Rust/Candle with this fix | 6.049s | 1.188x | 1,503,620 KB |

This PR's primary purpose is correctness for batched SAM inputs. The benchmark is
included only to show that the downstream batched path is restored.

## Review Notes

- The change is intentionally limited to SAM `ImageEncoderViT`; no generic
  operator semantics are changed.
- Other Candle transformer models already use `broadcast_add` for comparable
  positional embedding and mask additions.
- This can be reviewed independently from the CUDA performance PR stack.
