# PR 2: CPU Conv2d Tiled Direct-Stride Reads

Branch: `cpu-conv2d-tiled-direct-strides`  
Base: `main`  
Commit: `8d228de6`

## Suggested Title

Avoid input repack in tiled CPU conv2d

## Push And Open

```bash
cd /home/mahogny/github/claude/candle
git checkout cpu-conv2d-tiled-direct-strides
git status --short
git push -u <your-fork-remote> cpu-conv2d-tiled-direct-strides
gh pr create --base main --head <your-github-user>:cpu-conv2d-tiled-direct-strides --title "Avoid input repack in tiled CPU conv2d" --body-file /data/henriksson/github/claude/stardist-rs/prs/02-cpu-conv2d-tiled-direct-strides.md
```

## Suggested PR Body

### Summary

This removes the full NCHW-to-NHWC input repack from tiled CPU Conv2d. The tiled
kernel reads input directly through the source layout strides instead.

### Scope

- Remove the full input repack from `conv2d_tiled`.
- Read input directly using layout strides.
- Preserve tiled output ordering and padding/stride/dilation behavior.
- Keep non-contiguous input coverage.

### Validation

- `cargo fmt`
- `cargo test -p candle-core conv2d`
- `cargo test -p candle-core conv2d_noncontiguous_input`
- `git diff --check`

### Benchmark

Release microbenchmark on 2026-07-29:

| Shape | `main` | PR branch | Ratio |
| --- | ---: | ---: | ---: |
| `[1, 32, 192, 192] * [64, 32, 3, 3]`, padding 1 | 70.953 ms | 55.370 ms | 1.28x faster |

Checksums matched.

### Audit

Two audit passes found no major or medium issues. The change is scoped to tiled
direct-stride reads plus focused non-contiguous input coverage.
