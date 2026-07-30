# PR 1: CPU Conv2d 1x1 Fast Path

Branch: `cpu-conv2d-1x1-fast-path`  
Base: `main`  
Commit: `e370f71b`

## Suggested Title

Optimize CPU conv2d 1x1 batch-1 fast path

## Push And Open

```bash
cd /home/mahogny/github/claude/candle
git checkout cpu-conv2d-1x1-fast-path
git status --short
git push -u <your-fork-remote> cpu-conv2d-1x1-fast-path
gh pr create --base main --head <your-github-user>:cpu-conv2d-1x1-fast-path --title "Optimize CPU conv2d 1x1 batch-1 fast path" --body-file /data/henriksson/github/claude/stardist-rs/prs/01-cpu-conv2d-1x1-fast-path.md
```

## Suggested PR Body

### Summary

This adds a narrow CPU Conv2d fast path for contiguous NCHW batch-1 `1x1`
convolutions.

For this case the GEMM result is already in the desired output order
`[1, c_out, h, w]`, so the implementation can return it directly instead of
doing extra reshaping/copying.

### Scope

- Avoid reshaping contiguous NCHW input for batch-1 `1x1` convolutions.
- Return the GEMM result directly when the memory order already matches the
  output layout.
- Preserve existing behavior for non-contiguous and multi-batch inputs.

### Validation

- `cargo fmt`
- `cargo test -p candle-core conv2d`
- `cargo fmt --check`
- `git diff --check`

### Benchmark

Release microbenchmark on 2026-07-29:

| Shape | `main` | PR branch | Ratio |
| --- | ---: | ---: | ---: |
| `[1, 64, 192, 192] * [64, 64, 1, 1]` | 81.654 ms | 24.512 ms | 3.33x faster |

Checksums matched.

### Audit

Two audit passes found no major or medium issues. The diff is limited to the CPU
`1x1` Conv2d path and preserves the existing fallback paths.
