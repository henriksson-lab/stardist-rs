# PR 3: CPU Conv2d 3x3 Same-Padding Fast Path

Branch: `cpu-conv2d-3x3-same-fast-path`  
Base: `cpu-conv2d-tiled-direct-strides`  
Commit: `b7900d83`

## Suggested Title

Specialize tiled CPU conv2d for 3x3 same padding

## Push And Open

Open this after PR 2 is open or merged.

```bash
cd /home/mahogny/github/claude/candle
git checkout cpu-conv2d-3x3-same-fast-path
git status --short
git push -u <your-fork-remote> cpu-conv2d-3x3-same-fast-path
gh pr create --base cpu-conv2d-tiled-direct-strides --head <your-github-user>:cpu-conv2d-3x3-same-fast-path --title "Specialize tiled CPU conv2d for 3x3 same padding" --body-file /data/henriksson/github/claude/stardist-rs/prs/03-cpu-conv2d-3x3-same-fast-path.md
```

If PR 2 is merged first, rebase this branch onto updated `main` and open against
`main`.

## Suggested PR Body

### Summary

This adds an interior-pixel fast path for the common `3x3`, stride-1,
padding-1, dilation-1 tiled CPU Conv2d case.

Border pixels continue using the generic path.

### Scope

- Specialize only the common `k=3`, `stride=1`, `padding=1`, `dilation=1` case.
- Use the fast path only for interior pixels.
- Keep border handling on the generic implementation.

### Validation

- `cargo fmt`
- `cargo test -p candle-core conv2d`
- `cargo test -p candle-core conv2d_c_eq_h_eq_w`
- `cargo test -p candle-core conv2d_noncontiguous_input`
- `git diff --check`

### Benchmark

Release microbenchmark on 2026-07-29:

| Shape | `main` | PR 2 | PR 3 |
| --- | ---: | ---: | ---: |
| `[1, 32, 192, 192] * [64, 32, 3, 3]`, padding 1 | 70.953 ms | 55.370 ms | 44.814 ms |

PR 3 is 1.58x faster than `main` and 1.24x faster than PR 2. Checksums matched.

### Audit

Two audit passes found no major or medium issues. The incremental diff against
PR 2 is a one-file guarded interior specialization.
