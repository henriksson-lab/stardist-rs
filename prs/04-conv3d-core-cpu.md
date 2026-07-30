# PR 4: Core Conv3d API And CPU Backend

Branch: `conv3d-core-cpu`  
Base: `main`  
Commit: `b1814338`

## Suggested Title

Add core Conv3d API and CPU backend

## Push And Open

```bash
cd /home/mahogny/github/claude/candle
git checkout conv3d-core-cpu
git status --short
git push -u <your-fork-remote> conv3d-core-cpu
gh pr create --base main --head <your-github-user>:conv3d-core-cpu --title "Add core Conv3d API and CPU backend" --body-file /data/henriksson/github/claude/stardist-rs/prs/04-conv3d-core-cpu.md
```

## Suggested PR Body

### Summary

This adds the core Tensor Conv3d API and a CPU direct Conv3d backend.

The API supports per-axis `[usize; 3]` `padding`, `stride`, and `dilation`,
while retaining a scalar convenience wrapper consistent with existing Conv1d
and Conv2d APIs.

### Scope

- Add `ParamsConv3D`.
- Add `Tensor::conv3d` and `Tensor::conv3d_with_algo`.
- Add backend trait/storage plumbing.
- Add CPU direct Conv3d implementation.
- Add dummy CUDA/Metal backend stubs.
- Add reference tests, including anisotropic stride and grouped Conv3d.
- Add structured `Conv3dInvalidArgs` error.

### Validation

- `cargo fmt`
- `cargo test -p candle-core conv3d`
- `cargo test -p candle-core conv3d_cpu_plan_cases`
- `cargo test -p candle-core conv`
- `git diff --check`

Note: `cargo check -p candle-core --features cuda` was attempted on this branch
but local CUDA headers failed in `candle-kernels` before reaching the Rust
backend changes because `/usr/include` lacks `cuda_fp8.h`.

### Audit

Two audit passes found no major or medium issues. The API is generalized to
per-axis 3D parameters and shared reference tests are ready for CUDA/Metal
follow-ups.
