# PR 7: candle-nn Conv3d Module

Branch: `nn-conv3d`  
Base: `conv3d-core-cpu`  
Commit: `8e0fa052`

## Suggested Title

Add candle-nn Conv3d module

## Push And Open

Open this after PR 4 is open or merged.

```bash
cd /home/mahogny/github/claude/candle
git checkout nn-conv3d
git status --short
git push -u <your-fork-remote> nn-conv3d
gh pr create --base conv3d-core-cpu --head <your-github-user>:nn-conv3d --title "Add candle-nn Conv3d module" --body-file /data/henriksson/github/claude/stardist-rs/prs/07-nn-conv3d.md
```

If PR 4 is merged first, rebase this branch onto updated `main` and open against
`main`.

## Suggested PR Body

### Summary

This adds a high-level `candle-nn` Conv3d module on top of the core Conv3d API.

### Scope

- Add `Conv3d` and `Conv3dConfig`.
- Add `conv3d` and `conv3d_no_bias` constructors.
- Mirror Conv1d/Conv2d initialization and naming patterns.
- Use `[usize; 3]` for padding, stride, and dilation.
- Add tests for bias and no-bias paths.

### Validation

- `cargo fmt`
- `cargo test -p candle-nn conv3d`
- `cargo test -p candle-nn`
- `git diff --check`

### Audit

Two audit passes found no major or medium issues. The config uses per-axis 3D
parameters, bias broadcasting uses `(1, channels, 1, 1, 1)`, and public
re-exports are present.
