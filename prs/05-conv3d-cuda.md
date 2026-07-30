# PR 5: CUDA Conv3d Backend

Branch: `conv3d-cuda`  
Base: `conv3d-core-cpu`  
Commit: `75d13c2f`

## Suggested Title

Add CUDA Conv3d backend

## Push And Open

Open this after PR 4 is open or merged.

```bash
cd /home/mahogny/github/claude/candle
git checkout conv3d-cuda
git status --short
git push -u <your-fork-remote> conv3d-cuda
gh pr create --base conv3d-core-cpu --head <your-github-user>:conv3d-cuda --title "Add CUDA Conv3d backend" --body-file /data/henriksson/github/claude/stardist-rs/prs/05-conv3d-cuda.md
```

If PR 4 is merged first, rebase this branch onto updated `main` and open against
`main`.

## Suggested PR Body

### Summary

This wires Conv3d for CUDA using an `im2col3d` kernel plus GEMM, and adds cuDNN
Conv3d launch support where available.

### Scope

- Add CUDA `im2col3d`.
- Use per-axis depth/height/width stride, padding, and dilation.
- Wire CUDA backend `conv3d`.
- Add cuDNN Conv3d descriptor/launch path.
- Add CUDA parity tests using the same reference cases as CPU.

### Validation

Validated locally on Quadro RTX 5000 / sm75 using CUDA 12.8 through
`validation/conv3d-cuda-sm75`, which combines this clean PR branch with the
separate sm75 compatibility branch.

- `cargo fmt`
- `CUDA_HOME=/usr/local/cuda-12.8 CUDA_PATH=/usr/local/cuda-12.8 NVCC=/usr/local/cuda-12.8/bin/nvcc CUDA_COMPUTE_CAP=75 cargo test -p candle-core --features cuda conv3d`
- `CUDA_HOME=/usr/local/cuda-12.8 CUDA_PATH=/usr/local/cuda-12.8 NVCC=/usr/local/cuda-12.8/bin/nvcc CUDA_COMPUTE_CAP=75 cargo check -p candle-core --features cuda,cudnn`
- `git diff --check`

`cargo test -p candle-core --features cuda,cudnn conv3d` was attempted, but
local linking failed because `libcudnn` is not installed/found. The cuDNN Rust
path compiled under `cargo check`.

### Audit

Two audit passes found no major or medium issues. The branch diff against
`conv3d-core-cpu` is limited to CUDA Conv3d code and tests.
