# PR 6: Metal Conv3d Backend

Branch: `conv3d-metal`  
Base: `conv3d-core-cpu`  
Commit: `7e343ad2`

## Suggested Title

Add Metal Conv3d backend

## Push And Open

Run the macOS validation below before opening this PR.

```bash
cd /home/mahogny/github/claude/candle
git checkout conv3d-metal
git status --short
git push -u <your-fork-remote> conv3d-metal
gh pr create --base conv3d-core-cpu --head <your-github-user>:conv3d-metal --title "Add Metal Conv3d backend" --body-file /data/henriksson/github/claude/stardist-rs/prs/06-conv3d-metal.md
```

If PR 4 is merged first, rebase this branch onto updated `main` and open against
`main`.

## Suggested PR Body

### Summary

This wires Conv3d for Metal using an `im2col3d` shader plus matmul.

### Scope

- Add Metal `im2col3d` shader.
- Add Rust launcher `call_im2col3d_strided`.
- Wire Metal backend `conv3d`.
- Use per-axis depth/height/width stride, padding, and dilation.
- Add Metal parity test wrapper using the shared Conv3d reference cases.

### Validation

Linux-valid checks:

- `cargo fmt`
- `cargo test -p candle-core conv3d`
- `git diff --check`
- `git diff --check conv3d-core-cpu..conv3d-metal`

Linux cannot compile or run Metal. On macOS/Metal, run before submission:

```bash
cargo test -p candle-core --features metal conv3d
```

### Audit

Three source-level audit passes found no major or medium issues. The final audit
checked the branch against `conv3d-core-cpu` and confirmed the diff is limited
to:

- `candle-core/src/metal_backend/mod.rs`
- `candle-core/tests/conv_tests.rs`
- `candle-metal-kernels/src/kernels/convolution.rs`
- `candle-metal-kernels/src/metal_src/conv.metal`

Shader ABI, Rust launcher argument order, backend layout transforms, per-axis
parameters, and shared tests line up.
