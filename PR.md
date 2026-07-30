# Candle Upstream PR Plan

Fork: `/home/mahogny/github/claude/candle`

Goal: split the Candle changes needed by `stardist-rs` into reviewable upstream
pull requests. Keep each PR focused on one problem, with focused tests and
benchmarks where relevant.

## PR 1: CPU conv2d 1x1 fast path

Branch: `cpu-conv2d-1x1-fast-path`

Status: implemented and committed in the fork as `e370f71b`.

Scope:

- Avoid reshaping contiguous NCHW input for batch-1 `1x1` convolutions.
- Return the GEMM result directly for batch size 1 when the memory order already
  matches `[1, c_out, h, w]`.
- Keep existing behavior for non-contiguous and multi-batch inputs.

Validation:

- `cargo fmt`
- `cargo test -p candle-core conv2d` passed locally.
- `cargo fmt --check` passed locally on the branch.
- Two audit passes completed without major or medium findings:
  - audit 1 checked the branch diff and confirmed it is limited to the CPU
    `1x1` Conv2d path with existing behavior preserved for non-contiguous input;
  - audit 2 checked the changed file set, `git diff --check`, and reran the
    focused Conv2d tests directly on the branch.
- Microbenchmark on 2026-07-29 with release build, shape
  `[1, 64, 192, 192] * [64, 64, 1, 1]`: upstream `main` 81.654 ms/iter,
  PR branch 24.512 ms/iter, 3.33x faster. Checksums matched.

Reasoning:

This is a small CPU-only performance patch with low review risk. It is
independent of tiled im2col and Conv3d.

## PR 2: CPU conv2d tiled direct-stride input reads

Branch: `cpu-conv2d-tiled-direct-strides`

Status: implemented and committed in the fork as `8d228de6`.

Scope:

- Remove the full NCHW-to-NHWC input repack from `conv2d_tiled`.
- Read input directly using the source layout strides.
- Preserve current tiled output ordering and all padding/stride/dilation
  semantics.

Validation:

- `cargo fmt`
- `cargo test -p candle-core conv2d` passed locally.
- `cargo test -p candle-core conv2d_noncontiguous_input` passed locally.
- Two audit passes completed without major or medium findings:
  - audit 1 checked the branch diff and confirmed the change is limited to
    tiled direct-stride reads plus missing non-contiguous input coverage;
  - audit 2 checked the focused regression, changed file set, and
    `git diff --check`.
- Microbenchmark on 2026-07-29 with release build, shape
  `[1, 32, 192, 192] * [64, 32, 3, 3]`, padding 1: upstream `main`
  70.953 ms/iter, PR branch 55.370 ms/iter, 1.28x faster. Checksums matched.

Reasoning:

This is the main generic CPU improvement: it removes a full input copy and
reduces peak memory. Current Candle `main` already includes the non-contiguous
tiled correctness fix, so this PR should be framed as a performance follow-up.

## PR 3: CPU conv2d 3x3 same-padding interior fast path

Branch: `cpu-conv2d-3x3-same-fast-path`

Status: implemented as a stacked branch on top of
`cpu-conv2d-tiled-direct-strides` and committed in the fork as `b7900d83`.

Scope:

- Specialize the common `k=3`, `stride=1`, `padding=1`, `dilation=1` case.
- Use the fast path only for interior pixels.
- Keep border pixels on the generic path.

Validation:

- `cargo fmt`
- `cargo test -p candle-core conv2d` passed locally.
- `cargo test -p candle-core conv2d_c_eq_h_eq_w` passed locally.
- `cargo test -p candle-core conv2d_noncontiguous_input` passed locally.
- Two audit passes completed without major or medium findings:
  - audit 1 checked the incremental diff against PR 2 and confirmed it is a
    one-file guarded interior specialization;
  - audit 2 reran the tests that exercise the `3x3`, stride-1, padding-1 path.
- Microbenchmark on 2026-07-29 with release build, shape
  `[1, 32, 192, 192] * [64, 32, 3, 3]`, padding 1: upstream `main`
  70.953 ms/iter, PR 2 branch 55.370 ms/iter, PR 3 branch 44.814 ms/iter.
  PR 3 is 1.58x faster than `main` and 1.24x faster than PR 2. Checksums
  matched.

Reasoning:

This is more specialized than PR 2. Keeping it separate lets upstream accept the
generic direct-stride improvement even if they want more evidence for the `3x3`
specialization.

## PR 4: Core Conv3d API and CPU backend

Branch: `conv3d-core-cpu`

Status: implemented and committed in the fork as `b1814338`.

Scope:

- Add `ParamsConv3D`.
- Add `Tensor::conv3d` and `Tensor::conv3d_with_algo`.
- Add the backend trait/storage plumbing.
- Add a CPU direct Conv3d implementation.
- Add dummy CUDA/Metal backend stubs as needed.
- Add tests comparing Candle output with a small reference implementation.

Audit notes before implementation:

- Do not upstream the vendored scalar-only API as-is. StarDist already needs
  anisotropic 3D stride/padding handling, and ordinary 3D models often differ
  across depth/height/width. Use `[usize; 3]` internally for `padding`,
  `stride`, and `dilation`.
- To stay ergonomic and consistent with existing Candle scalar conv APIs, expose
  a scalar convenience wrapper plus a full 3D variant, for example
  `conv3d(...)` for scalar parameters and `conv3d_with_params(...)` or
  `conv3d_with_algo(...)` accepting `[usize; 3]`.
- Add a structured `Conv3dInvalidArgs` error variant instead of a plain
  `bail!` string for channel/group mismatches.
- Update the module comment in `candle-core/src/conv.rs`; current upstream says
  "1D and 2D Convolutions".
- Reuse the same reference-case tests for CPU, CUDA, and Metal. Include at least
  one anisotropic stride case such as `[1, 2, 2]`.

Validation:

- `cargo fmt`
- `cargo test -p candle-core conv3d` passed locally.
- `cargo test -p candle-core conv3d_cpu_plan_cases` passed locally.
- `cargo test -p candle-core conv` passed locally.
- `git diff --check` passed before commit.
- `cargo check -p candle-core --features cuda` was attempted, but failed in
  `candle-kernels` before reaching the Rust backend changes because local CUDA
  headers do not include `cuda_fp8.h`.
- Two audit passes completed without major or medium findings:
  - audit 1 checked the API shape, backend trait plumbing, structured error,
    CPU stride-aware implementation, and anisotropic reference tests;
  - audit 2 added grouped Conv3d coverage, reran the focused and broad
    convolution tests, checked the final diff, and confirmed only expected files
    were staged.

Reasoning:

This establishes the public API and correctness contract before introducing GPU
implementations. CUDA and Metal can then compare against the same CPU reference.

## PR 5: CUDA Conv3d backend

Branch: `conv3d-cuda`

Status: implemented as a stacked branch on top of `conv3d-core-cpu` and
committed in the fork as `75d13c2f`.

Scope:

- Add CUDA `im2col3d` support.
- Add cuDNN Conv3d launch support where available.
- Wire CUDA backend `conv3d`.
- Add CUDA tests comparing against CPU for small cases.

Audit notes before implementation:

- The vendored CUDA `im2col3d` path also uses scalar stride/padding/dilation.
  Generalize the kernel arguments to separate depth/height/width values before
  upstreaming.
- Keep cuDNN and non-cuDNN paths behaviorally equivalent. If cuDNN only accepts
  a subset of parameter combinations, document the fallback.

Validation:

- `cargo fmt`
- `CUDA_HOME=/usr/local/cuda-12.8 CUDA_PATH=/usr/local/cuda-12.8
  NVCC=/usr/local/cuda-12.8/bin/nvcc CUDA_COMPUTE_CAP=75
  cargo test -p candle-core --features cuda conv3d` passed locally on
  validation branch `validation/conv3d-cuda-sm75`, which combines this clean
  PR branch with the separate sm75 compatibility patch.
- `cargo check -p candle-core --features cuda,cudnn` passed locally with the
  same CUDA 12.8/sm75 environment on `validation/conv3d-cuda-sm75`.
- `cargo test -p candle-core --features cuda,cudnn conv3d` was attempted, but
  local linking failed because `libcudnn` is not installed/found. This is a
  local environment gap; the cuDNN Rust path compiled under `cargo check`.
- `git diff --check` passed before commit.
- Two audit passes completed without major or medium findings:
  - audit 1 checked the CUDA kernel ABI, per-axis parameter plumbing,
    non-cuDNN matmul route, cuDNN descriptor route, and CUDA parity tests;
  - audit 2 wrapped the long kernel macro call, reran CUDA Conv3d tests,
    compile-checked cuDNN, and confirmed the staged file set was limited to
    CUDA Conv3d code and tests.
- Local environment: Quadro RTX 5000 / sm75, CUDA 12.8 toolkit via
  `/usr/local/cuda-12.8`.

Reasoning:

This is hardware-specific and should be reviewed separately from the CPU API.

## PR 6: Metal Conv3d backend

Branch: `conv3d-metal`

Status: implemented and committed in the fork as `7e343ad2`.

Scope:

- Add Metal `im2col3d` support.
- Wire Metal backend `conv3d`.
- Add Metal tests using the same reference cases as CPU/CUDA.

Audit notes before implementation:

- The vendored Metal `im2col3d` path mirrors the scalar CUDA parameters. It
  should be generalized to `[usize; 3]` at the same time as the core API.
- Validate on macOS before PR. The local Linux workstation cannot execute the
  Metal tests.

Validation:

- `cargo fmt`
- `cargo test -p candle-core conv3d` passed locally after the shared test
  refactor.
- `cargo check -p candle-core --features metal` was attempted locally, but this
  Linux host stops in `objc2` with the expected Apple-platform compile error
  before Candle's Metal code can be checked.
- `git diff --check` passed before commit.
- Two audit passes completed without major or medium findings:
  - audit 1 checked the Metal im2col3d shader, Rust kernel launcher, backend
    layout transforms, per-axis parameter plumbing, and shared Metal test
    wrapper against the vendored implementation and existing 2D conventions;
  - audit 2 reran the shared Conv3d CPU regression, checked `git diff --check`,
    and verified the Metal symbols line up across shader, launcher, backend,
    and tests.
- A third source-level audit on 2026-07-29 checked the branch against
  `conv3d-core-cpu` and confirmed the PR diff is limited to
  `candle-core/src/metal_backend/mod.rs`, `candle-core/tests/conv_tests.rs`,
  `candle-metal-kernels/src/kernels/convolution.rs`, and
  `candle-metal-kernels/src/metal_src/conv.metal`; the shader ABI, Rust
  launcher argument order, backend layout transforms, per-axis
  stride/padding/dilation, and shared tests line up. `cargo test -p
  candle-core conv3d` and `git diff --check conv3d-core-cpu..conv3d-metal`
  passed locally.
- Upstream submission gate: still run `cargo test -p candle-core --features
  metal conv3d` on a macOS/Metal host before opening the PR. This is a runtime
  environment gate, not an unresolved candidate audit finding.

Reasoning:

Metal cannot be validated on the local Linux workstation, so it should remain a
separate PR with explicit macOS validation.

## PR 7: candle-nn Conv3d module

Branch: `nn-conv3d`

Status: implemented and committed in the fork as `8e0fa052`.

Scope:

- Add `Conv3d`, `Conv3dConfig`, `conv3d`, and `conv3d_no_bias`.
- Mirror existing `Conv1d` and `Conv2d` naming and initialization patterns.
- Add `candle-nn` tests using CPU and feature-gated CUDA/Metal cases where
  available.

Audit notes before implementation:

- `Conv3dConfig` should carry `[usize; 3]` for `padding`, `stride`, and
  `dilation`, matching a generalized core Conv3d API.
- Current Candle `main` has a Qwen3-VL-local `conv3d_temporal_2` helper. The
  generic `candle-nn` module should make that helper unnecessary in a later,
  separate cleanup PR.

Validation:

- `cargo fmt`
- `cargo test -p candle-nn conv3d` passed locally.
- `cargo test -p candle-nn` passed locally.
- `git diff --check` passed before commit.
- Two audit passes completed without major or medium findings:
  - audit 1 checked that `Conv3dConfig` uses `[usize; 3]`, `Conv3d::forward`
    calls core `conv3d_with_algo`, bias broadcasting uses
    `(1, channels, 1, 1, 1)`, constructors mirror Conv2d initialization, and
    public re-exports are present;
  - audit 2 ran the full `candle-nn` test suite and checked the final diff and
    staged file set.

Reasoning:

This depends on the core Conv3d API. Keeping the high-level module separate
makes the core backend review easier.

## PR 8: CUDA sm75/CUDA 12.8 kernel compatibility

Branch: `cuda-sm75-compat`

Status: implemented and committed in the fork as `005ea216`.

Source: `/home/mahogny/github/ripp/cellpose-rs/CANDLEFIX.md` and vendored
`vendor/candle-kernels-0.8.4`.

Scope:

- Make `candle-kernels` compile on Quadro RTX 5000 / sm75 with CUDA 12.8.
- Avoid compiling BF16/device-code paths that require unavailable architecture
  support.
- Keep BF16 support enabled on architectures that actually support it.

Audit notes before implementation:

- Current Candle `main` already has some compute-capability gating in
  `candle-kernels/build.rs`; first verify whether the cellpose
  `compatibility.cuh` patch is still needed on current `main`.
- Do not hardcode this workstation. Prefer architecture guards or build-script
  feature definitions that preserve newer GPU support.
- The local CUDA check currently fails because `cuda_fp8.h` is missing from
  `/usr/include`; distinguish that header-path/toolkit issue from sm75 BF16
  gating.

Validation:

- `CUDA_HOME=/usr/local/cuda-12.8 CUDA_PATH=/usr/local/cuda-12.8
  NVCC=/usr/local/cuda-12.8/bin/nvcc CUDA_COMPUTE_CAP=75
  cargo check -p candle-core --features cuda` passed locally.
- `git diff --check` passed before commit.
- Two audit passes completed without major or medium findings:
  - audit 1 checked that the diff is a one-file guard around the fallback
    `__hmax_nan`/`__hmin_nan` definitions and verified the original CUDA 12.8
    duplicate-definition failure was addressed;
  - audit 2 wrapped the preprocessor condition, reran the CUDA 12.8/sm75 check,
    and confirmed the patch stays scoped to compatibility.
- A BF16-capable sm80+ validation was not available locally.

Reasoning:

This is a compatibility PR, not a Conv3d PR. It should stay separate so
upstream can review architecture gating independently from new operators.

## PR 9: CUDA GEMM broadcast batch collapse

Branch: `cuda-f32-gemm-broadcast-collapse`

Status: implemented and committed in the fork as `8b5605fc`, stacked on
`cuda-sm75-compat` for local CUDA 12.8/sm75 validation.

Source: `/home/mahogny/github/ripp/cellpose-rs/CANDLEFIX.md` and vendored
`vendor/candle-core-0.8.4`.

Scope:

- Fix the slow floating-point GEMM case where a broadcasted RHS batch is treated
  as an inefficient strided batched GEMM.
- Collapse `(b, m, n, k)` to `(1, b * m, n, k)` only when layout constraints
  prove the LHS rows are contiguous and RHS is batch-broadcast.
- Ensure the ordinary FP32 path uses normal cuBLAS SGEMM when reduced precision
  is disabled.

Audit notes before implementation:

- Keep the optimization guarded by exact stride/layout checks; do not apply it
  to arbitrary non-contiguous tensors.
- Omit `CANDLE_LOG_F32_GEMM` from the upstream performance PR unless it is
  reworked into accepted tracing/instrumentation.
- Add a regression similar to
  `f32_matmul_broadcast_rhs_batch_stride_zero`.

Validation:

- `cargo fmt`
- `CUDA_HOME=/usr/local/cuda-12.8 CUDA_PATH=/usr/local/cuda-12.8
  NVCC=/usr/local/cuda-12.8/bin/nvcc CUDA_COMPUTE_CAP=75
  cargo test -p candle-core --features cuda
  f32_matmul_broadcast_rhs_batch_stride_zero` passed locally.
- `CUDA_HOME=/usr/local/cuda-12.8 CUDA_PATH=/usr/local/cuda-12.8
  NVCC=/usr/local/cuda-12.8/bin/nvcc CUDA_COMPUTE_CAP=75
  cargo check -p candle-core --features cuda` passed locally.
- `git diff --check` passed before commit.
- Two audit passes completed without major or medium findings:
  - audit 1 compared current Candle `main` with the cellpose vendored patch,
    found the reduced-precision control already present, and kept only the
    missing F32 broadcast collapse plus regression test;
  - audit 2 tightened the fast-path guard to require the collapsed RHS stride,
    reran the CUDA regression twice, ran the broader CUDA compile check, and
    confirmed the committed diff is one-file scoped;
  - audit 3 generalized the collapse decision into a shared helper used by F32
    and F64, added F64 CUDA regression coverage, kept F16/BF16 unchanged because
    their tensor-op/reduced-precision paths were not part of the tested need,
    and reran the focused CUDA regressions.
- Microbenchmark on 2026-07-29 with release CUDA build on Quadro RTX 5000
  / sm75, shape `[64, 256, 256] @ [256, 256]`: `cuda-sm75-compat` baseline
  3.648 ms/iter, PR branch 2.946 ms/iter, 1.24x faster. Checksums matched.

Reasoning:

This is a CUDA matmul performance/correctness patch that benefits Cellpose and
may also help StarDist, but it is independent of Conv3d.

## PR 10: CUDA F32 softmax last-dim 1024 fast path

Branch: `cuda-softmax-1024-f32`

Status: implemented and committed in the fork as `ccae8ef6`, stacked on
`cuda-sm75-compat` for local CUDA 12.8/sm75 validation.

Source: `/home/mahogny/github/ripp/cellpose-rs/CANDLEFIX.md` and vendored
`vendor/candle-kernels-0.8.4` plus `vendor/candle-nn-0.8.4`.

Scope:

- Add a specialized CUDA kernel for contiguous F32 softmax where the last
  dimension is exactly 1024.
- Dispatch from `candle-nn::ops::softmax_last_dim` only for supported
  dtype/layout/shape.
- Preserve the existing generic path for all other cases.

Audit notes before implementation:

- Check current Candle `main` for any existing softmax specialization before
  transplanting.
- Keep the 1024-specific path as a narrow fast path unless a clean generalized
  block-softmax falls out naturally.
- Include correctness comparison against the generic stable softmax expression.

Validation:

- `cargo fmt`
- `cargo test -p candle-nn softmax_last_dim_1024_f32_cpu` passed locally.
- `CUDA_HOME=/usr/local/cuda-12.8 CUDA_PATH=/usr/local/cuda-12.8
  NVCC=/usr/local/cuda-12.8/bin/nvcc CUDA_COMPUTE_CAP=75
  cargo test -p candle-nn --features cuda softmax_last_dim_1024_f32`
  passed locally.
- `CUDA_HOME=/usr/local/cuda-12.8 CUDA_PATH=/usr/local/cuda-12.8
  NVCC=/usr/local/cuda-12.8/bin/nvcc CUDA_COMPUTE_CAP=75
  cargo check -p candle-nn --features cuda` passed locally.
- `git diff --check` passed before commit.
- Two audit passes completed without major or medium findings:
  - audit 1 checked current Candle `main` has no existing 1024-wide softmax
    specialization and adapted the vendored patch to the current
    `PushKernelArg` builder launcher;
  - audit 2 checked the final two-file diff, confirmed the existing contiguous
    input validation is still the layout gate, and verified all non-F32 or
    non-1024 shapes continue through the existing generic kernel path.
- Microbenchmark on 2026-07-29 with release CUDA build on Quadro RTX 5000
  / sm75, shape `[8192, 1024]`: `cuda-sm75-compat` baseline 6.194 ms/iter,
  PR branch 4.789 ms/iter, 1.29x faster. Checksums matched within F32
  reduction precision.

Reasoning:

This is a standalone kernel+dispatch performance PR and should not be mixed
with linear or GEMM changes.

## PR 11: CUDA F32 last-dim bias-add fast path

Branch: `cuda-linear-bias-add-f32`

Status: implemented and committed in the fork as `635f8f43`, stacked on
`cuda-sm75-compat` for local CUDA 12.8/sm75 validation.

Source: `/home/mahogny/github/ripp/cellpose-rs/CANDLEFIX.md` and vendored
`vendor/candle-kernels-0.8.4` plus `vendor/candle-nn-0.8.4`.

Scope:

- Add a reusable private CUDA fast path for contiguous F32 tensors plus 1D bias
  addition along the last dimension.
- Use the helper from `Linear::forward` after matmul.
- Fall back to `broadcast_add` for unsupported dtype/layout/device cases.

Audit notes before implementation:

- Keep fallback behavior explicit and tested.
- Keep the helper private for now to avoid committing to a public API in a
  performance PR; use generalized naming so it can move to a broader ops module
  later if upstream wants that.
- Add 2D and higher-rank tests comparing `Linear::forward` to
  `matmul(...).broadcast_add(...)`.

Validation:

- `cargo fmt`
- `cargo test -p candle-nn bias_add` passed locally.
- `CUDA_HOME=/usr/local/cuda-12.8 CUDA_PATH=/usr/local/cuda-12.8
  NVCC=/usr/local/cuda-12.8/bin/nvcc CUDA_COMPUTE_CAP=75
  cargo test -p candle-nn --features cuda bias_add` passed locally.
- `CUDA_HOME=/usr/local/cuda-12.8 CUDA_PATH=/usr/local/cuda-12.8
  NVCC=/usr/local/cuda-12.8/bin/nvcc CUDA_COMPUTE_CAP=75
  cargo check -p candle-nn --features cuda` passed locally.
- `git diff --check` passed before commit.
- Two audit passes completed without major or medium findings:
  - audit 1 checked current `Linear::forward` already has contiguous 3D/4D
    matmul reshaping and kept that optimization intact while routing only the
    final bias add through the fast path;
  - audit 2 replaced the vendored catch-all fallback with explicit tensor
    preflight, so unsupported dtype/layout/device cases use `broadcast_add`
    but real CUDA kernel or launch failures are returned;
  - audit 3 generalized the kernel/custom-op naming to `bias_add_last_dim`,
    kept the helper reusable but private, added a direct 3D helper test in
    addition to the `Linear::forward` tests, and reran CPU/CUDA focused tests
    plus CUDA compile check.
- Microbenchmark on 2026-07-29 with release CUDA build on Quadro RTX 5000
  / sm75, `Linear::forward` shape `[16384, 64] @ [1024, 64]^T + [1024]`:
  `cuda-sm75-compat` baseline 10.689 ms/iter, PR branch 10.138 ms/iter,
  1.05x faster. Checksums matched.

Reasoning:

This touches a different kernel family and high-level module from softmax, so
it should be reviewed separately.

## External/optional: cudarc CUDA 12.8 library lookup

Source: `/home/mahogny/github/ripp/cellpose-rs/CANDLEFIX.md` and vendored
`vendor/cudarc-0.13.9`.

Scope:

- Prefer `/usr/local/cuda-12.8` dynamic library locations when loading CUDA
  libraries if upstream cudarc still lacks this.

Decision:

- Track this as a cudarc upstream candidate, not a Candle PR, unless Candle's
  current dependency/features still require a local workaround.

Validation:

- Build Candle CUDA features with the target machine's CUDA 12.8 installation.

## Local-only unless requested: Candle Transformers timing helpers

Source: `/home/mahogny/github/ripp/cellpose-rs/CANDLEFIX.md` and vendored
`vendor/candle-transformers-0.8.4`.

Decision:

- Do not include in the optimization PR stack by default. The helpers are useful
  diagnostics but not an upstreamable runtime fix as written.

## Branch-base audit

The CUDA branches were rebuilt once after audit because `cuda-sm75-compat` had
accidentally been stacked on `conv3d-core-cpu`, which made PR 8 and downstream
CUDA performance branches include unrelated Conv3d files in their diffs.

Current intended bases:

- PR 8 `cuda-sm75-compat`: base `main`; one-file diff
  `candle-kernels/src/compatibility.cuh`.
- PR 9 `cuda-f32-gemm-broadcast-collapse`: base `cuda-sm75-compat`; one-file
  diff `candle-core/src/cuda_backend/mod.rs`.
- PR 10 `cuda-softmax-1024-f32`: base `cuda-sm75-compat`; two-file diff
  `candle-kernels/src/reduce.cu` and `candle-nn/src/ops.rs`.
- PR 11 `cuda-linear-bias-add-f32`: base `cuda-sm75-compat`; two-file diff
  `candle-kernels/src/binary.cu` and `candle-nn/src/linear.rs`.
- PR 5 `conv3d-cuda`: base `conv3d-core-cpu`; four-file diff
  `candle-core/src/cuda_backend/cudnn.rs`,
  `candle-core/src/cuda_backend/mod.rs`, `candle-core/tests/conv_tests.rs`,
  and `candle-kernels/src/conv.cu`.

Validation after rebuild:

- `git diff --check` passed for rebuilt PR 5, PR 8, PR 9, PR 10, and PR 11.
- PR 8 CUDA 12.8/sm75 `cargo check -p candle-core --features cuda` passed.
- PR 9 CUDA 12.8/sm75 regression and compile checks passed.
- PR 10 CPU regression, CUDA regression, and CUDA compile checks passed.
- PR 11 CPU regression, CUDA regression, and CUDA compile checks passed.
- PR 5 CUDA Conv3d regression and `cuda,cudnn` compile check passed on
  validation branch `validation/conv3d-cuda-sm75`, which combines the clean
  PR 5 branch with the separate PR 8 sm75 compatibility patch.

## Not currently worth a PR

The CUDA MoE/BF16 build issue appears fixed on current Candle `main`:
`candle-kernels/build.rs` detects compute capability and defines
`NO_BF16_KERNEL` below sm80. Do not open a PR unless a current `main` build still
fails on sm75.

## Workflow

For each PR:

1. Start from clean Candle `main`.
2. Create the topic branch named above.
3. Transplant only the relevant change.
4. Add focused tests and, for performance PRs, a short benchmark note.
5. Run formatting and targeted tests.
6. Commit in Henriksson's name.
7. Push to the fork and open a PR against upstream Candle.
