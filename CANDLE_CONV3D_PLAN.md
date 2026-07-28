# Candle Conv3d Vendor Plan

Goal: add real Conv3d support to the vendored Candle crates used by
`stardist-rs`, then replace the local depth-loop Conv2d workaround in the
StarDist 3D Candle backend.

## Status

- Implemented native `Tensor::conv3d` in vendored `candle-core`.
- Implemented CPU Conv3d forward for NCDHW/OIDHW tensors.
- Added `candle-nn::Conv3d` and helper constructors.
- StarDist 3D Candle inference now uses `candle_nn::Conv3d` for uniform padding
  and stride, with the previous depth-loop path retained for anisotropic
  configs.
- Added cuDNN Conv3d descriptor/launch wiring; it compiles with
  `candle-cudnn`.
- Added non-cuDNN CUDA Conv3d through `im2col3d` plus CUDA matmul, so local
  CUDA 12.8 runs do not depend on the incompatible cuDNN installation.
- Fixed non-cuDNN CUDA convolution with non-contiguous kernels, which affected
  the anisotropic 3D fallback path.
- Added CPU tests covering the planned padding, stride, kernel, channel, output
  channel, and batch-size cases.
- Runtime CUDA benchmark passes locally with `candle-cuda,hdf5` on CUDA 12.8:
  labels exact, raw probability max diff `7.15e-7`, raw distance max diff
  `1.34e-5`, and raw 3D inference `0.098 s`.
- cuDNN runtime remains unavailable locally because CUDA 12.8 plus the Ollama
  CUDA 13 cuDNN bundle fails with `CUDNN_STATUS_NOT_INITIALIZED`; CUDA 13.2
  fails earlier at `CUBLAS_STATUS_NOT_INITIALIZED` on the installed
  driver/runtime stack.

## Scope

Implement enough Conv3d for StarDist 3D inference:

- F32 tensors.
- NCDHW input layout.
- OIDHW kernel layout.
- `groups = 1`.
- `dilation = 1`.
- symmetric padding.
- stride 1 and 2.
- CPU correctness first.
- CUDA/cuDNN acceleration second.

Do not optimize 3D StarDist postprocessing in this plan. That is a separate
geometry/polyhedron rendering bottleneck.

## 1. Vendor Candle Core And NN

Add local copies:

- `vendor/candle-core-0.9.2`
- `vendor/candle-nn-0.9.2`

Extend `[patch.crates-io]` in `Cargo.toml`:

```toml
candle-core = { path = "vendor/candle-core-0.9.2" }
candle-nn = { path = "vendor/candle-nn-0.9.2" }
candle-kernels = { path = "vendor/candle-kernels-0.9.2" }
```

Keep ignoring generated vendor metadata:

- `vendor/**/.cargo-ok`
- `vendor/**/.cargo_vcs_info.json`
- `vendor/**/Cargo.lock`

## 2. Add Tensor Conv3d In Candle Core

Mirror the existing Conv2d structure:

- Add `Op::Conv3D` in `candle-core/src/op.rs`.
- Add tensor methods in `candle-core/src/tensor.rs`:
  - `conv3d`
  - optionally `conv3d_with_algo` if cuDNN algorithm selection is needed.
- Add storage/backend dispatch in `candle-core/src/storage.rs`.

Initial API should match Candle style:

```rust
pub fn conv3d(
    &self,
    kernel: &Tensor,
    padding: usize,
    stride: usize,
    dilation: usize,
    groups: usize,
) -> Result<Tensor>
```

## 3. Implement CPU Conv3d

Add `candle-core/src/cpu_backend/conv3d.rs`.

Start with correctness rather than maximum speed:

- Direct loop or simple im2col plus GEMM.
- F32 support first.
- Batch, input channel, output channel loops.
- Kernel depth/height/width loops.
- Padding and stride support.

Required CPU test cases:

- padding 0, stride 1, kernel 1.
- padding 1, stride 1, kernel 3.
- padding 0, stride 2, kernel 3.
- multi-input-channel and multi-output-channel.
- batch size greater than 1.

## 4. Add CUDA/cuDNN Conv3d

For useful GPU speed, route Conv3d through cuDNN when the `cudnn` feature is
enabled.

Tasks:

- Extend Candle's cuDNN convolution wrapper to accept 5D NCDHW tensor
  descriptors.
- Use OIDHW filter descriptors.
- Support F32 first.
- Support padding/stride/dilation/groups in the same shape as Conv2d.
- Return a clear unsupported error when CUDA is enabled without a usable Conv3d
  path.

Avoid using the current StarDist fallback as the CUDA implementation: repeated
small Conv2d launches are the problem we are trying to remove.

## 5. Add Candle NN Conv3d

In `candle-nn/src/conv.rs`:

- Add `Conv3dConfig`, mirroring `Conv2dConfig`.
- Add `Conv3d { weight, bias, config }`.
- Implement `Module for Conv3d`.
- Add helpers:
  - `conv3d`
  - `conv3d_no_bias`

Export these from `candle-nn/src/lib.rs`.

## 6. Switch StarDist To Native Conv3d

In `src/model.rs`:

- Replace the local depth-loop `CandleConv3d` with `candle_nn::Conv3d`.
- Preserve the same Keras HDF5 weight layout checks.
- Keep the current depth-loop implementation temporarily as a private fallback
  or test-only reference until native Conv3d is proven.

## 7. Verify

Local correctness:

```bash
cargo test --features candle,hdf5 candle_3d --lib
cargo test --features candle,hdf5 candle_ --lib
cargo check --features candle,hdf5 --examples
```

CUDA benchmark:

```bash
CUDA_HOME=/usr/local/cuda-12.8 \
CUDA_PATH=/usr/local/cuda-12.8 \
PATH=/usr/local/cuda-12.8/bin:$PATH \
LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:$LD_LIBRARY_PATH \
CUDA_COMPUTE_CAP=75 \
cargo run --release --features candle-cuda,hdf5 \
  --example bench_candle_real_data -- \
  3d .tmp/bench_original_real_3d.npz cuda
```

Expected result:

- labels exact.
- raw probability and distance diffs comparable to CPU fixture tolerance.
- 3D raw inference no longer dominated by repeated small Conv2d launches.

Current local result:

- labels exact.
- raw probability max diff: `7.15e-7`.
- raw distance max diff: `1.34e-5`.
- raw 3D inference: `0.097 s`.
- sparse prediction: `0.173 s`.
- postprocess: `171.905 s`.
- peak RSS: `373.7 MiB`.

`cargo check --features candle-cudnn,hdf5 --examples` passes, so the cuDNN
Conv3d wiring compiles. A runtime cuDNN benchmark has not been possible on this
machine because the installed CUDA/cuDNN libraries are mismatched.

## 8. Benchmark Reporting

Report 3D as separate stages:

- raw model inference.
- sparse prediction.
- postprocess.
- peak RSS.
- parity.

Do not interpret postprocess speed as Conv3d speed. StarDist 3D polyhedron
rendering is a separate bottleneck.
