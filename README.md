# StarDist-rs

This is a Rust translation of [StarDist](https://github.com/stardist/stardist) - Object Detection with Star-convex Shapes

* 2026-08-01: CI added
* 2026-07-30: Replaced clipper with faster pure rust version
* 2026-07-28: Speed now higher than original
* 2026-07-27: Work on getting proper speed. 2d is ok with GPU, and 3d CUDA now has exact-label parity with postprocessing close to the original implementation.
* 2026-07-26: Initial translation

Translated from upstream StarDist commit `e80c6de700693bc228ed3c9ba1dc19c3785667ee`.

Note that this crate has non-Rust dependencies. The aim is to minimize them but it will take some time to get them to zero.

## Below is a blurb that we should add to all our crates; it is the latest version
## ⚠ This is an LLM-mediated faithful (hopefully) translation, not the original code!

Most users should probably first see if the existing original code works for them, unless they have
reason otherwise. The original source may have newer features and it has had more love in terms of
fixing bugs. In fact, we aim to replicate (logical) bugs if they are present, for the sake of reproducibility!
(but then we might have added a few more bugs in the process)

There are however cases when you might prefer this Rust version. We generally agree with
[this manifesto](https://rewrites.bio/) but more specifically:

* We have had many issues with ensuring that our software works using existing containers (Docker,
  PodMan, Singularity). One size does not fit all and it eats our resources trying to keep up with
  every way of delivering software
* Common package managers do not work well. It was great when we had a few Linux distributions with
  stable procedures, but now there are just too many ecosystems (Homebrew, Conda). Conda has an
  NP-complete resolver which does not scale. Homebrew is only so-stable. And our dependencies in
  Python still break. These can no longer be considered professional serious options. Meanwhile, Cargo
  enables multiple versions of packages to be available, even within the same program(!)
* The future is the web. We deploy software in the web browser, and until now that has meant
  Javascript. This is a language where even the == operator is broken. Typescript is one step up, but
  a game changer is the ability to compile Rust code into webassembly, enabling performance and
  sharing of code with the backend. Translating code to Rust enables new ways of deployment and
  running code in the browser has especial benefits for science - researchers do not have deep pockets
  to run servers, so pushing compute to the user enables deployment that otherwise would be impossible
* Old CLI-based utilities are bad for the environment(!). A large amount of compute resources are
  spent creating and communicating via small files, which we can bypass by using code as libraries.
  Even better, we can avoid frequent reloading of databases by hoisting this stage, with up to 100x
  speedups in some cases. Less compute means faster compute and less electricity wasted
* LLM-mediated translations may actually be safer to use than the original code. This article shows
  that
  [running the same code on different operating systems can give somewhat different answers](https://doi.org/10.1038/nbt.3820).
  This is a gap that Rust+Cargo can reduce. Typesafe interfaces also reduce coding mistakes and error
  handling, as opposed to typical command-line scripting

But:

* **LLM-mediated translation should still be considered experimental**. The LLM technology is immature and has
  sharp corners. But there are opportunities to reap, and the genie is not going back into the bottle.
  This translation is as much aimed to learn how to improve the translation technology and get feedback
  on the results.
* Translations are not endorsed by the original authors unless otherwise noted. **Do not send bug
  reports to the original developers**. Use our Github issues page instead.
* **Do not trust the benchmarks on this page**. They are used to audit the translation and not aimed to be precise
* **If you want improved performance, you should use this crate as a library, not as a CLI utility**.
  By calling the code directly you avoid startup cost and the need to read/write the results. You also
  get type-safety, and it will be easier to have multiple versions installed in parallel.
* **Check the original Github pages for information about the package**. This README is kept sparse on
  purpose. It is not meant to be the primary source of information.
* **If you are the author of the original code and wish to move to Rust, you can obtain ownership of
  this repository and crate**. Until then, our commitment is to offer an as-faithful-as-possible
  translation of a snapshot of your code. If we find serious bugs, we will report them to you.
  Otherwise we will just replicate them, to ensure comparability across studies that claim to use
  package XYZ v.666. Think of this like a fancy Ubuntu .deb-package of your software - that is how we
  treat it

This blurb might be out of date. Go to
[this page](https://github.com/henriksson-lab/rustification) for the latest information and further
information about how we approach translation.


## Usage

CLI-style prediction adapters are kept behind the optional `cli` feature.

By default, `stardist-rs` enables Burn CPU inference and pure-Rust HDF5 weight loading. This is the recommended default when CPU inference is required. For GPU inference, enable the Candle GPU feature for your platform instead, such as `candle-cuda` or `candle-metal`.

```toml
[dependencies]
stardist-rs = "0.1"
burn = { version = "0.21", default-features = false, features = ["std", "train", "autodiff", "flex"] }
```

For Candle GPU inference instead:

```toml
[dependencies]
stardist-rs = { version = "0.1", default-features = false, features = ["candle-cuda", "hdf5"] }
candle-core = "0.11"
```

To prefer Candle GPU when it is compiled in and available at runtime, while
falling back to Burn CPU otherwise, compile both backends and use
`preferred_inference_backend()`:

```toml
[dependencies]
stardist-rs = { version = "0.1", features = ["candle-cuda"] }
```

```rust
match stardist_rs::preferred_inference_backend() {
    Some(stardist_rs::PreferredInferenceBackend::CandleCuda) => {
        // Use the Candle CUDA model path.
    }
    Some(stardist_rs::PreferredInferenceBackend::BurnCpu) => {
        // Use the Burn CPU model path.
    }
    _ => {
        // No compiled backend is available for this runtime.
    }
}
```

Runnable examples are available under `examples/`:

```bash
cargo run --example config_thresholds
cargo run --example sample_data
cargo run --example bioimageio_helpers
```

### Compare against original Python StarDist

For local speed, peak-RSS, and parity checks on the bundled 2D real TIFF image,
generate an original Python StarDist artifact first, then run the Rust examples
against the same normalized tensor:

```bash
python3 scripts/bench_original_real_data.py 2d --out .tmp/bench_original_real_2d.npz
cargo run --release --features burn --example bench_burn_real_data -- 2d .tmp/bench_original_real_2d.npz
cargo run --release --features candle,hdf5 --example bench_candle_real_data -- 2d .tmp/bench_original_real_2d.npz cpu
```

To run and collect the 2D benchmark set in one JSON summary:

```bash
python3 scripts/bench_real_data.py --dimensions 2d --out .tmp/bench_real_data_summary.json
python3 scripts/bench_real_data.py --dimensions 2d --candle-cuda
CUDA_COMPUTE_CAP=75 python3 scripts/bench_real_data.py --dimensions 2d --candle-cuda --cuda-home /usr/local/cuda-12.8 --cuda-compute-cap 75
```

The Python script uses the untracked upstream checkout under `stardist/` and
writes timing/RSS JSON next to the NPZ. The Rust example loads the same input
and Python outputs, then reports Burn/Candle timing, peak RSS, and max/mean
absolute differences for raw predictions plus instance outputs. The benchmark
runner keeps the original Python baseline on CPU by default to avoid accidental
TensorFlow GPU failures; pass `--python-gpu` to benchmark original StarDist on
TensorFlow GPU. The
`--candle-cuda` runner option preflights for `nvcc`, then builds with
`--features candle-cuda,hdf5` only when the CUDA toolkit is available. Pass
`--cuda-home` to select a specific toolkit instead of the first `nvcc` on
`PATH`; pass `--cuda-compute-cap` if the benchmark environment cannot query the
driver directly. This checkout patches Candle through the
`mahogny/candle` `stardist-integration` branch, which includes the Conv3d and
CUDA sm75/Turing fixes needed for the local Quadro RTX 5000 benchmark.

Representative local 2D results on `assets/data/images/img2d.tif`
after switching 2D polygon clipping to `geo-clipper-pure-rs`:

| Backend/device | Python raw inference | Rust raw inference | Raw speed | Python sparse predict | Rust sparse predict | Sparse speed | Python postprocess | Rust postprocess | Post speed | Python peak RSS | Rust peak RSS | RSS | Parity |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Burn CPU | 0.288 s | 0.460 s | 1.60x slower | 0.182 s | 0.487 s | 2.67x slower | 0.295 s | 0.048 s | 6.1x faster | 1202.2 MiB | 262.6 MiB | 4.6x lower | labels exact; raw max diff `2.289e-5` |
| Candle CPU | 0.288 s | 0.695 s | 2.42x slower | 0.182 s | 0.662 s | 3.64x slower | 0.295 s | 0.049 s | 6.1x faster | 1202.2 MiB | 281.3 MiB | 4.3x lower | labels exact; raw max diff `2.289e-5` |
| Candle CUDA, Quadro RTX 5000 sm75 | 0.288 s | 0.049 s | 5.9x faster | 0.182 s | 0.098 s | 1.9x faster | 0.295 s | 0.047 s | 6.3x faster | 1202.2 MiB | 361.1 MiB | 3.3x lower | labels exact; raw max diff `4.578e-5` |

For an end-to-end sparse prediction plus instance-label postprocess path:

| Backend/device | Python total | Rust total | Total speed |
| --- | ---: | ---: | ---: |
| Burn CPU | 0.477 s | 0.535 s | 1.12x slower |
| Candle CPU | 0.477 s | 0.710 s | 1.49x slower |
| Candle CUDA, Quadro RTX 5000 sm75 | 0.477 s | 0.145 s | 3.3x faster |

These timings include TensorFlow/Python and Burn/Flex CPU behavior on this
machine. Treat them as translation diagnostics, not portable performance
claims.

Representative local 3D results on `assets/data/images/img3d.tif`:

| Backend/device | Python raw inference | Rust raw inference | Raw speed | Python sparse predict | Rust sparse predict | Sparse speed | Python postprocess | Rust postprocess | Post speed | Python peak RSS | Rust peak RSS | RSS | Parity |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Candle CUDA, Quadro RTX 5000 sm75 | 0.363 s | 0.110 s | 3.3x faster | 0.434 s | 0.202 s | 2.1x faster | 0.913 s | 0.529 s | 1.7x faster | 1282.9 MiB | 375.1 MiB | 3.4x lower | labels exact; raw prob max diff `7.15e-7`; raw dist max diff `1.34e-5` |

For the same end-to-end sparse prediction plus instance-label postprocess path,
3D Candle CUDA is 1.84x faster than the original Python code (1.347 s Python
total, 0.731 s Rust total).

The 3D CUDA row uses Candle's non-cuDNN CUDA Conv3d path on CUDA 12.8. The raw
network path has parity. The Rust NMS path skips the local brute-force
halfspace-intersection filter and goes directly from the cheap bounds to exact
rendered overlap, because the original implementation uses Qhull for that
filter.

### Run the bundled 2D demo model with Burn

The repository checkout includes the upstream StarDist demo config and weights under `stardist/models/examples`. The crates.io package includes the small configs/thresholds but intentionally excludes the large demo `.h5` weights; provide your own StarDist/Keras weights or use a full repository checkout for this example. Burn expects channels-first tensors, so a 2D grayscale image is shaped as `NCHW`.

```rust
use burn::tensor::Tensor;
use stardist_rs::{Config2D, model::burn as stardist_burn};
use stardist_rs::weights::load_keras_hdf5_weights;

type B = burn::backend::Flex;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let device = Default::default();

    let config = Config2D::from_json_file("assets/models/examples/2D_demo/config.json")?;
    let weights = load_keras_hdf5_weights("path/to/weights_best.h5")?;

    let model = stardist_burn::StarDist2D::<B>::init(config, &device)
        .load_keras_weights(&weights, &device)?;

    // Replace this with normalized image pixels in NCHW order.
    let image = Tensor::<B, 4>::zeros([1, 1, 64, 64], &device);
    let outputs = model.forward(image);

    assert_eq!(outputs.prob.dims(), [1, 1, 32, 32]);
    assert_eq!(outputs.dist.dims(), [1, 32, 32, 32]);
    Ok(())
}
```

For 3D inference, use `Config3D`, `stardist_burn::StarDist3D`, the `3D_demo` files, and an `NCDHW` input tensor:

```rust
let config = stardist_rs::Config3D::from_json_file("assets/models/examples/3D_demo/config.json")?;
let weights = stardist_rs::weights::load_keras_hdf5_weights(
    "path/to/3d_weights_best.h5",
)?;
let model = stardist_rs::model::burn::StarDist3D::<B>::init(config, &device)
    .load_keras_weights(&weights, &device)?;
let image = burn::tensor::Tensor::<B, 5>::zeros([1, 1, 8, 16, 16], &device);
let outputs = model.forward(image);
```

### Run the bundled 2D demo model with Candle

Candle covers the 2D U-Net inference path. Use the `candle-cuda` or
`candle-metal` crate feature for local GPU builds; the Rust API stays the same
except for the selected `candle_core::Device`.

```rust
use candle_core::{Device, Tensor};
use stardist_rs::{Config2D, model::candle as stardist_candle};
use stardist_rs::weights::load_keras_hdf5_weights;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let device = Device::Cpu;
    let config = Config2D::from_json_file("assets/models/examples/2D_demo/config.json")?;
    let weights = load_keras_hdf5_weights("path/to/weights_best.h5")?;

    let model = stardist_candle::StarDist2D::try_init(config, &device)?
        .load_keras_weights(&weights, &device)?;

    // Replace this with normalized image pixels in NCHW order.
    let image = Tensor::zeros((1, 1, 64, 64), candle_core::DType::F32, &device)?;
    let outputs = model.forward(&image)?;

    assert_eq!(outputs.prob.dims(), &[1, 1, 32, 32]);
    assert_eq!(outputs.dist.dims(), &[1, 32, 32, 32]);
    Ok(())
}
```

### Postprocess predictions without Burn

The geometry and non-maximum suppression utilities are plain Rust APIs. Use them when probability and radial-distance maps come from another inference runtime.

```rust
use stardist_rs::{non_maximum_suppression, polygons_to_label};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let shape = [3, 3];
    let n_rays = 4;
    let grid = [1, 1];

    let mut prob = vec![0.0; shape[0] * shape[1]];
    prob[4] = 0.95;

    // Distances are laid out as YX-ray for dense 2D predictions.
    let mut dist = vec![0.0; prob.len() * n_rays];
    dist[4 * n_rays..5 * n_rays].copy_from_slice(&[1.0, 1.0, 1.0, 1.0]);

    let nms = non_maximum_suppression(
        &dist,
        &prob,
        shape,
        n_rays,
        grid,
        None,
        0.4,
        0.5,
        true,
        true,
    )?;
    let labels = polygons_to_label(
        &nms.dist,
        &nms.points,
        shape,
        Some(&nms.prob),
        f32::NEG_INFINITY,
        [1.0, 1.0],
    )?;

    assert_eq!(labels.dim(), (3, 3));
    Ok(())
}
```

### BioImageIO helpers

The `bioimageio` module mirrors StarDist helper pieces such as metadata construction, the DeepImageJ postprocessing macro text, dependency environment generation, typed export metadata, and import of StarDist payload files from an extracted BioImageIO package. TensorFlow SavedModel execution/export and `bioimageio.core.build_model` remain explicit runtime boundaries because this crate does not embed TensorFlow or Python BioImageIO.

```rust
use stardist_rs::{
    export_bioimageio, BioimageioMode, BioimageioModelConfig, BioimageioThresholds,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let outdir = std::env::temp_dir().join("stardist-rs-bioimageio-readme");
    let model = BioimageioModelConfig {
        n_dim: 2,
        n_channel_in: 1,
        n_rays: 32,
        grid: vec![2, 2],
        axes_net: "YXC".to_string(),
        axes_out: "YXC".to_string(),
        axes_net_div_by: vec![16, 16, 1],
        is_multiclass: false,
        thresholds: BioimageioThresholds { prob: 0.5, nms: 0.4 },
        config_json: serde_json::json!({"n_dim": 2, "n_rays": 32}),
    };
    let export = export_bioimageio(
        &model,
        &outdir,
        &[0.0; 16],
        &[4, 4],
        Some("demo"),
        BioimageioMode::TensorflowSavedModelBundle,
        1.0,
        99.8,
        None,
        false,
    )?;

    println!("{}", export.zip_path.display());
    Ok(())
}
```





## License

BSD 3-Clause License

## How to cite

- Uwe Schmidt, Martin Weigert, Coleman Broaddus, and Gene Myers.
[*Cell Detection with Star-convex Polygons*](https://arxiv.org/abs/1806.03535).
International Conference on Medical Image Computing and Computer-Assisted Intervention (MICCAI), Granada, Spain, September 2018.

- Martin Weigert, Uwe Schmidt, Robert Haase, Ko Sugawara, and Gene Myers.
[*Star-convex Polyhedra for 3D Object Detection and Segmentation in Microscopy*](http://openaccess.thecvf.com/content_WACV_2020/papers/Weigert_Star-convex_Polyhedra_for_3D_Object_Detection_and_Segmentation_in_Microscopy_WACV_2020_paper.pdf).
The IEEE Winter Conference on Applications of Computer Vision (WACV), Snowmass Village, Colorado, March 2020.

- Martin Weigert and Uwe Schmidt.
[*Nuclei Instance Segmentation and Classification in Histopathology Images with Stardist*](https://arxiv.org/abs/2203.02284).
The IEEE International Symposium on Biomedical Imaging Challenges (ISBIC), Kolkata, India, March 2022.
