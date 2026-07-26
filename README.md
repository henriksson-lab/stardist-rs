# StarDist-rs

This is a Rust translation of [StarDist](https://github.com/stardist/stardist) - Object Detection with Star-convex Shapes

* 2026-07-26: Initial work

Translated from upstream StarDist commit `e80c6de700693bc228ed3c9ba1dc19c3785667ee`.


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

This crate is intended primarily as a library. CLI-style prediction adapters are kept behind the optional `cli` feature.

Enable the `burn` feature for the native Burn model implementation. The default feature set includes `hdf5`, which is needed for loading Keras `.h5` weights.

```toml
[dependencies]
stardist-rs = { version = "0.1", features = ["burn", "hdf5"] }
burn = { version = "0.21", default-features = false, features = ["std", "train", "autodiff", "flex"] }
```

Runnable examples are available under `examples/`:

```bash
cargo run --example config_thresholds
cargo run --example sample_data
cargo run --example bioimageio_helpers
```

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
