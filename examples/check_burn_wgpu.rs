#[cfg(feature = "burn-wgpu")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use burn::tensor::Tensor;
    use stardist_rs::{Config2D, model::burn::StarDist2D};

    type B = burn::backend::Wgpu;

    let device = burn::backend::wgpu::WgpuDevice::default();
    let config = Config2D::from_json_file("assets/models/examples/2D_demo/config.json")?;
    let model = StarDist2D::<B>::try_init(config, &device)?;
    #[cfg(feature = "hdf5")]
    let model = {
        let weights_path = std::path::Path::new("stardist/models/examples/2D_demo/weights_best.h5");
        if weights_path.exists() {
            model.load_keras_weights(
                &stardist_rs::weights::load_keras_hdf5_weights(weights_path)?,
                &device,
            )?
        } else {
            model
        }
    };
    let input = Tensor::<B, 4>::zeros([1, 1, 64, 64], &device);
    let outputs = model.forward(input);

    let prob_shape = outputs.prob.shape();
    let dist_shape = outputs.dist.shape();
    let _ = outputs.prob.into_data();
    let _ = outputs.dist.into_data();
    println!(
        "prob={:?} dist={:?}",
        prob_shape.dims::<4>(),
        dist_shape.dims::<4>()
    );
    Ok(())
}

#[cfg(not(feature = "burn-wgpu"))]
fn main() {
    eprintln!("requires --features burn-wgpu");
}
