#[cfg(all(feature = "candle-metal", feature = "hdf5"))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use candle_core::{Device, Tensor};
    use stardist_rs::{StarDist3D, weights::load_keras_hdf5_weights};

    let cpu = Device::Cpu;
    let metal = Device::new_metal(0)?;
    let stardist = StarDist3D::from_model_dir("assets/models/examples/3D_demo")?;
    let config = stardist.config.clone();
    let weights = load_keras_hdf5_weights("stardist/models/examples/3D_demo/weights_best.h5")?;
    let model_cpu = stardist_rs::model::candle::StarDist3D::init(config.clone(), &cpu)
        .load_keras_weights(&weights, &cpu)?;
    let model_metal = stardist_rs::model::candle::StarDist3D::init(config, &metal)
        .load_keras_weights(&weights, &metal)?;

    let input_len = 8 * 16 * 16;
    let input = (0..input_len)
        .map(|v| ((v % 251) as f32 - 103.0) / 97.0)
        .collect::<Vec<_>>();
    let mut cpu_layer = Tensor::from_vec(input.clone(), (1, 1, 8, 16, 16), &cpu)?;
    let mut metal_layer = Tensor::from_vec(input, (1, 1, 8, 16, 16), &metal)?;

    cpu_layer = model_cpu
        .initial_7
        .as_ref()
        .expect("missing initial_7")
        .forward(&cpu_layer)?;
    metal_layer = model_metal
        .initial_7
        .as_ref()
        .expect("missing initial_7")
        .forward(&metal_layer)?;
    compare("initial_7", &cpu_layer, &metal_layer, &cpu)?;

    cpu_layer = model_cpu
        .initial_3
        .as_ref()
        .expect("missing initial_3")
        .forward(&cpu_layer)?;
    metal_layer = model_metal
        .initial_3
        .as_ref()
        .expect("missing initial_3")
        .forward(&metal_layer)?;
    compare("initial_3", &cpu_layer, &metal_layer, &cpu)?;

    for (block_index, (cpu_block, metal_block)) in model_cpu
        .resnet_blocks
        .iter()
        .zip(&model_metal.resnet_blocks)
        .enumerate()
    {
        let shortcut_cpu = match &cpu_block.shortcut {
            Some(shortcut) => shortcut.forward(&cpu_layer)?,
            None => cpu_layer.clone(),
        };
        let shortcut_metal = match &metal_block.shortcut {
            Some(shortcut) => shortcut.forward(&metal_layer)?,
            None => metal_layer.clone(),
        };
        compare(
            &format!("block_{block_index}_shortcut"),
            &shortcut_cpu,
            &shortcut_metal,
            &cpu,
        )?;

        let mut residual_cpu = if cpu_block.pool.iter().any(|p| *p > 1) {
            same_for_stride(
                &cpu_layer,
                cpu_block.pool,
                model_cpu.config.resnet_kernel_size,
            )?
        } else {
            cpu_layer
        };
        let mut residual_metal = if metal_block.pool.iter().any(|p| *p > 1) {
            same_for_stride(
                &metal_layer,
                metal_block.pool,
                model_metal.config.resnet_kernel_size,
            )?
        } else {
            metal_layer
        };
        compare(
            &format!("block_{block_index}_residual_input"),
            &residual_cpu,
            &residual_metal,
            &cpu,
        )?;

        for (conv_index, (cpu_conv, metal_conv)) in
            cpu_block.convs.iter().zip(&metal_block.convs).enumerate()
        {
            residual_cpu = cpu_conv.forward(&residual_cpu)?;
            residual_metal = metal_conv.forward(&residual_metal)?;
            compare(
                &format!("block_{block_index}_conv_{conv_index}"),
                &residual_cpu,
                &residual_metal,
                &cpu,
            )?;
            if conv_index + 1 != cpu_block.convs.len() {
                residual_cpu = residual_cpu.relu()?;
                residual_metal = residual_metal.relu()?;
                compare(
                    &format!("block_{block_index}_relu_{conv_index}"),
                    &residual_cpu,
                    &residual_metal,
                    &cpu,
                )?;
            }
        }
        cpu_layer = residual_cpu.broadcast_add(&shortcut_cpu)?.relu()?;
        metal_layer = residual_metal.broadcast_add(&shortcut_metal)?.relu()?;
        compare(
            &format!("block_{block_index}_out"),
            &cpu_layer,
            &metal_layer,
            &cpu,
        )?;
    }

    if let (Some(cpu_features), Some(metal_features)) = (&model_cpu.features, &model_metal.features)
    {
        cpu_layer = cpu_features.forward(&cpu_layer)?.relu()?;
        metal_layer = metal_features.forward(&metal_layer)?.relu()?;
        compare("features", &cpu_layer, &metal_layer, &cpu)?;
    }
    compare(
        "prob_head",
        &model_cpu.prob.forward(&cpu_layer)?,
        &model_metal.prob.forward(&metal_layer)?,
        &cpu,
    )?;
    compare(
        "dist_head",
        &model_cpu.dist.forward(&cpu_layer)?,
        &model_metal.dist.forward(&metal_layer)?,
        &cpu,
    )?;
    Ok(())
}

#[cfg(all(feature = "candle-metal", feature = "hdf5"))]
fn same_for_stride(
    layer: &candle_core::Tensor,
    stride: [usize; 3],
    kernel: [usize; 3],
) -> candle_core::Result<candle_core::Tensor> {
    let (_batch, _channels, depth, height, width) = layer.dims5()?;
    let out_depth = depth.div_ceil(stride[0]);
    let out_height = height.div_ceil(stride[1]);
    let out_width = width.div_ceil(stride[2]);
    let pad_depth = ((out_depth - 1) * stride[0] + kernel[0]).saturating_sub(depth);
    let pad_height = ((out_height - 1) * stride[1] + kernel[1]).saturating_sub(height);
    let pad_width = ((out_width - 1) * stride[2] + kernel[2]).saturating_sub(width);
    layer
        .pad_with_zeros(2, pad_depth / 2, pad_depth - pad_depth / 2)?
        .pad_with_zeros(3, pad_height / 2, pad_height - pad_height / 2)?
        .pad_with_zeros(4, pad_width / 2, pad_width - pad_width / 2)
}

#[cfg(all(feature = "candle-metal", feature = "hdf5"))]
fn compare(
    name: &str,
    cpu_tensor: &candle_core::Tensor,
    metal_tensor: &candle_core::Tensor,
    cpu: &candle_core::Device,
) -> candle_core::Result<()> {
    let cpu_values = cpu_tensor.flatten_all()?.to_vec1::<f32>()?;
    let metal_values = metal_tensor
        .to_device(cpu)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    let mut max_abs = 0.0f32;
    let mut mean_abs = 0.0f64;
    let mut first = None;
    for (index, (lhs, rhs)) in cpu_values.iter().zip(&metal_values).enumerate() {
        let diff = (*lhs - *rhs).abs();
        max_abs = max_abs.max(diff);
        mean_abs += diff as f64;
        if first.is_none() && diff > 1e-4 {
            first = Some((index, *lhs, *rhs, diff));
        }
    }
    if !cpu_values.is_empty() {
        mean_abs /= cpu_values.len() as f64;
    }
    println!(
        "{name}: shape={:?} len={} max_abs={max_abs:.9} mean_abs={mean_abs:.9} first_mismatch={first:?}",
        cpu_tensor.shape(),
        cpu_values.len()
    );
    Ok(())
}

#[cfg(not(all(feature = "candle-metal", feature = "hdf5")))]
fn main() {
    eprintln!("requires --features candle-metal,hdf5");
}
