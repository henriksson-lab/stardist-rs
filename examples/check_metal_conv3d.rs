#[cfg(feature = "candle-metal")]
fn main() -> candle_core::Result<()> {
    use candle_core::Device;

    let cpu = Device::Cpu;
    let metal = Device::new_metal(0)?;
    check(&cpu, &metal, (1, 2, 3, 4, 5), (2, 2, 2, 3, 3), 1, 1)?;
    check(&cpu, &metal, (1, 3, 5, 6, 7), (4, 3, 1, 1, 1), 0, 1)?;
    check(&cpu, &metal, (1, 2, 6, 7, 8), (3, 2, 3, 3, 3), 1, 2)?;
    check(&cpu, &metal, (1, 1, 8, 16, 16), (32, 1, 7, 7, 7), 3, 1)?;
    check_conv2d(&cpu, &metal, (1, 32, 17, 17), (64, 32, 3, 3), 0, 2)?;
    check_sliced_conv2d_sum(&cpu, &metal, (1, 32, 10, 17, 17), (64, 32, 3, 3, 3), 2)?;
    check_same_for_stride(&cpu, &metal, (1, 2, 5, 7, 9), (3, 2, 3, 3, 3), [1, 2, 2])?;
    check_same_for_stride(&cpu, &metal, (1, 4, 6, 13, 15), (8, 4, 3, 3, 3), [1, 2, 2])?;
    Ok(())
}

#[cfg(feature = "candle-metal")]
fn check(
    cpu: &candle_core::Device,
    metal: &candle_core::Device,
    input_shape: (usize, usize, usize, usize, usize),
    kernel_shape: (usize, usize, usize, usize, usize),
    padding: usize,
    stride: usize,
) -> candle_core::Result<()> {
    use candle_core::Tensor;

    let input_len = input_shape.0 * input_shape.1 * input_shape.2 * input_shape.3 * input_shape.4;
    let kernel_len =
        kernel_shape.0 * kernel_shape.1 * kernel_shape.2 * kernel_shape.3 * kernel_shape.4;
    let input = (0..input_len)
        .map(|v| ((v % 251) as f32 - 103.0) / 97.0)
        .collect::<Vec<_>>();
    let kernel = (0..kernel_len)
        .map(|v| ((v % 257) as f32 - 117.0) / 113.0)
        .collect::<Vec<_>>();
    let input_cpu = Tensor::from_vec(input.clone(), input_shape, cpu)?;
    let kernel_cpu = Tensor::from_vec(kernel.clone(), kernel_shape, cpu)?;
    let input_metal = Tensor::from_vec(input, input_shape, metal)?;
    let kernel_metal = Tensor::from_vec(kernel, kernel_shape, metal)?;
    let out_metal = input_metal
        .conv3d(&kernel_metal, padding, stride, 1, 1)?
        .to_device(&cpu)?;
    let out_cpu = input_cpu.conv3d(&kernel_cpu, padding, stride, 1, 1)?;
    let cpu_values = out_cpu.flatten_all()?.to_vec1::<f32>()?;
    let metal_values = out_metal.flatten_all()?.to_vec1::<f32>()?;
    let mut max_abs = 0.0f32;
    let mut first = None;
    for (index, (lhs, rhs)) in cpu_values.iter().zip(&metal_values).enumerate() {
        let diff = (*lhs - *rhs).abs();
        if diff > max_abs {
            max_abs = diff;
        }
        if first.is_none() && diff > 1e-4 {
            first = Some((index, *lhs, *rhs, diff));
        }
    }
    println!(
        "input={input_shape:?} kernel={kernel_shape:?} padding={padding} stride={stride} shape={:?} len={} max_abs={max_abs} first_mismatch={first:?}",
        out_cpu.shape(),
        cpu_values.len()
    );
    Ok(())
}

#[cfg(feature = "candle-metal")]
fn check_sliced_conv2d_sum(
    cpu: &candle_core::Device,
    metal: &candle_core::Device,
    input_shape: (usize, usize, usize, usize, usize),
    kernel_shape: (usize, usize, usize, usize, usize),
    stride: usize,
) -> candle_core::Result<()> {
    use candle_core::Tensor;

    let input_len = input_shape.0 * input_shape.1 * input_shape.2 * input_shape.3 * input_shape.4;
    let kernel_len =
        kernel_shape.0 * kernel_shape.1 * kernel_shape.2 * kernel_shape.3 * kernel_shape.4;
    let input = (0..input_len)
        .map(|v| ((v % 251) as f32 - 103.0) / 97.0)
        .collect::<Vec<_>>();
    let kernel = (0..kernel_len)
        .map(|v| ((v % 257) as f32 - 117.0) / 113.0)
        .collect::<Vec<_>>();
    let input_cpu = Tensor::from_vec(input.clone(), input_shape, cpu)?;
    let kernel_cpu = Tensor::from_vec(kernel.clone(), kernel_shape, cpu)?;
    let input_metal = Tensor::from_vec(input, input_shape, metal)?;
    let kernel_metal = Tensor::from_vec(kernel, kernel_shape, metal)?;
    let out_cpu = sliced_conv2d_sum(&input_cpu, &kernel_cpu, stride, false)?;
    let out_metal =
        sliced_conv2d_sum(&input_metal, &kernel_metal, stride, false)?.to_device(cpu)?;
    compare(
        "sliced_conv2d_sum",
        input_shape,
        out_cpu.flatten_all()?.to_vec1::<f32>()?,
        out_metal.flatten_all()?.to_vec1::<f32>()?,
        out_cpu.shape(),
    );
    let out_metal_contiguous =
        sliced_conv2d_sum(&input_metal, &kernel_metal, stride, true)?.to_device(cpu)?;
    compare(
        "sliced_conv2d_sum_contiguous",
        input_shape,
        out_cpu.flatten_all()?.to_vec1::<f32>()?,
        out_metal_contiguous.flatten_all()?.to_vec1::<f32>()?,
        out_cpu.shape(),
    );
    Ok(())
}

#[cfg(feature = "candle-metal")]
fn sliced_conv2d_sum(
    input: &candle_core::Tensor,
    kernel: &candle_core::Tensor,
    stride: usize,
    contiguous: bool,
) -> candle_core::Result<candle_core::Tensor> {
    use candle_core::Tensor;

    let kernel_d = kernel.dim(2)?;
    let mut partial: Option<Tensor> = None;
    for kernel_z in 0..kernel_d {
        let mut input_slice = input.narrow(2, kernel_z, 1)?.squeeze(2)?;
        let mut weight_slice = kernel.narrow(2, kernel_z, 1)?.squeeze(2)?;
        if contiguous {
            input_slice = input_slice.contiguous()?;
            weight_slice = weight_slice.contiguous()?;
        }
        let next = input_slice.conv2d(&weight_slice, 0, stride, 1, 1)?;
        partial = Some(match partial {
            Some(accumulated) => accumulated.broadcast_add(&next)?,
            None => next,
        });
    }
    partial.ok_or_else(|| candle_core::Error::Msg("empty convolution depth".to_string()))
}

#[cfg(feature = "candle-metal")]
fn check_conv2d(
    cpu: &candle_core::Device,
    metal: &candle_core::Device,
    input_shape: (usize, usize, usize, usize),
    kernel_shape: (usize, usize, usize, usize),
    padding: usize,
    stride: usize,
) -> candle_core::Result<()> {
    use candle_core::Tensor;

    let input_len = input_shape.0 * input_shape.1 * input_shape.2 * input_shape.3;
    let kernel_len = kernel_shape.0 * kernel_shape.1 * kernel_shape.2 * kernel_shape.3;
    let input = (0..input_len)
        .map(|v| ((v % 251) as f32 - 103.0) / 97.0)
        .collect::<Vec<_>>();
    let kernel = (0..kernel_len)
        .map(|v| ((v % 257) as f32 - 117.0) / 113.0)
        .collect::<Vec<_>>();
    let input_cpu = Tensor::from_vec(input.clone(), input_shape, cpu)?;
    let kernel_cpu = Tensor::from_vec(kernel.clone(), kernel_shape, cpu)?;
    let input_metal = Tensor::from_vec(input, input_shape, metal)?;
    let kernel_metal = Tensor::from_vec(kernel, kernel_shape, metal)?;
    let out_cpu = input_cpu.conv2d(&kernel_cpu, padding, stride, 1, 1)?;
    let out_metal = input_metal
        .conv2d(&kernel_metal, padding, stride, 1, 1)?
        .to_device(cpu)?;
    compare(
        "conv2d",
        (
            input_shape.0,
            input_shape.1,
            1,
            input_shape.2,
            input_shape.3,
        ),
        out_cpu.flatten_all()?.to_vec1::<f32>()?,
        out_metal.flatten_all()?.to_vec1::<f32>()?,
        out_cpu.shape(),
    );
    Ok(())
}

#[cfg(feature = "candle-metal")]
fn check_same_for_stride(
    cpu: &candle_core::Device,
    metal: &candle_core::Device,
    input_shape: (usize, usize, usize, usize, usize),
    kernel_shape: (usize, usize, usize, usize, usize),
    stride: [usize; 3],
) -> candle_core::Result<()> {
    use candle_core::Tensor;

    let input_len = input_shape.0 * input_shape.1 * input_shape.2 * input_shape.3 * input_shape.4;
    let kernel_len =
        kernel_shape.0 * kernel_shape.1 * kernel_shape.2 * kernel_shape.3 * kernel_shape.4;
    let input = (0..input_len)
        .map(|v| ((v % 251) as f32 - 103.0) / 97.0)
        .collect::<Vec<_>>();
    let kernel = (0..kernel_len)
        .map(|v| ((v % 257) as f32 - 117.0) / 113.0)
        .collect::<Vec<_>>();
    let input_cpu = Tensor::from_vec(input.clone(), input_shape, cpu)?;
    let kernel_cpu = Tensor::from_vec(kernel.clone(), kernel_shape, cpu)?;
    let input_metal = Tensor::from_vec(input, input_shape, metal)?;
    let kernel_metal = Tensor::from_vec(kernel, kernel_shape, metal)?;

    let padded_cpu = same_for_stride(
        &input_cpu,
        stride,
        [kernel_shape.2, kernel_shape.3, kernel_shape.4],
    )?;
    let padded_metal = same_for_stride(
        &input_metal,
        stride,
        [kernel_shape.2, kernel_shape.3, kernel_shape.4],
    )?;
    compare(
        "same_for_stride_pad",
        input_shape,
        padded_cpu.flatten_all()?.to_vec1::<f32>()?,
        padded_metal
            .to_device(cpu)?
            .flatten_all()?
            .to_vec1::<f32>()?,
        padded_cpu.shape(),
    );

    let out_cpu = padded_cpu.conv3d(&kernel_cpu, 0, stride[0], 1, 1)?;
    let out_metal = padded_metal
        .conv3d(&kernel_metal, 0, stride[0], 1, 1)?
        .to_device(cpu)?;
    compare(
        "same_for_stride_conv",
        input_shape,
        out_cpu.flatten_all()?.to_vec1::<f32>()?,
        out_metal.flatten_all()?.to_vec1::<f32>()?,
        out_cpu.shape(),
    );
    Ok(())
}

#[cfg(feature = "candle-metal")]
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

#[cfg(feature = "candle-metal")]
fn compare(
    name: &str,
    input_shape: (usize, usize, usize, usize, usize),
    cpu_values: Vec<f32>,
    metal_values: Vec<f32>,
    shape: &candle_core::Shape,
) {
    let mut max_abs = 0.0f32;
    let mut first = None;
    for (index, (lhs, rhs)) in cpu_values.iter().zip(&metal_values).enumerate() {
        let diff = (*lhs - *rhs).abs();
        if diff > max_abs {
            max_abs = diff;
        }
        if first.is_none() && diff > 1e-4 {
            first = Some((index, *lhs, *rhs, diff));
        }
    }
    println!(
        "{name}: input={input_shape:?} shape={shape:?} len={} max_abs={max_abs} first_mismatch={first:?}",
        cpu_values.len()
    );
}

#[cfg(not(feature = "candle-metal"))]
fn main() {
    eprintln!("requires --features candle-metal");
}
