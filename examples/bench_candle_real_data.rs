#[cfg(all(feature = "candle", feature = "hdf5"))]
mod bench {
    use std::{
        env,
        error::Error,
        fs::File,
        path::{Path, PathBuf},
        time::Instant,
    };

    use candle_core::{Device, Tensor};
    use ndarray::{Array, IxDyn};
    use ndarray_npy::NpzReader;
    use stardist_rs::{
        PolyhedronRenderMode, StarDist2D, StarDist3D, StarDistDirectPrediction,
        weights::load_keras_hdf5_weights,
    };

    #[derive(Clone)]
    struct TensorFixture {
        shape: Vec<usize>,
        values: Vec<f32>,
    }

    #[derive(Clone)]
    struct LabelFixture {
        values: Vec<u32>,
    }

    pub fn main() -> Result<(), Box<dyn Error>> {
        let args = env::args().collect::<Vec<_>>();
        if args.len() < 3 || (args[1] != "2d" && args[1] != "3d") {
            eprintln!(
                "usage: cargo run --release --features candle,hdf5 --example bench_candle_real_data -- <2d|3d> <python_npz> [cpu|cuda|metal]"
            );
            std::process::exit(2);
        }
        match args[1].as_str() {
            "2d" => run_2d(Path::new(&args[2]), args.get(3).map(String::as_str))?,
            "3d" => run_3d(Path::new(&args[2]), args.get(3).map(String::as_str))?,
            _ => unreachable!(),
        }
        Ok(())
    }

    fn run_2d(npz_path: &Path, device_arg: Option<&str>) -> Result<(), Box<dyn Error>> {
        let mut npz = NpzReader::new(File::open(npz_path)?)?;
        let input = read_f32(&mut npz, "input_nchw.npy")?;
        let python_raw_prob = read_f32(&mut npz, "raw_prob_nchw.npy")?;
        let python_raw_dist = read_f32(&mut npz, "raw_dist_nchw.npy")?;
        let python_sparse_prob = read_f32(&mut npz, "sparse_prob.npy")?;
        let python_sparse_dist = read_f32(&mut npz, "sparse_dist.npy")?;
        let python_sparse_points = read_f32(&mut npz, "sparse_points.npy")?;
        let python_labels = read_u32(&mut npz, "labels.npy")?;
        let python_points = read_f32(&mut npz, "points.npy")?;
        let python_instances_prob = read_f32(&mut npz, "prob.npy")?;

        let input_shape = as_array::<4>(&input.shape)?;
        let device = device_from_arg(device_arg)?;
        let stardist = StarDist2D::from_model_dir("assets/models/examples/2D_demo")?;
        let config = stardist.config.clone();
        let model = stardist_rs::model::candle::StarDist2D::init(config.clone(), &device)
            .load_keras_weights(
                &load_keras_hdf5_weights(weights_path("2D_demo", "weights_best.h5"))?,
                &device,
            )?;

        let load_rss_kib = max_rss_kib();
        let tensor = Tensor::from_vec(
            input.values.clone(),
            (
                input_shape[0],
                input_shape[1],
                input_shape[2],
                input_shape[3],
            ),
            &device,
        )?;
        let started = Instant::now();
        let outputs = model.forward(&tensor)?;
        let inference_seconds = started.elapsed().as_secs_f64();
        let prob = tensor_to_vec(outputs.prob)?;
        let dist_nchw = tensor_to_vec(outputs.dist)?;

        let image_yx = nchw_single_image_to_yx(&input)?;
        let started = Instant::now();
        let sparse = stardist.predict_sparse(
            &image_yx,
            &[input_shape[2], input_shape[3]],
            None,
            Some("YX"),
            None,
            2,
            |x, x_shape, axes| predict_direct_2d(&model, &config, &device, x, x_shape, axes),
        )?;
        let predict_sparse_seconds = started.elapsed().as_secs_f64();
        let started = Instant::now();
        let instances = stardist._instances_from_prediction(
            [input_shape[2], input_shape[3]],
            &sparse.prob,
            [0, 0],
            &sparse.dist,
            Some(&sparse.points),
            None,
            None,
            None,
            true,
            None,
            None,
            true,
            true,
        )?;
        let postprocess_seconds = started.elapsed().as_secs_f64();
        let labels = instances.labels.expect("labels requested");
        let rust_labels = labels.iter().copied().collect::<Vec<_>>();

        print_report(
            "2d",
            device_arg.unwrap_or("cpu"),
            inference_seconds,
            predict_sparse_seconds,
            postprocess_seconds,
            load_rss_kib,
            &[
                ("raw_prob", compare_f32(&prob, &python_raw_prob.values)),
                ("raw_dist", compare_f32(&dist_nchw, &python_raw_dist.values)),
                (
                    "sparse_prob",
                    compare_f32(&sparse.prob, &python_sparse_prob.values),
                ),
                (
                    "sparse_dist",
                    compare_f32(&sparse.dist, &python_sparse_dist.values),
                ),
                (
                    "sparse_points",
                    compare_f32(
                        &flatten_points2(&sparse.points),
                        &python_sparse_points.values,
                    ),
                ),
                (
                    "instance_prob",
                    compare_f32(&instances.prob, &python_instances_prob.values),
                ),
                (
                    "points",
                    compare_f32(&flatten_points2(&instances.points), &python_points.values),
                ),
            ],
            compare_labels_u32(&rust_labels, &python_labels.values),
        );
        Ok(())
    }

    fn run_3d(npz_path: &Path, device_arg: Option<&str>) -> Result<(), Box<dyn Error>> {
        let mut npz = NpzReader::new(File::open(npz_path)?)?;
        let input = read_f32(&mut npz, "input_ncdhw.npy")?;
        let python_raw_prob = read_f32(&mut npz, "raw_prob_ncdhw.npy")?;
        let python_raw_dist = read_f32(&mut npz, "raw_dist_ncdhw.npy")?;
        let python_sparse_prob = read_f32(&mut npz, "sparse_prob.npy")?;
        let python_sparse_dist = read_f32(&mut npz, "sparse_dist.npy")?;
        let python_sparse_points = read_f32(&mut npz, "sparse_points.npy")?;
        let python_labels = read_u32(&mut npz, "labels.npy")?;
        let python_points = read_f32(&mut npz, "points.npy")?;
        let python_instances_prob = read_f32(&mut npz, "prob.npy")?;
        let python_instances_dist = read_f32(&mut npz, "dist.npy")?;

        let input_shape = as_array::<5>(&input.shape)?;
        let device = device_from_arg(device_arg)?;
        let stardist = StarDist3D::from_model_dir("assets/models/examples/3D_demo")?;
        let config = stardist.config.clone();
        let model = stardist_rs::model::candle::StarDist3D::init(config.clone(), &device)
            .load_keras_weights(
                &load_keras_hdf5_weights(weights_path("3D_demo", "weights_best.h5"))?,
                &device,
            )?;

        let load_rss_kib = max_rss_kib();
        let tensor = Tensor::from_vec(
            input.values.clone(),
            (
                input_shape[0],
                input_shape[1],
                input_shape[2],
                input_shape[3],
                input_shape[4],
            ),
            &device,
        )?;
        let started = Instant::now();
        let outputs = model.forward(&tensor)?;
        let inference_seconds = started.elapsed().as_secs_f64();
        let prob = tensor_to_vec(outputs.prob)?;
        let dist_ncdhw = tensor_to_vec(outputs.dist)?;

        let image_zyx = ncdhw_single_image_to_zyx(&input)?;
        let started = Instant::now();
        let sparse = stardist.predict_sparse(
            &image_zyx,
            &[input_shape[2], input_shape[3], input_shape[4]],
            None,
            Some("ZYX"),
            None,
            2,
            |x, x_shape, axes| predict_direct_3d(&model, &config, &device, x, x_shape, axes),
        )?;
        let predict_sparse_seconds = started.elapsed().as_secs_f64();
        let started = Instant::now();
        let instances = stardist._instances_from_prediction(
            [input_shape[2], input_shape[3], input_shape[4]],
            &sparse.prob,
            [0, 0, 0],
            &sparse.dist,
            Some(&sparse.points),
            None,
            None,
            None,
            None,
            true,
            None,
            None,
            true,
            true,
            true,
            PolyhedronRenderMode::Full,
        )?;
        let postprocess_seconds = started.elapsed().as_secs_f64();
        let labels = instances.labels.expect("labels requested");
        let rust_labels = labels.iter().copied().collect::<Vec<_>>();

        print_report(
            "3d",
            device_arg.unwrap_or("cpu"),
            inference_seconds,
            predict_sparse_seconds,
            postprocess_seconds,
            load_rss_kib,
            &[
                ("raw_prob", compare_f32(&prob, &python_raw_prob.values)),
                (
                    "raw_dist",
                    compare_f32(&dist_ncdhw, &python_raw_dist.values),
                ),
                (
                    "sparse_prob",
                    compare_f32(&sparse.prob, &python_sparse_prob.values),
                ),
                (
                    "sparse_dist",
                    compare_f32(&sparse.dist, &python_sparse_dist.values),
                ),
                (
                    "sparse_points",
                    compare_f32(
                        &flatten_points3(&sparse.points),
                        &python_sparse_points.values,
                    ),
                ),
                (
                    "instance_prob",
                    compare_f32(&instances.prob, &python_instances_prob.values),
                ),
                (
                    "instance_dist",
                    compare_f32(&instances.dist, &python_instances_dist.values),
                ),
                (
                    "points",
                    compare_f32(&flatten_points3(&instances.points), &python_points.values),
                ),
            ],
            compare_labels_i32(&rust_labels, &python_labels.values),
        );
        Ok(())
    }

    fn predict_direct_2d(
        model: &stardist_rs::model::candle::StarDist2D,
        config: &stardist_rs::Config2D,
        device: &Device,
        x: &[f32],
        x_shape: &[usize],
        axes: &str,
    ) -> Result<StarDistDirectPrediction, stardist_rs::StarDistPredictError> {
        if axes != "YXC" || x_shape.len() != 3 || x_shape[2] != 1 {
            return Err(stardist_rs::StarDistPredictError::OutputShapeMismatch);
        }
        let height = x_shape[0];
        let width = x_shape[1];
        let mut nchw = vec![0.0f32; height * width];
        for y in 0..height {
            for x_i in 0..width {
                nchw[y * width + x_i] = x[(y * width + x_i) * x_shape[2]];
            }
        }
        let input = Tensor::from_vec(nchw, (1, 1, height, width), device)
            .map_err(|_| stardist_rs::StarDistPredictError::OutputShapeMismatch)?;
        let outputs = model
            .forward(&input)
            .map_err(|_| stardist_rs::StarDistPredictError::OutputShapeMismatch)?;
        let prob_nchw = tensor_to_vec(outputs.prob)
            .map_err(|_| stardist_rs::StarDistPredictError::OutputShapeMismatch)?;
        let dist_nchw = tensor_to_vec(outputs.dist)
            .map_err(|_| stardist_rs::StarDistPredictError::OutputShapeMismatch)?;
        let prob_h = height / config.grid[0];
        let prob_w = width / config.grid[1];
        Ok(StarDistDirectPrediction {
            prob: nchw_prob_to_yxc(&prob_nchw, prob_h, prob_w),
            prob_shape: vec![prob_h, prob_w, 1],
            dist: nchw_dist_to_yxc(&dist_nchw, config.n_rays, prob_h, prob_w),
            dist_shape: vec![prob_h, prob_w, config.n_rays],
            prob_class: None,
            prob_class_shape: None,
        })
    }

    fn predict_direct_3d(
        model: &stardist_rs::model::candle::StarDist3D,
        config: &stardist_rs::Config3D,
        device: &Device,
        x: &[f32],
        x_shape: &[usize],
        axes: &str,
    ) -> Result<StarDistDirectPrediction, stardist_rs::StarDistPredictError> {
        if axes != "ZYXC" || x_shape.len() != 4 || x_shape[3] != 1 {
            return Err(stardist_rs::StarDistPredictError::OutputShapeMismatch);
        }
        let depth = x_shape[0];
        let height = x_shape[1];
        let width = x_shape[2];
        let mut ncdhw = vec![0.0f32; depth * height * width];
        for z in 0..depth {
            for y in 0..height {
                for x_i in 0..width {
                    ncdhw[(z * height + y) * width + x_i] =
                        x[((z * height + y) * width + x_i) * x_shape[3]];
                }
            }
        }
        let input = Tensor::from_vec(ncdhw, (1, 1, depth, height, width), device)
            .map_err(|_| stardist_rs::StarDistPredictError::OutputShapeMismatch)?;
        let outputs = model
            .forward(&input)
            .map_err(|_| stardist_rs::StarDistPredictError::OutputShapeMismatch)?;
        let prob_ncdhw = tensor_to_vec(outputs.prob)
            .map_err(|_| stardist_rs::StarDistPredictError::OutputShapeMismatch)?;
        let dist_ncdhw = tensor_to_vec(outputs.dist)
            .map_err(|_| stardist_rs::StarDistPredictError::OutputShapeMismatch)?;
        let prob_d = depth / config.grid[0];
        let prob_h = height / config.grid[1];
        let prob_w = width / config.grid[2];
        Ok(StarDistDirectPrediction {
            prob: ncdhw_prob_to_zyxc(&prob_ncdhw, prob_d, prob_h, prob_w),
            prob_shape: vec![prob_d, prob_h, prob_w, 1],
            dist: ncdhw_dist_to_zyxc(&dist_ncdhw, config.n_rays, prob_d, prob_h, prob_w),
            dist_shape: vec![prob_d, prob_h, prob_w, config.n_rays],
            prob_class: None,
            prob_class_shape: None,
        })
    }

    fn device_from_arg(arg: Option<&str>) -> Result<Device, Box<dyn Error>> {
        match arg.unwrap_or("cpu") {
            "cpu" => Ok(Device::Cpu),
            "cuda" => {
                #[cfg(feature = "candle-cuda")]
                {
                    print_cuda_runtime_report();
                    return Ok(Device::new_cuda(0)?);
                }
                #[cfg(not(feature = "candle-cuda"))]
                {
                    Err("CUDA support requires --features candle-cuda,hdf5".into())
                }
            }
            "metal" => {
                #[cfg(feature = "candle-metal")]
                {
                    return Ok(Device::new_metal(0)?);
                }
                #[cfg(not(feature = "candle-metal"))]
                {
                    Err("Metal support requires --features candle-metal,hdf5".into())
                }
            }
            other => Err(format!("unsupported Candle device {other:?}").into()),
        }
    }

    #[cfg(feature = "candle-cuda")]
    fn print_cuda_runtime_report() {
        let Ok(exe) = env::current_exe() else {
            return;
        };
        let Ok(output) = std::process::Command::new("ldd").arg(exe).output() else {
            return;
        };
        if !output.status.success() {
            return;
        }
        let text = String::from_utf8_lossy(&output.stdout);
        let libs = text
            .lines()
            .filter(|line| {
                line.contains("libcuda")
                    || line.contains("libcudart")
                    || line.contains("libcublas")
                    || line.contains("libcudnn")
                    || line.contains("libnvrtc")
                    || line.contains("libcurand")
            })
            .collect::<Vec<_>>();
        if libs.is_empty() {
            return;
        }
        eprintln!("CUDA runtime libraries linked by this benchmark:");
        for lib in &libs {
            eprintln!("  {}", lib.trim());
        }
        let has_cuda12 = libs.iter().any(|line| {
            line.contains("libcudart.so.12")
                || line.contains("libcublas.so.12")
                || line.contains("cuda-12")
        });
        let has_cuda13 = libs.iter().any(|line| {
            line.contains("libcudart.so.13")
                || line.contains("libcublas.so.13")
                || line.contains("mlx_cuda_v13")
                || line.contains("cuda/targets")
        });
        if has_cuda12 && has_cuda13 {
            eprintln!(
                "warning: CUDA 12 and CUDA 13 runtime libraries are mixed; this can cause CUBLAS_STATUS_NOT_INITIALIZED or CUDNN_STATUS_NOT_INITIALIZED"
            );
        }
    }

    fn tensor_to_vec(tensor: Tensor) -> candle_core::Result<Vec<f32>> {
        tensor.to_device(&Device::Cpu)?.flatten_all()?.to_vec1()
    }

    fn read_f32(npz: &mut NpzReader<File>, name: &str) -> Result<TensorFixture, Box<dyn Error>> {
        let array: Array<f32, IxDyn> = npz.by_name(name)?;
        Ok(TensorFixture {
            shape: array.shape().to_vec(),
            values: array.iter().copied().collect(),
        })
    }

    fn read_u32(npz: &mut NpzReader<File>, name: &str) -> Result<LabelFixture, Box<dyn Error>> {
        let array: Array<u32, IxDyn> = npz.by_name(name)?;
        Ok(LabelFixture {
            values: array.iter().copied().collect(),
        })
    }

    fn as_array<const N: usize>(shape: &[usize]) -> Result<[usize; N], Box<dyn Error>> {
        shape
            .try_into()
            .map_err(|_| format!("expected {N} dimensions, got {shape:?}").into())
    }

    fn weights_path(model: &str, file: &str) -> PathBuf {
        Path::new("stardist")
            .join("models")
            .join("examples")
            .join(model)
            .join(file)
    }

    fn max_rss_kib() -> u64 {
        let Ok(status) = std::fs::read_to_string("/proc/self/status") else {
            return 0;
        };
        for line in status.lines() {
            if let Some(rest) = line.strip_prefix("VmHWM:") {
                return rest
                    .split_whitespace()
                    .next()
                    .and_then(|value| value.parse().ok())
                    .unwrap_or(0);
            }
        }
        0
    }

    fn nchw_single_image_to_yx(input: &TensorFixture) -> Result<Vec<f32>, Box<dyn Error>> {
        let [batch, channels, height, width] = as_array::<4>(&input.shape)?;
        if batch != 1 || channels != 1 {
            return Err("expected single-channel NCHW 2D fixture".into());
        }
        Ok(input.values[0..height * width].to_vec())
    }

    fn ncdhw_single_image_to_zyx(input: &TensorFixture) -> Result<Vec<f32>, Box<dyn Error>> {
        let [batch, channels, depth, height, width] = as_array::<5>(&input.shape)?;
        if batch != 1 || channels != 1 {
            return Err("expected single-channel NCDHW 3D fixture".into());
        }
        Ok(input.values[0..depth * height * width].to_vec())
    }

    fn nchw_prob_to_yxc(prob: &[f32], height: usize, width: usize) -> Vec<f32> {
        let mut out = vec![0.0f32; height * width];
        for y in 0..height {
            for x in 0..width {
                out[y * width + x] = prob[y * width + x];
            }
        }
        out
    }

    fn nchw_dist_to_yxc(dist: &[f32], rays: usize, height: usize, width: usize) -> Vec<f32> {
        let mut out = vec![0.0f32; height * width * rays];
        for ray in 0..rays {
            for y in 0..height {
                for x in 0..width {
                    out[(y * width + x) * rays + ray] = dist[(ray * height + y) * width + x];
                }
            }
        }
        out
    }

    fn ncdhw_prob_to_zyxc(prob: &[f32], depth: usize, height: usize, width: usize) -> Vec<f32> {
        let mut out = vec![0.0f32; depth * height * width];
        for z in 0..depth {
            for y in 0..height {
                for x in 0..width {
                    out[(z * height + y) * width + x] = prob[(z * height + y) * width + x];
                }
            }
        }
        out
    }

    fn ncdhw_dist_to_zyxc(
        dist: &[f32],
        rays: usize,
        depth: usize,
        height: usize,
        width: usize,
    ) -> Vec<f32> {
        let mut out = vec![0.0f32; depth * height * width * rays];
        for ray in 0..rays {
            for z in 0..depth {
                for y in 0..height {
                    for x in 0..width {
                        out[((z * height + y) * width + x) * rays + ray] =
                            dist[((ray * depth + z) * height + y) * width + x];
                    }
                }
            }
        }
        out
    }

    #[derive(Clone, Copy)]
    struct F32Diff {
        len_rust: usize,
        len_python: usize,
        max_abs: f32,
        mean_abs: f64,
    }

    #[derive(Clone, Copy)]
    struct LabelDiff {
        len_rust: usize,
        len_python: usize,
        mismatches: usize,
    }

    fn compare_f32(rust: &[f32], python: &[f32]) -> F32Diff {
        let len = rust.len().min(python.len());
        let mut max_abs = 0.0f32;
        let mut sum_abs = 0.0f64;
        for i in 0..len {
            let diff = (rust[i] - python[i]).abs();
            max_abs = max_abs.max(diff);
            sum_abs += diff as f64;
        }
        F32Diff {
            len_rust: rust.len(),
            len_python: python.len(),
            max_abs,
            mean_abs: if len == 0 { 0.0 } else { sum_abs / len as f64 },
        }
    }

    fn compare_labels_u32(rust: &[u32], python: &[u32]) -> LabelDiff {
        let len = rust.len().min(python.len());
        let mismatches = (0..len).filter(|i| rust[*i] != python[*i]).count()
            + rust.len().max(python.len())
            - len;
        LabelDiff {
            len_rust: rust.len(),
            len_python: python.len(),
            mismatches,
        }
    }

    fn compare_labels_i32(rust: &[i32], python: &[u32]) -> LabelDiff {
        let len = rust.len().min(python.len());
        let mismatches = (0..len).filter(|i| rust[*i] != python[*i] as i32).count()
            + rust.len().max(python.len())
            - len;
        LabelDiff {
            len_rust: rust.len(),
            len_python: python.len(),
            mismatches,
        }
    }

    fn flatten_points2(points: &[[f32; 2]]) -> Vec<f32> {
        let mut out = Vec::with_capacity(points.len() * 2);
        for point in points {
            out.extend_from_slice(point);
        }
        out
    }

    fn flatten_points3(points: &[[f32; 3]]) -> Vec<f32> {
        let mut out = Vec::with_capacity(points.len() * 3);
        for point in points {
            out.extend_from_slice(point);
        }
        out
    }

    fn print_report(
        dimension: &str,
        device: &str,
        inference_seconds: f64,
        predict_sparse_seconds: f64,
        postprocess_seconds: f64,
        load_rss_kib: u64,
        f32_diffs: &[(&str, F32Diff)],
        label_diff: LabelDiff,
    ) {
        println!("{{");
        println!("  \"backend\": \"rust-candle\",");
        println!("  \"device\": \"{device}\",");
        println!("  \"dimension\": \"{dimension}\",");
        println!("  \"raw_inference_seconds\": {inference_seconds:.9},");
        println!("  \"predict_sparse_seconds\": {predict_sparse_seconds:.9},");
        println!("  \"postprocess_seconds\": {postprocess_seconds:.9},");
        println!("  \"load_max_rss_kib\": {load_rss_kib},");
        println!("  \"max_rss_kib\": {},", max_rss_kib());
        println!(
            "  \"labels\": {{ \"len_rust\": {}, \"len_python\": {}, \"mismatches\": {} }},",
            label_diff.len_rust, label_diff.len_python, label_diff.mismatches
        );
        println!("  \"arrays\": {{");
        for (i, (name, diff)) in f32_diffs.iter().enumerate() {
            let comma = if i + 1 == f32_diffs.len() { "" } else { "," };
            println!(
                "    \"{name}\": {{ \"len_rust\": {}, \"len_python\": {}, \"max_abs\": {:.9}, \"mean_abs\": {:.9} }}{comma}",
                diff.len_rust, diff.len_python, diff.max_abs, diff.mean_abs
            );
        }
        println!("  }}");
        println!("}}");
    }
}

#[cfg(all(feature = "candle", feature = "hdf5"))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    bench::main()
}

#[cfg(not(all(feature = "candle", feature = "hdf5")))]
fn main() {
    eprintln!("bench_candle_real_data requires --features candle,hdf5");
    std::process::exit(2);
}
