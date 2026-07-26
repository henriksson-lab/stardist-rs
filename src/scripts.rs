use std::path::{Path, PathBuf};

#[derive(Clone, Debug, PartialEq)]
pub struct PredictScriptArgs {
    pub input: Vec<PathBuf>,
    pub outdir: PathBuf,
    pub outname: String,
    pub model: String,
    pub registered_models: Vec<String>,
    pub axes: Option<String>,
    pub n_tiles: Option<Vec<usize>>,
    pub pnorm: [f32; 2],
    pub prob_thresh: Option<f32>,
    pub nms_thresh: Option<f32>,
    pub verbose: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub struct PredictScriptImage {
    pub data: Vec<f32>,
    pub shape: Vec<usize>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct PredictScriptLabels {
    pub data: Vec<u32>,
    pub shape: Vec<usize>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct PredictScriptOutput {
    pub input: PathBuf,
    pub output: PathBuf,
    pub axes: String,
    pub model: String,
    pub n_tiles: Option<Vec<usize>>,
}

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum PredictScriptError {
    #[error("at least one input file is required")]
    EmptyInput,
    #[error("unknown model: {model}")]
    UnknownModel {
        model: String,
        available: Vec<String>,
    },
    #[error("currently only 2d and 3d images are supported by the prediction script")]
    Unsupported2DInputDim,
    #[error("currently only 3d or 4d images are supported by the prediction script")]
    Unsupported3DInputDim,
    #[error("dimension of input ({image_ndim}) not the same as length of given axes ({axes_len})")]
    AxesLengthMismatch { image_ndim: usize, axes_len: usize },
    #[error("number of tiles does not match the script dimensionality")]
    TilesLengthMismatch,
    #[error("only tiff files supported in 3D for now")]
    NonTiff3DInput,
    #[error("normalization percentiles must be finite and increasing")]
    InvalidPercentiles,
    #[error("prediction callback returned labels with incompatible shape")]
    LabelShapeMismatch,
    #[error("failed script I/O")]
    Io(String),
}

pub mod predict2d {
    use super::*;

    pub fn main<L, P, W>(
        mut args: PredictScriptArgs,
        mut load_image: L,
        mut predict_instances: P,
        mut write_labels: W,
    ) -> Result<Vec<PredictScriptOutput>, PredictScriptError>
    where
        L: FnMut(&Path) -> Result<PredictScriptImage, PredictScriptError>,
        P: FnMut(
            &str,
            &[f32],
            &[usize],
            &str,
            Option<&[usize]>,
            Option<f32>,
            Option<f32>,
        ) -> Result<PredictScriptLabels, PredictScriptError>,
        W: FnMut(&Path, &PredictScriptLabels) -> Result<(), PredictScriptError>,
    {
        if args.input.is_empty() {
            return Err(PredictScriptError::EmptyInput);
        }
        let model = if Path::new(&args.model).is_dir() {
            args.model.clone()
        } else if args
            .registered_models
            .iter()
            .any(|name| name == &args.model)
        {
            args.model.clone()
        } else {
            return Err(PredictScriptError::UnknownModel {
                model: args.model,
                available: args.registered_models,
            });
        };
        if let Some(n_tiles) = &args.n_tiles {
            if n_tiles.len() != 2 {
                return Err(PredictScriptError::TilesLengthMismatch);
            }
        }
        if !args.pnorm[0].is_finite()
            || !args.pnorm[1].is_finite()
            || args.pnorm[0] >= args.pnorm[1]
        {
            return Err(PredictScriptError::InvalidPercentiles);
        }

        std::fs::create_dir_all(&args.outdir)
            .map_err(|err| PredictScriptError::Io(err.to_string()))?;
        let mut outputs = Vec::<PredictScriptOutput>::with_capacity(args.input.len());
        for fname in &args.input {
            let image = load_image(fname)?;
            if image.shape.len() != 2 && image.shape.len() != 3 {
                return Err(PredictScriptError::Unsupported2DInputDim);
            }
            if args.axes.is_none() {
                args.axes = Some(if image.shape.len() == 2 {
                    "YX".to_string()
                } else {
                    "YXC".to_string()
                });
            }
            let axes = args.axes.clone().unwrap();
            if axes.len() != image.shape.len() {
                return Err(PredictScriptError::AxesLengthMismatch {
                    image_ndim: image.shape.len(),
                    axes_len: axes.len(),
                });
            }

            let mut sorted = image
                .data
                .iter()
                .copied()
                .filter(|value| value.is_finite())
                .collect::<Vec<_>>();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let mut normalized = vec![0.0f32; image.data.len()];
            if !sorted.is_empty() {
                let pos_min = (args.pnorm[0] / 100.0) * (sorted.len().saturating_sub(1) as f32);
                let lo_min = pos_min.floor() as usize;
                let hi_min = pos_min.ceil() as usize;
                let frac_min = pos_min - lo_min as f32;
                let pmin = sorted[lo_min] * (1.0 - frac_min) + sorted[hi_min] * frac_min;
                let pos_max = (args.pnorm[1] / 100.0) * (sorted.len().saturating_sub(1) as f32);
                let lo_max = pos_max.floor() as usize;
                let hi_max = pos_max.ceil() as usize;
                let frac_max = pos_max - lo_max as f32;
                let pmax = sorted[lo_max] * (1.0 - frac_max) + sorted[hi_max] * frac_max;
                if pmax > pmin {
                    for (out, value) in normalized.iter_mut().zip(image.data.iter()) {
                        *out = (*value - pmin) / (pmax - pmin);
                    }
                }
            }

            let labels = predict_instances(
                &model,
                &normalized,
                &image.shape,
                &axes,
                args.n_tiles.as_deref(),
                args.prob_thresh,
                args.nms_thresh,
            )?;
            if labels.shape.len() != 2
                || labels.data.len() != labels.shape.iter().product::<usize>()
            {
                return Err(PredictScriptError::LabelShapeMismatch);
            }
            let stem = fname
                .file_stem()
                .and_then(|name| name.to_str())
                .unwrap_or("image");
            let output = args.outdir.join(args.outname.replace("{img}", stem));
            write_labels(&output, &labels)?;
            outputs.push(PredictScriptOutput {
                input: fname.clone(),
                output,
                axes,
                model: model.clone(),
                n_tiles: args.n_tiles.clone(),
            });
        }
        Ok(outputs)
    }
}

pub mod predict3d {
    use super::*;

    pub fn main<L, P, W>(
        mut args: PredictScriptArgs,
        mut load_image: L,
        mut predict_instances: P,
        mut write_labels: W,
    ) -> Result<Vec<PredictScriptOutput>, PredictScriptError>
    where
        L: FnMut(&Path) -> Result<PredictScriptImage, PredictScriptError>,
        P: FnMut(
            &str,
            &[f32],
            &[usize],
            &str,
            Option<&[usize]>,
            Option<f32>,
            Option<f32>,
        ) -> Result<PredictScriptLabels, PredictScriptError>,
        W: FnMut(&Path, &PredictScriptLabels) -> Result<(), PredictScriptError>,
    {
        if args.input.is_empty() {
            return Err(PredictScriptError::EmptyInput);
        }
        let model = if Path::new(&args.model).is_dir() {
            args.model.clone()
        } else if args
            .registered_models
            .iter()
            .any(|name| name == &args.model)
        {
            args.model.clone()
        } else {
            return Err(PredictScriptError::UnknownModel {
                model: args.model,
                available: args.registered_models,
            });
        };
        if let Some(n_tiles) = &args.n_tiles {
            if n_tiles.len() != 3 {
                return Err(PredictScriptError::TilesLengthMismatch);
            }
        }
        if !args.pnorm[0].is_finite()
            || !args.pnorm[1].is_finite()
            || args.pnorm[0] >= args.pnorm[1]
        {
            return Err(PredictScriptError::InvalidPercentiles);
        }

        std::fs::create_dir_all(&args.outdir)
            .map_err(|err| PredictScriptError::Io(err.to_string()))?;
        let mut outputs = Vec::<PredictScriptOutput>::with_capacity(args.input.len());
        for fname in &args.input {
            let suffix = fname
                .extension()
                .and_then(|value| value.to_str())
                .unwrap_or("")
                .to_ascii_lowercase();
            if suffix != "tif" && suffix != "tiff" {
                return Err(PredictScriptError::NonTiff3DInput);
            }
            let image = load_image(fname)?;
            if image.shape.len() != 3 && image.shape.len() != 4 {
                return Err(PredictScriptError::Unsupported3DInputDim);
            }
            if args.axes.is_none() {
                args.axes = Some(if image.shape.len() == 3 {
                    "ZYX".to_string()
                } else {
                    "ZYXC".to_string()
                });
            }
            let axes = args.axes.clone().unwrap();
            if axes.len() != image.shape.len() {
                return Err(PredictScriptError::AxesLengthMismatch {
                    image_ndim: image.shape.len(),
                    axes_len: axes.len(),
                });
            }

            let mut sorted = image
                .data
                .iter()
                .copied()
                .filter(|value| value.is_finite())
                .collect::<Vec<_>>();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let mut normalized = vec![0.0f32; image.data.len()];
            if !sorted.is_empty() {
                let pos_min = (args.pnorm[0] / 100.0) * (sorted.len().saturating_sub(1) as f32);
                let lo_min = pos_min.floor() as usize;
                let hi_min = pos_min.ceil() as usize;
                let frac_min = pos_min - lo_min as f32;
                let pmin = sorted[lo_min] * (1.0 - frac_min) + sorted[hi_min] * frac_min;
                let pos_max = (args.pnorm[1] / 100.0) * (sorted.len().saturating_sub(1) as f32);
                let lo_max = pos_max.floor() as usize;
                let hi_max = pos_max.ceil() as usize;
                let frac_max = pos_max - lo_max as f32;
                let pmax = sorted[lo_max] * (1.0 - frac_max) + sorted[hi_max] * frac_max;
                if pmax > pmin {
                    for (out, value) in normalized.iter_mut().zip(image.data.iter()) {
                        *out = (*value - pmin) / (pmax - pmin);
                    }
                }
            }

            let labels = predict_instances(
                &model,
                &normalized,
                &image.shape,
                &axes,
                args.n_tiles.as_deref(),
                args.prob_thresh,
                args.nms_thresh,
            )?;
            if labels.shape.len() != 3
                || labels.data.len() != labels.shape.iter().product::<usize>()
            {
                return Err(PredictScriptError::LabelShapeMismatch);
            }
            let stem = fname
                .file_stem()
                .and_then(|name| name.to_str())
                .unwrap_or("image");
            let output = args.outdir.join(args.outname.replace("{img}", stem));
            write_labels(&output, &labels)?;
            outputs.push(PredictScriptOutput {
                input: fname.clone(),
                output,
                axes,
                model: model.clone(),
                n_tiles: args.n_tiles.clone(),
            });
        }
        Ok(outputs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn predict2d_main_defaults_axes_normalizes_and_writes_named_output() {
        let outdir = std::env::temp_dir().join(format!(
            "stardist_rs_predict2d_script_{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&outdir);
        let args = PredictScriptArgs {
            input: vec![PathBuf::from("cell.tif")],
            outdir: outdir.clone(),
            outname: "{img}.stardist.tif".to_string(),
            model: "2D_demo".to_string(),
            registered_models: vec!["2D_demo".to_string()],
            axes: None,
            n_tiles: Some(vec![1, 1]),
            pnorm: [0.0, 100.0],
            prob_thresh: Some(0.4),
            nms_thresh: Some(0.3),
            verbose: false,
        };
        let mut wrote = Vec::<PathBuf>::new();

        let outputs = predict2d::main(
            args,
            |_| {
                Ok(PredictScriptImage {
                    data: vec![0.0, 5.0, 10.0, 15.0],
                    shape: vec![2, 2],
                })
            },
            |model, image, shape, axes, n_tiles, prob_thresh, nms_thresh| {
                assert_eq!(model, "2D_demo");
                assert_eq!(shape, &[2, 2]);
                assert_eq!(axes, "YX");
                assert_eq!(n_tiles, Some(&[1, 1][..]));
                assert_eq!(prob_thresh, Some(0.4));
                assert_eq!(nms_thresh, Some(0.3));
                assert!((image[0] - 0.0).abs() < 1e-6);
                assert!((image[1] - 1.0 / 3.0).abs() < 1e-6);
                assert!((image[2] - 2.0 / 3.0).abs() < 1e-6);
                assert!((image[3] - 1.0).abs() < 1e-6);
                Ok(PredictScriptLabels {
                    data: vec![0, 1, 0, 2],
                    shape: vec![2, 2],
                })
            },
            |path, labels| {
                assert_eq!(labels.shape, vec![2, 2]);
                wrote.push(path.to_path_buf());
                Ok(())
            },
        )
        .unwrap();

        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].axes, "YX");
        assert_eq!(outputs[0].output, outdir.join("cell.stardist.tif"));
        assert_eq!(wrote, vec![outdir.join("cell.stardist.tif")]);
        let _ = std::fs::remove_dir_all(outdir);
    }

    #[test]
    fn predict2d_main_reports_unknown_model_like_script_preflight() {
        let err = predict2d::main(
            PredictScriptArgs {
                input: vec![PathBuf::from("cell.tif")],
                outdir: PathBuf::from("."),
                outname: "{img}.stardist.tif".to_string(),
                model: "missing".to_string(),
                registered_models: vec!["2D_demo".to_string()],
                axes: None,
                n_tiles: None,
                pnorm: [1.0, 99.8],
                prob_thresh: None,
                nms_thresh: None,
                verbose: false,
            },
            |_| unreachable!(),
            |_, _, _, _, _, _, _| unreachable!(),
            |_, _| unreachable!(),
        )
        .unwrap_err();
        assert_eq!(
            err,
            PredictScriptError::UnknownModel {
                model: "missing".to_string(),
                available: vec!["2D_demo".to_string()],
            }
        );
    }

    #[test]
    fn predict3d_main_enforces_tiff_suffix_and_defaults_zyx_axes() {
        let non_tiff_err = predict3d::main(
            PredictScriptArgs {
                input: vec![PathBuf::from("volume.png")],
                outdir: PathBuf::from("."),
                outname: "{img}.stardist.tif".to_string(),
                model: "3D_demo".to_string(),
                registered_models: vec!["3D_demo".to_string()],
                axes: None,
                n_tiles: Some(vec![1, 1, 1]),
                pnorm: [1.0, 99.8],
                prob_thresh: None,
                nms_thresh: None,
                verbose: false,
            },
            |_| unreachable!(),
            |_, _, _, _, _, _, _| unreachable!(),
            |_, _| unreachable!(),
        )
        .unwrap_err();
        assert_eq!(non_tiff_err, PredictScriptError::NonTiff3DInput);

        let outdir = std::env::temp_dir().join(format!(
            "stardist_rs_predict3d_script_{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&outdir);
        let outputs = predict3d::main(
            PredictScriptArgs {
                input: vec![PathBuf::from("volume.tiff")],
                outdir: outdir.clone(),
                outname: "{img}.stardist.tif".to_string(),
                model: "3D_demo".to_string(),
                registered_models: vec!["3D_demo".to_string()],
                axes: None,
                n_tiles: Some(vec![1, 1, 1]),
                pnorm: [0.0, 100.0],
                prob_thresh: None,
                nms_thresh: None,
                verbose: false,
            },
            |_| {
                Ok(PredictScriptImage {
                    data: vec![1.0; 2 * 3 * 4],
                    shape: vec![2, 3, 4],
                })
            },
            |_, image, shape, axes, n_tiles, _, _| {
                assert_eq!(image.len(), 2 * 3 * 4);
                assert_eq!(shape, &[2, 3, 4]);
                assert_eq!(axes, "ZYX");
                assert_eq!(n_tiles, Some(&[1, 1, 1][..]));
                Ok(PredictScriptLabels {
                    data: vec![0; 2 * 3 * 4],
                    shape: vec![2, 3, 4],
                })
            },
            |_, _| Ok(()),
        )
        .unwrap();

        assert_eq!(outputs[0].axes, "ZYX");
        assert_eq!(outputs[0].output, outdir.join("volume.stardist.tif"));
        let _ = std::fs::remove_dir_all(outdir);
    }
}
