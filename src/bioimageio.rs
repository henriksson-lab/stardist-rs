use std::collections::BTreeMap;
use std::fs::{self, File};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use thiserror::Error;

pub const DEEPIMAGEJ_MACRO: &str = r#"
//*******************************************************************
// Date: July-2021
// Credits: StarDist, DeepImageJ
// URL:
//      https://github.com/stardist/stardist
//      https://deepimagej.github.io/deepimagej
// This macro was adapted from
// https://github.com/deepimagej/imagej-macros/blob/648caa867f6ccb459649d4d3799efa1e2e0c5204/StarDist2D_Post-processing.ijm
// Please cite the respective contributions when using this code.
//*******************************************************************
//  Macro to run StarDist postprocessing on 2D images.
//  StarDist and deepImageJ plugins need to be installed.
//  The macro assumes that the image to process is a stack in which
//  the first channel corresponds to the object probability map
//  and the remaining channels are the radial distances from each
//  pixel to the object boundary.
//*******************************************************************

// Get the name of the image to call it
getDimensions(width, height, channels, slices, frames);
name=getTitle();

probThresh={probThresh};
nmsThresh={nmsThresh};

// Isolate the detection probability scores
run("Make Substack...", "channels=1");
rename("scores");

// Isolate the oriented distances
run("Fire");
selectWindow(name);
run("Delete Slice", "delete=channel");
selectWindow(name);
run("Properties...", "channels=" + maxOf(channels, slices) - 1 + " slices=1 frames=1 pixel_width=1.0000 pixel_height=1.0000 voxel_depth=1.0000");
rename("distances");
run("royal");

// Run StarDist plugin
run("Command From Macro", "command=[de.csbdresden.stardist.StarDist2DNMS], args=['prob':'scores', 'dist':'distances', 'probThresh':'" + probThresh + "', 'nmsThresh':'" + nmsThresh + "', 'outputType':'Both', 'excludeBoundary':'2', 'roiPosition':'Stack', 'verbose':'false'], process=[false]");
"#;

pub const BIOIMAGEIO_MISSING_DEPENDENCIES: &str = "Required libraries are missing for bioimage.io model export.\nPlease install StarDist as follows: pip install 'stardist[bioimageio]'\n(You do not need to uninstall StarDist first.)";

#[derive(Debug, Error)]
pub enum BioimageioError {
    #[error("failed to write bioimageio file")]
    Io(#[from] std::io::Error),
    #[error("failed to read bioimageio zip archive")]
    Zip(#[from] zip::result::ZipError),
    #[error("tensorflow version must include a major version")]
    InvalidTensorflowVersion,
    #[error("export to keras format is not supported yet")]
    KerasExportUnsupported,
    #[error("tensorflow SavedModel execution is not available in this Rust crate")]
    TensorflowRuntimeUnavailable,
    #[error("tensorflow SavedModel export is not available in this Rust crate")]
    TensorflowExportUnavailable,
    #[error("Tensorflow SavedModel not supported for multiclass models yet")]
    TensorflowSavedModelMulticlassUnsupported,
    #[error("unsupported BioImageIO export mode {0}")]
    UnsupportedMode(String),
    #[error("invalid percentile values")]
    InvalidPercentiles,
    #[error("outpath has to be a folder or zip file")]
    InvalidOutpath,
    #[error("output path already exists")]
    OutputPathExists,
    #[error("bioimage.io model not compatible")]
    IncompatibleModel,
    #[error("couldn't find weights file '{0}'")]
    MissingWeights(String),
    #[error("{0}")]
    MissingDependencies(String),
}

#[derive(Clone, Debug, PartialEq)]
pub struct BioimageioCitation {
    pub text: String,
    pub doi: String,
}

#[derive(Clone, Debug, PartialEq)]
pub struct BioimageioAuthor {
    pub name: String,
    pub github_user: Option<String>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct StardistMetadata {
    pub description: String,
    pub authors: Vec<BioimageioAuthor>,
    pub git_repo: String,
    pub license: String,
    pub cite: Vec<BioimageioCitation>,
    pub tags: Vec<String>,
    pub covers: Vec<String>,
    pub documentation: PathBuf,
    pub dependencies: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum BioimageioMode {
    KerasHdf5,
    TensorflowSavedModelBundle,
    Other(String),
}

impl BioimageioMode {
    pub fn as_str(&self) -> &str {
        match self {
            Self::KerasHdf5 => "keras_hdf5",
            Self::TensorflowSavedModelBundle => "tensorflow_saved_model_bundle",
            Self::Other(mode) => mode.as_str(),
        }
    }
}

impl From<&str> for BioimageioMode {
    fn from(value: &str) -> Self {
        match value {
            "keras_hdf5" => Self::KerasHdf5,
            "tensorflow_saved_model_bundle" => Self::TensorflowSavedModelBundle,
            other => Self::Other(other.to_string()),
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct BioimageioThresholds {
    pub prob: f32,
    pub nms: f32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct BioimageioModelConfig {
    pub n_dim: usize,
    pub n_channel_in: usize,
    pub n_rays: usize,
    pub grid: Vec<usize>,
    pub axes_net: String,
    pub axes_out: String,
    pub axes_net_div_by: Vec<usize>,
    pub is_multiclass: bool,
    pub thresholds: BioimageioThresholds,
    pub config_json: serde_json::Value,
}

#[derive(Clone, Debug, PartialEq)]
pub struct BioimageioWeightsModelMetadata {
    pub weight_uri: PathBuf,
    pub test_inputs: Vec<PathBuf>,
    pub test_outputs: Vec<PathBuf>,
    pub config: serde_json::Value,
    pub tensorflow_version: Option<String>,
    pub input_names: Vec<String>,
    pub input_min_shape: Vec<Vec<usize>>,
    pub input_step: Vec<Vec<usize>>,
    pub input_axes: Vec<String>,
    pub input_data_range: Vec<[String; 2]>,
    pub preprocessing: Vec<Vec<BioimageioPreprocessing>>,
    pub output_names: Vec<String>,
    pub output_data_range: Vec<[String; 2]>,
    pub output_axes: Vec<String>,
    pub output_reference: Vec<String>,
    pub output_scale: Vec<Vec<f32>>,
    pub output_offset: Vec<Vec<f32>>,
    pub halo: Vec<Vec<usize>>,
    pub attachments: Vec<PathBuf>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct BioimageioPreprocessing {
    pub name: String,
    pub mode: String,
    pub axes: String,
    pub min_percentile: f32,
    pub max_percentile: f32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct BioimageioExport {
    pub name: String,
    pub outdir: PathBuf,
    pub zip_path: PathBuf,
    pub add_deepimagej_config: bool,
    pub metadata: StardistMetadata,
    pub model_metadata: BioimageioWeightsModelMetadata,
    pub overwrite_spec_kwargs: BTreeMap<String, serde_json::Value>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ImportedBioimageio {
    pub n_dim: usize,
    pub outpath: PathBuf,
    pub config_path: PathBuf,
    pub thresholds_path: PathBuf,
    pub weights_path: PathBuf,
    pub bioimageio_dir: PathBuf,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BioimageioImport {
    pub metadata: &'static str,
    pub build_model: &'static str,
    pub bioimageio_core: &'static str,
    pub xarray: &'static str,
}

pub fn _import(error: bool, available: bool) -> Result<Option<BioimageioImport>, BioimageioError> {
    if available {
        Ok(Some(BioimageioImport {
            metadata: "importlib_metadata::metadata",
            build_model: "bioimageio.core.build_spec::build_model",
            bioimageio_core: "bioimageio.core",
            xarray: "xarray",
        }))
    } else if error {
        Err(BioimageioError::MissingDependencies(
            BIOIMAGEIO_MISSING_DEPENDENCIES.to_string(),
        ))
    } else {
        Ok(None)
    }
}

pub fn _create_stardist_dependencies(
    outdir: impl AsRef<Path>,
    tf_version: &str,
    stardist_version: &str,
    reqs_conda: &[String],
) -> Result<String, BioimageioError> {
    let outdir = outdir.as_ref();
    fs::create_dir_all(outdir)?;
    let tf_major = tf_version
        .split(|c: char| !c.is_ascii_digit())
        .find(|part| !part.is_empty())
        .ok_or(BioimageioError::InvalidTensorflowVersion)?
        .parse::<u64>()
        .map_err(|_| BioimageioError::InvalidTensorflowVersion)?;
    let tf_minor = tf_version
        .split(|c: char| !c.is_ascii_digit())
        .filter(|part| !part.is_empty())
        .nth(1)
        .unwrap_or("0")
        .parse::<u64>()
        .map_err(|_| BioimageioError::InvalidTensorflowVersion)?;

    let path = outdir.join("environment.yaml");
    let mut file = File::create(&path)?;
    writeln!(file, "name: stardist")?;
    writeln!(file, "channels:")?;
    writeln!(file, "- defaults")?;
    writeln!(file, "- conda-forge")?;
    writeln!(file, "dependencies:")?;
    if tf_major == 1 {
        writeln!(file, "- python>=3.7,<3.8")?;
    } else {
        writeln!(file, "- python>=3.7")?;
    }
    for req in reqs_conda {
        writeln!(file, "- {req}")?;
    }
    writeln!(file, "- pip")?;
    writeln!(file, "- pip:")?;
    writeln!(file, "  - stardist>={stardist_version}")?;
    writeln!(
        file,
        "  - tensorflow>={}.{},<{}",
        tf_major,
        tf_minor,
        tf_major + 1
    )?;
    Ok(format!("conda:{}", path.display()))
}

pub fn _create_stardist_doc(outdir: impl AsRef<Path>) -> Result<PathBuf, BioimageioError> {
    let outdir = outdir.as_ref();
    fs::create_dir_all(outdir)?;
    let doc_path = outdir.join("README.md");
    let mut file = File::create(&doc_path)?;
    file.write_all(
        b"# StarDist Model\nThis is a model for object detection with star-convex shapes.\nPlease see the [StarDist repository](https://github.com/stardist/stardist) for details.",
    )?;
    Ok(doc_path)
}

pub fn _get_stardist_metadata(
    outdir: impl AsRef<Path>,
    n_dim: usize,
    package_summary: &str,
    package_author: &str,
    package_home_page: &str,
    package_license: &str,
    dependencies: Option<String>,
) -> Result<StardistMetadata, BioimageioError> {
    let outdir = outdir.as_ref();
    let doi_2d = "https://doi.org/10.1007/978-3-030-00934-2_30";
    let doi_3d = "https://doi.org/10.1109/WACV45572.2020.9093435";
    let mut authors = Vec::<BioimageioAuthor>::new();
    for name in package_author.split(',') {
        let name = name.trim();
        let github_user = match name {
            "Martin Weigert" => Some("maweigert".to_string()),
            "Uwe Schmidt" => Some("uschmidt83".to_string()),
            _ => None,
        };
        authors.push(BioimageioAuthor {
            name: name.to_string(),
            github_user,
        });
    }

    Ok(StardistMetadata {
        description: package_summary.to_string(),
        authors,
        git_repo: package_home_page.to_string(),
        license: package_license.to_string(),
        cite: vec![
            BioimageioCitation {
                text: "Cell Detection with Star-Convex Polygons".to_string(),
                doi: doi_2d.to_string(),
            },
            BioimageioCitation {
                text:
                    "Star-convex Polyhedra for 3D Object Detection and Segmentation in Microscopy"
                        .to_string(),
                doi: doi_3d.to_string(),
            },
        ],
        tags: vec![
            "fluorescence-light-microscopy".to_string(),
            "whole-slide-imaging".to_string(),
            "other".to_string(),
            format!("{n_dim}d"),
            "cells".to_string(),
            "nuclei".to_string(),
            "tensorflow".to_string(),
            "fiji".to_string(),
            "unet".to_string(),
            "instance-segmentation".to_string(),
            "object-detection".to_string(),
            "stardist".to_string(),
        ],
        covers: vec![
            "https://raw.githubusercontent.com/stardist/stardist/main/images/stardist_logo.jpg"
                .to_string(),
        ],
        documentation: _create_stardist_doc(outdir)?,
        dependencies,
    })
}

pub fn _predict_tf(
    _model_path: impl AsRef<Path>,
    _test_input: &[f32],
    _test_input_shape: &[usize],
) -> Result<Vec<f32>, BioimageioError> {
    Err(BioimageioError::TensorflowRuntimeUnavailable)
}

pub fn _get_weights_and_model_metadata(
    outdir: impl AsRef<Path>,
    model: &BioimageioModelConfig,
    test_input: &[f32],
    test_input_shape: &[usize],
    test_input_axes: Option<&str>,
    test_input_norm_axes: &str,
    mode: BioimageioMode,
    min_percentile: f32,
    max_percentile: f32,
    tensorflow_version: Option<&str>,
) -> Result<BioimageioWeightsModelMetadata, BioimageioError> {
    if !(0.0 <= min_percentile && min_percentile < max_percentile && max_percentile <= 100.0) {
        return Err(BioimageioError::InvalidPercentiles);
    }
    let outdir = outdir.as_ref();
    fs::create_dir_all(outdir)?;
    match &mode {
        BioimageioMode::KerasHdf5 => return Err(BioimageioError::KerasExportUnsupported),
        BioimageioMode::TensorflowSavedModelBundle => {
            if model.is_multiclass {
                return Err(BioimageioError::TensorflowSavedModelMulticlassUnsupported);
            }
        }
        BioimageioMode::Other(other) => {
            return Err(BioimageioError::UnsupportedMode(other.clone()));
        }
    }

    let ndim_tensor = model.axes_out.len() + 1;
    let net_axes_in = model.axes_net.to_ascii_lowercase();
    let net_axes_out = model.axes_out.to_ascii_lowercase();
    let input_axes = format!("b{net_axes_in}");
    let output_axes = format!("b{net_axes_out}");
    let mut input_min_shape = model.axes_net_div_by.clone();
    if let Some(c_axis) = model.axes_net.find('C') {
        if c_axis < input_min_shape.len() {
            input_min_shape[c_axis] = model.n_channel_in;
        }
    }
    let mut input_step = model.axes_net_div_by.clone();
    if let Some(c_axis) = model.axes_net.find('C') {
        if c_axis < input_step.len() {
            input_step[c_axis] = 0;
        }
    }
    input_min_shape.insert(0, 1);
    input_step.insert(0, 0);

    let mut output_scale = vec![1.0f32; ndim_tensor];
    if let Some(c_axis) = output_axes.find('c') {
        output_scale[c_axis] = 0.0;
    }
    let output_n_channels = 1 + model.n_rays;
    let mut output_offset = vec![0.0f32; ndim_tensor];
    if let Some(c_axis) = output_axes.find('c') {
        output_offset[c_axis] = output_n_channels as f32 / 2.0;
    }
    let mut halo = vec![0usize; ndim_tensor];
    for i in 1..ndim_tensor.min(model.axes_net_div_by.len() + 1) {
        halo[i] = ((model.axes_net_div_by[i - 1] + 7) / 8) * 8;
    }
    for (value, ha) in input_min_shape.iter_mut().zip(halo.iter()) {
        *value += 2 * *ha;
    }
    for i in 1..input_min_shape.len().min(model.axes_net_div_by.len() + 1) {
        let div_by = model.axes_net_div_by[i - 1].max(1);
        let rem = input_min_shape[i] % div_by;
        if rem != 0 {
            input_min_shape[i] += div_by - rem;
        }
    }

    let axes_img = test_input_axes.unwrap_or(&model.axes_net);
    let axes_norm = test_input_norm_axes
        .chars()
        .filter(|axis| *axis != 'S' && model.axes_net.contains(*axis))
        .collect::<String>()
        .to_ascii_lowercase();
    let in_path = outdir.join("test_input.json");
    let mut input_file = File::create(&in_path)?;
    serde_json::to_writer_pretty(
        &mut input_file,
        &serde_json::json!({
            "axes": axes_img,
            "shape": test_input_shape,
            "data": test_input,
        }),
    )
    .map_err(std::io::Error::other)?;

    let assets_uri = outdir.join("TF_SavedModel.zip");
    let out_path = outdir.join("test_output.json");
    let mut output_file = File::create(&out_path)?;
    serde_json::to_writer_pretty(
        &mut output_file,
        &serde_json::json!({
            "runtime": "tensorflow_saved_model_bundle",
            "status": "not executed by stardist-rs",
        }),
    )
    .map_err(std::io::Error::other)?;

    let weights_file = outdir.join("stardist_weights.h5");
    File::create(&weights_file)?;
    let mut attachments = vec![weights_file.clone()];
    let mut stardist_config = serde_json::json!({
        "stardist": {
            "python_version": serde_json::Value::Null,
            "thresholds": {
                "prob": model.thresholds.prob,
                "nms": model.thresholds.nms,
            },
            "weights": "stardist_weights.h5",
            "config": model.config_json,
        }
    });
    if model.n_dim == 2 {
        let macro_file = outdir.join("stardist_postprocessing.ijm");
        let macro_text = DEEPIMAGEJ_MACRO
            .replace("{probThresh}", &model.thresholds.prob.to_string())
            .replace("{nmsThresh}", &model.thresholds.nms.to_string());
        fs::write(&macro_file, macro_text)?;
        attachments.push(macro_file.clone());
        stardist_config["stardist"]["postprocessing_macro"] =
            serde_json::Value::String("stardist_postprocessing.ijm".to_string());
    }

    Ok(BioimageioWeightsModelMetadata {
        weight_uri: assets_uri,
        test_inputs: vec![in_path],
        test_outputs: vec![out_path],
        config: stardist_config,
        tensorflow_version: tensorflow_version.map(str::to_string),
        input_names: vec!["input".to_string()],
        input_min_shape: vec![input_min_shape],
        input_step: vec![input_step],
        input_axes: vec![input_axes],
        input_data_range: vec![["-inf".to_string(), "inf".to_string()]],
        preprocessing: vec![vec![BioimageioPreprocessing {
            name: "scale_range".to_string(),
            mode: "per_sample".to_string(),
            axes: axes_norm,
            min_percentile,
            max_percentile,
        }]],
        output_names: vec!["output".to_string()],
        output_data_range: vec![["-inf".to_string(), "inf".to_string()]],
        output_axes: vec![output_axes],
        output_reference: vec!["input".to_string()],
        output_scale: vec![output_scale],
        output_offset: vec![output_offset],
        halo: vec![halo],
        attachments,
    })
}

pub fn export_bioimageio(
    model: &BioimageioModelConfig,
    outpath: impl AsRef<Path>,
    test_input: &[f32],
    test_input_shape: &[usize],
    name: Option<&str>,
    mode: BioimageioMode,
    min_percentile: f32,
    max_percentile: f32,
    overwrite_spec_kwargs: Option<BTreeMap<String, serde_json::Value>>,
    generate_default_deps: bool,
) -> Result<BioimageioExport, BioimageioError> {
    if !(0.0 <= min_percentile && min_percentile < max_percentile && max_percentile <= 100.0) {
        return Err(BioimageioError::InvalidPercentiles);
    }
    let name = name.unwrap_or("stardist").to_string();
    let outpath = outpath.as_ref();
    let (outdir, zip_path) = if outpath.extension().is_none() {
        (outpath.to_path_buf(), outpath.join(format!("{name}.zip")))
    } else if outpath.extension().and_then(|s| s.to_str()) == Some("zip") {
        (
            outpath
                .parent()
                .map(Path::to_path_buf)
                .unwrap_or_else(|| PathBuf::from(".")),
            outpath.to_path_buf(),
        )
    } else {
        return Err(BioimageioError::InvalidOutpath);
    };
    fs::create_dir_all(&outdir)?;
    let dependencies = if generate_default_deps {
        Some(_create_stardist_dependencies(
            &outdir,
            "2.0.0",
            crate::STARDIST_VERSION,
            &[],
        )?)
    } else {
        None
    };
    let metadata = _get_stardist_metadata(
        &outdir,
        model.n_dim,
        "StarDist Model",
        "Martin Weigert, Uwe Schmidt",
        "https://github.com/stardist/stardist",
        "BSD-3-Clause",
        dependencies,
    )?;
    let model_metadata = _get_weights_and_model_metadata(
        &outdir,
        model,
        test_input,
        test_input_shape,
        None,
        "ZYX",
        mode,
        min_percentile,
        max_percentile,
        Some("2.0.0"),
    )?;

    Ok(BioimageioExport {
        name,
        outdir,
        zip_path,
        add_deepimagej_config: model.n_dim == 2,
        metadata,
        model_metadata,
        overwrite_spec_kwargs: overwrite_spec_kwargs.unwrap_or_default(),
    })
}

pub fn import_bioimageio(
    source: impl AsRef<Path>,
    outpath: impl AsRef<Path>,
) -> Result<ImportedBioimageio, BioimageioError> {
    let source = source.as_ref();
    let outpath = outpath.as_ref();
    if outpath.exists() {
        return Err(BioimageioError::OutputPathExists);
    }
    fs::create_dir_all(outpath)?;
    let bioimageio_dir = outpath.join("bioimageio");
    fs::create_dir_all(&bioimageio_dir)?;

    if source.is_dir() {
        copy_dir_recursive(source, &bioimageio_dir)?;
    } else {
        let file = File::open(source)?;
        let mut archive = zip::ZipArchive::new(file)?;
        archive.extract(&bioimageio_dir)?;
    }

    let config_source = find_file_named(&bioimageio_dir, "config.json")
        .ok_or(BioimageioError::IncompatibleModel)?;
    let thresholds_source = find_file_named(&bioimageio_dir, "thresholds.json")
        .ok_or(BioimageioError::IncompatibleModel)?;
    let weights_source = find_file_named(&bioimageio_dir, "stardist_weights.h5")
        .or_else(|| find_file_named(&bioimageio_dir, "weights_bioimageio.h5"))
        .ok_or_else(|| BioimageioError::MissingWeights("stardist_weights.h5".to_string()))?;

    let config_path = outpath.join("config.json");
    let thresholds_path = outpath.join("thresholds.json");
    let weights_path = outpath.join("weights_bioimageio.h5");
    fs::copy(&config_source, &config_path)?;
    fs::copy(&thresholds_source, &thresholds_path)?;
    fs::copy(&weights_source, &weights_path)?;

    let mut config_text = String::new();
    File::open(&config_path)?.read_to_string(&mut config_text)?;
    let config: serde_json::Value =
        serde_json::from_str(&config_text).map_err(std::io::Error::other)?;
    let n_dim = config
        .get("n_dim")
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(2) as usize;

    Ok(ImportedBioimageio {
        n_dim,
        outpath: outpath.to_path_buf(),
        config_path,
        thresholds_path,
        weights_path,
        bioimageio_dir,
    })
}

fn copy_dir_recursive(source: &Path, destination: &Path) -> Result<(), BioimageioError> {
    fs::create_dir_all(destination)?;
    for entry in fs::read_dir(source)? {
        let entry = entry?;
        let source_path = entry.path();
        let destination_path = destination.join(entry.file_name());
        if source_path.is_dir() {
            copy_dir_recursive(&source_path, &destination_path)?;
        } else {
            fs::copy(&source_path, &destination_path)?;
        }
    }
    Ok(())
}

fn find_file_named(root: &Path, name: &str) -> Option<PathBuf> {
    let entries = fs::read_dir(root).ok()?;
    for entry in entries.flatten() {
        let path = entry.path();
        if path.file_name().and_then(|s| s.to_str()) == Some(name) {
            return Some(path);
        }
        if path.is_dir() {
            if let Some(found) = find_file_named(&path, name) {
                return Some(found);
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn import_reports_optional_bioimageio_dependency_status_like_python() {
        assert_eq!(_import(false, false).unwrap(), None);
        assert_eq!(
            _import(true, false).unwrap_err().to_string(),
            BIOIMAGEIO_MISSING_DEPENDENCIES
        );
        assert_eq!(
            _import(true, true).unwrap(),
            Some(BioimageioImport {
                metadata: "importlib_metadata::metadata",
                build_model: "bioimageio.core.build_spec::build_model",
                bioimageio_core: "bioimageio.core",
                xarray: "xarray",
            })
        );
    }

    #[test]
    fn create_stardist_doc_writes_expected_readme_text() {
        let dir =
            std::env::temp_dir().join(format!("stardist_rs_bioimageio_doc_{}", std::process::id()));
        let _ = std::fs::remove_file(dir.join("README.md"));
        let _ = std::fs::remove_dir(&dir);

        let path = _create_stardist_doc(&dir).unwrap();
        let text = std::fs::read_to_string(&path).unwrap();
        assert_eq!(
            text,
            "# StarDist Model\nThis is a model for object detection with star-convex shapes.\nPlease see the [StarDist repository](https://github.com/stardist/stardist) for details."
        );
        let _ = std::fs::remove_file(path);
        let _ = std::fs::remove_dir(dir);
    }

    #[test]
    fn create_stardist_dependencies_writes_environment_yaml() {
        let dir =
            std::env::temp_dir().join(format!("stardist_rs_bioimageio_env_{}", std::process::id()));
        let _ = std::fs::remove_file(dir.join("environment.yaml"));
        let _ = std::fs::remove_dir(&dir);

        let uri = _create_stardist_dependencies(
            &dir,
            "2.13.1",
            "0.9.2",
            &["bioimageio.core>=0.5".to_string()],
        )
        .unwrap();
        assert!(uri.starts_with("conda:"));
        let text = std::fs::read_to_string(dir.join("environment.yaml")).unwrap();
        assert!(text.contains("- python>=3.7\n"));
        assert!(text.contains("- bioimageio.core>=0.5\n"));
        assert!(text.contains("  - stardist>=0.9.2\n"));
        assert!(text.contains("  - tensorflow>=2.13,<3\n"));
        let _ = std::fs::remove_file(dir.join("environment.yaml"));
        let _ = std::fs::remove_dir(dir);
    }

    #[test]
    fn get_stardist_metadata_matches_static_bioimageio_fields() {
        let dir = std::env::temp_dir().join(format!(
            "stardist_rs_bioimageio_metadata_{}",
            std::process::id()
        ));
        let _ = std::fs::remove_file(dir.join("README.md"));
        let _ = std::fs::remove_dir(&dir);

        let metadata = _get_stardist_metadata(
            &dir,
            3,
            "summary",
            "Martin Weigert, Uwe Schmidt, Someone Else",
            "https://github.com/stardist/stardist",
            "BSD-3-Clause",
            Some("conda:/tmp/environment.yaml".to_string()),
        )
        .unwrap();
        assert_eq!(metadata.description, "summary");
        assert_eq!(
            metadata.authors[0].github_user,
            Some("maweigert".to_string())
        );
        assert_eq!(
            metadata.authors[1].github_user,
            Some("uschmidt83".to_string())
        );
        assert_eq!(metadata.authors[2].github_user, None);
        assert!(metadata.tags.contains(&"3d".to_string()));
        assert_eq!(
            metadata.dependencies,
            Some("conda:/tmp/environment.yaml".to_string())
        );
        assert!(metadata.documentation.ends_with("README.md"));
        let _ = std::fs::remove_file(dir.join("README.md"));
        let _ = std::fs::remove_dir(dir);
    }

    #[test]
    fn predict_tf_reports_runtime_boundary() {
        assert!(matches!(
            _predict_tf("model.zip", &[0.0], &[1, 1]),
            Err(BioimageioError::TensorflowRuntimeUnavailable)
        ));
    }

    #[test]
    fn weights_and_model_metadata_match_savedmodel_export_contract() {
        let dir = std::env::temp_dir().join(format!(
            "stardist_rs_bioimageio_weights_{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        let model = BioimageioModelConfig {
            n_dim: 2,
            n_channel_in: 1,
            n_rays: 32,
            grid: vec![2, 2],
            axes_net: "YXC".to_string(),
            axes_out: "YXC".to_string(),
            axes_net_div_by: vec![16, 16, 1],
            is_multiclass: false,
            thresholds: BioimageioThresholds {
                prob: 0.5,
                nms: 0.4,
            },
            config_json: serde_json::json!({"n_dim": 2, "n_rays": 32}),
        };
        let metadata = _get_weights_and_model_metadata(
            &dir,
            &model,
            &[0.0; 16],
            &[4, 4],
            Some("YX"),
            "YX",
            BioimageioMode::TensorflowSavedModelBundle,
            1.0,
            99.8,
            Some("2.13.1"),
        )
        .unwrap();
        assert_eq!(metadata.input_names, vec!["input"]);
        assert_eq!(metadata.output_names, vec!["output"]);
        assert_eq!(metadata.input_axes, vec!["byxc"]);
        assert_eq!(metadata.output_axes, vec!["byxc"]);
        assert_eq!(metadata.output_scale[0][3], 0.0);
        assert_eq!(metadata.output_offset[0][3], 16.5);
        assert!(metadata.attachments.iter().any(|p| {
            p.file_name().and_then(|s| s.to_str()) == Some("stardist_postprocessing.ijm")
        }));
        assert!(dir.join("test_input.json").exists());
        assert!(dir.join("test_output.json").exists());
        assert!(dir.join("stardist_weights.h5").exists());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn export_bioimageio_builds_library_export_plan() {
        let dir = std::env::temp_dir().join(format!(
            "stardist_rs_bioimageio_export_{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        let model = BioimageioModelConfig {
            n_dim: 3,
            n_channel_in: 1,
            n_rays: 96,
            grid: vec![1, 2, 2],
            axes_net: "ZYXC".to_string(),
            axes_out: "ZYXC".to_string(),
            axes_net_div_by: vec![4, 16, 16, 1],
            is_multiclass: false,
            thresholds: BioimageioThresholds {
                prob: 0.5,
                nms: 0.4,
            },
            config_json: serde_json::json!({"n_dim": 3, "n_rays": 96}),
        };
        let export = export_bioimageio(
            &model,
            &dir,
            &[0.0; 8],
            &[2, 2, 2],
            Some("demo"),
            BioimageioMode::TensorflowSavedModelBundle,
            1.0,
            99.8,
            None,
            false,
        )
        .unwrap();
        assert_eq!(export.name, "demo");
        assert_eq!(export.zip_path, dir.join("demo.zip"));
        assert!(!export.add_deepimagej_config);
        assert_eq!(export.metadata.tags.iter().any(|tag| tag == "3d"), true);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn import_bioimageio_copies_stardist_payload_from_directory() {
        let root = std::env::temp_dir().join(format!(
            "stardist_rs_bioimageio_import_{}",
            std::process::id()
        ));
        let source = root.join("source");
        let out = root.join("out");
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(source.join("attachments")).unwrap();
        std::fs::write(source.join("config.json"), r#"{"n_dim":3}"#).unwrap();
        std::fs::write(source.join("thresholds.json"), r#"{"prob":0.5,"nms":0.4}"#).unwrap();
        std::fs::write(
            source.join("attachments").join("stardist_weights.h5"),
            b"weights",
        )
        .unwrap();

        let imported = import_bioimageio(&source, &out).unwrap();
        assert_eq!(imported.n_dim, 3);
        assert!(imported.config_path.exists());
        assert!(imported.thresholds_path.exists());
        assert!(imported.weights_path.exists());
        assert!(imported.bioimageio_dir.join("attachments").exists());
        let _ = std::fs::remove_dir_all(&root);
    }
}
