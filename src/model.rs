use std::collections::BTreeMap;

use ndarray::{Array2, Array3};

use crate::{ClassAssignment, Config2D, Config3D};

#[derive(Clone, Debug, PartialEq)]
pub struct StarDist2D {
    pub config: Config2D,
    pub thresholds: StarDistThresholds,
}

#[derive(Clone, Debug, PartialEq)]
pub struct StarDist3D {
    pub config: Config3D,
    pub thresholds: StarDistThresholds,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ConfigClass {
    Config2D,
    Config3D,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StarDistThresholds {
    pub prob: f32,
    pub nms: f32,
}

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum ThresholdsError {
    #[error("prob threshold must be finite and between 0 and 1")]
    InvalidProb,
    #[error("nms threshold must be finite and between 0 and 1")]
    InvalidNms,
}

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum OptimizeThresholdsError {
    #[error("nms_threshs must not be empty")]
    EmptyNmsThresholds,
    #[error(transparent)]
    Thresholds(#[from] ThresholdsError),
    #[error(transparent)]
    Utils(#[from] crate::UtilsError),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ClassesArg {
    Auto,
    String(String),
    Scalar(i32),
    List(Vec<ClassAssignment>),
}

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum ClassesArgError {
    #[error("only 'auto' is supported as string argument for classes")]
    UnsupportedString,
    #[error("using classes = 'auto' for n_classes > 1 is not supported")]
    AutoMulticlassUnsupported,
    #[error("len(classes) does not match training data length")]
    WrongLength,
}

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum AxesDivByError {
    #[error("unsupported backbone")]
    UnsupportedBackbone,
    #[error("axes contain duplicate entries")]
    DuplicateAxis,
    #[error("query axes are empty")]
    EmptyAxes,
}

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum AxesTileOverlapError {
    #[error("unsupported backbone")]
    UnsupportedBackbone,
    #[error("axes contain duplicate entries")]
    DuplicateAxis,
    #[error("query axes are empty")]
    EmptyAxes,
    #[error("tile overlap value is unavailable for this architecture")]
    Unavailable,
}

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum AxesError {
    #[error("axes contain duplicate entries")]
    DuplicateAxis,
    #[error("axes length does not match image dimensions")]
    DimensionMismatch,
    #[error("config axes must contain channel axis")]
    MissingConfigChannelAxis,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PadMode {
    Reflect,
    Edge,
    Constant,
}

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum ResizerError {
    #[error("axes contain duplicate entries")]
    DuplicateAxis,
    #[error("axes length does not match image dimensions")]
    DimensionMismatch,
    #[error("input length does not match shape")]
    ShapeMismatch,
    #[error("axes_div_by must match image dimensions")]
    AxesDivByMismatch,
    #[error("grid must divide axes_div_by")]
    GridDivisibility,
    #[error("resizer must run before before after/filter_points")]
    MissingBeforeState,
    #[error("network output shape is inconsistent with padded input shape and grid")]
    InconsistentNetworkShape,
    #[error("point dimensionality does not match spatial axes")]
    PointDimensionMismatch,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MaskedPenalty {
    Abs,
    Square,
}

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum LossError {
    #[error("channels must be positive")]
    InvalidChannels,
    #[error("tensor length must match and be divisible by channels")]
    ShapeMismatch,
    #[error("mask length must be 1, channels, or tensor length")]
    MaskShapeMismatch,
    #[error("weights length must be 1, channels, or tensor length")]
    WeightsShapeMismatch,
}

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum StarDistDataError {
    #[error("foreground_prob must be between 0 and 1")]
    InvalidForegroundProb,
    #[error("patch_size, grid, and maxfilter_patch_size must have the same dimensionality")]
    WrongPatchDimension,
    #[error("label image length does not match shape")]
    ShapeMismatch,
    #[error("image shape must include the configured channel axis")]
    ChannelShapeMismatch,
    #[error("classes must be provided for each source image when n_classes is set")]
    ClassesShapeMismatch,
    #[error(transparent)]
    SamplePatches(#[from] crate::sample_patches::SamplePatchesError),
    #[error(transparent)]
    Geometry(#[from] crate::geometry::GeometryError),
    #[error(transparent)]
    Utils(#[from] crate::UtilsError),
}

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum StarDistTrainError {
    #[error("X and Y can't be empty and must have same length")]
    EmptyOrMismatchedData,
    #[error("validation_data must have length 2, or length 3 when n_classes is set")]
    InvalidValidationData,
    #[error("train patch size is not divisible by the required model axes divisibility")]
    PatchSizeNotDivisible,
    #[error("epochs * steps_per_epoch overflowed")]
    LengthOverflow,
    #[error("unsupported configured distance loss {0}")]
    UnsupportedDistanceLoss(String),
    #[error("train_loss_weights length is incompatible with n_classes")]
    InvalidLossWeights,
    #[error("train_class_weights length is incompatible with n_classes")]
    InvalidClassWeights,
    #[error(transparent)]
    ClassesArg(#[from] ClassesArgError),
    #[error(transparent)]
    AxesDivBy(#[from] AxesDivByError),
    #[error(transparent)]
    Data(#[from] StarDistDataError),
    #[error(transparent)]
    Rays(#[from] crate::RaysError),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StarDistTrainDistLoss {
    Mae,
    Mse,
    Iou,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StarDistTrainCallback {
    Checkpoint,
    TensorBoard,
    ReduceLrOnPlateau,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StarDistCheckpointCallback {
    pub filepath: String,
    pub save_best_only: bool,
    pub save_weights_only: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum StarDistTrainingFinishedAction {
    SaveLastWeights { filepath: String },
    LoadBestWeights { prefer: String },
    RemoveEpochWeights { filepath: String },
}

#[derive(Clone, Debug, PartialEq)]
pub struct StarDistPreparedTraining {
    pub optimizer: String,
    pub learning_rate: f32,
    pub dist_loss: StarDistTrainDistLoss,
    pub losses: Vec<String>,
    pub loss_weights: Vec<f32>,
    pub metrics: Vec<String>,
    pub callbacks: Vec<StarDistTrainCallback>,
    pub checkpoint_callbacks: Vec<StarDistCheckpointCallback>,
    pub tensorboard_log_dir: Option<String>,
    pub training_finished: Vec<StarDistTrainingFinishedAction>,
    pub model_prepared: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub struct StarDistDataBase {
    pub n_channel: Option<usize>,
    pub patch_size: Vec<usize>,
    pub grid: Vec<usize>,
    pub foreground_prob: f32,
    pub maxfilter_patch_size: Vec<usize>,
    pub sample_ind_cache: bool,
    pub ind_cache_fg: BTreeMap<usize, Vec<Vec<usize>>>,
    pub ind_cache_all: BTreeMap<usize, Vec<Vec<usize>>>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct StarDistData2D {
    pub base: StarDistDataBase,
    pub n_rays: usize,
    pub n_classes: Option<usize>,
    pub classes: Option<Vec<ClassAssignment>>,
    pub shape_completion: bool,
    pub b: usize,
}

#[derive(Clone, Debug, PartialEq)]
pub struct StarDistData2DBatch {
    pub x: Vec<f32>,
    pub x_shape: [usize; 4],
    pub prob: Vec<f32>,
    pub prob_shape: [usize; 4],
    pub dist: Vec<f32>,
    pub dist_shape: [usize; 4],
    pub prob_class: Option<Vec<f32>>,
    pub prob_class_shape: Option<[usize; 4]>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct StarDistData3D {
    pub base: StarDistDataBase,
    pub rays: crate::Rays,
    pub anisotropy: Option<[f32; 3]>,
    pub n_classes: Option<usize>,
    pub classes: Option<Vec<ClassAssignment>>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct StarDistData3DBatch {
    pub x: Vec<f32>,
    pub x_shape: [usize; 5],
    pub prob: Vec<f32>,
    pub prob_shape: [usize; 5],
    pub dist: Vec<f32>,
    pub dist_shape: [usize; 5],
    pub prob_class: Option<Vec<f32>>,
    pub prob_class_shape: Option<[usize; 5]>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct StarDist2DTrainSetup {
    pub epochs: usize,
    pub steps_per_epoch: usize,
    pub train_length: usize,
    pub validation_n_take: usize,
    pub classes: Option<Vec<ClassAssignment>>,
    pub validation_classes: Option<Vec<ClassAssignment>>,
    pub prepared_training: StarDistPreparedTraining,
    pub data_train: StarDistData2D,
    pub data_val: StarDistData2D,
}

#[derive(Clone, Debug, PartialEq)]
pub struct StarDist3DTrainSetup {
    pub epochs: usize,
    pub steps_per_epoch: usize,
    pub train_length: usize,
    pub validation_n_take: usize,
    pub classes: Option<Vec<ClassAssignment>>,
    pub validation_classes: Option<Vec<ClassAssignment>>,
    pub prepared_training: StarDistPreparedTraining,
    pub data_train: StarDistData3D,
    pub data_val: StarDistData3D,
}

#[derive(Clone, Debug, PartialEq)]
pub struct StarDistPadAndCropResizer {
    pub grid: Vec<(char, usize)>,
    pub mode: PadMode,
    pub constant_value: f32,
    pub pad: Vec<(char, [usize; 2])>,
    pub padded_shape: Vec<(char, usize)>,
}

pub fn _is_multiclass(n_classes: Option<usize>) -> bool {
    n_classes.is_some()
}

pub fn _parse_classes_arg(
    n_classes: Option<usize>,
    classes: ClassesArg,
    length: usize,
) -> Result<Option<Vec<ClassAssignment>>, ClassesArgError> {
    match classes {
        ClassesArg::Auto => {
            if n_classes.is_none() {
                Ok(None)
            } else if n_classes == Some(1) {
                Ok(Some(vec![ClassAssignment::Single(Some(1)); length]))
            } else {
                Err(ClassesArgError::AutoMulticlassUnsupported)
            }
        }
        ClassesArg::String(classes) => {
            if classes != "auto" {
                return Err(ClassesArgError::UnsupportedString);
            }
            if n_classes.is_none() {
                Ok(None)
            } else if n_classes == Some(1) {
                Ok(Some(vec![ClassAssignment::Single(Some(1)); length]))
            } else {
                Err(ClassesArgError::AutoMulticlassUnsupported)
            }
        }
        ClassesArg::Scalar(class_id) => {
            Ok(Some(vec![ClassAssignment::Single(Some(class_id)); length]))
        }
        ClassesArg::List(classes) => {
            if classes.len() != length {
                return Err(ClassesArgError::WrongLength);
            }
            Ok(Some(classes))
        }
    }
}

pub fn _tf_version_at_least(tf_version: &str, version_string: &str) -> bool {
    let tf_parts = tf_version
        .split(|c: char| !c.is_ascii_digit())
        .filter(|part| !part.is_empty())
        .map(|part| part.parse::<u64>().unwrap_or(0))
        .collect::<Vec<_>>();
    let version_parts = version_string
        .split(|c: char| !c.is_ascii_digit())
        .filter(|part| !part.is_empty())
        .map(|part| part.parse::<u64>().unwrap_or(0))
        .collect::<Vec<_>>();
    let n = tf_parts.len().max(version_parts.len());
    for i in 0..n {
        let tf = *tf_parts.get(i).unwrap_or(&0);
        let requested = *version_parts.get(i).unwrap_or(&0);
        if tf > requested {
            return true;
        } else if tf < requested {
            return false;
        }
    }
    true
}

impl Default for StarDistThresholds {
    fn default() -> Self {
        Self {
            prob: 0.5,
            nms: 0.4,
        }
    }
}

impl StarDistThresholds {
    pub fn new(prob: Option<f32>, nms: Option<f32>) -> Self {
        Self {
            prob: prob
                .filter(|v| v.is_finite() && *v > 0.0 && *v < 1.0)
                .unwrap_or(0.5),
            nms: nms
                .filter(|v| v.is_finite() && *v > 0.0 && *v < 1.0)
                .unwrap_or(0.4),
        }
    }
}

pub fn generic_masked_loss<F, R>(
    mask: &[f32],
    y_true: &[f32],
    y_pred: &[f32],
    channels: usize,
    weights: &[f32],
    norm_by_mask: bool,
    reg_weight: f32,
    mut loss: F,
    mut reg_penalty: R,
) -> Result<Vec<f32>, LossError>
where
    F: FnMut(f32, f32) -> f32,
    R: FnMut(f32) -> f32,
{
    if channels == 0 {
        return Err(LossError::InvalidChannels);
    }
    if y_true.len() != y_pred.len() || y_true.len() % channels != 0 {
        return Err(LossError::ShapeMismatch);
    }
    if !(mask.len() == 1 || mask.len() == channels || mask.len() == y_true.len()) {
        return Err(LossError::MaskShapeMismatch);
    }
    if !(weights.len() == 1 || weights.len() == channels || weights.len() == y_true.len()) {
        return Err(LossError::WeightsShapeMismatch);
    }

    let eps = f32::EPSILON;
    let mut mask_sum = 0.0f32;
    for i in 0..y_true.len() {
        let mask_value = if mask.len() == 1 {
            mask[0]
        } else if mask.len() == channels {
            mask[i % channels]
        } else {
            mask[i]
        };
        mask_sum += mask_value;
    }
    let norm_mask = if norm_by_mask {
        mask_sum / y_true.len() as f32 + eps
    } else {
        1.0
    };

    let n_rows = y_true.len() / channels;
    let mut out = Vec::<f32>::with_capacity(n_rows);
    for row in 0..n_rows {
        let mut actual_loss = 0.0f32;
        let mut reg_loss = 0.0f32;
        for c in 0..channels {
            let i = row * channels + c;
            let mask_value = if mask.len() == 1 {
                mask[0]
            } else if mask.len() == channels {
                mask[c]
            } else {
                mask[i]
            };
            let weight_value = if weights.len() == 1 {
                weights[0]
            } else if weights.len() == channels {
                weights[c]
            } else {
                weights[i]
            };
            actual_loss += mask_value * weight_value * loss(y_true[i], y_pred[i]);
            if reg_weight > 0.0 {
                reg_loss += (1.0 - mask_value) * reg_penalty(y_pred[i]);
            }
        }
        actual_loss /= channels as f32;
        reg_loss /= channels as f32;
        if reg_weight > 0.0 {
            out.push(actual_loss / norm_mask + reg_weight * reg_loss);
        } else {
            out.push(actual_loss / norm_mask);
        }
    }
    Ok(out)
}

pub fn masked_loss(
    mask: &[f32],
    y_true: &[f32],
    y_pred: &[f32],
    channels: usize,
    penalty: MaskedPenalty,
    reg_weight: f32,
    norm_by_mask: bool,
) -> Result<Vec<f32>, LossError> {
    match penalty {
        MaskedPenalty::Abs => generic_masked_loss(
            mask,
            y_true,
            y_pred,
            channels,
            &[1.0],
            norm_by_mask,
            reg_weight,
            |yt, yp| (yt - yp).abs(),
            |yp| yp.abs(),
        ),
        MaskedPenalty::Square => generic_masked_loss(
            mask,
            y_true,
            y_pred,
            channels,
            &[1.0],
            norm_by_mask,
            reg_weight,
            |yt, yp| {
                let diff = yt - yp;
                diff * diff
            },
            |yp| yp.abs(),
        ),
    }
}

pub fn masked_loss_mae(
    mask: &[f32],
    y_true: &[f32],
    y_pred: &[f32],
    channels: usize,
    reg_weight: f32,
    norm_by_mask: bool,
) -> Result<Vec<f32>, LossError> {
    masked_loss(
        mask,
        y_true,
        y_pred,
        channels,
        MaskedPenalty::Abs,
        reg_weight,
        norm_by_mask,
    )
}

pub fn masked_loss_mse(
    mask: &[f32],
    y_true: &[f32],
    y_pred: &[f32],
    channels: usize,
    reg_weight: f32,
    norm_by_mask: bool,
) -> Result<Vec<f32>, LossError> {
    masked_loss(
        mask,
        y_true,
        y_pred,
        channels,
        MaskedPenalty::Square,
        reg_weight,
        norm_by_mask,
    )
}

pub fn masked_metric_mae(
    mask: &[f32],
    y_true: &[f32],
    y_pred: &[f32],
    channels: usize,
) -> Result<Vec<f32>, LossError> {
    masked_loss(
        mask,
        y_true,
        y_pred,
        channels,
        MaskedPenalty::Abs,
        0.0,
        true,
    )
}

pub fn masked_metric_mse(
    mask: &[f32],
    y_true: &[f32],
    y_pred: &[f32],
    channels: usize,
) -> Result<Vec<f32>, LossError> {
    masked_loss(
        mask,
        y_true,
        y_pred,
        channels,
        MaskedPenalty::Square,
        0.0,
        true,
    )
}

pub fn kld(y_true: &[f32], y_pred: &[f32]) -> Result<f32, LossError> {
    if y_true.len() != y_pred.len() {
        return Err(LossError::ShapeMismatch);
    }
    let eps = f32::EPSILON;
    let mut sum = 0.0f32;
    let mut count = 0usize;
    for i in 0..y_true.len() {
        if y_true[i] >= 0.0 {
            let yt = y_true[i].clamp(eps, 1.0);
            let yp = y_pred[i].clamp(eps, 1.0);
            let bce_pred_left = if yt == 0.0 { 0.0 } else { yt * yp.ln() };
            let bce_pred_right = if yt == 1.0 {
                0.0
            } else {
                (1.0 - yt) * (1.0 - yp).ln()
            };
            let bce_true_left = if yt == 0.0 { 0.0 } else { yt * yt.ln() };
            let bce_true_right = if yt == 1.0 {
                0.0
            } else {
                (1.0 - yt) * (1.0 - yt).ln()
            };
            let bce_pred = -(bce_pred_left + bce_pred_right);
            let bce_true = -(bce_true_left + bce_true_right);
            sum += bce_pred - bce_true;
            count += 1;
        }
    }
    if count == 0 {
        Ok(0.0)
    } else {
        Ok(sum / count as f32)
    }
}

pub fn masked_loss_iou(
    mask: &[f32],
    y_true: &[f32],
    y_pred: &[f32],
    channels: usize,
    reg_weight: f32,
    norm_by_mask: bool,
) -> Result<Vec<f32>, LossError> {
    if channels == 0 {
        return Err(LossError::InvalidChannels);
    }
    if y_true.len() != y_pred.len() || y_true.len() % channels != 0 {
        return Err(LossError::ShapeMismatch);
    }
    let mut per_channel_loss = Vec::<f32>::with_capacity(y_true.len());
    let eps = f32::EPSILON;
    for row in 0..(y_true.len() / channels) {
        let mut inter = 0.0f32;
        let mut union = 0.0f32;
        for c in 0..channels {
            let i = row * channels + c;
            inter += y_pred[i].signum() * y_true[i].min(y_pred[i]).powi(2);
            union += y_true[i].max(y_pred[i]).powi(2);
        }
        inter /= channels as f32;
        union /= channels as f32;
        let loss = 1.0 - inter / (union + eps);
        for _ in 0..channels {
            per_channel_loss.push(loss);
        }
    }
    generic_masked_loss(
        mask,
        &per_channel_loss,
        &vec![0.0; per_channel_loss.len()],
        channels,
        &[1.0],
        norm_by_mask,
        reg_weight,
        |yt, _yp| yt,
        |yp| yp.abs(),
    )
}

pub fn masked_metric_iou(
    mask: &[f32],
    y_true: &[f32],
    y_pred: &[f32],
    channels: usize,
    reg_weight: f32,
    norm_by_mask: bool,
) -> Result<Vec<f32>, LossError> {
    if channels == 0 {
        return Err(LossError::InvalidChannels);
    }
    if y_true.len() != y_pred.len() || y_true.len() % channels != 0 {
        return Err(LossError::ShapeMismatch);
    }
    let mut per_channel_iou = Vec::<f32>::with_capacity(y_true.len());
    let eps = f32::EPSILON;
    for row in 0..(y_true.len() / channels) {
        let mut inter = 0.0f32;
        let mut union = 0.0f32;
        for c in 0..channels {
            let i = row * channels + c;
            let yp = y_pred[i].max(0.0);
            inter += y_true[i].min(yp).powi(2);
            union += y_true[i].max(yp).powi(2);
        }
        inter /= channels as f32;
        union /= channels as f32;
        let iou = inter / (union + eps);
        for _ in 0..channels {
            per_channel_iou.push(iou);
        }
    }
    generic_masked_loss(
        mask,
        &per_channel_iou,
        &vec![0.0; per_channel_iou.len()],
        channels,
        &[1.0],
        norm_by_mask,
        reg_weight,
        |yt, _yp| yt,
        |yp| yp.abs(),
    )
}

pub fn weighted_categorical_crossentropy(
    weights: &[f32],
    ndim: usize,
    y_true: &[f32],
    y_pred: &[f32],
    channels: usize,
) -> Result<Vec<f32>, LossError> {
    if channels == 0 || !(ndim == 2 || ndim == 3) {
        return Err(LossError::InvalidChannels);
    }
    if weights.len() != channels || y_true.len() != y_pred.len() || y_true.len() % channels != 0 {
        return Err(LossError::ShapeMismatch);
    }
    let eps = f32::EPSILON;
    let n_rows = y_true.len() / channels;
    let mut out = Vec::<f32>::with_capacity(n_rows);
    for row in 0..n_rows {
        let mut pred_sum = 0.0f32;
        for c in 0..channels {
            pred_sum += y_pred[row * channels + c] + eps;
        }
        let mut loss = 0.0f32;
        for c in 0..channels {
            let i = row * channels + c;
            let mask = if y_true[i] >= 0.0 { 1.0 } else { 0.0 };
            let pred = (y_pred[i] / pred_sum).clamp(eps, 1.0 - eps);
            loss -= weights[c] * mask * y_true[i] * pred.ln();
        }
        out.push(loss);
    }
    Ok(out)
}

impl StarDistDataBase {
    pub fn new(
        n_channel: Option<usize>,
        patch_size: Vec<usize>,
        grid: Vec<usize>,
        foreground_prob: f32,
        maxfilter_patch_size: Option<Vec<usize>>,
        sample_ind_cache: bool,
    ) -> Result<Self, StarDistDataError> {
        if !(0.0..=1.0).contains(&foreground_prob) {
            return Err(StarDistDataError::InvalidForegroundProb);
        }
        let maxfilter_patch_size = maxfilter_patch_size.unwrap_or_else(|| patch_size.clone());
        if patch_size.len() != grid.len() || maxfilter_patch_size.len() != patch_size.len() {
            return Err(StarDistDataError::WrongPatchDimension);
        }
        Ok(Self {
            n_channel,
            patch_size,
            grid,
            foreground_prob,
            maxfilter_patch_size,
            sample_ind_cache,
            ind_cache_fg: BTreeMap::new(),
            ind_cache_all: BTreeMap::new(),
        })
    }

    pub fn get_valid_inds(
        &mut self,
        k: usize,
        y: &[i32],
        y_shape: &[usize],
        foreground_prob: Option<f32>,
        random_value: f32,
    ) -> Result<Vec<Vec<usize>>, StarDistDataError> {
        if y.len() != y_shape.iter().product::<usize>() {
            return Err(StarDistDataError::ShapeMismatch);
        }
        if y_shape.len() != self.patch_size.len()
            || self.maxfilter_patch_size.len() != self.patch_size.len()
        {
            return Err(StarDistDataError::WrongPatchDimension);
        }
        let foreground_prob = foreground_prob.unwrap_or(self.foreground_prob);
        if !(0.0..=1.0).contains(&foreground_prob) {
            return Err(StarDistDataError::InvalidForegroundProb);
        }
        let foreground_only = random_value < foreground_prob;
        if foreground_only {
            if let Some(inds) = self.ind_cache_fg.get(&k) {
                return Ok(inds.clone());
            }
        } else if let Some(inds) = self.ind_cache_all.get(&k) {
            return Ok(inds.clone());
        }

        let inds = if foreground_only {
            let ndim = y_shape.len();
            let mut filter = vec![false; y.len()];
            let mut coord = vec![0usize; ndim];
            let mut source_coord = vec![0usize; ndim];
            for (idx, filter_value) in filter.iter_mut().enumerate() {
                let mut rem = idx;
                for dim in (0..ndim).rev() {
                    coord[dim] = rem % y_shape[dim];
                    rem /= y_shape[dim];
                }
                let mut has_foreground = false;
                let mut window_len = 1usize;
                for dim in 0..ndim {
                    window_len *= self.maxfilter_patch_size[dim];
                }
                for window_idx in 0..window_len {
                    let mut window_rem = window_idx;
                    let mut inside = true;
                    for dim in (0..ndim).rev() {
                        let offset = window_rem % self.maxfilter_patch_size[dim];
                        window_rem /= self.maxfilter_patch_size[dim];
                        let center_offset = self.maxfilter_patch_size[dim] / 2;
                        let source = coord[dim] as isize + offset as isize - center_offset as isize;
                        if source < 0 || source >= y_shape[dim] as isize {
                            inside = false;
                            break;
                        }
                        source_coord[dim] = source as usize;
                    }
                    if !inside {
                        continue;
                    }
                    let mut source_index = 0usize;
                    for dim in 0..ndim {
                        source_index = source_index * y_shape[dim] + source_coord[dim];
                    }
                    if y[source_index] > 0 {
                        has_foreground = true;
                        break;
                    }
                }
                *filter_value = has_foreground;
            }
            crate::sample_patches::get_valid_inds(y_shape, &self.patch_size, Some(&filter))?
        } else {
            crate::sample_patches::get_valid_inds(y_shape, &self.patch_size, None)?
        };

        if foreground_only && inds.first().map(|v| v.is_empty()).unwrap_or(true) {
            return self.get_valid_inds(k, y, y_shape, Some(0.0), random_value);
        }
        if self.sample_ind_cache {
            if foreground_only {
                self.ind_cache_fg.insert(k, inds.clone());
            } else {
                self.ind_cache_all.insert(k, inds.clone());
            }
        }
        Ok(inds)
    }

    pub fn channels_as_tuple(
        &self,
        x: &[f32],
        shape: &[usize],
    ) -> Result<Vec<Vec<f32>>, StarDistDataError> {
        if x.len() != shape.iter().product::<usize>() {
            return Err(StarDistDataError::ShapeMismatch);
        }
        if let Some(n_channel) = self.n_channel {
            if shape.is_empty() || shape[shape.len() - 1] != n_channel {
                return Err(StarDistDataError::ChannelShapeMismatch);
            }
            let spatial_len = x.len() / n_channel;
            let mut channels = vec![Vec::<f32>::with_capacity(spatial_len); n_channel];
            for pixel in 0..spatial_len {
                for channel in 0..n_channel {
                    channels[channel].push(x[pixel * n_channel + channel]);
                }
            }
            Ok(channels)
        } else {
            Ok(vec![x.to_vec()])
        }
    }
}

impl StarDistData2D {
    pub fn new(
        base: StarDistDataBase,
        n_rays: usize,
        n_classes: Option<usize>,
        classes: Option<Vec<ClassAssignment>>,
        shape_completion: bool,
        b: usize,
    ) -> Result<Self, StarDistDataError> {
        if shape_completion && b > 0 && base.grid.iter().any(|g| b % *g != 0) {
            return Err(StarDistDataError::WrongPatchDimension);
        }
        Ok(Self {
            base,
            n_rays,
            n_classes,
            classes,
            shape_completion,
            b,
        })
    }

    pub fn __getitem__(
        &mut self,
        idx: &[usize],
        x_images: &[&[f32]],
        x_shapes: &[[usize; 3]],
        y_images: &[&[i32]],
        y_shapes: &[[usize; 2]],
        random_values: &[f32],
        seed: u64,
    ) -> Result<StarDistData2DBatch, StarDistDataError> {
        if idx.is_empty()
            || random_values.len() < idx.len()
            || x_images.len() != y_images.len()
            || x_shapes.len() != x_images.len()
            || y_shapes.len() != y_images.len()
        {
            return Err(StarDistDataError::ShapeMismatch);
        }
        if self.base.patch_size.len() != 2 || self.base.grid.len() != 2 {
            return Err(StarDistDataError::WrongPatchDimension);
        }
        if let Some(classes) = &self.classes {
            if classes.len() != x_images.len() {
                return Err(StarDistDataError::ClassesShapeMismatch);
            }
        }

        let batch = idx.len();
        let patch_h = self.base.patch_size[0];
        let patch_w = self.base.patch_size[1];
        let grid_y = self.base.grid[0];
        let grid_x = self.base.grid[1];
        let crop_b = if self.shape_completion { self.b } else { 0 };
        if crop_b * 2 >= patch_h || crop_b * 2 >= patch_w {
            return Err(StarDistDataError::WrongPatchDimension);
        }
        let x_h = patch_h - 2 * crop_b;
        let x_w = patch_w - 2 * crop_b;
        let out_h = (x_h - 1) / grid_y + 1;
        let out_w = (x_w - 1) / grid_x + 1;
        let n_channel = self.base.n_channel.unwrap_or(1);
        let mut x_batch = Vec::<f32>::with_capacity(batch * x_h * x_w * n_channel);
        let mut prob_batch = Vec::<f32>::with_capacity(batch * out_h * out_w);
        let mut dist_batch = Vec::<f32>::with_capacity(batch * out_h * out_w * (self.n_rays + 1));
        let mut prob_class_batch = if let Some(n_classes) = self.n_classes {
            Some(Vec::<f32>::with_capacity(
                batch * out_h * out_w * (n_classes + 1),
            ))
        } else {
            None
        };

        for (batch_i, k) in idx.iter().enumerate() {
            let k = *k;
            if k >= x_images.len() {
                return Err(StarDistDataError::ShapeMismatch);
            }
            if x_shapes[k][0] != y_shapes[k][0] || x_shapes[k][1] != y_shapes[k][1] {
                return Err(StarDistDataError::ShapeMismatch);
            }
            if x_images[k].len() != x_shapes[k].iter().product::<usize>()
                || y_images[k].len() != y_shapes[k].iter().product::<usize>()
            {
                return Err(StarDistDataError::ShapeMismatch);
            }
            if self.base.n_channel.is_some() && x_shapes[k][2] != n_channel {
                return Err(StarDistDataError::ChannelShapeMismatch);
            }
            if self.base.n_channel.is_none() && x_shapes[k][2] != 1 {
                return Err(StarDistDataError::ChannelShapeMismatch);
            }

            let valid_inds = self.base.get_valid_inds(
                k,
                y_images[k],
                &[y_shapes[k][0], y_shapes[k][1]],
                None,
                random_values[batch_i],
            )?;
            let y_as_f32 = y_images[k].iter().map(|v| *v as f32).collect::<Vec<_>>();
            let mut datas = Vec::<&[f32]>::with_capacity(n_channel + 1);
            datas.push(&y_as_f32);
            let channel_data = self.base.channels_as_tuple(
                x_images[k],
                &[x_shapes[k][0], x_shapes[k][1], x_shapes[k][2]],
            )?;
            for channel in &channel_data {
                datas.push(channel);
            }
            let patches = crate::sample_patches::sample_patches(
                &datas,
                &[y_shapes[k][0], y_shapes[k][1]],
                &self.base.patch_size,
                1,
                Some(&valid_inds),
                seed + batch_i as u64,
            )?;
            let mut y_patch = patches[0].iter().map(|v| *v as i32).collect::<Vec<i32>>();
            if self.base.n_channel.is_none() {
                for y in 0..x_h {
                    let source_y = y + crop_b;
                    for x in 0..x_w {
                        let source_x = x + crop_b;
                        x_batch.push(patches[1][source_y * patch_w + source_x]);
                    }
                }
            } else {
                for y in 0..x_h {
                    let source_y = y + crop_b;
                    for x in 0..x_w {
                        let source_x = x + crop_b;
                        let pixel = source_y * patch_w + source_x;
                        for channel in 0..n_channel {
                            x_batch.push(patches[channel + 1][pixel]);
                        }
                    }
                }
            }

            let mut mask_neg_labels = vec![false; out_h * out_w];
            let mut has_neg_labels = false;
            for y in 0..out_h {
                for x in 0..out_w {
                    let source = ((y * grid_y) + crop_b) * patch_w + ((x * grid_x) + crop_b);
                    if y_patch[source] < 0 {
                        mask_neg_labels[y * out_w + x] = true;
                        has_neg_labels = true;
                    }
                }
            }
            if has_neg_labels {
                for value in &mut y_patch {
                    if *value < 0 {
                        *value = 0;
                    }
                }
            }

            let mut y_subsampled = Vec::<i32>::with_capacity(out_h * out_w);
            for y in 0..out_h {
                for x in 0..out_w {
                    y_subsampled
                        .push(y_patch[((y * grid_y) + crop_b) * patch_w + ((x * grid_x) + crop_b)]);
                }
            }
            let mut prob = crate::edt_prob(&y_subsampled, &[out_h, out_w], None)?;
            if has_neg_labels {
                for i in 0..prob.len() {
                    if mask_neg_labels[i] {
                        prob[i] = -1.0;
                    }
                }
            }
            prob_batch.extend_from_slice(&prob);

            if self.shape_completion {
                let mut y_cleared = y_patch.clone();
                let mut border_labels = Vec::<i32>::new();
                for x in 0..patch_w {
                    let top = y_cleared[x];
                    if top > 0 && !border_labels.contains(&top) {
                        border_labels.push(top);
                    }
                    let bottom = y_cleared[(patch_h - 1) * patch_w + x];
                    if bottom > 0 && !border_labels.contains(&bottom) {
                        border_labels.push(bottom);
                    }
                }
                for y in 0..patch_h {
                    let left = y_cleared[y * patch_w];
                    if left > 0 && !border_labels.contains(&left) {
                        border_labels.push(left);
                    }
                    let right = y_cleared[y * patch_w + patch_w - 1];
                    if right > 0 && !border_labels.contains(&right) {
                        border_labels.push(right);
                    }
                }
                if !border_labels.is_empty() {
                    for value in &mut y_cleared {
                        if border_labels.contains(value) {
                            *value = 0;
                        }
                    }
                }
                let labels_u16 = y_cleared
                    .iter()
                    .map(|v| (*v).max(0) as u16)
                    .collect::<Vec<u16>>();
                let dist = crate::star_dist(&labels_u16, [patch_h, patch_w], self.n_rays, [1, 1])?;
                let mut y_cleared_subsampled = Vec::<i32>::with_capacity(out_h * out_w);
                for y in 0..out_h {
                    for x in 0..out_w {
                        y_cleared_subsampled.push(
                            y_cleared[((y * grid_y) + crop_b) * patch_w + ((x * grid_x) + crop_b)],
                        );
                    }
                }
                let dist_mask = crate::edt_prob(&y_cleared_subsampled, &[out_h, out_w], None)?;
                for y in 0..out_h {
                    for x in 0..out_w {
                        for ray in 0..self.n_rays {
                            dist_batch.push(dist[[y * grid_y + crop_b, x * grid_x + crop_b, ray]]);
                        }
                        dist_batch.push(dist_mask[y * out_w + x]);
                    }
                }
            } else {
                let labels_u16 = y_patch
                    .iter()
                    .map(|v| (*v).max(0) as u16)
                    .collect::<Vec<u16>>();
                let dist = crate::star_dist(
                    &labels_u16,
                    [patch_h, patch_w],
                    self.n_rays,
                    [grid_y, grid_x],
                )?;
                for y in 0..out_h {
                    for x in 0..out_w {
                        for ray in 0..self.n_rays {
                            dist_batch.push(dist[[y, x, ray]]);
                        }
                        dist_batch.push(prob[y * out_w + x]);
                    }
                }
            }

            if let Some(n_classes) = self.n_classes {
                let classes = self
                    .classes
                    .as_ref()
                    .ok_or(StarDistDataError::ClassesShapeMismatch)?;
                let (prob_class_full, _) = crate::mask_to_categorical(
                    &y_patch,
                    &[patch_h, patch_w],
                    n_classes,
                    classes[k].clone(),
                    false,
                )?;
                let channels = n_classes + 1;
                if let Some(prob_class_batch) = &mut prob_class_batch {
                    for y in 0..out_h {
                        for x in 0..out_w {
                            let source = (((y * grid_y) + crop_b) * patch_w
                                + ((x * grid_x) + crop_b))
                                * channels;
                            for channel in 0..channels {
                                let mut value = prob_class_full[source + channel];
                                if has_neg_labels && mask_neg_labels[y * out_w + x] {
                                    value = -1.0;
                                }
                                prob_class_batch.push(value);
                            }
                        }
                    }
                }
            }
        }

        Ok(StarDistData2DBatch {
            x: x_batch,
            x_shape: [batch, x_h, x_w, n_channel],
            prob: prob_batch,
            prob_shape: [batch, out_h, out_w, 1],
            dist: dist_batch,
            dist_shape: [batch, out_h, out_w, self.n_rays + 1],
            prob_class: prob_class_batch,
            prob_class_shape: self
                .n_classes
                .map(|n_classes| [batch, out_h, out_w, n_classes + 1]),
        })
    }
}

impl StarDistData3D {
    pub fn new(
        base: StarDistDataBase,
        rays: crate::Rays,
        anisotropy: Option<[f32; 3]>,
        n_classes: Option<usize>,
        classes: Option<Vec<ClassAssignment>>,
    ) -> Self {
        Self {
            base,
            rays,
            anisotropy,
            n_classes,
            classes,
        }
    }

    pub fn __getitem__(
        &mut self,
        idx: &[usize],
        x_images: &[&[f32]],
        x_shapes: &[[usize; 4]],
        y_images: &[&[i32]],
        y_shapes: &[[usize; 3]],
        random_values: &[f32],
        seed: u64,
    ) -> Result<StarDistData3DBatch, StarDistDataError> {
        if idx.is_empty()
            || random_values.len() < idx.len()
            || x_images.len() != y_images.len()
            || x_shapes.len() != x_images.len()
            || y_shapes.len() != y_images.len()
        {
            return Err(StarDistDataError::ShapeMismatch);
        }
        if self.base.patch_size.len() != 3 || self.base.grid.len() != 3 {
            return Err(StarDistDataError::WrongPatchDimension);
        }
        if let Some(classes) = &self.classes {
            if classes.len() != x_images.len() {
                return Err(StarDistDataError::ClassesShapeMismatch);
            }
        }

        let batch = idx.len();
        let patch_d = self.base.patch_size[0];
        let patch_h = self.base.patch_size[1];
        let patch_w = self.base.patch_size[2];
        let grid_z = self.base.grid[0];
        let grid_y = self.base.grid[1];
        let grid_x = self.base.grid[2];
        let out_d = (patch_d - 1) / grid_z + 1;
        let out_h = (patch_h - 1) / grid_y + 1;
        let out_w = (patch_w - 1) / grid_x + 1;
        let n_channel = self.base.n_channel.unwrap_or(1);
        let n_rays = self.rays.vertices.len();
        let mut x_batch =
            Vec::<f32>::with_capacity(batch * patch_d * patch_h * patch_w * n_channel);
        let mut prob_batch = Vec::<f32>::with_capacity(batch * out_d * out_h * out_w);
        let mut dist_batch =
            Vec::<f32>::with_capacity(batch * out_d * out_h * out_w * (n_rays + 1));
        let mut prob_class_batch = if let Some(n_classes) = self.n_classes {
            Some(Vec::<f32>::with_capacity(
                batch * out_d * out_h * out_w * (n_classes + 1),
            ))
        } else {
            None
        };

        for (batch_i, k) in idx.iter().enumerate() {
            let k = *k;
            if k >= x_images.len() {
                return Err(StarDistDataError::ShapeMismatch);
            }
            if x_shapes[k][0] != y_shapes[k][0]
                || x_shapes[k][1] != y_shapes[k][1]
                || x_shapes[k][2] != y_shapes[k][2]
            {
                return Err(StarDistDataError::ShapeMismatch);
            }
            if x_images[k].len() != x_shapes[k].iter().product::<usize>()
                || y_images[k].len() != y_shapes[k].iter().product::<usize>()
            {
                return Err(StarDistDataError::ShapeMismatch);
            }
            if self.base.n_channel.is_some() && x_shapes[k][3] != n_channel {
                return Err(StarDistDataError::ChannelShapeMismatch);
            }
            if self.base.n_channel.is_none() && x_shapes[k][3] != 1 {
                return Err(StarDistDataError::ChannelShapeMismatch);
            }

            let valid_inds = self.base.get_valid_inds(
                k,
                y_images[k],
                &[y_shapes[k][0], y_shapes[k][1], y_shapes[k][2]],
                None,
                random_values[batch_i],
            )?;
            let y_as_f32 = y_images[k].iter().map(|v| *v as f32).collect::<Vec<_>>();
            let mut datas = Vec::<&[f32]>::with_capacity(n_channel + 1);
            datas.push(&y_as_f32);
            let channel_data = self.base.channels_as_tuple(
                x_images[k],
                &[
                    x_shapes[k][0],
                    x_shapes[k][1],
                    x_shapes[k][2],
                    x_shapes[k][3],
                ],
            )?;
            for channel in &channel_data {
                datas.push(channel);
            }
            let patches = crate::sample_patches::sample_patches(
                &datas,
                &[y_shapes[k][0], y_shapes[k][1], y_shapes[k][2]],
                &self.base.patch_size,
                1,
                Some(&valid_inds),
                seed + batch_i as u64,
            )?;
            let mut y_patch = patches[0].iter().map(|v| *v as i32).collect::<Vec<i32>>();
            if self.base.n_channel.is_none() {
                x_batch.extend_from_slice(&patches[1]);
            } else {
                for voxel in 0..(patch_d * patch_h * patch_w) {
                    for channel in 0..n_channel {
                        x_batch.push(patches[channel + 1][voxel]);
                    }
                }
            }

            let mut mask_neg_labels = vec![false; out_d * out_h * out_w];
            let mut has_neg_labels = false;
            for z in 0..out_d {
                for y in 0..out_h {
                    for x in 0..out_w {
                        let source =
                            ((z * grid_z) * patch_h + (y * grid_y)) * patch_w + (x * grid_x);
                        if y_patch[source] < 0 {
                            mask_neg_labels[(z * out_h + y) * out_w + x] = true;
                            has_neg_labels = true;
                        }
                    }
                }
            }
            if has_neg_labels {
                for value in &mut y_patch {
                    if *value < 0 {
                        *value = 0;
                    }
                }
            }

            let edt = crate::edt_prob(
                &y_patch,
                &[patch_d, patch_h, patch_w],
                self.anisotropy.as_ref().map(|a| &a[..]),
            )?;
            let mut prob = Vec::<f32>::with_capacity(out_d * out_h * out_w);
            for z in 0..out_d {
                for y in 0..out_h {
                    for x in 0..out_w {
                        let source =
                            ((z * grid_z) * patch_h + (y * grid_y)) * patch_w + (x * grid_x);
                        prob.push(edt[source]);
                    }
                }
            }
            if has_neg_labels {
                for i in 0..prob.len() {
                    if mask_neg_labels[i] {
                        prob[i] = -1.0;
                    }
                }
            }
            prob_batch.extend_from_slice(&prob);

            let labels_u16 = y_patch
                .iter()
                .map(|v| (*v).max(0) as u16)
                .collect::<Vec<u16>>();
            let dist = crate::star_dist3d(
                &labels_u16,
                [patch_d, patch_h, patch_w],
                &self.rays,
                [grid_z, grid_y, grid_x],
            )?;
            for z in 0..out_d {
                for y in 0..out_h {
                    for x in 0..out_w {
                        for ray in 0..n_rays {
                            dist_batch.push(dist[[z, y, x, ray]]);
                        }
                        dist_batch.push(prob[(z * out_h + y) * out_w + x]);
                    }
                }
            }

            if let Some(n_classes) = self.n_classes {
                let classes = self
                    .classes
                    .as_ref()
                    .ok_or(StarDistDataError::ClassesShapeMismatch)?;
                let (prob_class_full, _) = crate::mask_to_categorical(
                    &y_patch,
                    &[patch_d, patch_h, patch_w],
                    n_classes,
                    classes[k].clone(),
                    false,
                )?;
                let channels = n_classes + 1;
                if let Some(prob_class_batch) = &mut prob_class_batch {
                    for z in 0..out_d {
                        for y in 0..out_h {
                            for x in 0..out_w {
                                let source = (((z * grid_z) * patch_h + (y * grid_y)) * patch_w
                                    + (x * grid_x))
                                    * channels;
                                for channel in 0..channels {
                                    let mut value = prob_class_full[source + channel];
                                    if has_neg_labels
                                        && mask_neg_labels[(z * out_h + y) * out_w + x]
                                    {
                                        value = -1.0;
                                    }
                                    prob_class_batch.push(value);
                                }
                            }
                        }
                    }
                }
            }
        }

        Ok(StarDistData3DBatch {
            x: x_batch,
            x_shape: [batch, patch_d, patch_h, patch_w, n_channel],
            prob: prob_batch,
            prob_shape: [batch, out_d, out_h, out_w, 1],
            dist: dist_batch,
            dist_shape: [batch, out_d, out_h, out_w, n_rays + 1],
            prob_class: prob_class_batch,
            prob_class_shape: self
                .n_classes
                .map(|n_classes| [batch, out_d, out_h, out_w, n_classes + 1]),
        })
    }
}

impl StarDistPadAndCropResizer {
    pub fn new(grid: Vec<(char, usize)>, mode: PadMode, constant_value: f32) -> Self {
        let grid = grid
            .into_iter()
            .map(|(axis, value)| (axis.to_ascii_uppercase(), value))
            .collect();
        Self {
            grid,
            mode,
            constant_value,
            pad: Vec::new(),
            padded_shape: Vec::new(),
        }
    }

    pub fn before(
        &mut self,
        x: &[f32],
        shape: &[usize],
        axes: &str,
        axes_div_by: &[usize],
    ) -> Result<(Vec<f32>, Vec<usize>), ResizerError> {
        if axes.len() != shape.len() {
            return Err(ResizerError::DimensionMismatch);
        }
        if axes_div_by.len() != shape.len() {
            return Err(ResizerError::AxesDivByMismatch);
        }
        if x.len() != shape.iter().product::<usize>() {
            return Err(ResizerError::ShapeMismatch);
        }
        let mut normalized_axes = Vec::<char>::with_capacity(axes.len());
        for axis in axes.chars() {
            let axis = axis.to_ascii_uppercase();
            if normalized_axes.contains(&axis) {
                return Err(ResizerError::DuplicateAxis);
            }
            normalized_axes.push(axis);
        }
        for (axis, axis_div_by) in normalized_axes.iter().zip(axes_div_by.iter()) {
            let mut grid = 1usize;
            for (grid_axis, grid_value) in &self.grid {
                if *grid_axis == *axis {
                    grid = *grid_value;
                    break;
                }
            }
            if *axis_div_by % grid != 0 {
                return Err(ResizerError::GridDivisibility);
            }
        }

        self.pad.clear();
        self.padded_shape.clear();
        let mut out_shape = Vec::<usize>::with_capacity(shape.len());
        for i in 0..shape.len() {
            let div_n = axes_div_by[i];
            let pad_after = (div_n - shape[i] % div_n) % div_n;
            self.pad.push((normalized_axes[i], [0, pad_after]));
            out_shape.push(shape[i] + pad_after);
            if normalized_axes[i] != 'C' {
                self.padded_shape
                    .push((normalized_axes[i], shape[i] + pad_after));
            }
        }

        let mut in_strides = vec![1usize; shape.len()];
        if !shape.is_empty() {
            for i in (0..shape.len() - 1).rev() {
                in_strides[i] = in_strides[i + 1] * shape[i + 1];
            }
        }
        let mut out_strides = vec![1usize; out_shape.len()];
        if !out_shape.is_empty() {
            for i in (0..out_shape.len() - 1).rev() {
                out_strides[i] = out_strides[i + 1] * out_shape[i + 1];
            }
        }
        let out_len = out_shape.iter().product::<usize>();
        let mut x_pad = vec![self.constant_value; out_len];
        for (out_index, value) in x_pad.iter_mut().enumerate() {
            let mut remainder = out_index;
            let mut in_index = 0usize;
            let mut constant_pad = false;
            for axis_i in 0..out_shape.len() {
                let coord = remainder / out_strides[axis_i];
                remainder %= out_strides[axis_i];
                let source_coord = if coord < shape[axis_i] {
                    coord
                } else if self.mode == PadMode::Constant {
                    constant_pad = true;
                    0
                } else if self.mode == PadMode::Edge || shape[axis_i] <= 1 {
                    shape[axis_i] - 1
                } else {
                    let period = 2 * shape[axis_i] - 2;
                    let folded = coord % period;
                    if folded < shape[axis_i] {
                        folded
                    } else {
                        period - folded
                    }
                };
                in_index += source_coord * in_strides[axis_i];
            }
            if !constant_pad {
                *value = x[in_index];
            }
        }

        Ok((x_pad, out_shape))
    }

    pub fn after(
        &self,
        x: &[f32],
        shape: &[usize],
        axes: &str,
    ) -> Result<(Vec<f32>, Vec<usize>), ResizerError> {
        if self.pad.is_empty() || self.padded_shape.is_empty() {
            return Err(ResizerError::MissingBeforeState);
        }
        if axes.len() != shape.len() {
            return Err(ResizerError::DimensionMismatch);
        }
        if x.len() != shape.iter().product::<usize>() {
            return Err(ResizerError::ShapeMismatch);
        }
        let mut normalized_axes = Vec::<char>::with_capacity(axes.len());
        for axis in axes.chars() {
            let axis = axis.to_ascii_uppercase();
            if normalized_axes.contains(&axis) {
                return Err(ResizerError::DuplicateAxis);
            }
            normalized_axes.push(axis);
        }
        let mut out_shape = Vec::<usize>::with_capacity(shape.len());
        for i in 0..shape.len() {
            let axis = normalized_axes[i];
            let mut padded_axis_shape = shape[i];
            for (padded_axis, value) in &self.padded_shape {
                if *padded_axis == axis {
                    padded_axis_shape = *value;
                    break;
                }
            }
            let mut grid = 1usize;
            for (grid_axis, grid_value) in &self.grid {
                if *grid_axis == axis {
                    grid = *grid_value;
                    break;
                }
            }
            if padded_axis_shape != shape[i] * grid {
                return Err(ResizerError::InconsistentNetworkShape);
            }
            let mut pad_after = 0usize;
            for (pad_axis, pad_value) in &self.pad {
                if *pad_axis == axis {
                    pad_after = pad_value[1];
                    break;
                }
            }
            let crop_after = if pad_after >= grid {
                pad_after / grid
            } else {
                0
            };
            out_shape.push(shape[i] - crop_after);
        }

        let mut in_strides = vec![1usize; shape.len()];
        if !shape.is_empty() {
            for i in (0..shape.len() - 1).rev() {
                in_strides[i] = in_strides[i + 1] * shape[i + 1];
            }
        }
        let mut out_strides = vec![1usize; out_shape.len()];
        if !out_shape.is_empty() {
            for i in (0..out_shape.len() - 1).rev() {
                out_strides[i] = out_strides[i + 1] * out_shape[i + 1];
            }
        }
        let out_len = out_shape.iter().product::<usize>();
        let mut cropped = vec![0.0f32; out_len];
        for (out_index, value) in cropped.iter_mut().enumerate() {
            let mut remainder = out_index;
            let mut in_index = 0usize;
            for axis_i in 0..out_shape.len() {
                let coord = remainder / out_strides[axis_i];
                remainder %= out_strides[axis_i];
                in_index += coord * in_strides[axis_i];
            }
            *value = x[in_index];
        }
        Ok((cropped, out_shape))
    }

    pub fn filter_points<const N: usize>(
        &self,
        ndim: usize,
        points: &[[f32; N]],
        axes: &str,
    ) -> Result<Vec<usize>, ResizerError> {
        if self.pad.is_empty() || self.padded_shape.is_empty() {
            return Err(ResizerError::MissingBeforeState);
        }
        if axes.len() != ndim {
            return Err(ResizerError::DimensionMismatch);
        }
        let mut normalized_axes = Vec::<char>::with_capacity(axes.len());
        for axis in axes.chars() {
            let axis = axis.to_ascii_uppercase();
            if normalized_axes.contains(&axis) {
                return Err(ResizerError::DuplicateAxis);
            }
            normalized_axes.push(axis);
        }
        let mut bounds = Vec::<f32>::new();
        for axis in normalized_axes {
            if matches!(axis, 'Z' | 'Y' | 'X') {
                let mut padded_axis_shape = None;
                for (padded_axis, value) in &self.padded_shape {
                    if *padded_axis == axis {
                        padded_axis_shape = Some(*value);
                        break;
                    }
                }
                let mut pad_after = None;
                for (pad_axis, pad_value) in &self.pad {
                    if *pad_axis == axis {
                        pad_after = Some(pad_value[1]);
                        break;
                    }
                }
                bounds.push(
                    (padded_axis_shape.ok_or(ResizerError::MissingBeforeState)?
                        - pad_after.ok_or(ResizerError::MissingBeforeState)?)
                        as f32,
                );
            }
        }
        if bounds.len() != N {
            return Err(ResizerError::PointDimensionMismatch);
        }
        let mut indices = Vec::<usize>::new();
        for (i, point) in points.iter().enumerate() {
            let mut inside = true;
            for axis_i in 0..N {
                if point[axis_i] >= bounds[axis_i] {
                    inside = false;
                    break;
                }
            }
            if inside {
                indices.push(i);
            }
        }
        Ok(indices)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct StarDist2DOutputs<T> {
    pub prob: T,
    pub dist: T,
}

#[derive(Clone, Debug, PartialEq)]
pub struct StarDistDirectPrediction {
    pub prob: Vec<f32>,
    pub prob_shape: Vec<usize>,
    pub dist: Vec<f32>,
    pub dist_shape: Vec<usize>,
    pub prob_class: Option<Vec<f32>>,
    pub prob_class_shape: Option<Vec<usize>>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct StarDistPredictSetup {
    pub x: Vec<f32>,
    pub x_shape: Vec<usize>,
    pub axes: String,
    pub axes_net: String,
    pub axes_net_div_by: Vec<usize>,
    pub n_tiles: Vec<usize>,
    pub grid: Vec<usize>,
    pub channel: usize,
    pub resizer: StarDistPadAndCropResizer,
}

#[derive(Clone, Debug, PartialEq)]
pub struct StarDistSparsePrediction<const N: usize> {
    pub prob: Vec<f32>,
    pub dist: Vec<f32>,
    pub points: Vec<[f32; N]>,
    pub prob_class: Option<Vec<f32>>,
    pub prob_class_channels: Option<usize>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct StarDistPrediction {
    pub prob: Vec<f32>,
    pub prob_shape: Vec<usize>,
    pub dist: Vec<f32>,
    pub dist_shape: Vec<usize>,
    pub prob_class: Option<Vec<f32>>,
    pub prob_class_shape: Option<Vec<usize>>,
}

#[derive(Debug, thiserror::Error)]
pub enum StarDistPredictError {
    #[error("input image length does not match shape")]
    ShapeMismatch,
    #[error("n_tiles must be an iterable of length equal to image dimensions")]
    TilesDimensionMismatch,
    #[error("all values of n_tiles must be integer values >= 1")]
    InvalidTiles,
    #[error("channel axis is missing")]
    MissingChannelAxis,
    #[error("input channel count does not match config.n_channel_in")]
    ChannelMismatch,
    #[error("tiled sparse prediction is not implemented in this direct-output wrapper")]
    TiledPredictionUnsupported,
    #[error("direct predictor returned incompatible output shapes")]
    OutputShapeMismatch,
    #[error("multiclass output is required by this model config")]
    MissingClassOutput,
    #[error("overlap_label is not supported for 2D")]
    OverlapLabel2DUnsupported,
    #[error(transparent)]
    Postprocess2D(#[from] StarDist2DPostprocessError),
    #[error(transparent)]
    Postprocess3D(#[from] StarDist3DPostprocessError),
    #[error(transparent)]
    Axes(#[from] AxesError),
    #[error(transparent)]
    AxesDivBy(#[from] AxesDivByError),
    #[error(transparent)]
    Resizer(#[from] ResizerError),
    #[error(transparent)]
    Nms(#[from] crate::nms::NmsError),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StarDistBuildLayer {
    pub name: String,
    pub kind: String,
    pub filters: Option<usize>,
    pub kernel: Vec<usize>,
    pub pool: Vec<usize>,
    pub activation: Option<String>,
    pub source: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StarDistBuildGraph {
    pub ndim: usize,
    pub backbone: String,
    pub input_shape: Vec<Option<usize>>,
    pub layers: Vec<StarDistBuildLayer>,
    pub outputs: Vec<String>,
}

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum StarDistBuildError {
    #[error("unsupported backbone")]
    UnsupportedBackbone,
    #[error("grid cannot be reached by Python build pooling loop")]
    InvalidGrid,
}

#[derive(Clone, Debug, PartialEq)]
pub struct StarDist2DInstances {
    pub labels: Option<Array2<u32>>,
    pub coord: Array3<f32>,
    pub points: Vec<[f32; 2]>,
    pub prob: Vec<f32>,
    pub class_prob: Option<Vec<f32>>,
    pub class_prob_channels: Option<usize>,
    pub class_id: Option<Vec<usize>>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct StarDist2DPredictInstancesResult {
    pub instances: StarDist2DInstances,
    pub prediction: Option<StarDistPrediction>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct StarDist3DInstances {
    pub labels: Option<Array3<u32>>,
    pub dist: Vec<f32>,
    pub points: Vec<[f32; 3]>,
    pub prob: Vec<f32>,
    pub rays: crate::Rays,
    pub rays_vertices: Vec<[f32; 3]>,
    pub rays_faces: Vec<[usize; 3]>,
    pub class_prob: Option<Vec<f32>>,
    pub class_prob_channels: Option<usize>,
    pub class_id: Option<Vec<usize>>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct StarDist3DPredictInstancesResult {
    pub instances: StarDist3DInstances,
    pub prediction: Option<StarDistPrediction>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct StarDistBigPrediction {
    pub labels: Vec<i32>,
    pub labels_shape: Vec<usize>,
    pub polys: crate::BigPolys,
}

#[derive(Clone, Debug, PartialEq)]
pub struct StarDistBigResult {
    pub labels: Option<Vec<i32>>,
    pub labels_shape: Vec<usize>,
    pub polys: crate::BigPolys,
    pub n_blocks: usize,
}

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum StarDistPredictInstancesBigError {
    #[error("input image length does not match shape")]
    ShapeMismatch,
    #[error("axes length does not match image dimensions")]
    DimensionMismatch,
    #[error("label image contains negative labels")]
    NegativeLabel,
    #[error("labels_out must have the output shape")]
    LabelsOutShapeMismatch,
    #[error(transparent)]
    AxesDivBy(#[from] AxesDivByError),
    #[error(transparent)]
    AxesTileOverlap(#[from] AxesTileOverlapError),
    #[error(transparent)]
    Big(#[from] crate::BigError),
    #[error(transparent)]
    Matching(#[from] crate::MatchingError),
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StarDist2DScale {
    pub y: f32,
    pub x: f32,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StarDist3DScale {
    pub z: f32,
    pub y: f32,
    pub x: f32,
}

#[derive(Debug, thiserror::Error)]
pub enum StarDist2DPostprocessError {
    #[error(transparent)]
    Nms(#[from] crate::nms::NmsError),
    #[error(transparent)]
    Geometry(#[from] crate::geometry::GeometryError),
    #[error("prob_class must have one row per input candidate/pixel")]
    ClassProbShapeMismatch,
    #[error("scale must be non-zero for X and Y")]
    InvalidScale,
}

#[derive(Debug, thiserror::Error)]
pub enum StarDist3DPostprocessError {
    #[error(transparent)]
    Nms(#[from] crate::nms::NmsError),
    #[error(transparent)]
    Geometry(#[from] crate::geometry::GeometryError),
    #[error(transparent)]
    Rays(#[from] crate::RaysError),
    #[error("prob_class must have one row per input candidate/pixel")]
    ClassProbShapeMismatch,
    #[error("scale must be non-zero for X, Y, and Z")]
    InvalidScale,
}

impl StarDist2D {
    pub fn new(config: Config2D) -> Self {
        Self {
            config,
            thresholds: StarDistThresholds::default(),
        }
    }

    pub fn thresholds(&self) -> StarDistThresholds {
        self.thresholds
    }

    pub fn set_thresholds(&mut self, d: StarDistThresholds) -> Result<(), ThresholdsError> {
        if !d.prob.is_finite() || d.prob <= 0.0 || d.prob >= 1.0 {
            return Err(ThresholdsError::InvalidProb);
        }
        if !d.nms.is_finite() || d.nms <= 0.0 || d.nms >= 1.0 {
            return Err(ThresholdsError::InvalidNms);
        }
        self.thresholds = d;
        Ok(())
    }

    pub fn optimize_thresholds<F>(
        &mut self,
        y_val: &[&[u32]],
        yhat_prob: &[&[f32]],
        nms_threshs: &[f32],
        iou_threshs: &[f32],
        measure: crate::OptimizeThresholdMeasure,
        bracket: Option<[f32; 2]>,
        tol: f32,
        maxiter: usize,
        mut predict_instances: F,
    ) -> Result<StarDistThresholds, OptimizeThresholdsError>
    where
        F: FnMut(usize, f32, f32) -> Result<Vec<u32>, crate::UtilsError>,
    {
        if nms_threshs.is_empty() {
            return Err(OptimizeThresholdsError::EmptyNmsThresholds);
        }
        let mut opt_prob_thresh = 0.5f32;
        let mut opt_measure = f32::NEG_INFINITY;
        let mut opt_nms_thresh = 0.4f32;
        for nms_thresh in nms_threshs {
            let (prob_thresh, value) = crate::optimize_threshold(
                y_val,
                yhat_prob,
                *nms_thresh,
                measure,
                iou_threshs,
                bracket,
                tol,
                maxiter,
                |i, prob_thresh, nms_thresh| predict_instances(i, prob_thresh, nms_thresh),
            )?;
            if value > opt_measure {
                opt_prob_thresh = prob_thresh;
                opt_measure = value;
                opt_nms_thresh = *nms_thresh;
            }
        }
        let opt_threshs = StarDistThresholds {
            prob: opt_prob_thresh,
            nms: opt_nms_thresh,
        };
        self.set_thresholds(opt_threshs)?;
        Ok(opt_threshs)
    }

    pub fn _config_class(&self) -> ConfigClass {
        ConfigClass::Config2D
    }

    pub fn _build(&self) -> Result<StarDistBuildGraph, StarDistBuildError> {
        if self.config.backbone != "unet" {
            return Err(StarDistBuildError::UnsupportedBackbone);
        }
        let mut layers = Vec::<StarDistBuildLayer>::new();
        layers.push(StarDistBuildLayer {
            name: "input".to_string(),
            kind: "Input".to_string(),
            filters: None,
            kernel: Vec::new(),
            pool: Vec::new(),
            activation: None,
            source: "input".to_string(),
        });

        let mut pooled = [1usize, 1usize];
        let mut pooled_source = "input".to_string();
        let mut stage = 0usize;
        while pooled != self.config.grid {
            let mut pool = [1usize, 1usize];
            for axis in 0..2 {
                if self.config.grid[axis] > pooled[axis] {
                    pool[axis] = 2;
                }
            }
            if pool == [1, 1] {
                return Err(StarDistBuildError::InvalidGrid);
            }
            for axis in 0..2 {
                pooled[axis] *= pool[axis];
            }
            for conv in 0..self.config.unet_n_conv_per_depth {
                let name = format!("pre_grid_{stage}_conv_{conv}");
                layers.push(StarDistBuildLayer {
                    name: name.clone(),
                    kind: "Conv2D".to_string(),
                    filters: Some(self.config.unet_n_filter_base),
                    kernel: self.config.unet_kernel_size.to_vec(),
                    pool: Vec::new(),
                    activation: Some(self.config.unet_activation.clone()),
                    source: pooled_source,
                });
                pooled_source = name;
            }
            let name = format!("pre_grid_{stage}_max_pool");
            layers.push(StarDistBuildLayer {
                name: name.clone(),
                kind: "MaxPooling2D".to_string(),
                filters: None,
                kernel: Vec::new(),
                pool: pool.to_vec(),
                activation: None,
                source: pooled_source,
            });
            pooled_source = name;
            stage += 1;
        }

        layers.push(StarDistBuildLayer {
            name: "unet_block".to_string(),
            kind: "unet_block".to_string(),
            filters: Some(self.config.unet_n_filter_base),
            kernel: self.config.unet_kernel_size.to_vec(),
            pool: self.config.unet_pool.to_vec(),
            activation: Some(self.config.unet_activation.clone()),
            source: pooled_source,
        });
        let mut output_source = "unet_block".to_string();
        if self.config.net_conv_after_unet > 0 {
            layers.push(StarDistBuildLayer {
                name: "features".to_string(),
                kind: "Conv2D".to_string(),
                filters: Some(self.config.net_conv_after_unet),
                kernel: self.config.unet_kernel_size.to_vec(),
                pool: Vec::new(),
                activation: Some(self.config.unet_activation.clone()),
                source: output_source,
            });
            output_source = "features".to_string();
        }
        layers.push(StarDistBuildLayer {
            name: "prob".to_string(),
            kind: "Conv2D".to_string(),
            filters: Some(1),
            kernel: vec![1, 1],
            pool: Vec::new(),
            activation: Some("sigmoid".to_string()),
            source: output_source.clone(),
        });
        layers.push(StarDistBuildLayer {
            name: "dist".to_string(),
            kind: "Conv2D".to_string(),
            filters: Some(self.config.n_rays),
            kernel: vec![1, 1],
            pool: Vec::new(),
            activation: Some("linear".to_string()),
            source: output_source.clone(),
        });
        let mut outputs = vec!["prob".to_string(), "dist".to_string()];
        if _is_multiclass(self.config.n_classes) {
            let class_source = if self.config.net_conv_after_unet > 0 {
                layers.push(StarDistBuildLayer {
                    name: "features_class".to_string(),
                    kind: "Conv2D".to_string(),
                    filters: Some(self.config.net_conv_after_unet),
                    kernel: self.config.unet_kernel_size.to_vec(),
                    pool: Vec::new(),
                    activation: Some(self.config.unet_activation.clone()),
                    source: "unet_block".to_string(),
                });
                "features_class".to_string()
            } else {
                "unet_block".to_string()
            };
            layers.push(StarDistBuildLayer {
                name: "prob_class".to_string(),
                kind: "Conv2D".to_string(),
                filters: Some(self.config.n_classes.unwrap() + 1),
                kernel: vec![1, 1],
                pool: Vec::new(),
                activation: Some("softmax".to_string()),
                source: class_source,
            });
            outputs.push("prob_class".to_string());
        }

        Ok(StarDistBuildGraph {
            ndim: 2,
            backbone: self.config.backbone.clone(),
            input_shape: self.config.net_input_shape.to_vec(),
            layers,
            outputs,
        })
    }

    pub fn _axes_div_by(&self, query_axes: &str) -> Result<Vec<usize>, AxesDivByError> {
        if self.config.backbone != "unet" {
            return Err(AxesDivByError::UnsupportedBackbone);
        }
        if query_axes.is_empty() {
            return Err(AxesDivByError::EmptyAxes);
        }
        let mut normalized_axes = Vec::<char>::with_capacity(query_axes.len());
        for axis in query_axes.chars() {
            let axis = axis.to_ascii_uppercase();
            if normalized_axes.contains(&axis) {
                return Err(AxesDivByError::DuplicateAxis);
            }
            normalized_axes.push(axis);
        }

        let mut div_by = Vec::<(char, usize)>::new();
        for (i, axis) in self
            .config
            .axes
            .chars()
            .filter(|axis| *axis != 'C')
            .enumerate()
        {
            let pool = self.config.unet_pool[i].pow(self.config.unet_n_depth as u32);
            div_by.push((axis, pool * self.config.grid[i]));
        }

        let mut result = Vec::<usize>::with_capacity(normalized_axes.len());
        for axis in normalized_axes {
            let mut value = 1usize;
            for (div_axis, div_value) in &div_by {
                if *div_axis == axis {
                    value = *div_value;
                    break;
                }
            }
            result.push(value);
        }
        Ok(result)
    }

    pub fn _axes_tile_overlap(&self, query_axes: &str) -> Result<Vec<usize>, AxesTileOverlapError> {
        if self.config.backbone != "unet" {
            return Err(AxesTileOverlapError::UnsupportedBackbone);
        }
        if query_axes.is_empty() {
            return Err(AxesTileOverlapError::EmptyAxes);
        }
        let mut normalized_axes = Vec::<char>::with_capacity(query_axes.len());
        for axis in query_axes.chars() {
            let axis = axis.to_ascii_uppercase();
            if normalized_axes.contains(&axis) {
                return Err(AxesTileOverlapError::DuplicateAxis);
            }
            normalized_axes.push(axis);
        }
        let mut overlap = Vec::<(char, usize)>::new();
        for (i, axis) in self
            .config
            .axes
            .chars()
            .filter(|axis| *axis != 'C')
            .enumerate()
        {
            let grid = self.config.grid[i];
            let log_grid = if grid > 0 && grid.is_power_of_two() {
                grid.trailing_zeros() as usize
            } else {
                return Err(AxesTileOverlapError::Unavailable);
            };
            let n_depth = self.config.unet_n_depth + log_grid;
            let kern_size = self.config.unet_kernel_size[i];
            let pool_size = self.config.unet_pool[i];
            let value = match (n_depth, kern_size, pool_size) {
                (1, 3, 1) => 6,
                (1, 5, 1) => 12,
                (1, 7, 1) => 18,
                (2, 3, 1) => 10,
                (2, 5, 1) => 20,
                (2, 7, 1) => 30,
                (3, 3, 1) => 14,
                (3, 5, 1) => 28,
                (3, 7, 1) => 42,
                (4, 3, 1) => 18,
                (4, 5, 1) => 36,
                (4, 7, 1) => 54,
                (5, 3, 1) => 22,
                (5, 5, 1) => 44,
                (5, 7, 1) => 66,
                (1, 3, 2) => 9,
                (1, 5, 2) => 17,
                (1, 7, 2) => 25,
                (2, 3, 2) => 22,
                (2, 5, 2) => 43,
                (2, 7, 2) => 62,
                (3, 3, 2) => 46,
                (3, 5, 2) => 92,
                (3, 7, 2) => 138,
                (4, 3, 2) => 94,
                (4, 5, 2) => 188,
                (4, 7, 2) => 282,
                (5, 3, 2) => 190,
                (5, 5, 2) => 380,
                (5, 7, 2) => 570,
                (1, 3, 4) => 14,
                (1, 5, 4) => 27,
                (1, 7, 4) => 38,
                (2, 3, 4) => 58,
                (2, 5, 4) => 116,
                (2, 7, 4) => 158,
                (3, 3, 4) => 234,
                (3, 5, 4) => 468,
                (3, 7, 4) => 638,
                (4, 3, 4) => 938,
                (4, 5, 4) => 1876,
                (4, 7, 4) => 2558,
                _ => return Err(AxesTileOverlapError::Unavailable),
            };
            overlap.push((axis, value));
        }

        let mut result = Vec::<usize>::with_capacity(normalized_axes.len());
        for axis in normalized_axes {
            let mut value = 0usize;
            for (overlap_axis, overlap_value) in &overlap {
                if *overlap_axis == axis {
                    value = *overlap_value;
                    break;
                }
            }
            result.push(value);
        }
        Ok(result)
    }

    pub fn _compute_receptive_field(
        &self,
        img_size: Option<&[usize]>,
    ) -> Result<Vec<(usize, usize)>, AxesTileOverlapError> {
        if let Some(img_size) = img_size {
            if img_size.len() != self.config.n_dim
                || img_size.iter().any(|size| !size.is_power_of_two())
            {
                return Err(AxesTileOverlapError::Unavailable);
            }
        }
        let axes = self.config.axes.replace('C', "");
        let overlap = self._axes_tile_overlap(&axes)?;
        let mut receptive_field = Vec::<(usize, usize)>::with_capacity(overlap.len());
        for value in overlap {
            receptive_field.push((value, value));
        }
        Ok(receptive_field)
    }

    pub fn _normalize_axes(
        &self,
        img_shape: &[usize],
        axes: Option<&str>,
    ) -> Result<String, AxesError> {
        let axes = if let Some(axes) = axes {
            axes.to_string()
        } else {
            if !self.config.axes.contains('C') {
                return Err(AxesError::MissingConfigChannelAxis);
            }
            if img_shape.len() == self.config.axes.len() - 1 && self.config.n_channel_in == 1 {
                self.config.axes.replace('C', "")
            } else {
                self.config.axes.clone()
            }
        };
        if axes.len() != img_shape.len() {
            return Err(AxesError::DimensionMismatch);
        }
        let mut normalized = String::with_capacity(axes.len());
        for axis in axes.chars() {
            let axis = axis.to_ascii_uppercase();
            if normalized.contains(axis) {
                return Err(AxesError::DuplicateAxis);
            }
            normalized.push(axis);
        }
        Ok(normalized)
    }

    pub fn _guess_n_tiles(
        &self,
        img_shape: &[usize],
        axes: Option<&str>,
    ) -> Result<Vec<usize>, AxesError> {
        let axes = self._normalize_axes(img_shape, axes)?;
        let mut spatial_shape = Vec::<usize>::with_capacity(self.config.n_dim);
        let mut channel_index = None;
        for (i, axis) in axes.chars().enumerate() {
            if axis == 'C' {
                channel_index = Some(i);
            } else {
                spatial_shape.push(img_shape[i]);
            }
        }
        let b = (self.config.train_batch_size as f32).powf(1.0 / self.config.n_dim as f32);
        let mut n_tiles = Vec::<usize>::with_capacity(spatial_shape.len());
        for (s, p) in spatial_shape
            .iter()
            .zip(self.config.train_patch_size.iter())
        {
            n_tiles.push((*s as f32 / (*p as f32 * b)).ceil() as usize);
        }
        if let Some(channel_index) = channel_index {
            n_tiles.insert(channel_index, 1);
        }
        Ok(n_tiles)
    }

    pub fn _predict_setup(
        &self,
        img: &[f32],
        img_shape: &[usize],
        axes: Option<&str>,
        n_tiles: Option<&[usize]>,
    ) -> Result<StarDistPredictSetup, StarDistPredictError> {
        if img.len() != img_shape.iter().product::<usize>() {
            return Err(StarDistPredictError::ShapeMismatch);
        }
        let n_tiles = if let Some(n_tiles) = n_tiles {
            if n_tiles.len() != img_shape.len() {
                return Err(StarDistPredictError::TilesDimensionMismatch);
            }
            let mut checked = Vec::<usize>::with_capacity(n_tiles.len());
            for value in n_tiles {
                if *value < 1 {
                    return Err(StarDistPredictError::InvalidTiles);
                }
                checked.push(*value);
            }
            checked
        } else {
            vec![1; img_shape.len()]
        };

        let axes = self._normalize_axes(img_shape, axes)?;
        let axes_net = self.config.axes.clone();
        let axes_net_chars = axes_net.chars().collect::<Vec<_>>();
        let axes_chars = axes.chars().collect::<Vec<_>>();
        let channel = axes_net_chars
            .iter()
            .position(|axis| *axis == 'C')
            .ok_or(StarDistPredictError::MissingChannelAxis)?;
        let axes_net_div_by = self._axes_div_by(&axes_net)?;

        let mut x_shape = Vec::<usize>::with_capacity(axes_net_chars.len());
        for axis in &axes_net_chars {
            if let Some(pos) = axes_chars.iter().position(|candidate| candidate == axis) {
                x_shape.push(img_shape[pos]);
            } else if *axis == 'C' && self.config.n_channel_in == 1 {
                x_shape.push(1);
            } else {
                return Err(StarDistPredictError::OutputShapeMismatch);
            }
        }
        if x_shape[channel] != self.config.n_channel_in {
            return Err(StarDistPredictError::ChannelMismatch);
        }

        let mut in_strides = vec![1usize; img_shape.len()];
        if !img_shape.is_empty() {
            for i in (0..img_shape.len() - 1).rev() {
                in_strides[i] = in_strides[i + 1] * img_shape[i + 1];
            }
        }
        let mut out_strides = vec![1usize; x_shape.len()];
        if !x_shape.is_empty() {
            for i in (0..x_shape.len() - 1).rev() {
                out_strides[i] = out_strides[i + 1] * x_shape[i + 1];
            }
        }
        let mut x = vec![0.0f32; x_shape.iter().product::<usize>()];
        for out_index in 0..x.len() {
            let mut remainder = out_index;
            let mut in_index = 0usize;
            for (axis_i, axis) in axes_net_chars.iter().enumerate() {
                let coord = remainder / out_strides[axis_i];
                remainder %= out_strides[axis_i];
                if let Some(pos) = axes_chars.iter().position(|candidate| candidate == axis) {
                    in_index += coord * in_strides[pos];
                }
            }
            x[out_index] = img[in_index];
        }

        let mut n_tiles_net = Vec::<usize>::with_capacity(axes_net_chars.len());
        for axis in &axes_net_chars {
            if let Some(pos) = axes_chars.iter().position(|candidate| candidate == axis) {
                n_tiles_net.push(n_tiles[pos]);
            } else {
                n_tiles_net.push(1);
            }
        }

        let grid = self.config.grid.to_vec();
        let grid_for_resizer = axes_net
            .chars()
            .filter(|axis| *axis != 'C')
            .zip(grid.iter().copied())
            .collect::<Vec<_>>();
        let mut resizer = StarDistPadAndCropResizer::new(grid_for_resizer, PadMode::Reflect, 0.0);
        let (x, x_shape) = resizer.before(&x, &x_shape, &axes_net, &axes_net_div_by)?;

        Ok(StarDistPredictSetup {
            x,
            x_shape,
            axes,
            axes_net,
            axes_net_div_by,
            n_tiles: n_tiles_net,
            grid,
            channel,
            resizer,
        })
    }

    pub fn _predict_generator<F>(
        &self,
        img: &[f32],
        img_shape: &[usize],
        axes: Option<&str>,
        n_tiles: Option<&[usize]>,
        mut predict_direct: F,
    ) -> Result<StarDistPrediction, StarDistPredictError>
    where
        F: FnMut(&[f32], &[usize], &str) -> Result<StarDistDirectPrediction, StarDistPredictError>,
    {
        let setup = self._predict_setup(img, img_shape, axes, n_tiles)?;
        if setup.n_tiles.iter().product::<usize>() > 1 {
            return Err(StarDistPredictError::TiledPredictionUnsupported);
        }

        let results = predict_direct(&setup.x, &setup.x_shape, &setup.axes_net)?;
        if results.prob_shape.len() != setup.axes_net.len()
            || results.dist_shape.len() != setup.axes_net.len()
            || results.prob.len() != results.prob_shape.iter().product::<usize>()
            || results.dist.len() != results.dist_shape.iter().product::<usize>()
            || results.prob_shape[setup.channel] != 1
            || results.dist_shape[setup.channel] != self.config.n_rays
        {
            return Err(StarDistPredictError::OutputShapeMismatch);
        }

        let (prob_resized, prob_shape_resized) =
            setup
                .resizer
                .after(&results.prob, &results.prob_shape, &setup.axes_net)?;
        let (dist_resized, dist_shape_resized) =
            setup
                .resizer
                .after(&results.dist, &results.dist_shape, &setup.axes_net)?;

        let mut spatial_shape = Vec::<usize>::with_capacity(self.config.n_dim);
        for (i, axis) in setup.axes_net.chars().enumerate() {
            if axis != 'C' {
                spatial_shape.push(prob_shape_resized[i]);
            }
        }
        if spatial_shape.len() != self.config.n_dim {
            return Err(StarDistPredictError::OutputShapeMismatch);
        }

        let mut prob_strides = vec![1usize; prob_shape_resized.len()];
        if !prob_shape_resized.is_empty() {
            for i in (0..prob_shape_resized.len() - 1).rev() {
                prob_strides[i] = prob_strides[i + 1] * prob_shape_resized[i + 1];
            }
        }
        let mut dist_strides = vec![1usize; dist_shape_resized.len()];
        if !dist_shape_resized.is_empty() {
            for i in (0..dist_shape_resized.len() - 1).rev() {
                dist_strides[i] = dist_strides[i + 1] * dist_shape_resized[i + 1];
            }
        }

        let mut prob = Vec::<f32>::with_capacity(spatial_shape.iter().product::<usize>());
        let mut dist = Vec::<f32>::with_capacity(prob.capacity() * self.config.n_rays);
        for y in 0..spatial_shape[0] {
            for x in 0..spatial_shape[1] {
                let mut prob_index = 0usize;
                let mut dist_base = 0usize;
                let mut spatial_i = 0usize;
                for (axis_i, axis) in setup.axes_net.chars().enumerate() {
                    let coord = if axis == 'C' {
                        0
                    } else {
                        let value = if spatial_i == 0 { y } else { x };
                        spatial_i += 1;
                        value
                    };
                    prob_index += coord * prob_strides[axis_i];
                    dist_base += coord * dist_strides[axis_i];
                }
                prob.push(prob_resized[prob_index]);
                for ray in 0..self.config.n_rays {
                    dist.push(
                        (dist_resized[dist_base + ray * dist_strides[setup.channel]]).max(1e-3),
                    );
                }
            }
        }

        let (prob_class, prob_class_shape) = if _is_multiclass(self.config.n_classes) {
            let class_values = results
                .prob_class
                .ok_or(StarDistPredictError::MissingClassOutput)?;
            let class_shape = results
                .prob_class_shape
                .ok_or(StarDistPredictError::MissingClassOutput)?;
            if class_shape.len() != setup.axes_net.len()
                || class_values.len() != class_shape.iter().product::<usize>()
            {
                return Err(StarDistPredictError::OutputShapeMismatch);
            }
            let channels = class_shape[setup.channel];
            let (class_resized, class_shape_resized) =
                setup
                    .resizer
                    .after(&class_values, &class_shape, &setup.axes_net)?;
            let mut class_strides = vec![1usize; class_shape_resized.len()];
            if !class_shape_resized.is_empty() {
                for i in (0..class_shape_resized.len() - 1).rev() {
                    class_strides[i] = class_strides[i + 1] * class_shape_resized[i + 1];
                }
            }
            let mut selected =
                Vec::<f32>::with_capacity(spatial_shape.iter().product::<usize>() * channels);
            for y in 0..spatial_shape[0] {
                for x in 0..spatial_shape[1] {
                    let mut base = 0usize;
                    let mut spatial_i = 0usize;
                    for (axis_i, axis) in setup.axes_net.chars().enumerate() {
                        let coord = if axis == 'C' {
                            0
                        } else {
                            let value = if spatial_i == 0 { y } else { x };
                            spatial_i += 1;
                            value
                        };
                        base += coord * class_strides[axis_i];
                    }
                    for class_i in 0..channels {
                        selected.push(class_resized[base + class_i * class_strides[setup.channel]]);
                    }
                }
            }
            let mut shape = spatial_shape.clone();
            shape.push(channels);
            (Some(selected), Some(shape))
        } else {
            (None, None)
        };

        let mut dist_shape = spatial_shape.clone();
        dist_shape.push(self.config.n_rays);
        Ok(StarDistPrediction {
            prob,
            prob_shape: spatial_shape,
            dist,
            dist_shape,
            prob_class,
            prob_class_shape,
        })
    }

    pub fn predict<F>(
        &self,
        img: &[f32],
        img_shape: &[usize],
        axes: Option<&str>,
        n_tiles: Option<&[usize]>,
        predict_direct: F,
    ) -> Result<StarDistPrediction, StarDistPredictError>
    where
        F: FnMut(&[f32], &[usize], &str) -> Result<StarDistDirectPrediction, StarDistPredictError>,
    {
        self._predict_generator(img, img_shape, axes, n_tiles, predict_direct)
    }

    pub fn _predict_sparse_generator<F>(
        &self,
        img: &[f32],
        img_shape: &[usize],
        prob_thresh: Option<f32>,
        axes: Option<&str>,
        n_tiles: Option<&[usize]>,
        b: usize,
        mut predict_direct: F,
    ) -> Result<StarDistSparsePrediction<2>, StarDistPredictError>
    where
        F: FnMut(&[f32], &[usize], &str) -> Result<StarDistDirectPrediction, StarDistPredictError>,
    {
        let prob_thresh = prob_thresh.unwrap_or(self.thresholds.prob);
        let setup = self._predict_setup(img, img_shape, axes, n_tiles)?;
        if setup.n_tiles.iter().product::<usize>() > 1 {
            return Err(StarDistPredictError::TiledPredictionUnsupported);
        }

        let results = predict_direct(&setup.x, &setup.x_shape, &setup.axes_net)?;
        if results.prob_shape.len() != setup.axes_net.len()
            || results.dist_shape.len() != setup.axes_net.len()
            || results.prob.len() != results.prob_shape.iter().product::<usize>()
            || results.dist.len() != results.dist_shape.iter().product::<usize>()
            || results.prob_shape[setup.channel] != 1
            || results.dist_shape[setup.channel] != self.config.n_rays
        {
            return Err(StarDistPredictError::OutputShapeMismatch);
        }

        let (prob_resized, prob_shape_resized) =
            setup
                .resizer
                .after(&results.prob, &results.prob_shape, &setup.axes_net)?;
        let (dist_resized, dist_shape_resized) =
            setup
                .resizer
                .after(&results.dist, &results.dist_shape, &setup.axes_net)?;

        let mut spatial_shape = Vec::<usize>::with_capacity(2);
        for (i, axis) in setup.axes_net.chars().enumerate() {
            if axis != 'C' {
                spatial_shape.push(prob_shape_resized[i]);
            }
        }
        if spatial_shape.len() != 2 {
            return Err(StarDistPredictError::OutputShapeMismatch);
        }

        let mut out_strides = vec![1usize; prob_shape_resized.len()];
        if !prob_shape_resized.is_empty() {
            for i in (0..prob_shape_resized.len() - 1).rev() {
                out_strides[i] = out_strides[i + 1] * prob_shape_resized[i + 1];
            }
        }
        let mut dist_strides = vec![1usize; dist_shape_resized.len()];
        if !dist_shape_resized.is_empty() {
            for i in (0..dist_shape_resized.len() - 1).rev() {
                dist_strides[i] = dist_strides[i + 1] * dist_shape_resized[i + 1];
            }
        }
        let mut prob = Vec::<f32>::with_capacity(spatial_shape.iter().product::<usize>());
        let mut dist = Vec::<f32>::with_capacity(prob.capacity() * self.config.n_rays);
        for y in 0..spatial_shape[0] {
            for x in 0..spatial_shape[1] {
                let mut prob_index = 0usize;
                let mut dist_base = 0usize;
                let mut spatial_i = 0usize;
                for (axis_i, axis) in setup.axes_net.chars().enumerate() {
                    let coord = if axis == 'C' {
                        0
                    } else {
                        let value = if spatial_i == 0 { y } else { x };
                        spatial_i += 1;
                        value
                    };
                    prob_index += coord * out_strides[axis_i];
                    dist_base += coord * dist_strides[axis_i];
                }
                prob.push(prob_resized[prob_index]);
                for ray in 0..self.config.n_rays {
                    let mut index = dist_base;
                    index += ray * dist_strides[setup.channel];
                    dist.push(dist_resized[index].max(1e-3));
                }
            }
        }

        let mask = crate::nms::_ind_prob_thresh(
            &prob,
            &spatial_shape,
            prob_thresh,
            Some(&[[b, b], [b, b]]),
        )?;
        let mut proba = Vec::<f32>::new();
        let mut dista = Vec::<f32>::new();
        let mut pointsa = Vec::<[f32; 2]>::new();
        let mut kept_indices = Vec::<usize>::new();
        for (i, keep) in mask.iter().enumerate() {
            if *keep {
                let y = i / spatial_shape[1];
                let x = i % spatial_shape[1];
                proba.push(prob[i]);
                dista
                    .extend_from_slice(&dist[i * self.config.n_rays..(i + 1) * self.config.n_rays]);
                pointsa.push([
                    (y * self.config.grid[0]) as f32,
                    (x * self.config.grid[1]) as f32,
                ]);
                kept_indices.push(i);
            }
        }

        let keep_after_resize =
            setup
                .resizer
                .filter_points::<2>(setup.x_shape.len(), &pointsa, &setup.axes_net)?;
        let mut prob_filtered = Vec::<f32>::with_capacity(keep_after_resize.len());
        let mut dist_filtered =
            Vec::<f32>::with_capacity(keep_after_resize.len() * self.config.n_rays);
        let mut points_filtered = Vec::<[f32; 2]>::with_capacity(keep_after_resize.len());
        let mut kept_filtered = Vec::<usize>::with_capacity(keep_after_resize.len());
        for i in keep_after_resize {
            prob_filtered.push(proba[i]);
            dist_filtered
                .extend_from_slice(&dista[i * self.config.n_rays..(i + 1) * self.config.n_rays]);
            points_filtered.push(pointsa[i]);
            kept_filtered.push(kept_indices[i]);
        }

        let (prob_class, prob_class_channels) = if _is_multiclass(self.config.n_classes) {
            let class_values = results
                .prob_class
                .ok_or(StarDistPredictError::MissingClassOutput)?;
            let class_shape = results
                .prob_class_shape
                .ok_or(StarDistPredictError::MissingClassOutput)?;
            if class_shape.len() != setup.axes_net.len()
                || class_values.len() != class_shape.iter().product::<usize>()
            {
                return Err(StarDistPredictError::OutputShapeMismatch);
            }
            let channels = class_shape[setup.channel];
            let (class_resized, class_shape_resized) =
                setup
                    .resizer
                    .after(&class_values, &class_shape, &setup.axes_net)?;
            let mut class_strides = vec![1usize; class_shape_resized.len()];
            if !class_shape_resized.is_empty() {
                for i in (0..class_shape_resized.len() - 1).rev() {
                    class_strides[i] = class_strides[i + 1] * class_shape_resized[i + 1];
                }
            }
            let mut selected = Vec::<f32>::with_capacity(kept_filtered.len() * channels);
            for i in kept_filtered {
                let y = i / spatial_shape[1];
                let x = i % spatial_shape[1];
                let mut base = 0usize;
                let mut spatial_i = 0usize;
                for (axis_i, axis) in setup.axes_net.chars().enumerate() {
                    let coord = if axis == 'C' {
                        0
                    } else {
                        let value = if spatial_i == 0 { y } else { x };
                        spatial_i += 1;
                        value
                    };
                    base += coord * class_strides[axis_i];
                }
                for class_i in 0..channels {
                    selected.push(class_resized[base + class_i * class_strides[setup.channel]]);
                }
            }
            (Some(selected), Some(channels))
        } else {
            (None, None)
        };

        Ok(StarDistSparsePrediction {
            prob: prob_filtered,
            dist: dist_filtered,
            points: points_filtered,
            prob_class,
            prob_class_channels,
        })
    }

    pub fn predict_sparse<F>(
        &self,
        img: &[f32],
        img_shape: &[usize],
        prob_thresh: Option<f32>,
        axes: Option<&str>,
        n_tiles: Option<&[usize]>,
        b: usize,
        predict_direct: F,
    ) -> Result<StarDistSparsePrediction<2>, StarDistPredictError>
    where
        F: FnMut(&[f32], &[usize], &str) -> Result<StarDistDirectPrediction, StarDistPredictError>,
    {
        self._predict_sparse_generator(
            img,
            img_shape,
            prob_thresh,
            axes,
            n_tiles,
            b,
            predict_direct,
        )
    }

    pub fn _checkpoint_callbacks(
        &self,
        logdir: Option<&str>,
        keras3: bool,
    ) -> Vec<StarDistCheckpointCallback> {
        let mut callbacks = Vec::<StarDistCheckpointCallback>::new();
        if let Some(logdir) = logdir {
            let suffix = if keras3 { ".weights.h5" } else { "" };
            if !self.config.train_checkpoint.is_empty() {
                let mut filepath = String::with_capacity(
                    logdir.len() + 1 + self.config.train_checkpoint.len() + suffix.len(),
                );
                filepath.push_str(logdir.trim_end_matches('/'));
                filepath.push('/');
                filepath.push_str(&self.config.train_checkpoint);
                filepath.push_str(suffix);
                if keras3 {
                    let len_without_suffix = filepath.len() - suffix.len();
                    filepath.truncate(len_without_suffix);
                }
                callbacks.push(StarDistCheckpointCallback {
                    filepath,
                    save_best_only: true,
                    save_weights_only: true,
                });
            }
            if !self.config.train_checkpoint_epoch.is_empty() {
                let mut filepath = String::with_capacity(
                    logdir.len() + 1 + self.config.train_checkpoint_epoch.len() + suffix.len(),
                );
                filepath.push_str(logdir.trim_end_matches('/'));
                filepath.push('/');
                filepath.push_str(&self.config.train_checkpoint_epoch);
                filepath.push_str(suffix);
                if keras3 {
                    let len_without_suffix = filepath.len() - suffix.len();
                    filepath.truncate(len_without_suffix);
                }
                callbacks.push(StarDistCheckpointCallback {
                    filepath,
                    save_best_only: false,
                    save_weights_only: true,
                });
            }
        }
        callbacks
    }

    pub fn _training_finished(&self, logdir: Option<&str>) -> Vec<StarDistTrainingFinishedAction> {
        let mut actions = Vec::<StarDistTrainingFinishedAction>::new();
        if let Some(logdir) = logdir {
            if !self.config.train_checkpoint_last.is_empty() {
                let mut filepath = String::with_capacity(
                    logdir.len() + 1 + self.config.train_checkpoint_last.len(),
                );
                filepath.push_str(logdir.trim_end_matches('/'));
                filepath.push('/');
                filepath.push_str(&self.config.train_checkpoint_last);
                actions.push(StarDistTrainingFinishedAction::SaveLastWeights { filepath });
            }
            if !self.config.train_checkpoint.is_empty() {
                actions.push(StarDistTrainingFinishedAction::LoadBestWeights {
                    prefer: self.config.train_checkpoint.clone(),
                });
            }
            if !self.config.train_checkpoint_epoch.is_empty() {
                let mut filepath = String::with_capacity(
                    logdir.len() + 1 + self.config.train_checkpoint_epoch.len(),
                );
                filepath.push_str(logdir.trim_end_matches('/'));
                filepath.push('/');
                filepath.push_str(&self.config.train_checkpoint_epoch);
                actions.push(StarDistTrainingFinishedAction::RemoveEpochWeights { filepath });
            }
        }
        actions
    }

    pub fn prepare_for_training(
        &self,
        optimizer: Option<&str>,
        basedir_is_some: bool,
    ) -> Result<StarDistPreparedTraining, StarDistTrainError> {
        let optimizer = optimizer.unwrap_or("Adam").to_string();
        let dist_loss = match self.config.train_dist_loss.as_str() {
            "mae" => StarDistTrainDistLoss::Mae,
            "mse" => StarDistTrainDistLoss::Mse,
            "iou" => StarDistTrainDistLoss::Iou,
            other => {
                return Err(StarDistTrainError::UnsupportedDistanceLoss(
                    other.to_string(),
                ));
            }
        };
        let expected_loss_weights = if _is_multiclass(self.config.n_classes) {
            3
        } else {
            2
        };
        if self.config.train_loss_weights.len() != expected_loss_weights {
            return Err(StarDistTrainError::InvalidLossWeights);
        }
        let expected_class_weights = if let Some(n_classes) = self.config.n_classes {
            n_classes + 1
        } else {
            2
        };
        if self.config.train_class_weights.len() != expected_class_weights {
            return Err(StarDistTrainError::InvalidClassWeights);
        }

        let mut losses = vec!["prob_loss".to_string(), "dist_loss".to_string()];
        if _is_multiclass(self.config.n_classes) {
            losses.push("prob_class_loss".to_string());
        }
        let metrics = vec![
            "prob:kld".to_string(),
            "dist:relevant_mae".to_string(),
            "dist:relevant_mse".to_string(),
            "dist:dist_iou_metric".to_string(),
        ];
        let mut callbacks = Vec::<StarDistTrainCallback>::new();
        if basedir_is_some {
            callbacks.push(StarDistTrainCallback::Checkpoint);
            if self.config.train_tensorboard {
                callbacks.push(StarDistTrainCallback::TensorBoard);
            }
        }
        callbacks.insert(0, StarDistTrainCallback::ReduceLrOnPlateau);
        let checkpoint_callbacks = if basedir_is_some {
            self._checkpoint_callbacks(Some("."), false)
        } else {
            Vec::new()
        };
        let tensorboard_log_dir = if basedir_is_some && self.config.train_tensorboard {
            Some("./logs".to_string())
        } else {
            None
        };
        let training_finished = if basedir_is_some {
            self._training_finished(Some("."))
        } else {
            Vec::new()
        };

        Ok(StarDistPreparedTraining {
            optimizer,
            learning_rate: self.config.train_learning_rate,
            dist_loss,
            losses,
            loss_weights: self.config.train_loss_weights.clone(),
            metrics,
            callbacks,
            checkpoint_callbacks,
            tensorboard_log_dir,
            training_finished,
            model_prepared: true,
        })
    }

    pub fn train(
        &self,
        x_len: usize,
        y_len: usize,
        validation_data_len: usize,
        validation_tuple_len: usize,
        classes: ClassesArg,
        validation_classes: Option<ClassesArg>,
        n_channel: Option<usize>,
        epochs: Option<usize>,
        steps_per_epoch: Option<usize>,
    ) -> Result<StarDist2DTrainSetup, StarDistTrainError> {
        if x_len == 0 || x_len != y_len {
            return Err(StarDistTrainError::EmptyOrMismatchedData);
        }
        let epochs = epochs.unwrap_or(self.config.train_epochs);
        let steps_per_epoch = steps_per_epoch.unwrap_or(self.config.train_steps_per_epoch);
        let train_length = epochs
            .checked_mul(steps_per_epoch)
            .ok_or(StarDistTrainError::LengthOverflow)?;

        let classes = _parse_classes_arg(self.config.n_classes, classes, x_len)?;
        let classes = if _is_multiclass(self.config.n_classes) {
            classes
        } else {
            None
        };

        let valid_validation_tuple_len = if _is_multiclass(self.config.n_classes) {
            validation_tuple_len == 2 || validation_tuple_len == 3
        } else {
            validation_tuple_len == 2
        };
        if !valid_validation_tuple_len {
            return Err(StarDistTrainError::InvalidValidationData);
        }
        let validation_classes = if _is_multiclass(self.config.n_classes) {
            let validation_classes = if validation_tuple_len == 2 {
                ClassesArg::Auto
            } else {
                validation_classes.unwrap_or(ClassesArg::Auto)
            };
            _parse_classes_arg(
                self.config.n_classes,
                validation_classes,
                validation_data_len,
            )?
        } else {
            None
        };

        let axes = self.config.axes.replace('C', "");
        let div_by = self._axes_div_by(&axes)?;
        let b = if self.config.train_shape_completion {
            self.config.train_completion_crop
        } else {
            0
        };
        for (p, d) in self.config.train_patch_size.iter().zip(div_by.iter()) {
            let effective = p
                .checked_sub(2 * b)
                .ok_or(StarDistTrainError::PatchSizeNotDivisible)?;
            if effective % d != 0 {
                return Err(StarDistTrainError::PatchSizeNotDivisible);
            }
        }

        let validation_n_take = self
            .config
            .train_n_val_patches
            .unwrap_or(validation_data_len);
        let prepared_training = self.prepare_for_training(None, true)?;
        let train_base = StarDistDataBase::new(
            n_channel,
            self.config.train_patch_size.to_vec(),
            self.config.grid.to_vec(),
            self.config.train_foreground_only,
            None,
            self.config.train_sample_cache,
        )?;
        let val_base = StarDistDataBase::new(
            n_channel,
            self.config.train_patch_size.to_vec(),
            self.config.grid.to_vec(),
            self.config.train_foreground_only,
            None,
            self.config.train_sample_cache,
        )?;
        let data_train = StarDistData2D::new(
            train_base,
            self.config.n_rays,
            self.config.n_classes,
            classes.clone(),
            self.config.train_shape_completion,
            self.config.train_completion_crop,
        )?;
        let data_val = StarDistData2D::new(
            val_base,
            self.config.n_rays,
            self.config.n_classes,
            validation_classes.clone(),
            self.config.train_shape_completion,
            self.config.train_completion_crop,
        )?;

        Ok(StarDist2DTrainSetup {
            epochs,
            steps_per_epoch,
            train_length,
            validation_n_take,
            classes,
            validation_classes,
            prepared_training,
            data_train,
            data_val,
        })
    }

    pub fn _instances_from_prediction(
        &self,
        img_shape: [usize; 2],
        prob: &[f32],
        prob_shape: [usize; 2],
        dist: &[f32],
        points: Option<&[[f32; 2]]>,
        prob_class: Option<(&[f32], usize)>,
        prob_thresh: Option<f32>,
        nms_thresh: Option<f32>,
        return_labels: bool,
        scale: Option<StarDist2DScale>,
        b: Option<[[usize; 2]; 2]>,
        use_bbox: bool,
        use_kdtree: bool,
    ) -> Result<StarDist2DInstances, StarDist2DPostprocessError> {
        let prob_thresh = prob_thresh.unwrap_or(self.thresholds.prob);
        let nms_thresh = nms_thresh.unwrap_or(self.thresholds.nms);

        let (mut pointsi, probi, disti, indsi) = if let Some(points) = points {
            let nms = crate::nms::non_maximum_suppression_sparse(
                dist,
                prob,
                points,
                self.config.n_rays,
                b,
                nms_thresh,
                use_bbox,
                use_kdtree,
            )?;
            (nms.points, nms.prob, nms.dist, Some(nms.indices))
        } else {
            let nms = crate::nms::non_maximum_suppression(
                dist,
                prob,
                prob_shape,
                self.config.n_rays,
                self.config.grid,
                b,
                nms_thresh,
                prob_thresh,
                use_bbox,
                use_kdtree,
            )?;
            (nms.points, nms.prob, nms.dist, None)
        };

        let rescale = if let Some(scale) = scale {
            if scale.y == 0.0 || scale.x == 0.0 {
                return Err(StarDist2DPostprocessError::InvalidScale);
            }
            let rescale = [1.0 / scale.y, 1.0 / scale.x];
            for p in &mut pointsi {
                p[0] *= rescale[0];
                p[1] *= rescale[1];
            }
            rescale
        } else {
            [1.0, 1.0]
        };

        let labels = if return_labels {
            Some(crate::geometry::polygons_to_label(
                &disti,
                &pointsi,
                img_shape,
                Some(&probi),
                f32::NEG_INFINITY,
                rescale,
            )?)
        } else {
            None
        };

        let coord = if pointsi.is_empty() {
            Array3::<f32>::zeros((0, 2, self.config.n_rays))
        } else {
            crate::geometry::dist_to_coord(&disti, &pointsi, self.config.n_rays, rescale)?
        };

        let (class_prob, class_prob_channels, class_id) = if let Some((prob_class, channels)) =
            prob_class
        {
            if channels == 0 {
                return Err(StarDist2DPostprocessError::ClassProbShapeMismatch);
            }
            let mut selected = Vec::with_capacity(pointsi.len() * channels);
            if let Some(indsi) = indsi {
                if prob_class.len() % channels != 0 || prob_class.len() / channels < prob.len() {
                    return Err(StarDist2DPostprocessError::ClassProbShapeMismatch);
                }
                for i in indsi {
                    let start = i * channels;
                    selected.extend_from_slice(&prob_class[start..start + channels]);
                }
            } else {
                if prob_class.len() != prob_shape[0] * prob_shape[1] * channels {
                    return Err(StarDist2DPostprocessError::ClassProbShapeMismatch);
                }
                for p in &pointsi {
                    let y = (p[0] as usize) / self.config.grid[0];
                    let x = (p[1] as usize) / self.config.grid[1];
                    let start = (y * prob_shape[1] + x) * channels;
                    selected.extend_from_slice(&prob_class[start..start + channels]);
                }
            }

            let mut ids = Vec::with_capacity(pointsi.len());
            for row in selected.chunks(channels) {
                let mut best = 0usize;
                let mut best_value = row[0];
                for (i, value) in row.iter().enumerate().skip(1) {
                    if *value > best_value {
                        best = i;
                        best_value = *value;
                    }
                }
                ids.push(best);
            }
            (Some(selected), Some(channels), Some(ids))
        } else {
            (None, None, None)
        };

        Ok(StarDist2DInstances {
            labels,
            coord,
            points: pointsi,
            prob: probi,
            class_prob,
            class_prob_channels,
            class_id,
        })
    }

    pub fn _predict_instances_generator<F>(
        &self,
        img: &[f32],
        img_shape: &[usize],
        axes: Option<&str>,
        sparse: bool,
        prob_thresh: Option<f32>,
        nms_thresh: Option<f32>,
        scale: Option<StarDist2DScale>,
        n_tiles: Option<&[usize]>,
        return_labels: bool,
        overlap_label: Option<u32>,
        return_predict: bool,
        b: usize,
        use_bbox: bool,
        use_kdtree: bool,
        mut predict_direct: F,
    ) -> Result<StarDist2DPredictInstancesResult, StarDistPredictError>
    where
        F: FnMut(&[f32], &[usize], &str) -> Result<StarDistDirectPrediction, StarDistPredictError>,
    {
        if overlap_label.is_some() {
            return Err(StarDistPredictError::OverlapLabel2DUnsupported);
        }
        let axes_normalized = self._normalize_axes(img_shape, axes)?;
        let axes_chars = axes_normalized.chars().collect::<Vec<_>>();
        let axes_net_chars = self.config.axes.chars().collect::<Vec<_>>();
        let mut img_shape_net = Vec::<usize>::with_capacity(axes_net_chars.len());
        for axis in &axes_net_chars {
            if let Some(pos) = axes_chars.iter().position(|candidate| candidate == axis) {
                img_shape_net.push(img_shape[pos]);
            } else if *axis == 'C' && self.config.n_channel_in == 1 {
                img_shape_net.push(1);
            } else {
                return Err(StarDistPredictError::OutputShapeMismatch);
            }
        }
        let mut shape_inst = Vec::<usize>::with_capacity(2);
        for (i, axis) in axes_net_chars.iter().enumerate() {
            if *axis != 'C' {
                shape_inst.push(img_shape_net[i]);
            }
        }
        if shape_inst.len() != 2 {
            return Err(StarDistPredictError::OutputShapeMismatch);
        }
        let shape_inst = [shape_inst[0], shape_inst[1]];

        if sparse && !return_predict {
            let sparse_prediction = self._predict_sparse_generator(
                img,
                img_shape,
                prob_thresh,
                axes,
                n_tiles,
                b,
                |x, x_shape, axes| predict_direct(x, x_shape, axes),
            )?;
            let prob_class = if let (Some(prob_class), Some(channels)) = (
                sparse_prediction.prob_class.as_ref(),
                sparse_prediction.prob_class_channels,
            ) {
                Some((prob_class.as_slice(), channels))
            } else {
                None
            };
            let instances = self._instances_from_prediction(
                shape_inst,
                &sparse_prediction.prob,
                [sparse_prediction.prob.len(), 1],
                &sparse_prediction.dist,
                Some(&sparse_prediction.points),
                prob_class,
                prob_thresh,
                nms_thresh,
                return_labels,
                scale,
                Some([[b, b], [b, b]]),
                use_bbox,
                use_kdtree,
            )?;
            Ok(StarDist2DPredictInstancesResult {
                instances,
                prediction: None,
            })
        } else {
            let prediction =
                self._predict_generator(img, img_shape, axes, n_tiles, predict_direct)?;
            if prediction.prob_shape.len() != 2 || prediction.dist_shape.len() != 3 {
                return Err(StarDistPredictError::OutputShapeMismatch);
            }
            let prob_class = if let (Some(prob_class), Some(prob_class_shape)) = (
                prediction.prob_class.as_ref(),
                prediction.prob_class_shape.as_ref(),
            ) {
                if prob_class_shape.len() != 3 {
                    return Err(StarDistPredictError::OutputShapeMismatch);
                }
                Some((prob_class.as_slice(), prob_class_shape[2]))
            } else {
                None
            };
            let instances = self._instances_from_prediction(
                shape_inst,
                &prediction.prob,
                [prediction.prob_shape[0], prediction.prob_shape[1]],
                &prediction.dist,
                None,
                prob_class,
                prob_thresh,
                nms_thresh,
                return_labels,
                scale,
                Some([[b, b], [b, b]]),
                use_bbox,
                use_kdtree,
            )?;
            Ok(StarDist2DPredictInstancesResult {
                instances,
                prediction: if return_predict {
                    Some(prediction)
                } else {
                    None
                },
            })
        }
    }

    pub fn predict_instances<F>(
        &self,
        img: &[f32],
        img_shape: &[usize],
        axes: Option<&str>,
        sparse: bool,
        prob_thresh: Option<f32>,
        nms_thresh: Option<f32>,
        scale: Option<StarDist2DScale>,
        n_tiles: Option<&[usize]>,
        return_labels: bool,
        overlap_label: Option<u32>,
        return_predict: bool,
        b: usize,
        use_bbox: bool,
        use_kdtree: bool,
        predict_direct: F,
    ) -> Result<StarDist2DPredictInstancesResult, StarDistPredictError>
    where
        F: FnMut(&[f32], &[usize], &str) -> Result<StarDistDirectPrediction, StarDistPredictError>,
    {
        self._predict_instances_generator(
            img,
            img_shape,
            axes,
            sparse,
            prob_thresh,
            nms_thresh,
            scale,
            n_tiles,
            return_labels,
            overlap_label,
            return_predict,
            b,
            use_bbox,
            use_kdtree,
            predict_direct,
        )
    }

    pub fn predict_instances_big<T, F>(
        &self,
        img: &[T],
        img_shape: &[usize],
        axes: &str,
        block_size: &[usize],
        min_overlap: &[usize],
        context: Option<&[usize]>,
        labels_out: Option<Vec<i32>>,
        mut predict_instances: F,
    ) -> Result<StarDistBigResult, StarDistPredictInstancesBigError>
    where
        T: Clone,
        F: FnMut(
            &[T],
            &[usize],
            &str,
        ) -> Result<StarDistBigPrediction, StarDistPredictInstancesBigError>,
    {
        let n = img_shape.len();
        if img.len() != img_shape.iter().product::<usize>() {
            return Err(StarDistPredictInstancesBigError::ShapeMismatch);
        }
        if axes.chars().count() != n
            || block_size.len() != n
            || min_overlap.len() != n
            || context.is_some_and(|context| context.len() != n)
        {
            return Err(StarDistPredictInstancesBigError::DimensionMismatch);
        }

        let mut grid = self._axes_div_by(axes)?;
        let axes_out = self
            .config
            .axes
            .chars()
            .filter(|axis| *axis != 'C')
            .collect::<String>();
        let mut shape_out = Vec::<usize>::with_capacity(axes_out.len());
        for axis_out in axes_out.chars() {
            let mut found = None;
            for (i, axis) in axes.chars().enumerate() {
                if axis == axis_out {
                    found = Some(img_shape[i]);
                    break;
                }
            }
            shape_out.push(found.ok_or(StarDistPredictInstancesBigError::DimensionMismatch)?);
        }

        let mut block_size = block_size.to_vec();
        let mut min_overlap = min_overlap.to_vec();
        let mut context = if let Some(context) = context {
            context.to_vec()
        } else {
            self._axes_tile_overlap(axes)?
        };
        for (i, axis) in axes.chars().enumerate() {
            if axis == 'C' {
                block_size[i] = img_shape[i];
                min_overlap[i] = 0;
                context[i] = 0;
                grid[i] = 1;
            }
        }
        for i in 0..n {
            block_size[i] = crate::big::_grid_divisible(grid[i], block_size[i])?;
            min_overlap[i] = crate::big::_grid_divisible(grid[i], min_overlap[i])?;
            context[i] = crate::big::_grid_divisible(grid[i], context[i])?;
        }

        let blocks =
            crate::BlockND::cover(img_shape, axes, &block_size, &min_overlap, &context, &grid)?;
        let mut labels_out = if let Some(labels_out) = labels_out {
            if labels_out.len() != shape_out.iter().product::<usize>() {
                return Err(StarDistPredictInstancesBigError::LabelsOutShapeMismatch);
            }
            Some(labels_out)
        } else {
            Some(vec![0i32; shape_out.iter().product::<usize>()])
        };
        let mut polys_all = crate::BigPolys {
            entries: Vec::new(),
        };
        let mut label_offset = 1u32;

        for block in &blocks {
            let (tile, tile_shape) = block.read(img, img_shape, Some(axes))?;
            let prediction = predict_instances(&tile, &tile_shape, axes)?;
            let (cropped_labels, cropped_shape) = block.crop_context(
                &prediction.labels,
                &prediction.labels_shape,
                Some(&axes_out),
            )?;
            let (filtered_labels, filtered_polys) = block.filter_objects_with_polys(
                &cropped_labels,
                &cropped_shape,
                &prediction.polys,
                Some(&axes_out),
            )?;

            let mut relabel_input = Vec::<u32>::with_capacity(filtered_labels.len());
            for label in &filtered_labels {
                if *label < 0 {
                    return Err(StarDistPredictInstancesBigError::NegativeLabel);
                }
                relabel_input.push(*label as u32);
            }
            let relabeled = crate::relabel_sequential(&relabel_input, label_offset)?;
            let relabeled_labels = relabeled
                .relabeled
                .iter()
                .map(|label| *label as i32)
                .collect::<Vec<_>>();

            if let Some(labels_out) = &mut labels_out {
                block.write(
                    labels_out,
                    &shape_out,
                    &relabeled_labels,
                    &cropped_shape,
                    Some(&axes_out),
                )?;
            }

            let mut object_count = relabeled
                .relabeled
                .iter()
                .copied()
                .filter(|label| *label > 0)
                .collect::<Vec<_>>();
            object_count.sort_unstable();
            object_count.dedup();
            let mut object_count = object_count.len();

            for (key, value) in filtered_polys.entries {
                if key == "prob" {
                    if let crate::BigPolysValue::F32 { shape, .. } = &value {
                        if !shape.is_empty() {
                            object_count = shape[0];
                        }
                    }
                }
                let is_object_key = crate::OBJECT_KEYS.contains(&key.as_str());
                let mut existing_index = None;
                for (i, (existing_key, _)) in polys_all.entries.iter().enumerate() {
                    if *existing_key == key {
                        existing_index = Some(i);
                        break;
                    }
                }
                if let Some(existing_index) = existing_index {
                    if is_object_key {
                        match (&mut polys_all.entries[existing_index].1, value) {
                            (
                                crate::BigPolysValue::F32 {
                                    values: values_out,
                                    shape: shape_out,
                                },
                                crate::BigPolysValue::F32 { values, shape },
                            ) => {
                                if shape_out.len() != shape.len()
                                    || shape_out[1..] != shape[1..]
                                    || values.len() != shape.iter().product::<usize>()
                                {
                                    return Err(crate::BigError::PolysShapeMismatch.into());
                                }
                                shape_out[0] += shape[0];
                                values_out.extend(values);
                            }
                            (
                                crate::BigPolysValue::I32 {
                                    values: values_out,
                                    shape: shape_out,
                                },
                                crate::BigPolysValue::I32 { values, shape },
                            ) => {
                                if shape_out.len() != shape.len()
                                    || shape_out[1..] != shape[1..]
                                    || values.len() != shape.iter().product::<usize>()
                                {
                                    return Err(crate::BigError::PolysShapeMismatch.into());
                                }
                                shape_out[0] += shape[0];
                                values_out.extend(values);
                            }
                            (
                                crate::BigPolysValue::Usize {
                                    values: values_out,
                                    shape: shape_out,
                                },
                                crate::BigPolysValue::Usize { values, shape },
                            ) => {
                                if shape_out.len() != shape.len()
                                    || shape_out[1..] != shape[1..]
                                    || values.len() != shape.iter().product::<usize>()
                                {
                                    return Err(crate::BigError::PolysShapeMismatch.into());
                                }
                                shape_out[0] += shape[0];
                                values_out.extend(values);
                            }
                            (
                                crate::BigPolysValue::Bool {
                                    values: values_out,
                                    shape: shape_out,
                                },
                                crate::BigPolysValue::Bool { values, shape },
                            ) => {
                                if shape_out.len() != shape.len()
                                    || shape_out[1..] != shape[1..]
                                    || values.len() != shape.iter().product::<usize>()
                                {
                                    return Err(crate::BigError::PolysShapeMismatch.into());
                                }
                                shape_out[0] += shape[0];
                                values_out.extend(values);
                            }
                            _ => return Err(crate::BigError::PolysShapeMismatch.into()),
                        }
                    }
                } else {
                    polys_all.entries.push((key, value));
                }
            }
            label_offset += object_count as u32;
        }

        Ok(StarDistBigResult {
            labels: labels_out,
            labels_shape: shape_out,
            polys: polys_all,
            n_blocks: blocks.len(),
        })
    }
}

impl StarDist3D {
    pub fn new(config: Config3D) -> Self {
        Self {
            config,
            thresholds: StarDistThresholds::default(),
        }
    }

    pub fn thresholds(&self) -> StarDistThresholds {
        self.thresholds
    }

    pub fn set_thresholds(&mut self, d: StarDistThresholds) -> Result<(), ThresholdsError> {
        if !d.prob.is_finite() || d.prob <= 0.0 || d.prob >= 1.0 {
            return Err(ThresholdsError::InvalidProb);
        }
        if !d.nms.is_finite() || d.nms <= 0.0 || d.nms >= 1.0 {
            return Err(ThresholdsError::InvalidNms);
        }
        self.thresholds = d;
        Ok(())
    }

    pub fn optimize_thresholds<F>(
        &mut self,
        y_val: &[&[u32]],
        yhat_prob: &[&[f32]],
        nms_threshs: &[f32],
        iou_threshs: &[f32],
        measure: crate::OptimizeThresholdMeasure,
        bracket: Option<[f32; 2]>,
        tol: f32,
        maxiter: usize,
        mut predict_instances: F,
    ) -> Result<StarDistThresholds, OptimizeThresholdsError>
    where
        F: FnMut(usize, f32, f32) -> Result<Vec<u32>, crate::UtilsError>,
    {
        if nms_threshs.is_empty() {
            return Err(OptimizeThresholdsError::EmptyNmsThresholds);
        }
        let mut opt_prob_thresh = 0.5f32;
        let mut opt_measure = f32::NEG_INFINITY;
        let mut opt_nms_thresh = 0.4f32;
        for nms_thresh in nms_threshs {
            let (prob_thresh, value) = crate::optimize_threshold(
                y_val,
                yhat_prob,
                *nms_thresh,
                measure,
                iou_threshs,
                bracket,
                tol,
                maxiter,
                |i, prob_thresh, nms_thresh| predict_instances(i, prob_thresh, nms_thresh),
            )?;
            if value > opt_measure {
                opt_prob_thresh = prob_thresh;
                opt_measure = value;
                opt_nms_thresh = *nms_thresh;
            }
        }
        let opt_threshs = StarDistThresholds {
            prob: opt_prob_thresh,
            nms: opt_nms_thresh,
        };
        self.set_thresholds(opt_threshs)?;
        Ok(opt_threshs)
    }

    pub fn _config_class(&self) -> ConfigClass {
        ConfigClass::Config3D
    }

    pub fn _build(&self) -> Result<StarDistBuildGraph, StarDistBuildError> {
        if self.config.backbone == "unet" {
            self._build_unet()
        } else if self.config.backbone == "resnet" {
            self._build_resnet()
        } else {
            Err(StarDistBuildError::UnsupportedBackbone)
        }
    }

    pub fn _build_unet(&self) -> Result<StarDistBuildGraph, StarDistBuildError> {
        if self.config.backbone != "unet" {
            return Err(StarDistBuildError::UnsupportedBackbone);
        }
        let mut layers = Vec::<StarDistBuildLayer>::new();
        layers.push(StarDistBuildLayer {
            name: "input".to_string(),
            kind: "Input".to_string(),
            filters: None,
            kernel: Vec::new(),
            pool: Vec::new(),
            activation: None,
            source: "input".to_string(),
        });

        let mut pooled = [1usize, 1usize, 1usize];
        let mut pooled_source = "input".to_string();
        let mut stage = 0usize;
        while pooled != self.config.grid {
            let mut pool = [1usize, 1usize, 1usize];
            for axis in 0..3 {
                if self.config.grid[axis] > pooled[axis] {
                    pool[axis] = 2;
                }
            }
            if pool == [1, 1, 1] {
                return Err(StarDistBuildError::InvalidGrid);
            }
            for axis in 0..3 {
                pooled[axis] *= pool[axis];
            }
            for conv in 0..self.config.unet_n_conv_per_depth {
                let name = format!("pre_grid_{stage}_conv_{conv}");
                layers.push(StarDistBuildLayer {
                    name: name.clone(),
                    kind: "Conv3D".to_string(),
                    filters: Some(self.config.unet_n_filter_base),
                    kernel: self.config.unet_kernel_size.to_vec(),
                    pool: Vec::new(),
                    activation: Some(self.config.unet_activation.clone()),
                    source: pooled_source,
                });
                pooled_source = name;
            }
            let name = format!("pre_grid_{stage}_max_pool");
            layers.push(StarDistBuildLayer {
                name: name.clone(),
                kind: "MaxPooling3D".to_string(),
                filters: None,
                kernel: Vec::new(),
                pool: pool.to_vec(),
                activation: None,
                source: pooled_source,
            });
            pooled_source = name;
            stage += 1;
        }

        layers.push(StarDistBuildLayer {
            name: "unet_block".to_string(),
            kind: "unet_block".to_string(),
            filters: Some(self.config.unet_n_filter_base),
            kernel: self.config.unet_kernel_size.to_vec(),
            pool: self.config.unet_pool.to_vec(),
            activation: Some(self.config.unet_activation.clone()),
            source: pooled_source,
        });
        let mut output_source = "unet_block".to_string();
        if self.config.net_conv_after_unet > 0 {
            layers.push(StarDistBuildLayer {
                name: "features".to_string(),
                kind: "Conv3D".to_string(),
                filters: Some(self.config.net_conv_after_unet),
                kernel: self.config.unet_kernel_size.to_vec(),
                pool: Vec::new(),
                activation: Some(self.config.unet_activation.clone()),
                source: output_source,
            });
            output_source = "features".to_string();
        }
        layers.push(StarDistBuildLayer {
            name: "prob".to_string(),
            kind: "Conv3D".to_string(),
            filters: Some(1),
            kernel: vec![1, 1, 1],
            pool: Vec::new(),
            activation: Some("sigmoid".to_string()),
            source: output_source.clone(),
        });
        layers.push(StarDistBuildLayer {
            name: "dist".to_string(),
            kind: "Conv3D".to_string(),
            filters: Some(self.config.n_rays),
            kernel: vec![1, 1, 1],
            pool: Vec::new(),
            activation: Some("linear".to_string()),
            source: output_source.clone(),
        });
        let mut outputs = vec!["prob".to_string(), "dist".to_string()];
        if _is_multiclass(self.config.n_classes) {
            let class_source = if self.config.net_conv_after_unet > 0 {
                layers.push(StarDistBuildLayer {
                    name: "features_class".to_string(),
                    kind: "Conv3D".to_string(),
                    filters: Some(self.config.net_conv_after_unet),
                    kernel: self.config.unet_kernel_size.to_vec(),
                    pool: Vec::new(),
                    activation: Some(self.config.unet_activation.clone()),
                    source: "unet_block".to_string(),
                });
                "features_class".to_string()
            } else {
                "unet_block".to_string()
            };
            layers.push(StarDistBuildLayer {
                name: "prob_class".to_string(),
                kind: "Conv3D".to_string(),
                filters: Some(self.config.n_classes.unwrap() + 1),
                kernel: vec![1, 1, 1],
                pool: Vec::new(),
                activation: Some("softmax".to_string()),
                source: class_source,
            });
            outputs.push("prob_class".to_string());
        }

        Ok(StarDistBuildGraph {
            ndim: 3,
            backbone: self.config.backbone.clone(),
            input_shape: self.config.net_input_shape.to_vec(),
            layers,
            outputs,
        })
    }

    pub fn _build_resnet(&self) -> Result<StarDistBuildGraph, StarDistBuildError> {
        if self.config.backbone != "resnet" {
            return Err(StarDistBuildError::UnsupportedBackbone);
        }
        let mut layers = Vec::<StarDistBuildLayer>::new();
        layers.push(StarDistBuildLayer {
            name: "input".to_string(),
            kind: "Input".to_string(),
            filters: None,
            kernel: Vec::new(),
            pool: Vec::new(),
            activation: None,
            source: "input".to_string(),
        });
        layers.push(StarDistBuildLayer {
            name: "conv3d_initial_7".to_string(),
            kind: "Conv3D".to_string(),
            filters: Some(self.config.resnet_n_filter_base),
            kernel: vec![7, 7, 7],
            pool: Vec::new(),
            activation: None,
            source: "input".to_string(),
        });
        layers.push(StarDistBuildLayer {
            name: "conv3d_initial_3".to_string(),
            kind: "Conv3D".to_string(),
            filters: Some(self.config.resnet_n_filter_base),
            kernel: vec![3, 3, 3],
            pool: Vec::new(),
            activation: None,
            source: "conv3d_initial_7".to_string(),
        });

        let mut pooled = [1usize, 1usize, 1usize];
        let mut n_filter = self.config.resnet_n_filter_base;
        let mut source = "conv3d_initial_3".to_string();
        for block in 0..self.config.resnet_n_blocks {
            let mut pool = [1usize, 1usize, 1usize];
            for axis in 0..3 {
                if self.config.grid[axis] > pooled[axis] {
                    pool[axis] = 2;
                }
            }
            for axis in 0..3 {
                pooled[axis] *= pool[axis];
            }
            if pool.iter().any(|value| *value > 1) {
                n_filter *= 2;
            }
            let name = format!("resnet_block_{block}");
            layers.push(StarDistBuildLayer {
                name: name.clone(),
                kind: "resnet_block".to_string(),
                filters: Some(n_filter),
                kernel: self.config.resnet_kernel_size.to_vec(),
                pool: pool.to_vec(),
                activation: Some(self.config.resnet_activation.clone()),
                source,
            });
            source = name;
        }
        let layer_base = source.clone();
        if self.config.net_conv_after_resnet > 0 {
            layers.push(StarDistBuildLayer {
                name: "features".to_string(),
                kind: "Conv3D".to_string(),
                filters: Some(self.config.net_conv_after_resnet),
                kernel: self.config.resnet_kernel_size.to_vec(),
                pool: Vec::new(),
                activation: Some(self.config.resnet_activation.clone()),
                source: layer_base.clone(),
            });
            source = "features".to_string();
        }
        layers.push(StarDistBuildLayer {
            name: "prob".to_string(),
            kind: "Conv3D".to_string(),
            filters: Some(1),
            kernel: vec![1, 1, 1],
            pool: Vec::new(),
            activation: Some("sigmoid".to_string()),
            source: source.clone(),
        });
        layers.push(StarDistBuildLayer {
            name: "dist".to_string(),
            kind: "Conv3D".to_string(),
            filters: Some(self.config.n_rays),
            kernel: vec![1, 1, 1],
            pool: Vec::new(),
            activation: Some("linear".to_string()),
            source: source.clone(),
        });
        let mut outputs = vec!["prob".to_string(), "dist".to_string()];
        if _is_multiclass(self.config.n_classes) {
            let class_source = if self.config.net_conv_after_resnet > 0 {
                layers.push(StarDistBuildLayer {
                    name: "features_class".to_string(),
                    kind: "Conv3D".to_string(),
                    filters: Some(self.config.net_conv_after_resnet),
                    kernel: self.config.resnet_kernel_size.to_vec(),
                    pool: Vec::new(),
                    activation: Some(self.config.resnet_activation.clone()),
                    source: layer_base,
                });
                "features_class".to_string()
            } else {
                layer_base
            };
            layers.push(StarDistBuildLayer {
                name: "prob_class".to_string(),
                kind: "Conv3D".to_string(),
                filters: Some(self.config.n_classes.unwrap() + 1),
                kernel: vec![1, 1, 1],
                pool: Vec::new(),
                activation: Some("softmax".to_string()),
                source: class_source,
            });
            outputs.push("prob_class".to_string());
        }

        Ok(StarDistBuildGraph {
            ndim: 3,
            backbone: self.config.backbone.clone(),
            input_shape: self.config.net_input_shape.to_vec(),
            layers,
            outputs,
        })
    }

    pub fn _axes_div_by(&self, query_axes: &str) -> Result<Vec<usize>, AxesDivByError> {
        if query_axes.is_empty() {
            return Err(AxesDivByError::EmptyAxes);
        }
        let mut normalized_axes = Vec::<char>::with_capacity(query_axes.len());
        for axis in query_axes.chars() {
            let axis = axis.to_ascii_uppercase();
            if normalized_axes.contains(&axis) {
                return Err(AxesDivByError::DuplicateAxis);
            }
            normalized_axes.push(axis);
        }

        if self.config.backbone == "unet" {
            let mut div_by = Vec::<(char, usize)>::new();
            for (i, axis) in self
                .config
                .axes
                .chars()
                .filter(|axis| *axis != 'C')
                .enumerate()
            {
                let pool = self.config.unet_pool[i].pow(self.config.unet_n_depth as u32);
                div_by.push((axis, pool * self.config.grid[i]));
            }
            let mut result = Vec::<usize>::with_capacity(normalized_axes.len());
            for axis in normalized_axes {
                let mut value = 1usize;
                for (div_axis, div_value) in &div_by {
                    if *div_axis == axis {
                        value = *div_value;
                        break;
                    }
                }
                result.push(value);
            }
            Ok(result)
        } else if self.config.backbone == "resnet" {
            let mut grid_by_axis = Vec::<(char, usize)>::new();
            for (i, axis) in self
                .config
                .axes
                .chars()
                .filter(|axis| *axis != 'C')
                .enumerate()
            {
                grid_by_axis.push((axis, self.config.grid[i]));
            }
            let mut result = Vec::<usize>::with_capacity(normalized_axes.len());
            for axis in normalized_axes {
                let mut value = 1usize;
                for (grid_axis, grid_value) in &grid_by_axis {
                    if *grid_axis == axis {
                        value = *grid_value;
                        break;
                    }
                }
                result.push(value);
            }
            Ok(result)
        } else {
            Err(AxesDivByError::UnsupportedBackbone)
        }
    }

    pub fn _axes_tile_overlap(&self, query_axes: &str) -> Result<Vec<usize>, AxesTileOverlapError> {
        if query_axes.is_empty() {
            return Err(AxesTileOverlapError::EmptyAxes);
        }
        let mut normalized_axes = Vec::<char>::with_capacity(query_axes.len());
        for axis in query_axes.chars() {
            let axis = axis.to_ascii_uppercase();
            if normalized_axes.contains(&axis) {
                return Err(AxesTileOverlapError::DuplicateAxis);
            }
            normalized_axes.push(axis);
        }

        let mut overlap = Vec::<(char, usize)>::new();
        if self.config.backbone == "resnet" {
            let mut pooled = [1usize; 3];
            let mut jump = [1usize; 3];
            let mut radius = [0usize; 3];
            for axis in 0..3 {
                radius[axis] += (7usize - 1) / 2 * jump[axis];
                radius[axis] += (3usize - 1) / 2 * jump[axis];
            }
            for _ in 0..self.config.resnet_n_blocks {
                let mut pool = [1usize; 3];
                for axis in 0..3 {
                    if self.config.grid[axis] > pooled[axis] {
                        pool[axis] = 2;
                    }
                    pooled[axis] *= pool[axis];
                    if pool[axis] > 1 {
                        radius[axis] += (self.config.resnet_kernel_size[axis] - 1) * jump[axis];
                    } else {
                        radius[axis] += (self.config.resnet_kernel_size[axis] - 1) / 2 * jump[axis];
                    }
                    jump[axis] *= pool[axis];
                    radius[axis] += (self.config.resnet_kernel_size[axis] - 1) / 2 * jump[axis];
                    radius[axis] += (self.config.resnet_kernel_size[axis] - 1) / 2 * jump[axis];
                }
            }
            for axis in 0..3 {
                if self.config.net_conv_after_resnet > 0 {
                    radius[axis] += (self.config.resnet_kernel_size[axis] - 1) / 2 * jump[axis];
                }
            }
            for (i, axis) in self
                .config
                .axes
                .chars()
                .filter(|axis| *axis != 'C')
                .enumerate()
            {
                overlap.push((axis, radius[i]));
            }
        } else if self.config.backbone == "unet" {
            for (i, axis) in self
                .config
                .axes
                .chars()
                .filter(|axis| *axis != 'C')
                .enumerate()
            {
                let grid = self.config.grid[i];
                let log_grid = if grid > 0 && grid.is_power_of_two() {
                    grid.trailing_zeros() as usize
                } else {
                    return Err(AxesTileOverlapError::Unavailable);
                };
                let n_depth = self.config.unet_n_depth + log_grid;
                let kern_size = self.config.unet_kernel_size[i];
                let pool_size = self.config.unet_pool[i];
                let value = match (n_depth, kern_size, pool_size) {
                    (1, 3, 1) => 6,
                    (1, 5, 1) => 12,
                    (1, 7, 1) => 18,
                    (2, 3, 1) => 10,
                    (2, 5, 1) => 20,
                    (2, 7, 1) => 30,
                    (3, 3, 1) => 14,
                    (3, 5, 1) => 28,
                    (3, 7, 1) => 42,
                    (4, 3, 1) => 18,
                    (4, 5, 1) => 36,
                    (4, 7, 1) => 54,
                    (5, 3, 1) => 22,
                    (5, 5, 1) => 44,
                    (5, 7, 1) => 66,
                    (1, 3, 2) => 9,
                    (1, 5, 2) => 17,
                    (1, 7, 2) => 25,
                    (2, 3, 2) => 22,
                    (2, 5, 2) => 43,
                    (2, 7, 2) => 62,
                    (3, 3, 2) => 46,
                    (3, 5, 2) => 92,
                    (3, 7, 2) => 138,
                    (4, 3, 2) => 94,
                    (4, 5, 2) => 188,
                    (4, 7, 2) => 282,
                    (5, 3, 2) => 190,
                    (5, 5, 2) => 380,
                    (5, 7, 2) => 570,
                    (1, 3, 4) => 14,
                    (1, 5, 4) => 27,
                    (1, 7, 4) => 38,
                    (2, 3, 4) => 58,
                    (2, 5, 4) => 116,
                    (2, 7, 4) => 158,
                    (3, 3, 4) => 234,
                    (3, 5, 4) => 468,
                    (3, 7, 4) => 638,
                    (4, 3, 4) => 938,
                    (4, 5, 4) => 1876,
                    (4, 7, 4) => 2558,
                    _ => return Err(AxesTileOverlapError::Unavailable),
                };
                overlap.push((axis, value));
            }
        } else {
            return Err(AxesTileOverlapError::UnsupportedBackbone);
        }

        let mut result = Vec::<usize>::with_capacity(normalized_axes.len());
        for axis in normalized_axes {
            let mut value = 0usize;
            for (overlap_axis, overlap_value) in &overlap {
                if *overlap_axis == axis {
                    value = *overlap_value;
                    break;
                }
            }
            result.push(value);
        }
        Ok(result)
    }

    pub fn _compute_receptive_field(
        &self,
        img_size: Option<&[usize]>,
    ) -> Result<Vec<(usize, usize)>, AxesTileOverlapError> {
        if let Some(img_size) = img_size {
            if img_size.len() != self.config.n_dim
                || img_size.iter().any(|size| !size.is_power_of_two())
            {
                return Err(AxesTileOverlapError::Unavailable);
            }
        }
        let axes = self.config.axes.replace('C', "");
        let overlap = self._axes_tile_overlap(&axes)?;
        let mut receptive_field = Vec::<(usize, usize)>::with_capacity(overlap.len());
        for value in overlap {
            receptive_field.push((value, value));
        }
        Ok(receptive_field)
    }

    pub fn _normalize_axes(
        &self,
        img_shape: &[usize],
        axes: Option<&str>,
    ) -> Result<String, AxesError> {
        let axes = if let Some(axes) = axes {
            axes.to_string()
        } else {
            if !self.config.axes.contains('C') {
                return Err(AxesError::MissingConfigChannelAxis);
            }
            if img_shape.len() == self.config.axes.len() - 1 && self.config.n_channel_in == 1 {
                self.config.axes.replace('C', "")
            } else {
                self.config.axes.clone()
            }
        };
        if axes.len() != img_shape.len() {
            return Err(AxesError::DimensionMismatch);
        }
        let mut normalized = String::with_capacity(axes.len());
        for axis in axes.chars() {
            let axis = axis.to_ascii_uppercase();
            if normalized.contains(axis) {
                return Err(AxesError::DuplicateAxis);
            }
            normalized.push(axis);
        }
        Ok(normalized)
    }

    pub fn _guess_n_tiles(
        &self,
        img_shape: &[usize],
        axes: Option<&str>,
    ) -> Result<Vec<usize>, AxesError> {
        let axes = self._normalize_axes(img_shape, axes)?;
        let mut spatial_shape = Vec::<usize>::with_capacity(self.config.n_dim);
        let mut channel_index = None;
        for (i, axis) in axes.chars().enumerate() {
            if axis == 'C' {
                channel_index = Some(i);
            } else {
                spatial_shape.push(img_shape[i]);
            }
        }
        let b = (self.config.train_batch_size as f32).powf(1.0 / self.config.n_dim as f32);
        let mut n_tiles = Vec::<usize>::with_capacity(spatial_shape.len());
        for (s, p) in spatial_shape
            .iter()
            .zip(self.config.train_patch_size.iter())
        {
            n_tiles.push((*s as f32 / (*p as f32 * b)).ceil() as usize);
        }
        if let Some(channel_index) = channel_index {
            n_tiles.insert(channel_index, 1);
        }
        Ok(n_tiles)
    }

    pub fn _predict_setup(
        &self,
        img: &[f32],
        img_shape: &[usize],
        axes: Option<&str>,
        n_tiles: Option<&[usize]>,
    ) -> Result<StarDistPredictSetup, StarDistPredictError> {
        if img.len() != img_shape.iter().product::<usize>() {
            return Err(StarDistPredictError::ShapeMismatch);
        }
        let n_tiles = if let Some(n_tiles) = n_tiles {
            if n_tiles.len() != img_shape.len() {
                return Err(StarDistPredictError::TilesDimensionMismatch);
            }
            let mut checked = Vec::<usize>::with_capacity(n_tiles.len());
            for value in n_tiles {
                if *value < 1 {
                    return Err(StarDistPredictError::InvalidTiles);
                }
                checked.push(*value);
            }
            checked
        } else {
            vec![1; img_shape.len()]
        };

        let axes = self._normalize_axes(img_shape, axes)?;
        let axes_net = self.config.axes.clone();
        let axes_net_chars = axes_net.chars().collect::<Vec<_>>();
        let axes_chars = axes.chars().collect::<Vec<_>>();
        let channel = axes_net_chars
            .iter()
            .position(|axis| *axis == 'C')
            .ok_or(StarDistPredictError::MissingChannelAxis)?;
        let axes_net_div_by = self._axes_div_by(&axes_net)?;

        let mut x_shape = Vec::<usize>::with_capacity(axes_net_chars.len());
        for axis in &axes_net_chars {
            if let Some(pos) = axes_chars.iter().position(|candidate| candidate == axis) {
                x_shape.push(img_shape[pos]);
            } else if *axis == 'C' && self.config.n_channel_in == 1 {
                x_shape.push(1);
            } else {
                return Err(StarDistPredictError::OutputShapeMismatch);
            }
        }
        if x_shape[channel] != self.config.n_channel_in {
            return Err(StarDistPredictError::ChannelMismatch);
        }

        let mut in_strides = vec![1usize; img_shape.len()];
        if !img_shape.is_empty() {
            for i in (0..img_shape.len() - 1).rev() {
                in_strides[i] = in_strides[i + 1] * img_shape[i + 1];
            }
        }
        let mut out_strides = vec![1usize; x_shape.len()];
        if !x_shape.is_empty() {
            for i in (0..x_shape.len() - 1).rev() {
                out_strides[i] = out_strides[i + 1] * x_shape[i + 1];
            }
        }
        let mut x = vec![0.0f32; x_shape.iter().product::<usize>()];
        for out_index in 0..x.len() {
            let mut remainder = out_index;
            let mut in_index = 0usize;
            for (axis_i, axis) in axes_net_chars.iter().enumerate() {
                let coord = remainder / out_strides[axis_i];
                remainder %= out_strides[axis_i];
                if let Some(pos) = axes_chars.iter().position(|candidate| candidate == axis) {
                    in_index += coord * in_strides[pos];
                }
            }
            x[out_index] = img[in_index];
        }

        let mut n_tiles_net = Vec::<usize>::with_capacity(axes_net_chars.len());
        for axis in &axes_net_chars {
            if let Some(pos) = axes_chars.iter().position(|candidate| candidate == axis) {
                n_tiles_net.push(n_tiles[pos]);
            } else {
                n_tiles_net.push(1);
            }
        }

        let grid = self.config.grid.to_vec();
        let grid_for_resizer = axes_net
            .chars()
            .filter(|axis| *axis != 'C')
            .zip(grid.iter().copied())
            .collect::<Vec<_>>();
        let mut resizer = StarDistPadAndCropResizer::new(grid_for_resizer, PadMode::Reflect, 0.0);
        let (x, x_shape) = resizer.before(&x, &x_shape, &axes_net, &axes_net_div_by)?;

        Ok(StarDistPredictSetup {
            x,
            x_shape,
            axes,
            axes_net,
            axes_net_div_by,
            n_tiles: n_tiles_net,
            grid,
            channel,
            resizer,
        })
    }

    pub fn _predict_generator<F>(
        &self,
        img: &[f32],
        img_shape: &[usize],
        axes: Option<&str>,
        n_tiles: Option<&[usize]>,
        mut predict_direct: F,
    ) -> Result<StarDistPrediction, StarDistPredictError>
    where
        F: FnMut(&[f32], &[usize], &str) -> Result<StarDistDirectPrediction, StarDistPredictError>,
    {
        let setup = self._predict_setup(img, img_shape, axes, n_tiles)?;
        if setup.n_tiles.iter().product::<usize>() > 1 {
            return Err(StarDistPredictError::TiledPredictionUnsupported);
        }

        let results = predict_direct(&setup.x, &setup.x_shape, &setup.axes_net)?;
        if results.prob_shape.len() != setup.axes_net.len()
            || results.dist_shape.len() != setup.axes_net.len()
            || results.prob.len() != results.prob_shape.iter().product::<usize>()
            || results.dist.len() != results.dist_shape.iter().product::<usize>()
            || results.prob_shape[setup.channel] != 1
            || results.dist_shape[setup.channel] != self.config.n_rays
        {
            return Err(StarDistPredictError::OutputShapeMismatch);
        }

        let (prob_resized, prob_shape_resized) =
            setup
                .resizer
                .after(&results.prob, &results.prob_shape, &setup.axes_net)?;
        let (dist_resized, dist_shape_resized) =
            setup
                .resizer
                .after(&results.dist, &results.dist_shape, &setup.axes_net)?;

        let mut spatial_shape = Vec::<usize>::with_capacity(self.config.n_dim);
        for (i, axis) in setup.axes_net.chars().enumerate() {
            if axis != 'C' {
                spatial_shape.push(prob_shape_resized[i]);
            }
        }
        if spatial_shape.len() != self.config.n_dim {
            return Err(StarDistPredictError::OutputShapeMismatch);
        }

        let mut prob_strides = vec![1usize; prob_shape_resized.len()];
        if !prob_shape_resized.is_empty() {
            for i in (0..prob_shape_resized.len() - 1).rev() {
                prob_strides[i] = prob_strides[i + 1] * prob_shape_resized[i + 1];
            }
        }
        let mut dist_strides = vec![1usize; dist_shape_resized.len()];
        if !dist_shape_resized.is_empty() {
            for i in (0..dist_shape_resized.len() - 1).rev() {
                dist_strides[i] = dist_strides[i + 1] * dist_shape_resized[i + 1];
            }
        }

        let mut prob = Vec::<f32>::with_capacity(spatial_shape.iter().product::<usize>());
        let mut dist = Vec::<f32>::with_capacity(prob.capacity() * self.config.n_rays);
        for z in 0..spatial_shape[0] {
            for y in 0..spatial_shape[1] {
                for x in 0..spatial_shape[2] {
                    let mut prob_index = 0usize;
                    let mut dist_base = 0usize;
                    let mut spatial_i = 0usize;
                    for (axis_i, axis) in setup.axes_net.chars().enumerate() {
                        let coord = if axis == 'C' {
                            0
                        } else {
                            let value = if spatial_i == 0 {
                                z
                            } else if spatial_i == 1 {
                                y
                            } else {
                                x
                            };
                            spatial_i += 1;
                            value
                        };
                        prob_index += coord * prob_strides[axis_i];
                        dist_base += coord * dist_strides[axis_i];
                    }
                    prob.push(prob_resized[prob_index]);
                    for ray in 0..self.config.n_rays {
                        dist.push(
                            (dist_resized[dist_base + ray * dist_strides[setup.channel]]).max(1e-3),
                        );
                    }
                }
            }
        }

        let (prob_class, prob_class_shape) = if _is_multiclass(self.config.n_classes) {
            let class_values = results
                .prob_class
                .ok_or(StarDistPredictError::MissingClassOutput)?;
            let class_shape = results
                .prob_class_shape
                .ok_or(StarDistPredictError::MissingClassOutput)?;
            if class_shape.len() != setup.axes_net.len()
                || class_values.len() != class_shape.iter().product::<usize>()
            {
                return Err(StarDistPredictError::OutputShapeMismatch);
            }
            let channels = class_shape[setup.channel];
            let (class_resized, class_shape_resized) =
                setup
                    .resizer
                    .after(&class_values, &class_shape, &setup.axes_net)?;
            let mut class_strides = vec![1usize; class_shape_resized.len()];
            if !class_shape_resized.is_empty() {
                for i in (0..class_shape_resized.len() - 1).rev() {
                    class_strides[i] = class_strides[i + 1] * class_shape_resized[i + 1];
                }
            }
            let mut selected =
                Vec::<f32>::with_capacity(spatial_shape.iter().product::<usize>() * channels);
            for z in 0..spatial_shape[0] {
                for y in 0..spatial_shape[1] {
                    for x in 0..spatial_shape[2] {
                        let mut base = 0usize;
                        let mut spatial_i = 0usize;
                        for (axis_i, axis) in setup.axes_net.chars().enumerate() {
                            let coord = if axis == 'C' {
                                0
                            } else {
                                let value = if spatial_i == 0 {
                                    z
                                } else if spatial_i == 1 {
                                    y
                                } else {
                                    x
                                };
                                spatial_i += 1;
                                value
                            };
                            base += coord * class_strides[axis_i];
                        }
                        for class_i in 0..channels {
                            selected
                                .push(class_resized[base + class_i * class_strides[setup.channel]]);
                        }
                    }
                }
            }
            let mut shape = spatial_shape.clone();
            shape.push(channels);
            (Some(selected), Some(shape))
        } else {
            (None, None)
        };

        let mut dist_shape = spatial_shape.clone();
        dist_shape.push(self.config.n_rays);
        Ok(StarDistPrediction {
            prob,
            prob_shape: spatial_shape,
            dist,
            dist_shape,
            prob_class,
            prob_class_shape,
        })
    }

    pub fn predict<F>(
        &self,
        img: &[f32],
        img_shape: &[usize],
        axes: Option<&str>,
        n_tiles: Option<&[usize]>,
        predict_direct: F,
    ) -> Result<StarDistPrediction, StarDistPredictError>
    where
        F: FnMut(&[f32], &[usize], &str) -> Result<StarDistDirectPrediction, StarDistPredictError>,
    {
        self._predict_generator(img, img_shape, axes, n_tiles, predict_direct)
    }

    pub fn _predict_sparse_generator<F>(
        &self,
        img: &[f32],
        img_shape: &[usize],
        prob_thresh: Option<f32>,
        axes: Option<&str>,
        n_tiles: Option<&[usize]>,
        b: usize,
        mut predict_direct: F,
    ) -> Result<StarDistSparsePrediction<3>, StarDistPredictError>
    where
        F: FnMut(&[f32], &[usize], &str) -> Result<StarDistDirectPrediction, StarDistPredictError>,
    {
        let prob_thresh = prob_thresh.unwrap_or(self.thresholds.prob);
        let setup = self._predict_setup(img, img_shape, axes, n_tiles)?;
        if setup.n_tiles.iter().product::<usize>() > 1 {
            return Err(StarDistPredictError::TiledPredictionUnsupported);
        }

        let results = predict_direct(&setup.x, &setup.x_shape, &setup.axes_net)?;
        if results.prob_shape.len() != setup.axes_net.len()
            || results.dist_shape.len() != setup.axes_net.len()
            || results.prob.len() != results.prob_shape.iter().product::<usize>()
            || results.dist.len() != results.dist_shape.iter().product::<usize>()
            || results.prob_shape[setup.channel] != 1
            || results.dist_shape[setup.channel] != self.config.n_rays
        {
            return Err(StarDistPredictError::OutputShapeMismatch);
        }

        let (prob_resized, prob_shape_resized) =
            setup
                .resizer
                .after(&results.prob, &results.prob_shape, &setup.axes_net)?;
        let (dist_resized, dist_shape_resized) =
            setup
                .resizer
                .after(&results.dist, &results.dist_shape, &setup.axes_net)?;

        let mut spatial_shape = Vec::<usize>::with_capacity(3);
        for (i, axis) in setup.axes_net.chars().enumerate() {
            if axis != 'C' {
                spatial_shape.push(prob_shape_resized[i]);
            }
        }
        if spatial_shape.len() != 3 {
            return Err(StarDistPredictError::OutputShapeMismatch);
        }

        let mut out_strides = vec![1usize; prob_shape_resized.len()];
        if !prob_shape_resized.is_empty() {
            for i in (0..prob_shape_resized.len() - 1).rev() {
                out_strides[i] = out_strides[i + 1] * prob_shape_resized[i + 1];
            }
        }
        let mut dist_strides = vec![1usize; dist_shape_resized.len()];
        if !dist_shape_resized.is_empty() {
            for i in (0..dist_shape_resized.len() - 1).rev() {
                dist_strides[i] = dist_strides[i + 1] * dist_shape_resized[i + 1];
            }
        }
        let mut prob = Vec::<f32>::with_capacity(spatial_shape.iter().product::<usize>());
        let mut dist = Vec::<f32>::with_capacity(prob.capacity() * self.config.n_rays);
        for z in 0..spatial_shape[0] {
            for y in 0..spatial_shape[1] {
                for x in 0..spatial_shape[2] {
                    let mut prob_index = 0usize;
                    let mut dist_base = 0usize;
                    let mut spatial_i = 0usize;
                    for (axis_i, axis) in setup.axes_net.chars().enumerate() {
                        let coord = if axis == 'C' {
                            0
                        } else {
                            let value = if spatial_i == 0 {
                                z
                            } else if spatial_i == 1 {
                                y
                            } else {
                                x
                            };
                            spatial_i += 1;
                            value
                        };
                        prob_index += coord * out_strides[axis_i];
                        dist_base += coord * dist_strides[axis_i];
                    }
                    prob.push(prob_resized[prob_index]);
                    for ray in 0..self.config.n_rays {
                        let mut index = dist_base;
                        index += ray * dist_strides[setup.channel];
                        dist.push(dist_resized[index].max(1e-3));
                    }
                }
            }
        }

        let mask = crate::nms::_ind_prob_thresh(
            &prob,
            &spatial_shape,
            prob_thresh,
            Some(&[[b, b], [b, b], [b, b]]),
        )?;
        let mut proba = Vec::<f32>::new();
        let mut dista = Vec::<f32>::new();
        let mut pointsa = Vec::<[f32; 3]>::new();
        let mut kept_indices = Vec::<usize>::new();
        for (i, keep) in mask.iter().enumerate() {
            if *keep {
                let z = i / (spatial_shape[1] * spatial_shape[2]);
                let rem = i % (spatial_shape[1] * spatial_shape[2]);
                let y = rem / spatial_shape[2];
                let x = rem % spatial_shape[2];
                proba.push(prob[i]);
                dista
                    .extend_from_slice(&dist[i * self.config.n_rays..(i + 1) * self.config.n_rays]);
                pointsa.push([
                    (z * self.config.grid[0]) as f32,
                    (y * self.config.grid[1]) as f32,
                    (x * self.config.grid[2]) as f32,
                ]);
                kept_indices.push(i);
            }
        }

        let keep_after_resize =
            setup
                .resizer
                .filter_points::<3>(setup.x_shape.len(), &pointsa, &setup.axes_net)?;
        let mut prob_filtered = Vec::<f32>::with_capacity(keep_after_resize.len());
        let mut dist_filtered =
            Vec::<f32>::with_capacity(keep_after_resize.len() * self.config.n_rays);
        let mut points_filtered = Vec::<[f32; 3]>::with_capacity(keep_after_resize.len());
        let mut kept_filtered = Vec::<usize>::with_capacity(keep_after_resize.len());
        for i in keep_after_resize {
            prob_filtered.push(proba[i]);
            dist_filtered
                .extend_from_slice(&dista[i * self.config.n_rays..(i + 1) * self.config.n_rays]);
            points_filtered.push(pointsa[i]);
            kept_filtered.push(kept_indices[i]);
        }

        let (prob_class, prob_class_channels) = if _is_multiclass(self.config.n_classes) {
            let class_values = results
                .prob_class
                .ok_or(StarDistPredictError::MissingClassOutput)?;
            let class_shape = results
                .prob_class_shape
                .ok_or(StarDistPredictError::MissingClassOutput)?;
            if class_shape.len() != setup.axes_net.len()
                || class_values.len() != class_shape.iter().product::<usize>()
            {
                return Err(StarDistPredictError::OutputShapeMismatch);
            }
            let channels = class_shape[setup.channel];
            let (class_resized, class_shape_resized) =
                setup
                    .resizer
                    .after(&class_values, &class_shape, &setup.axes_net)?;
            let mut class_strides = vec![1usize; class_shape_resized.len()];
            if !class_shape_resized.is_empty() {
                for i in (0..class_shape_resized.len() - 1).rev() {
                    class_strides[i] = class_strides[i + 1] * class_shape_resized[i + 1];
                }
            }
            let mut selected = Vec::<f32>::with_capacity(kept_filtered.len() * channels);
            for i in kept_filtered {
                let z = i / (spatial_shape[1] * spatial_shape[2]);
                let rem = i % (spatial_shape[1] * spatial_shape[2]);
                let y = rem / spatial_shape[2];
                let x = rem % spatial_shape[2];
                let mut base = 0usize;
                let mut spatial_i = 0usize;
                for (axis_i, axis) in setup.axes_net.chars().enumerate() {
                    let coord = if axis == 'C' {
                        0
                    } else {
                        let value = if spatial_i == 0 {
                            z
                        } else if spatial_i == 1 {
                            y
                        } else {
                            x
                        };
                        spatial_i += 1;
                        value
                    };
                    base += coord * class_strides[axis_i];
                }
                for class_i in 0..channels {
                    selected.push(class_resized[base + class_i * class_strides[setup.channel]]);
                }
            }
            (Some(selected), Some(channels))
        } else {
            (None, None)
        };

        Ok(StarDistSparsePrediction {
            prob: prob_filtered,
            dist: dist_filtered,
            points: points_filtered,
            prob_class,
            prob_class_channels,
        })
    }

    pub fn predict_sparse<F>(
        &self,
        img: &[f32],
        img_shape: &[usize],
        prob_thresh: Option<f32>,
        axes: Option<&str>,
        n_tiles: Option<&[usize]>,
        b: usize,
        predict_direct: F,
    ) -> Result<StarDistSparsePrediction<3>, StarDistPredictError>
    where
        F: FnMut(&[f32], &[usize], &str) -> Result<StarDistDirectPrediction, StarDistPredictError>,
    {
        self._predict_sparse_generator(
            img,
            img_shape,
            prob_thresh,
            axes,
            n_tiles,
            b,
            predict_direct,
        )
    }

    pub fn _checkpoint_callbacks(
        &self,
        logdir: Option<&str>,
        keras3: bool,
    ) -> Vec<StarDistCheckpointCallback> {
        let mut callbacks = Vec::<StarDistCheckpointCallback>::new();
        if let Some(logdir) = logdir {
            let suffix = if keras3 { ".weights.h5" } else { "" };
            if !self.config.train_checkpoint.is_empty() {
                let mut filepath = String::with_capacity(
                    logdir.len() + 1 + self.config.train_checkpoint.len() + suffix.len(),
                );
                filepath.push_str(logdir.trim_end_matches('/'));
                filepath.push('/');
                filepath.push_str(&self.config.train_checkpoint);
                filepath.push_str(suffix);
                if keras3 {
                    let len_without_suffix = filepath.len() - suffix.len();
                    filepath.truncate(len_without_suffix);
                }
                callbacks.push(StarDistCheckpointCallback {
                    filepath,
                    save_best_only: true,
                    save_weights_only: true,
                });
            }
            if !self.config.train_checkpoint_epoch.is_empty() {
                let mut filepath = String::with_capacity(
                    logdir.len() + 1 + self.config.train_checkpoint_epoch.len() + suffix.len(),
                );
                filepath.push_str(logdir.trim_end_matches('/'));
                filepath.push('/');
                filepath.push_str(&self.config.train_checkpoint_epoch);
                filepath.push_str(suffix);
                if keras3 {
                    let len_without_suffix = filepath.len() - suffix.len();
                    filepath.truncate(len_without_suffix);
                }
                callbacks.push(StarDistCheckpointCallback {
                    filepath,
                    save_best_only: false,
                    save_weights_only: true,
                });
            }
        }
        callbacks
    }

    pub fn _training_finished(&self, logdir: Option<&str>) -> Vec<StarDistTrainingFinishedAction> {
        let mut actions = Vec::<StarDistTrainingFinishedAction>::new();
        if let Some(logdir) = logdir {
            if !self.config.train_checkpoint_last.is_empty() {
                let mut filepath = String::with_capacity(
                    logdir.len() + 1 + self.config.train_checkpoint_last.len(),
                );
                filepath.push_str(logdir.trim_end_matches('/'));
                filepath.push('/');
                filepath.push_str(&self.config.train_checkpoint_last);
                actions.push(StarDistTrainingFinishedAction::SaveLastWeights { filepath });
            }
            if !self.config.train_checkpoint.is_empty() {
                actions.push(StarDistTrainingFinishedAction::LoadBestWeights {
                    prefer: self.config.train_checkpoint.clone(),
                });
            }
            if !self.config.train_checkpoint_epoch.is_empty() {
                let mut filepath = String::with_capacity(
                    logdir.len() + 1 + self.config.train_checkpoint_epoch.len(),
                );
                filepath.push_str(logdir.trim_end_matches('/'));
                filepath.push('/');
                filepath.push_str(&self.config.train_checkpoint_epoch);
                actions.push(StarDistTrainingFinishedAction::RemoveEpochWeights { filepath });
            }
        }
        actions
    }

    pub fn prepare_for_training(
        &self,
        optimizer: Option<&str>,
        basedir_is_some: bool,
    ) -> Result<StarDistPreparedTraining, StarDistTrainError> {
        let optimizer = optimizer.unwrap_or("Adam").to_string();
        let dist_loss = match self.config.train_dist_loss.as_str() {
            "mae" => StarDistTrainDistLoss::Mae,
            "mse" => StarDistTrainDistLoss::Mse,
            "iou" => StarDistTrainDistLoss::Iou,
            other => {
                return Err(StarDistTrainError::UnsupportedDistanceLoss(
                    other.to_string(),
                ));
            }
        };
        let expected_loss_weights = if _is_multiclass(self.config.n_classes) {
            3
        } else {
            2
        };
        if self.config.train_loss_weights.len() != expected_loss_weights {
            return Err(StarDistTrainError::InvalidLossWeights);
        }
        let expected_class_weights = if let Some(n_classes) = self.config.n_classes {
            n_classes + 1
        } else {
            2
        };
        if self.config.train_class_weights.len() != expected_class_weights {
            return Err(StarDistTrainError::InvalidClassWeights);
        }

        let mut losses = vec!["prob_loss".to_string(), "dist_loss".to_string()];
        if _is_multiclass(self.config.n_classes) {
            losses.push("prob_class_loss".to_string());
        }
        let metrics = vec![
            "prob:kld".to_string(),
            "dist:relevant_mae".to_string(),
            "dist:relevant_mse".to_string(),
            "dist:dist_iou_metric".to_string(),
        ];
        let mut callbacks = Vec::<StarDistTrainCallback>::new();
        if basedir_is_some {
            callbacks.push(StarDistTrainCallback::Checkpoint);
            if self.config.train_tensorboard {
                callbacks.push(StarDistTrainCallback::TensorBoard);
            }
        }
        callbacks.insert(0, StarDistTrainCallback::ReduceLrOnPlateau);
        let checkpoint_callbacks = if basedir_is_some {
            self._checkpoint_callbacks(Some("."), false)
        } else {
            Vec::new()
        };
        let tensorboard_log_dir = if basedir_is_some && self.config.train_tensorboard {
            Some("./logs".to_string())
        } else {
            None
        };
        let training_finished = if basedir_is_some {
            self._training_finished(Some("."))
        } else {
            Vec::new()
        };

        Ok(StarDistPreparedTraining {
            optimizer,
            learning_rate: self.config.train_learning_rate,
            dist_loss,
            losses,
            loss_weights: self.config.train_loss_weights.clone(),
            metrics,
            callbacks,
            checkpoint_callbacks,
            tensorboard_log_dir,
            training_finished,
            model_prepared: true,
        })
    }

    pub fn train(
        &self,
        x_len: usize,
        y_len: usize,
        validation_data_len: usize,
        validation_tuple_len: usize,
        classes: ClassesArg,
        validation_classes: Option<ClassesArg>,
        n_channel: Option<usize>,
        epochs: Option<usize>,
        steps_per_epoch: Option<usize>,
    ) -> Result<StarDist3DTrainSetup, StarDistTrainError> {
        if x_len == 0 || x_len != y_len {
            return Err(StarDistTrainError::EmptyOrMismatchedData);
        }
        let epochs = epochs.unwrap_or(self.config.train_epochs);
        let steps_per_epoch = steps_per_epoch.unwrap_or(self.config.train_steps_per_epoch);
        let train_length = epochs
            .checked_mul(steps_per_epoch)
            .ok_or(StarDistTrainError::LengthOverflow)?;

        let classes = _parse_classes_arg(self.config.n_classes, classes, x_len)?;
        let classes = if _is_multiclass(self.config.n_classes) {
            classes
        } else {
            None
        };

        let valid_validation_tuple_len = if _is_multiclass(self.config.n_classes) {
            validation_tuple_len == 2 || validation_tuple_len == 3
        } else {
            validation_tuple_len == 2
        };
        if !valid_validation_tuple_len {
            return Err(StarDistTrainError::InvalidValidationData);
        }
        let validation_classes = if _is_multiclass(self.config.n_classes) {
            let validation_classes = if validation_tuple_len == 2 {
                ClassesArg::Auto
            } else {
                validation_classes.unwrap_or(ClassesArg::Auto)
            };
            _parse_classes_arg(
                self.config.n_classes,
                validation_classes,
                validation_data_len,
            )?
        } else {
            None
        };

        let axes = self.config.axes.replace('C', "");
        let div_by = self._axes_div_by(&axes)?;
        for (p, d) in self.config.train_patch_size.iter().zip(div_by.iter()) {
            if p % d != 0 {
                return Err(StarDistTrainError::PatchSizeNotDivisible);
            }
        }

        let validation_n_take = self
            .config
            .train_n_val_patches
            .unwrap_or(validation_data_len);
        let prepared_training = self.prepare_for_training(None, true)?;
        let rays = crate::rays_from_json(&self.config.rays_json)?;
        let train_base = StarDistDataBase::new(
            n_channel,
            self.config.train_patch_size.to_vec(),
            self.config.grid.to_vec(),
            self.config.train_foreground_only,
            None,
            self.config.train_sample_cache,
        )?;
        let val_base = StarDistDataBase::new(
            n_channel,
            self.config.train_patch_size.to_vec(),
            self.config.grid.to_vec(),
            self.config.train_foreground_only,
            None,
            self.config.train_sample_cache,
        )?;
        let data_train = StarDistData3D::new(
            train_base,
            rays.clone(),
            Some(self.config.anisotropy),
            self.config.n_classes,
            classes.clone(),
        );
        let data_val = StarDistData3D::new(
            val_base,
            rays,
            Some(self.config.anisotropy),
            self.config.n_classes,
            validation_classes.clone(),
        );

        Ok(StarDist3DTrainSetup {
            epochs,
            steps_per_epoch,
            train_length,
            validation_n_take,
            classes,
            validation_classes,
            prepared_training,
            data_train,
            data_val,
        })
    }

    pub fn _instances_from_prediction(
        &self,
        img_shape: [usize; 3],
        prob: &[f32],
        prob_shape: [usize; 3],
        dist: &[f32],
        points: Option<&[[f32; 3]]>,
        prob_class: Option<(&[f32], usize)>,
        prob_thresh: Option<f32>,
        nms_thresh: Option<f32>,
        overlap_label: Option<u32>,
        return_labels: bool,
        scale: Option<StarDist3DScale>,
        b: Option<[[usize; 2]; 3]>,
        use_bbox: bool,
        use_kdtree: bool,
        use_gravity: bool,
        render_mode: crate::PolyhedronRenderMode,
    ) -> Result<StarDist3DInstances, StarDist3DPostprocessError> {
        let prob_thresh = prob_thresh.unwrap_or(self.thresholds.prob);
        let nms_thresh = nms_thresh.unwrap_or(self.thresholds.nms);
        let mut rays = crate::rays_from_json(&self.config.rays_json)?;

        let (mut pointsi, probi, disti, indsi) = if let Some(points) = points {
            let nms = crate::nms::non_maximum_suppression_3d_sparse(
                dist, prob, points, &rays, b, nms_thresh, use_bbox, use_kdtree, false,
            )?;
            (nms.points, nms.prob, nms.dist, Some(nms.indices))
        } else {
            let nms = crate::nms::non_maximum_suppression_3d(
                dist,
                prob,
                prob_shape,
                &rays,
                self.config.grid,
                b,
                nms_thresh,
                prob_thresh,
                use_bbox,
                use_kdtree,
                use_gravity,
            )?;
            (nms.points, nms.prob, nms.dist, None)
        };

        let (class_prob, class_prob_channels, class_id) = if let Some((prob_class, channels)) =
            prob_class
        {
            if channels == 0 {
                return Err(StarDist3DPostprocessError::ClassProbShapeMismatch);
            }
            let mut selected = Vec::with_capacity(pointsi.len() * channels);
            if let Some(indsi) = &indsi {
                if prob_class.len() % channels != 0 || prob_class.len() / channels < prob.len() {
                    return Err(StarDist3DPostprocessError::ClassProbShapeMismatch);
                }
                for i in indsi {
                    let start = i * channels;
                    selected.extend_from_slice(&prob_class[start..start + channels]);
                }
            } else {
                if prob_class.len() != prob_shape[0] * prob_shape[1] * prob_shape[2] * channels {
                    return Err(StarDist3DPostprocessError::ClassProbShapeMismatch);
                }
                for p in &pointsi {
                    let z = (p[0] as usize) / self.config.grid[0];
                    let y = (p[1] as usize) / self.config.grid[1];
                    let x = (p[2] as usize) / self.config.grid[2];
                    let start = ((z * prob_shape[1] + y) * prob_shape[2] + x) * channels;
                    selected.extend_from_slice(&prob_class[start..start + channels]);
                }
            }

            let mut ids = Vec::with_capacity(pointsi.len());
            for row in selected.chunks(channels) {
                let mut best = 0usize;
                let mut best_value = row[0];
                for (i, value) in row.iter().enumerate().skip(1) {
                    if *value > best_value {
                        best = i;
                        best_value = *value;
                    }
                }
                ids.push(best);
            }
            (Some(selected), Some(channels), Some(ids))
        } else {
            (None, None, None)
        };

        if let Some(scale) = scale {
            if scale.z == 0.0 || scale.y == 0.0 || scale.x == 0.0 {
                return Err(StarDist3DPostprocessError::InvalidScale);
            }
            let rescale = [1.0 / scale.z, 1.0 / scale.y, 1.0 / scale.x];
            for p in &mut pointsi {
                p[0] *= rescale[0];
                p[1] *= rescale[1];
                p[2] *= rescale[2];
            }
            rays = rays.copy(rescale);
        }

        let labels = if return_labels {
            Some(crate::geometry::polyhedron_to_label(
                &disti,
                &pointsi,
                &rays,
                img_shape,
                Some(&probi),
                f32::NEG_INFINITY,
                None,
                render_mode,
                overlap_label,
            )?)
        } else {
            None
        };

        let rays_vertices = rays.vertices.clone();
        let rays_faces = rays.faces.clone();
        Ok(StarDist3DInstances {
            labels,
            dist: disti,
            points: pointsi,
            prob: probi,
            rays,
            rays_vertices,
            rays_faces,
            class_prob,
            class_prob_channels,
            class_id,
        })
    }

    pub fn _predict_instances_generator<F>(
        &self,
        img: &[f32],
        img_shape: &[usize],
        axes: Option<&str>,
        sparse: bool,
        prob_thresh: Option<f32>,
        nms_thresh: Option<f32>,
        scale: Option<StarDist3DScale>,
        n_tiles: Option<&[usize]>,
        return_labels: bool,
        overlap_label: Option<u32>,
        return_predict: bool,
        b: usize,
        use_bbox: bool,
        use_kdtree: bool,
        use_gravity: bool,
        render_mode: crate::PolyhedronRenderMode,
        mut predict_direct: F,
    ) -> Result<StarDist3DPredictInstancesResult, StarDistPredictError>
    where
        F: FnMut(&[f32], &[usize], &str) -> Result<StarDistDirectPrediction, StarDistPredictError>,
    {
        let axes_normalized = self._normalize_axes(img_shape, axes)?;
        let axes_chars = axes_normalized.chars().collect::<Vec<_>>();
        let axes_net_chars = self.config.axes.chars().collect::<Vec<_>>();
        let mut img_shape_net = Vec::<usize>::with_capacity(axes_net_chars.len());
        for axis in &axes_net_chars {
            if let Some(pos) = axes_chars.iter().position(|candidate| candidate == axis) {
                img_shape_net.push(img_shape[pos]);
            } else if *axis == 'C' && self.config.n_channel_in == 1 {
                img_shape_net.push(1);
            } else {
                return Err(StarDistPredictError::OutputShapeMismatch);
            }
        }
        let mut shape_inst = Vec::<usize>::with_capacity(3);
        for (i, axis) in axes_net_chars.iter().enumerate() {
            if *axis != 'C' {
                shape_inst.push(img_shape_net[i]);
            }
        }
        if shape_inst.len() != 3 {
            return Err(StarDistPredictError::OutputShapeMismatch);
        }
        let shape_inst = [shape_inst[0], shape_inst[1], shape_inst[2]];

        if sparse && !return_predict {
            let sparse_prediction = self._predict_sparse_generator(
                img,
                img_shape,
                prob_thresh,
                axes,
                n_tiles,
                b,
                |x, x_shape, axes| predict_direct(x, x_shape, axes),
            )?;
            let prob_class = if let (Some(prob_class), Some(channels)) = (
                sparse_prediction.prob_class.as_ref(),
                sparse_prediction.prob_class_channels,
            ) {
                Some((prob_class.as_slice(), channels))
            } else {
                None
            };
            let instances = self._instances_from_prediction(
                shape_inst,
                &sparse_prediction.prob,
                [sparse_prediction.prob.len(), 1, 1],
                &sparse_prediction.dist,
                Some(&sparse_prediction.points),
                prob_class,
                prob_thresh,
                nms_thresh,
                overlap_label,
                return_labels,
                scale,
                Some([[b, b], [b, b], [b, b]]),
                use_bbox,
                use_kdtree,
                use_gravity,
                render_mode,
            )?;
            Ok(StarDist3DPredictInstancesResult {
                instances,
                prediction: None,
            })
        } else {
            let prediction =
                self._predict_generator(img, img_shape, axes, n_tiles, predict_direct)?;
            if prediction.prob_shape.len() != 3 || prediction.dist_shape.len() != 4 {
                return Err(StarDistPredictError::OutputShapeMismatch);
            }
            let prob_class = if let (Some(prob_class), Some(prob_class_shape)) = (
                prediction.prob_class.as_ref(),
                prediction.prob_class_shape.as_ref(),
            ) {
                if prob_class_shape.len() != 4 {
                    return Err(StarDistPredictError::OutputShapeMismatch);
                }
                Some((prob_class.as_slice(), prob_class_shape[3]))
            } else {
                None
            };
            let instances = self._instances_from_prediction(
                shape_inst,
                &prediction.prob,
                [
                    prediction.prob_shape[0],
                    prediction.prob_shape[1],
                    prediction.prob_shape[2],
                ],
                &prediction.dist,
                None,
                prob_class,
                prob_thresh,
                nms_thresh,
                overlap_label,
                return_labels,
                scale,
                Some([[b, b], [b, b], [b, b]]),
                use_bbox,
                use_kdtree,
                use_gravity,
                render_mode,
            )?;
            Ok(StarDist3DPredictInstancesResult {
                instances,
                prediction: if return_predict {
                    Some(prediction)
                } else {
                    None
                },
            })
        }
    }

    pub fn predict_instances<F>(
        &self,
        img: &[f32],
        img_shape: &[usize],
        axes: Option<&str>,
        sparse: bool,
        prob_thresh: Option<f32>,
        nms_thresh: Option<f32>,
        scale: Option<StarDist3DScale>,
        n_tiles: Option<&[usize]>,
        return_labels: bool,
        overlap_label: Option<u32>,
        return_predict: bool,
        b: usize,
        use_bbox: bool,
        use_kdtree: bool,
        use_gravity: bool,
        render_mode: crate::PolyhedronRenderMode,
        predict_direct: F,
    ) -> Result<StarDist3DPredictInstancesResult, StarDistPredictError>
    where
        F: FnMut(&[f32], &[usize], &str) -> Result<StarDistDirectPrediction, StarDistPredictError>,
    {
        self._predict_instances_generator(
            img,
            img_shape,
            axes,
            sparse,
            prob_thresh,
            nms_thresh,
            scale,
            n_tiles,
            return_labels,
            overlap_label,
            return_predict,
            b,
            use_bbox,
            use_kdtree,
            use_gravity,
            render_mode,
            predict_direct,
        )
    }

    pub fn predict_instances_big<T, F>(
        &self,
        img: &[T],
        img_shape: &[usize],
        axes: &str,
        block_size: &[usize],
        min_overlap: &[usize],
        context: Option<&[usize]>,
        labels_out: Option<Vec<i32>>,
        mut predict_instances: F,
    ) -> Result<StarDistBigResult, StarDistPredictInstancesBigError>
    where
        T: Clone,
        F: FnMut(
            &[T],
            &[usize],
            &str,
        ) -> Result<StarDistBigPrediction, StarDistPredictInstancesBigError>,
    {
        let n = img_shape.len();
        if img.len() != img_shape.iter().product::<usize>() {
            return Err(StarDistPredictInstancesBigError::ShapeMismatch);
        }
        if axes.chars().count() != n
            || block_size.len() != n
            || min_overlap.len() != n
            || context.is_some_and(|context| context.len() != n)
        {
            return Err(StarDistPredictInstancesBigError::DimensionMismatch);
        }

        let mut grid = self._axes_div_by(axes)?;
        let axes_out = self
            .config
            .axes
            .chars()
            .filter(|axis| *axis != 'C')
            .collect::<String>();
        let mut shape_out = Vec::<usize>::with_capacity(axes_out.len());
        for axis_out in axes_out.chars() {
            let mut found = None;
            for (i, axis) in axes.chars().enumerate() {
                if axis == axis_out {
                    found = Some(img_shape[i]);
                    break;
                }
            }
            shape_out.push(found.ok_or(StarDistPredictInstancesBigError::DimensionMismatch)?);
        }

        let mut block_size = block_size.to_vec();
        let mut min_overlap = min_overlap.to_vec();
        let mut context = if let Some(context) = context {
            context.to_vec()
        } else {
            self._axes_tile_overlap(axes)?
        };
        for (i, axis) in axes.chars().enumerate() {
            if axis == 'C' {
                block_size[i] = img_shape[i];
                min_overlap[i] = 0;
                context[i] = 0;
                grid[i] = 1;
            }
        }
        for i in 0..n {
            block_size[i] = crate::big::_grid_divisible(grid[i], block_size[i])?;
            min_overlap[i] = crate::big::_grid_divisible(grid[i], min_overlap[i])?;
            context[i] = crate::big::_grid_divisible(grid[i], context[i])?;
        }

        let blocks =
            crate::BlockND::cover(img_shape, axes, &block_size, &min_overlap, &context, &grid)?;
        let mut labels_out = if let Some(labels_out) = labels_out {
            if labels_out.len() != shape_out.iter().product::<usize>() {
                return Err(StarDistPredictInstancesBigError::LabelsOutShapeMismatch);
            }
            Some(labels_out)
        } else {
            Some(vec![0i32; shape_out.iter().product::<usize>()])
        };
        let mut polys_all = crate::BigPolys {
            entries: Vec::new(),
        };
        let mut label_offset = 1u32;

        for block in &blocks {
            let (tile, tile_shape) = block.read(img, img_shape, Some(axes))?;
            let prediction = predict_instances(&tile, &tile_shape, axes)?;
            let (cropped_labels, cropped_shape) = block.crop_context(
                &prediction.labels,
                &prediction.labels_shape,
                Some(&axes_out),
            )?;
            let (filtered_labels, filtered_polys) = block.filter_objects_with_polys(
                &cropped_labels,
                &cropped_shape,
                &prediction.polys,
                Some(&axes_out),
            )?;

            let mut relabel_input = Vec::<u32>::with_capacity(filtered_labels.len());
            for label in &filtered_labels {
                if *label < 0 {
                    return Err(StarDistPredictInstancesBigError::NegativeLabel);
                }
                relabel_input.push(*label as u32);
            }
            let relabeled = crate::relabel_sequential(&relabel_input, label_offset)?;
            let relabeled_labels = relabeled
                .relabeled
                .iter()
                .map(|label| *label as i32)
                .collect::<Vec<_>>();

            if let Some(labels_out) = &mut labels_out {
                block.write(
                    labels_out,
                    &shape_out,
                    &relabeled_labels,
                    &cropped_shape,
                    Some(&axes_out),
                )?;
            }

            let mut object_count = relabeled
                .relabeled
                .iter()
                .copied()
                .filter(|label| *label > 0)
                .collect::<Vec<_>>();
            object_count.sort_unstable();
            object_count.dedup();
            let mut object_count = object_count.len();

            for (key, value) in filtered_polys.entries {
                if key == "prob" {
                    if let crate::BigPolysValue::F32 { shape, .. } = &value {
                        if !shape.is_empty() {
                            object_count = shape[0];
                        }
                    }
                }
                let is_object_key = crate::OBJECT_KEYS.contains(&key.as_str());
                let mut existing_index = None;
                for (i, (existing_key, _)) in polys_all.entries.iter().enumerate() {
                    if *existing_key == key {
                        existing_index = Some(i);
                        break;
                    }
                }
                if let Some(existing_index) = existing_index {
                    if is_object_key {
                        match (&mut polys_all.entries[existing_index].1, value) {
                            (
                                crate::BigPolysValue::F32 {
                                    values: values_out,
                                    shape: shape_out,
                                },
                                crate::BigPolysValue::F32 { values, shape },
                            ) => {
                                if shape_out.len() != shape.len()
                                    || shape_out[1..] != shape[1..]
                                    || values.len() != shape.iter().product::<usize>()
                                {
                                    return Err(crate::BigError::PolysShapeMismatch.into());
                                }
                                shape_out[0] += shape[0];
                                values_out.extend(values);
                            }
                            (
                                crate::BigPolysValue::I32 {
                                    values: values_out,
                                    shape: shape_out,
                                },
                                crate::BigPolysValue::I32 { values, shape },
                            ) => {
                                if shape_out.len() != shape.len()
                                    || shape_out[1..] != shape[1..]
                                    || values.len() != shape.iter().product::<usize>()
                                {
                                    return Err(crate::BigError::PolysShapeMismatch.into());
                                }
                                shape_out[0] += shape[0];
                                values_out.extend(values);
                            }
                            (
                                crate::BigPolysValue::Usize {
                                    values: values_out,
                                    shape: shape_out,
                                },
                                crate::BigPolysValue::Usize { values, shape },
                            ) => {
                                if shape_out.len() != shape.len()
                                    || shape_out[1..] != shape[1..]
                                    || values.len() != shape.iter().product::<usize>()
                                {
                                    return Err(crate::BigError::PolysShapeMismatch.into());
                                }
                                shape_out[0] += shape[0];
                                values_out.extend(values);
                            }
                            (
                                crate::BigPolysValue::Bool {
                                    values: values_out,
                                    shape: shape_out,
                                },
                                crate::BigPolysValue::Bool { values, shape },
                            ) => {
                                if shape_out.len() != shape.len()
                                    || shape_out[1..] != shape[1..]
                                    || values.len() != shape.iter().product::<usize>()
                                {
                                    return Err(crate::BigError::PolysShapeMismatch.into());
                                }
                                shape_out[0] += shape[0];
                                values_out.extend(values);
                            }
                            _ => return Err(crate::BigError::PolysShapeMismatch.into()),
                        }
                    }
                } else {
                    polys_all.entries.push((key, value));
                }
            }
            label_offset += object_count as u32;
        }

        Ok(StarDistBigResult {
            labels: labels_out,
            labels_shape: shape_out,
            polys: polys_all,
            n_blocks: blocks.len(),
        })
    }
}

#[cfg(feature = "burn")]
pub mod burn {
    use ::burn::module::{AutodiffModule, Module, Param};
    use ::burn::nn::conv::{Conv2d, Conv2dConfig, Conv3d, Conv3dConfig};
    use ::burn::nn::interpolate::{Interpolate2d, Interpolate2dConfig, InterpolateMode};
    use ::burn::nn::pool::{MaxPool2d, MaxPool2dConfig};
    use ::burn::nn::{PaddingConfig2d, PaddingConfig3d};
    use ::burn::optim::{GradientsParams, LearningRate, Optimizer};
    use ::burn::prelude::*;
    use ::burn::record::DefaultRecorder;
    use ::burn::tensor::TensorData;
    use ::burn::tensor::activation::{relu, sigmoid, softmax};
    use ::burn::tensor::backend::AutodiffBackend;
    use ::burn::tensor::ops::PadMode as BurnPadMode;
    use std::io::Write;
    use std::path::PathBuf;

    #[derive(Debug, thiserror::Error)]
    pub enum BurnWeightError {
        #[error("missing Keras weight tensor {0}")]
        Missing(String),
        #[error("wrong shape for {name}: expected {expected:?}, got {actual:?}")]
        Shape {
            name: String,
            expected: Vec<usize>,
            actual: Vec<usize>,
        },
    }

    #[derive(Debug, thiserror::Error)]
    pub enum BurnTrainError {
        #[error("batch tensor shape does not match its buffer length")]
        ShapeMismatch,
        #[error("training requires at least one batch")]
        EmptyTrainingBatches,
        #[error("epochs * steps_per_epoch overflowed")]
        LengthOverflow,
        #[error("could not read scalar loss from Burn tensor")]
        LossScalarUnavailable,
        #[error("train_loss_weights must contain at least probability and distance weights")]
        InvalidLossWeights,
        #[error("multiclass training requires class targets and class prediction output")]
        MissingClassLossInput,
        #[error("train_class_weights length must match n_classes + 1")]
        InvalidClassWeights,
        #[error("unsupported configured distance loss {0}")]
        UnsupportedDistanceLoss(String),
        #[error("TensorBoard image values must match a positive 2D image shape")]
        InvalidTensorBoardImage,
        #[error(transparent)]
        Data(#[from] crate::StarDistDataError),
        #[error(transparent)]
        Recorder(#[from] ::burn::record::RecorderError),
        #[error(transparent)]
        Io(#[from] std::io::Error),
    }

    #[derive(Module, Debug)]
    pub struct StarDist2D<B: Backend> {
        pub config: crate::Config2D,
        pub conv2d_1: Conv2d<B>,
        pub conv2d_2: Conv2d<B>,
        pub max_pooling2d_1: MaxPool2d,
        pub down_level_0_no_0: Conv2d<B>,
        pub down_level_0_no_1: Conv2d<B>,
        pub max_0: MaxPool2d,
        pub down_level_1_no_0: Conv2d<B>,
        pub down_level_1_no_1: Conv2d<B>,
        pub max_1: MaxPool2d,
        pub down_level_2_no_0: Conv2d<B>,
        pub down_level_2_no_1: Conv2d<B>,
        pub max_2: MaxPool2d,
        pub middle_0: Conv2d<B>,
        pub middle_2: Conv2d<B>,
        pub up_sampling2d_1: Interpolate2d,
        pub up_level_2_no_0: Conv2d<B>,
        pub up_level_2_no_2: Conv2d<B>,
        pub up_sampling2d_2: Interpolate2d,
        pub up_level_1_no_0: Conv2d<B>,
        pub up_level_1_no_2: Conv2d<B>,
        pub up_sampling2d_3: Interpolate2d,
        pub up_level_0_no_0: Conv2d<B>,
        pub up_level_0_no_2: Conv2d<B>,
        pub features: Conv2d<B>,
        pub prob: Conv2d<B>,
        pub dist: Conv2d<B>,
        pub features_class: Option<Conv2d<B>>,
        pub prob_class: Option<Conv2d<B>>,
    }

    #[derive(Clone, Debug)]
    pub struct StarDist2DOutputs<B: Backend> {
        pub prob: Tensor<B, 4>,
        pub dist: Tensor<B, 4>,
        pub prob_class: Option<Tensor<B, 4>>,
    }

    #[derive(Module, Debug)]
    pub struct StarDist3D<B: Backend> {
        pub config: crate::config::Config3D,
        pub conv3d_1: Conv3d<B>,
        pub conv3d_2: Conv3d<B>,
        pub conv3d_3: Conv3d<B>,
        pub conv3d_4: Conv3d<B>,
        pub conv3d_5: Conv3d<B>,
        pub conv3d_6: Conv3d<B>,
        pub conv3d_7: Conv3d<B>,
        pub conv3d_8: Conv3d<B>,
        pub conv3d_9: Conv3d<B>,
        pub conv3d_10: Conv3d<B>,
        pub conv3d_11: Conv3d<B>,
        pub conv3d_12: Conv3d<B>,
        pub conv3d_13: Conv3d<B>,
        pub conv3d_14: Conv3d<B>,
        pub conv3d_15: Conv3d<B>,
        pub features: Conv3d<B>,
        pub prob: Conv3d<B>,
        pub dist: Conv3d<B>,
        pub features_class: Option<Conv3d<B>>,
        pub prob_class: Option<Conv3d<B>>,
    }

    #[derive(Clone, Debug)]
    pub struct StarDist3DOutputs<B: Backend> {
        pub prob: Tensor<B, 5>,
        pub dist: Tensor<B, 5>,
        pub prob_class: Option<Tensor<B, 5>>,
    }

    #[derive(Clone, Debug)]
    pub struct StarDistData2DBatchTensors<B: Backend> {
        pub x: Tensor<B, 4>,
        pub prob: Tensor<B, 4>,
        pub dist: Tensor<B, 4>,
        pub prob_class: Option<Tensor<B, 4>>,
    }

    #[derive(Clone, Debug)]
    pub struct StarDistData3DBatchTensors<B: Backend> {
        pub x: Tensor<B, 5>,
        pub prob: Tensor<B, 5>,
        pub dist: Tensor<B, 5>,
        pub prob_class: Option<Tensor<B, 5>>,
    }

    #[derive(Clone, Debug, PartialEq)]
    pub struct StarDistBurnTrainHistory {
        pub loss: Vec<f32>,
        pub val_loss: Vec<f32>,
        pub learning_rates: Vec<LearningRate>,
        pub checkpoint_files: Vec<String>,
        pub log_files: Vec<String>,
        pub event_files: Vec<String>,
    }

    #[derive(Clone, Debug, PartialEq, Eq)]
    pub struct StarDistBurnCheckpointConfig {
        pub best: Option<String>,
        pub epoch: Option<String>,
        pub last: Option<String>,
    }

    #[derive(Clone, Debug, PartialEq, Eq)]
    pub struct StarDistBurnTensorBoardConfig {
        pub log_dir: String,
    }

    pub fn write_tensorboard_scalar_events(
        log_dir: &str,
        loss: &[f32],
        val_loss: &[f32],
    ) -> Result<String, BurnTrainError> {
        write_tensorboard_scalar_events_with_learning_rates(log_dir, loss, val_loss, None)
    }

    pub fn write_tensorboard_scalar_events_with_learning_rates(
        log_dir: &str,
        loss: &[f32],
        val_loss: &[f32],
        learning_rates: Option<&[LearningRate]>,
    ) -> Result<String, BurnTrainError> {
        std::fs::create_dir_all(log_dir)?;
        let wall_time = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|duration| duration.as_secs_f64())
            .unwrap_or(0.0);
        let event_path = PathBuf::from(log_dir).join(format!(
            "events.out.tfevents.{}.stardist-rs",
            wall_time.trunc() as u64
        ));
        let mut file = std::fs::File::create(&event_path)?;
        let mut crc32c_table = [0u32; 256];
        for (i, entry) in crc32c_table.iter_mut().enumerate() {
            let mut crc = i as u32;
            for _ in 0..8 {
                if crc & 1 != 0 {
                    crc = (crc >> 1) ^ 0x82f6_3b78;
                } else {
                    crc >>= 1;
                }
            }
            *entry = crc;
        }
        let crc32c = |bytes: &[u8]| -> u32 {
            let mut crc = 0xffff_ffffu32;
            for byte in bytes {
                let index = ((crc ^ u32::from(*byte)) & 0xff) as usize;
                crc = crc32c_table[index] ^ (crc >> 8);
            }
            !crc
        };
        let masked_crc32c = |bytes: &[u8]| -> u32 {
            let crc = crc32c(bytes);
            crc.rotate_right(15).wrapping_add(0xa282_ead8)
        };
        let append_varint = |buffer: &mut Vec<u8>, mut value: u64| {
            while value >= 0x80 {
                buffer.push((value as u8) | 0x80);
                value >>= 7;
            }
            buffer.push(value as u8);
        };
        for epoch in 0..loss.len() {
            let mut values = Vec::<(&str, f32)>::with_capacity(3);
            values.push(("loss", loss[epoch]));
            if epoch < val_loss.len() {
                values.push(("val_loss", val_loss[epoch]));
            }
            if let Some(learning_rates) = learning_rates {
                if epoch < learning_rates.len() {
                    values.push(("lr", learning_rates[epoch] as f32));
                }
            }
            for (tag, scalar) in values {
                let mut summary_value = Vec::<u8>::new();
                summary_value.push(0x0a);
                append_varint(&mut summary_value, tag.len() as u64);
                summary_value.extend_from_slice(tag.as_bytes());
                summary_value.push(0x15);
                summary_value.extend_from_slice(&scalar.to_le_bytes());

                let mut summary = Vec::<u8>::new();
                summary.push(0x0a);
                append_varint(&mut summary, summary_value.len() as u64);
                summary.extend_from_slice(&summary_value);

                let mut event = Vec::<u8>::new();
                event.push(0x09);
                event.extend_from_slice(&wall_time.to_le_bytes());
                event.push(0x10);
                append_varint(&mut event, (epoch + 1) as u64);
                event.push(0x2a);
                append_varint(&mut event, summary.len() as u64);
                event.extend_from_slice(&summary);

                let length = event.len() as u64;
                let length_bytes = length.to_le_bytes();
                file.write_all(&length_bytes)?;
                file.write_all(&masked_crc32c(&length_bytes).to_le_bytes())?;
                file.write_all(&event)?;
                file.write_all(&masked_crc32c(&event).to_le_bytes())?;
            }
        }
        Ok(event_path.display().to_string())
    }

    #[derive(Clone, Debug)]
    struct ReduceLrOnPlateauState {
        current_lr: LearningRate,
        factor: LearningRate,
        patience: usize,
        min_delta: f32,
        best: f32,
        wait: usize,
    }

    impl ReduceLrOnPlateauState {
        fn new(initial_lr: LearningRate, config: &crate::config::TrainReduceLr) -> Self {
            Self {
                current_lr: initial_lr,
                factor: config.factor as LearningRate,
                patience: config.patience,
                min_delta: config.min_delta,
                best: f32::INFINITY,
                wait: 0,
            }
        }

        fn update(&mut self, value: Option<f32>) {
            let Some(value) = value else {
                return;
            };
            if value < self.best - self.min_delta {
                self.best = value;
                self.wait = 0;
                return;
            }
            self.wait += 1;
            if self.wait >= self.patience
                && self.factor.is_finite()
                && self.factor > 0.0
                && self.factor < 1.0
            {
                self.current_lr *= self.factor;
                self.wait = 0;
            }
        }
    }

    pub fn write_tensorboard_image_events(
        log_dir: &str,
        tag_prefix: &str,
        images: &[(&[f32], [usize; 2])],
    ) -> Result<String, BurnTrainError> {
        if images.is_empty() {
            return Err(BurnTrainError::InvalidTensorBoardImage);
        }
        let image_log_dir = PathBuf::from(log_dir).join("images");
        std::fs::create_dir_all(&image_log_dir)?;
        let wall_time = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|duration| duration.as_secs_f64())
            .unwrap_or(0.0);
        let event_path = image_log_dir.join(format!(
            "events.out.tfevents.{}.stardist-rs.images",
            wall_time.trunc() as u64
        ));
        let mut file = std::fs::File::create(&event_path)?;

        let mut crc32c_table = [0u32; 256];
        for (i, entry) in crc32c_table.iter_mut().enumerate() {
            let mut crc = i as u32;
            for _ in 0..8 {
                if crc & 1 != 0 {
                    crc = (crc >> 1) ^ 0x82f6_3b78;
                } else {
                    crc >>= 1;
                }
            }
            *entry = crc;
        }
        let crc32c = |bytes: &[u8]| -> u32 {
            let mut crc = 0xffff_ffffu32;
            for byte in bytes {
                let index = ((crc ^ u32::from(*byte)) & 0xff) as usize;
                crc = crc32c_table[index] ^ (crc >> 8);
            }
            !crc
        };
        let masked_crc32c = |bytes: &[u8]| -> u32 {
            let crc = crc32c(bytes);
            crc.rotate_right(15).wrapping_add(0xa282_ead8)
        };
        let append_varint = |buffer: &mut Vec<u8>, mut value: u64| {
            while value >= 0x80 {
                buffer.push((value as u8) | 0x80);
                value >>= 7;
            }
            buffer.push(value as u8);
        };
        let mut png_crc32_table = [0u32; 256];
        for (i, entry) in png_crc32_table.iter_mut().enumerate() {
            let mut crc = i as u32;
            for _ in 0..8 {
                if crc & 1 != 0 {
                    crc = (crc >> 1) ^ 0xedb8_8320;
                } else {
                    crc >>= 1;
                }
            }
            *entry = crc;
        }
        let png_crc32 = |bytes: &[u8]| -> u32 {
            let mut crc = 0xffff_ffffu32;
            for byte in bytes {
                let index = ((crc ^ u32::from(*byte)) & 0xff) as usize;
                crc = png_crc32_table[index] ^ (crc >> 8);
            }
            !crc
        };
        let append_png_chunk = |png: &mut Vec<u8>, chunk_type: &[u8; 4], data: &[u8]| {
            png.extend_from_slice(&(data.len() as u32).to_be_bytes());
            png.extend_from_slice(chunk_type);
            png.extend_from_slice(data);
            let mut crc_input = Vec::<u8>::with_capacity(4 + data.len());
            crc_input.extend_from_slice(chunk_type);
            crc_input.extend_from_slice(data);
            png.extend_from_slice(&png_crc32(&crc_input).to_be_bytes());
        };

        for (image_index, (image, [height, width])) in images.iter().enumerate() {
            if *height == 0 || *width == 0 || image.len() != height * width {
                return Err(BurnTrainError::InvalidTensorBoardImage);
            }
            let mut min_value = f32::INFINITY;
            let mut max_value = f32::NEG_INFINITY;
            for value in *image {
                if value.is_finite() {
                    min_value = min_value.min(*value);
                    max_value = max_value.max(*value);
                }
            }
            if !min_value.is_finite() || !max_value.is_finite() {
                min_value = 0.0;
                max_value = 1.0;
            }
            let mut scanlines = Vec::<u8>::with_capacity(height * (width + 1));
            for y in 0..*height {
                scanlines.push(0);
                for x in 0..*width {
                    let mut value = image[y * width + x];
                    if !value.is_finite() {
                        value = min_value;
                    }
                    let normalized = if max_value > min_value {
                        (value - min_value) / (max_value - min_value)
                    } else {
                        value.clamp(0.0, 1.0)
                    };
                    scanlines.push((normalized.clamp(0.0, 1.0) * 255.0).round() as u8);
                }
            }
            let mut zlib = Vec::<u8>::new();
            zlib.extend_from_slice(&[0x78, 0x01]);
            let mut offset = 0usize;
            while offset < scanlines.len() {
                let remaining = scanlines.len() - offset;
                let len = remaining.min(65_535);
                zlib.push(if offset + len == scanlines.len() {
                    1
                } else {
                    0
                });
                zlib.extend_from_slice(&(len as u16).to_le_bytes());
                zlib.extend_from_slice(&(!(len as u16)).to_le_bytes());
                zlib.extend_from_slice(&scanlines[offset..offset + len]);
                offset += len;
            }
            let mut adler_a = 1u32;
            let mut adler_b = 0u32;
            for byte in &scanlines {
                adler_a = (adler_a + u32::from(*byte)) % 65_521;
                adler_b = (adler_b + adler_a) % 65_521;
            }
            zlib.extend_from_slice(&((adler_b << 16) | adler_a).to_be_bytes());

            let mut png = Vec::<u8>::new();
            png.extend_from_slice(&[137, 80, 78, 71, 13, 10, 26, 10]);
            let mut ihdr = Vec::<u8>::with_capacity(13);
            ihdr.extend_from_slice(&(*width as u32).to_be_bytes());
            ihdr.extend_from_slice(&(*height as u32).to_be_bytes());
            ihdr.extend_from_slice(&[8, 0, 0, 0, 0]);
            append_png_chunk(&mut png, b"IHDR", &ihdr);
            append_png_chunk(&mut png, b"IDAT", &zlib);
            append_png_chunk(&mut png, b"IEND", &[]);

            let tag = format!("{tag_prefix}/{image_index}");
            let mut image_summary = Vec::<u8>::new();
            image_summary.push(0x08);
            append_varint(&mut image_summary, *height as u64);
            image_summary.push(0x10);
            append_varint(&mut image_summary, *width as u64);
            image_summary.push(0x18);
            append_varint(&mut image_summary, 1);
            image_summary.push(0x22);
            append_varint(&mut image_summary, png.len() as u64);
            image_summary.extend_from_slice(&png);

            let mut summary_value = Vec::<u8>::new();
            summary_value.push(0x0a);
            append_varint(&mut summary_value, tag.len() as u64);
            summary_value.extend_from_slice(tag.as_bytes());
            summary_value.push(0x22);
            append_varint(&mut summary_value, image_summary.len() as u64);
            summary_value.extend_from_slice(&image_summary);

            let mut summary = Vec::<u8>::new();
            summary.push(0x0a);
            append_varint(&mut summary, summary_value.len() as u64);
            summary.extend_from_slice(&summary_value);

            let mut event = Vec::<u8>::new();
            event.push(0x09);
            event.extend_from_slice(&wall_time.to_le_bytes());
            event.push(0x10);
            append_varint(&mut event, image_index as u64);
            event.push(0x2a);
            append_varint(&mut event, summary.len() as u64);
            event.extend_from_slice(&summary);

            let length = event.len() as u64;
            let length_bytes = length.to_le_bytes();
            file.write_all(&length_bytes)?;
            file.write_all(&masked_crc32c(&length_bytes).to_le_bytes())?;
            file.write_all(&event)?;
            file.write_all(&masked_crc32c(&event).to_le_bytes())?;
        }
        Ok(event_path.display().to_string())
    }

    pub fn stardist_data2d_batch_to_tensors<B: Backend>(
        batch: &crate::StarDistData2DBatch,
        device: &B::Device,
    ) -> Result<StarDistData2DBatchTensors<B>, BurnTrainError> {
        let [batch_n, height, width, channels] = batch.x_shape;
        if batch.x.len() != batch_n * height * width * channels {
            return Err(BurnTrainError::ShapeMismatch);
        }
        let [prob_batch, prob_h, prob_w, prob_c] = batch.prob_shape;
        if prob_batch != batch_n
            || prob_c != 1
            || batch.prob.len() != prob_batch * prob_h * prob_w * prob_c
        {
            return Err(BurnTrainError::ShapeMismatch);
        }
        let [dist_batch, dist_h, dist_w, dist_c] = batch.dist_shape;
        if dist_batch != batch_n || batch.dist.len() != dist_batch * dist_h * dist_w * dist_c {
            return Err(BurnTrainError::ShapeMismatch);
        }

        let mut x = vec![0.0f32; batch.x.len()];
        for n in 0..batch_n {
            for y in 0..height {
                for x_pos in 0..width {
                    for c in 0..channels {
                        x[((n * channels + c) * height + y) * width + x_pos] =
                            batch.x[((n * height + y) * width + x_pos) * channels + c];
                    }
                }
            }
        }
        let mut prob = vec![0.0f32; batch.prob.len()];
        for n in 0..prob_batch {
            for y in 0..prob_h {
                for x_pos in 0..prob_w {
                    prob[(n * prob_h + y) * prob_w + x_pos] =
                        batch.prob[((n * prob_h + y) * prob_w + x_pos) * prob_c];
                }
            }
        }
        let mut dist = vec![0.0f32; batch.dist.len()];
        for n in 0..dist_batch {
            for y in 0..dist_h {
                for x_pos in 0..dist_w {
                    for c in 0..dist_c {
                        dist[((n * dist_c + c) * dist_h + y) * dist_w + x_pos] =
                            batch.dist[((n * dist_h + y) * dist_w + x_pos) * dist_c + c];
                    }
                }
            }
        }
        let prob_class = if let Some(prob_class) = &batch.prob_class {
            let shape = batch
                .prob_class_shape
                .ok_or(BurnTrainError::ShapeMismatch)?;
            let [class_batch, class_h, class_w, class_c] = shape;
            if class_batch != batch_n
                || prob_class.len() != class_batch * class_h * class_w * class_c
            {
                return Err(BurnTrainError::ShapeMismatch);
            }
            let mut values = vec![0.0f32; prob_class.len()];
            for n in 0..class_batch {
                for y in 0..class_h {
                    for x_pos in 0..class_w {
                        for c in 0..class_c {
                            values[((n * class_c + c) * class_h + y) * class_w + x_pos] =
                                prob_class[((n * class_h + y) * class_w + x_pos) * class_c + c];
                        }
                    }
                }
            }
            Some(Tensor::<B, 4>::from_data(
                TensorData::new(values, [class_batch, class_c, class_h, class_w]),
                device,
            ))
        } else {
            if batch.prob_class_shape.is_some() {
                return Err(BurnTrainError::ShapeMismatch);
            }
            None
        };

        Ok(StarDistData2DBatchTensors {
            x: Tensor::<B, 4>::from_data(
                TensorData::new(x, [batch_n, channels, height, width]),
                device,
            ),
            prob: Tensor::<B, 4>::from_data(
                TensorData::new(prob, [prob_batch, 1, prob_h, prob_w]),
                device,
            ),
            dist: Tensor::<B, 4>::from_data(
                TensorData::new(dist, [dist_batch, dist_c, dist_h, dist_w]),
                device,
            ),
            prob_class,
        })
    }

    pub fn stardist_data3d_batch_to_tensors<B: Backend>(
        batch: &crate::StarDistData3DBatch,
        device: &B::Device,
    ) -> Result<StarDistData3DBatchTensors<B>, BurnTrainError> {
        let [batch_n, depth, height, width, channels] = batch.x_shape;
        if batch.x.len() != batch_n * depth * height * width * channels {
            return Err(BurnTrainError::ShapeMismatch);
        }
        let [prob_batch, prob_d, prob_h, prob_w, prob_c] = batch.prob_shape;
        if prob_batch != batch_n
            || prob_c != 1
            || batch.prob.len() != prob_batch * prob_d * prob_h * prob_w * prob_c
        {
            return Err(BurnTrainError::ShapeMismatch);
        }
        let [dist_batch, dist_d, dist_h, dist_w, dist_c] = batch.dist_shape;
        if dist_batch != batch_n
            || batch.dist.len() != dist_batch * dist_d * dist_h * dist_w * dist_c
        {
            return Err(BurnTrainError::ShapeMismatch);
        }

        let mut x = vec![0.0f32; batch.x.len()];
        for n in 0..batch_n {
            for z in 0..depth {
                for y in 0..height {
                    for x_pos in 0..width {
                        for c in 0..channels {
                            x[(((n * channels + c) * depth + z) * height + y) * width + x_pos] =
                                batch.x[(((n * depth + z) * height + y) * width + x_pos)
                                    * channels
                                    + c];
                        }
                    }
                }
            }
        }
        let mut prob = vec![0.0f32; batch.prob.len()];
        for n in 0..prob_batch {
            for z in 0..prob_d {
                for y in 0..prob_h {
                    for x_pos in 0..prob_w {
                        prob[((n * prob_d + z) * prob_h + y) * prob_w + x_pos] =
                            batch.prob[(((n * prob_d + z) * prob_h + y) * prob_w + x_pos) * prob_c];
                    }
                }
            }
        }
        let mut dist = vec![0.0f32; batch.dist.len()];
        for n in 0..dist_batch {
            for z in 0..dist_d {
                for y in 0..dist_h {
                    for x_pos in 0..dist_w {
                        for c in 0..dist_c {
                            dist[(((n * dist_c + c) * dist_d + z) * dist_h + y) * dist_w + x_pos] =
                                batch.dist[(((n * dist_d + z) * dist_h + y) * dist_w + x_pos)
                                    * dist_c
                                    + c];
                        }
                    }
                }
            }
        }
        let prob_class = if let Some(prob_class) = &batch.prob_class {
            let shape = batch
                .prob_class_shape
                .ok_or(BurnTrainError::ShapeMismatch)?;
            let [class_batch, class_d, class_h, class_w, class_c] = shape;
            if class_batch != batch_n
                || prob_class.len() != class_batch * class_d * class_h * class_w * class_c
            {
                return Err(BurnTrainError::ShapeMismatch);
            }
            let mut values = vec![0.0f32; prob_class.len()];
            for n in 0..class_batch {
                for z in 0..class_d {
                    for y in 0..class_h {
                        for x_pos in 0..class_w {
                            for c in 0..class_c {
                                values[(((n * class_c + c) * class_d + z) * class_h + y)
                                    * class_w
                                    + x_pos] = prob_class[(((n * class_d + z) * class_h + y)
                                    * class_w
                                    + x_pos)
                                    * class_c
                                    + c];
                            }
                        }
                    }
                }
            }
            Some(Tensor::<B, 5>::from_data(
                TensorData::new(values, [class_batch, class_c, class_d, class_h, class_w]),
                device,
            ))
        } else {
            if batch.prob_class_shape.is_some() {
                return Err(BurnTrainError::ShapeMismatch);
            }
            None
        };

        Ok(StarDistData3DBatchTensors {
            x: Tensor::<B, 5>::from_data(
                TensorData::new(x, [batch_n, channels, depth, height, width]),
                device,
            ),
            prob: Tensor::<B, 5>::from_data(
                TensorData::new(prob, [prob_batch, 1, prob_d, prob_h, prob_w]),
                device,
            ),
            dist: Tensor::<B, 5>::from_data(
                TensorData::new(dist, [dist_batch, dist_c, dist_d, dist_h, dist_w]),
                device,
            ),
            prob_class,
        })
    }

    pub fn prob_loss<B: Backend, const D: usize>(
        y_true: Tensor<B, D>,
        y_pred: Tensor<B, D>,
    ) -> Tensor<B, 1> {
        let eps = f32::EPSILON;
        let mask = y_true.clone().greater_equal_elem(0.0).float();
        let y_true = y_true.clamp(0.0, 1.0);
        let y_pred = y_pred.clamp(eps, 1.0 - eps);
        let one_true = y_true.ones_like();
        let one_pred = y_pred.ones_like();
        let bce = (y_true.clone() * y_pred.clone().log()
            + (one_true - y_true) * (one_pred - y_pred).log())
        .mul_scalar(-1.0);
        let loss = (mask.clone() * bce).sum();
        let norm = mask.sum().add_scalar(eps);
        loss / norm
    }

    pub fn dist_loss<B: Backend, const D: usize>(
        dist_true_mask: Tensor<B, D>,
        dist_pred: Tensor<B, D>,
        n_rays: usize,
        penalty: crate::MaskedPenalty,
        reg_weight: f32,
        norm_by_mask: bool,
    ) -> Tensor<B, 1> {
        let eps = f32::EPSILON;
        let dist_true = dist_true_mask.clone().narrow(1, 0, n_rays);
        let dist_mask = dist_true_mask.narrow(1, n_rays, 1);
        let raw_loss = match penalty {
            crate::MaskedPenalty::Abs => (dist_true - dist_pred.clone()).abs(),
            crate::MaskedPenalty::Square => (dist_true - dist_pred.clone()).powf_scalar(2.0),
        };
        let actual_loss = (dist_mask.clone() * raw_loss).mean_dim(1);
        let norm_mask = if norm_by_mask {
            dist_mask.clone().mean().add_scalar(eps)
        } else {
            dist_mask.clone().full_like(1.0).mean()
        };
        let norm_mask = norm_mask.expand(actual_loss.shape());
        if reg_weight > 0.0 {
            let reg_loss = (dist_mask.full_like(1.0) - dist_mask) * dist_pred.abs();
            (actual_loss / norm_mask + reg_loss.mean_dim(1).mul_scalar(reg_weight)).mean()
        } else {
            (actual_loss / norm_mask).mean()
        }
    }

    pub fn dist_loss_mae<B: Backend, const D: usize>(
        dist_true_mask: Tensor<B, D>,
        dist_pred: Tensor<B, D>,
        n_rays: usize,
        reg_weight: f32,
        norm_by_mask: bool,
    ) -> Tensor<B, 1> {
        dist_loss(
            dist_true_mask,
            dist_pred,
            n_rays,
            crate::MaskedPenalty::Abs,
            reg_weight,
            norm_by_mask,
        )
    }

    pub fn dist_loss_mse<B: Backend, const D: usize>(
        dist_true_mask: Tensor<B, D>,
        dist_pred: Tensor<B, D>,
        n_rays: usize,
        reg_weight: f32,
        norm_by_mask: bool,
    ) -> Tensor<B, 1> {
        dist_loss(
            dist_true_mask,
            dist_pred,
            n_rays,
            crate::MaskedPenalty::Square,
            reg_weight,
            norm_by_mask,
        )
    }

    pub fn weighted_categorical_crossentropy_loss<B: Backend, const D: usize>(
        weights: &[f32],
        y_true: Tensor<B, D>,
        y_pred: Tensor<B, D>,
    ) -> Result<Tensor<B, 1>, BurnTrainError> {
        let shape = y_true.dims();
        if D < 3 || y_pred.dims() != shape {
            return Err(BurnTrainError::ShapeMismatch);
        }
        let channels = shape[1];
        if channels == 0 || weights.len() != channels {
            return Err(BurnTrainError::ShapeMismatch);
        }
        let eps = f32::EPSILON;
        let device = y_pred.device();
        let spatial_stride = shape[2..].iter().product::<usize>();
        let len = shape.iter().product::<usize>();
        let mut weight_values = Vec::<f32>::with_capacity(len);
        for i in 0..len {
            let channel = (i / spatial_stride) % channels;
            weight_values.push(weights[channel]);
        }
        let weight_tensor =
            Tensor::<B, D>::from_data(TensorData::new(weight_values, shape), &device);
        let mask = y_true.clone().greater_equal_elem(0.0).float();
        let pred_sum = y_pred.clone().add_scalar(eps).sum_dim(1);
        let pred_sum = pred_sum.expand(shape);
        let y_pred = (y_pred / pred_sum).clamp(eps, 1.0 - eps);
        Ok((weight_tensor * mask * y_true * y_pred.log())
            .sum_dim(1)
            .mul_scalar(-1.0)
            .mean())
    }

    pub fn stardist_2d_loss<B: Backend>(
        outputs: StarDist2DOutputs<B>,
        prob_true: Tensor<B, 4>,
        dist_true_mask: Tensor<B, 4>,
        prob_class_true: Option<Tensor<B, 4>>,
        config: &crate::Config2D,
    ) -> Result<Tensor<B, 1>, BurnTrainError> {
        if config.train_loss_weights.len() < 2 {
            return Err(BurnTrainError::InvalidLossWeights);
        }
        let prob = prob_loss(prob_true, outputs.prob).mul_scalar(config.train_loss_weights[0]);
        let dist = match config.train_dist_loss.as_str() {
            "mae" => dist_loss_mae(
                dist_true_mask,
                outputs.dist,
                config.n_rays,
                config.train_background_reg,
                true,
            ),
            "mse" => dist_loss_mse(
                dist_true_mask,
                outputs.dist,
                config.n_rays,
                config.train_background_reg,
                true,
            ),
            other => return Err(BurnTrainError::UnsupportedDistanceLoss(other.to_string())),
        }
        .mul_scalar(config.train_loss_weights[1]);
        if let Some(n_classes) = config.n_classes {
            if config.train_loss_weights.len() != 3 {
                return Err(BurnTrainError::InvalidLossWeights);
            }
            if config.train_class_weights.len() != n_classes + 1 {
                return Err(BurnTrainError::InvalidClassWeights);
            }
            let prob_class_true = prob_class_true.ok_or(BurnTrainError::MissingClassLossInput)?;
            let prob_class = outputs
                .prob_class
                .ok_or(BurnTrainError::MissingClassLossInput)?;
            let class_loss = weighted_categorical_crossentropy_loss(
                &config.train_class_weights,
                prob_class_true,
                prob_class,
            )?
            .mul_scalar(config.train_loss_weights[2]);
            Ok(prob + dist + class_loss)
        } else {
            Ok(prob + dist)
        }
    }

    pub fn stardist_3d_loss<B: Backend>(
        outputs: StarDist3DOutputs<B>,
        prob_true: Tensor<B, 5>,
        dist_true_mask: Tensor<B, 5>,
        prob_class_true: Option<Tensor<B, 5>>,
        config: &crate::Config3D,
    ) -> Result<Tensor<B, 1>, BurnTrainError> {
        if config.train_loss_weights.len() < 2 {
            return Err(BurnTrainError::InvalidLossWeights);
        }
        let prob = prob_loss(prob_true, outputs.prob).mul_scalar(config.train_loss_weights[0]);
        let dist = match config.train_dist_loss.as_str() {
            "mae" => dist_loss_mae(
                dist_true_mask,
                outputs.dist,
                config.n_rays,
                config.train_background_reg,
                true,
            ),
            "mse" => dist_loss_mse(
                dist_true_mask,
                outputs.dist,
                config.n_rays,
                config.train_background_reg,
                true,
            ),
            other => return Err(BurnTrainError::UnsupportedDistanceLoss(other.to_string())),
        }
        .mul_scalar(config.train_loss_weights[1]);
        if let Some(n_classes) = config.n_classes {
            if config.train_loss_weights.len() != 3 {
                return Err(BurnTrainError::InvalidLossWeights);
            }
            if config.train_class_weights.len() != n_classes + 1 {
                return Err(BurnTrainError::InvalidClassWeights);
            }
            let prob_class_true = prob_class_true.ok_or(BurnTrainError::MissingClassLossInput)?;
            let prob_class = outputs
                .prob_class
                .ok_or(BurnTrainError::MissingClassLossInput)?;
            let class_loss = weighted_categorical_crossentropy_loss(
                &config.train_class_weights,
                prob_class_true,
                prob_class,
            )?
            .mul_scalar(config.train_loss_weights[2]);
            Ok(prob + dist + class_loss)
        } else {
            Ok(prob + dist)
        }
    }

    pub fn stardist_2d_train_step<B, O>(
        model: StarDist2D<B>,
        optimizer: &mut O,
        batch: StarDistData2DBatchTensors<B>,
        learning_rate: LearningRate,
    ) -> Result<(StarDist2D<B>, Tensor<B, 1>), BurnTrainError>
    where
        B: AutodiffBackend,
        StarDist2D<B>: AutodiffModule<B>,
        O: Optimizer<StarDist2D<B>, B>,
    {
        let config = model.config.clone();
        let outputs = model.forward(batch.x);
        let loss = stardist_2d_loss(outputs, batch.prob, batch.dist, batch.prob_class, &config)?;
        let grads = GradientsParams::from_grads(loss.clone().backward(), &model);
        let model = optimizer.step(learning_rate, model, grads);
        Ok((model, loss))
    }

    pub fn stardist_3d_train_step<B, O>(
        model: StarDist3D<B>,
        optimizer: &mut O,
        batch: StarDistData3DBatchTensors<B>,
        learning_rate: LearningRate,
    ) -> Result<(StarDist3D<B>, Tensor<B, 1>), BurnTrainError>
    where
        B: AutodiffBackend,
        StarDist3D<B>: AutodiffModule<B>,
        O: Optimizer<StarDist3D<B>, B>,
    {
        let config = model.config.clone();
        let outputs = model.forward(batch.x);
        let loss = stardist_3d_loss(outputs, batch.prob, batch.dist, batch.prob_class, &config)?;
        let grads = GradientsParams::from_grads(loss.clone().backward(), &model);
        let model = optimizer.step(learning_rate, model, grads);
        Ok((model, loss))
    }

    pub fn stardist_2d_fit_batches<B, O>(
        mut model: StarDist2D<B>,
        optimizer: &mut O,
        train_batches: &[StarDistData2DBatchTensors<B>],
        validation_batch: Option<StarDistData2DBatchTensors<B>>,
        epochs: usize,
        steps_per_epoch: usize,
        learning_rate: LearningRate,
        device: &B::Device,
        checkpoint: Option<StarDistBurnCheckpointConfig>,
        tensorboard: Option<StarDistBurnTensorBoardConfig>,
    ) -> Result<(StarDist2D<B>, StarDistBurnTrainHistory), BurnTrainError>
    where
        B: AutodiffBackend,
        StarDist2D<B>: AutodiffModule<B>,
        O: Optimizer<StarDist2D<B>, B>,
    {
        if train_batches.is_empty() {
            return Err(BurnTrainError::EmptyTrainingBatches);
        }
        let total_steps = epochs
            .checked_mul(steps_per_epoch)
            .ok_or(BurnTrainError::LengthOverflow)?;
        let mut history = StarDistBurnTrainHistory {
            loss: Vec::with_capacity(epochs),
            val_loss: Vec::with_capacity(epochs),
            learning_rates: Vec::with_capacity(epochs),
            checkpoint_files: Vec::new(),
            log_files: Vec::new(),
            event_files: Vec::new(),
        };
        let log_file_path = if let Some(tensorboard) = &tensorboard {
            std::fs::create_dir_all(&tensorboard.log_dir)?;
            let path = PathBuf::from(&tensorboard.log_dir).join("scalars.tsv");
            let mut file = std::fs::File::create(&path)?;
            writeln!(file, "epoch\tloss\tval_loss")?;
            history.log_files.push(path.display().to_string());
            Some(path)
        } else {
            None
        };
        let recorder = DefaultRecorder::new();
        let mut best_loss = f32::INFINITY;
        let mut reduce_lr =
            ReduceLrOnPlateauState::new(learning_rate, &model.config.train_reduce_lr);
        let mut step = 0usize;
        for epoch in 0..epochs {
            history.learning_rates.push(reduce_lr.current_lr);
            let mut epoch_loss = 0.0f32;
            for _ in 0..steps_per_epoch {
                let batch = train_batches[step % train_batches.len()].clone();
                let (updated, loss) =
                    stardist_2d_train_step(model, optimizer, batch, reduce_lr.current_lr)?;
                let loss = loss.into_data();
                let loss = loss
                    .as_slice::<f32>()
                    .map_err(|_| BurnTrainError::LossScalarUnavailable)?;
                if loss.len() != 1 {
                    return Err(BurnTrainError::LossScalarUnavailable);
                }
                epoch_loss += loss[0];
                model = updated;
                step += 1;
            }
            if steps_per_epoch > 0 {
                history.loss.push(epoch_loss / steps_per_epoch as f32);
            }
            let mut current_val_loss = None;
            if let Some(validation_batch) = &validation_batch {
                let config = model.config.clone();
                let outputs = model.forward(validation_batch.x.clone());
                let val_loss = stardist_2d_loss(
                    outputs,
                    validation_batch.prob.clone(),
                    validation_batch.dist.clone(),
                    validation_batch.prob_class.clone(),
                    &config,
                )?
                .into_data();
                let val_loss = val_loss
                    .as_slice::<f32>()
                    .map_err(|_| BurnTrainError::LossScalarUnavailable)?;
                if val_loss.len() != 1 {
                    return Err(BurnTrainError::LossScalarUnavailable);
                }
                current_val_loss = Some(val_loss[0]);
                history.val_loss.push(val_loss[0]);
            }
            if let Some(checkpoint) = &checkpoint {
                if let (Some(best), Some(val_loss)) = (&checkpoint.best, current_val_loss) {
                    if val_loss < best_loss {
                        best_loss = val_loss;
                        model.clone().save_file(best, &recorder)?;
                        let mut path = PathBuf::from(best);
                        path.set_extension("mpk");
                        history.checkpoint_files.push(path.display().to_string());
                    }
                }
                if let Some(epoch) = &checkpoint.epoch {
                    model.clone().save_file(epoch, &recorder)?;
                    let mut path = PathBuf::from(epoch);
                    path.set_extension("mpk");
                    history.checkpoint_files.push(path.display().to_string());
                }
            }
            if let Some(log_file_path) = &log_file_path {
                let mut file = std::fs::OpenOptions::new()
                    .append(true)
                    .open(log_file_path)?;
                let loss = history.loss.last().copied().unwrap_or(0.0);
                let val_loss = current_val_loss
                    .map(|value| value.to_string())
                    .unwrap_or_else(|| "NA".to_string());
                writeln!(file, "{}\t{}\t{}", epoch + 1, loss, val_loss)?;
            }
            reduce_lr.update(current_val_loss);
        }
        if let Some(checkpoint) = &checkpoint {
            if let Some(last) = &checkpoint.last {
                model.clone().save_file(last, &recorder)?;
                let mut path = PathBuf::from(last);
                path.set_extension("mpk");
                history.checkpoint_files.push(path.display().to_string());
            }
            if let Some(best) = &checkpoint.best {
                if best_loss.is_finite() {
                    model = model.load_file(best, &recorder, device)?;
                }
            }
        }
        if let Some(tensorboard) = &tensorboard {
            let event_file = write_tensorboard_scalar_events_with_learning_rates(
                &tensorboard.log_dir,
                &history.loss,
                &history.val_loss,
                Some(&history.learning_rates),
            )?;
            history.event_files.push(event_file);
        }
        debug_assert_eq!(step, total_steps);
        Ok((model, history))
    }

    pub fn stardist_3d_fit_batches<B, O>(
        mut model: StarDist3D<B>,
        optimizer: &mut O,
        train_batches: &[StarDistData3DBatchTensors<B>],
        validation_batch: Option<StarDistData3DBatchTensors<B>>,
        epochs: usize,
        steps_per_epoch: usize,
        learning_rate: LearningRate,
        device: &B::Device,
        checkpoint: Option<StarDistBurnCheckpointConfig>,
        tensorboard: Option<StarDistBurnTensorBoardConfig>,
    ) -> Result<(StarDist3D<B>, StarDistBurnTrainHistory), BurnTrainError>
    where
        B: AutodiffBackend,
        StarDist3D<B>: AutodiffModule<B>,
        O: Optimizer<StarDist3D<B>, B>,
    {
        if train_batches.is_empty() {
            return Err(BurnTrainError::EmptyTrainingBatches);
        }
        let total_steps = epochs
            .checked_mul(steps_per_epoch)
            .ok_or(BurnTrainError::LengthOverflow)?;
        let mut history = StarDistBurnTrainHistory {
            loss: Vec::with_capacity(epochs),
            val_loss: Vec::with_capacity(epochs),
            learning_rates: Vec::with_capacity(epochs),
            checkpoint_files: Vec::new(),
            log_files: Vec::new(),
            event_files: Vec::new(),
        };
        let log_file_path = if let Some(tensorboard) = &tensorboard {
            std::fs::create_dir_all(&tensorboard.log_dir)?;
            let path = PathBuf::from(&tensorboard.log_dir).join("scalars.tsv");
            let mut file = std::fs::File::create(&path)?;
            writeln!(file, "epoch\tloss\tval_loss")?;
            history.log_files.push(path.display().to_string());
            Some(path)
        } else {
            None
        };
        let recorder = DefaultRecorder::new();
        let mut best_loss = f32::INFINITY;
        let mut reduce_lr =
            ReduceLrOnPlateauState::new(learning_rate, &model.config.train_reduce_lr);
        let mut step = 0usize;
        for epoch in 0..epochs {
            history.learning_rates.push(reduce_lr.current_lr);
            let mut epoch_loss = 0.0f32;
            for _ in 0..steps_per_epoch {
                let batch = train_batches[step % train_batches.len()].clone();
                let (updated, loss) =
                    stardist_3d_train_step(model, optimizer, batch, reduce_lr.current_lr)?;
                let loss = loss.into_data();
                let loss = loss
                    .as_slice::<f32>()
                    .map_err(|_| BurnTrainError::LossScalarUnavailable)?;
                if loss.len() != 1 {
                    return Err(BurnTrainError::LossScalarUnavailable);
                }
                epoch_loss += loss[0];
                model = updated;
                step += 1;
            }
            if steps_per_epoch > 0 {
                history.loss.push(epoch_loss / steps_per_epoch as f32);
            }
            let mut current_val_loss = None;
            if let Some(validation_batch) = &validation_batch {
                let config = model.config.clone();
                let outputs = model.forward(validation_batch.x.clone());
                let val_loss = stardist_3d_loss(
                    outputs,
                    validation_batch.prob.clone(),
                    validation_batch.dist.clone(),
                    validation_batch.prob_class.clone(),
                    &config,
                )?
                .into_data();
                let val_loss = val_loss
                    .as_slice::<f32>()
                    .map_err(|_| BurnTrainError::LossScalarUnavailable)?;
                if val_loss.len() != 1 {
                    return Err(BurnTrainError::LossScalarUnavailable);
                }
                current_val_loss = Some(val_loss[0]);
                history.val_loss.push(val_loss[0]);
            }
            if let Some(checkpoint) = &checkpoint {
                if let (Some(best), Some(val_loss)) = (&checkpoint.best, current_val_loss) {
                    if val_loss < best_loss {
                        best_loss = val_loss;
                        model.clone().save_file(best, &recorder)?;
                        let mut path = PathBuf::from(best);
                        path.set_extension("mpk");
                        history.checkpoint_files.push(path.display().to_string());
                    }
                }
                if let Some(epoch) = &checkpoint.epoch {
                    model.clone().save_file(epoch, &recorder)?;
                    let mut path = PathBuf::from(epoch);
                    path.set_extension("mpk");
                    history.checkpoint_files.push(path.display().to_string());
                }
            }
            if let Some(log_file_path) = &log_file_path {
                let mut file = std::fs::OpenOptions::new()
                    .append(true)
                    .open(log_file_path)?;
                let loss = history.loss.last().copied().unwrap_or(0.0);
                let val_loss = current_val_loss
                    .map(|value| value.to_string())
                    .unwrap_or_else(|| "NA".to_string());
                writeln!(file, "{}\t{}\t{}", epoch + 1, loss, val_loss)?;
            }
            reduce_lr.update(current_val_loss);
        }
        if let Some(checkpoint) = &checkpoint {
            if let Some(last) = &checkpoint.last {
                model.clone().save_file(last, &recorder)?;
                let mut path = PathBuf::from(last);
                path.set_extension("mpk");
                history.checkpoint_files.push(path.display().to_string());
            }
            if let Some(best) = &checkpoint.best {
                if best_loss.is_finite() {
                    model = model.load_file(best, &recorder, device)?;
                }
            }
        }
        if let Some(tensorboard) = &tensorboard {
            let event_file = write_tensorboard_scalar_events_with_learning_rates(
                &tensorboard.log_dir,
                &history.loss,
                &history.val_loss,
                Some(&history.learning_rates),
            )?;
            history.event_files.push(event_file);
        }
        debug_assert_eq!(step, total_steps);
        Ok((model, history))
    }

    pub fn stardist_2d_fit_images<B, O>(
        mut model: StarDist2D<B>,
        optimizer: &mut O,
        mut data_train: crate::StarDistData2D,
        x_images: &[&[f32]],
        x_shapes: &[[usize; 3]],
        y_images: &[&[i32]],
        y_shapes: &[[usize; 2]],
        validation_data: Option<(
            crate::StarDistData2D,
            &[&[f32]],
            &[[usize; 3]],
            &[&[i32]],
            &[[usize; 2]],
            usize,
        )>,
        epochs: usize,
        steps_per_epoch: usize,
        batch_size: usize,
        learning_rate: LearningRate,
        device: &B::Device,
        seed: u64,
        checkpoint: Option<StarDistBurnCheckpointConfig>,
        tensorboard: Option<StarDistBurnTensorBoardConfig>,
    ) -> Result<(StarDist2D<B>, StarDistBurnTrainHistory), BurnTrainError>
    where
        B: AutodiffBackend,
        StarDist2D<B>: AutodiffModule<B>,
        O: Optimizer<StarDist2D<B>, B>,
    {
        if x_images.is_empty()
            || x_images.len() != y_images.len()
            || x_shapes.len() != x_images.len()
            || y_shapes.len() != y_images.len()
            || batch_size == 0
        {
            return Err(BurnTrainError::EmptyTrainingBatches);
        }
        let total_steps = epochs
            .checked_mul(steps_per_epoch)
            .ok_or(BurnTrainError::LengthOverflow)?;
        let mut tensorboard_images = Vec::<(Vec<f32>, [usize; 2])>::new();
        let validation_batch = if let Some((
            mut data_val,
            val_x_images,
            val_x_shapes,
            val_y_images,
            val_y_shapes,
            n_take,
        )) = validation_data
        {
            if val_x_images.is_empty()
                || val_x_images.len() != val_y_images.len()
                || val_x_shapes.len() != val_x_images.len()
                || val_y_shapes.len() != val_y_images.len()
                || n_take == 0
            {
                return Err(BurnTrainError::EmptyTrainingBatches);
            }
            let mut idx = Vec::<usize>::with_capacity(n_take);
            let mut random_values = Vec::<f32>::with_capacity(n_take);
            let mut state = seed ^ 0xa076_1d64_78bd_642f;
            for i in 0..n_take {
                idx.push(i % val_x_images.len());
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                random_values.push(((state >> 40) as f32) / ((1u64 << 24) as f32));
            }
            let val_batch = data_val.__getitem__(
                &idx,
                val_x_images,
                val_x_shapes,
                val_y_images,
                val_y_shapes,
                &random_values,
                seed ^ 0xe703_7ed1_a0b4_28db,
            )?;
            if tensorboard.is_some() {
                let [batch_n, height, width, channels] = val_batch.x_shape;
                if channels == 0 || val_batch.x.len() != batch_n * height * width * channels {
                    return Err(BurnTrainError::ShapeMismatch);
                }
                for sample in 0..batch_n.min(3) {
                    let mut image = Vec::<f32>::with_capacity(height * width);
                    for y in 0..height {
                        for x in 0..width {
                            image.push(val_batch.x[((sample * height + y) * width + x) * channels]);
                        }
                    }
                    tensorboard_images.push((image, [height, width]));
                }
            }
            Some(stardist_data2d_batch_to_tensors(&val_batch, device)?)
        } else {
            None
        };
        let mut history = StarDistBurnTrainHistory {
            loss: Vec::with_capacity(epochs),
            val_loss: Vec::with_capacity(epochs),
            learning_rates: Vec::with_capacity(epochs),
            checkpoint_files: Vec::new(),
            log_files: Vec::new(),
            event_files: Vec::new(),
        };
        let log_file_path = if let Some(tensorboard) = &tensorboard {
            std::fs::create_dir_all(&tensorboard.log_dir)?;
            let path = PathBuf::from(&tensorboard.log_dir).join("scalars.tsv");
            let mut file = std::fs::File::create(&path)?;
            writeln!(file, "epoch\tloss\tval_loss")?;
            history.log_files.push(path.display().to_string());
            Some(path)
        } else {
            None
        };
        let recorder = DefaultRecorder::new();
        let mut best_loss = f32::INFINITY;
        let mut reduce_lr =
            ReduceLrOnPlateauState::new(learning_rate, &model.config.train_reduce_lr);
        let mut state = seed;
        let mut global_step = 0usize;
        for epoch in 0..epochs {
            history.learning_rates.push(reduce_lr.current_lr);
            let mut epoch_loss = 0.0f32;
            for _ in 0..steps_per_epoch {
                let mut idx = Vec::<usize>::with_capacity(batch_size);
                let mut random_values = Vec::<f32>::with_capacity(batch_size);
                for batch_i in 0..batch_size {
                    idx.push((global_step * batch_size + batch_i) % x_images.len());
                    state = state
                        .wrapping_mul(6364136223846793005)
                        .wrapping_add(1442695040888963407);
                    random_values.push(((state >> 40) as f32) / ((1u64 << 24) as f32));
                }
                let batch = data_train.__getitem__(
                    &idx,
                    x_images,
                    x_shapes,
                    y_images,
                    y_shapes,
                    &random_values,
                    seed.wrapping_add(global_step as u64),
                )?;
                let batch = stardist_data2d_batch_to_tensors(&batch, device)?;
                let (updated, loss) =
                    stardist_2d_train_step(model, optimizer, batch, reduce_lr.current_lr)?;
                let loss = loss.into_data();
                let loss = loss
                    .as_slice::<f32>()
                    .map_err(|_| BurnTrainError::LossScalarUnavailable)?;
                if loss.len() != 1 {
                    return Err(BurnTrainError::LossScalarUnavailable);
                }
                epoch_loss += loss[0];
                model = updated;
                global_step += 1;
            }
            if steps_per_epoch > 0 {
                history.loss.push(epoch_loss / steps_per_epoch as f32);
            }
            let mut current_val_loss = None;
            if let Some(validation_batch) = &validation_batch {
                let config = model.config.clone();
                let outputs = model.forward(validation_batch.x.clone());
                let val_loss = stardist_2d_loss(
                    outputs,
                    validation_batch.prob.clone(),
                    validation_batch.dist.clone(),
                    validation_batch.prob_class.clone(),
                    &config,
                )?
                .into_data();
                let val_loss = val_loss
                    .as_slice::<f32>()
                    .map_err(|_| BurnTrainError::LossScalarUnavailable)?;
                if val_loss.len() != 1 {
                    return Err(BurnTrainError::LossScalarUnavailable);
                }
                current_val_loss = Some(val_loss[0]);
                history.val_loss.push(val_loss[0]);
            }
            if let Some(checkpoint) = &checkpoint {
                if let (Some(best), Some(val_loss)) = (&checkpoint.best, current_val_loss) {
                    if val_loss < best_loss {
                        best_loss = val_loss;
                        model.clone().save_file(best, &recorder)?;
                        let mut path = PathBuf::from(best);
                        path.set_extension("mpk");
                        history.checkpoint_files.push(path.display().to_string());
                    }
                }
                if let Some(epoch) = &checkpoint.epoch {
                    model.clone().save_file(epoch, &recorder)?;
                    let mut path = PathBuf::from(epoch);
                    path.set_extension("mpk");
                    history.checkpoint_files.push(path.display().to_string());
                }
            }
            if let Some(log_file_path) = &log_file_path {
                let mut file = std::fs::OpenOptions::new()
                    .append(true)
                    .open(log_file_path)?;
                let loss = history.loss.last().copied().unwrap_or(0.0);
                let val_loss = current_val_loss
                    .map(|value| value.to_string())
                    .unwrap_or_else(|| "NA".to_string());
                writeln!(file, "{}\t{}\t{}", epoch + 1, loss, val_loss)?;
            }
            reduce_lr.update(current_val_loss);
        }
        if let Some(checkpoint) = &checkpoint {
            if let Some(last) = &checkpoint.last {
                model.clone().save_file(last, &recorder)?;
                let mut path = PathBuf::from(last);
                path.set_extension("mpk");
                history.checkpoint_files.push(path.display().to_string());
            }
            if let Some(best) = &checkpoint.best {
                if best_loss.is_finite() {
                    model = model.load_file(best, &recorder, device)?;
                }
            }
        }
        if let Some(tensorboard) = &tensorboard {
            let event_file = write_tensorboard_scalar_events_with_learning_rates(
                &tensorboard.log_dir,
                &history.loss,
                &history.val_loss,
                Some(&history.learning_rates),
            )?;
            history.event_files.push(event_file);
            if !tensorboard_images.is_empty() {
                let image_refs = tensorboard_images
                    .iter()
                    .map(|(image, shape)| (image.as_slice(), *shape))
                    .collect::<Vec<_>>();
                let event_file = write_tensorboard_image_events(
                    &tensorboard.log_dir,
                    "validation/input",
                    &image_refs,
                )?;
                history.event_files.push(event_file);
            }
        }
        debug_assert_eq!(global_step, total_steps);
        Ok((model, history))
    }

    pub fn stardist_3d_fit_images<B, O>(
        mut model: StarDist3D<B>,
        optimizer: &mut O,
        mut data_train: crate::StarDistData3D,
        x_images: &[&[f32]],
        x_shapes: &[[usize; 4]],
        y_images: &[&[i32]],
        y_shapes: &[[usize; 3]],
        validation_data: Option<(
            crate::StarDistData3D,
            &[&[f32]],
            &[[usize; 4]],
            &[&[i32]],
            &[[usize; 3]],
            usize,
        )>,
        epochs: usize,
        steps_per_epoch: usize,
        batch_size: usize,
        learning_rate: LearningRate,
        device: &B::Device,
        seed: u64,
        checkpoint: Option<StarDistBurnCheckpointConfig>,
        tensorboard: Option<StarDistBurnTensorBoardConfig>,
    ) -> Result<(StarDist3D<B>, StarDistBurnTrainHistory), BurnTrainError>
    where
        B: AutodiffBackend,
        StarDist3D<B>: AutodiffModule<B>,
        O: Optimizer<StarDist3D<B>, B>,
    {
        if x_images.is_empty()
            || x_images.len() != y_images.len()
            || x_shapes.len() != x_images.len()
            || y_shapes.len() != y_images.len()
            || batch_size == 0
        {
            return Err(BurnTrainError::EmptyTrainingBatches);
        }
        let total_steps = epochs
            .checked_mul(steps_per_epoch)
            .ok_or(BurnTrainError::LengthOverflow)?;
        let mut tensorboard_images = Vec::<(Vec<f32>, [usize; 2])>::new();
        let validation_batch = if let Some((
            mut data_val,
            val_x_images,
            val_x_shapes,
            val_y_images,
            val_y_shapes,
            n_take,
        )) = validation_data
        {
            if val_x_images.is_empty()
                || val_x_images.len() != val_y_images.len()
                || val_x_shapes.len() != val_x_images.len()
                || val_y_shapes.len() != val_y_images.len()
                || n_take == 0
            {
                return Err(BurnTrainError::EmptyTrainingBatches);
            }
            let mut idx = Vec::<usize>::with_capacity(n_take);
            let mut random_values = Vec::<f32>::with_capacity(n_take);
            let mut state = seed ^ 0xa076_1d64_78bd_642f;
            for i in 0..n_take {
                idx.push(i % val_x_images.len());
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                random_values.push(((state >> 40) as f32) / ((1u64 << 24) as f32));
            }
            let val_batch = data_val.__getitem__(
                &idx,
                val_x_images,
                val_x_shapes,
                val_y_images,
                val_y_shapes,
                &random_values,
                seed ^ 0xe703_7ed1_a0b4_28db,
            )?;
            if tensorboard.is_some() {
                let [batch_n, depth, height, width, channels] = val_batch.x_shape;
                if channels == 0 || val_batch.x.len() != batch_n * depth * height * width * channels
                {
                    return Err(BurnTrainError::ShapeMismatch);
                }
                let z = depth / 2;
                for sample in 0..batch_n.min(3) {
                    let mut image = Vec::<f32>::with_capacity(height * width);
                    for y in 0..height {
                        for x in 0..width {
                            image.push(
                                val_batch.x
                                    [(((sample * depth + z) * height + y) * width + x) * channels],
                            );
                        }
                    }
                    tensorboard_images.push((image, [height, width]));
                }
            }
            Some(stardist_data3d_batch_to_tensors(&val_batch, device)?)
        } else {
            None
        };
        let mut history = StarDistBurnTrainHistory {
            loss: Vec::with_capacity(epochs),
            val_loss: Vec::with_capacity(epochs),
            learning_rates: Vec::with_capacity(epochs),
            checkpoint_files: Vec::new(),
            log_files: Vec::new(),
            event_files: Vec::new(),
        };
        let log_file_path = if let Some(tensorboard) = &tensorboard {
            std::fs::create_dir_all(&tensorboard.log_dir)?;
            let path = PathBuf::from(&tensorboard.log_dir).join("scalars.tsv");
            let mut file = std::fs::File::create(&path)?;
            writeln!(file, "epoch\tloss\tval_loss")?;
            history.log_files.push(path.display().to_string());
            Some(path)
        } else {
            None
        };
        let recorder = DefaultRecorder::new();
        let mut best_loss = f32::INFINITY;
        let mut reduce_lr =
            ReduceLrOnPlateauState::new(learning_rate, &model.config.train_reduce_lr);
        let mut state = seed;
        let mut global_step = 0usize;
        for epoch in 0..epochs {
            history.learning_rates.push(reduce_lr.current_lr);
            let mut epoch_loss = 0.0f32;
            for _ in 0..steps_per_epoch {
                let mut idx = Vec::<usize>::with_capacity(batch_size);
                let mut random_values = Vec::<f32>::with_capacity(batch_size);
                for batch_i in 0..batch_size {
                    idx.push((global_step * batch_size + batch_i) % x_images.len());
                    state = state
                        .wrapping_mul(6364136223846793005)
                        .wrapping_add(1442695040888963407);
                    random_values.push(((state >> 40) as f32) / ((1u64 << 24) as f32));
                }
                let batch = data_train.__getitem__(
                    &idx,
                    x_images,
                    x_shapes,
                    y_images,
                    y_shapes,
                    &random_values,
                    seed.wrapping_add(global_step as u64),
                )?;
                let batch = stardist_data3d_batch_to_tensors(&batch, device)?;
                let (updated, loss) =
                    stardist_3d_train_step(model, optimizer, batch, reduce_lr.current_lr)?;
                let loss = loss.into_data();
                let loss = loss
                    .as_slice::<f32>()
                    .map_err(|_| BurnTrainError::LossScalarUnavailable)?;
                if loss.len() != 1 {
                    return Err(BurnTrainError::LossScalarUnavailable);
                }
                epoch_loss += loss[0];
                model = updated;
                global_step += 1;
            }
            if steps_per_epoch > 0 {
                history.loss.push(epoch_loss / steps_per_epoch as f32);
            }
            let mut current_val_loss = None;
            if let Some(validation_batch) = &validation_batch {
                let config = model.config.clone();
                let outputs = model.forward(validation_batch.x.clone());
                let val_loss = stardist_3d_loss(
                    outputs,
                    validation_batch.prob.clone(),
                    validation_batch.dist.clone(),
                    validation_batch.prob_class.clone(),
                    &config,
                )?
                .into_data();
                let val_loss = val_loss
                    .as_slice::<f32>()
                    .map_err(|_| BurnTrainError::LossScalarUnavailable)?;
                if val_loss.len() != 1 {
                    return Err(BurnTrainError::LossScalarUnavailable);
                }
                current_val_loss = Some(val_loss[0]);
                history.val_loss.push(val_loss[0]);
            }
            if let Some(checkpoint) = &checkpoint {
                if let (Some(best), Some(val_loss)) = (&checkpoint.best, current_val_loss) {
                    if val_loss < best_loss {
                        best_loss = val_loss;
                        model.clone().save_file(best, &recorder)?;
                        let mut path = PathBuf::from(best);
                        path.set_extension("mpk");
                        history.checkpoint_files.push(path.display().to_string());
                    }
                }
                if let Some(epoch) = &checkpoint.epoch {
                    model.clone().save_file(epoch, &recorder)?;
                    let mut path = PathBuf::from(epoch);
                    path.set_extension("mpk");
                    history.checkpoint_files.push(path.display().to_string());
                }
            }
            if let Some(log_file_path) = &log_file_path {
                let mut file = std::fs::OpenOptions::new()
                    .append(true)
                    .open(log_file_path)?;
                let loss = history.loss.last().copied().unwrap_or(0.0);
                let val_loss = current_val_loss
                    .map(|value| value.to_string())
                    .unwrap_or_else(|| "NA".to_string());
                writeln!(file, "{}\t{}\t{}", epoch + 1, loss, val_loss)?;
            }
            reduce_lr.update(current_val_loss);
        }
        if let Some(checkpoint) = &checkpoint {
            if let Some(last) = &checkpoint.last {
                model.clone().save_file(last, &recorder)?;
                let mut path = PathBuf::from(last);
                path.set_extension("mpk");
                history.checkpoint_files.push(path.display().to_string());
            }
            if let Some(best) = &checkpoint.best {
                if best_loss.is_finite() {
                    model = model.load_file(best, &recorder, device)?;
                }
            }
        }
        if let Some(tensorboard) = &tensorboard {
            let event_file = write_tensorboard_scalar_events_with_learning_rates(
                &tensorboard.log_dir,
                &history.loss,
                &history.val_loss,
                Some(&history.learning_rates),
            )?;
            history.event_files.push(event_file);
            if !tensorboard_images.is_empty() {
                let image_refs = tensorboard_images
                    .iter()
                    .map(|(image, shape)| (image.as_slice(), *shape))
                    .collect::<Vec<_>>();
                let event_file = write_tensorboard_image_events(
                    &tensorboard.log_dir,
                    "validation/input_z",
                    &image_refs,
                )?;
                history.event_files.push(event_file);
            }
        }
        debug_assert_eq!(global_step, total_steps);
        Ok((model, history))
    }

    impl<B: Backend> StarDist2D<B> {
        pub fn init(config: crate::Config2D, device: &B::Device) -> Self {
            Self {
                conv2d_1: Conv2dConfig::new([config.n_channel_in, 32], config.unet_kernel_size)
                    .with_padding(PaddingConfig2d::Same)
                    .init(device),
                conv2d_2: Conv2dConfig::new([32, 32], config.unet_kernel_size)
                    .with_padding(PaddingConfig2d::Same)
                    .init(device),
                max_pooling2d_1: MaxPool2dConfig::new([2, 2]).init(),
                down_level_0_no_0: Conv2dConfig::new([32, 32], config.unet_kernel_size)
                    .with_padding(PaddingConfig2d::Same)
                    .init(device),
                down_level_0_no_1: Conv2dConfig::new([32, 32], config.unet_kernel_size)
                    .with_padding(PaddingConfig2d::Same)
                    .init(device),
                max_0: MaxPool2dConfig::new(config.unet_pool).init(),
                down_level_1_no_0: Conv2dConfig::new([32, 64], config.unet_kernel_size)
                    .with_padding(PaddingConfig2d::Same)
                    .init(device),
                down_level_1_no_1: Conv2dConfig::new([64, 64], config.unet_kernel_size)
                    .with_padding(PaddingConfig2d::Same)
                    .init(device),
                max_1: MaxPool2dConfig::new(config.unet_pool).init(),
                down_level_2_no_0: Conv2dConfig::new([64, 128], config.unet_kernel_size)
                    .with_padding(PaddingConfig2d::Same)
                    .init(device),
                down_level_2_no_1: Conv2dConfig::new([128, 128], config.unet_kernel_size)
                    .with_padding(PaddingConfig2d::Same)
                    .init(device),
                max_2: MaxPool2dConfig::new(config.unet_pool).init(),
                middle_0: Conv2dConfig::new([128, 256], config.unet_kernel_size)
                    .with_padding(PaddingConfig2d::Same)
                    .init(device),
                middle_2: Conv2dConfig::new([256, 128], config.unet_kernel_size)
                    .with_padding(PaddingConfig2d::Same)
                    .init(device),
                up_sampling2d_1: Interpolate2dConfig::new()
                    .with_scale_factor(Some([2.0, 2.0]))
                    .with_mode(InterpolateMode::Nearest)
                    .init(),
                up_level_2_no_0: Conv2dConfig::new([256, 128], config.unet_kernel_size)
                    .with_padding(PaddingConfig2d::Same)
                    .init(device),
                up_level_2_no_2: Conv2dConfig::new([128, 64], config.unet_kernel_size)
                    .with_padding(PaddingConfig2d::Same)
                    .init(device),
                up_sampling2d_2: Interpolate2dConfig::new()
                    .with_scale_factor(Some([2.0, 2.0]))
                    .with_mode(InterpolateMode::Nearest)
                    .init(),
                up_level_1_no_0: Conv2dConfig::new([128, 64], config.unet_kernel_size)
                    .with_padding(PaddingConfig2d::Same)
                    .init(device),
                up_level_1_no_2: Conv2dConfig::new([64, 32], config.unet_kernel_size)
                    .with_padding(PaddingConfig2d::Same)
                    .init(device),
                up_sampling2d_3: Interpolate2dConfig::new()
                    .with_scale_factor(Some([2.0, 2.0]))
                    .with_mode(InterpolateMode::Nearest)
                    .init(),
                up_level_0_no_0: Conv2dConfig::new([64, 32], config.unet_kernel_size)
                    .with_padding(PaddingConfig2d::Same)
                    .init(device),
                up_level_0_no_2: Conv2dConfig::new([32, 32], config.unet_kernel_size)
                    .with_padding(PaddingConfig2d::Same)
                    .init(device),
                features: Conv2dConfig::new(
                    [32, config.net_conv_after_unet],
                    config.unet_kernel_size,
                )
                .with_padding(PaddingConfig2d::Same)
                .init(device),
                prob: Conv2dConfig::new([config.net_conv_after_unet, 1], [1, 1])
                    .with_padding(PaddingConfig2d::Same)
                    .init(device),
                dist: Conv2dConfig::new([config.net_conv_after_unet, config.n_rays], [1, 1])
                    .with_padding(PaddingConfig2d::Same)
                    .init(device),
                features_class: if config.n_classes.is_some() {
                    Some(
                        Conv2dConfig::new(
                            [32, config.net_conv_after_unet],
                            config.unet_kernel_size,
                        )
                        .with_padding(PaddingConfig2d::Same)
                        .init(device),
                    )
                } else {
                    None
                },
                prob_class: config.n_classes.map(|n_classes| {
                    Conv2dConfig::new([config.net_conv_after_unet, n_classes + 1], [1, 1])
                        .with_padding(PaddingConfig2d::Same)
                        .init(device)
                }),
                config,
            }
        }

        pub fn forward(&self, input: Tensor<B, 4>) -> StarDist2DOutputs<B> {
            let pooled_img = relu(self.conv2d_1.forward(input));
            let pooled_img = relu(self.conv2d_2.forward(pooled_img));
            let pooled_img = self.max_pooling2d_1.forward(pooled_img);

            let down_level_0 = relu(self.down_level_0_no_0.forward(pooled_img));
            let down_level_0 = relu(self.down_level_0_no_1.forward(down_level_0));
            let max_0 = self.max_0.forward(down_level_0.clone());

            let down_level_1 = relu(self.down_level_1_no_0.forward(max_0));
            let down_level_1 = relu(self.down_level_1_no_1.forward(down_level_1));
            let max_1 = self.max_1.forward(down_level_1.clone());

            let down_level_2 = relu(self.down_level_2_no_0.forward(max_1));
            let down_level_2 = relu(self.down_level_2_no_1.forward(down_level_2));
            let max_2 = self.max_2.forward(down_level_2.clone());

            let middle = relu(self.middle_0.forward(max_2));
            let middle = relu(self.middle_2.forward(middle));

            let up = self.up_sampling2d_1.forward(middle);
            let up = Tensor::cat(vec![up, down_level_2], 1);
            let up = relu(self.up_level_2_no_0.forward(up));
            let up = relu(self.up_level_2_no_2.forward(up));

            let up = self.up_sampling2d_2.forward(up);
            let up = Tensor::cat(vec![up, down_level_1], 1);
            let up = relu(self.up_level_1_no_0.forward(up));
            let up = relu(self.up_level_1_no_2.forward(up));

            let up = self.up_sampling2d_3.forward(up);
            let up = Tensor::cat(vec![up, down_level_0], 1);
            let up = relu(self.up_level_0_no_0.forward(up));
            let up = relu(self.up_level_0_no_2.forward(up));

            let features = relu(self.features.forward(up.clone()));
            let prob = sigmoid(self.prob.forward(features.clone()));
            let dist = self.dist.forward(features);
            let prob_class = if let (Some(features_class), Some(prob_class)) =
                (&self.features_class, &self.prob_class)
            {
                let class_features = relu(features_class.forward(up));
                Some(softmax(prob_class.forward(class_features), 1))
            } else {
                None
            };
            StarDist2DOutputs {
                prob,
                dist,
                prob_class,
            }
        }

        pub fn load_keras_weights(
            mut self,
            weights: &crate::KerasWeights,
            device: &B::Device,
        ) -> Result<Self, BurnWeightError> {
            load_conv2d(
                &mut self.conv2d_1,
                weights,
                "conv2d_1/conv2d_1",
                [32, self.config.n_channel_in, 3, 3],
                device,
            )?;
            load_conv2d(
                &mut self.conv2d_2,
                weights,
                "conv2d_2/conv2d_2",
                [32, 32, 3, 3],
                device,
            )?;
            load_conv2d(
                &mut self.down_level_0_no_0,
                weights,
                "down_level_0_no_0/down_level_0_no_0",
                [32, 32, 3, 3],
                device,
            )?;
            load_conv2d(
                &mut self.down_level_0_no_1,
                weights,
                "down_level_0_no_1/down_level_0_no_1",
                [32, 32, 3, 3],
                device,
            )?;
            load_conv2d(
                &mut self.down_level_1_no_0,
                weights,
                "down_level_1_no_0/down_level_1_no_0",
                [64, 32, 3, 3],
                device,
            )?;
            load_conv2d(
                &mut self.down_level_1_no_1,
                weights,
                "down_level_1_no_1/down_level_1_no_1",
                [64, 64, 3, 3],
                device,
            )?;
            load_conv2d(
                &mut self.down_level_2_no_0,
                weights,
                "down_level_2_no_0/down_level_2_no_0",
                [128, 64, 3, 3],
                device,
            )?;
            load_conv2d(
                &mut self.down_level_2_no_1,
                weights,
                "down_level_2_no_1/down_level_2_no_1",
                [128, 128, 3, 3],
                device,
            )?;
            load_conv2d(
                &mut self.middle_0,
                weights,
                "middle_0/middle_0",
                [256, 128, 3, 3],
                device,
            )?;
            load_conv2d(
                &mut self.middle_2,
                weights,
                "middle_2/middle_2",
                [128, 256, 3, 3],
                device,
            )?;
            load_conv2d(
                &mut self.up_level_2_no_0,
                weights,
                "up_level_2_no_0/up_level_2_no_0",
                [128, 256, 3, 3],
                device,
            )?;
            load_conv2d(
                &mut self.up_level_2_no_2,
                weights,
                "up_level_2_no_2/up_level_2_no_2",
                [64, 128, 3, 3],
                device,
            )?;
            load_conv2d(
                &mut self.up_level_1_no_0,
                weights,
                "up_level_1_no_0/up_level_1_no_0",
                [64, 128, 3, 3],
                device,
            )?;
            load_conv2d(
                &mut self.up_level_1_no_2,
                weights,
                "up_level_1_no_2/up_level_1_no_2",
                [32, 64, 3, 3],
                device,
            )?;
            load_conv2d(
                &mut self.up_level_0_no_0,
                weights,
                "up_level_0_no_0/up_level_0_no_0",
                [32, 64, 3, 3],
                device,
            )?;
            load_conv2d(
                &mut self.up_level_0_no_2,
                weights,
                "up_level_0_no_2/up_level_0_no_2",
                [32, 32, 3, 3],
                device,
            )?;
            load_conv2d(
                &mut self.features,
                weights,
                "features/features",
                [128, 32, 3, 3],
                device,
            )?;
            load_conv2d(&mut self.prob, weights, "prob/prob", [1, 128, 1, 1], device)?;
            load_conv2d(
                &mut self.dist,
                weights,
                "dist/dist",
                [self.config.n_rays, 128, 1, 1],
                device,
            )?;
            if let Some(features_class) = &mut self.features_class {
                load_conv2d(
                    features_class,
                    weights,
                    "features_class/features_class",
                    [128, 32, 3, 3],
                    device,
                )?;
            }
            if let Some(prob_class) = &mut self.prob_class {
                let n_classes = self.config.n_classes.unwrap_or(0);
                load_conv2d(
                    prob_class,
                    weights,
                    "prob_class/prob_class",
                    [n_classes + 1, 128, 1, 1],
                    device,
                )?;
            }
            Ok(self)
        }
    }

    impl<B: Backend> StarDist3D<B> {
        pub fn init(config: crate::config::Config3D, device: &B::Device) -> Self {
            Self {
                conv3d_1: Conv3dConfig::new([config.n_channel_in, 32], [7, 7, 7])
                    .with_padding(PaddingConfig3d::Same)
                    .init(device),
                conv3d_2: Conv3dConfig::new([32, 32], [3, 3, 3])
                    .with_padding(PaddingConfig3d::Same)
                    .init(device),
                conv3d_3: Conv3dConfig::new([32, 64], config.resnet_kernel_size)
                    .with_stride([1, 2, 2])
                    .with_padding(PaddingConfig3d::Valid)
                    .init(device),
                conv3d_4: Conv3dConfig::new([64, 64], config.resnet_kernel_size)
                    .with_padding(PaddingConfig3d::Same)
                    .init(device),
                conv3d_5: Conv3dConfig::new([64, 64], config.resnet_kernel_size)
                    .with_padding(PaddingConfig3d::Same)
                    .init(device),
                conv3d_6: Conv3dConfig::new([32, 64], [1, 1, 1])
                    .with_stride([1, 2, 2])
                    .with_padding(PaddingConfig3d::Valid)
                    .init(device),
                conv3d_7: Conv3dConfig::new([64, 64], config.resnet_kernel_size)
                    .with_padding(PaddingConfig3d::Same)
                    .init(device),
                conv3d_8: Conv3dConfig::new([64, 64], config.resnet_kernel_size)
                    .with_padding(PaddingConfig3d::Same)
                    .init(device),
                conv3d_9: Conv3dConfig::new([64, 64], config.resnet_kernel_size)
                    .with_padding(PaddingConfig3d::Same)
                    .init(device),
                conv3d_10: Conv3dConfig::new([64, 64], config.resnet_kernel_size)
                    .with_padding(PaddingConfig3d::Same)
                    .init(device),
                conv3d_11: Conv3dConfig::new([64, 64], config.resnet_kernel_size)
                    .with_padding(PaddingConfig3d::Same)
                    .init(device),
                conv3d_12: Conv3dConfig::new([64, 64], config.resnet_kernel_size)
                    .with_padding(PaddingConfig3d::Same)
                    .init(device),
                conv3d_13: Conv3dConfig::new([64, 64], config.resnet_kernel_size)
                    .with_padding(PaddingConfig3d::Same)
                    .init(device),
                conv3d_14: Conv3dConfig::new([64, 64], config.resnet_kernel_size)
                    .with_padding(PaddingConfig3d::Same)
                    .init(device),
                conv3d_15: Conv3dConfig::new([64, 64], config.resnet_kernel_size)
                    .with_padding(PaddingConfig3d::Same)
                    .init(device),
                features: Conv3dConfig::new(
                    [64, config.net_conv_after_resnet],
                    config.resnet_kernel_size,
                )
                .with_padding(PaddingConfig3d::Same)
                .init(device),
                prob: Conv3dConfig::new([config.net_conv_after_resnet, 1], [1, 1, 1])
                    .with_padding(PaddingConfig3d::Same)
                    .init(device),
                dist: Conv3dConfig::new([config.net_conv_after_resnet, config.n_rays], [1, 1, 1])
                    .with_padding(PaddingConfig3d::Same)
                    .init(device),
                features_class: if config.n_classes.is_some() {
                    Some(
                        Conv3dConfig::new(
                            [64, config.net_conv_after_resnet],
                            config.resnet_kernel_size,
                        )
                        .with_padding(PaddingConfig3d::Same)
                        .init(device),
                    )
                } else {
                    None
                },
                prob_class: config.n_classes.map(|n_classes| {
                    Conv3dConfig::new([config.net_conv_after_resnet, n_classes + 1], [1, 1, 1])
                        .with_padding(PaddingConfig3d::Same)
                        .init(device)
                }),
                config,
            }
        }

        pub fn forward(&self, input: Tensor<B, 5>) -> StarDist3DOutputs<B> {
            let layer = self.conv3d_1.forward(input);
            let layer = self.conv3d_2.forward(layer);

            let shortcut = self.conv3d_6.forward(layer.clone());
            let [_batch, _channel, depth, height, width] = layer.dims();
            let out_depth = depth.div_ceil(1);
            let out_height = height.div_ceil(2);
            let out_width = width.div_ceil(2);
            let pad_depth = ((out_depth - 1) * 1 + 3).saturating_sub(depth);
            let pad_height = ((out_height - 1) * 2 + 3).saturating_sub(height);
            let pad_width = ((out_width - 1) * 2 + 3).saturating_sub(width);
            let layer_same = layer.pad(
                [
                    (0, 0),
                    (0, 0),
                    (pad_depth / 2, pad_depth - pad_depth / 2),
                    (pad_height / 2, pad_height - pad_height / 2),
                    (pad_width / 2, pad_width - pad_width / 2),
                ],
                BurnPadMode::Constant(0.0),
            );
            let block = relu(self.conv3d_3.forward(layer_same));
            let block = relu(self.conv3d_4.forward(block));
            let block = self.conv3d_5.forward(block);
            let layer = relu(block + shortcut);

            let shortcut = layer.clone();
            let block = relu(self.conv3d_7.forward(layer));
            let block = relu(self.conv3d_8.forward(block));
            let block = self.conv3d_9.forward(block);
            let layer = relu(block + shortcut);

            let shortcut = layer.clone();
            let block = relu(self.conv3d_10.forward(layer));
            let block = relu(self.conv3d_11.forward(block));
            let block = self.conv3d_12.forward(block);
            let layer = relu(block + shortcut);

            let shortcut = layer.clone();
            let block = relu(self.conv3d_13.forward(layer));
            let block = relu(self.conv3d_14.forward(block));
            let block = self.conv3d_15.forward(block);
            let layer = relu(block + shortcut);

            let layer_base = layer;
            let features = relu(self.features.forward(layer_base.clone()));
            let prob = sigmoid(self.prob.forward(features.clone()));
            let dist = self.dist.forward(features);
            let prob_class = if let (Some(features_class), Some(prob_class)) =
                (&self.features_class, &self.prob_class)
            {
                let class_features = relu(features_class.forward(layer_base));
                Some(softmax(prob_class.forward(class_features), 1))
            } else {
                None
            };
            StarDist3DOutputs {
                prob,
                dist,
                prob_class,
            }
        }

        pub fn load_keras_weights(
            mut self,
            weights: &crate::KerasWeights,
            device: &B::Device,
        ) -> Result<Self, BurnWeightError> {
            load_conv3d(
                &mut self.conv3d_1,
                weights,
                "conv3d_1/conv3d_1",
                [32, 1, 7, 7, 7],
                device,
            )?;
            load_conv3d(
                &mut self.conv3d_2,
                weights,
                "conv3d_2/conv3d_2",
                [32, 32, 3, 3, 3],
                device,
            )?;
            load_conv3d(
                &mut self.conv3d_3,
                weights,
                "conv3d_3/conv3d_3",
                [64, 32, 3, 3, 3],
                device,
            )?;
            load_conv3d(
                &mut self.conv3d_4,
                weights,
                "conv3d_4/conv3d_4",
                [64, 64, 3, 3, 3],
                device,
            )?;
            load_conv3d(
                &mut self.conv3d_5,
                weights,
                "conv3d_5/conv3d_5",
                [64, 64, 3, 3, 3],
                device,
            )?;
            load_conv3d(
                &mut self.conv3d_6,
                weights,
                "conv3d_6/conv3d_6",
                [64, 32, 1, 1, 1],
                device,
            )?;
            load_conv3d(
                &mut self.conv3d_7,
                weights,
                "conv3d_7/conv3d_7",
                [64, 64, 3, 3, 3],
                device,
            )?;
            load_conv3d(
                &mut self.conv3d_8,
                weights,
                "conv3d_8/conv3d_8",
                [64, 64, 3, 3, 3],
                device,
            )?;
            load_conv3d(
                &mut self.conv3d_9,
                weights,
                "conv3d_9/conv3d_9",
                [64, 64, 3, 3, 3],
                device,
            )?;
            load_conv3d(
                &mut self.conv3d_10,
                weights,
                "conv3d_10/conv3d_10",
                [64, 64, 3, 3, 3],
                device,
            )?;
            load_conv3d(
                &mut self.conv3d_11,
                weights,
                "conv3d_11/conv3d_11",
                [64, 64, 3, 3, 3],
                device,
            )?;
            load_conv3d(
                &mut self.conv3d_12,
                weights,
                "conv3d_12/conv3d_12",
                [64, 64, 3, 3, 3],
                device,
            )?;
            load_conv3d(
                &mut self.conv3d_13,
                weights,
                "conv3d_13/conv3d_13",
                [64, 64, 3, 3, 3],
                device,
            )?;
            load_conv3d(
                &mut self.conv3d_14,
                weights,
                "conv3d_14/conv3d_14",
                [64, 64, 3, 3, 3],
                device,
            )?;
            load_conv3d(
                &mut self.conv3d_15,
                weights,
                "conv3d_15/conv3d_15",
                [64, 64, 3, 3, 3],
                device,
            )?;
            load_conv3d(
                &mut self.features,
                weights,
                "features/features",
                [128, 64, 3, 3, 3],
                device,
            )?;
            load_conv3d(
                &mut self.prob,
                weights,
                "prob/prob",
                [1, 128, 1, 1, 1],
                device,
            )?;
            load_conv3d(
                &mut self.dist,
                weights,
                "dist/dist",
                [self.config.n_rays, 128, 1, 1, 1],
                device,
            )?;
            if let Some(features_class) = &mut self.features_class {
                load_conv3d(
                    features_class,
                    weights,
                    "features_class/features_class",
                    [128, 64, 3, 3, 3],
                    device,
                )?;
            }
            if let Some(prob_class) = &mut self.prob_class {
                let n_classes = self.config.n_classes.unwrap_or(0);
                load_conv3d(
                    prob_class,
                    weights,
                    "prob_class/prob_class",
                    [n_classes + 1, 128, 1, 1, 1],
                    device,
                )?;
            }
            Ok(self)
        }
    }

    fn load_conv2d<B: Backend>(
        layer: &mut Conv2d<B>,
        weights: &crate::KerasWeights,
        prefix: &str,
        expected_kernel_shape: [usize; 4],
        device: &B::Device,
    ) -> Result<(), BurnWeightError> {
        let kernel_name = format!("{prefix}/kernel:0");
        let kernel = weights
            .get(&kernel_name)
            .ok_or_else(|| BurnWeightError::Missing(kernel_name.clone()))?;
        if kernel.shape != expected_kernel_shape {
            return Err(BurnWeightError::Shape {
                name: kernel_name,
                expected: expected_kernel_shape.to_vec(),
                actual: kernel.shape.clone(),
            });
        }
        layer.weight = Param::from_tensor(Tensor::<B, 4>::from_data(
            TensorData::new(kernel.values.clone(), expected_kernel_shape),
            device,
        ));

        let bias_name = format!("{prefix}/bias:0");
        let bias = weights
            .get(&bias_name)
            .ok_or_else(|| BurnWeightError::Missing(bias_name.clone()))?;
        if bias.shape != vec![expected_kernel_shape[0]] {
            return Err(BurnWeightError::Shape {
                name: bias_name,
                expected: vec![expected_kernel_shape[0]],
                actual: bias.shape.clone(),
            });
        }
        layer.bias = Some(Param::from_tensor(Tensor::<B, 1>::from_data(
            TensorData::new(bias.values.clone(), [expected_kernel_shape[0]]),
            device,
        )));
        Ok(())
    }

    fn load_conv3d<B: Backend>(
        layer: &mut Conv3d<B>,
        weights: &crate::KerasWeights,
        prefix: &str,
        expected_kernel_shape: [usize; 5],
        device: &B::Device,
    ) -> Result<(), BurnWeightError> {
        let kernel_name = format!("{prefix}/kernel:0");
        let kernel = weights
            .get(&kernel_name)
            .ok_or_else(|| BurnWeightError::Missing(kernel_name.clone()))?;
        if kernel.shape != expected_kernel_shape {
            return Err(BurnWeightError::Shape {
                name: kernel_name,
                expected: expected_kernel_shape.to_vec(),
                actual: kernel.shape.clone(),
            });
        }
        layer.weight = Param::from_tensor(Tensor::<B, 5>::from_data(
            TensorData::new(kernel.values.clone(), expected_kernel_shape),
            device,
        ));

        let bias_name = format!("{prefix}/bias:0");
        let bias = weights
            .get(&bias_name)
            .ok_or_else(|| BurnWeightError::Missing(bias_name.clone()))?;
        if bias.shape != vec![expected_kernel_shape[0]] {
            return Err(BurnWeightError::Shape {
                name: bias_name,
                expected: vec![expected_kernel_shape[0]],
                actual: bias.shape.clone(),
            });
        }
        layer.bias = Some(Param::from_tensor(Tensor::<B, 1>::from_data(
            TensorData::new(bias.values.clone(), [expected_kernel_shape[0]]),
            device,
        )));
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_2d_demo_config() {
        let config =
            Config2D::from_json_file("assets/models/examples/2D_demo/config.json").unwrap();
        assert_eq!(config.n_dim, 2);
        assert_eq!(config.n_rays, 32);
        assert_eq!(config.grid, [2, 2]);
        assert_eq!(config.n_classes, None);
        assert_eq!(config.backbone, "unet");
        assert_eq!(config.train_foreground_only, 0.9);
        assert!(config.train_sample_cache);
        assert_eq!(config.train_loss_weights, vec![1.0, 0.2]);
        assert_eq!(config.train_class_weights, vec![1.0, 1.0]);
    }

    #[test]
    fn reads_3d_demo_config() {
        let config =
            crate::Config3D::from_json_file("assets/models/examples/3D_demo/config.json").unwrap();
        assert_eq!(config.n_dim, 3);
        assert_eq!(config.n_rays, 96);
        assert_eq!(config.grid, [1, 2, 2]);
        assert_eq!(config.n_classes, None);
        assert_eq!(config.backbone, "resnet");
        assert_eq!(config.rays_json.name, "Rays_GoldenSpiral");
        assert_eq!(config.rays_json.kwargs.anisotropy, Some([2.0, 1.0, 1.0]));
        assert_eq!(config.unet_n_depth, 2);
        assert_eq!(config.unet_kernel_size, [3, 3, 3]);
        assert_eq!(config.unet_n_filter_base, 32);
        assert_eq!(config.unet_n_conv_per_depth, 2);
        assert_eq!(config.unet_pool, [2, 2, 2]);
        assert_eq!(config.unet_activation, "relu");
        assert_eq!(config.unet_last_activation, "relu");
        assert!(!config.unet_batch_norm);
        assert_eq!(config.unet_dropout, 0.0);
        assert_eq!(config.unet_expansion, 2);
        assert_eq!(config.unet_prefix, "");
        assert_eq!(config.net_conv_after_unet, 128);
        assert_eq!(config.train_foreground_only, 0.9);
        assert!(config.train_sample_cache);
        assert_eq!(config.train_loss_weights, vec![1.0, 0.2]);
        assert_eq!(config.train_class_weights, vec![1.0, 1.0]);
    }

    #[test]
    fn config_class_methods_match_python_properties() {
        let config2d =
            Config2D::from_json_file("assets/models/examples/2D_demo/config.json").unwrap();
        let config3d =
            crate::Config3D::from_json_file("assets/models/examples/3D_demo/config.json").unwrap();
        assert_eq!(
            StarDist2D::new(config2d)._config_class(),
            ConfigClass::Config2D
        );
        assert_eq!(
            StarDist3D::new(config3d)._config_class(),
            ConfigClass::Config3D
        );
    }

    #[test]
    fn build_2d_records_unet_graph_and_multiclass_head_like_model2d_build() {
        let mut config =
            Config2D::from_json_file("assets/models/examples/2D_demo/config.json").unwrap();
        config.grid = [2, 2];
        config.unet_n_conv_per_depth = 2;
        config.unet_n_filter_base = 16;
        config.net_conv_after_unet = 8;
        config.n_rays = 5;
        config.n_classes = Some(2);
        let graph = StarDist2D::new(config)._build().unwrap();

        assert_eq!(graph.ndim, 2);
        assert_eq!(graph.backbone, "unet");
        assert_eq!(graph.outputs, vec!["prob", "dist", "prob_class"]);
        assert!(graph.layers.iter().any(|layer| {
            layer.name == "pre_grid_0_max_pool"
                && layer.kind == "MaxPooling2D"
                && layer.pool == vec![2, 2]
        }));
        assert!(graph.layers.iter().any(|layer| {
            layer.name == "features"
                && layer.kind == "Conv2D"
                && layer.filters == Some(8)
                && layer.source == "unet_block"
        }));
        assert!(graph.layers.iter().any(|layer| {
            layer.name == "prob_class"
                && layer.filters == Some(3)
                && layer.activation.as_deref() == Some("softmax")
        }));
    }

    #[test]
    fn build_3d_dispatches_to_resnet_graph_like_model3d_build() {
        let mut config =
            crate::Config3D::from_json_file("assets/models/examples/3D_demo/config.json").unwrap();
        config.grid = [1, 2, 2];
        config.resnet_n_blocks = 2;
        config.resnet_n_filter_base = 16;
        config.net_conv_after_resnet = 32;
        config.n_rays = 9;
        config.n_classes = Some(1);
        let graph = StarDist3D::new(config)._build().unwrap();

        assert_eq!(graph.ndim, 3);
        assert_eq!(graph.backbone, "resnet");
        assert_eq!(graph.outputs, vec!["prob", "dist", "prob_class"]);
        assert!(graph.layers.iter().any(|layer| {
            layer.name == "conv3d_initial_7"
                && layer.kind == "Conv3D"
                && layer.kernel == vec![7, 7, 7]
                && layer.filters == Some(16)
        }));
        assert!(graph.layers.iter().any(|layer| {
            layer.name == "resnet_block_0"
                && layer.kind == "resnet_block"
                && layer.pool == vec![1, 2, 2]
                && layer.filters == Some(32)
        }));
        assert!(graph.layers.iter().any(|layer| {
            layer.name == "prob_class"
                && layer.filters == Some(2)
                && layer.activation.as_deref() == Some("softmax")
        }));
    }

    #[test]
    fn build_3d_unet_records_unet_graph_like_model3d_build_unet() {
        let mut config =
            crate::Config3D::from_json_file("assets/models/examples/3D_demo/config.json").unwrap();
        config.backbone = "unet".to_string();
        config.grid = [1, 2, 2];
        config.unet_n_conv_per_depth = 2;
        config.unet_n_filter_base = 16;
        config.net_conv_after_unet = 32;
        config.n_rays = 9;
        config.n_classes = Some(1);
        let graph = StarDist3D::new(config)._build().unwrap();

        assert_eq!(graph.ndim, 3);
        assert_eq!(graph.backbone, "unet");
        assert_eq!(graph.outputs, vec!["prob", "dist", "prob_class"]);
        assert!(graph.layers.iter().any(|layer| {
            layer.name == "pre_grid_0_max_pool"
                && layer.kind == "MaxPooling3D"
                && layer.pool == vec![1, 2, 2]
        }));
        assert!(graph.layers.iter().any(|layer| {
            layer.name == "features"
                && layer.kind == "Conv3D"
                && layer.filters == Some(32)
                && layer.source == "unet_block"
        }));
        assert!(graph.layers.iter().any(|layer| {
            layer.name == "prob_class"
                && layer.filters == Some(2)
                && layer.activation.as_deref() == Some("softmax")
        }));
    }

    #[test]
    fn predict_setup_2d_inserts_singleton_channel_and_validates_tiles_like_base() {
        let mut config =
            Config2D::from_json_file("assets/models/examples/2D_demo/config.json").unwrap();
        config.grid = [1, 1];
        config.unet_pool = [1, 1];
        config.unet_n_depth = 1;
        let model = StarDist2D::new(config);
        let img = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];

        let setup = model
            ._predict_setup(&img, &[2, 3], None, Some(&[1, 1]))
            .unwrap();
        assert_eq!(setup.axes, "YX");
        assert_eq!(setup.axes_net, "YXC");
        assert_eq!(setup.x_shape, vec![2, 3, 1]);
        assert_eq!(setup.x, img);
        assert_eq!(setup.channel, 2);
        assert_eq!(setup.n_tiles, vec![1, 1, 1]);

        assert!(matches!(
            model._predict_setup(&img, &[2, 3], None, Some(&[1])),
            Err(StarDistPredictError::TilesDimensionMismatch)
        ));
        assert!(matches!(
            model._predict_setup(&img, &[2, 3], None, Some(&[1, 0])),
            Err(StarDistPredictError::InvalidTiles)
        ));
    }

    #[test]
    fn predict_2d_returns_prob_dist_and_class_with_channels_last() {
        let mut config =
            Config2D::from_json_file("assets/models/examples/2D_demo/config.json").unwrap();
        config.grid = [2, 3];
        config.unet_pool = [1, 1];
        config.unet_n_depth = 1;
        config.n_rays = 4;
        config.n_classes = Some(1);
        let model = StarDist2D::new(config);
        let img = vec![0.0; 3 * 4];

        let prediction = model
            .predict(&img, &[3, 4], None, None, |x, x_shape, axes| {
                assert_eq!(axes, "YXC");
                assert_eq!(x_shape, &[4, 6, 1]);
                assert_eq!(x.len(), 4 * 6);
                let prob = vec![0.1, 0.2, 0.3, 0.4];
                let mut dist = vec![2.0f32; 2 * 2 * 4];
                dist[0] = 0.0;
                let prob_class = vec![0.9, 0.1, 0.2, 0.8, 0.7, 0.3, 0.4, 0.6];
                Ok(StarDistDirectPrediction {
                    prob,
                    prob_shape: vec![2, 2, 1],
                    dist,
                    dist_shape: vec![2, 2, 4],
                    prob_class: Some(prob_class),
                    prob_class_shape: Some(vec![2, 2, 2]),
                })
            })
            .unwrap();

        assert_eq!(prediction.prob_shape, vec![2, 2]);
        assert_eq!(prediction.prob, vec![0.1, 0.2, 0.3, 0.4]);
        assert_eq!(prediction.dist_shape, vec![2, 2, 4]);
        assert_eq!(prediction.dist[0], 1e-3);
        assert_eq!(prediction.dist[1..4], [2.0, 2.0, 2.0]);
        assert_eq!(prediction.prob_class_shape, Some(vec![2, 2, 2]));
        assert_eq!(
            prediction.prob_class.unwrap(),
            vec![0.9, 0.1, 0.2, 0.8, 0.7, 0.3, 0.4, 0.6]
        );
    }

    #[test]
    fn predict_3d_returns_prob_and_dist_with_channels_last() {
        let mut config =
            crate::Config3D::from_json_file("assets/models/examples/3D_demo/config.json").unwrap();
        config.grid = [1, 2, 3];
        config.n_rays = 4;
        let model = StarDist3D::new(config);
        let img = vec![0.0; 2 * 3 * 4];

        let prediction = model
            .predict(&img, &[2, 3, 4], None, None, |x, x_shape, axes| {
                assert_eq!(axes, "ZYXC");
                assert_eq!(x_shape, &[2, 4, 6, 1]);
                assert_eq!(x.len(), 2 * 4 * 6);
                let prob = (0..8).map(|i| i as f32 / 10.0).collect::<Vec<_>>();
                let mut dist = vec![3.0f32; 2 * 2 * 2 * 4];
                dist[7] = 0.0;
                Ok(StarDistDirectPrediction {
                    prob,
                    prob_shape: vec![2, 2, 2, 1],
                    dist,
                    dist_shape: vec![2, 2, 2, 4],
                    prob_class: None,
                    prob_class_shape: None,
                })
            })
            .unwrap();

        assert_eq!(prediction.prob_shape, vec![2, 2, 2]);
        assert_eq!(
            prediction.prob,
            vec![0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
        );
        assert_eq!(prediction.dist_shape, vec![2, 2, 2, 4]);
        assert_eq!(prediction.dist[7], 1e-3);
        assert_eq!(prediction.prob_class, None);
    }

    #[test]
    fn predict_sparse_2d_thresholds_distances_points_and_class_rows() {
        let mut config =
            Config2D::from_json_file("assets/models/examples/2D_demo/config.json").unwrap();
        config.grid = [2, 3];
        config.unet_pool = [1, 1];
        config.unet_n_depth = 1;
        config.n_rays = 4;
        config.n_classes = Some(1);
        let model = StarDist2D::new(config);
        let img = vec![0.0; 3 * 4];

        let sparse = model
            .predict_sparse(
                &img,
                &[3, 4],
                Some(0.5),
                None,
                None,
                0,
                |x, x_shape, axes| {
                    assert_eq!(axes, "YXC");
                    assert_eq!(x_shape, &[4, 6, 1]);
                    assert_eq!(x.len(), 4 * 6);
                    let prob_shape = vec![2, 2, 1];
                    let mut prob = vec![0.1f32; 2 * 2];
                    prob[1] = 0.8;
                    prob[2] = 0.9;
                    let dist_shape = vec![2, 2, 4];
                    let mut dist = vec![2.0f32; 2 * 2 * 4];
                    dist[4] = 0.0;
                    let prob_class_shape = vec![2, 2, 2];
                    let mut prob_class = vec![0.0f32; 2 * 2 * 2];
                    prob_class[2] = 0.2;
                    prob_class[3] = 0.8;
                    prob_class[4] = 0.7;
                    prob_class[5] = 0.3;
                    Ok(StarDistDirectPrediction {
                        prob,
                        prob_shape,
                        dist,
                        dist_shape,
                        prob_class: Some(prob_class),
                        prob_class_shape: Some(prob_class_shape),
                    })
                },
            )
            .unwrap();

        assert_eq!(sparse.prob, vec![0.8, 0.9]);
        assert_eq!(sparse.points, vec![[0.0, 3.0], [2.0, 0.0]]);
        assert_eq!(sparse.dist.len(), 8);
        assert_eq!(sparse.dist[0], 1e-3);
        assert_eq!(sparse.prob_class_channels, Some(2));
        assert_eq!(sparse.prob_class.unwrap(), vec![0.2, 0.8, 0.7, 0.3]);
    }

    #[test]
    fn predict_sparse_3d_thresholds_distances_and_points() {
        let mut config =
            crate::Config3D::from_json_file("assets/models/examples/3D_demo/config.json").unwrap();
        config.grid = [1, 2, 3];
        config.n_rays = 4;
        let model = StarDist3D::new(config);
        let img = vec![0.0; 2 * 3 * 4];

        let sparse = model
            .predict_sparse(
                &img,
                &[2, 3, 4],
                Some(0.5),
                None,
                None,
                0,
                |x, x_shape, axes| {
                    assert_eq!(axes, "ZYXC");
                    assert_eq!(x_shape, &[2, 4, 6, 1]);
                    assert_eq!(x.len(), 2 * 4 * 6);
                    let prob_shape = vec![2, 2, 2, 1];
                    let mut prob = vec![0.1f32; 2 * 2 * 2];
                    prob[(1 * 2 + 1) * 2 + 1] = 0.75;
                    let dist_shape = vec![2, 2, 2, 4];
                    let dist = vec![3.0f32; 2 * 2 * 2 * 4];
                    Ok(StarDistDirectPrediction {
                        prob,
                        prob_shape,
                        dist,
                        dist_shape,
                        prob_class: None,
                        prob_class_shape: None,
                    })
                },
            )
            .unwrap();

        assert_eq!(sparse.prob, vec![0.75]);
        assert_eq!(sparse.points, vec![[1.0, 2.0, 3.0]]);
        assert_eq!(sparse.dist, vec![3.0, 3.0, 3.0, 3.0]);
        assert_eq!(sparse.prob_class, None);
    }

    #[test]
    fn predict_instances_2d_forces_dense_prediction_when_return_predict_is_requested() {
        let mut config =
            Config2D::from_json_file("assets/models/examples/2D_demo/config.json").unwrap();
        config.grid = [1, 1];
        config.unet_pool = [1, 1];
        config.unet_n_depth = 1;
        config.n_rays = 4;
        let model = StarDist2D::new(config);
        let img = vec![0.0; 2 * 2];

        let result = model
            .predict_instances(
                &img,
                &[2, 2],
                None,
                true,
                Some(0.5),
                Some(0.5),
                None,
                None,
                false,
                None,
                true,
                0,
                false,
                false,
                |_, _, _| {
                    let mut prob = vec![0.0f32; 2 * 2];
                    prob[3] = 0.9;
                    Ok(StarDistDirectPrediction {
                        prob,
                        prob_shape: vec![2, 2, 1],
                        dist: vec![1.0; 2 * 2 * 4],
                        dist_shape: vec![2, 2, 4],
                        prob_class: None,
                        prob_class_shape: None,
                    })
                },
            )
            .unwrap();

        assert!(result.prediction.is_some());
        assert_eq!(result.instances.labels, None);
        assert_eq!(result.instances.points, vec![[1.0, 1.0]]);
        assert_eq!(result.instances.prob, vec![0.9]);
    }

    #[test]
    fn predict_instances_3d_sparse_path_returns_instances_without_dense_prediction() {
        let mut config =
            crate::Config3D::from_json_file("assets/models/examples/3D_demo/config.json").unwrap();
        let rays = crate::RaysGoldenSpiral::new(8, None).unwrap().into_rays();
        config.n_rays = 8;
        config.grid = [1, 1, 1];
        config.rays_json = rays.to_json();
        let model = StarDist3D::new(config);
        let img = vec![0.0; 2 * 2 * 2];

        let result = model
            .predict_instances(
                &img,
                &[2, 2, 2],
                None,
                true,
                Some(0.5),
                Some(0.5),
                None,
                None,
                false,
                None,
                false,
                0,
                false,
                false,
                false,
                crate::PolyhedronRenderMode::Bbox,
                |_, _, _| {
                    let mut prob = vec![0.0f32; 2 * 2 * 2];
                    prob[7] = 0.9;
                    Ok(StarDistDirectPrediction {
                        prob,
                        prob_shape: vec![2, 2, 2, 1],
                        dist: vec![1.0; 2 * 2 * 2 * 8],
                        dist_shape: vec![2, 2, 2, 8],
                        prob_class: None,
                        prob_class_shape: None,
                    })
                },
            )
            .unwrap();

        assert_eq!(result.prediction, None);
        assert_eq!(result.instances.labels, None);
        assert_eq!(result.instances.points, vec![[1.0, 1.0, 1.0]]);
        assert_eq!(result.instances.prob, vec![0.9]);
        assert_eq!(result.instances.rays_vertices.len(), 8);
    }

    #[test]
    fn predict_instances_big_2d_crops_filters_relabels_and_concatenates_block_predictions() {
        let mut config =
            Config2D::from_json_file("assets/models/examples/2D_demo/config.json").unwrap();
        config.axes = "YXC".to_string();
        config.grid = [1, 1];
        config.unet_pool = [1, 1];
        config.unet_n_depth = 1;
        let model = StarDist2D::new(config);
        let img = vec![0u8; 8 * 8];
        let mut calls = 0usize;

        let result = model
            .predict_instances_big(
                &img,
                &[8, 8],
                "YX",
                &[6, 6],
                &[2, 2],
                Some(&[1, 1]),
                None,
                |_, tile_shape, _| {
                    calls += 1;
                    let y = tile_shape[0] / 2;
                    let x = tile_shape[1] / 2;
                    let mut labels = vec![0i32; tile_shape[0] * tile_shape[1]];
                    labels[y * tile_shape[1] + x] = 1;
                    Ok(StarDistBigPrediction {
                        labels,
                        labels_shape: tile_shape.to_vec(),
                        polys: crate::BigPolys {
                            entries: vec![
                                (
                                    "prob".to_string(),
                                    crate::BigPolysValue::F32 {
                                        values: vec![0.9],
                                        shape: vec![1],
                                    },
                                ),
                                (
                                    "points".to_string(),
                                    crate::BigPolysValue::F32 {
                                        values: vec![y as f32, x as f32],
                                        shape: vec![1, 2],
                                    },
                                ),
                                (
                                    "coord".to_string(),
                                    crate::BigPolysValue::F32 {
                                        values: vec![y as f32, x as f32],
                                        shape: vec![1, 2, 1],
                                    },
                                ),
                            ],
                        },
                    })
                },
            )
            .unwrap();

        assert_eq!(calls, 4);
        assert_eq!(result.n_blocks, 4);
        assert_eq!(result.labels_shape, vec![8, 8]);
        let labels = result.labels.unwrap();
        let mut unique_labels = labels
            .iter()
            .copied()
            .filter(|label| *label > 0)
            .collect::<Vec<_>>();
        unique_labels.sort_unstable();
        unique_labels.dedup();
        assert_eq!(unique_labels, vec![1, 2, 3, 4]);
        assert_eq!(
            result
                .polys
                .entries
                .iter()
                .find_map(|(key, value)| {
                    if key == "prob" {
                        if let crate::BigPolysValue::F32 { values, shape } = value {
                            Some((values.clone(), shape.clone()))
                        } else {
                            None
                        }
                    } else {
                        None
                    }
                })
                .unwrap(),
            (vec![0.9, 0.9, 0.9, 0.9], vec![4])
        );
    }

    #[test]
    fn predict_instances_big_3d_crops_filters_relabels_and_concatenates_block_predictions() {
        let mut config =
            Config3D::from_json_file("assets/models/examples/3D_demo/config.json").unwrap();
        config.axes = "ZYXC".to_string();
        config.grid = [1, 1, 1];
        let model = StarDist3D::new(config);
        let img = vec![0u8; 8 * 8 * 8];
        let mut calls = 0usize;

        let result = model
            .predict_instances_big(
                &img,
                &[8, 8, 8],
                "ZYX",
                &[6, 6, 6],
                &[2, 2, 2],
                Some(&[1, 1, 1]),
                None,
                |_, tile_shape, _| {
                    calls += 1;
                    let z = tile_shape[0] / 2;
                    let y = tile_shape[1] / 2;
                    let x = tile_shape[2] / 2;
                    let mut labels = vec![0i32; tile_shape[0] * tile_shape[1] * tile_shape[2]];
                    labels[(z * tile_shape[1] + y) * tile_shape[2] + x] = 1;
                    Ok(StarDistBigPrediction {
                        labels,
                        labels_shape: tile_shape.to_vec(),
                        polys: crate::BigPolys {
                            entries: vec![
                                (
                                    "prob".to_string(),
                                    crate::BigPolysValue::F32 {
                                        values: vec![0.8],
                                        shape: vec![1],
                                    },
                                ),
                                (
                                    "points".to_string(),
                                    crate::BigPolysValue::F32 {
                                        values: vec![z as f32, y as f32, x as f32],
                                        shape: vec![1, 3],
                                    },
                                ),
                                (
                                    "coord".to_string(),
                                    crate::BigPolysValue::F32 {
                                        values: vec![z as f32, y as f32, x as f32],
                                        shape: vec![1, 3, 1],
                                    },
                                ),
                            ],
                        },
                    })
                },
            )
            .unwrap();

        assert_eq!(calls, 8);
        assert_eq!(result.n_blocks, 8);
        assert_eq!(result.labels_shape, vec![8, 8, 8]);
        let labels = result.labels.unwrap();
        let mut unique_labels = labels
            .iter()
            .copied()
            .filter(|label| *label > 0)
            .collect::<Vec<_>>();
        unique_labels.sort_unstable();
        unique_labels.dedup();
        assert_eq!(unique_labels, vec![1, 2, 3, 4, 5, 6, 7, 8]);
        assert_eq!(
            result
                .polys
                .entries
                .iter()
                .find_map(|(key, value)| {
                    if key == "prob" {
                        if let crate::BigPolysValue::F32 { values, shape } = value {
                            Some((values.clone(), shape.clone()))
                        } else {
                            None
                        }
                    } else {
                        None
                    }
                })
                .unwrap(),
            (vec![0.8; 8], vec![8])
        );
    }

    #[test]
    fn prepare_for_training_2d_matches_python_compile_choices() {
        let config =
            Config2D::from_json_file("assets/models/examples/2D_demo/config.json").unwrap();
        let model = StarDist2D::new(config);
        let prepared = model.prepare_for_training(None, true).unwrap();
        assert_eq!(prepared.optimizer, "Adam");
        assert_eq!(prepared.learning_rate, 0.0003);
        assert_eq!(prepared.dist_loss, StarDistTrainDistLoss::Mae);
        assert_eq!(prepared.losses, vec!["prob_loss", "dist_loss"]);
        assert_eq!(prepared.loss_weights, vec![1.0, 0.2]);
        assert_eq!(
            prepared.metrics,
            vec![
                "prob:kld",
                "dist:relevant_mae",
                "dist:relevant_mse",
                "dist:dist_iou_metric"
            ]
        );
        assert_eq!(
            prepared.callbacks,
            vec![
                StarDistTrainCallback::ReduceLrOnPlateau,
                StarDistTrainCallback::Checkpoint,
                StarDistTrainCallback::TensorBoard,
            ]
        );
        assert_eq!(
            prepared.checkpoint_callbacks,
            vec![
                StarDistCheckpointCallback {
                    filepath: "./weights_best.h5".to_string(),
                    save_best_only: true,
                    save_weights_only: true,
                },
                StarDistCheckpointCallback {
                    filepath: "./weights_now.h5".to_string(),
                    save_best_only: false,
                    save_weights_only: true,
                },
            ]
        );
        assert_eq!(prepared.tensorboard_log_dir, Some("./logs".to_string()));
        assert_eq!(
            prepared.training_finished,
            vec![
                StarDistTrainingFinishedAction::SaveLastWeights {
                    filepath: "./weights_last.h5".to_string(),
                },
                StarDistTrainingFinishedAction::LoadBestWeights {
                    prefer: "weights_best.h5".to_string(),
                },
                StarDistTrainingFinishedAction::RemoveEpochWeights {
                    filepath: "./weights_now.h5".to_string(),
                },
            ]
        );
        assert!(prepared.model_prepared);
    }

    #[test]
    fn checkpoint_callbacks_and_training_finished_match_csbdeep_base_model() {
        let config =
            Config2D::from_json_file("assets/models/examples/2D_demo/config.json").unwrap();
        let model = StarDist2D::new(config);
        assert_eq!(
            model._checkpoint_callbacks(Some("models/demo"), false),
            vec![
                StarDistCheckpointCallback {
                    filepath: "models/demo/weights_best.h5".to_string(),
                    save_best_only: true,
                    save_weights_only: true,
                },
                StarDistCheckpointCallback {
                    filepath: "models/demo/weights_now.h5".to_string(),
                    save_best_only: false,
                    save_weights_only: true,
                },
            ]
        );
        assert_eq!(
            model._checkpoint_callbacks(Some("models/demo"), true),
            model._checkpoint_callbacks(Some("models/demo"), false),
        );
        assert!(model._checkpoint_callbacks(None, false).is_empty());
        assert_eq!(
            model._training_finished(Some("models/demo")),
            vec![
                StarDistTrainingFinishedAction::SaveLastWeights {
                    filepath: "models/demo/weights_last.h5".to_string(),
                },
                StarDistTrainingFinishedAction::LoadBestWeights {
                    prefer: "weights_best.h5".to_string(),
                },
                StarDistTrainingFinishedAction::RemoveEpochWeights {
                    filepath: "models/demo/weights_now.h5".to_string(),
                },
            ]
        );
        assert!(model._training_finished(None).is_empty());
    }

    #[test]
    fn prepare_for_training_3d_adds_multiclass_loss_branch() {
        let mut config =
            crate::Config3D::from_json_file("assets/models/examples/3D_demo/config.json").unwrap();
        config.n_classes = Some(2);
        config.train_dist_loss = "iou".to_string();
        config.train_loss_weights = vec![1.0, 0.2, 1.0];
        config.train_class_weights = vec![1.0, 1.0, 1.0];
        config.train_tensorboard = false;
        let model = StarDist3D::new(config);
        let prepared = model
            .prepare_for_training(Some("CustomOptimizer"), true)
            .unwrap();
        assert_eq!(prepared.optimizer, "CustomOptimizer");
        assert_eq!(prepared.dist_loss, StarDistTrainDistLoss::Iou);
        assert_eq!(
            prepared.losses,
            vec!["prob_loss", "dist_loss", "prob_class_loss"]
        );
        assert_eq!(prepared.loss_weights, vec![1.0, 0.2, 1.0]);
        assert_eq!(
            prepared.callbacks,
            vec![
                StarDistTrainCallback::ReduceLrOnPlateau,
                StarDistTrainCallback::Checkpoint,
            ]
        );
    }

    #[test]
    fn prepare_for_training_validates_distance_loss_and_weight_lengths() {
        let mut config =
            Config2D::from_json_file("assets/models/examples/2D_demo/config.json").unwrap();
        config.train_dist_loss = "bad".to_string();
        assert_eq!(
            StarDist2D::new(config)
                .prepare_for_training(None, true)
                .unwrap_err(),
            StarDistTrainError::UnsupportedDistanceLoss("bad".to_string())
        );

        let mut config =
            Config2D::from_json_file("assets/models/examples/2D_demo/config.json").unwrap();
        config.n_classes = Some(1);
        config.train_loss_weights = vec![1.0, 0.2];
        assert_eq!(
            StarDist2D::new(config)
                .prepare_for_training(None, true)
                .unwrap_err(),
            StarDistTrainError::InvalidLossWeights
        );

        let mut config =
            Config2D::from_json_file("assets/models/examples/2D_demo/config.json").unwrap();
        config.n_classes = Some(2);
        config.train_loss_weights = vec![1.0, 0.2, 1.0];
        config.train_class_weights = vec![1.0, 1.0];
        assert_eq!(
            StarDist2D::new(config)
                .prepare_for_training(None, true)
                .unwrap_err(),
            StarDistTrainError::InvalidClassWeights
        );
    }

    #[test]
    fn train_2d_builds_generator_setup_like_python_prefit_path() {
        let config =
            Config2D::from_json_file("assets/models/examples/2D_demo/config.json").unwrap();
        let model = StarDist2D::new(config);
        let setup = model
            .train(
                3,
                3,
                2,
                2,
                ClassesArg::Auto,
                None,
                Some(1),
                Some(2),
                Some(5),
            )
            .unwrap();
        assert_eq!(setup.epochs, 2);
        assert_eq!(setup.steps_per_epoch, 5);
        assert_eq!(setup.train_length, 10);
        assert_eq!(setup.validation_n_take, 2);
        assert_eq!(setup.classes, None);
        assert_eq!(setup.validation_classes, None);
        assert_eq!(
            setup.prepared_training.losses,
            vec!["prob_loss", "dist_loss"]
        );
        assert_eq!(setup.data_train.base.patch_size, vec![256, 256]);
        assert_eq!(setup.data_train.base.grid, vec![2, 2]);
        assert_eq!(setup.data_train.base.foreground_prob, 0.9);
        assert!(setup.data_train.base.sample_ind_cache);
    }

    #[test]
    fn train_2d_validates_validation_tuple_and_patch_divisibility() {
        let config =
            Config2D::from_json_file("assets/models/examples/2D_demo/config.json").unwrap();
        let model = StarDist2D::new(config.clone());
        assert_eq!(
            model
                .train(1, 1, 1, 3, ClassesArg::Auto, None, Some(1), None, None)
                .unwrap_err(),
            StarDistTrainError::InvalidValidationData
        );

        let mut config = config;
        config.train_patch_size = [255, 256];
        let model = StarDist2D::new(config);
        assert_eq!(
            model
                .train(1, 1, 1, 2, ClassesArg::Auto, None, Some(1), None, None)
                .unwrap_err(),
            StarDistTrainError::PatchSizeNotDivisible
        );
    }

    #[test]
    fn train_2d_broadcasts_scalar_classes_like_python() {
        let mut config =
            Config2D::from_json_file("assets/models/examples/2D_demo/config.json").unwrap();
        config.n_classes = Some(2);
        config.train_loss_weights = vec![1.0, 0.2, 1.0];
        config.train_class_weights = vec![1.0, 1.0, 1.0];
        let model = StarDist2D::new(config);
        let setup = model
            .train(
                3,
                3,
                2,
                3,
                ClassesArg::Scalar(2),
                Some(ClassesArg::Scalar(1)),
                Some(1),
                Some(1),
                Some(1),
            )
            .unwrap();

        assert_eq!(
            setup.classes,
            Some(vec![
                ClassAssignment::Single(Some(2)),
                ClassAssignment::Single(Some(2)),
                ClassAssignment::Single(Some(2)),
            ])
        );
        assert_eq!(
            setup.validation_classes,
            Some(vec![
                ClassAssignment::Single(Some(1)),
                ClassAssignment::Single(Some(1)),
            ])
        );
        assert_eq!(setup.data_train.classes, setup.classes);
        assert_eq!(setup.data_val.classes, setup.validation_classes);
    }

    #[test]
    fn train_3d_builds_multiclass_generator_setup_like_python_prefit_path() {
        let mut config =
            crate::Config3D::from_json_file("assets/models/examples/3D_demo/config.json").unwrap();
        config.n_classes = Some(1);
        config.train_loss_weights = vec![1.0, 0.2, 1.0];
        config.train_class_weights = vec![1.0, 1.0];
        let model = StarDist3D::new(config);
        let setup = model
            .train(
                2,
                2,
                1,
                2,
                ClassesArg::Scalar(1),
                Some(ClassesArg::Scalar(1)),
                Some(1),
                Some(4),
                Some(3),
            )
            .unwrap();
        assert_eq!(setup.epochs, 4);
        assert_eq!(setup.steps_per_epoch, 3);
        assert_eq!(setup.train_length, 12);
        assert_eq!(setup.validation_n_take, 1);
        assert_eq!(
            setup.classes,
            Some(vec![
                ClassAssignment::Single(Some(1)),
                ClassAssignment::Single(Some(1)),
            ])
        );
        assert_eq!(
            setup.validation_classes,
            Some(vec![ClassAssignment::Single(Some(1))])
        );
        assert_eq!(
            setup.prepared_training.losses,
            vec!["prob_loss", "dist_loss", "prob_class_loss"]
        );
        assert_eq!(setup.data_train.base.patch_size, vec![48, 96, 96]);
        assert_eq!(setup.data_train.base.grid, vec![1, 2, 2]);
        assert_eq!(setup.data_train.rays.vertices.len(), 96);
        assert_eq!(setup.data_train.anisotropy, Some([2.0, 1.0, 1.0]));
    }

    #[test]
    fn train_3d_validates_data_lengths_and_patch_divisibility() {
        let config =
            crate::Config3D::from_json_file("assets/models/examples/3D_demo/config.json").unwrap();
        let model = StarDist3D::new(config.clone());
        assert_eq!(
            model
                .train(0, 0, 1, 2, ClassesArg::Auto, None, Some(1), None, None)
                .unwrap_err(),
            StarDistTrainError::EmptyOrMismatchedData
        );

        let mut config = config;
        config.train_patch_size = [48, 95, 96];
        let model = StarDist3D::new(config);
        assert_eq!(
            model
                .train(1, 1, 1, 2, ClassesArg::Auto, None, Some(1), None, None)
                .unwrap_err(),
            StarDistTrainError::PatchSizeNotDivisible
        );
    }

    #[test]
    fn thresholds_default_and_loaded_values_match_python_base_model() {
        assert_eq!(
            StarDistThresholds::default(),
            StarDistThresholds {
                prob: 0.5,
                nms: 0.4
            }
        );
        assert_eq!(
            StarDistThresholds::new(Some(0.7), Some(0.3)),
            StarDistThresholds {
                prob: 0.7,
                nms: 0.3
            }
        );
        assert_eq!(
            StarDistThresholds::new(Some(1.5), Some(f32::NAN)),
            StarDistThresholds {
                prob: 0.5,
                nms: 0.4
            }
        );
    }

    #[test]
    fn model_threshold_getter_and_setter_validate_values() {
        let config =
            Config2D::from_json_file("assets/models/examples/2D_demo/config.json").unwrap();
        let mut model = StarDist2D::new(config);
        assert_eq!(model.thresholds(), StarDistThresholds::default());
        model
            .set_thresholds(StarDistThresholds {
                prob: 0.6,
                nms: 0.2,
            })
            .unwrap();
        assert_eq!(
            model.thresholds(),
            StarDistThresholds {
                prob: 0.6,
                nms: 0.2
            }
        );
        assert_eq!(
            model
                .set_thresholds(StarDistThresholds {
                    prob: 0.0,
                    nms: 0.2
                })
                .unwrap_err(),
            ThresholdsError::InvalidProb
        );
        assert_eq!(
            model
                .set_thresholds(StarDistThresholds {
                    prob: 0.2,
                    nms: 1.0
                })
                .unwrap_err(),
            ThresholdsError::InvalidNms
        );
    }

    #[test]
    fn optimize_thresholds_selects_best_nms_and_updates_model_thresholds() {
        let config =
            Config2D::from_json_file("assets/models/examples/2D_demo/config.json").unwrap();
        let mut model = StarDist2D::new(config);
        let y_true_image = [1, 1, 0, 0];
        let y_val = [&y_true_image[..]];
        let prob = [0.8, 0.8, 0.1, 0.1];
        let yhat_prob = [&prob[..]];
        let thresholds = model
            .optimize_thresholds(
                &y_val,
                &yhat_prob,
                &[0.2, 0.4],
                &[0.5],
                crate::OptimizeThresholdMeasure::Accuracy,
                Some([0.0, 1.0]),
                1e-3,
                32,
                |_i, prob_thresh, nms_thresh| {
                    if nms_thresh > 0.3 && prob_thresh <= 0.7 {
                        Ok(vec![1, 1, 0, 0])
                    } else {
                        Ok(vec![0, 0, 0, 0])
                    }
                },
            )
            .unwrap();
        assert_eq!(thresholds.nms, 0.4);
        assert!((0.0..=0.7).contains(&thresholds.prob));
        assert_eq!(model.thresholds(), thresholds);
    }

    #[test]
    fn optimize_thresholds_rejects_empty_nms_list() {
        let config =
            Config2D::from_json_file("assets/models/examples/2D_demo/config.json").unwrap();
        let mut model = StarDist2D::new(config);
        let y_true_image = [1, 1, 0, 0];
        let y_val = [&y_true_image[..]];
        let prob = [0.8, 0.8, 0.1, 0.1];
        let yhat_prob = [&prob[..]];
        let err = model
            .optimize_thresholds(
                &y_val,
                &yhat_prob,
                &[],
                &[0.5],
                crate::OptimizeThresholdMeasure::Accuracy,
                Some([0.0, 1.0]),
                1e-3,
                32,
                |_i, _prob_thresh, _nms_thresh| Ok(vec![1, 1, 0, 0]),
            )
            .unwrap_err();
        assert_eq!(err, OptimizeThresholdsError::EmptyNmsThresholds);
    }

    #[test]
    fn axes_div_by_2d_matches_unet_pool_depth_and_grid() {
        let config =
            Config2D::from_json_file("assets/models/examples/2D_demo/config.json").unwrap();
        let model = StarDist2D::new(config);
        assert_eq!(model._axes_div_by("YXC").unwrap(), vec![16, 16, 1]);
        assert_eq!(model._axes_div_by("CYX").unwrap(), vec![1, 16, 16]);
    }

    #[test]
    fn axes_div_by_3d_resnet_matches_grid() {
        let config =
            crate::Config3D::from_json_file("assets/models/examples/3D_demo/config.json").unwrap();
        let model = StarDist3D::new(config);
        assert_eq!(model._axes_div_by("ZYXC").unwrap(), vec![1, 2, 2, 1]);
        assert_eq!(model._axes_div_by("CXYZ").unwrap(), vec![1, 2, 2, 1]);
    }

    #[test]
    fn axes_div_by_3d_unet_matches_pool_depth_and_grid() {
        let mut config =
            crate::Config3D::from_json_file("assets/models/examples/3D_demo/config.json").unwrap();
        config.backbone = "unet".to_string();
        config.grid = [1, 2, 4];
        config.unet_pool = [2, 2, 2];
        config.unet_n_depth = 2;
        let model = StarDist3D::new(config);
        assert_eq!(model._axes_div_by("ZYXC").unwrap(), vec![4, 8, 16, 1]);
        assert_eq!(model._axes_div_by("CXYZ").unwrap(), vec![1, 16, 8, 4]);
    }

    #[test]
    fn axes_tile_overlap_matches_python_demo_values() {
        let config =
            Config2D::from_json_file("assets/models/examples/2D_demo/config.json").unwrap();
        let model = StarDist2D::new(config);
        assert_eq!(model._axes_tile_overlap("YXC").unwrap(), vec![94, 94, 0]);
        assert_eq!(model._axes_tile_overlap("CYX").unwrap(), vec![0, 94, 94]);

        let config =
            crate::Config3D::from_json_file("assets/models/examples/3D_demo/config.json").unwrap();
        let model = StarDist3D::new(config);
        assert_eq!(
            model._axes_tile_overlap("ZYXC").unwrap(),
            vec![17, 30, 30, 0]
        );
        assert_eq!(
            model._axes_tile_overlap("CXYZ").unwrap(),
            vec![0, 30, 30, 17]
        );
    }

    #[test]
    fn axes_tile_overlap_3d_unet_uses_configured_depth_kernel_pool_and_grid() {
        let mut config =
            crate::Config3D::from_json_file("assets/models/examples/3D_demo/config.json").unwrap();
        config.backbone = "unet".to_string();
        config.grid = [1, 2, 4];
        config.unet_n_depth = 2;
        config.unet_kernel_size = [3, 5, 7];
        config.unet_pool = [2, 2, 2];
        let model = StarDist3D::new(config);

        assert_eq!(
            model._axes_tile_overlap("ZYXC").unwrap(),
            vec![22, 92, 282, 0]
        );
        assert_eq!(
            model._axes_tile_overlap("CXYZ").unwrap(),
            vec![0, 282, 92, 22]
        );
    }

    #[test]
    fn compute_receptive_field_returns_symmetric_architecture_overlap() {
        let config =
            Config2D::from_json_file("assets/models/examples/2D_demo/config.json").unwrap();
        let model = StarDist2D::new(config);
        assert_eq!(
            model._compute_receptive_field(None).unwrap(),
            vec![(94, 94), (94, 94)]
        );
        assert_eq!(
            model
                ._compute_receptive_field(Some(&[128, 96]))
                .unwrap_err(),
            AxesTileOverlapError::Unavailable
        );

        let config =
            crate::Config3D::from_json_file("assets/models/examples/3D_demo/config.json").unwrap();
        let model = StarDist3D::new(config);
        assert_eq!(
            model._compute_receptive_field(None).unwrap(),
            vec![(17, 17), (30, 30), (30, 30)]
        );
        assert_eq!(
            model._compute_receptive_field(Some(&[64, 64])).unwrap_err(),
            AxesTileOverlapError::Unavailable
        );
    }

    #[test]
    fn axes_div_by_rejects_duplicate_query_axes() {
        let config =
            Config2D::from_json_file("assets/models/examples/2D_demo/config.json").unwrap();
        let model = StarDist2D::new(config);
        assert_eq!(
            model._axes_div_by("YY").unwrap_err(),
            AxesDivByError::DuplicateAxis
        );
    }

    #[test]
    fn axes_tile_overlap_rejects_duplicate_query_axes() {
        let config =
            Config2D::from_json_file("assets/models/examples/2D_demo/config.json").unwrap();
        let model = StarDist2D::new(config);
        assert_eq!(
            model._axes_tile_overlap("YY").unwrap_err(),
            AxesTileOverlapError::DuplicateAxis
        );
    }

    #[test]
    fn normalize_axes_2d_drops_singleton_channel_axis_when_image_has_no_channel() {
        let config =
            Config2D::from_json_file("assets/models/examples/2D_demo/config.json").unwrap();
        let model = StarDist2D::new(config);
        assert_eq!(model._normalize_axes(&[512, 256], None).unwrap(), "YX");
        assert_eq!(model._normalize_axes(&[512, 256, 1], None).unwrap(), "YXC");
        assert_eq!(
            model._normalize_axes(&[512, 256], Some("xy")).unwrap(),
            "XY"
        );
    }

    #[test]
    fn normalize_axes_rejects_duplicate_or_wrong_length_axes() {
        let config =
            Config2D::from_json_file("assets/models/examples/2D_demo/config.json").unwrap();
        let model = StarDist2D::new(config);
        assert_eq!(
            model._normalize_axes(&[512, 256], Some("YY")).unwrap_err(),
            AxesError::DuplicateAxis
        );
        assert_eq!(
            model._normalize_axes(&[512, 256], Some("YXC")).unwrap_err(),
            AxesError::DimensionMismatch
        );
    }

    #[test]
    fn guess_n_tiles_2d_matches_base_formula_with_and_without_channel_axis() {
        let config =
            Config2D::from_json_file("assets/models/examples/2D_demo/config.json").unwrap();
        let model = StarDist2D::new(config);
        assert_eq!(
            model._guess_n_tiles(&[1024, 512], None).unwrap(),
            vec![2, 1]
        );
        assert_eq!(
            model._guess_n_tiles(&[1024, 512, 1], None).unwrap(),
            vec![2, 1, 1]
        );
    }

    #[test]
    fn guess_n_tiles_3d_matches_base_formula() {
        let config =
            crate::Config3D::from_json_file("assets/models/examples/3D_demo/config.json").unwrap();
        let model = StarDist3D::new(config);
        assert_eq!(
            model._guess_n_tiles(&[96, 192, 192], None).unwrap(),
            vec![2, 2, 2]
        );
        assert_eq!(
            model._guess_n_tiles(&[96, 192, 192, 1], None).unwrap(),
            vec![2, 2, 2, 1]
        );
    }

    #[test]
    fn pad_and_crop_resizer_before_pads_only_at_end() {
        let mut resizer =
            StarDistPadAndCropResizer::new(vec![('Y', 2), ('X', 2)], PadMode::Reflect, 0.0);
        let x = (0..15).map(|v| v as f32).collect::<Vec<_>>();
        let (padded, shape) = resizer.before(&x, &[3, 5], "YX", &[4, 4]).unwrap();
        assert_eq!(shape, vec![4, 8]);
        assert_eq!(resizer.pad, vec![('Y', [0, 1]), ('X', [0, 3])]);
        assert_eq!(resizer.padded_shape, vec![('Y', 4), ('X', 8)]);
        assert_eq!(&padded[0..5], &[0.0, 1.0, 2.0, 3.0, 4.0]);
        assert_eq!(&padded[5..8], &[3.0, 2.0, 1.0]);
        assert_eq!(&padded[24..29], &[5.0, 6.0, 7.0, 8.0, 9.0]);
    }

    #[test]
    fn pad_and_crop_resizer_after_crops_network_output_by_grid_scaled_pad() {
        let mut resizer =
            StarDistPadAndCropResizer::new(vec![('Y', 2), ('X', 2)], PadMode::Reflect, 0.0);
        let x = (0..15).map(|v| v as f32).collect::<Vec<_>>();
        resizer.before(&x, &[3, 5], "YX", &[4, 4]).unwrap();
        let y = (0..8).map(|v| v as f32).collect::<Vec<_>>();
        let (cropped, shape) = resizer.after(&y, &[2, 4], "YX").unwrap();
        assert_eq!(shape, vec![2, 3]);
        assert_eq!(cropped, vec![0.0, 1.0, 2.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn pad_and_crop_resizer_filter_points_returns_indices_inside_crop_region() {
        let mut resizer =
            StarDistPadAndCropResizer::new(vec![('Y', 2), ('X', 2)], PadMode::Reflect, 0.0);
        let x = (0..15).map(|v| v as f32).collect::<Vec<_>>();
        resizer.before(&x, &[3, 5], "YX", &[4, 4]).unwrap();
        let points = [[2.9, 4.9], [3.0, 0.0], [0.0, 5.0], [1.0, 2.0]];
        let indices = resizer.filter_points::<2>(2, &points, "YX").unwrap();
        assert_eq!(indices, vec![0, 3]);
    }

    #[test]
    fn is_multiclass_matches_config_n_classes_check() {
        assert!(!_is_multiclass(None));
        assert!(_is_multiclass(Some(1)));
        assert!(_is_multiclass(Some(3)));
    }

    #[test]
    fn tf_version_at_least_matches_packaging_version_order_for_numeric_releases() {
        assert!(_tf_version_at_least("2.2.0", "2.2.0"));
        assert!(_tf_version_at_least("2.13.1", "2.2.0"));
        assert!(!_tf_version_at_least("2.1.9", "2.2.0"));
        assert!(!_tf_version_at_least("1.15.5", "2.0.0"));
    }

    #[test]
    fn masked_losses_match_channel_reduction_and_mask_normalization() {
        let mask = [1.0, 0.0, 1.0, 1.0];
        let y_true = [1.0, 3.0, 5.0, 7.0];
        let y_pred = [2.0, 1.0, 4.0, 10.0];
        let mae = masked_loss_mae(&mask, &y_true, &y_pred, 2, 0.0, true).unwrap();
        let mse = masked_loss_mse(&mask, &y_true, &y_pred, 2, 0.0, true).unwrap();
        assert!((mae[0] - (0.5 / 0.75)).abs() < 1e-6);
        assert!((mae[1] - (2.0 / 0.75)).abs() < 1e-6);
        assert!((mse[0] - (0.5 / 0.75)).abs() < 1e-6);
        assert!((mse[1] - (5.0 / 0.75)).abs() < 1e-6);
    }

    #[test]
    fn masked_metrics_are_masked_losses_without_regularization() {
        let mask = [1.0, 0.0];
        let y_true = [1.0, 3.0];
        let y_pred = [2.0, 1.0];
        let metric = masked_metric_mae(&mask, &y_true, &y_pred, 2).unwrap();
        let loss = masked_loss(&mask, &y_true, &y_pred, 2, MaskedPenalty::Abs, 0.0, true).unwrap();
        assert_eq!(metric, loss);
    }

    #[test]
    fn kld_ignores_negative_targets_like_python_mask() {
        let value = kld(&[1.0, -1.0, 0.5], &[0.5, 0.2, 0.25]).unwrap();
        let expected = ((-(0.5f32.ln()))
            + (-(0.5 * 0.25f32.ln() + 0.5 * 0.75f32.ln())
                + (0.5 * 0.5f32.ln() + 0.5 * 0.5f32.ln())))
            / 2.0;
        assert!((value - expected).abs() < 1e-6);
    }

    #[test]
    fn masked_iou_loss_and_metric_match_stardist_formulas() {
        let mask = [1.0, 1.0];
        let y_true = [1.0, 2.0];
        let y_pred = [1.0, 1.0];
        let loss = masked_loss_iou(&mask, &y_true, &y_pred, 2, 0.0, false).unwrap();
        let metric = masked_metric_iou(&mask, &y_true, &y_pred, 2, 0.0, false).unwrap();
        let iou = 1.0 / 2.5;
        assert!((loss[0] - (1.0 - iou)).abs() < 1e-6);
        assert!((metric[0] - iou).abs() < 1e-6);
    }

    #[test]
    fn weighted_categorical_crossentropy_ignores_negative_class_targets() {
        let loss = weighted_categorical_crossentropy(
            &[1.0, 2.0],
            2,
            &[1.0, 0.0, -1.0, -1.0],
            &[0.25, 0.75, 0.5, 0.5],
            2,
        )
        .unwrap();
        assert!((loss[0] + (0.25f32 / (1.0 + 2.0 * f32::EPSILON)).ln()).abs() < 1e-5);
        assert_eq!(loss[1], 0.0);
    }

    #[cfg(feature = "burn")]
    #[test]
    fn burn_prob_loss_masks_negative_targets() {
        type B = ::burn::backend::Flex;
        let device = Default::default();
        let y_true = ::burn::tensor::Tensor::<B, 4>::from_data(
            ::burn::tensor::TensorData::new(vec![1.0, 0.0, -1.0, 1.0], [1, 1, 2, 2]),
            &device,
        );
        let y_pred = ::burn::tensor::Tensor::<B, 4>::from_data(
            ::burn::tensor::TensorData::new(vec![0.8, 0.2, 0.9, 0.4], [1, 1, 2, 2]),
            &device,
        );
        let loss = burn::prob_loss(y_true, y_pred).into_data();
        let loss = loss.as_slice::<f32>().unwrap()[0];
        let expected = -((0.8f32).ln() + (1.0f32 - 0.2f32).ln() + (0.4f32).ln()) / 3.0;
        assert!((loss - expected).abs() < 1e-6, "{loss} != {expected}");
    }

    #[cfg(feature = "burn")]
    #[test]
    fn burn_dist_loss_uses_last_channel_as_mask() {
        type B = ::burn::backend::Flex;
        let device = Default::default();
        let dist_true_mask = ::burn::tensor::Tensor::<B, 4>::from_data(
            ::burn::tensor::TensorData::new(vec![1.0, 3.0, 2.0, 4.0, 1.0, 0.0], [1, 3, 1, 2]),
            &device,
        );
        let dist_pred = ::burn::tensor::Tensor::<B, 4>::from_data(
            ::burn::tensor::TensorData::new(vec![1.5, 10.0, 1.0, 20.0], [1, 2, 1, 2]),
            &device,
        );
        let loss = burn::dist_loss_mae(dist_true_mask, dist_pred, 2, 0.0, true).into_data();
        let loss = loss.as_slice::<f32>().unwrap()[0];
        assert!((loss - 0.75).abs() < 1e-5, "{loss}");
    }

    #[cfg(feature = "burn")]
    #[test]
    fn burn_weighted_categorical_crossentropy_masks_negative_targets() {
        type B = ::burn::backend::Flex;
        let device = Default::default();
        let y_true = ::burn::tensor::Tensor::<B, 4>::from_data(
            ::burn::tensor::TensorData::new(vec![1.0, -1.0, 0.0, -1.0], [1, 2, 1, 2]),
            &device,
        );
        let y_pred = ::burn::tensor::Tensor::<B, 4>::from_data(
            ::burn::tensor::TensorData::new(vec![0.25, 0.5, 0.75, 0.5], [1, 2, 1, 2]),
            &device,
        );
        let loss = burn::weighted_categorical_crossentropy_loss(&[1.0, 2.0], y_true, y_pred)
            .unwrap()
            .into_data();
        let loss = loss.as_slice::<f32>().unwrap()[0];
        let expected = -((0.25f32 / (1.0 + 2.0 * f32::EPSILON)).ln()) / 2.0;
        assert!((loss - expected).abs() < 1e-5, "{loss} != {expected}");
    }

    #[cfg(feature = "burn")]
    #[test]
    fn burn_2d_stardist_loss_includes_multiclass_component() {
        type B = ::burn::backend::Flex;
        let device = Default::default();
        let mut config =
            Config2D::from_json_file("assets/models/examples/2D_demo/config.json").unwrap();
        config.n_rays = 1;
        config.n_classes = Some(1);
        config.train_loss_weights = vec![1.0, 1.0, 1.0];
        config.train_class_weights = vec![1.0, 1.0];
        config.train_background_reg = 0.0;

        let outputs = burn::StarDist2DOutputs::<B> {
            prob: ::burn::tensor::Tensor::<B, 4>::from_data(
                ::burn::tensor::TensorData::new(vec![0.5], [1, 1, 1, 1]),
                &device,
            ),
            dist: ::burn::tensor::Tensor::<B, 4>::from_data(
                ::burn::tensor::TensorData::new(vec![1.0], [1, 1, 1, 1]),
                &device,
            ),
            prob_class: Some(::burn::tensor::Tensor::<B, 4>::from_data(
                ::burn::tensor::TensorData::new(vec![0.25, 0.75], [1, 2, 1, 1]),
                &device,
            )),
        };
        let prob_true = ::burn::tensor::Tensor::<B, 4>::from_data(
            ::burn::tensor::TensorData::new(vec![1.0], [1, 1, 1, 1]),
            &device,
        );
        let dist_true_mask = ::burn::tensor::Tensor::<B, 4>::from_data(
            ::burn::tensor::TensorData::new(vec![2.0, 1.0], [1, 2, 1, 1]),
            &device,
        );
        let prob_class_true = ::burn::tensor::Tensor::<B, 4>::from_data(
            ::burn::tensor::TensorData::new(vec![1.0, 0.0], [1, 2, 1, 1]),
            &device,
        );

        let loss = burn::stardist_2d_loss(
            outputs,
            prob_true,
            dist_true_mask,
            Some(prob_class_true),
            &config,
        )
        .unwrap()
        .into_data();
        let loss = loss.as_slice::<f32>().unwrap()[0];
        let expected = -(0.5f32.ln()) + 1.0 - (0.25f32 / (1.0 + 2.0 * f32::EPSILON)).ln();
        assert!((loss - expected).abs() < 1e-5, "{loss} != {expected}");
    }

    #[cfg(feature = "burn")]
    #[test]
    fn burn_2d_model_adds_multiclass_head_when_configured() {
        type B = ::burn::backend::Flex;
        let device = Default::default();
        let mut config =
            Config2D::from_json_file("assets/models/examples/2D_demo/config.json").unwrap();
        config.n_classes = Some(2);
        config.n_channel_out = 1 + config.n_rays + 3;
        config.train_loss_weights = vec![1.0, 0.2, 1.0];
        config.train_class_weights = vec![1.0, 1.0, 1.0];

        let model = burn::StarDist2D::<B>::init(config, &device);
        let input = ::burn::tensor::Tensor::<B, 4>::zeros([1, 1, 64, 64], &device);
        let outputs = model.forward(input);
        assert_eq!(outputs.prob.dims(), [1, 1, 32, 32]);
        assert_eq!(outputs.dist.dims(), [1, 32, 32, 32]);
        assert_eq!(outputs.prob_class.unwrap().dims(), [1, 3, 32, 32]);
    }

    #[cfg(feature = "burn")]
    #[test]
    fn burn_2d_train_step_runs_backward_and_optimizer_update() {
        type B = ::burn::backend::Autodiff<::burn::backend::Flex>;
        let device = Default::default();
        let mut config =
            Config2D::from_json_file("assets/models/examples/2D_demo/config.json").unwrap();
        config.train_dist_loss = "mae".to_string();
        config.train_loss_weights = vec![1.0, 0.2];
        config.train_background_reg = 0.0;

        let model = burn::StarDist2D::<B>::init(config.clone(), &device);
        let batch = burn::StarDistData2DBatchTensors {
            x: ::burn::tensor::Tensor::<B, 4>::zeros([1, 1, 64, 64], &device),
            prob: ::burn::tensor::Tensor::<B, 4>::zeros([1, 1, 32, 32], &device),
            dist: ::burn::tensor::Tensor::<B, 4>::ones([1, config.n_rays + 1, 32, 32], &device),
            prob_class: None,
        };
        let mut optimizer = ::burn::optim::AdamConfig::new().init::<B, burn::StarDist2D<B>>();

        let (model, loss) =
            burn::stardist_2d_train_step(model, &mut optimizer, batch, 1e-4).unwrap();
        let loss = loss.into_data();
        let loss = loss.as_slice::<f32>().unwrap()[0];
        assert!(loss.is_finite(), "{loss}");

        let outputs = model.forward(::burn::tensor::Tensor::<B, 4>::zeros(
            [1, 1, 64, 64],
            &device,
        ));
        assert_eq!(outputs.prob.dims(), [1, 1, 32, 32]);
        assert_eq!(outputs.dist.dims(), [1, 32, 32, 32]);
    }

    #[cfg(feature = "burn")]
    #[test]
    fn burn_2d_fit_batches_records_epoch_and_validation_losses() {
        type B = ::burn::backend::Autodiff<::burn::backend::Flex>;
        let device = Default::default();
        let mut config =
            Config2D::from_json_file("assets/models/examples/2D_demo/config.json").unwrap();
        config.train_dist_loss = "mae".to_string();
        config.train_loss_weights = vec![1.0, 0.2];
        config.train_background_reg = 0.0;

        let model = burn::StarDist2D::<B>::init(config.clone(), &device);
        let batch = burn::StarDistData2DBatchTensors {
            x: ::burn::tensor::Tensor::<B, 4>::zeros([1, 1, 64, 64], &device),
            prob: ::burn::tensor::Tensor::<B, 4>::zeros([1, 1, 32, 32], &device),
            dist: ::burn::tensor::Tensor::<B, 4>::ones([1, config.n_rays + 1, 32, 32], &device),
            prob_class: None,
        };
        let validation_batch = batch.clone();
        let mut optimizer = ::burn::optim::AdamConfig::new().init::<B, burn::StarDist2D<B>>();

        let (_model, history) = burn::stardist_2d_fit_batches(
            model,
            &mut optimizer,
            &[batch],
            Some(validation_batch),
            2,
            1,
            1e-4,
            &device,
            None,
            None,
        )
        .unwrap();

        assert_eq!(history.loss.len(), 2);
        assert_eq!(history.val_loss.len(), 2);
        assert_eq!(history.learning_rates, vec![1e-4, 1e-4]);
        assert!(history.loss.iter().all(|v| v.is_finite()));
        assert!(history.val_loss.iter().all(|v| v.is_finite()));
        assert!(history.checkpoint_files.is_empty());
        assert!(history.log_files.is_empty());
    }

    #[cfg(feature = "burn")]
    #[test]
    fn burn_2d_fit_batches_reduces_learning_rate_on_plateau() {
        type B = ::burn::backend::Autodiff<::burn::backend::Flex>;
        let device = Default::default();
        let mut config =
            Config2D::from_json_file("assets/models/examples/2D_demo/config.json").unwrap();
        config.train_dist_loss = "mae".to_string();
        config.train_loss_weights = vec![1.0, 0.2];
        config.train_background_reg = 0.0;
        config.train_reduce_lr.factor = 0.25;
        config.train_reduce_lr.patience = 1;
        config.train_reduce_lr.min_delta = f32::MAX;

        let model = burn::StarDist2D::<B>::init(config.clone(), &device);
        let batch = burn::StarDistData2DBatchTensors {
            x: ::burn::tensor::Tensor::<B, 4>::zeros([1, 1, 64, 64], &device),
            prob: ::burn::tensor::Tensor::<B, 4>::zeros([1, 1, 32, 32], &device),
            dist: ::burn::tensor::Tensor::<B, 4>::ones([1, config.n_rays + 1, 32, 32], &device),
            prob_class: None,
        };
        let validation_batch = batch.clone();
        let mut optimizer = ::burn::optim::AdamConfig::new().init::<B, burn::StarDist2D<B>>();

        let (_model, history) = burn::stardist_2d_fit_batches(
            model,
            &mut optimizer,
            &[batch],
            Some(validation_batch),
            3,
            1,
            1e-4,
            &device,
            None,
            None,
        )
        .unwrap();

        assert_eq!(history.learning_rates.len(), 3);
        assert_eq!(history.learning_rates[0], 1e-4);
        assert_eq!(history.learning_rates[1], 1e-4);
        assert_eq!(history.learning_rates[2], 2.5e-5);
    }

    #[cfg(feature = "burn")]
    #[test]
    fn burn_2d_fit_batches_writes_burn_checkpoints() {
        type B = ::burn::backend::Autodiff<::burn::backend::Flex>;
        let device = Default::default();
        let mut config =
            Config2D::from_json_file("assets/models/examples/2D_demo/config.json").unwrap();
        config.train_dist_loss = "mae".to_string();
        config.train_loss_weights = vec![1.0, 0.2];
        config.train_background_reg = 0.0;

        let root = std::env::temp_dir().join(format!(
            "stardist_rs_burn_checkpoint_{}",
            std::process::id()
        ));
        let best = root.join("weights_best");
        let epoch = root.join("weights_now");
        let last = root.join("weights_last");
        for path in [&best, &epoch, &last] {
            let mut with_ext = path.clone();
            with_ext.set_extension("mpk");
            let _ = std::fs::remove_file(with_ext);
        }

        let model = burn::StarDist2D::<B>::init(config.clone(), &device);
        let batch = burn::StarDistData2DBatchTensors {
            x: ::burn::tensor::Tensor::<B, 4>::zeros([1, 1, 64, 64], &device),
            prob: ::burn::tensor::Tensor::<B, 4>::zeros([1, 1, 32, 32], &device),
            dist: ::burn::tensor::Tensor::<B, 4>::ones([1, config.n_rays + 1, 32, 32], &device),
            prob_class: None,
        };
        let validation_batch = batch.clone();
        let mut optimizer = ::burn::optim::AdamConfig::new().init::<B, burn::StarDist2D<B>>();

        let (_model, history) = burn::stardist_2d_fit_batches(
            model,
            &mut optimizer,
            &[batch],
            Some(validation_batch),
            1,
            1,
            1e-4,
            &device,
            Some(burn::StarDistBurnCheckpointConfig {
                best: Some(best.display().to_string()),
                epoch: Some(epoch.display().to_string()),
                last: Some(last.display().to_string()),
            }),
            None,
        )
        .unwrap();

        assert_eq!(history.checkpoint_files.len(), 3);
        for file in &history.checkpoint_files {
            assert!(std::path::Path::new(file).exists(), "{file}");
        }
        for path in [&best, &epoch, &last] {
            let mut with_ext = path.clone();
            with_ext.set_extension("mpk");
            let _ = std::fs::remove_file(with_ext);
        }
    }

    #[cfg(feature = "burn")]
    #[test]
    fn burn_2d_fit_batches_writes_scalar_log_file() {
        type B = ::burn::backend::Autodiff<::burn::backend::Flex>;
        let device = Default::default();
        let mut config =
            Config2D::from_json_file("assets/models/examples/2D_demo/config.json").unwrap();
        config.train_dist_loss = "mae".to_string();
        config.train_loss_weights = vec![1.0, 0.2];
        config.train_background_reg = 0.0;

        let log_dir =
            std::env::temp_dir().join(format!("stardist_rs_burn_logs_{}", std::process::id()));
        let log_file = log_dir.join("scalars.tsv");
        let _ = std::fs::remove_file(&log_file);
        if let Ok(entries) = std::fs::read_dir(&log_dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path
                    .file_name()
                    .and_then(|name| name.to_str())
                    .is_some_and(|name| name.starts_with("events.out.tfevents."))
                {
                    let _ = std::fs::remove_file(path);
                }
            }
        }
        let _ = std::fs::remove_dir(&log_dir);

        let model = burn::StarDist2D::<B>::init(config.clone(), &device);
        let batch = burn::StarDistData2DBatchTensors {
            x: ::burn::tensor::Tensor::<B, 4>::zeros([1, 1, 64, 64], &device),
            prob: ::burn::tensor::Tensor::<B, 4>::zeros([1, 1, 32, 32], &device),
            dist: ::burn::tensor::Tensor::<B, 4>::ones([1, config.n_rays + 1, 32, 32], &device),
            prob_class: None,
        };
        let validation_batch = batch.clone();
        let mut optimizer = ::burn::optim::AdamConfig::new().init::<B, burn::StarDist2D<B>>();

        let (_model, history) = burn::stardist_2d_fit_batches(
            model,
            &mut optimizer,
            &[batch],
            Some(validation_batch),
            2,
            1,
            1e-4,
            &device,
            None,
            Some(burn::StarDistBurnTensorBoardConfig {
                log_dir: log_dir.display().to_string(),
            }),
        )
        .unwrap();

        assert_eq!(history.log_files, vec![log_file.display().to_string()]);
        assert_eq!(history.event_files.len(), 1);
        assert!(history.event_files[0].contains("events.out.tfevents."));
        let contents = std::fs::read_to_string(&log_file).unwrap();
        let lines = contents.lines().collect::<Vec<_>>();
        assert_eq!(lines.len(), 3);
        assert_eq!(lines[0], "epoch\tloss\tval_loss");
        assert!(lines[1].starts_with("1\t"));
        assert!(lines[2].starts_with("2\t"));
        let event_bytes = std::fs::read(&history.event_files[0]).unwrap();
        assert!(event_bytes.len() > 32);
        assert!(event_bytes.windows(4).any(|window| window == b"loss"));
        assert!(event_bytes.windows(8).any(|window| window == b"val_loss"));
        assert!(event_bytes.windows(2).any(|window| window == b"lr"));
        let _ = std::fs::remove_file(&log_file);
        let _ = std::fs::remove_file(&history.event_files[0]);
        let _ = std::fs::remove_dir(&log_dir);
    }

    #[cfg(feature = "burn")]
    #[test]
    fn burn_2d_fit_images_trains_from_translated_data_generator() {
        type B = ::burn::backend::Autodiff<::burn::backend::Flex>;
        let device = Default::default();
        let mut config =
            Config2D::from_json_file("assets/models/examples/2D_demo/config.json").unwrap();
        config.train_patch_size = [64, 64];
        config.train_batch_size = 1;
        config.train_dist_loss = "mae".to_string();
        config.train_loss_weights = vec![1.0, 0.2];
        config.train_background_reg = 0.0;

        let mut x = vec![0.0f32; 64 * 64];
        let mut y = vec![0i32; 64 * 64];
        for yy in 24..40 {
            for xx in 24..40 {
                x[yy * 64 + xx] = 1.0;
                y[yy * 64 + xx] = 1;
            }
        }
        let x_images = [&x[..]];
        let y_images = [&y[..]];
        let x_shapes = [[64, 64, 1]];
        let y_shapes = [[64, 64]];
        let setup = StarDist2D::new(config.clone())
            .train(
                1,
                1,
                1,
                2,
                ClassesArg::Auto,
                None,
                Some(1),
                Some(1),
                Some(1),
            )
            .unwrap();
        let log_dir = std::env::temp_dir().join(format!(
            "stardist_rs_burn_image_logs_{}",
            std::process::id()
        ));
        let log_file = log_dir.join("scalars.tsv");
        let image_dir = log_dir.join("images");
        let _ = std::fs::remove_file(&log_file);
        if let Ok(entries) = std::fs::read_dir(&log_dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path
                    .file_name()
                    .and_then(|name| name.to_str())
                    .is_some_and(|name| name.starts_with("events.out.tfevents."))
                {
                    let _ = std::fs::remove_file(path);
                }
            }
        }
        if let Ok(entries) = std::fs::read_dir(&image_dir) {
            for entry in entries.flatten() {
                let _ = std::fs::remove_file(entry.path());
            }
        }
        let _ = std::fs::remove_dir(&image_dir);
        let _ = std::fs::remove_dir(&log_dir);
        let model = burn::StarDist2D::<B>::init(config.clone(), &device);
        let mut optimizer = ::burn::optim::AdamConfig::new().init::<B, burn::StarDist2D<B>>();

        let (_model, history) = burn::stardist_2d_fit_images(
            model,
            &mut optimizer,
            setup.data_train,
            &x_images,
            &x_shapes,
            &y_images,
            &y_shapes,
            Some((
                setup.data_val,
                &x_images,
                &x_shapes,
                &y_images,
                &y_shapes,
                setup.validation_n_take,
            )),
            setup.epochs,
            setup.steps_per_epoch,
            config.train_batch_size,
            config.train_learning_rate as f64,
            &device,
            7,
            None,
            Some(burn::StarDistBurnTensorBoardConfig {
                log_dir: log_dir.display().to_string(),
            }),
        )
        .unwrap();

        assert_eq!(history.loss.len(), 1);
        assert_eq!(history.val_loss.len(), 1);
        assert!(history.loss[0].is_finite(), "{:?}", history.loss);
        assert!(history.val_loss[0].is_finite(), "{:?}", history.val_loss);
        assert!(history.checkpoint_files.is_empty());
        assert_eq!(history.log_files, vec![log_file.display().to_string()]);
        assert_eq!(history.event_files.len(), 2);
        let image_event_file = history
            .event_files
            .iter()
            .find(|path| path.contains("images"))
            .unwrap();
        let image_event_bytes = std::fs::read(image_event_file).unwrap();
        assert!(
            image_event_bytes
                .windows(16)
                .any(|window| window == b"validation/input")
        );
        assert!(
            image_event_bytes
                .windows(8)
                .any(|window| window == [137, 80, 78, 71, 13, 10, 26, 10])
        );
        let _ = std::fs::remove_file(&log_file);
        for event_file in &history.event_files {
            let _ = std::fs::remove_file(event_file);
        }
        let _ = std::fs::remove_dir(&image_dir);
        let _ = std::fs::remove_dir(&log_dir);
    }

    #[cfg(feature = "burn")]
    #[test]
    fn burn_2d_fit_images_supports_shape_completion_batch_path() {
        type B = ::burn::backend::Autodiff<::burn::backend::Flex>;
        let device = Default::default();
        let mut config =
            Config2D::from_json_file("assets/models/examples/2D_demo/config.json").unwrap();
        config.train_patch_size = [64, 64];
        config.train_completion_crop = 8;
        config.train_shape_completion = true;
        config.train_batch_size = 1;
        config.train_dist_loss = "mae".to_string();
        config.train_loss_weights = vec![1.0, 0.2];
        config.train_background_reg = 0.0;

        let mut x = vec![0.0f32; 64 * 64];
        let mut y = vec![0i32; 64 * 64];
        for yy in 24..40 {
            for xx in 24..40 {
                x[yy * 64 + xx] = 1.0;
                y[yy * 64 + xx] = 1;
            }
        }
        for xx in 0..8 {
            y[xx] = 2;
        }
        let x_images = [&x[..]];
        let y_images = [&y[..]];
        let x_shapes = [[64, 64, 1]];
        let y_shapes = [[64, 64]];
        let setup = StarDist2D::new(config.clone())
            .train(
                1,
                1,
                1,
                2,
                ClassesArg::Auto,
                None,
                Some(1),
                Some(1),
                Some(1),
            )
            .unwrap();
        let model = burn::StarDist2D::<B>::init(config.clone(), &device);
        let mut optimizer = ::burn::optim::AdamConfig::new().init::<B, burn::StarDist2D<B>>();

        let (_model, history) = burn::stardist_2d_fit_images(
            model,
            &mut optimizer,
            setup.data_train,
            &x_images,
            &x_shapes,
            &y_images,
            &y_shapes,
            None,
            setup.epochs,
            setup.steps_per_epoch,
            config.train_batch_size,
            config.train_learning_rate as f64,
            &device,
            13,
            None,
            None,
        )
        .unwrap();

        assert_eq!(history.loss.len(), 1);
        assert!(history.val_loss.is_empty());
        assert!(history.loss[0].is_finite(), "{:?}", history.loss);
    }

    #[cfg(feature = "burn")]
    #[test]
    fn burn_3d_model_adds_multiclass_head_when_configured() {
        type B = ::burn::backend::Flex;
        let device = Default::default();
        let mut config =
            crate::Config3D::from_json_file("assets/models/examples/3D_demo/config.json").unwrap();
        config.n_classes = Some(2);
        config.n_channel_out = 1 + config.n_rays + 3;
        config.train_loss_weights = vec![1.0, 0.2, 1.0];
        config.train_class_weights = vec![1.0, 1.0, 1.0];

        let model = burn::StarDist3D::<B>::init(config, &device);
        let input = ::burn::tensor::Tensor::<B, 5>::zeros([1, 1, 8, 16, 16], &device);
        let outputs = model.forward(input);
        assert_eq!(outputs.prob.dims(), [1, 1, 8, 8, 8]);
        assert_eq!(outputs.dist.dims(), [1, 96, 8, 8, 8]);
        assert_eq!(outputs.prob_class.unwrap().dims(), [1, 3, 8, 8, 8]);
    }

    #[cfg(feature = "burn")]
    #[test]
    fn burn_3d_fit_images_trains_from_translated_data_generator() {
        type B = ::burn::backend::Autodiff<::burn::backend::Flex>;
        let device = Default::default();
        let mut config =
            crate::Config3D::from_json_file("assets/models/examples/3D_demo/config.json").unwrap();
        config.train_patch_size = [8, 16, 16];
        config.train_batch_size = 1;
        config.train_dist_loss = "mae".to_string();
        config.train_loss_weights = vec![1.0, 0.2];
        config.train_background_reg = 0.0;

        let mut x = vec![0.0f32; 8 * 16 * 16];
        let mut y = vec![0i32; 8 * 16 * 16];
        for zz in 2..6 {
            for yy in 5..11 {
                for xx in 5..11 {
                    x[(zz * 16 + yy) * 16 + xx] = 1.0;
                    y[(zz * 16 + yy) * 16 + xx] = 1;
                }
            }
        }
        let x_images = [&x[..]];
        let y_images = [&y[..]];
        let x_shapes = [[8, 16, 16, 1]];
        let y_shapes = [[8, 16, 16]];
        let setup = StarDist3D::new(config.clone())
            .train(
                1,
                1,
                1,
                2,
                ClassesArg::Auto,
                None,
                Some(1),
                Some(1),
                Some(1),
            )
            .unwrap();
        let log_dir = std::env::temp_dir().join(format!(
            "stardist_rs_burn_3d_image_logs_{}",
            std::process::id()
        ));
        let log_file = log_dir.join("scalars.tsv");
        let image_dir = log_dir.join("images");
        let _ = std::fs::remove_file(&log_file);
        if let Ok(entries) = std::fs::read_dir(&log_dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path
                    .file_name()
                    .and_then(|name| name.to_str())
                    .is_some_and(|name| name.starts_with("events.out.tfevents."))
                {
                    let _ = std::fs::remove_file(path);
                }
            }
        }
        if let Ok(entries) = std::fs::read_dir(&image_dir) {
            for entry in entries.flatten() {
                let _ = std::fs::remove_file(entry.path());
            }
        }
        let _ = std::fs::remove_dir(&image_dir);
        let _ = std::fs::remove_dir(&log_dir);
        let model = burn::StarDist3D::<B>::init(config.clone(), &device);
        let mut optimizer = ::burn::optim::AdamConfig::new().init::<B, burn::StarDist3D<B>>();

        let (_model, history) = burn::stardist_3d_fit_images(
            model,
            &mut optimizer,
            setup.data_train,
            &x_images,
            &x_shapes,
            &y_images,
            &y_shapes,
            Some((
                setup.data_val,
                &x_images,
                &x_shapes,
                &y_images,
                &y_shapes,
                setup.validation_n_take,
            )),
            setup.epochs,
            setup.steps_per_epoch,
            config.train_batch_size,
            config.train_learning_rate as f64,
            &device,
            11,
            None,
            Some(burn::StarDistBurnTensorBoardConfig {
                log_dir: log_dir.display().to_string(),
            }),
        )
        .unwrap();

        assert_eq!(history.loss.len(), 1);
        assert_eq!(history.val_loss.len(), 1);
        assert!(history.loss[0].is_finite(), "{:?}", history.loss);
        assert!(history.val_loss[0].is_finite(), "{:?}", history.val_loss);
        assert_eq!(history.log_files, vec![log_file.display().to_string()]);
        assert_eq!(history.event_files.len(), 2);
        let image_event_file = history
            .event_files
            .iter()
            .find(|path| path.contains("images"))
            .unwrap();
        let image_event_bytes = std::fs::read(image_event_file).unwrap();
        assert!(
            image_event_bytes
                .windows(18)
                .any(|window| window == b"validation/input_z")
        );
        assert!(
            image_event_bytes
                .windows(8)
                .any(|window| window == [137, 80, 78, 71, 13, 10, 26, 10])
        );
        let _ = std::fs::remove_file(&log_file);
        for event_file in &history.event_files {
            let _ = std::fs::remove_file(event_file);
        }
        let _ = std::fs::remove_dir(&image_dir);
        let _ = std::fs::remove_dir(&log_dir);
    }

    #[test]
    fn stardist_data_base_channels_as_tuple_splits_last_axis() {
        let data = StarDistDataBase::new(Some(2), vec![2, 2], vec![1, 1], 0.0, None, true).unwrap();
        let x = [1.0, 10.0, 2.0, 20.0, 3.0, 30.0, 4.0, 40.0];
        let channels = data.channels_as_tuple(&x, &[2, 2, 2]).unwrap();
        assert_eq!(
            channels,
            vec![vec![1.0, 2.0, 3.0, 4.0], vec![10.0, 20.0, 30.0, 40.0]]
        );
        let single = StarDistDataBase::new(None, vec![2, 2], vec![1, 1], 0.0, None, true).unwrap();
        assert_eq!(
            single.channels_as_tuple(&x[0..4], &[2, 2]).unwrap(),
            vec![vec![1.0, 10.0, 2.0, 20.0]]
        );
    }

    #[test]
    fn stardist_data_base_get_valid_inds_uses_all_cache() {
        let mut data =
            StarDistDataBase::new(None, vec![3, 3], vec![1, 1], 0.0, None, true).unwrap();
        let y = [0; 25];
        let inds = data.get_valid_inds(2, &y, &[5, 5], None, 0.5).unwrap();
        assert_eq!(inds[0], vec![1, 1, 1, 2, 2, 2, 3, 3, 3]);
        assert_eq!(inds[1], vec![1, 2, 3, 1, 2, 3, 1, 2, 3]);
        assert!(data.ind_cache_all.contains_key(&2));
    }

    #[test]
    fn stardist_data_base_get_valid_inds_prefers_foreground_and_falls_back_when_empty() {
        let mut data =
            StarDistDataBase::new(None, vec![3, 3], vec![1, 1], 1.0, None, true).unwrap();
        let mut y = [0; 25];
        y[2 * 5 + 2] = 1;
        let inds = data.get_valid_inds(1, &y, &[5, 5], None, 0.0).unwrap();
        assert_eq!(inds[0], vec![1, 1, 1, 2, 2, 2, 3, 3, 3]);
        assert_eq!(inds[1], vec![1, 2, 3, 1, 2, 3, 1, 2, 3]);
        assert!(data.ind_cache_fg.contains_key(&1));

        let mut empty_fg =
            StarDistDataBase::new(None, vec![3, 3], vec![1, 1], 1.0, None, true).unwrap();
        let y_empty = [0; 25];
        let fallback = empty_fg
            .get_valid_inds(0, &y_empty, &[5, 5], None, 0.0)
            .unwrap();
        assert_eq!(fallback[0].len(), 9);
        assert!(empty_fg.ind_cache_all.contains_key(&0));
    }

    #[test]
    fn stardist_data2d_getitem_builds_prob_and_dist_targets() {
        let base = StarDistDataBase::new(None, vec![3, 3], vec![1, 1], 0.0, None, true).unwrap();
        let mut data = StarDistData2D::new(base, 4, None, None, false, 0).unwrap();
        let x = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let y = [0, 0, 0, 0, 1, 0, 0, 0, 0];
        let batch = data
            .__getitem__(&[0], &[&x], &[[3, 3, 1]], &[&y], &[[3, 3]], &[0.5], 11)
            .unwrap();
        assert_eq!(batch.x_shape, [1, 3, 3, 1]);
        assert_eq!(batch.x, x);
        assert_eq!(batch.prob_shape, [1, 3, 3, 1]);
        assert_eq!(batch.dist_shape, [1, 3, 3, 5]);
        assert_eq!(batch.prob[4], 1.0);
        assert_eq!(batch.dist[4 * 5 + 4], 1.0);
        assert!(batch.prob_class.is_none());
    }

    #[test]
    fn stardist_data2d_getitem_masks_negative_labels_and_builds_class_targets() {
        let base = StarDistDataBase::new(None, vec![3, 3], vec![1, 1], 0.0, None, true).unwrap();
        let mut data = StarDistData2D::new(
            base,
            4,
            Some(1),
            Some(vec![ClassAssignment::Single(Some(1))]),
            false,
            0,
        )
        .unwrap();
        let x = [1.0; 9];
        let y = [0, 0, 0, 0, -1, 0, 0, 0, 0];
        let batch = data
            .__getitem__(&[0], &[&x], &[[3, 3, 1]], &[&y], &[[3, 3]], &[0.5], 11)
            .unwrap();
        assert_eq!(batch.prob[4], -1.0);
        assert_eq!(batch.dist[4 * 5 + 4], -1.0);
        assert_eq!(batch.prob_class_shape, Some([1, 3, 3, 2]));
        let prob_class = batch.prob_class.unwrap();
        assert_eq!(&prob_class[4 * 2..4 * 2 + 2], &[-1.0, -1.0]);
    }

    #[test]
    fn stardist_data2d_getitem_shape_completion_crops_x_and_clears_border_labels() {
        let base = StarDistDataBase::new(None, vec![5, 5], vec![1, 1], 0.0, None, true).unwrap();
        let mut data = StarDistData2D::new(base, 4, None, None, true, 1).unwrap();
        let x = (0..25).map(|v| v as f32).collect::<Vec<_>>();
        let mut y = vec![0i32; 25];
        y[0] = 1;
        y[1] = 1;
        y[5] = 1;
        y[12] = 2;

        let batch = data
            .__getitem__(&[0], &[&x], &[[5, 5, 1]], &[&y], &[[5, 5]], &[0.5], 11)
            .unwrap();

        assert_eq!(batch.x_shape, [1, 3, 3, 1]);
        assert_eq!(
            batch.x,
            vec![6.0, 7.0, 8.0, 11.0, 12.0, 13.0, 16.0, 17.0, 18.0]
        );
        assert_eq!(batch.prob_shape, [1, 3, 3, 1]);
        assert_eq!(batch.dist_shape, [1, 3, 3, 5]);
        assert_eq!(batch.prob[4], 1.0);
        assert_eq!(batch.dist[4 * 5 + 4], 1.0);
        assert_eq!(batch.dist[0 * 5 + 4], 0.0);
    }

    #[test]
    fn stardist_data3d_getitem_builds_prob_and_dist_targets() {
        let base =
            StarDistDataBase::new(None, vec![3, 3, 3], vec![1, 1, 1], 0.0, None, true).unwrap();
        let rays = crate::Rays {
            name: "Rays_Explicit".to_string(),
            kwargs: crate::RaysKwargs {
                n: 6,
                anisotropy: None,
                ..Default::default()
            },
            vertices: vec![
                [1.0, 0.0, 0.0],
                [-1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, -1.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, -1.0],
            ],
            faces: Vec::new(),
        };
        let mut data = StarDistData3D::new(base, rays, None, None, None);
        let x = (0..27).map(|v| v as f32).collect::<Vec<_>>();
        let mut y = vec![0i32; 27];
        y[(1 * 3 + 1) * 3 + 1] = 1;
        let batch = data
            .__getitem__(
                &[0],
                &[&x],
                &[[3, 3, 3, 1]],
                &[&y],
                &[[3, 3, 3]],
                &[0.5],
                11,
            )
            .unwrap();
        assert_eq!(batch.x_shape, [1, 3, 3, 3, 1]);
        assert_eq!(batch.x, x);
        assert_eq!(batch.prob_shape, [1, 3, 3, 3, 1]);
        assert_eq!(batch.dist_shape, [1, 3, 3, 3, 7]);
        let center = (1 * 3 + 1) * 3 + 1;
        assert_eq!(batch.prob[center], 1.0);
        assert_eq!(batch.dist[center * 7 + 6], 1.0);
        assert!(batch.prob_class.is_none());
    }

    #[test]
    fn stardist_data3d_getitem_masks_negative_labels_and_builds_class_targets() {
        let base =
            StarDistDataBase::new(None, vec![3, 3, 3], vec![1, 1, 1], 0.0, None, true).unwrap();
        let rays = crate::Rays {
            name: "Rays_Explicit".to_string(),
            kwargs: crate::RaysKwargs {
                n: 6,
                anisotropy: None,
                ..Default::default()
            },
            vertices: vec![
                [1.0, 0.0, 0.0],
                [-1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, -1.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, -1.0],
            ],
            faces: Vec::new(),
        };
        let mut data = StarDistData3D::new(
            base,
            rays,
            None,
            Some(1),
            Some(vec![ClassAssignment::Single(Some(1))]),
        );
        let x = vec![1.0; 27];
        let mut y = vec![0i32; 27];
        y[(1 * 3 + 1) * 3 + 1] = -1;
        let batch = data
            .__getitem__(
                &[0],
                &[&x],
                &[[3, 3, 3, 1]],
                &[&y],
                &[[3, 3, 3]],
                &[0.5],
                11,
            )
            .unwrap();
        let center = (1 * 3 + 1) * 3 + 1;
        assert_eq!(batch.prob[center], -1.0);
        assert_eq!(batch.dist[center * 7 + 6], -1.0);
        assert_eq!(batch.prob_class_shape, Some([1, 3, 3, 3, 2]));
        let prob_class = batch.prob_class.unwrap();
        assert_eq!(&prob_class[center * 2..center * 2 + 2], &[-1.0, -1.0]);
    }

    #[test]
    fn parse_classes_arg_auto_matches_python_branches() {
        assert_eq!(_parse_classes_arg(None, ClassesArg::Auto, 2).unwrap(), None);
        assert_eq!(
            _parse_classes_arg(Some(1), ClassesArg::Auto, 3).unwrap(),
            Some(vec![
                ClassAssignment::Single(Some(1)),
                ClassAssignment::Single(Some(1)),
                ClassAssignment::Single(Some(1)),
            ])
        );
        assert_eq!(
            _parse_classes_arg(Some(2), ClassesArg::Auto, 1).unwrap_err(),
            ClassesArgError::AutoMulticlassUnsupported
        );
    }

    #[test]
    fn parse_classes_arg_string_and_list_validate_like_python() {
        assert_eq!(
            _parse_classes_arg(Some(1), ClassesArg::String("manual".to_string()), 1).unwrap_err(),
            ClassesArgError::UnsupportedString
        );
        assert_eq!(
            _parse_classes_arg(Some(3), ClassesArg::Scalar(2), 2).unwrap(),
            Some(vec![
                ClassAssignment::Single(Some(2)),
                ClassAssignment::Single(Some(2)),
            ])
        );
        let classes = vec![
            ClassAssignment::Single(Some(1)),
            ClassAssignment::Dict(vec![(3, Some(1))]),
        ];
        assert_eq!(
            _parse_classes_arg(Some(2), ClassesArg::List(classes.clone()), 2).unwrap(),
            Some(classes)
        );
        assert_eq!(
            _parse_classes_arg(
                Some(2),
                ClassesArg::List(vec![ClassAssignment::Single(Some(1))]),
                2,
            )
            .unwrap_err(),
            ClassesArgError::WrongLength
        );
    }

    #[test]
    fn instances_from_prediction_dense_returns_labels_and_coords() {
        let mut config =
            Config2D::from_json_file("assets/models/examples/2D_demo/config.json").unwrap();
        config.n_rays = 8;
        config.grid = [1, 1];
        let model = StarDist2D::new(config);
        let mut prob = vec![0.0; 9];
        prob[4] = 0.9;
        let dist = vec![1.0; 9 * 8];

        let instances = model
            ._instances_from_prediction(
                [3, 3],
                &prob,
                [3, 3],
                &dist,
                None,
                None,
                Some(0.5),
                Some(0.5),
                true,
                None,
                None,
                true,
                false,
            )
            .unwrap();

        assert_eq!(instances.points, vec![[1.0, 1.0]]);
        assert_eq!(instances.prob, vec![0.9]);
        assert_eq!(instances.coord.shape(), &[1, 2, 8]);
        assert!(instances.labels.unwrap().iter().any(|v| *v == 1));
    }

    #[test]
    fn instances_from_prediction_sparse_returns_original_class_indices() {
        let mut config =
            Config2D::from_json_file("assets/models/examples/2D_demo/config.json").unwrap();
        config.n_rays = 8;
        config.grid = [1, 1];
        let model = StarDist2D::new(config);
        let prob = [0.8, 0.9];
        let dist = vec![2.0; 2 * 8];
        let points = [[5.0, 5.0], [5.0, 5.0]];
        let prob_class = [0.1, 0.9, 0.7, 0.3];

        let instances = model
            ._instances_from_prediction(
                [11, 11],
                &prob,
                [1, 2],
                &dist,
                Some(&points),
                Some((&prob_class, 2)),
                Some(0.5),
                Some(0.5),
                false,
                None,
                None,
                true,
                false,
            )
            .unwrap();

        assert_eq!(instances.points, vec![[5.0, 5.0]]);
        assert_eq!(instances.prob, vec![0.9]);
        assert_eq!(instances.class_prob, Some(vec![0.7, 0.3]));
        assert_eq!(instances.class_prob_channels, Some(2));
        assert_eq!(instances.class_id, Some(vec![0]));
        assert_eq!(instances.labels, None);
    }

    #[test]
    fn instances_from_prediction_3d_dense_returns_labels_and_rays() {
        let mut config =
            crate::Config3D::from_json_file("assets/models/examples/3D_demo/config.json").unwrap();
        let rays = crate::RaysGoldenSpiral::new(8, None).unwrap().into_rays();
        config.n_rays = 8;
        config.grid = [1, 1, 1];
        config.rays_json = rays.to_json();
        let model = StarDist3D::new(config);
        let mut prob = vec![0.0; 8];
        prob[7] = 0.9;
        let dist = vec![1.0; 8 * 8];

        let instances = model
            ._instances_from_prediction(
                [4, 4, 4],
                &prob,
                [2, 2, 2],
                &dist,
                None,
                None,
                Some(0.5),
                Some(0.5),
                None,
                true,
                None,
                None,
                true,
                false,
                false,
                crate::PolyhedronRenderMode::Bbox,
            )
            .unwrap();

        assert_eq!(instances.points, vec![[1.0, 1.0, 1.0]]);
        assert_eq!(instances.prob, vec![0.9]);
        assert_eq!(instances.rays_vertices.len(), 8);
        assert_eq!(instances.rays_faces.len(), 12);
        assert!(instances.labels.unwrap().iter().any(|v| *v == 1));
    }

    #[test]
    fn instances_from_prediction_3d_sparse_returns_class_indices() {
        let mut config =
            crate::Config3D::from_json_file("assets/models/examples/3D_demo/config.json").unwrap();
        let rays = crate::RaysGoldenSpiral::new(8, None).unwrap().into_rays();
        config.n_rays = 8;
        config.grid = [1, 1, 1];
        config.rays_json = rays.to_json();
        let model = StarDist3D::new(config);
        let prob = [0.8, 0.9];
        let points = [[5.0, 5.0, 5.0], [5.0, 5.0, 5.0]];
        let dist = vec![2.0; 2 * 8];
        let prob_class = [0.2, 0.8, 0.6, 0.4];

        let instances = model
            ._instances_from_prediction(
                [10, 10, 10],
                &prob,
                [1, 1, 2],
                &dist,
                Some(&points),
                Some((&prob_class, 2)),
                Some(0.5),
                Some(0.5),
                None,
                false,
                None,
                None,
                true,
                false,
                true,
                crate::PolyhedronRenderMode::Bbox,
            )
            .unwrap();

        assert_eq!(instances.points, vec![[5.0, 5.0, 5.0]]);
        assert_eq!(instances.prob, vec![0.9]);
        assert_eq!(instances.class_prob, Some(vec![0.6, 0.4]));
        assert_eq!(instances.class_prob_channels, Some(2));
        assert_eq!(instances.class_id, Some(vec![0]));
        assert_eq!(instances.labels, None);
    }

    #[cfg(all(feature = "burn", feature = "hdf5"))]
    #[test]
    fn burn_2d_model_loads_keras_weights_and_runs_forward() {
        type B = ::burn::backend::Flex;

        let weights_path = "stardist/models/examples/2D_demo/weights_best.h5";
        if !std::path::Path::new(weights_path).exists() {
            return;
        }
        let device = Default::default();
        let config =
            Config2D::from_json_file("assets/models/examples/2D_demo/config.json").unwrap();
        let weights = crate::weights::load_keras_hdf5_weights(weights_path).unwrap();
        let model = burn::StarDist2D::<B>::init(config, &device)
            .load_keras_weights(&weights, &device)
            .unwrap();
        let input = ::burn::tensor::Tensor::<B, 4>::zeros([1, 1, 64, 64], &device);
        let outputs = model.forward(input);
        assert_eq!(outputs.prob.dims(), [1, 1, 32, 32]);
        assert_eq!(outputs.dist.dims(), [1, 32, 32, 32]);
    }

    #[cfg(all(feature = "burn", feature = "hdf5"))]
    #[test]
    fn burn_2d_model_matches_python_fixture_when_available() {
        type B = ::burn::backend::Flex;

        let fixture_path = "tests/fixtures/2d_demo_inference.npz";
        let weights_path = "stardist/models/examples/2D_demo/weights_best.h5";
        if !std::path::Path::new(fixture_path).exists() {
            return;
        }
        if !std::path::Path::new(weights_path).exists() {
            return;
        }

        let device = Default::default();
        let config =
            Config2D::from_json_file("assets/models/examples/2D_demo/config.json").unwrap();
        let weights = crate::weights::load_keras_hdf5_weights(weights_path).unwrap();
        let fixture = crate::fixtures::load_stardist_2d_inference_fixture(fixture_path).unwrap();
        let model = burn::StarDist2D::<B>::init(config, &device)
            .load_keras_weights(&weights, &device)
            .unwrap();
        let input = ::burn::tensor::Tensor::<B, 4>::from_data(
            ::burn::tensor::TensorData::new(fixture.input_nchw.values, [1, 1, 64, 64]),
            &device,
        );
        let outputs = model.forward(input);
        let prob_data = outputs.prob.into_data();
        let dist_data = outputs.dist.into_data();
        let prob = prob_data.as_slice::<f32>().unwrap();
        let dist = dist_data.as_slice::<f32>().unwrap();

        assert_eq!(fixture.prob_nchw.shape, vec![1, 1, 32, 32]);
        assert_eq!(fixture.dist_nchw.shape, vec![1, 32, 32, 32]);
        for i in 0..prob.len() {
            assert!(
                (prob[i] - fixture.prob_nchw.values[i]).abs() < 1e-4,
                "prob mismatch at {i}: rust={} python={}",
                prob[i],
                fixture.prob_nchw.values[i]
            );
        }
        for i in 0..dist.len() {
            assert!(
                (dist[i] - fixture.dist_nchw.values[i]).abs() < 1e-3,
                "dist mismatch at {i}: rust={} python={}",
                dist[i],
                fixture.dist_nchw.values[i]
            );
        }
    }

    #[cfg(all(feature = "burn", feature = "hdf5"))]
    #[test]
    fn burn_2d_model_matches_python_instances_fixture_when_available() {
        type B = ::burn::backend::Flex;

        let fixture_path = "tests/fixtures/2d_demo_instances.npz";
        let weights_path = "stardist/models/examples/2D_demo/weights_best.h5";
        if !std::path::Path::new(fixture_path).exists() {
            return;
        }
        if !std::path::Path::new(weights_path).exists() {
            return;
        }

        let device = Default::default();
        let config =
            Config2D::from_json_file("assets/models/examples/2D_demo/config.json").unwrap();
        let weights = crate::weights::load_keras_hdf5_weights(weights_path).unwrap();
        let fixture = crate::fixtures::load_stardist_2d_instances_fixture(fixture_path).unwrap();
        let model = burn::StarDist2D::<B>::init(config.clone(), &device)
            .load_keras_weights(&weights, &device)
            .unwrap();
        let input = ::burn::tensor::Tensor::<B, 4>::from_data(
            ::burn::tensor::TensorData::new(fixture.input_nchw.values, [1, 1, 64, 64]),
            &device,
        );
        let outputs = model.forward(input);
        let prob_data = outputs.prob.into_data();
        let dist_data = outputs.dist.into_data();
        let prob = prob_data.as_slice::<f32>().unwrap();
        let dist_nchw = dist_data.as_slice::<f32>().unwrap();
        let mut dist_yxc = vec![0.0f32; 32 * 32 * config.n_rays];
        for ray in 0..config.n_rays {
            for y in 0..32 {
                for x in 0..32 {
                    dist_yxc[(y * 32 + x) * config.n_rays + ray] =
                        dist_nchw[(ray * 32 + y) * 32 + x];
                }
            }
        }

        let instances = StarDist2D::new(config)
            ._instances_from_prediction(
                [64, 64],
                prob,
                [32, 32],
                &dist_yxc,
                None,
                None,
                None,
                None,
                true,
                None,
                None,
                true,
                true,
            )
            .unwrap();

        let labels = instances.labels.unwrap();
        assert_eq!(fixture.labels.shape, vec![64, 64]);
        assert_eq!(
            labels.iter().copied().collect::<Vec<_>>(),
            fixture.labels.values
        );
        assert_eq!(fixture.points.shape, vec![instances.points.len(), 2]);
        for (i, point) in instances.points.iter().enumerate() {
            assert!(
                (point[0] - fixture.points.values[i * 2]).abs() < 1e-4,
                "point y mismatch at {i}: rust={} python={}",
                point[0],
                fixture.points.values[i * 2]
            );
            assert!(
                (point[1] - fixture.points.values[i * 2 + 1]).abs() < 1e-4,
                "point x mismatch at {i}: rust={} python={}",
                point[1],
                fixture.points.values[i * 2 + 1]
            );
        }
        assert_eq!(instances.prob.len(), fixture.prob.values.len());
        for i in 0..instances.prob.len() {
            assert!(
                (instances.prob[i] - fixture.prob.values[i]).abs() < 1e-4,
                "instance prob mismatch at {i}: rust={} python={}",
                instances.prob[i],
                fixture.prob.values[i]
            );
        }
        assert_eq!(fixture.coord.shape, instances.coord.shape());
        for (i, value) in instances.coord.iter().enumerate() {
            assert!(
                (*value - fixture.coord.values[i]).abs() < 1e-3,
                "coord mismatch at {i}: rust={} python={}",
                *value,
                fixture.coord.values[i]
            );
        }
    }

    #[cfg(all(feature = "burn", feature = "hdf5"))]
    #[test]
    fn burn_3d_model_loads_keras_weights_and_runs_forward() {
        type B = ::burn::backend::Flex;

        let weights_path = "stardist/models/examples/3D_demo/weights_best.h5";
        if !std::path::Path::new(weights_path).exists() {
            return;
        }
        let device = Default::default();
        let config =
            crate::Config3D::from_json_file("assets/models/examples/3D_demo/config.json").unwrap();
        let weights = crate::weights::load_keras_hdf5_weights(weights_path).unwrap();
        let model = burn::StarDist3D::<B>::init(config, &device)
            .load_keras_weights(&weights, &device)
            .unwrap();
        let input = ::burn::tensor::Tensor::<B, 5>::zeros([1, 1, 8, 16, 16], &device);
        let outputs = model.forward(input);
        assert_eq!(outputs.prob.dims(), [1, 1, 8, 8, 8]);
        assert_eq!(outputs.dist.dims(), [1, 96, 8, 8, 8]);
    }

    #[cfg(all(feature = "burn", feature = "hdf5"))]
    #[test]
    fn burn_3d_model_matches_python_fixture_when_available() {
        type B = ::burn::backend::Flex;

        let fixture_path = "tests/fixtures/3d_demo_inference.npz";
        let weights_path = "stardist/models/examples/3D_demo/weights_best.h5";
        if !std::path::Path::new(fixture_path).exists() {
            return;
        }
        if !std::path::Path::new(weights_path).exists() {
            return;
        }

        let device = Default::default();
        let config =
            crate::Config3D::from_json_file("assets/models/examples/3D_demo/config.json").unwrap();
        let weights = crate::weights::load_keras_hdf5_weights(weights_path).unwrap();
        let fixture = crate::fixtures::load_stardist_3d_inference_fixture(fixture_path).unwrap();
        let model = burn::StarDist3D::<B>::init(config, &device)
            .load_keras_weights(&weights, &device)
            .unwrap();
        let input = ::burn::tensor::Tensor::<B, 5>::from_data(
            ::burn::tensor::TensorData::new(fixture.input_ncdhw.values, [1, 1, 8, 16, 16]),
            &device,
        );
        let outputs = model.forward(input);
        let prob_data = outputs.prob.into_data();
        let dist_data = outputs.dist.into_data();
        let prob = prob_data.as_slice::<f32>().unwrap();
        let dist = dist_data.as_slice::<f32>().unwrap();

        assert_eq!(fixture.prob_ncdhw.shape, vec![1, 1, 8, 8, 8]);
        assert_eq!(fixture.dist_ncdhw.shape, vec![1, 96, 8, 8, 8]);
        for i in 0..prob.len() {
            assert!(
                (prob[i] - fixture.prob_ncdhw.values[i]).abs() < 1e-4,
                "prob mismatch at {i}: rust={} python={}",
                prob[i],
                fixture.prob_ncdhw.values[i]
            );
        }
        for i in 0..dist.len() {
            assert!(
                (dist[i] - fixture.dist_ncdhw.values[i]).abs() < 1e-3,
                "dist mismatch at {i}: rust={} python={}",
                dist[i],
                fixture.dist_ncdhw.values[i]
            );
        }
    }

    #[cfg(all(feature = "burn", feature = "hdf5"))]
    #[test]
    fn burn_3d_model_matches_python_instances_fixture_when_available() {
        type B = ::burn::backend::Flex;

        let fixture_path = "tests/fixtures/3d_demo_instances.npz";
        let weights_path = "stardist/models/examples/3D_demo/weights_best.h5";
        if !std::path::Path::new(fixture_path).exists() {
            return;
        }
        if !std::path::Path::new(weights_path).exists() {
            return;
        }

        let device = Default::default();
        let config =
            crate::Config3D::from_json_file("assets/models/examples/3D_demo/config.json").unwrap();
        let weights = crate::weights::load_keras_hdf5_weights(weights_path).unwrap();
        let fixture = crate::fixtures::load_stardist_3d_instances_fixture(fixture_path).unwrap();
        let model = burn::StarDist3D::<B>::init(config.clone(), &device)
            .load_keras_weights(&weights, &device)
            .unwrap();
        let input = ::burn::tensor::Tensor::<B, 5>::from_data(
            ::burn::tensor::TensorData::new(fixture.input_ncdhw.values, [1, 1, 8, 16, 16]),
            &device,
        );
        let outputs = model.forward(input);
        let prob_data = outputs.prob.into_data();
        let dist_data = outputs.dist.into_data();
        let prob = prob_data.as_slice::<f32>().unwrap();
        let dist_ncdhw = dist_data.as_slice::<f32>().unwrap();
        let mut dist_zyxc = vec![0.0f32; 8 * 8 * 8 * config.n_rays];
        for ray in 0..config.n_rays {
            for z in 0..8 {
                for y in 0..8 {
                    for x in 0..8 {
                        dist_zyxc[((z * 8 + y) * 8 + x) * config.n_rays + ray] =
                            dist_ncdhw[((ray * 8 + z) * 8 + y) * 8 + x];
                    }
                }
            }
        }

        let mut stardist = StarDist3D::new(config);
        stardist
            .set_thresholds(StarDistThresholds {
                prob: 0.7079326,
                nms: 0.3,
            })
            .unwrap();
        let instances = stardist
            ._instances_from_prediction(
                [8, 16, 16],
                prob,
                [8, 8, 8],
                &dist_zyxc,
                None,
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
                crate::PolyhedronRenderMode::Full,
            )
            .unwrap();

        let labels = instances.labels.unwrap();
        assert_eq!(fixture.labels.shape, vec![8, 16, 16]);
        assert_eq!(
            labels.iter().copied().collect::<Vec<_>>(),
            fixture.labels.values
        );
        assert_eq!(fixture.points.shape, vec![instances.points.len(), 3]);
        for (i, point) in instances.points.iter().enumerate() {
            for axis in 0..3 {
                assert!(
                    (point[axis] - fixture.points.values[i * 3 + axis]).abs() < 1e-4,
                    "point axis {axis} mismatch at {i}: rust={} python={}",
                    point[axis],
                    fixture.points.values[i * 3 + axis]
                );
            }
        }
        assert_eq!(fixture.dist.shape, vec![instances.points.len(), 96]);
        for i in 0..instances.dist.len() {
            assert!(
                (instances.dist[i] - fixture.dist.values[i]).abs() < 1e-3,
                "dist mismatch at {i}: rust={} python={}",
                instances.dist[i],
                fixture.dist.values[i]
            );
        }
        assert_eq!(instances.prob.len(), fixture.prob.values.len());
        for i in 0..instances.prob.len() {
            assert!(
                (instances.prob[i] - fixture.prob.values[i]).abs() < 1e-4,
                "prob mismatch at {i}: rust={} python={}",
                instances.prob[i],
                fixture.prob.values[i]
            );
        }
    }
}
