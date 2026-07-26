use std::{fs::File, path::Path};

use serde::Deserialize;

use crate::RaysJson;

#[derive(Clone, Debug, Deserialize, PartialEq)]
pub struct Config2D {
    pub n_dim: usize,
    pub axes: String,
    pub n_channel_in: usize,
    pub n_channel_out: usize,
    pub train_checkpoint: String,
    pub train_checkpoint_last: String,
    pub train_checkpoint_epoch: String,
    pub n_rays: usize,
    pub grid: [usize; 2],
    #[serde(default)]
    pub n_classes: Option<usize>,
    pub backbone: String,
    pub unet_n_depth: usize,
    pub unet_kernel_size: [usize; 2],
    pub unet_n_filter_base: usize,
    pub unet_n_conv_per_depth: usize,
    pub unet_pool: [usize; 2],
    pub unet_activation: String,
    pub unet_last_activation: String,
    pub unet_batch_norm: bool,
    pub unet_dropout: f32,
    pub unet_prefix: String,
    pub net_conv_after_unet: usize,
    pub net_input_shape: [Option<usize>; 3],
    pub net_mask_shape: [Option<usize>; 3],
    pub train_shape_completion: bool,
    pub train_completion_crop: usize,
    pub train_patch_size: [usize; 2],
    pub train_background_reg: f32,
    #[serde(default = "default_train_foreground_only")]
    pub train_foreground_only: f32,
    #[serde(default = "default_train_sample_cache")]
    pub train_sample_cache: bool,
    pub train_dist_loss: String,
    pub train_loss_weights: Vec<f32>,
    #[serde(default = "default_train_class_weights_single_class")]
    pub train_class_weights: Vec<f32>,
    pub train_epochs: usize,
    pub train_steps_per_epoch: usize,
    pub train_learning_rate: f32,
    pub train_batch_size: usize,
    pub train_n_val_patches: Option<usize>,
    pub train_tensorboard: bool,
    pub train_reduce_lr: TrainReduceLr,
    pub use_gpu: bool,
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
pub struct TrainReduceLr {
    pub factor: f32,
    pub patience: usize,
    pub min_delta: f32,
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
pub struct Config3D {
    pub n_dim: usize,
    pub axes: String,
    pub n_channel_in: usize,
    pub n_channel_out: usize,
    pub train_checkpoint: String,
    pub train_checkpoint_last: String,
    pub train_checkpoint_epoch: String,
    pub n_rays: usize,
    pub grid: [usize; 3],
    pub anisotropy: [f32; 3],
    pub backbone: String,
    pub rays_json: RaysJson,
    #[serde(default)]
    pub n_classes: Option<usize>,
    #[serde(default = "default_unet_n_depth_3d")]
    pub unet_n_depth: usize,
    #[serde(default = "default_unet_kernel_size_3d")]
    pub unet_kernel_size: [usize; 3],
    #[serde(default = "default_unet_n_filter_base")]
    pub unet_n_filter_base: usize,
    #[serde(default = "default_unet_n_conv_per_depth")]
    pub unet_n_conv_per_depth: usize,
    #[serde(default = "default_unet_pool_3d")]
    pub unet_pool: [usize; 3],
    #[serde(default = "default_relu")]
    pub unet_activation: String,
    #[serde(default = "default_relu")]
    pub unet_last_activation: String,
    #[serde(default)]
    pub unet_batch_norm: bool,
    #[serde(default)]
    pub unet_dropout: f32,
    #[serde(default = "default_unet_expansion")]
    pub unet_expansion: usize,
    #[serde(default)]
    pub unet_prefix: String,
    #[serde(default = "default_net_conv_after_unet")]
    pub net_conv_after_unet: usize,
    pub resnet_n_blocks: usize,
    pub resnet_kernel_size: [usize; 3],
    pub resnet_kernel_init: String,
    pub resnet_n_filter_base: usize,
    pub resnet_n_conv_per_block: usize,
    pub resnet_activation: String,
    pub resnet_batch_norm: bool,
    pub net_conv_after_resnet: usize,
    pub net_input_shape: [Option<usize>; 4],
    pub net_mask_shape: [Option<usize>; 4],
    pub train_patch_size: [usize; 3],
    pub train_background_reg: f32,
    #[serde(default = "default_train_foreground_only")]
    pub train_foreground_only: f32,
    #[serde(default = "default_train_sample_cache")]
    pub train_sample_cache: bool,
    pub train_dist_loss: String,
    pub train_loss_weights: Vec<f32>,
    #[serde(default = "default_train_class_weights_single_class")]
    pub train_class_weights: Vec<f32>,
    pub train_epochs: usize,
    pub train_steps_per_epoch: usize,
    pub train_learning_rate: f32,
    pub train_batch_size: usize,
    pub train_n_val_patches: Option<usize>,
    pub train_tensorboard: bool,
    pub train_reduce_lr: TrainReduceLr,
    pub use_gpu: bool,
}

impl Config2D {
    pub fn from_json_file(path: impl AsRef<Path>) -> Result<Self, ConfigError> {
        let file = File::open(path)?;
        let config = serde_json::from_reader(file)?;
        Ok(config)
    }
}

impl Config3D {
    pub fn from_json_file(path: impl AsRef<Path>) -> Result<Self, ConfigError> {
        let file = File::open(path)?;
        let config = serde_json::from_reader(file)?;
        Ok(config)
    }
}

#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    #[error("failed to read config file")]
    Io(#[from] std::io::Error),
    #[error("failed to parse config JSON")]
    Json(#[from] serde_json::Error),
}

fn default_train_foreground_only() -> f32 {
    0.9
}

fn default_train_sample_cache() -> bool {
    true
}

fn default_train_class_weights_single_class() -> Vec<f32> {
    vec![1.0, 1.0]
}

fn default_unet_n_depth_3d() -> usize {
    2
}

fn default_unet_kernel_size_3d() -> [usize; 3] {
    [3, 3, 3]
}

fn default_unet_n_filter_base() -> usize {
    32
}

fn default_unet_n_conv_per_depth() -> usize {
    2
}

fn default_unet_pool_3d() -> [usize; 3] {
    [2, 2, 2]
}

fn default_relu() -> String {
    "relu".to_string()
}

fn default_unet_expansion() -> usize {
    2
}

fn default_net_conv_after_unet() -> usize {
    128
}
