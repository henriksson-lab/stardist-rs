use std::{fs::File, path::Path};

use ndarray::{Array, IxDyn};
use ndarray_npy::{NpzReader, ReadNpzError};

#[derive(Clone, Debug)]
pub struct FixtureTensor {
    pub shape: Vec<usize>,
    pub values: Vec<f32>,
}

#[derive(Clone, Debug)]
pub struct FixtureTensorU32 {
    pub shape: Vec<usize>,
    pub values: Vec<u32>,
}

#[derive(Clone, Debug)]
pub struct StarDist2DInferenceFixture {
    pub input_nhwc: FixtureTensor,
    pub input_nchw: FixtureTensor,
    pub prob_nhwc: FixtureTensor,
    pub prob_nchw: FixtureTensor,
    pub dist_nhwc: FixtureTensor,
    pub dist_nchw: FixtureTensor,
}

#[derive(Clone, Debug)]
pub struct StarDist3DInferenceFixture {
    pub input_ndhwc: FixtureTensor,
    pub input_ncdhw: FixtureTensor,
    pub prob_ndhwc: FixtureTensor,
    pub prob_ncdhw: FixtureTensor,
    pub dist_ndhwc: FixtureTensor,
    pub dist_ncdhw: FixtureTensor,
}

#[derive(Clone, Debug)]
pub struct StarDist2DInstancesFixture {
    pub input_nchw: FixtureTensor,
    pub labels: FixtureTensorU32,
    pub coord: FixtureTensor,
    pub points: FixtureTensor,
    pub prob: FixtureTensor,
}

#[derive(Clone, Debug)]
pub struct StarDist3DInstancesFixture {
    pub input_ncdhw: FixtureTensor,
    pub labels: FixtureTensorU32,
    pub dist: FixtureTensor,
    pub points: FixtureTensor,
    pub prob: FixtureTensor,
}

pub fn load_stardist_2d_inference_fixture(
    path: impl AsRef<Path>,
) -> Result<StarDist2DInferenceFixture, FixtureError> {
    let file = File::open(path)?;
    let mut npz = NpzReader::new(file)?;

    let input_nhwc: Array<f32, IxDyn> = npz.by_name("input_nhwc.npy")?;
    let input_nchw: Array<f32, IxDyn> = npz.by_name("input_nchw.npy")?;
    let prob_nhwc: Array<f32, IxDyn> = npz.by_name("prob_nhwc.npy")?;
    let prob_nchw: Array<f32, IxDyn> = npz.by_name("prob_nchw.npy")?;
    let dist_nhwc: Array<f32, IxDyn> = npz.by_name("dist_nhwc.npy")?;
    let dist_nchw: Array<f32, IxDyn> = npz.by_name("dist_nchw.npy")?;

    Ok(StarDist2DInferenceFixture {
        input_nhwc: FixtureTensor {
            shape: input_nhwc.shape().to_vec(),
            values: input_nhwc.iter().copied().collect(),
        },
        input_nchw: FixtureTensor {
            shape: input_nchw.shape().to_vec(),
            values: input_nchw.iter().copied().collect(),
        },
        prob_nhwc: FixtureTensor {
            shape: prob_nhwc.shape().to_vec(),
            values: prob_nhwc.iter().copied().collect(),
        },
        prob_nchw: FixtureTensor {
            shape: prob_nchw.shape().to_vec(),
            values: prob_nchw.iter().copied().collect(),
        },
        dist_nhwc: FixtureTensor {
            shape: dist_nhwc.shape().to_vec(),
            values: dist_nhwc.iter().copied().collect(),
        },
        dist_nchw: FixtureTensor {
            shape: dist_nchw.shape().to_vec(),
            values: dist_nchw.iter().copied().collect(),
        },
    })
}

pub fn load_stardist_3d_inference_fixture(
    path: impl AsRef<Path>,
) -> Result<StarDist3DInferenceFixture, FixtureError> {
    let file = File::open(path)?;
    let mut npz = NpzReader::new(file)?;

    let input_ndhwc: Array<f32, IxDyn> = npz.by_name("input_ndhwc.npy")?;
    let input_ncdhw: Array<f32, IxDyn> = npz.by_name("input_ncdhw.npy")?;
    let prob_ndhwc: Array<f32, IxDyn> = npz.by_name("prob_ndhwc.npy")?;
    let prob_ncdhw: Array<f32, IxDyn> = npz.by_name("prob_ncdhw.npy")?;
    let dist_ndhwc: Array<f32, IxDyn> = npz.by_name("dist_ndhwc.npy")?;
    let dist_ncdhw: Array<f32, IxDyn> = npz.by_name("dist_ncdhw.npy")?;

    Ok(StarDist3DInferenceFixture {
        input_ndhwc: FixtureTensor {
            shape: input_ndhwc.shape().to_vec(),
            values: input_ndhwc.iter().copied().collect(),
        },
        input_ncdhw: FixtureTensor {
            shape: input_ncdhw.shape().to_vec(),
            values: input_ncdhw.iter().copied().collect(),
        },
        prob_ndhwc: FixtureTensor {
            shape: prob_ndhwc.shape().to_vec(),
            values: prob_ndhwc.iter().copied().collect(),
        },
        prob_ncdhw: FixtureTensor {
            shape: prob_ncdhw.shape().to_vec(),
            values: prob_ncdhw.iter().copied().collect(),
        },
        dist_ndhwc: FixtureTensor {
            shape: dist_ndhwc.shape().to_vec(),
            values: dist_ndhwc.iter().copied().collect(),
        },
        dist_ncdhw: FixtureTensor {
            shape: dist_ncdhw.shape().to_vec(),
            values: dist_ncdhw.iter().copied().collect(),
        },
    })
}

pub fn load_stardist_2d_instances_fixture(
    path: impl AsRef<Path>,
) -> Result<StarDist2DInstancesFixture, FixtureError> {
    let file = File::open(path)?;
    let mut npz = NpzReader::new(file)?;

    let input_nchw: Array<f32, IxDyn> = npz.by_name("input_nchw.npy")?;
    let labels: Array<u32, IxDyn> = npz.by_name("labels.npy")?;
    let coord: Array<f32, IxDyn> = npz.by_name("coord.npy")?;
    let points: Array<f32, IxDyn> = npz.by_name("points.npy")?;
    let prob: Array<f32, IxDyn> = npz.by_name("prob.npy")?;

    Ok(StarDist2DInstancesFixture {
        input_nchw: FixtureTensor {
            shape: input_nchw.shape().to_vec(),
            values: input_nchw.iter().copied().collect(),
        },
        labels: FixtureTensorU32 {
            shape: labels.shape().to_vec(),
            values: labels.iter().copied().collect(),
        },
        coord: FixtureTensor {
            shape: coord.shape().to_vec(),
            values: coord.iter().copied().collect(),
        },
        points: FixtureTensor {
            shape: points.shape().to_vec(),
            values: points.iter().copied().collect(),
        },
        prob: FixtureTensor {
            shape: prob.shape().to_vec(),
            values: prob.iter().copied().collect(),
        },
    })
}

pub fn load_stardist_3d_instances_fixture(
    path: impl AsRef<Path>,
) -> Result<StarDist3DInstancesFixture, FixtureError> {
    let file = File::open(path)?;
    let mut npz = NpzReader::new(file)?;

    let input_ncdhw: Array<f32, IxDyn> = npz.by_name("input_ncdhw.npy")?;
    let labels: Array<u32, IxDyn> = npz.by_name("labels.npy")?;
    let dist: Array<f32, IxDyn> = npz.by_name("dist.npy")?;
    let points: Array<f32, IxDyn> = npz.by_name("points.npy")?;
    let prob: Array<f32, IxDyn> = npz.by_name("prob.npy")?;

    Ok(StarDist3DInstancesFixture {
        input_ncdhw: FixtureTensor {
            shape: input_ncdhw.shape().to_vec(),
            values: input_ncdhw.iter().copied().collect(),
        },
        labels: FixtureTensorU32 {
            shape: labels.shape().to_vec(),
            values: labels.iter().copied().collect(),
        },
        dist: FixtureTensor {
            shape: dist.shape().to_vec(),
            values: dist.iter().copied().collect(),
        },
        points: FixtureTensor {
            shape: points.shape().to_vec(),
            values: points.iter().copied().collect(),
        },
        prob: FixtureTensor {
            shape: prob.shape().to_vec(),
            values: prob.iter().copied().collect(),
        },
    })
}

#[derive(Debug, thiserror::Error)]
pub enum FixtureError {
    #[error("failed to read fixture file")]
    Io(#[from] std::io::Error),
    #[error("failed to read NPZ fixture")]
    Npz(#[from] ReadNpzError),
}
