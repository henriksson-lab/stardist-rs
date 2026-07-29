use std::collections::BTreeMap;
use std::path::Path;

#[derive(Clone, Debug, PartialEq)]
pub struct KerasWeight {
    pub name: String,
    pub shape: Vec<usize>,
    pub values: Vec<f32>,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct KerasWeights {
    pub tensors: BTreeMap<String, KerasWeight>,
}

impl KerasWeights {
    pub fn get(&self, name: &str) -> Option<&KerasWeight> {
        self.tensors.get(name)
    }
}

pub fn keras_conv2d_kernel_to_burn(shape: [usize; 4], values: &[f32]) -> KerasWeight {
    let [kh, kw, channels_in, channels_out] = shape;
    assert_eq!(values.len(), kh * kw * channels_in * channels_out);
    let mut converted = vec![0.0; values.len()];
    for y in 0..kh {
        for x in 0..kw {
            for c_in in 0..channels_in {
                for c_out in 0..channels_out {
                    let keras_index = (((y * kw + x) * channels_in + c_in) * channels_out) + c_out;
                    let burn_index = (((c_out * channels_in + c_in) * kh + y) * kw) + x;
                    converted[burn_index] = values[keras_index];
                }
            }
        }
    }
    KerasWeight {
        name: String::new(),
        shape: vec![channels_out, channels_in, kh, kw],
        values: converted,
    }
}

pub fn keras_conv3d_kernel_to_burn(shape: [usize; 5], values: &[f32]) -> KerasWeight {
    let [kd, kh, kw, channels_in, channels_out] = shape;
    assert_eq!(values.len(), kd * kh * kw * channels_in * channels_out);
    let mut converted = vec![0.0; values.len()];
    for z in 0..kd {
        for y in 0..kh {
            for x in 0..kw {
                for c_in in 0..channels_in {
                    for c_out in 0..channels_out {
                        let keras_index =
                            ((((z * kh + y) * kw + x) * channels_in + c_in) * channels_out) + c_out;
                        let burn_index =
                            ((((c_out * channels_in + c_in) * kd + z) * kh + y) * kw) + x;
                        converted[burn_index] = values[keras_index];
                    }
                }
            }
        }
    }
    KerasWeight {
        name: String::new(),
        shape: vec![channels_out, channels_in, kd, kh, kw],
        values: converted,
    }
}

#[cfg(feature = "hdf5")]
pub fn load_keras_hdf5_weights(path: impl AsRef<Path>) -> Result<KerasWeights, WeightError> {
    let file = hdf5::File::open(path)?;
    let mut weights = KerasWeights::default();
    let mut groups = vec![String::from("/")];
    while let Some(group_name) = groups.pop() {
        let group = file.group(&group_name)?;
        for member_name in group.member_names()? {
            let full_name = if group_name == "/" {
                format!("/{member_name}")
            } else {
                format!("{group_name}/{member_name}")
            };
            match file.loc_type_by_name(&full_name)? {
                hdf5::LocationType::Group => groups.push(full_name),
                hdf5::LocationType::Dataset => {
                    let dataset = file.dataset(&full_name)?;
                    let shape = dataset.shape();
                    let values = dataset.read_raw::<f32>()?;
                    let key = full_name.trim_start_matches('/').to_string();
                    let mut weight = match shape.as_slice() {
                        [kh, kw, channels_in, channels_out] if key.ends_with("kernel:0") => {
                            keras_conv2d_kernel_to_burn(
                                [*kh, *kw, *channels_in, *channels_out],
                                &values,
                            )
                        }
                        [kd, kh, kw, channels_in, channels_out] if key.ends_with("kernel:0") => {
                            keras_conv3d_kernel_to_burn(
                                [*kd, *kh, *kw, *channels_in, *channels_out],
                                &values,
                            )
                        }
                        _ => KerasWeight {
                            name: String::new(),
                            shape,
                            values,
                        },
                    };
                    weight.name = key;
                    weights.tensors.insert(weight.name.clone(), weight);
                }
                hdf5::LocationType::NamedDatatype | hdf5::LocationType::TypeMap => {}
            }
        }
    }
    Ok(weights)
}

#[cfg(not(feature = "hdf5"))]
pub fn load_keras_hdf5_weights(_path: impl AsRef<Path>) -> Result<KerasWeights, WeightError> {
    Err(WeightError::Hdf5FeatureDisabled)
}

#[derive(Debug, thiserror::Error)]
pub enum WeightError {
    #[cfg(feature = "hdf5")]
    #[error("failed to read HDF5 weights")]
    Hdf5(#[from] hdf5::Error),
    #[error("HDF5 support is disabled")]
    Hdf5FeatureDisabled,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn converts_keras_conv2d_kernel_to_burn_order() {
        let weight = keras_conv2d_kernel_to_burn(
            [2, 3, 2, 2],
            &(0..24).map(|v| v as f32).collect::<Vec<_>>(),
        );
        assert_eq!(weight.shape, vec![2, 2, 2, 3]);
        assert_eq!(weight.values[0], 0.0);
        assert_eq!(weight.values[1], 4.0);
        assert_eq!(weight.values[2], 8.0);
        assert_eq!(weight.values[3], 12.0);
        assert_eq!(weight.values[4], 16.0);
        assert_eq!(weight.values[5], 20.0);
        assert_eq!(weight.values[6], 2.0);
        assert_eq!(weight.values[12], 1.0);
    }

    #[test]
    fn converts_keras_conv3d_kernel_to_burn_order() {
        let weight = keras_conv3d_kernel_to_burn(
            [2, 2, 2, 2, 2],
            &(0..32).map(|v| v as f32).collect::<Vec<_>>(),
        );
        assert_eq!(weight.shape, vec![2, 2, 2, 2, 2]);
        assert_eq!(weight.values[0], 0.0);
        assert_eq!(weight.values[1], 4.0);
        assert_eq!(weight.values[2], 8.0);
        assert_eq!(weight.values[3], 12.0);
        assert_eq!(weight.values[4], 16.0);
        assert_eq!(weight.values[8], 2.0);
        assert_eq!(weight.values[16], 1.0);
    }

    #[cfg(feature = "hdf5")]
    #[test]
    fn loads_2d_demo_keras_hdf5_weights() {
        let path = "stardist/models/examples/2D_demo/weights_best.h5";
        if !std::path::Path::new(path).exists() {
            return;
        }
        let weights = load_keras_hdf5_weights(path).unwrap();
        assert_eq!(
            weights.get("conv2d_1/conv2d_1/kernel:0").unwrap().shape,
            vec![32, 1, 3, 3]
        );
        assert_eq!(
            weights.get("features/features/kernel:0").unwrap().shape,
            vec![128, 32, 3, 3]
        );
        assert_eq!(
            weights.get("prob/prob/kernel:0").unwrap().shape,
            vec![1, 128, 1, 1]
        );
        assert_eq!(
            weights.get("dist/dist/kernel:0").unwrap().shape,
            vec![32, 128, 1, 1]
        );
    }

    #[cfg(feature = "hdf5")]
    #[test]
    fn loads_3d_demo_keras_hdf5_weights() {
        let path = "stardist/models/examples/3D_demo/weights_best.h5";
        if !std::path::Path::new(path).exists() {
            return;
        }
        let weights = load_keras_hdf5_weights(path).unwrap();
        assert_eq!(
            weights.get("conv3d_1/conv3d_1/kernel:0").unwrap().shape,
            vec![32, 1, 7, 7, 7]
        );
        assert_eq!(
            weights.get("conv3d_6/conv3d_6/kernel:0").unwrap().shape,
            vec![64, 32, 1, 1, 1]
        );
        assert_eq!(
            weights.get("features/features/kernel:0").unwrap().shape,
            vec![128, 64, 3, 3, 3]
        );
        assert_eq!(
            weights.get("dist/dist/kernel:0").unwrap().shape,
            vec![96, 128, 1, 1, 1]
        );
    }
}
