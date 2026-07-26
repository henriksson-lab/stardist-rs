#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum SamplePatchesError {
    #[error("patch_size length must match data dimensionality")]
    WrongPatchDimension,
    #[error("all input shapes must be the same")]
    ShapeMismatch,
    #[error("data length does not match shape")]
    DataLengthMismatch,
    #[error("patch_size must be positive and no larger than data shape")]
    InvalidPatchSize,
    #[error("valid_inds must contain one index array per data dimension")]
    ValidIndsDimensionMismatch,
    #[error("valid_inds arrays must have the same length")]
    ValidIndsLengthMismatch,
    #[error("valid_inds center is outside the valid patch region")]
    ValidIndsOutOfBounds,
    #[error("no regions to sample from")]
    NoRegionsToSample,
    #[error("patch_filter mask length does not match image shape")]
    PatchFilterShapeMismatch,
}

pub fn get_valid_inds(
    img_shape: &[usize],
    patch_size: &[usize],
    patch_filter: Option<&[bool]>,
) -> Result<Vec<Vec<usize>>, SamplePatchesError> {
    if patch_size.len() != img_shape.len() {
        return Err(SamplePatchesError::WrongPatchDimension);
    }
    if !patch_size
        .iter()
        .zip(img_shape.iter())
        .all(|(p, s)| *p > 0 && *p <= *s)
    {
        return Err(SamplePatchesError::InvalidPatchSize);
    }
    let n_dim = img_shape.len();
    let mut border_start = Vec::with_capacity(n_dim);
    let mut border_stop = Vec::with_capacity(n_dim);
    for dim in 0..n_dim {
        border_start.push(patch_size[dim] / 2);
        border_stop.push(img_shape[dim] - patch_size[dim] + patch_size[dim] / 2 + 1);
    }

    let mut valid_inds = vec![Vec::<usize>::new(); n_dim];
    if let Some(patch_filter) = patch_filter {
        if patch_filter.len() != img_shape.iter().product::<usize>() {
            return Err(SamplePatchesError::PatchFilterShapeMismatch);
        }
        let mut coord = vec![0usize; n_dim];
        for idx in 0..patch_filter.len() {
            let mut rem = idx;
            for dim in (0..n_dim).rev() {
                coord[dim] = rem % img_shape[dim];
                rem /= img_shape[dim];
            }
            if !patch_filter[idx] {
                continue;
            }
            let mut valid = true;
            for dim in 0..n_dim {
                if coord[dim] < border_start[dim] || coord[dim] >= border_stop[dim] {
                    valid = false;
                    break;
                }
            }
            if valid {
                for dim in 0..n_dim {
                    valid_inds[dim].push(coord[dim]);
                }
            }
        }
    } else {
        let n_valid = border_start
            .iter()
            .zip(border_stop.iter())
            .map(|(start, stop)| stop - start)
            .product::<usize>();
        for dim in 0..n_dim {
            valid_inds[dim].reserve(n_valid);
        }
        for flat in 0..n_valid {
            let mut rem = flat;
            let mut coord = vec![0usize; n_dim];
            for dim in (0..n_dim).rev() {
                let len = border_stop[dim] - border_start[dim];
                coord[dim] = border_start[dim] + rem % len;
                rem /= len;
            }
            for dim in 0..n_dim {
                valid_inds[dim].push(coord[dim]);
            }
        }
    }

    Ok(valid_inds)
}

pub fn sample_patches(
    datas: &[&[f32]],
    data_shape: &[usize],
    patch_size: &[usize],
    n_samples: usize,
    valid_inds: Option<&[Vec<usize>]>,
    seed: u64,
) -> Result<Vec<Vec<f32>>, SamplePatchesError> {
    if datas.is_empty() {
        return Err(SamplePatchesError::ShapeMismatch);
    }
    if patch_size.len() != data_shape.len() {
        return Err(SamplePatchesError::WrongPatchDimension);
    }
    let data_len = data_shape.iter().product::<usize>();
    if datas.iter().any(|data| data.len() != data_len) {
        return Err(SamplePatchesError::DataLengthMismatch);
    }
    if !patch_size
        .iter()
        .zip(data_shape.iter())
        .all(|(p, s)| *p > 0 && *p <= *s)
    {
        return Err(SamplePatchesError::InvalidPatchSize);
    }

    let owned_valid_inds;
    let valid_inds = if let Some(valid_inds) = valid_inds {
        valid_inds
    } else {
        owned_valid_inds = get_valid_inds(data_shape, patch_size, None)?;
        &owned_valid_inds
    };

    if valid_inds.len() != data_shape.len() {
        return Err(SamplePatchesError::ValidIndsDimensionMismatch);
    }
    let n_valid = valid_inds[0].len();
    if n_valid == 0 {
        return Err(SamplePatchesError::NoRegionsToSample);
    }
    if valid_inds.iter().any(|v| v.len() != n_valid) {
        return Err(SamplePatchesError::ValidIndsLengthMismatch);
    }
    for point in 0..n_valid {
        for dim in 0..data_shape.len() {
            let r = valid_inds[dim][point];
            let start = r as isize - (patch_size[dim] / 2) as isize;
            let stop = r + patch_size[dim] - patch_size[dim] / 2;
            if start < 0 || stop > data_shape[dim] {
                return Err(SamplePatchesError::ValidIndsOutOfBounds);
            }
        }
    }

    let replace = n_valid < n_samples;
    let mut state = seed;
    let mut idx = Vec::<usize>::with_capacity(n_samples);
    if replace {
        for _ in 0..n_samples {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let r = ((state >> 11) as f64) * (1.0 / ((1u64 << 53) as f64));
            idx.push(((r * n_valid as f64).floor() as usize).min(n_valid - 1));
        }
    } else {
        let mut pool = (0..n_valid).collect::<Vec<_>>();
        for i in 0..n_samples {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let r = ((state >> 11) as f64) * (1.0 / ((1u64 << 53) as f64));
            let j = i + ((r * (n_valid - i) as f64).floor() as usize).min(n_valid - i - 1);
            pool.swap(i, j);
            idx.push(pool[i]);
        }
    }

    let patch_len = patch_size.iter().product::<usize>();
    let mut res = Vec::with_capacity(datas.len());
    for data in datas {
        let mut sampled = Vec::<f32>::with_capacity(n_samples * patch_len);
        for chosen in &idx {
            let mut patch_coord = vec![0usize; patch_size.len()];
            for flat in 0..patch_len {
                let mut rem = flat;
                for dim in (0..patch_size.len()).rev() {
                    patch_coord[dim] = rem % patch_size[dim];
                    rem /= patch_size[dim];
                }
                let mut src_idx = 0usize;
                let mut stride = 1usize;
                for dim in (0..data_shape.len()).rev() {
                    let center = valid_inds[dim][*chosen];
                    let start = center - patch_size[dim] / 2;
                    src_idx += (start + patch_coord[dim]) * stride;
                    stride *= data_shape[dim];
                }
                sampled.push(data[src_idx]);
            }
        }
        res.push(sampled);
    }

    Ok(res)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn get_valid_inds_without_filter_matches_meshgrid_indexing() {
        let inds = get_valid_inds(&[4, 5], &[2, 3], None).unwrap();
        assert_eq!(inds[0], vec![1, 1, 1, 2, 2, 2, 3, 3, 3]);
        assert_eq!(inds[1], vec![1, 2, 3, 1, 2, 3, 1, 2, 3]);
    }

    #[test]
    fn get_valid_inds_applies_patch_filter_inside_valid_border() {
        let mut filter = vec![false; 4 * 5];
        filter[1 * 5 + 1] = true;
        filter[2 * 5 + 2] = true;
        filter[0] = true;
        let inds = get_valid_inds(&[4, 5], &[3, 3], Some(&filter)).unwrap();
        assert_eq!(inds[0], vec![1, 2]);
        assert_eq!(inds[1], vec![1, 2]);
    }

    #[test]
    fn sample_patches_extracts_same_patch_from_multiple_arrays() {
        let data0 = (0..16).map(|x| x as f32).collect::<Vec<_>>();
        let data1 = (100..116).map(|x| x as f32).collect::<Vec<_>>();
        let valid_inds = vec![vec![1], vec![1]];
        let patches =
            sample_patches(&[&data0, &data1], &[4, 4], &[2, 2], 1, Some(&valid_inds), 7).unwrap();
        assert_eq!(patches[0], vec![0.0, 1.0, 4.0, 5.0]);
        assert_eq!(patches[1], vec![100.0, 101.0, 104.0, 105.0]);
    }

    #[test]
    fn sample_patches_samples_without_replacement_when_enough_regions() {
        let data = (0..9).map(|x| x as f32).collect::<Vec<_>>();
        let valid_inds = vec![vec![0, 0, 0], vec![0, 1, 2]];
        let patches = sample_patches(&[&data], &[3, 3], &[1, 1], 3, Some(&valid_inds), 3).unwrap();
        let mut values = patches[0].clone();
        values.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert_eq!(values, vec![0.0, 1.0, 2.0]);
    }

    #[test]
    fn sample_patches_rejects_invalid_inputs() {
        let data = [0.0f32; 4];
        assert_eq!(
            get_valid_inds(&[2, 2], &[3, 1], None).unwrap_err(),
            SamplePatchesError::InvalidPatchSize
        );
        assert_eq!(
            get_valid_inds(&[2, 2], &[1, 1], Some(&[true])).unwrap_err(),
            SamplePatchesError::PatchFilterShapeMismatch
        );
        assert_eq!(
            sample_patches(&[&data], &[2, 2], &[1], 1, None, 1).unwrap_err(),
            SamplePatchesError::WrongPatchDimension
        );
        assert_eq!(
            sample_patches(&[&data], &[2, 2], &[1, 1], 1, Some(&[vec![], vec![]]), 1).unwrap_err(),
            SamplePatchesError::NoRegionsToSample
        );
        assert_eq!(
            sample_patches(&[&data], &[2, 2], &[2, 2], 1, Some(&[vec![0], vec![0]]), 1)
                .unwrap_err(),
            SamplePatchesError::ValidIndsOutOfBounds
        );
    }
}
