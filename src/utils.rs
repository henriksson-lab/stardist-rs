use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::path::{Path, PathBuf};

use crate::matching::{MatchingCriterion, MatchingError, matching_dataset};

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum GridError {
    #[error("grid must contain exactly {expected} values")]
    WrongLength { expected: usize },
    #[error("grid values must be positive powers of two")]
    NotPowerOfTwo,
}

pub fn _is_power_of_2(i: usize) -> bool {
    i > 0 && (i & (i - 1)) == 0
}

pub fn _normalize_grid<const N: usize>(grid: &[usize]) -> Result<[usize; N], GridError> {
    if grid.len() != N {
        return Err(GridError::WrongLength { expected: N });
    }
    let mut normalized = [1; N];
    for i in 0..N {
        if !_is_power_of_2(grid[i]) {
            return Err(GridError::NotPowerOfTwo);
        }
        normalized[i] = grid[i];
    }
    Ok(normalized)
}

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum UtilsError {
    #[error("label image should be 2- or 3-dimensional")]
    WrongLabelDimension,
    #[error("label image length does not match shape")]
    ShapeMismatch,
    #[error("anisotropy must be None or have one value per image dimension")]
    AnisotropyShapeMismatch,
    #[error("patch_size and grid must have the same length")]
    PatchGridMismatch,
    #[error("sample_points mask must be 2-dimensional")]
    WrongMaskDimension,
    #[error("probability map length does not match mask")]
    ProbShapeMismatch,
    #[error("no valid sample points")]
    NoSamplePoints,
    #[error("probability weights must have a positive finite sum")]
    InvalidProbabilityWeights,
    #[error("label arrays must contain non-negative integers")]
    NegativeLabel,
    #[error("n_classes must be a positive integer")]
    InvalidClassCount,
    #[error("all positive labels must be present in class dict")]
    MissingClassLabel,
    #[error("wrong class id {class_id} for n_classes={n_classes}")]
    WrongClassId { class_id: i32, n_classes: usize },
    #[error("invalid threshold optimization input")]
    InvalidThresholdOptimizationInput,
    #[error("polygon ROI coordinates must have matching non-empty x/y lengths")]
    RoiShapeMismatch,
    #[error("polygon ROI coordinates are outside ImageJ int16 bounds")]
    RoiCoordinateOutOfRange,
    #[error("failed to write ImageJ ROI zip")]
    RoiWriteFailed,
    #[error(transparent)]
    Matching(#[from] MatchingError),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ClassAssignment {
    Single(Option<i32>),
    Dict(Vec<(i32, Option<i32>)>),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ArrayDType {
    Bool,
    Int,
    UInt,
    Float,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OptimizeThresholdMeasure {
    Precision,
    Recall,
    Accuracy,
    F1,
    MeanTrueScore,
    MeanMatchedScore,
    PanopticQuality,
}

pub fn gputools_available() -> bool {
    false
}

pub fn path_absolute(path_relative: impl AsRef<Path>) -> PathBuf {
    let mut base_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    base_path.push("assets");
    base_path.push(path_relative);
    base_path
}

pub fn edt_prob(
    lbl_img: &[i32],
    shape: &[usize],
    anisotropy: Option<&[f32]>,
) -> Result<Vec<f32>, UtilsError> {
    _edt_prob_scipy(lbl_img, shape, anisotropy)
}

pub fn _edt_prob_edt(
    lbl_img: &[i32],
    shape: &[usize],
    anisotropy: Option<&[f32]>,
) -> Result<Vec<f32>, UtilsError> {
    let n = shape.len();
    if n != 2 && n != 3 {
        return Err(UtilsError::WrongLabelDimension);
    }
    if lbl_img.len() != shape.iter().product::<usize>() {
        return Err(UtilsError::ShapeMismatch);
    }
    if let Some(anisotropy) = anisotropy {
        if anisotropy.len() != n {
            return Err(UtilsError::AnisotropyShapeMismatch);
        }
    }
    if shape.iter().any(|dim| *dim == 0) {
        return Ok(vec![0.0; lbl_img.len()]);
    }

    let mut labels = BTreeSet::new();
    for label in lbl_img {
        if *label > 0 {
            labels.insert(*label);
        }
    }
    let constant_img =
        !lbl_img.is_empty() && lbl_img.iter().all(|label| *label == lbl_img[0]) && lbl_img[0] > 0;

    let mut prob = vec![0.0f32; lbl_img.len()];
    if n == 2 {
        let h = shape[0];
        let w = shape[1];
        let ay = anisotropy.map(|a| a[0] as f64).unwrap_or(1.0);
        let ax = anisotropy.map(|a| a[1] as f64).unwrap_or(1.0);
        for label in labels {
            let mut object_points = Vec::<(usize, usize, usize)>::new();
            let mut background_points = Vec::<(isize, isize)>::new();
            for y in 0..h {
                for x in 0..w {
                    let idx = y * w + x;
                    if lbl_img[idx] == label {
                        object_points.push((y, x, idx));
                    } else {
                        background_points.push((y as isize, x as isize));
                    }
                }
            }
            if constant_img {
                for y in 0..h {
                    background_points.push((y as isize, -1));
                    background_points.push((y as isize, w as isize));
                }
                for x in 0..w {
                    background_points.push((-1, x as isize));
                    background_points.push((h as isize, x as isize));
                }
            }
            let mut distances = Vec::<(usize, f32)>::with_capacity(object_points.len());
            let mut max_dist = 0.0f32;
            for (y, x, idx) in object_points {
                let mut best = f64::INFINITY;
                for (by, bx) in &background_points {
                    let dy = (y as isize - *by) as f64 * ay;
                    let dx = (x as isize - *bx) as f64 * ax;
                    let dist2 = dy * dy + dx * dx;
                    if dist2 < best {
                        best = dist2;
                    }
                }
                let dist = best.sqrt() as f32;
                max_dist = max_dist.max(dist);
                distances.push((idx, dist));
            }
            for (idx, dist) in distances {
                prob[idx] = dist / (max_dist + 1e-10);
            }
        }
    } else {
        let d = shape[0];
        let h = shape[1];
        let w = shape[2];
        let az = anisotropy.map(|a| a[0] as f64).unwrap_or(1.0);
        let ay = anisotropy.map(|a| a[1] as f64).unwrap_or(1.0);
        let ax = anisotropy.map(|a| a[2] as f64).unwrap_or(1.0);
        for label in labels {
            let mut object_points = Vec::<(usize, usize, usize, usize)>::new();
            let mut background_points = Vec::<(isize, isize, isize)>::new();
            for z in 0..d {
                for y in 0..h {
                    for x in 0..w {
                        let idx = (z * h + y) * w + x;
                        if lbl_img[idx] == label {
                            object_points.push((z, y, x, idx));
                        } else {
                            background_points.push((z as isize, y as isize, x as isize));
                        }
                    }
                }
            }
            if constant_img {
                for z in 0..d {
                    for y in 0..h {
                        background_points.push((z as isize, y as isize, -1));
                        background_points.push((z as isize, y as isize, w as isize));
                    }
                }
                for z in 0..d {
                    for x in 0..w {
                        background_points.push((z as isize, -1, x as isize));
                        background_points.push((z as isize, h as isize, x as isize));
                    }
                }
                for y in 0..h {
                    for x in 0..w {
                        background_points.push((-1, y as isize, x as isize));
                        background_points.push((d as isize, y as isize, x as isize));
                    }
                }
            }
            let mut distances = Vec::<(usize, f32)>::with_capacity(object_points.len());
            let mut max_dist = 0.0f32;
            for (z, y, x, idx) in object_points {
                let mut best = f64::INFINITY;
                for (bz, by, bx) in &background_points {
                    let dz = (z as isize - *bz) as f64 * az;
                    let dy = (y as isize - *by) as f64 * ay;
                    let dx = (x as isize - *bx) as f64 * ax;
                    let dist2 = dz * dz + dy * dy + dx * dx;
                    if dist2 < best {
                        best = dist2;
                    }
                }
                let dist = best.sqrt() as f32;
                max_dist = max_dist.max(dist);
                distances.push((idx, dist));
            }
            for (idx, dist) in distances {
                prob[idx] = dist / (max_dist + 1e-10);
            }
        }
    }
    Ok(prob)
}

pub fn _edt_prob_scipy(
    lbl_img: &[i32],
    shape: &[usize],
    anisotropy: Option<&[f32]>,
) -> Result<Vec<f32>, UtilsError> {
    let n = shape.len();
    if n != 2 && n != 3 {
        return Err(UtilsError::WrongLabelDimension);
    }
    if lbl_img.len() != shape.iter().product::<usize>() {
        return Err(UtilsError::ShapeMismatch);
    }
    if let Some(anisotropy) = anisotropy {
        if anisotropy.len() != n {
            return Err(UtilsError::AnisotropyShapeMismatch);
        }
    }
    if shape.iter().any(|dim| *dim == 0) {
        return Ok(vec![0.0; lbl_img.len()]);
    }

    let constant_img =
        !lbl_img.is_empty() && lbl_img.iter().all(|label| *label == lbl_img[0]) && lbl_img[0] > 0;
    let mut labels = BTreeSet::new();
    for label in lbl_img {
        if *label > 0 {
            labels.insert(*label);
        }
    }
    let mut prob = vec![0.0f32; lbl_img.len()];

    if n == 2 {
        let h0 = shape[0];
        let w0 = shape[1];
        let (h, w, offset_y, offset_x, padded) = if constant_img {
            (h0 + 2, w0 + 2, 1usize, 1usize, true)
        } else {
            (h0, w0, 0usize, 0usize, false)
        };
        let ay = anisotropy.map(|a| a[0] as f64).unwrap_or(1.0);
        let ax = anisotropy.map(|a| a[1] as f64).unwrap_or(1.0);
        for label in labels {
            let mut min_y = usize::MAX;
            let mut min_x = usize::MAX;
            let mut max_y = 0usize;
            let mut max_x = 0usize;
            for py in 0..h {
                for px in 0..w {
                    let in_original = py >= offset_y
                        && py < offset_y + h0
                        && px >= offset_x
                        && px < offset_x + w0;
                    let value = if in_original {
                        lbl_img[(py - offset_y) * w0 + px - offset_x]
                    } else {
                        0
                    };
                    if value == label {
                        min_y = min_y.min(py);
                        min_x = min_x.min(px);
                        max_y = max_y.max(py + 1);
                        max_x = max_x.max(px + 1);
                    }
                }
            }
            if min_y == usize::MAX {
                continue;
            }
            let grow_y0 = if min_y > 0 { min_y - 1 } else { min_y };
            let grow_x0 = if min_x > 0 { min_x - 1 } else { min_x };
            let grow_y1 = if max_y < h { max_y + 1 } else { max_y };
            let grow_x1 = if max_x < w { max_x + 1 } else { max_x };
            let mut distances = Vec::<(usize, f32)>::new();
            let mut max_dist = 0.0f32;
            for py in min_y..max_y {
                for px in min_x..max_x {
                    let original_y = py as isize - offset_y as isize;
                    let original_x = px as isize - offset_x as isize;
                    let in_original = original_y >= 0
                        && original_y < h0 as isize
                        && original_x >= 0
                        && original_x < w0 as isize;
                    if !in_original
                        || lbl_img[original_y as usize * w0 + original_x as usize] != label
                    {
                        continue;
                    }
                    let mut best = f64::INFINITY;
                    for by in grow_y0..grow_y1 {
                        for bx in grow_x0..grow_x1 {
                            let by_original = by as isize - offset_y as isize;
                            let bx_original = bx as isize - offset_x as isize;
                            let by_in_original = by_original >= 0
                                && by_original < h0 as isize
                                && bx_original >= 0
                                && bx_original < w0 as isize;
                            let value = if by_in_original {
                                lbl_img[by_original as usize * w0 + bx_original as usize]
                            } else {
                                0
                            };
                            if value != label {
                                let dy = (py as isize - by as isize) as f64 * ay;
                                let dx = (px as isize - bx as isize) as f64 * ax;
                                let dist2 = dy * dy + dx * dx;
                                if dist2 < best {
                                    best = dist2;
                                }
                            }
                        }
                    }
                    let dist = best.sqrt() as f32;
                    max_dist = max_dist.max(dist);
                    distances.push((original_y as usize * w0 + original_x as usize, dist));
                }
            }
            for (idx, dist) in distances {
                prob[idx] = dist / (max_dist + 1e-10);
            }
        }
        if padded {
            return Ok(prob);
        }
    } else {
        let d0 = shape[0];
        let h0 = shape[1];
        let w0 = shape[2];
        let (d, h, w, offset_z, offset_y, offset_x, padded) = if constant_img {
            (d0 + 2, h0 + 2, w0 + 2, 1usize, 1usize, 1usize, true)
        } else {
            (d0, h0, w0, 0usize, 0usize, 0usize, false)
        };
        let az = anisotropy.map(|a| a[0] as f64).unwrap_or(1.0);
        let ay = anisotropy.map(|a| a[1] as f64).unwrap_or(1.0);
        let ax = anisotropy.map(|a| a[2] as f64).unwrap_or(1.0);
        for label in labels {
            let mut min_z = usize::MAX;
            let mut min_y = usize::MAX;
            let mut min_x = usize::MAX;
            let mut max_z = 0usize;
            let mut max_y = 0usize;
            let mut max_x = 0usize;
            for pz in 0..d {
                for py in 0..h {
                    for px in 0..w {
                        let in_original = pz >= offset_z
                            && pz < offset_z + d0
                            && py >= offset_y
                            && py < offset_y + h0
                            && px >= offset_x
                            && px < offset_x + w0;
                        let value = if in_original {
                            lbl_img[((pz - offset_z) * h0 + py - offset_y) * w0 + px - offset_x]
                        } else {
                            0
                        };
                        if value == label {
                            min_z = min_z.min(pz);
                            min_y = min_y.min(py);
                            min_x = min_x.min(px);
                            max_z = max_z.max(pz + 1);
                            max_y = max_y.max(py + 1);
                            max_x = max_x.max(px + 1);
                        }
                    }
                }
            }
            if min_z == usize::MAX {
                continue;
            }
            let grow_z0 = if min_z > 0 { min_z - 1 } else { min_z };
            let grow_y0 = if min_y > 0 { min_y - 1 } else { min_y };
            let grow_x0 = if min_x > 0 { min_x - 1 } else { min_x };
            let grow_z1 = if max_z < d { max_z + 1 } else { max_z };
            let grow_y1 = if max_y < h { max_y + 1 } else { max_y };
            let grow_x1 = if max_x < w { max_x + 1 } else { max_x };
            let mut distances = Vec::<(usize, f32)>::new();
            let mut max_dist = 0.0f32;
            for pz in min_z..max_z {
                for py in min_y..max_y {
                    for px in min_x..max_x {
                        let original_z = pz as isize - offset_z as isize;
                        let original_y = py as isize - offset_y as isize;
                        let original_x = px as isize - offset_x as isize;
                        let in_original = original_z >= 0
                            && original_z < d0 as isize
                            && original_y >= 0
                            && original_y < h0 as isize
                            && original_x >= 0
                            && original_x < w0 as isize;
                        if !in_original
                            || lbl_img[((original_z as usize * h0 + original_y as usize) * w0)
                                + original_x as usize]
                                != label
                        {
                            continue;
                        }
                        let mut best = f64::INFINITY;
                        for bz in grow_z0..grow_z1 {
                            for by in grow_y0..grow_y1 {
                                for bx in grow_x0..grow_x1 {
                                    let bz_original = bz as isize - offset_z as isize;
                                    let by_original = by as isize - offset_y as isize;
                                    let bx_original = bx as isize - offset_x as isize;
                                    let b_in_original = bz_original >= 0
                                        && bz_original < d0 as isize
                                        && by_original >= 0
                                        && by_original < h0 as isize
                                        && bx_original >= 0
                                        && bx_original < w0 as isize;
                                    let value = if b_in_original {
                                        lbl_img[((bz_original as usize * h0
                                            + by_original as usize)
                                            * w0)
                                            + bx_original as usize]
                                    } else {
                                        0
                                    };
                                    if value != label {
                                        let dz = (pz as isize - bz as isize) as f64 * az;
                                        let dy = (py as isize - by as isize) as f64 * ay;
                                        let dx = (px as isize - bx as isize) as f64 * ax;
                                        let dist2 = dz * dz + dy * dy + dx * dx;
                                        if dist2 < best {
                                            best = dist2;
                                        }
                                    }
                                }
                            }
                        }
                        let dist = best.sqrt() as f32;
                        max_dist = max_dist.max(dist);
                        distances.push((
                            (original_z as usize * h0 + original_y as usize) * w0
                                + original_x as usize,
                            dist,
                        ));
                    }
                }
            }
            for (idx, dist) in distances {
                prob[idx] = dist / (max_dist + 1e-10);
            }
        }
        if padded {
            return Ok(prob);
        }
    }
    Ok(prob)
}

pub fn _invert_dict(d: &[(i32, Option<i32>)]) -> BTreeMap<Option<i32>, Vec<i32>> {
    let mut res = BTreeMap::<Option<i32>, Vec<i32>>::new();
    for (k, v) in d {
        res.entry(*v).or_default().push(*k);
    }
    res
}

pub fn mask_to_categorical(
    y: &[i32],
    shape: &[usize],
    n_classes: usize,
    classes: ClassAssignment,
    return_cls_dict: bool,
) -> Result<(Vec<f32>, Option<BTreeMap<Option<i32>, Vec<i32>>>), UtilsError> {
    if y.len() != shape.iter().product::<usize>() {
        return Err(UtilsError::ShapeMismatch);
    }
    for label in y {
        if *label < 0 {
            return Err(UtilsError::NegativeLabel);
        }
    }
    if n_classes < 1 {
        return Err(UtilsError::InvalidClassCount);
    }

    let mut y_labels = BTreeSet::new();
    for label in y {
        if *label > 0 {
            y_labels.insert(*label);
        }
    }

    let classes = match classes {
        ClassAssignment::Single(class_id) => {
            let mut mapped = Vec::with_capacity(y_labels.len());
            for label in &y_labels {
                mapped.push((*label, class_id));
            }
            mapped
        }
        ClassAssignment::Dict(classes) => classes,
    };

    let mut class_labels = BTreeSet::new();
    for (label, _) in &classes {
        class_labels.insert(*label);
    }
    if !y_labels.is_subset(&class_labels) {
        return Err(UtilsError::MissingClassLabel);
    }

    let cls_dict = _invert_dict(&classes);
    let channels = n_classes + 1;
    let mut y_mask = vec![0.0f32; y.len() * channels];

    for (cls, labels) in &cls_dict {
        if cls.is_none() {
            for label in labels {
                for idx in 0..y.len() {
                    if y[idx] == *label {
                        let offset = idx * channels;
                        for channel in 0..channels {
                            y_mask[offset + channel] = -1.0;
                        }
                    }
                }
            }
        } else if let Some(cls) = cls {
            if *cls >= 0 && (*cls as usize) <= n_classes {
                for label in labels {
                    for idx in 0..y.len() {
                        if y[idx] == *label {
                            y_mask[idx * channels + *cls as usize] = 1.0;
                        }
                    }
                }
            } else {
                return Err(UtilsError::WrongClassId {
                    class_id: *cls,
                    n_classes,
                });
            }
        }
    }

    for idx in 0..y.len() {
        y_mask[idx * channels] = if y[idx] == 0 { 1.0 } else { 0.0 };
    }

    if return_cls_dict {
        Ok((y_mask, Some(cls_dict)))
    } else {
        Ok((y_mask, None))
    }
}

pub fn sample_points(
    n_samples: usize,
    mask: &[bool],
    shape: &[usize],
    prob: Option<&[f32]>,
    b: Option<usize>,
    seed: u64,
) -> Result<Vec<[usize; 2]>, UtilsError> {
    if shape.len() != 2 {
        return Err(UtilsError::WrongMaskDimension);
    }
    if mask.len() != shape.iter().product::<usize>() {
        return Err(UtilsError::ShapeMismatch);
    }
    if let Some(prob) = prob {
        if prob.len() != mask.len() {
            return Err(UtilsError::ProbShapeMismatch);
        }
    }

    let h = shape[0];
    let w = shape[1];
    let mut points = Vec::<[usize; 2]>::new();
    for y in 0..h {
        for x in 0..w {
            let keep_boundary = if let Some(b) = b {
                if b > 0 {
                    y >= b && y < h.saturating_sub(b) && x >= b && x < w.saturating_sub(b)
                } else {
                    true
                }
            } else {
                true
            };
            if mask[y * w + x] && keep_boundary {
                points.push([y, x]);
            }
        }
    }

    if points.is_empty() && n_samples > 0 {
        return Err(UtilsError::NoSamplePoints);
    }

    let mut state = seed;
    let mut sampled = Vec::with_capacity(n_samples);
    if let Some(prob) = prob {
        let mut weights = Vec::with_capacity(points.len());
        let mut sum = 0.0f64;
        for point in &points {
            let value = prob[point[0] * w + point[1]] as f64;
            if value.is_sign_negative() || !value.is_finite() {
                return Err(UtilsError::InvalidProbabilityWeights);
            }
            weights.push(value);
            sum += value;
        }
        if !(sum.is_finite() && sum > 0.0) {
            return Err(UtilsError::InvalidProbabilityWeights);
        }
        for _ in 0..n_samples {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let mut r = ((state >> 11) as f64) * (1.0 / ((1u64 << 53) as f64));
            r *= sum;
            let mut acc = 0.0f64;
            let mut chosen = points.len() - 1;
            for i in 0..weights.len() {
                acc += weights[i];
                if r < acc {
                    chosen = i;
                    break;
                }
            }
            sampled.push(points[chosen]);
        }
    } else {
        for _ in 0..n_samples {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let r = ((state >> 11) as f64) * (1.0 / ((1u64 << 53) as f64));
            let chosen = (r * points.len() as f64).floor() as usize;
            sampled.push(points[chosen.min(points.len() - 1)]);
        }
    }

    Ok(sampled)
}

pub fn _fill_label_holes(lbl_img: &[i32], shape: &[usize]) -> Result<Vec<i32>, UtilsError> {
    let n = shape.len();
    if n != 2 && n != 3 {
        return Err(UtilsError::WrongLabelDimension);
    }
    if lbl_img.len() != shape.iter().product::<usize>() {
        return Err(UtilsError::ShapeMismatch);
    }
    if shape.iter().any(|dim| *dim == 0) {
        return Ok(vec![0i32; lbl_img.len()]);
    }

    let mut labels = BTreeSet::new();
    for label in lbl_img {
        if *label != 0 {
            labels.insert(*label);
        }
    }

    let mut lbl_img_filled = vec![0i32; lbl_img.len()];
    for label in labels {
        let mut outside = vec![false; lbl_img.len()];
        let mut queue = VecDeque::new();
        if n == 2 {
            let h = shape[0];
            let w = shape[1];
            for y in 0..h {
                for x in [0, w - 1] {
                    let idx = y * w + x;
                    if lbl_img[idx] != label && !outside[idx] {
                        outside[idx] = true;
                        queue.push_back(idx);
                    }
                }
            }
            for x in 0..w {
                for y in [0, h - 1] {
                    let idx = y * w + x;
                    if lbl_img[idx] != label && !outside[idx] {
                        outside[idx] = true;
                        queue.push_back(idx);
                    }
                }
            }
            while let Some(idx) = queue.pop_front() {
                let y = idx / w;
                let x = idx % w;
                if y > 0 {
                    let next = (y - 1) * w + x;
                    if lbl_img[next] != label && !outside[next] {
                        outside[next] = true;
                        queue.push_back(next);
                    }
                }
                if y + 1 < h {
                    let next = (y + 1) * w + x;
                    if lbl_img[next] != label && !outside[next] {
                        outside[next] = true;
                        queue.push_back(next);
                    }
                }
                if x > 0 {
                    let next = y * w + x - 1;
                    if lbl_img[next] != label && !outside[next] {
                        outside[next] = true;
                        queue.push_back(next);
                    }
                }
                if x + 1 < w {
                    let next = y * w + x + 1;
                    if lbl_img[next] != label && !outside[next] {
                        outside[next] = true;
                        queue.push_back(next);
                    }
                }
            }
        } else {
            let d = shape[0];
            let h = shape[1];
            let w = shape[2];
            for z in 0..d {
                for y in 0..h {
                    for x in [0, w - 1] {
                        let idx = (z * h + y) * w + x;
                        if lbl_img[idx] != label && !outside[idx] {
                            outside[idx] = true;
                            queue.push_back(idx);
                        }
                    }
                }
            }
            for z in 0..d {
                for x in 0..w {
                    for y in [0, h - 1] {
                        let idx = (z * h + y) * w + x;
                        if lbl_img[idx] != label && !outside[idx] {
                            outside[idx] = true;
                            queue.push_back(idx);
                        }
                    }
                }
            }
            for y in 0..h {
                for x in 0..w {
                    for z in [0, d - 1] {
                        let idx = (z * h + y) * w + x;
                        if lbl_img[idx] != label && !outside[idx] {
                            outside[idx] = true;
                            queue.push_back(idx);
                        }
                    }
                }
            }
            while let Some(idx) = queue.pop_front() {
                let z = idx / (h * w);
                let rem = idx % (h * w);
                let y = rem / w;
                let x = rem % w;
                if z > 0 {
                    let next = ((z - 1) * h + y) * w + x;
                    if lbl_img[next] != label && !outside[next] {
                        outside[next] = true;
                        queue.push_back(next);
                    }
                }
                if z + 1 < d {
                    let next = ((z + 1) * h + y) * w + x;
                    if lbl_img[next] != label && !outside[next] {
                        outside[next] = true;
                        queue.push_back(next);
                    }
                }
                if y > 0 {
                    let next = (z * h + y - 1) * w + x;
                    if lbl_img[next] != label && !outside[next] {
                        outside[next] = true;
                        queue.push_back(next);
                    }
                }
                if y + 1 < h {
                    let next = (z * h + y + 1) * w + x;
                    if lbl_img[next] != label && !outside[next] {
                        outside[next] = true;
                        queue.push_back(next);
                    }
                }
                if x > 0 {
                    let next = (z * h + y) * w + x - 1;
                    if lbl_img[next] != label && !outside[next] {
                        outside[next] = true;
                        queue.push_back(next);
                    }
                }
                if x + 1 < w {
                    let next = (z * h + y) * w + x + 1;
                    if lbl_img[next] != label && !outside[next] {
                        outside[next] = true;
                        queue.push_back(next);
                    }
                }
            }
        }
        for idx in 0..lbl_img.len() {
            if lbl_img[idx] == label || !outside[idx] {
                lbl_img_filled[idx] = label;
            }
        }
    }
    Ok(lbl_img_filled)
}

pub fn fill_label_holes(lbl_img: &[i32], shape: &[usize]) -> Result<Vec<i32>, UtilsError> {
    let n = shape.len();
    if n != 2 && n != 3 {
        return Err(UtilsError::WrongLabelDimension);
    }
    if lbl_img.len() != shape.iter().product::<usize>() {
        return Err(UtilsError::ShapeMismatch);
    }
    if shape.iter().any(|dim| *dim == 0) {
        return Ok(vec![0i32; lbl_img.len()]);
    }

    let max_label = lbl_img
        .iter()
        .copied()
        .filter(|x| *x > 0)
        .max()
        .unwrap_or(0);
    let mut lbl_img_filled = vec![0i32; lbl_img.len()];

    if n == 2 {
        let h = shape[0];
        let w = shape[1];
        let mut min_y = vec![usize::MAX; max_label as usize + 1];
        let mut min_x = vec![usize::MAX; max_label as usize + 1];
        let mut max_y = vec![0usize; max_label as usize + 1];
        let mut max_x = vec![0usize; max_label as usize + 1];
        let mut seen = vec![false; max_label as usize + 1];
        for y in 0..h {
            for x in 0..w {
                let label = lbl_img[y * w + x];
                if label <= 0 {
                    continue;
                }
                let label = label as usize;
                seen[label] = true;
                min_y[label] = min_y[label].min(y);
                min_x[label] = min_x[label].min(x);
                max_y[label] = max_y[label].max(y + 1);
                max_x[label] = max_x[label].max(x + 1);
            }
        }
        for label in 1..=max_label as usize {
            if !seen[label] {
                continue;
            }
            let grow_y0 = if min_y[label] > 0 {
                min_y[label] - 1
            } else {
                min_y[label]
            };
            let grow_x0 = if min_x[label] > 0 {
                min_x[label] - 1
            } else {
                min_x[label]
            };
            let grow_y1 = if max_y[label] < h {
                max_y[label] + 1
            } else {
                max_y[label]
            };
            let grow_x1 = if max_x[label] < w {
                max_x[label] + 1
            } else {
                max_x[label]
            };
            let gh = grow_y1 - grow_y0;
            let gw = grow_x1 - grow_x0;
            let mut outside = vec![false; gh * gw];
            let mut queue = VecDeque::new();
            for ly in 0..gh {
                for lx in [0, gw - 1] {
                    let idx = ly * gw + lx;
                    let src = (grow_y0 + ly) * w + grow_x0 + lx;
                    if lbl_img[src] != label as i32 && !outside[idx] {
                        outside[idx] = true;
                        queue.push_back(idx);
                    }
                }
            }
            for lx in 0..gw {
                for ly in [0, gh - 1] {
                    let idx = ly * gw + lx;
                    let src = (grow_y0 + ly) * w + grow_x0 + lx;
                    if lbl_img[src] != label as i32 && !outside[idx] {
                        outside[idx] = true;
                        queue.push_back(idx);
                    }
                }
            }
            while let Some(idx) = queue.pop_front() {
                let ly = idx / gw;
                let lx = idx % gw;
                if ly > 0 {
                    let next = (ly - 1) * gw + lx;
                    let src = (grow_y0 + ly - 1) * w + grow_x0 + lx;
                    if lbl_img[src] != label as i32 && !outside[next] {
                        outside[next] = true;
                        queue.push_back(next);
                    }
                }
                if ly + 1 < gh {
                    let next = (ly + 1) * gw + lx;
                    let src = (grow_y0 + ly + 1) * w + grow_x0 + lx;
                    if lbl_img[src] != label as i32 && !outside[next] {
                        outside[next] = true;
                        queue.push_back(next);
                    }
                }
                if lx > 0 {
                    let next = ly * gw + lx - 1;
                    let src = (grow_y0 + ly) * w + grow_x0 + lx - 1;
                    if lbl_img[src] != label as i32 && !outside[next] {
                        outside[next] = true;
                        queue.push_back(next);
                    }
                }
                if lx + 1 < gw {
                    let next = ly * gw + lx + 1;
                    let src = (grow_y0 + ly) * w + grow_x0 + lx + 1;
                    if lbl_img[src] != label as i32 && !outside[next] {
                        outside[next] = true;
                        queue.push_back(next);
                    }
                }
            }
            for y in min_y[label]..max_y[label] {
                for x in min_x[label]..max_x[label] {
                    let local = (y - grow_y0) * gw + x - grow_x0;
                    if lbl_img[y * w + x] == label as i32 || !outside[local] {
                        lbl_img_filled[y * w + x] = label as i32;
                    }
                }
            }
        }
    } else {
        let d = shape[0];
        let h = shape[1];
        let w = shape[2];
        let mut min_z = vec![usize::MAX; max_label as usize + 1];
        let mut min_y = vec![usize::MAX; max_label as usize + 1];
        let mut min_x = vec![usize::MAX; max_label as usize + 1];
        let mut max_z = vec![0usize; max_label as usize + 1];
        let mut max_y = vec![0usize; max_label as usize + 1];
        let mut max_x = vec![0usize; max_label as usize + 1];
        let mut seen = vec![false; max_label as usize + 1];
        for z in 0..d {
            for y in 0..h {
                for x in 0..w {
                    let label = lbl_img[(z * h + y) * w + x];
                    if label <= 0 {
                        continue;
                    }
                    let label = label as usize;
                    seen[label] = true;
                    min_z[label] = min_z[label].min(z);
                    min_y[label] = min_y[label].min(y);
                    min_x[label] = min_x[label].min(x);
                    max_z[label] = max_z[label].max(z + 1);
                    max_y[label] = max_y[label].max(y + 1);
                    max_x[label] = max_x[label].max(x + 1);
                }
            }
        }
        for label in 1..=max_label as usize {
            if !seen[label] {
                continue;
            }
            let grow_z0 = if min_z[label] > 0 {
                min_z[label] - 1
            } else {
                min_z[label]
            };
            let grow_y0 = if min_y[label] > 0 {
                min_y[label] - 1
            } else {
                min_y[label]
            };
            let grow_x0 = if min_x[label] > 0 {
                min_x[label] - 1
            } else {
                min_x[label]
            };
            let grow_z1 = if max_z[label] < d {
                max_z[label] + 1
            } else {
                max_z[label]
            };
            let grow_y1 = if max_y[label] < h {
                max_y[label] + 1
            } else {
                max_y[label]
            };
            let grow_x1 = if max_x[label] < w {
                max_x[label] + 1
            } else {
                max_x[label]
            };
            let gd = grow_z1 - grow_z0;
            let gh = grow_y1 - grow_y0;
            let gw = grow_x1 - grow_x0;
            let mut outside = vec![false; gd * gh * gw];
            let mut queue = VecDeque::new();
            for lz in 0..gd {
                for ly in 0..gh {
                    for lx in [0, gw - 1] {
                        let idx = (lz * gh + ly) * gw + lx;
                        let src = ((grow_z0 + lz) * h + grow_y0 + ly) * w + grow_x0 + lx;
                        if lbl_img[src] != label as i32 && !outside[idx] {
                            outside[idx] = true;
                            queue.push_back(idx);
                        }
                    }
                }
            }
            for lz in 0..gd {
                for lx in 0..gw {
                    for ly in [0, gh - 1] {
                        let idx = (lz * gh + ly) * gw + lx;
                        let src = ((grow_z0 + lz) * h + grow_y0 + ly) * w + grow_x0 + lx;
                        if lbl_img[src] != label as i32 && !outside[idx] {
                            outside[idx] = true;
                            queue.push_back(idx);
                        }
                    }
                }
            }
            for ly in 0..gh {
                for lx in 0..gw {
                    for lz in [0, gd - 1] {
                        let idx = (lz * gh + ly) * gw + lx;
                        let src = ((grow_z0 + lz) * h + grow_y0 + ly) * w + grow_x0 + lx;
                        if lbl_img[src] != label as i32 && !outside[idx] {
                            outside[idx] = true;
                            queue.push_back(idx);
                        }
                    }
                }
            }
            while let Some(idx) = queue.pop_front() {
                let lz = idx / (gh * gw);
                let rem = idx % (gh * gw);
                let ly = rem / gw;
                let lx = rem % gw;
                if lz > 0 {
                    let next = ((lz - 1) * gh + ly) * gw + lx;
                    let src = ((grow_z0 + lz - 1) * h + grow_y0 + ly) * w + grow_x0 + lx;
                    if lbl_img[src] != label as i32 && !outside[next] {
                        outside[next] = true;
                        queue.push_back(next);
                    }
                }
                if lz + 1 < gd {
                    let next = ((lz + 1) * gh + ly) * gw + lx;
                    let src = ((grow_z0 + lz + 1) * h + grow_y0 + ly) * w + grow_x0 + lx;
                    if lbl_img[src] != label as i32 && !outside[next] {
                        outside[next] = true;
                        queue.push_back(next);
                    }
                }
                if ly > 0 {
                    let next = (lz * gh + ly - 1) * gw + lx;
                    let src = ((grow_z0 + lz) * h + grow_y0 + ly - 1) * w + grow_x0 + lx;
                    if lbl_img[src] != label as i32 && !outside[next] {
                        outside[next] = true;
                        queue.push_back(next);
                    }
                }
                if ly + 1 < gh {
                    let next = (lz * gh + ly + 1) * gw + lx;
                    let src = ((grow_z0 + lz) * h + grow_y0 + ly + 1) * w + grow_x0 + lx;
                    if lbl_img[src] != label as i32 && !outside[next] {
                        outside[next] = true;
                        queue.push_back(next);
                    }
                }
                if lx > 0 {
                    let next = (lz * gh + ly) * gw + lx - 1;
                    let src = ((grow_z0 + lz) * h + grow_y0 + ly) * w + grow_x0 + lx - 1;
                    if lbl_img[src] != label as i32 && !outside[next] {
                        outside[next] = true;
                        queue.push_back(next);
                    }
                }
                if lx + 1 < gw {
                    let next = (lz * gh + ly) * gw + lx + 1;
                    let src = ((grow_z0 + lz) * h + grow_y0 + ly) * w + grow_x0 + lx + 1;
                    if lbl_img[src] != label as i32 && !outside[next] {
                        outside[next] = true;
                        queue.push_back(next);
                    }
                }
            }
            for z in min_z[label]..max_z[label] {
                for y in min_y[label]..max_y[label] {
                    for x in min_x[label]..max_x[label] {
                        let local = ((z - grow_z0) * gh + y - grow_y0) * gw + x - grow_x0;
                        if lbl_img[(z * h + y) * w + x] == label as i32 || !outside[local] {
                            lbl_img_filled[(z * h + y) * w + x] = label as i32;
                        }
                    }
                }
            }
        }
    }

    if lbl_img.iter().any(|label| *label < 0) {
        let mut negative_input = vec![0i32; lbl_img.len()];
        for idx in 0..lbl_img.len() {
            negative_input[idx] = -lbl_img[idx].min(0);
        }
        let lbl_neg_filled = fill_label_holes(&negative_input, shape)?;
        for idx in 0..lbl_img.len() {
            if lbl_neg_filled[idx] > 0 {
                lbl_img_filled[idx] = -lbl_neg_filled[idx];
            }
        }
    }

    Ok(lbl_img_filled)
}

pub fn calculate_extents(lbl: &[u16], shape: &[usize]) -> Result<Vec<f32>, UtilsError> {
    let n = shape.len();
    if n != 2 && n != 3 {
        return Err(UtilsError::WrongLabelDimension);
    }
    if lbl.len() != shape.iter().product::<usize>() {
        return Err(UtilsError::ShapeMismatch);
    }

    let mut max_label = 0u16;
    for value in lbl {
        max_label = max_label.max(*value);
    }
    if max_label == 0 {
        return Ok(vec![0.0; n]);
    }

    let mut min_coord = vec![vec![usize::MAX; n]; max_label as usize + 1];
    let mut max_coord = vec![vec![0usize; n]; max_label as usize + 1];
    let mut seen = vec![false; max_label as usize + 1];

    if n == 2 {
        for y in 0..shape[0] {
            for x in 0..shape[1] {
                let label = lbl[y * shape[1] + x] as usize;
                if label == 0 {
                    continue;
                }
                seen[label] = true;
                min_coord[label][0] = min_coord[label][0].min(y);
                min_coord[label][1] = min_coord[label][1].min(x);
                max_coord[label][0] = max_coord[label][0].max(y + 1);
                max_coord[label][1] = max_coord[label][1].max(x + 1);
            }
        }
    } else {
        for z in 0..shape[0] {
            for y in 0..shape[1] {
                for x in 0..shape[2] {
                    let label = lbl[(z * shape[1] + y) * shape[2] + x] as usize;
                    if label == 0 {
                        continue;
                    }
                    seen[label] = true;
                    min_coord[label][0] = min_coord[label][0].min(z);
                    min_coord[label][1] = min_coord[label][1].min(y);
                    min_coord[label][2] = min_coord[label][2].min(x);
                    max_coord[label][0] = max_coord[label][0].max(z + 1);
                    max_coord[label][1] = max_coord[label][1].max(y + 1);
                    max_coord[label][2] = max_coord[label][2].max(x + 1);
                }
            }
        }
    }

    let mut extents = vec![Vec::<f32>::new(); n];
    for label in 1..=max_label as usize {
        if !seen[label] {
            continue;
        }
        for axis in 0..n {
            extents[axis].push((max_coord[label][axis] - min_coord[label][axis]) as f32);
        }
    }

    let mut result = Vec::with_capacity(n);
    for values in &mut extents {
        if values.is_empty() {
            result.push(0.0);
        } else {
            values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let mid = values.len() / 2;
            if values.len() % 2 == 0 {
                result.push(0.5 * (values[mid - 1] + values[mid]));
            } else {
                result.push(values[mid]);
            }
        }
    }
    Ok(result)
}

pub fn optimize_threshold<F>(
    y_true: &[&[u32]],
    yhat_prob: &[&[f32]],
    nms_thresh: f32,
    measure: OptimizeThresholdMeasure,
    iou_threshs: &[f32],
    bracket: Option<[f32; 2]>,
    tol: f32,
    maxiter: usize,
    mut predict_instances: F,
) -> Result<(f32, f32), UtilsError>
where
    F: FnMut(usize, f32, f32) -> Result<Vec<u32>, UtilsError>,
{
    if y_true.len() != yhat_prob.len()
        || y_true.is_empty()
        || !nms_thresh.is_finite()
        || !tol.is_finite()
        || tol <= 0.0
        || maxiter == 0
    {
        return Err(UtilsError::InvalidThresholdOptimizationInput);
    }

    let bracket = if let Some(bracket) = bracket {
        bracket
    } else {
        let mut max_prob = f32::NEG_INFINITY;
        for prob in yhat_prob {
            for value in *prob {
                if value.is_finite() {
                    max_prob = max_prob.max(*value);
                }
            }
        }
        if !max_prob.is_finite() {
            return Err(UtilsError::InvalidThresholdOptimizationInput);
        }
        [max_prob / 2.0, max_prob]
    };
    if !bracket[0].is_finite() || !bracket[1].is_finite() || bracket[0] >= bracket[1] {
        return Err(UtilsError::InvalidThresholdOptimizationInput);
    }

    let mut values = Vec::<(u32, f32, f32)>::new();
    let mut eval = |thr: f32| -> Result<f32, UtilsError> {
        let prob_thresh = thr.clamp(bracket[0], bracket[1]);
        let key = prob_thresh.to_bits();
        for (cached_key, _, value) in &values {
            if *cached_key == key {
                return Ok(*value);
            }
        }

        let mut y_instances = Vec::<Vec<u32>>::with_capacity(y_true.len());
        for i in 0..y_true.len() {
            y_instances.push(predict_instances(i, prob_thresh, nms_thresh)?);
        }
        let y_pred = y_instances
            .iter()
            .map(|labels| labels.as_slice())
            .collect::<Vec<_>>();
        let stats = matching_dataset(
            y_true,
            &y_pred,
            iou_threshs,
            MatchingCriterion::Iou,
            false,
            false,
            true,
        )?;
        let mut value = 0.0f32;
        for s in &stats {
            value += match measure {
                OptimizeThresholdMeasure::Precision => s.precision,
                OptimizeThresholdMeasure::Recall => s.recall,
                OptimizeThresholdMeasure::Accuracy => s.accuracy,
                OptimizeThresholdMeasure::F1 => s.f1,
                OptimizeThresholdMeasure::MeanTrueScore => s.mean_true_score,
                OptimizeThresholdMeasure::MeanMatchedScore => s.mean_matched_score,
                OptimizeThresholdMeasure::PanopticQuality => s.panoptic_quality,
            };
        }
        value /= stats.len() as f32;
        values.push((key, prob_thresh, value));
        Ok(value)
    };

    let mut a = bracket[0];
    let mut b = bracket[1];
    let inv_phi = (5.0_f32.sqrt() - 1.0) / 2.0;
    let mut c = b - inv_phi * (b - a);
    let mut d = a + inv_phi * (b - a);
    let mut fc = eval(c)?;
    let mut fd = eval(d)?;

    for _ in 0..maxiter {
        if (b - a).abs() <= tol {
            break;
        }
        if fc >= fd {
            b = d;
            d = c;
            fd = fc;
            c = b - inv_phi * (b - a);
            fc = eval(c)?;
        } else {
            a = c;
            c = d;
            fc = fd;
            d = a + inv_phi * (b - a);
            fd = eval(d)?;
        }
    }

    let prob_thresh = 0.5 * (a + b);
    let value = eval(prob_thresh)?;
    Ok((prob_thresh, value))
}

pub fn grid_divisible_patch_size(
    patch_size: &[usize],
    grid: &[usize],
) -> Result<Vec<usize>, UtilsError> {
    if patch_size.len() != grid.len() {
        return Err(UtilsError::PatchGridMismatch);
    }
    let mut patch_size_divisible = Vec::with_capacity(patch_size.len());
    for i in 0..patch_size.len() {
        let sh = patch_size[i];
        let g = grid[i];
        patch_size_divisible.push(sh.div_ceil(g) * g);
    }
    Ok(patch_size_divisible)
}

pub fn _is_floatarray(dtype: ArrayDType) -> bool {
    matches!(dtype, ArrayDType::Float)
}

pub fn abspath(root: impl AsRef<Path>, relpath: impl AsRef<Path>) -> std::io::Result<PathBuf> {
    let root = root.as_ref();
    let relpath = relpath.as_ref();
    let path = if root.is_dir() {
        root.join(relpath)
    } else {
        root.parent().unwrap_or_else(|| Path::new("")).join(relpath)
    };
    let absolute = if path.is_absolute() {
        path
    } else {
        std::env::current_dir()?.join(path)
    };
    let mut normalized = PathBuf::new();
    for component in absolute.components() {
        match component {
            std::path::Component::CurDir => {}
            std::path::Component::ParentDir => {
                normalized.pop();
            }
            _ => normalized.push(component.as_os_str()),
        }
    }
    Ok(normalized)
}

pub fn polyroi_bytearray(
    x: &[f32],
    y: &[f32],
    pos: Option<i32>,
    subpixel: bool,
) -> Result<Vec<u8>, UtilsError> {
    if x.is_empty() || x.len() != y.len() {
        return Err(UtilsError::RoiShapeMismatch);
    }
    let mut x_raw = Vec::<f32>::with_capacity(x.len());
    let mut y_raw = Vec::<f32>::with_capacity(y.len());
    let mut x_round = Vec::<i32>::with_capacity(x.len());
    let mut y_round = Vec::<i32>::with_capacity(y.len());
    for i in 0..x.len() {
        x_raw.push(x[i] + 0.5);
        y_raw.push(y[i] + 0.5);
        x_round.push(x_raw[i].round() as i32);
        y_round.push(y_raw[i].round() as i32);
    }
    let top = *y_round.iter().min().ok_or(UtilsError::RoiShapeMismatch)?;
    let left = *x_round.iter().min().ok_or(UtilsError::RoiShapeMismatch)?;
    let bottom = *y_round.iter().max().ok_or(UtilsError::RoiShapeMismatch)?;
    let right = *x_round.iter().max().ok_or(UtilsError::RoiShapeMismatch)?;
    let n_coords = x.len();
    if n_coords > u16::MAX as usize {
        return Err(UtilsError::RoiCoordinateOutOfRange);
    }
    for value in [top, left, bottom, right] {
        if value < i16::MIN as i32 || value > i16::MAX as i32 {
            return Err(UtilsError::RoiCoordinateOutOfRange);
        }
    }
    if let Some(pos) = pos {
        if pos == i32::MIN {
            return Err(UtilsError::RoiCoordinateOutOfRange);
        }
    }

    let bytes_header = 64usize;
    let bytes_total = bytes_header + n_coords * 2 * 2 + usize::from(subpixel) * n_coords * 2 * 4;
    let mut bytes = vec![0u8; bytes_total];
    bytes[0..4].copy_from_slice(b"Iout");
    bytes[4..6].copy_from_slice(&(227i16).to_be_bytes());
    bytes[6..8].copy_from_slice(&(0i16).to_be_bytes());
    bytes[8..10].copy_from_slice(&(top as i16).to_be_bytes());
    bytes[10..12].copy_from_slice(&(left as i16).to_be_bytes());
    bytes[12..14].copy_from_slice(&(bottom as i16).to_be_bytes());
    bytes[14..16].copy_from_slice(&(right as i16).to_be_bytes());
    bytes[16..18].copy_from_slice(&(n_coords as u16).to_be_bytes());
    if subpixel {
        bytes[50..52].copy_from_slice(&(128i16).to_be_bytes());
    }
    if let Some(pos) = pos {
        bytes[56..60].copy_from_slice(&pos.to_be_bytes());
    }

    for i in 0..n_coords {
        let x_offset = x_round[i] - left;
        let y_offset = y_round[i] - top;
        if x_offset < i16::MIN as i32
            || x_offset > i16::MAX as i32
            || y_offset < i16::MIN as i32
            || y_offset > i16::MAX as i32
        {
            return Err(UtilsError::RoiCoordinateOutOfRange);
        }
        let xs = bytes_header + 2 * i;
        let ys = xs + 2 * n_coords;
        bytes[xs..xs + 2].copy_from_slice(&(x_offset as i16).to_be_bytes());
        bytes[ys..ys + 2].copy_from_slice(&(y_offset as i16).to_be_bytes());
    }

    if subpixel {
        let base1 = bytes_header + n_coords * 2 * 2;
        let base2 = base1 + n_coords * 4;
        for i in 0..n_coords {
            let xs = base1 + 4 * i;
            let ys = base2 + 4 * i;
            bytes[xs..xs + 4].copy_from_slice(&x_raw[i].to_be_bytes());
            bytes[ys..ys + 4].copy_from_slice(&y_raw[i].to_be_bytes());
        }
    }

    Ok(bytes)
}

pub fn export_imagej_rois(
    fname: impl AsRef<Path>,
    polygons: &[Vec<Vec<[f32; 2]>>],
    set_position: bool,
    subpixel: bool,
) -> Result<PathBuf, UtilsError> {
    let mut fname = fname.as_ref().to_path_buf();
    if fname.extension().and_then(|ext| ext.to_str()) == Some("zip") {
        fname.set_extension("");
    }
    let zip_path = fname.with_extension("zip");
    let file = std::fs::File::create(&zip_path).map_err(|_| UtilsError::RoiWriteFailed)?;
    let mut roizip = zip::ZipWriter::new(file);
    let options =
        zip::write::SimpleFileOptions::default().compression_method(zip::CompressionMethod::Stored);
    for (pos_i, polygroup) in polygons.iter().enumerate() {
        let pos = pos_i + 1;
        for (poly_i, poly) in polygroup.iter().enumerate() {
            let mut x = Vec::<f32>::with_capacity(poly.len());
            let mut y = Vec::<f32>::with_capacity(poly.len());
            for point in poly {
                y.push(point[0]);
                x.push(point[1]);
            }
            let roi = polyroi_bytearray(
                &x,
                &y,
                if set_position { Some(pos as i32) } else { None },
                subpixel,
            )?;
            roizip
                .start_file(format!("{pos:03}_{:03}.roi", poly_i + 1), options)
                .map_err(|_| UtilsError::RoiWriteFailed)?;
            std::io::Write::write_all(&mut roizip, &roi).map_err(|_| UtilsError::RoiWriteFailed)?;
        }
    }
    roizip.finish().map_err(|_| UtilsError::RoiWriteFailed)?;
    Ok(zip_path)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn calculate_extents_returns_zero_without_objects() {
        let extents = calculate_extents(&[0; 9], &[3, 3]).unwrap();
        assert_eq!(extents, vec![0.0, 0.0]);
    }

    #[test]
    fn optimize_threshold_maximizes_matching_measure_with_prediction_closure() {
        let y_true_image = [1, 1, 0, 0];
        let y_true = [&y_true_image[..]];
        let prob = [0.8, 0.8, 0.1, 0.1];
        let yhat_prob = [&prob[..]];
        let (thr, value) = optimize_threshold(
            &y_true,
            &yhat_prob,
            0.4,
            OptimizeThresholdMeasure::Accuracy,
            &[0.5],
            Some([0.0, 1.0]),
            1e-3,
            32,
            |_i, prob_thresh, _nms_thresh| {
                if prob_thresh <= 0.7 {
                    Ok(vec![1, 1, 0, 0])
                } else {
                    Ok(vec![0, 0, 0, 0])
                }
            },
        )
        .unwrap();
        assert!((0.0..=0.7).contains(&thr));
        assert!((value - 1.0).abs() < 1e-6);
    }

    #[test]
    fn optimize_threshold_validates_input_like_python_scalar_checks() {
        let y_true_image = [1, 1, 0, 0];
        let y_true = [&y_true_image[..]];
        let prob = [0.8, 0.8, 0.1, 0.1];
        let yhat_prob = [&prob[..]];
        let err = optimize_threshold(
            &y_true,
            &yhat_prob,
            f32::NAN,
            OptimizeThresholdMeasure::Accuracy,
            &[0.5],
            Some([0.0, 1.0]),
            1e-3,
            32,
            |_i, _prob_thresh, _nms_thresh| Ok(vec![1, 1, 0, 0]),
        )
        .unwrap_err();
        assert_eq!(err, UtilsError::InvalidThresholdOptimizationInput);
    }

    #[test]
    fn gputools_available_is_false_without_python_gputools() {
        assert!(!gputools_available());
    }

    #[test]
    fn path_absolute_points_inside_vendored_stardist_package() {
        let path = path_absolute("models/examples/2D_demo/config.json");
        assert!(path.is_absolute());
        assert_eq!(
            path.file_name().and_then(|x| x.to_str()),
            Some("config.json")
        );
        assert!(path.ends_with("assets/models/examples/2D_demo/config.json"));
    }

    #[test]
    fn is_floatarray_matches_float_dtype_only() {
        assert!(_is_floatarray(ArrayDType::Float));
        assert!(!_is_floatarray(ArrayDType::Bool));
        assert!(!_is_floatarray(ArrayDType::Int));
        assert!(!_is_floatarray(ArrayDType::UInt));
    }

    #[test]
    fn abspath_joins_against_directory_or_parent_like_python() {
        let manifest = Path::new(env!("CARGO_MANIFEST_DIR"));
        let from_dir = abspath(manifest, "src/../Cargo.toml").unwrap();
        assert_eq!(from_dir, manifest.join("Cargo.toml"));

        let from_file = abspath(manifest.join("src/lib.rs"), "utils.rs").unwrap();
        assert_eq!(from_file, manifest.join("src/utils.rs"));
    }

    #[test]
    fn edt_prob_scipy_normalizes_each_2d_object() {
        let labels = [
            0, 0, 0, 0, 0, //
            0, 1, 1, 1, 0, //
            0, 1, 1, 1, 0, //
            0, 1, 1, 1, 0, //
            0, 0, 0, 0, 0, //
        ];
        let prob = _edt_prob_scipy(&labels, &[5, 5], None).unwrap();
        assert_eq!(prob[2 * 5 + 2], 1.0);
        assert!((prob[1 * 5 + 1] - 0.5).abs() < 1e-6);
        assert_eq!(prob[0], 0.0);
    }

    #[test]
    fn edt_prob_handles_constant_positive_2d_image_like_padded_python_case() {
        let labels = [1; 9];
        let prob = edt_prob(&labels, &[3, 3], None).unwrap();
        assert_eq!(prob[1 * 3 + 1], 1.0);
        assert!((prob[0] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn edt_prob_scipy_normalizes_3d_object() {
        let labels = [1; 27];
        let prob = _edt_prob_scipy(&labels, &[3, 3, 3], None).unwrap();
        assert_eq!(prob[(1 * 3 + 1) * 3 + 1], 1.0);
        assert!((prob[0] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn edt_prob_edt_uses_anisotropy_and_validates_shape() {
        let labels = [
            0, 0, 0, 0, 0, //
            0, 1, 1, 1, 0, //
            0, 1, 1, 1, 0, //
            0, 1, 1, 1, 0, //
            0, 0, 0, 0, 0, //
        ];
        let isotropic = _edt_prob_edt(&labels, &[5, 5], None).unwrap();
        let anisotropic = _edt_prob_edt(&labels, &[5, 5], Some(&[2.0, 1.0])).unwrap();
        assert!(anisotropic[1 * 5 + 2] > isotropic[1 * 5 + 2]);
        assert_eq!(
            _edt_prob_scipy(&labels, &[5, 5], Some(&[1.0])).unwrap_err(),
            UtilsError::AnisotropyShapeMismatch
        );
    }

    #[test]
    fn invert_dict_groups_labels_by_class() {
        let inverted = _invert_dict(&[(1, Some(2)), (2, Some(1)), (3, Some(2)), (4, None)]);
        assert_eq!(inverted.get(&Some(2)), Some(&vec![1, 3]));
        assert_eq!(inverted.get(&Some(1)), Some(&vec![2]));
        assert_eq!(inverted.get(&None), Some(&vec![4]));
    }

    #[test]
    fn mask_to_categorical_broadcasts_scalar_class() {
        let y = [0, 1, 2, 0];
        let (mask, cls_dict) =
            mask_to_categorical(&y, &[2, 2], 2, ClassAssignment::Single(Some(1)), false).unwrap();
        assert_eq!(cls_dict, None);
        assert_eq!(
            mask,
            vec![
                1.0, 0.0, 0.0, //
                0.0, 1.0, 0.0, //
                0.0, 1.0, 0.0, //
                1.0, 0.0, 0.0, //
            ]
        );
    }

    #[test]
    fn mask_to_categorical_uses_label_dict_and_returns_inverse() {
        let y = [0, 1, 2, 3];
        let (mask, cls_dict) = mask_to_categorical(
            &y,
            &[2, 2],
            2,
            ClassAssignment::Dict(vec![(1, Some(2)), (2, Some(1)), (3, Some(2))]),
            true,
        )
        .unwrap();
        assert_eq!(
            mask,
            vec![
                1.0, 0.0, 0.0, //
                0.0, 0.0, 1.0, //
                0.0, 1.0, 0.0, //
                0.0, 0.0, 1.0, //
            ]
        );
        let cls_dict = cls_dict.unwrap();
        assert_eq!(cls_dict.get(&Some(2)), Some(&vec![1, 3]));
        assert_eq!(cls_dict.get(&Some(1)), Some(&vec![2]));
    }

    #[test]
    fn mask_to_categorical_marks_ignored_objects_except_background_channel() {
        let y = [0, 1, 2];
        let (mask, _) = mask_to_categorical(
            &y,
            &[3],
            2,
            ClassAssignment::Dict(vec![(1, None), (2, Some(2))]),
            false,
        )
        .unwrap();
        assert_eq!(
            mask,
            vec![
                1.0, 0.0, 0.0, //
                0.0, -1.0, -1.0, //
                0.0, 0.0, 1.0, //
            ]
        );
    }

    #[test]
    fn mask_to_categorical_rejects_invalid_inputs_like_python() {
        assert_eq!(
            mask_to_categorical(&[-1], &[1], 1, ClassAssignment::Single(Some(1)), false)
                .unwrap_err(),
            UtilsError::NegativeLabel
        );
        assert_eq!(
            mask_to_categorical(&[1], &[1], 0, ClassAssignment::Single(Some(1)), false)
                .unwrap_err(),
            UtilsError::InvalidClassCount
        );
        assert_eq!(
            mask_to_categorical(
                &[1, 2],
                &[2],
                2,
                ClassAssignment::Dict(vec![(1, Some(1))]),
                false
            )
            .unwrap_err(),
            UtilsError::MissingClassLabel
        );
        assert_eq!(
            mask_to_categorical(&[1], &[1], 2, ClassAssignment::Single(Some(3)), false)
                .unwrap_err(),
            UtilsError::WrongClassId {
                class_id: 3,
                n_classes: 2
            }
        );
    }

    #[test]
    fn sample_points_samples_masked_points_with_replacement() {
        let mask = [
            false, true, false, //
            true, false, false, //
            false, false, true, //
        ];
        let points = sample_points(6, &mask, &[3, 3], None, None, 17).unwrap();
        assert_eq!(points.len(), 6);
        for point in points {
            assert!(matches!(point, [0, 1] | [1, 0] | [2, 2]));
        }
    }

    #[test]
    fn sample_points_excludes_boundary_like_python() {
        let mask = [true; 25];
        let points = sample_points(10, &mask, &[5, 5], None, Some(1), 123).unwrap();
        assert_eq!(points.len(), 10);
        for [y, x] in points {
            assert!((1..4).contains(&y));
            assert!((1..4).contains(&x));
        }
    }

    #[test]
    fn sample_points_uses_probability_weights() {
        let mask = [true, true, true, true];
        let prob = [0.0, 0.0, 0.0, 5.0];
        let points = sample_points(5, &mask, &[2, 2], Some(&prob), Some(0), 99).unwrap();
        assert_eq!(points, vec![[1, 1]; 5]);
    }

    #[test]
    fn sample_points_rejects_invalid_inputs() {
        assert_eq!(
            sample_points(1, &[true], &[1], None, None, 1).unwrap_err(),
            UtilsError::WrongMaskDimension
        );
        assert_eq!(
            sample_points(1, &[true], &[1, 1], Some(&[0.0, 1.0]), None, 1).unwrap_err(),
            UtilsError::ProbShapeMismatch
        );
        assert_eq!(
            sample_points(1, &[false; 4], &[2, 2], None, None, 1).unwrap_err(),
            UtilsError::NoSamplePoints
        );
        assert_eq!(
            sample_points(1, &[true, true], &[1, 2], Some(&[0.0, 0.0]), None, 1).unwrap_err(),
            UtilsError::InvalidProbabilityWeights
        );
    }

    #[test]
    fn fill_label_holes_fills_enclosed_2d_hole() {
        let labels = [
            1, 1, 1, 0, //
            1, 0, 1, 0, //
            1, 1, 1, 0, //
            0, 0, 0, 0, //
        ];
        let filled = fill_label_holes(&labels, &[4, 4]).unwrap();
        assert_eq!(
            filled,
            vec![
                1, 1, 1, 0, //
                1, 1, 1, 0, //
                1, 1, 1, 0, //
                0, 0, 0, 0, //
            ]
        );
    }

    #[test]
    fn fill_label_holes_preserves_border_connected_background() {
        let labels = [
            1, 1, 1, 0, //
            1, 0, 0, 0, //
            1, 1, 1, 0, //
            0, 0, 0, 0, //
        ];
        let filled = fill_label_holes(&labels, &[4, 4]).unwrap();
        assert_eq!(filled, labels);
    }

    #[test]
    fn fill_label_holes_preserves_and_fills_negative_labels() {
        let labels = [
            -1, -1, -1, 0, //
            -1, 0, -1, 0, //
            -1, -1, -1, 0, //
            0, 0, 0, 0, //
        ];
        let filled = fill_label_holes(&labels, &[4, 4]).unwrap();
        assert_eq!(
            filled,
            vec![
                -1, -1, -1, 0, //
                -1, -1, -1, 0, //
                -1, -1, -1, 0, //
                0, 0, 0, 0, //
            ]
        );
    }

    #[test]
    fn fill_label_holes_fills_enclosed_3d_hole() {
        let mut labels = vec![1i32; 3 * 3 * 3];
        labels[(1 * 3 + 1) * 3 + 1] = 0;
        let filled = fill_label_holes(&labels, &[3, 3, 3]).unwrap();
        assert_eq!(filled, vec![1; 3 * 3 * 3]);
    }

    #[test]
    fn underscore_fill_label_holes_processes_all_nonzero_labels() {
        let labels = [
            1, 1, 1, 0, -2, -2, -2, //
            1, 0, 1, 0, -2, 0, -2, //
            1, 1, 1, 0, -2, -2, -2, //
        ];
        let filled = _fill_label_holes(&labels, &[3, 7]).unwrap();
        assert_eq!(
            filled,
            vec![
                1, 1, 1, 0, -2, -2, -2, //
                1, 1, 1, 0, -2, -2, -2, //
                1, 1, 1, 0, -2, -2, -2, //
            ]
        );
    }

    #[test]
    fn calculate_extents_matches_region_bbox_median_2d() {
        let labels = [
            1, 1, 0, 0, 0, //
            1, 1, 0, 2, 2, //
            0, 0, 0, 2, 2, //
            0, 0, 0, 2, 2, //
        ];
        let extents = calculate_extents(&labels, &[4, 5]).unwrap();
        assert_eq!(extents, vec![2.5, 2.0]);
    }

    #[test]
    fn calculate_extents_matches_region_bbox_median_3d() {
        let mut labels = vec![0u16; 3 * 4 * 5];
        labels[6] = 1;
        labels[11] = 1;
        labels[(2 * 4 + 2) * 5 + 3] = 2;
        labels[(2 * 4 + 3) * 5 + 4] = 2;
        let extents = calculate_extents(&labels, &[3, 4, 5]).unwrap();
        assert_eq!(extents, vec![1.0, 2.0, 1.5]);
    }

    #[test]
    fn grid_divisible_patch_size_rounds_up_like_python() {
        let size = grid_divisible_patch_size(&[63, 64, 65], &[16, 16, 32]).unwrap();
        assert_eq!(size, vec![64, 64, 96]);
    }

    #[test]
    fn polyroi_bytearray_matches_imagej_polygon_header_layout() {
        let roi = polyroi_bytearray(&[1.0, 4.0, 4.0], &[2.0, 2.0, 6.0], Some(7), true).unwrap();
        assert_eq!(&roi[0..4], b"Iout");
        assert_eq!(i16::from_be_bytes([roi[4], roi[5]]), 227);
        assert_eq!(i16::from_be_bytes([roi[6], roi[7]]), 0);
        assert_eq!(i16::from_be_bytes([roi[8], roi[9]]), 3);
        assert_eq!(i16::from_be_bytes([roi[10], roi[11]]), 2);
        assert_eq!(i16::from_be_bytes([roi[12], roi[13]]), 7);
        assert_eq!(i16::from_be_bytes([roi[14], roi[15]]), 5);
        assert_eq!(u16::from_be_bytes([roi[16], roi[17]]), 3);
        assert_eq!(i16::from_be_bytes([roi[50], roi[51]]), 128);
        assert_eq!(i32::from_be_bytes([roi[56], roi[57], roi[58], roi[59]]), 7);
        assert_eq!(i16::from_be_bytes([roi[64], roi[65]]), 0);
        assert_eq!(i16::from_be_bytes([roi[70], roi[71]]), 0);
        assert_eq!(
            f32::from_be_bytes([roi[76], roi[77], roi[78], roi[79]]),
            1.5
        );
        assert_eq!(
            f32::from_be_bytes([roi[88], roi[89], roi[90], roi[91]]),
            2.5
        );
    }

    #[test]
    fn export_imagej_rois_writes_positioned_roi_zip() {
        let path = std::env::temp_dir().join(format!(
            "stardist_rs_imagej_rois_{}.zip",
            std::process::id()
        ));
        let _ = std::fs::remove_file(&path);
        let polygons = vec![
            vec![vec![[2.0, 1.0], [2.0, 4.0], [6.0, 4.0]]],
            vec![vec![[0.0, 0.0], [0.0, 2.0], [2.0, 0.0]]],
        ];
        let written = export_imagej_rois(&path, &polygons, true, false).unwrap();
        assert_eq!(written, path);

        let file = std::fs::File::open(&written).unwrap();
        let mut zip = zip::ZipArchive::new(file).unwrap();
        assert_eq!(zip.len(), 2);
        {
            let mut roi = zip.by_name("001_001.roi").unwrap();
            let mut bytes = Vec::new();
            std::io::Read::read_to_end(&mut roi, &mut bytes).unwrap();
            assert_eq!(&bytes[0..4], b"Iout");
            assert_eq!(
                i32::from_be_bytes([bytes[56], bytes[57], bytes[58], bytes[59]]),
                1
            );
            assert_eq!(bytes.len(), 64 + 3 * 2 * 2);
        }
        {
            let mut roi = zip.by_name("002_001.roi").unwrap();
            let mut bytes = Vec::new();
            std::io::Read::read_to_end(&mut roi, &mut bytes).unwrap();
            assert_eq!(
                i32::from_be_bytes([bytes[56], bytes[57], bytes[58], bytes[59]]),
                2
            );
        }
        let _ = std::fs::remove_file(&written);
    }
}
