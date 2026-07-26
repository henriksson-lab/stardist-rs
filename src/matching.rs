#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum MatchingError {
    #[error("labels must be non-negative integers")]
    NegativeLabel,
    #[error("labels must be sequential non-negative integers")]
    NonSequentialLabel,
    #[error("label arrays must have the same shape")]
    ShapeMismatch,
    #[error("y_true and y_pred must have the same number of images")]
    DatasetLengthMismatch,
    #[error("overlap shape must match overlap data length")]
    OverlapShapeMismatch,
    #[error("offset must be strictly positive")]
    InvalidOffset,
    #[error("matching criterion is not supported")]
    UnsupportedCriterion,
    #[error("ys must have two or more entries")]
    TooFewImages,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RelabelSequential {
    pub relabeled: Vec<u32>,
    pub forward_map: Vec<u32>,
    pub inverse_map: Vec<u32>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MatchingCriterion {
    Iou,
    Iot,
    Iop,
}

#[derive(Clone, Debug, PartialEq)]
pub struct MatchingStats {
    pub criterion: MatchingCriterion,
    pub thresh: f32,
    pub fp: u32,
    pub tp: u32,
    pub fn_: u32,
    pub precision: f32,
    pub recall: f32,
    pub accuracy: f32,
    pub f1: f32,
    pub n_true: u32,
    pub n_pred: u32,
    pub mean_true_score: f32,
    pub mean_matched_score: f32,
    pub panoptic_quality: f32,
    pub matched_pairs: Option<Vec<(u32, u32)>>,
    pub matched_scores: Option<Vec<f32>>,
    pub matched_tps: Option<Vec<usize>>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct DatasetMatchingStats {
    pub criterion: MatchingCriterion,
    pub thresh: f32,
    pub fp: u32,
    pub tp: u32,
    pub fn_: u32,
    pub precision: f32,
    pub recall: f32,
    pub accuracy: f32,
    pub f1: f32,
    pub n_true: u32,
    pub n_pred: u32,
    pub mean_true_score: f32,
    pub mean_matched_score: f32,
    pub panoptic_quality: f32,
    pub by_image: bool,
}

pub fn label_are_sequential(y: &[u32]) -> bool {
    let mut max_label = 0u32;
    for label in y {
        max_label = max_label.max(*label);
    }
    let mut present = vec![false; max_label as usize + 1];
    for label in y {
        present[*label as usize] = true;
    }
    for label in 1..=max_label as usize {
        if !present[label] {
            return false;
        }
    }
    true
}

pub fn is_array_of_integers() -> bool {
    true
}

pub fn _check_label_array(y: &[u32], check_sequential: bool) -> Result<bool, MatchingError> {
    if y.is_empty() {
        return Ok(true);
    }
    if check_sequential && !label_are_sequential(y) {
        return Err(MatchingError::NonSequentialLabel);
    }
    Ok(true)
}

pub fn label_overlap(
    x: &[u32],
    y: &[u32],
    check: bool,
) -> Result<(Vec<u32>, [usize; 2]), MatchingError> {
    if check {
        _check_label_array(x, true)?;
        _check_label_array(y, true)?;
        if x.len() != y.len() {
            return Err(MatchingError::ShapeMismatch);
        }
    }
    Ok(_label_overlap(x, y))
}

pub fn _label_overlap(x: &[u32], y: &[u32]) -> (Vec<u32>, [usize; 2]) {
    let mut max_x = 0u32;
    let mut max_y = 0u32;
    for value in x {
        max_x = max_x.max(*value);
    }
    for value in y {
        max_y = max_y.max(*value);
    }
    let rows = max_x as usize + 1;
    let cols = max_y as usize + 1;
    let mut overlap = vec![0u32; rows * cols];
    for i in 0..x.len().min(y.len()) {
        overlap[x[i] as usize * cols + y[i] as usize] += 1;
    }
    (overlap, [rows, cols])
}

pub fn _safe_divide(x: f32, y: f32, eps: f32) -> f32 {
    if y.abs() > eps { x / y } else { 0.0 }
}

pub fn intersection_over_union(
    overlap: &[u32],
    shape: [usize; 2],
) -> Result<Vec<f32>, MatchingError> {
    if overlap.len() != shape[0] * shape[1] {
        return Err(MatchingError::OverlapShapeMismatch);
    }
    let mut total = 0u32;
    for value in overlap {
        total += *value;
    }
    if total == 0 {
        return Ok(overlap.iter().map(|value| *value as f32).collect());
    }
    let mut n_pixels_pred = vec![0u32; shape[1]];
    let mut n_pixels_true = vec![0u32; shape[0]];
    for r in 0..shape[0] {
        for c in 0..shape[1] {
            let value = overlap[r * shape[1] + c];
            n_pixels_true[r] += value;
            n_pixels_pred[c] += value;
        }
    }
    let mut result = vec![0.0f32; overlap.len()];
    for r in 0..shape[0] {
        for c in 0..shape[1] {
            let value = overlap[r * shape[1] + c] as f32;
            let denom = (n_pixels_pred[c] + n_pixels_true[r]) as f32 - value;
            result[r * shape[1] + c] = _safe_divide(value, denom, 1e-10);
        }
    }
    Ok(result)
}

pub fn intersection_over_true(
    overlap: &[u32],
    shape: [usize; 2],
) -> Result<Vec<f32>, MatchingError> {
    if overlap.len() != shape[0] * shape[1] {
        return Err(MatchingError::OverlapShapeMismatch);
    }
    let mut total = 0u32;
    for value in overlap {
        total += *value;
    }
    if total == 0 {
        return Ok(overlap.iter().map(|value| *value as f32).collect());
    }
    let mut n_pixels_true = vec![0u32; shape[0]];
    for r in 0..shape[0] {
        for c in 0..shape[1] {
            n_pixels_true[r] += overlap[r * shape[1] + c];
        }
    }
    let mut result = vec![0.0f32; overlap.len()];
    for r in 0..shape[0] {
        for c in 0..shape[1] {
            result[r * shape[1] + c] = _safe_divide(
                overlap[r * shape[1] + c] as f32,
                n_pixels_true[r] as f32,
                1e-10,
            );
        }
    }
    Ok(result)
}

pub fn intersection_over_pred(
    overlap: &[u32],
    shape: [usize; 2],
) -> Result<Vec<f32>, MatchingError> {
    if overlap.len() != shape[0] * shape[1] {
        return Err(MatchingError::OverlapShapeMismatch);
    }
    let mut total = 0u32;
    for value in overlap {
        total += *value;
    }
    if total == 0 {
        return Ok(overlap.iter().map(|value| *value as f32).collect());
    }
    let mut n_pixels_pred = vec![0u32; shape[1]];
    for r in 0..shape[0] {
        for c in 0..shape[1] {
            n_pixels_pred[c] += overlap[r * shape[1] + c];
        }
    }
    let mut result = vec![0.0f32; overlap.len()];
    for r in 0..shape[0] {
        for c in 0..shape[1] {
            result[r * shape[1] + c] = _safe_divide(
                overlap[r * shape[1] + c] as f32,
                n_pixels_pred[c] as f32,
                1e-10,
            );
        }
    }
    Ok(result)
}

pub fn precision(tp: u32, fp: u32, _fn_: u32) -> f32 {
    if tp > 0 {
        tp as f32 / (tp + fp) as f32
    } else {
        0.0
    }
}

pub fn recall(tp: u32, _fp: u32, fn_: u32) -> f32 {
    if tp > 0 {
        tp as f32 / (tp + fn_) as f32
    } else {
        0.0
    }
}

pub fn accuracy(tp: u32, fp: u32, fn_: u32) -> f32 {
    if tp > 0 {
        tp as f32 / (tp + fp + fn_) as f32
    } else {
        0.0
    }
}

pub fn f1(tp: u32, fp: u32, fn_: u32) -> f32 {
    if tp > 0 {
        (2 * tp) as f32 / (2 * tp + fp + fn_) as f32
    } else {
        0.0
    }
}

pub fn matching(
    y_true: &[u32],
    y_pred: &[u32],
    thresh: Option<f32>,
    criterion: MatchingCriterion,
    report_matches: bool,
) -> Result<MatchingStats, MatchingError> {
    _check_label_array(y_true, false)?;
    _check_label_array(y_pred, false)?;
    if y_true.len() != y_pred.len() {
        return Err(MatchingError::ShapeMismatch);
    }
    let thresh = thresh.unwrap_or(0.0);

    let relabel_true = relabel_sequential(y_true, 1)?;
    let relabel_pred = relabel_sequential(y_pred, 1)?;
    let (overlap, overlap_shape) =
        label_overlap(&relabel_true.relabeled, &relabel_pred.relabeled, false)?;
    let scores_full = match criterion {
        MatchingCriterion::Iou => intersection_over_union(&overlap, overlap_shape)?,
        MatchingCriterion::Iot => intersection_over_true(&overlap, overlap_shape)?,
        MatchingCriterion::Iop => intersection_over_pred(&overlap, overlap_shape)?,
    };

    let n_true = overlap_shape[0].saturating_sub(1);
    let n_pred = overlap_shape[1].saturating_sub(1);
    let n_matched = n_true.min(n_pred);
    let mut scores = vec![0.0f32; n_true * n_pred];
    for r in 0..n_true {
        for c in 0..n_pred {
            scores[r * n_pred + c] = scores_full[(r + 1) * overlap_shape[1] + c + 1];
        }
    }

    let mut assigned_pairs = Vec::<(usize, usize)>::new();
    let not_trivial = n_matched > 0;
    if not_trivial {
        if n_true <= n_pred {
            let states = 1usize << n_pred;
            let mut dp = vec![f32::NEG_INFINITY; states];
            let mut parent = vec![usize::MAX; states];
            dp[0] = 0.0;
            for r in 0..n_true {
                let mut next = vec![f32::NEG_INFINITY; states];
                let mut next_parent = vec![usize::MAX; states];
                for mask in 0..states {
                    if mask.count_ones() as usize != r || !dp[mask].is_finite() {
                        continue;
                    }
                    for c in 0..n_pred {
                        if mask & (1usize << c) != 0 {
                            continue;
                        }
                        let score = scores[r * n_pred + c];
                        let gain = if score >= thresh { 1.0 } else { 0.0 }
                            + score / (2.0 * n_matched as f32);
                        let next_mask = mask | (1usize << c);
                        let candidate = dp[mask] + gain;
                        if candidate > next[next_mask] {
                            next[next_mask] = candidate;
                            next_parent[next_mask] = c;
                        }
                    }
                }
                for mask in 0..states {
                    if next[mask] > dp[mask] {
                        dp[mask] = next[mask];
                        parent[mask] = next_parent[mask];
                    }
                }
            }
            let mut best_mask = 0usize;
            let mut best_score = f32::NEG_INFINITY;
            for mask in 0..states {
                if mask.count_ones() as usize == n_true && dp[mask] > best_score {
                    best_score = dp[mask];
                    best_mask = mask;
                }
            }
            let mut mask = best_mask;
            let mut reversed_pairs = Vec::<(usize, usize)>::with_capacity(n_true);
            for r_rev in (0..n_true).rev() {
                let c = parent[mask];
                reversed_pairs.push((r_rev, c));
                mask &= !(1usize << c);
            }
            reversed_pairs.reverse();
            assigned_pairs = reversed_pairs;
        } else {
            let states = 1usize << n_true;
            let mut dp = vec![f32::NEG_INFINITY; states];
            let mut parent = vec![usize::MAX; states];
            dp[0] = 0.0;
            for c in 0..n_pred {
                let mut next = vec![f32::NEG_INFINITY; states];
                let mut next_parent = vec![usize::MAX; states];
                for mask in 0..states {
                    if mask.count_ones() as usize != c || !dp[mask].is_finite() {
                        continue;
                    }
                    for r in 0..n_true {
                        if mask & (1usize << r) != 0 {
                            continue;
                        }
                        let score = scores[r * n_pred + c];
                        let gain = if score >= thresh { 1.0 } else { 0.0 }
                            + score / (2.0 * n_matched as f32);
                        let next_mask = mask | (1usize << r);
                        let candidate = dp[mask] + gain;
                        if candidate > next[next_mask] {
                            next[next_mask] = candidate;
                            next_parent[next_mask] = r;
                        }
                    }
                }
                for mask in 0..states {
                    if next[mask] > dp[mask] {
                        dp[mask] = next[mask];
                        parent[mask] = next_parent[mask];
                    }
                }
            }
            let mut best_mask = 0usize;
            let mut best_score = f32::NEG_INFINITY;
            for mask in 0..states {
                if mask.count_ones() as usize == n_pred && dp[mask] > best_score {
                    best_score = dp[mask];
                    best_mask = mask;
                }
            }
            let mut mask = best_mask;
            let mut pairs_by_pred = Vec::<(usize, usize)>::with_capacity(n_pred);
            for c_rev in (0..n_pred).rev() {
                let r = parent[mask];
                pairs_by_pred.push((r, c_rev));
                mask &= !(1usize << r);
            }
            pairs_by_pred.reverse();
            pairs_by_pred.sort_by_key(|(r, _)| *r);
            assigned_pairs = pairs_by_pred;
        }
    }

    let mut match_ok = Vec::<bool>::with_capacity(assigned_pairs.len());
    let mut tp = 0u32;
    let mut sum_matched_score = 0.0f32;
    for (r, c) in &assigned_pairs {
        let score = scores[*r * n_pred + *c];
        let ok = score >= thresh;
        match_ok.push(ok);
        if ok {
            tp += 1;
            sum_matched_score += score;
        }
    }
    let fp = n_pred as u32 - tp;
    let fn_ = n_true as u32 - tp;
    let mean_matched_score = _safe_divide(sum_matched_score, tp as f32, 1e-10);
    let mean_true_score = _safe_divide(sum_matched_score, n_true as f32, 1e-10);
    let panoptic_quality = _safe_divide(
        sum_matched_score,
        tp as f32 + fp as f32 / 2.0 + fn_ as f32 / 2.0,
        1e-10,
    );

    let (matched_pairs, matched_scores, matched_tps) = if report_matches {
        let mut pairs = Vec::<(u32, u32)>::with_capacity(assigned_pairs.len());
        let mut pair_scores = Vec::<f32>::with_capacity(assigned_pairs.len());
        let mut tps = Vec::<usize>::new();
        for (i, (r, c)) in assigned_pairs.iter().enumerate() {
            pairs.push((
                relabel_true.inverse_map[r + 1],
                relabel_pred.inverse_map[c + 1],
            ));
            pair_scores.push(scores[*r * n_pred + *c]);
            if match_ok[i] {
                tps.push(i);
            }
        }
        (Some(pairs), Some(pair_scores), Some(tps))
    } else {
        (None, None, None)
    };

    Ok(MatchingStats {
        criterion,
        thresh,
        fp,
        tp,
        fn_,
        precision: precision(tp, fp, fn_),
        recall: recall(tp, fp, fn_),
        accuracy: accuracy(tp, fp, fn_),
        f1: f1(tp, fp, fn_),
        n_true: n_true as u32,
        n_pred: n_pred as u32,
        mean_true_score,
        mean_matched_score,
        panoptic_quality,
        matched_pairs,
        matched_scores,
        matched_tps,
    })
}

pub fn matching_dataset(
    y_true: &[&[u32]],
    y_pred: &[&[u32]],
    thresh: &[f32],
    criterion: MatchingCriterion,
    by_image: bool,
    _show_progress: bool,
    _parallel: bool,
) -> Result<Vec<DatasetMatchingStats>, MatchingError> {
    if y_true.len() != y_pred.len() {
        return Err(MatchingError::DatasetLengthMismatch);
    }
    let mut pairs = Vec::<(&[u32], &[u32])>::with_capacity(y_true.len());
    for i in 0..y_true.len() {
        pairs.push((y_true[i], y_pred[i]));
    }
    matching_dataset_lazy(
        &pairs,
        thresh,
        criterion,
        by_image,
        _show_progress,
        _parallel,
    )
}

pub fn matching_dataset_lazy(
    y_gen: &[(&[u32], &[u32])],
    thresh: &[f32],
    criterion: MatchingCriterion,
    by_image: bool,
    _show_progress: bool,
    _parallel: bool,
) -> Result<Vec<DatasetMatchingStats>, MatchingError> {
    let thresh = if thresh.is_empty() {
        vec![0.5]
    } else {
        thresh.to_vec()
    };
    let mut stats_all = Vec::<Vec<MatchingStats>>::with_capacity(y_gen.len());
    for (y_t, y_p) in y_gen {
        let mut image_stats = Vec::<MatchingStats>::with_capacity(thresh.len());
        for thr in &thresh {
            image_stats.push(matching(y_t, y_p, Some(*thr), criterion, false)?);
        }
        stats_all.push(image_stats);
    }

    let n_images = stats_all.len();
    let mut accumulate = Vec::<DatasetMatchingStats>::with_capacity(thresh.len());
    for (i, thr) in thresh.iter().enumerate() {
        let mut fp = 0u32;
        let mut tp = 0u32;
        let mut fn_ = 0u32;
        let mut precision_sum = 0.0f32;
        let mut recall_sum = 0.0f32;
        let mut accuracy_sum = 0.0f32;
        let mut f1_sum = 0.0f32;
        let mut n_true = 0u32;
        let mut n_pred = 0u32;
        let mut mean_true_score_sum = 0.0f32;
        let mut mean_matched_score_sum = 0.0f32;
        let mut panoptic_quality_sum = 0.0f32;

        for stats in &stats_all {
            let s = &stats[i];
            fp += s.fp;
            tp += s.tp;
            fn_ += s.fn_;
            precision_sum += s.precision;
            recall_sum += s.recall;
            accuracy_sum += s.accuracy;
            f1_sum += s.f1;
            n_true += s.n_true;
            n_pred += s.n_pred;
            if by_image {
                mean_true_score_sum += s.mean_true_score;
            } else {
                mean_true_score_sum += s.mean_true_score * s.n_true as f32;
            }
            mean_matched_score_sum += s.mean_matched_score;
            panoptic_quality_sum += s.panoptic_quality;
        }

        if by_image {
            let denom = n_images as f32;
            accumulate.push(DatasetMatchingStats {
                criterion,
                thresh: *thr,
                fp,
                tp,
                fn_,
                precision: _safe_divide(precision_sum, denom, 1e-10),
                recall: _safe_divide(recall_sum, denom, 1e-10),
                accuracy: _safe_divide(accuracy_sum, denom, 1e-10),
                f1: _safe_divide(f1_sum, denom, 1e-10),
                n_true,
                n_pred,
                mean_true_score: _safe_divide(mean_true_score_sum, denom, 1e-10),
                mean_matched_score: _safe_divide(mean_matched_score_sum, denom, 1e-10),
                panoptic_quality: _safe_divide(panoptic_quality_sum, denom, 1e-10),
                by_image,
            });
        } else {
            let sum_matched_score = mean_true_score_sum;
            let mean_matched_score = _safe_divide(sum_matched_score, tp as f32, 1e-10);
            let mean_true_score = _safe_divide(sum_matched_score, n_true as f32, 1e-10);
            let panoptic_quality = _safe_divide(
                sum_matched_score,
                tp as f32 + fp as f32 / 2.0 + fn_ as f32 / 2.0,
                1e-10,
            );
            accumulate.push(DatasetMatchingStats {
                criterion,
                thresh: *thr,
                fp,
                tp,
                fn_,
                precision: precision(tp, fp, fn_),
                recall: recall(tp, fp, fn_),
                accuracy: accuracy(tp, fp, fn_),
                f1: f1(tp, fp, fn_),
                n_true,
                n_pred,
                mean_true_score,
                mean_matched_score,
                panoptic_quality,
                by_image,
            });
        }
    }

    Ok(accumulate)
}

pub fn relabel_sequential(
    label_field: &[u32],
    offset: u32,
) -> Result<RelabelSequential, MatchingError> {
    if offset == 0 {
        return Err(MatchingError::InvalidOffset);
    }
    let mut max_label = 0u32;
    for label in label_field {
        max_label = max_label.max(*label);
    }
    let mut labels = label_field.to_vec();
    labels.sort_unstable();
    labels.dedup();
    let labels0 = labels
        .into_iter()
        .filter(|label| *label != 0)
        .collect::<Vec<_>>();
    let new_max_label = offset - 1 + labels0.len() as u32;
    let mut new_labels0 = Vec::with_capacity(labels0.len());
    for label in offset..=new_max_label {
        new_labels0.push(label);
    }
    let mut forward_map = vec![0u32; max_label as usize + 1];
    for (old, new) in labels0.iter().zip(new_labels0.iter()) {
        forward_map[*old as usize] = *new;
    }
    let mut inverse_map = vec![0u32; new_max_label as usize + 1];
    for (i, old) in labels0.iter().enumerate() {
        inverse_map[offset as usize + i] = *old;
    }
    let mut relabeled = Vec::with_capacity(label_field.len());
    for label in label_field {
        relabeled.push(forward_map[*label as usize]);
    }
    Ok(RelabelSequential {
        relabeled,
        forward_map,
        inverse_map,
    })
}

pub fn group_matching_labels(
    ys: &[&[u32]],
    thresh: f32,
    criterion: MatchingCriterion,
) -> Result<Vec<Vec<u32>>, MatchingError> {
    if ys.len() <= 1 {
        return Err(MatchingError::TooFewImages);
    }
    for y in ys {
        _check_label_array(y, false)?;
    }
    let image_len = ys[0].len();
    for y in ys {
        if y.len() != image_len {
            return Err(MatchingError::ShapeMismatch);
        }
    }

    let mut ys_grouped = Vec::<Vec<u32>>::with_capacity(ys.len());
    ys_grouped.push(ys[0].to_vec());
    let mut next_id = 0u32;
    for label in &ys_grouped[0] {
        next_id = next_id.max(*label + 1);
    }

    for i in 0..ys.len() - 1 {
        let y_prev = &ys_grouped[i];
        let y = ys[i + 1];
        let res = matching(y_prev, y, Some(thresh), criterion, true)?;
        let mut relabel = Vec::<(u32, u32)>::new();
        if let (Some(matched_pairs), Some(matched_tps)) = (&res.matched_pairs, &res.matched_tps) {
            for index in matched_tps {
                let pair = matched_pairs[*index];
                relabel.push((pair.1, pair.0));
            }
        }

        let mut labels = y.to_vec();
        labels.sort_unstable();
        labels.dedup();
        let mut y_grouped = vec![0u32; y.len()];
        for label in labels {
            if label == 0 {
                continue;
            }
            let mut mapped = None;
            for (from, to) in &relabel {
                if *from == label {
                    mapped = Some(*to);
                    break;
                }
            }
            let out_label = if let Some(mapped) = mapped {
                mapped
            } else {
                let assigned = next_id;
                next_id += 1;
                assigned
            };
            for idx in 0..y.len() {
                if y[idx] == label {
                    y_grouped[idx] = out_label;
                }
            }
        }
        ys_grouped.push(y_grouped);
    }

    Ok(ys_grouped)
}

pub fn _shuffle_labels(y: &[u32], seed: u64) -> Result<Vec<u32>, MatchingError> {
    _check_label_array(y, false)?;
    let mut y2 = vec![0u32; y.len()];
    let mut ids = y.to_vec();
    ids.sort_unstable();
    ids.dedup();
    ids.retain(|label| *label != 0);

    let mut permuted = ids.clone();
    let mut state = seed;
    for i in (1..permuted.len()).rev() {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        let r = ((state >> 11) as f64) * (1.0 / ((1u64 << 53) as f64));
        let j = (r * (i + 1) as f64).floor() as usize;
        permuted.swap(i, j.min(i));
    }

    for (from, to) in ids.iter().zip(permuted.iter()) {
        for idx in 0..y.len() {
            if y[idx] == *from {
                y2[idx] = *to;
            }
        }
    }
    Ok(y2)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn label_are_sequential_matches_python_set_rule() {
        assert!(label_are_sequential(&[0, 1, 2, 2]));
        assert!(label_are_sequential(&[0]));
        assert!(!label_are_sequential(&[0, 1, 3]));
    }

    #[test]
    fn label_overlap_counts_joint_labels() {
        let x = [0, 1, 1, 2, 2, 2];
        let y = [0, 1, 2, 2, 0, 2];
        let (overlap, shape) = label_overlap(&x, &y, true).unwrap();
        assert_eq!(shape, [3, 3]);
        assert_eq!(
            overlap,
            vec![
                1, 0, 0, //
                0, 1, 1, //
                1, 0, 2, //
            ]
        );
    }

    #[test]
    fn overlap_criteria_match_expected_ratios() {
        let overlap = [
            0, 0, 0, //
            0, 2, 1, //
            0, 1, 3, //
        ];
        let iou = intersection_over_union(&overlap, [3, 3]).unwrap();
        assert!((iou[1 * 3 + 1] - 0.5).abs() < 1e-6);
        assert!((iou[2 * 3 + 2] - 0.6).abs() < 1e-6);
        let iot = intersection_over_true(&overlap, [3, 3]).unwrap();
        assert!((iot[1 * 3 + 1] - 2.0 / 3.0).abs() < 1e-6);
        let iop = intersection_over_pred(&overlap, [3, 3]).unwrap();
        assert!((iop[2 * 3 + 2] - 3.0 / 4.0).abs() < 1e-6);
    }

    #[test]
    fn scalar_metrics_return_zero_without_true_positives() {
        assert_eq!(precision(0, 3, 4), 0.0);
        assert_eq!(recall(0, 3, 4), 0.0);
        assert_eq!(accuracy(0, 3, 4), 0.0);
        assert_eq!(f1(0, 3, 4), 0.0);
        assert!((precision(2, 1, 3) - 2.0 / 3.0).abs() < 1e-6);
        assert!((recall(2, 1, 3) - 2.0 / 5.0).abs() < 1e-6);
        assert!((accuracy(2, 1, 3) - 2.0 / 6.0).abs() < 1e-6);
        assert!((f1(2, 1, 3) - 4.0 / 8.0).abs() < 1e-6);
    }

    #[test]
    fn relabel_sequential_matches_skimage_example() {
        let relabeled = relabel_sequential(&[1, 1, 5, 5, 8, 99, 42], 1).unwrap();
        assert_eq!(relabeled.relabeled, vec![1, 1, 2, 2, 3, 5, 4]);
        assert_eq!(relabeled.forward_map[1], 1);
        assert_eq!(relabeled.forward_map[5], 2);
        assert_eq!(relabeled.forward_map[8], 3);
        assert_eq!(relabeled.forward_map[42], 4);
        assert_eq!(relabeled.forward_map[99], 5);
        assert_eq!(relabeled.inverse_map, vec![0, 1, 5, 8, 42, 99]);

        let relabeled_offset = relabel_sequential(&[1, 1, 5, 5, 8, 99, 42], 5).unwrap();
        assert_eq!(relabeled_offset.relabeled, vec![5, 5, 6, 6, 7, 9, 8]);
    }

    #[test]
    fn matching_returns_python_example_counts_for_shifted_object() {
        let mut y_true = vec![0u32; 25];
        y_true[1 * 5 + 1] = 1;
        y_true[1 * 5 + 2] = 1;
        y_true[2 * 5 + 1] = 1;
        y_true[2 * 5 + 2] = 1;
        let mut y_pred = vec![0u32; 25];
        y_pred[3 * 5 + 1] = 1;
        y_pred[3 * 5 + 2] = 1;
        y_pred[4 * 5 + 1] = 1;
        y_pred[4 * 5 + 2] = 1;
        let stats = matching(&y_true, &y_pred, Some(0.5), MatchingCriterion::Iou, false).unwrap();
        assert_eq!(stats.fp, 1);
        assert_eq!(stats.tp, 0);
        assert_eq!(stats.fn_, 1);
        assert_eq!(stats.precision, 0.0);
        assert_eq!(stats.recall, 0.0);
        assert_eq!(stats.accuracy, 0.0);
        assert_eq!(stats.f1, 0.0);
        assert_eq!(stats.n_true, 1);
        assert_eq!(stats.n_pred, 1);
        assert_eq!(stats.mean_true_score, 0.0);
        assert_eq!(stats.mean_matched_score, 0.0);
        assert_eq!(stats.panoptic_quality, 0.0);
        assert_eq!(stats.matched_pairs, None);
    }

    #[test]
    fn matching_reports_pairs_scores_and_true_positive_indices() {
        let y_true = [0, 5, 5, 0, 9, 9];
        let y_pred = [0, 2, 2, 0, 7, 7];
        let stats = matching(&y_true, &y_pred, Some(0.5), MatchingCriterion::Iou, true).unwrap();
        assert_eq!(stats.tp, 2);
        assert_eq!(stats.fp, 0);
        assert_eq!(stats.fn_, 0);
        assert_eq!(stats.precision, 1.0);
        assert_eq!(stats.recall, 1.0);
        assert_eq!(stats.accuracy, 1.0);
        assert_eq!(stats.f1, 1.0);
        assert_eq!(stats.mean_true_score, 1.0);
        assert_eq!(stats.mean_matched_score, 1.0);
        assert_eq!(stats.panoptic_quality, 1.0);
        assert_eq!(stats.matched_pairs, Some(vec![(5, 2), (9, 7)]));
        assert_eq!(stats.matched_scores, Some(vec![1.0, 1.0]));
        assert_eq!(stats.matched_tps, Some(vec![0, 1]));
    }

    #[test]
    fn matching_uses_requested_overlap_criterion() {
        let y_true = [1, 1, 1, 0];
        let y_pred = [2, 2, 0, 0];
        let iou = matching(&y_true, &y_pred, Some(0.8), MatchingCriterion::Iou, false).unwrap();
        let iot = matching(&y_true, &y_pred, Some(0.8), MatchingCriterion::Iot, false).unwrap();
        let iop = matching(&y_true, &y_pred, Some(0.8), MatchingCriterion::Iop, false).unwrap();
        assert_eq!(iou.tp, 0);
        assert_eq!(iot.tp, 0);
        assert_eq!(iop.tp, 1);
        assert_eq!(iop.mean_true_score, 1.0);

        let y_true = [1, 1, 0, 0];
        let y_pred = [2, 2, 2, 0];
        let iot = matching(&y_true, &y_pred, Some(0.8), MatchingCriterion::Iot, false).unwrap();
        let iop = matching(&y_true, &y_pred, Some(0.8), MatchingCriterion::Iop, false).unwrap();
        assert_eq!(iot.tp, 1);
        assert_eq!(iop.tp, 0);
        assert_eq!(iot.mean_true_score, 1.0);
    }

    #[test]
    fn matching_dataset_aggregates_counts_globally() {
        let y_true0 = [0, 1, 1, 0];
        let y_pred0 = [0, 2, 2, 0];
        let y_true1 = [0, 1, 1, 0];
        let y_pred1 = [2, 2, 0, 0];
        let stats = matching_dataset(
            &[&y_true0, &y_true1],
            &[&y_pred0, &y_pred1],
            &[0.75],
            MatchingCriterion::Iou,
            false,
            false,
            false,
        )
        .unwrap();
        assert_eq!(stats.len(), 1);
        let stats = &stats[0];
        assert_eq!(stats.tp, 1);
        assert_eq!(stats.fp, 1);
        assert_eq!(stats.fn_, 1);
        assert_eq!(stats.n_true, 2);
        assert_eq!(stats.n_pred, 2);
        assert!((stats.precision - 0.5).abs() < 1e-6);
        assert!((stats.recall - 0.5).abs() < 1e-6);
        assert!((stats.accuracy - 1.0 / 3.0).abs() < 1e-6);
        assert!((stats.f1 - 0.5).abs() < 1e-6);
        assert!((stats.mean_true_score - 0.5).abs() < 1e-6);
        assert!((stats.mean_matched_score - 1.0).abs() < 1e-6);
        assert!((stats.panoptic_quality - 0.5).abs() < 1e-6);
        assert!(!stats.by_image);
    }

    #[test]
    fn matching_dataset_lazy_averages_by_image_and_handles_multiple_thresholds() {
        let y_true0 = [0, 1, 1, 0];
        let y_pred0 = [0, 2, 2, 0];
        let y_true1 = [0, 1, 1, 0];
        let y_pred1 = [2, 2, 0, 0];
        let pairs = [(&y_true0[..], &y_pred0[..]), (&y_true1[..], &y_pred1[..])];
        let stats = matching_dataset_lazy(
            &pairs,
            &[0.3, 0.75],
            MatchingCriterion::Iou,
            true,
            false,
            false,
        )
        .unwrap();
        assert_eq!(stats.len(), 2);
        assert_eq!(stats[0].thresh, 0.3);
        assert_eq!(stats[1].thresh, 0.75);
        assert_eq!(stats[0].tp, 2);
        assert_eq!(stats[1].tp, 1);
        assert!((stats[0].precision - 1.0).abs() < 1e-6);
        assert!((stats[1].precision - 0.5).abs() < 1e-6);
        assert!((stats[1].mean_true_score - 0.5).abs() < 1e-6);
        assert!(stats[1].by_image);
    }

    #[test]
    fn matching_dataset_rejects_mismatched_dataset_lengths() {
        let y_true = [0, 1];
        let y_pred = [0, 1];
        assert_eq!(
            matching_dataset(
                &[&y_true, &y_true],
                &[&y_pred],
                &[0.5],
                MatchingCriterion::Iou,
                false,
                false,
                false,
            )
            .unwrap_err(),
            MatchingError::DatasetLengthMismatch
        );
    }

    #[test]
    fn group_matching_labels_keeps_matched_ids_and_assigns_new_ids() {
        let y0 = [
            1, 1, 0, 0, //
            1, 1, 0, 2, //
            0, 0, 0, 2, //
            0, 0, 0, 2, //
        ];
        let y1 = [
            5, 5, 0, 0, //
            5, 5, 0, 9, //
            0, 0, 0, 9, //
            7, 7, 0, 9, //
        ];
        let grouped = group_matching_labels(&[&y0, &y1], 0.5, MatchingCriterion::Iou).unwrap();
        assert_eq!(grouped[0], y0);
        assert_eq!(
            grouped[1],
            vec![
                1, 1, 0, 0, //
                1, 1, 0, 2, //
                0, 0, 0, 2, //
                3, 3, 0, 2, //
            ]
        );
    }

    #[test]
    fn group_matching_labels_chains_against_previous_grouped_frame() {
        let y0 = [1, 1, 0, 0];
        let y1 = [0, 2, 2, 0];
        let y2 = [0, 0, 4, 4];
        let grouped = group_matching_labels(&[&y0, &y1, &y2], 0.3, MatchingCriterion::Iou).unwrap();
        assert_eq!(grouped[0], vec![1, 1, 0, 0]);
        assert_eq!(grouped[1], vec![0, 1, 1, 0]);
        assert_eq!(grouped[2], vec![0, 0, 1, 1]);
    }

    #[test]
    fn group_matching_labels_rejects_invalid_sequences() {
        let y0 = [0, 1];
        let y1 = [0, 1, 1];
        assert_eq!(
            group_matching_labels(&[&y0], 0.5, MatchingCriterion::Iou).unwrap_err(),
            MatchingError::TooFewImages
        );
        assert_eq!(
            group_matching_labels(&[&y0, &y1], 0.5, MatchingCriterion::Iou).unwrap_err(),
            MatchingError::ShapeMismatch
        );
    }

    #[test]
    fn shuffle_labels_is_deterministic_and_preserves_regions() {
        let y = [
            0, 1, 1, 0, //
            2, 2, 0, 3, //
            0, 0, 3, 3, //
        ];
        let shuffled = _shuffle_labels(&y, 5).unwrap();
        assert_eq!(shuffled, _shuffle_labels(&y, 5).unwrap());
        assert_eq!(shuffled[0], 0);
        assert_eq!(shuffled[3], 0);
        assert_eq!(shuffled[1], shuffled[2]);
        assert_eq!(shuffled[4], shuffled[5]);
        assert_eq!(shuffled[7], shuffled[10]);
        assert_eq!(shuffled[7], shuffled[11]);

        let mut original_ids = y.iter().copied().filter(|v| *v != 0).collect::<Vec<_>>();
        original_ids.sort_unstable();
        original_ids.dedup();
        let mut shuffled_ids = shuffled
            .iter()
            .copied()
            .filter(|v| *v != 0)
            .collect::<Vec<_>>();
        shuffled_ids.sort_unstable();
        shuffled_ids.dedup();
        assert_eq!(shuffled_ids, original_ids);
    }

    #[test]
    fn shuffle_labels_handles_background_only_image() {
        assert_eq!(_shuffle_labels(&[0, 0, 0], 9).unwrap(), vec![0, 0, 0]);
    }

    #[test]
    fn validation_errors_match_boundaries() {
        assert_eq!(
            label_overlap(&[0, 2], &[0, 1], true).unwrap_err(),
            MatchingError::NonSequentialLabel
        );
        assert_eq!(
            label_overlap(&[0, 1], &[0], true).unwrap_err(),
            MatchingError::ShapeMismatch
        );
        assert_eq!(
            intersection_over_union(&[0, 1], [2, 2]).unwrap_err(),
            MatchingError::OverlapShapeMismatch
        );
        assert_eq!(
            relabel_sequential(&[0, 1], 0).unwrap_err(),
            MatchingError::InvalidOffset
        );
    }
}
