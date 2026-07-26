use std::collections::BTreeSet;
use std::time::{SystemTime, UNIX_EPOCH};

use crate::matching::{MatchingCriterion, MatchingError, matching};

#[derive(Debug, thiserror::Error, PartialEq)]
pub enum PlotError {
    #[error("x and y coordinates must have the same length")]
    CoordinateLengthMismatch,
    #[error("coordinate array shape does not match coordinate data")]
    CoordShapeMismatch,
    #[error("score image shape does not match score data")]
    ScoreShapeMismatch,
    #[error("polygon index is out of bounds")]
    PolygonIndexOutOfBounds,
    #[error("number of polygons and scores must match")]
    ScoresLengthMismatch,
    #[error("number of polygons and points must match")]
    PointsLengthMismatch,
    #[error("colormap must include background plus one color per polygon")]
    CmapTooSmall,
    #[error("show_dist requires points")]
    ShowDistRequiresPoints,
    #[error("label image shape does not match label data")]
    LabelShapeMismatch,
    #[error("image shape does not match image data")]
    ImageShapeMismatch,
    #[error("img should be 2 or 3 dimensional")]
    InvalidImageDimension,
    #[error("single-color colormap color must have length 3 or 4")]
    InvalidColorLength,
    #[error("label colormap does not contain requested label")]
    LabelCmapTooSmall,
    #[error(transparent)]
    Matching(#[from] MatchingError),
}

#[derive(Clone, Debug, PartialEq)]
pub enum PlotDrawCommand {
    Point {
        point: [f32; 2],
        markersize: f32,
        color: [f32; 3],
    },
    Line {
        points: Vec<[f32; 2]>,
        linewidth: f32,
        color: [f32; 3],
        dashed: bool,
    },
    LineCollection {
        lines: Vec<[[f32; 2]; 2]>,
        linewidth: f32,
        color: [f32; 3],
    },
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum PlotImage<'a> {
    Gray { data: &'a [f32], shape: [usize; 2] },
    Channels { data: &'a [f32], shape: [usize; 3] },
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum PlotRange {
    Scalar(f32),
    Range(f32, f32),
}

impl From<f32> for PlotRange {
    fn from(value: f32) -> Self {
        Self::Scalar(value)
    }
}

impl From<(f32, f32)> for PlotRange {
    fn from(value: (f32, f32)) -> Self {
        Self::Range(value.0, value.1)
    }
}

impl From<[f32; 2]> for PlotRange {
    fn from(value: [f32; 2]) -> Self {
        Self::Range(value[0], value[1])
    }
}

pub fn random_hls(
    n: usize,
    h0: impl Into<PlotRange>,
    l0: impl Into<PlotRange>,
    s0: impl Into<PlotRange>,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let h0 = h0.into();
    let l0 = l0.into();
    let s0 = s0.into();
    let seed = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_nanos() as u64)
        .unwrap_or(0);
    let mut state = seed ^ ((n as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15));
    let mut h = Vec::<f32>::with_capacity(n);
    let mut l = Vec::<f32>::with_capacity(n);
    let mut s = Vec::<f32>::with_capacity(n);

    for _ in 0..n {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        let rh = ((state >> 40) as f32) / ((1u64 << 24) as f32);
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        let rl = ((state >> 40) as f32) / ((1u64 << 24) as f32);
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        let rs = ((state >> 40) as f32) / ((1u64 << 24) as f32);

        let (ha, hb) = match h0 {
            PlotRange::Scalar(value) => (value, value),
            PlotRange::Range(a, b) => (a, b),
        };
        let (la, lb) = match l0 {
            PlotRange::Scalar(value) => (value, value),
            PlotRange::Range(a, b) => (a, b),
        };
        let (sa, sb) = match s0 {
            PlotRange::Scalar(value) => (value, value),
            PlotRange::Range(a, b) => (a, b),
        };
        h.push(ha + (hb - ha) * rh);
        l.push(la + (lb - la) * rl);
        s.push(sa + (sb - sa) * rs);
    }
    (h, l, s)
}

pub fn cmap_from_hls(h: &[f32], l: &[f32], s: &[f32]) -> Vec<[f32; 3]> {
    let n = h.len().min(l.len()).min(s.len());
    let mut cols = Vec::<[f32; 3]>::with_capacity(n);

    for i in 0..n {
        let mut hue = h[i] % 1.0;
        if hue < 0.0 {
            hue += 1.0;
        }
        let lum = l[i].clamp(0.0, 1.0);
        let sat = s[i].clamp(0.0, 1.0);

        let (r, g, b) = if sat == 0.0 {
            (lum, lum, lum)
        } else {
            let m2 = if lum <= 0.5 {
                lum * (1.0 + sat)
            } else {
                lum + sat - lum * sat
            };
            let m1 = 2.0 * lum - m2;
            let mut rgb = [0.0; 3];
            for (j, value) in [hue + 1.0 / 3.0, hue, hue - 1.0 / 3.0].iter().enumerate() {
                let mut value = *value;
                if value < 0.0 {
                    value += 1.0;
                }
                if value > 1.0 {
                    value -= 1.0;
                }
                rgb[j] = if 6.0 * value < 1.0 {
                    m1 + (m2 - m1) * value * 6.0
                } else if 2.0 * value < 1.0 {
                    m2
                } else if 3.0 * value < 2.0 {
                    m1 + (m2 - m1) * (2.0 / 3.0 - value) * 6.0
                } else {
                    m1
                };
            }
            (rgb[0], rgb[1], rgb[2])
        };
        cols.push([r, g, b]);
    }

    if let Some(first) = cols.first_mut() {
        *first = [0.0, 0.0, 0.0];
    }
    cols
}

pub fn random_label_cmap(
    n: usize,
    h: impl Into<PlotRange>,
    l: impl Into<PlotRange>,
    s: impl Into<PlotRange>,
    seed: Option<u64>,
) -> Vec<[f32; 3]> {
    let h = h.into();
    let l = l.into();
    let s = s.into();
    let mut state = seed.unwrap_or_else(|| {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|duration| duration.as_nanos() as u64)
            .unwrap_or(0)
    });
    let mut hues = Vec::<f32>::with_capacity(n);
    let mut lums = Vec::<f32>::with_capacity(n);
    let mut sats = Vec::<f32>::with_capacity(n);

    for _ in 0..n {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        let rh = ((state >> 40) as f32) / ((1u64 << 24) as f32);
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        let rl = ((state >> 40) as f32) / ((1u64 << 24) as f32);
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        let rs = ((state >> 40) as f32) / ((1u64 << 24) as f32);

        let (ha, hb) = match h {
            PlotRange::Scalar(value) => (value, value),
            PlotRange::Range(a, b) => (a, b),
        };
        let (la, lb) = match l {
            PlotRange::Scalar(value) => (value, value),
            PlotRange::Range(a, b) => (a, b),
        };
        let (sa, sb) = match s {
            PlotRange::Scalar(value) => (value, value),
            PlotRange::Range(a, b) => (a, b),
        };
        hues.push(ha + (hb - ha) * rh);
        lums.push(la + (lb - la) * rl);
        sats.push(sa + (sb - sa) * rs);
    }

    let mut cols = Vec::<[f32; 3]>::with_capacity(n);
    for i in 0..n {
        let mut hue = hues[i] % 1.0;
        if hue < 0.0 {
            hue += 1.0;
        }
        let lum = lums[i].clamp(0.0, 1.0);
        let sat = sats[i].clamp(0.0, 1.0);

        let (red, green, blue) = if sat == 0.0 {
            (lum, lum, lum)
        } else {
            let m2 = if lum <= 0.5 {
                lum * (1.0 + sat)
            } else {
                lum + sat - lum * sat
            };
            let m1 = 2.0 * lum - m2;
            let mut rgb = [0.0; 3];
            for (j, value) in [hue + 1.0 / 3.0, hue, hue - 1.0 / 3.0].iter().enumerate() {
                let mut value = *value;
                if value < 0.0 {
                    value += 1.0;
                }
                if value > 1.0 {
                    value -= 1.0;
                }
                rgb[j] = if 6.0 * value < 1.0 {
                    m1 + (m2 - m1) * value * 6.0
                } else if 2.0 * value < 1.0 {
                    m2
                } else if 3.0 * value < 2.0 {
                    m1 + (m2 - m1) * (2.0 / 3.0 - value) * 6.0
                } else {
                    m1
                };
            }
            (rgb[0], rgb[1], rgb[2])
        };
        cols.push([red, green, blue]);
    }

    if let Some(first) = cols.first_mut() {
        *first = [0.0, 0.0, 0.0];
    }
    cols
}

pub fn _single_color_integer_cmap(
    labels: &[u32],
    color: &[f32],
) -> Result<Vec<[f32; 4]>, PlotError> {
    if !(color.len() == 3 || color.len() == 4) {
        return Err(PlotError::InvalidColorLength);
    }
    let color = [
        color[0],
        color[1],
        color[2],
        if color.len() == 4 { color[3] } else { 1.0 },
    ];
    let mut rendered = Vec::<[f32; 4]>::with_capacity(labels.len());
    for label in labels {
        if *label > 0 {
            rendered.push(color);
        } else {
            rendered.push([0.0, 0.0, 0.0, color[3]]);
        }
    }
    Ok(rendered)
}

pub fn render_label(
    lbl: &[u32],
    lbl_shape: [usize; 2],
    img: Option<PlotImage<'_>>,
    cmap: Option<&[[f32; 4]]>,
    alpha: f32,
    alpha_boundary: Option<f32>,
    normalize_img: bool,
) -> Result<Vec<[f32; 4]>, PlotError> {
    let height = lbl_shape[0];
    let width = lbl_shape[1];
    if lbl.len() != height * width {
        return Err(PlotError::LabelShapeMismatch);
    }
    let alpha = alpha.clamp(0.0, 1.0);
    let alpha_boundary = alpha_boundary.unwrap_or(alpha);

    let mut im_img = vec![[0.0, 0.0, 0.0, 1.0]; lbl.len()];
    if let Some(img) = img {
        match img {
            PlotImage::Gray { data, shape } => {
                if shape != lbl_shape || data.len() != height * width {
                    return Err(PlotError::ImageShapeMismatch);
                }
                let mut min_value = f32::INFINITY;
                let mut max_value = f32::NEG_INFINITY;
                if normalize_img {
                    for value in data {
                        min_value = min_value.min(*value);
                        max_value = max_value.max(*value);
                    }
                }
                for i in 0..data.len() {
                    let value = if normalize_img && max_value > min_value {
                        (data[i] - min_value) / (max_value - min_value)
                    } else if normalize_img {
                        0.0
                    } else {
                        data[i]
                    }
                    .clamp(0.0, 1.0);
                    im_img[i] = [value, value, value, 1.0];
                }
            }
            PlotImage::Channels { data, shape } => {
                if shape[0] != height || shape[1] != width || shape[2] == 0 {
                    return Err(PlotError::ImageShapeMismatch);
                }
                let channels = shape[2];
                if data.len() != height * width * channels {
                    return Err(PlotError::ImageShapeMismatch);
                }
                let mut min_value = f32::INFINITY;
                let mut max_value = f32::NEG_INFINITY;
                if normalize_img {
                    for value in data {
                        min_value = min_value.min(*value);
                        max_value = max_value.max(*value);
                    }
                }
                for i in 0..height * width {
                    let mut pixel = [1.0, 1.0, 1.0, 1.0];
                    for c in 0..channels.min(4) {
                        let value = data[i * channels + c];
                        pixel[c] = if normalize_img && max_value > min_value {
                            (value - min_value) / (max_value - min_value)
                        } else if normalize_img {
                            0.0
                        } else {
                            value
                        }
                        .clamp(0.0, 1.0);
                    }
                    im_img[i] = pixel;
                }
            }
        }
    }

    let default_cmap;
    let colors = if let Some(cmap) = cmap {
        cmap
    } else {
        let max_label = lbl.iter().copied().max().unwrap_or(0) as usize;
        let rgb = random_label_cmap(max_label + 1, (0.0, 1.0), (0.4, 1.0), (0.2, 0.8), None);
        default_cmap = rgb
            .iter()
            .map(|color| [color[0], color[1], color[2], 1.0])
            .collect::<Vec<_>>();
        &default_cmap
    };

    let mut im = im_img.clone();
    let mut im_lbl = Vec::<[f32; 4]>::with_capacity(lbl.len());
    for label in lbl {
        let color = colors
            .get(*label as usize)
            .copied()
            .ok_or(PlotError::LabelCmapTooSmall)?;
        im_lbl.push(color);
    }

    for i in 0..lbl.len() {
        if lbl[i] > 0 {
            for c in 0..4 {
                im[i][c] = alpha * im_lbl[i][c] + (1.0 - alpha) * im_img[i][c];
            }
        }
    }
    for row in 0..height {
        for col in 0..width {
            let i = row * width + col;
            if lbl[i] == 0 {
                continue;
            }
            let mut boundary = false;
            for dr in [-1isize, 0, 1] {
                for dc in [-1isize, 0, 1] {
                    if dr == 0 && dc == 0 {
                        continue;
                    }
                    let rr = row as isize + dr;
                    let cc = col as isize + dc;
                    if rr < 0 || cc < 0 || rr >= height as isize || cc >= width as isize {
                        boundary = true;
                    } else if lbl[rr as usize * width + cc as usize] != lbl[i] {
                        boundary = true;
                    }
                }
            }
            if boundary {
                for c in 0..4 {
                    im[i][c] =
                        alpha_boundary * im_lbl[i][c] + (1.0 - alpha_boundary) * im_img[i][c];
                }
            }
        }
    }
    Ok(im)
}

pub fn render_label_pred(
    y_true: &[u32],
    y_pred: &[u32],
    shape: [usize; 2],
    img: Option<PlotImage<'_>>,
    tp_alpha: f32,
    fp_alpha: f32,
    fn_alpha: f32,
    thresh: f32,
    criterion: MatchingCriterion,
    normalize_img: bool,
) -> Result<Vec<[f32; 4]>, PlotError> {
    let height = shape[0];
    let width = shape[1];
    if y_true.len() != height * width || y_pred.len() != height * width {
        return Err(PlotError::LabelShapeMismatch);
    }
    let res = matching(y_true, y_pred, Some(thresh), criterion, true)?;
    let matched_pairs = res
        .matched_pairs
        .expect("report_matches=true returns matched pairs");
    let matched_scores = res
        .matched_scores
        .expect("report_matches=true returns matched scores");

    let mut all_true = BTreeSet::<u32>::new();
    let mut all_pred = BTreeSet::<u32>::new();
    for label in y_true {
        if *label != 0 {
            all_true.insert(*label);
        }
    }
    for label in y_pred {
        if *label != 0 {
            all_pred.insert(*label);
        }
    }

    let mut tp_true = BTreeSet::<u32>::new();
    let mut tp_pred = BTreeSet::<u32>::new();
    for (pair, score) in matched_pairs.iter().zip(&matched_scores) {
        if *score >= thresh {
            tp_true.insert(pair.0);
            tp_pred.insert(pair.1);
        }
    }
    let fn_labels = all_true
        .difference(&tp_true)
        .copied()
        .collect::<BTreeSet<_>>();
    let fp_labels = all_pred
        .difference(&tp_pred)
        .copied()
        .collect::<BTreeSet<_>>();
    debug_assert_eq!(res.tp as usize, tp_pred.len());
    debug_assert_eq!(res.fp as usize, fp_labels.len());
    debug_assert_eq!(res.fn_ as usize, fn_labels.len());

    let n0 = y_pred.iter().copied().max().unwrap_or(0) as usize + 1;
    let (h_tp, l_tp, s_tp) = random_hls(n0, (0.25, 0.35), (0.4, 0.6), (0.5, 0.7));
    let cmap_tp = cmap_from_hls(&h_tp, &l_tp, &s_tp)
        .iter()
        .map(|color| [color[0], color[1], color[2], 1.0])
        .collect::<Vec<_>>();
    let (h_fp, l_fp, s_fp) = random_hls(n0, (0.0, 0.1), (0.4, 0.6), (0.5, 0.7));
    let cmap_fp = cmap_from_hls(&h_fp, &l_fp, &s_fp)
        .iter()
        .map(|color| [color[0], color[1], color[2], 1.0])
        .collect::<Vec<_>>();
    let n_true = y_true.iter().copied().max().unwrap_or(0) as usize + 1;
    let (h_fn, l_fn, s_fn) = random_hls(n_true, (0.6, 0.7), (0.4, 0.6), (0.5, 0.7));
    let cmap_fn = cmap_from_hls(&h_fn, &l_fn, &s_fn)
        .iter()
        .map(|color| [color[0], color[1], color[2], 1.0])
        .collect::<Vec<_>>();

    let mut im_img = vec![[0.0, 0.0, 0.0, 1.0]; y_true.len()];
    if let Some(img) = img {
        match img {
            PlotImage::Gray {
                data,
                shape: img_shape,
            } => {
                if img_shape != shape || data.len() != height * width {
                    return Err(PlotError::ImageShapeMismatch);
                }
                let mut min_value = f32::INFINITY;
                let mut max_value = f32::NEG_INFINITY;
                if normalize_img {
                    for value in data {
                        min_value = min_value.min(*value);
                        max_value = max_value.max(*value);
                    }
                }
                for i in 0..data.len() {
                    let value = if normalize_img && max_value > min_value {
                        (data[i] - min_value) / (max_value - min_value)
                    } else if normalize_img {
                        0.0
                    } else {
                        data[i]
                    }
                    .clamp(0.0, 1.0);
                    im_img[i] = [value, value, value, 1.0];
                }
            }
            PlotImage::Channels {
                data,
                shape: img_shape,
            } => {
                if img_shape[0] != height || img_shape[1] != width || img_shape[2] == 0 {
                    return Err(PlotError::ImageShapeMismatch);
                }
                let channels = img_shape[2];
                if data.len() != height * width * channels {
                    return Err(PlotError::ImageShapeMismatch);
                }
                let mut min_value = f32::INFINITY;
                let mut max_value = f32::NEG_INFINITY;
                if normalize_img {
                    for value in data {
                        min_value = min_value.min(*value);
                        max_value = max_value.max(*value);
                    }
                }
                for i in 0..height * width {
                    let mut pixel = [1.0, 1.0, 1.0, 1.0];
                    for c in 0..channels.min(4) {
                        let value = data[i * channels + c];
                        pixel[c] = if normalize_img && max_value > min_value {
                            (value - min_value) / (max_value - min_value)
                        } else if normalize_img {
                            0.0
                        } else {
                            value
                        }
                        .clamp(0.0, 1.0);
                    }
                    im_img[i] = pixel;
                }
            }
        }
    }

    let mut im = im_img.clone();
    for i in 0..y_true.len() {
        if tp_pred.contains(&y_pred[i]) {
            let color = cmap_tp[y_pred[i] as usize];
            for c in 0..4 {
                im[i][c] = tp_alpha * color[c] + (1.0 - tp_alpha) * im_img[i][c];
            }
        }
        if fp_labels.contains(&y_pred[i]) {
            let color = cmap_fp[y_pred[i] as usize];
            for c in 0..4 {
                im[i][c] = fp_alpha * color[c] + (1.0 - fp_alpha) * im_img[i][c];
            }
        }
        if fn_labels.contains(&y_true[i]) {
            let color = cmap_fn[y_true[i] as usize];
            for c in 0..4 {
                im[i][c] = fn_alpha * color[c] + (1.0 - fn_alpha) * im_img[i][c];
            }
        }
    }
    Ok(im)
}

pub fn _plot_polygon(
    x: &[f32],
    y: &[f32],
    score: f32,
    color: [f32; 3],
) -> Result<PlotDrawCommand, PlotError> {
    if x.len() != y.len() {
        return Err(PlotError::CoordinateLengthMismatch);
    }
    let mut points = Vec::<[f32; 2]>::with_capacity(x.len() + usize::from(!x.is_empty()));
    for i in 0..x.len() {
        points.push([x[i], y[i]]);
    }
    if let Some(first) = points.first().copied() {
        points.push(first);
    }
    Ok(PlotDrawCommand::Line {
        points,
        linewidth: score,
        color,
        dashed: true,
    })
}

pub fn draw_polygons(
    coord: &[f32],
    coord_shape: [usize; 4],
    score: &[f32],
    score_shape: [usize; 2],
    poly_idx: &[[usize; 2]],
    grid: [usize; 2],
    cmap: Option<&[[f32; 3]]>,
    show_dist: bool,
) -> Result<Vec<PlotDrawCommand>, PlotError> {
    let height = coord_shape[0];
    let width = coord_shape[1];
    let n_axes = coord_shape[2];
    let n_vertices = coord_shape[3];
    if n_axes != 2 || coord.len() != height * width * n_axes * n_vertices {
        return Err(PlotError::CoordShapeMismatch);
    }
    if score_shape != [height, width] || score.len() != height * width {
        return Err(PlotError::ScoreShapeMismatch);
    }

    let mut polygons = Vec::<Vec<[f32; 2]>>::with_capacity(poly_idx.len());
    let mut points = Vec::<[f32; 2]>::with_capacity(poly_idx.len());
    let mut scores = Vec::<f32>::with_capacity(poly_idx.len());
    for index in poly_idx {
        let row = index[0];
        let col = index[1];
        if row >= height || col >= width {
            return Err(PlotError::PolygonIndexOutOfBounds);
        }
        let mut polygon = Vec::<[f32; 2]>::with_capacity(n_vertices);
        for vertex in 0..n_vertices {
            let y = coord[(((row * width + col) * n_axes) * n_vertices) + vertex];
            let x = coord[(((row * width + col) * n_axes + 1) * n_vertices) + vertex];
            polygon.push([y, x]);
        }
        polygons.push(polygon);
        points.push([row as f32 * grid[0] as f32, col as f32 * grid[1] as f32]);
        scores.push(score[row * width + col]);
    }

    _draw_polygons(
        &polygons,
        Some(&points),
        Some(&scores),
        None,
        cmap,
        show_dist,
    )
}

pub fn _draw_polygons(
    polygons: &[Vec<[f32; 2]>],
    points: Option<&[[f32; 2]]>,
    scores: Option<&[f32]>,
    _grid: Option<[usize; 2]>,
    cmap: Option<&[[f32; 3]]>,
    show_dist: bool,
) -> Result<Vec<PlotDrawCommand>, PlotError> {
    let mut default_points = Vec::<[f32; 2]>::new();
    let points = if let Some(points) = points {
        points
    } else {
        default_points.resize(polygons.len(), [f32::NAN, f32::NAN]);
        &default_points
    };

    let mut default_scores = Vec::<f32>::new();
    let scores = if let Some(scores) = scores {
        scores
    } else {
        default_scores.resize(polygons.len(), 1.0);
        &default_scores
    };

    let default_cmap;
    let colors = if let Some(cmap) = cmap {
        cmap
    } else {
        default_cmap =
            random_label_cmap(polygons.len() + 1, (0.0, 1.0), (0.4, 1.0), (0.2, 0.8), None);
        &default_cmap
    };

    if polygons.len() != scores.len() {
        return Err(PlotError::ScoresLengthMismatch);
    }
    if polygons.len() != points.len() {
        return Err(PlotError::PointsLengthMismatch);
    }
    if colors.len().saturating_sub(1) < polygons.len() {
        return Err(PlotError::CmapTooSmall);
    }
    if show_dist
        && points
            .iter()
            .any(|point| point[0].is_nan() || point[1].is_nan())
    {
        return Err(PlotError::ShowDistRequiresPoints);
    }

    let mut commands = Vec::<PlotDrawCommand>::new();
    for i in 0..polygons.len() {
        let point = points[i];
        let poly = &polygons[i];
        let score = scores[i];
        let color = colors[i + 1];

        if !point[0].is_nan() && !point[1].is_nan() {
            commands.push(PlotDrawCommand::Point {
                point: [point[1], point[0]],
                markersize: 8.0 * score,
                color,
            });
        }

        if show_dist {
            let mut lines = Vec::<[[f32; 2]; 2]>::with_capacity(poly.len());
            for vertex in poly {
                lines.push([[vertex[1], vertex[0]], [point[1], point[0]]]);
            }
            commands.push(PlotDrawCommand::LineCollection {
                lines,
                linewidth: 0.4,
                color,
            });
        }

        let mut x = Vec::<f32>::with_capacity(poly.len());
        let mut y = Vec::<f32>::with_capacity(poly.len());
        for vertex in poly {
            x.push(vertex[1]);
            y.push(vertex[0]);
        }
        commands.push(_plot_polygon(&x, &y, 3.0 * score, color)?);
    }
    Ok(commands)
}

pub fn match_labels(y0: &[u32], y: &[u32]) -> Result<Vec<u32>, MatchingError> {
    let res = matching(y0, y, Some(0.1), MatchingCriterion::Iou, true)?;
    let matched_pairs = res
        .matched_pairs
        .expect("report_matches=true returns matched pairs");
    if matched_pairs.is_empty() {
        return Ok(y.to_vec());
    }

    let mut ind_matched0 = Vec::<u32>::with_capacity(matched_pairs.len());
    let mut ind_matched = Vec::<u32>::with_capacity(matched_pairs.len());
    for (label0, label) in &matched_pairs {
        ind_matched0.push(*label0);
        ind_matched.push(*label);
    }

    let mut all_y = BTreeSet::<u32>::new();
    for label in y {
        if *label != 0 {
            all_y.insert(*label);
        }
    }
    let ind_matched_set = ind_matched.iter().copied().collect::<BTreeSet<_>>();
    let ind_unmatched = all_y
        .difference(&ind_matched_set)
        .copied()
        .collect::<Vec<_>>();

    let max_matched0 = ind_matched0.iter().copied().max().unwrap_or(0);
    let matched0_set = ind_matched0.iter().copied().collect::<BTreeSet<_>>();
    let mut leftover_labels = (1..max_matched0)
        .filter(|label| !matched0_set.contains(label))
        .collect::<BTreeSet<_>>();
    if ind_unmatched.len() > leftover_labels.len() {
        let n_extra = ind_unmatched.len() - leftover_labels.len();
        for i in 0..n_extra {
            leftover_labels.insert(max_matched0 + 1 + i as u32);
        }
    }

    let mut u = vec![0u32; y.len()];
    for (label0, label) in matched_pairs {
        for (i, value) in y.iter().enumerate() {
            if *value == label {
                u[i] = label0;
            }
        }
    }
    for (label, label2) in ind_unmatched.iter().zip(leftover_labels.iter()) {
        for (i, value) in y.iter().enumerate() {
            if value == label {
                u[i] = *label2;
            }
        }
    }
    Ok(u)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plot_polygon_closes_path_and_uses_score_as_linewidth() {
        let command = _plot_polygon(&[1.0, 2.0], &[3.0, 4.0], 0.75, [0.1, 0.2, 0.3]).unwrap();
        assert_eq!(
            command,
            PlotDrawCommand::Line {
                points: vec![[1.0, 3.0], [2.0, 4.0], [1.0, 3.0]],
                linewidth: 0.75,
                color: [0.1, 0.2, 0.3],
                dashed: true,
            }
        );
        assert_eq!(
            _plot_polygon(&[1.0], &[2.0, 3.0], 1.0, [1.0, 0.0, 0.0]).unwrap_err(),
            PlotError::CoordinateLengthMismatch
        );
    }

    #[test]
    fn draw_polygons_selects_dense_polygons_points_and_scores() {
        let coord = [
            0.0, 1.0, 2.0, 10.0, 11.0, 12.0, //
            3.0, 4.0, 5.0, 13.0, 14.0, 15.0, //
            6.0, 7.0, 8.0, 16.0, 17.0, 18.0, //
            9.0, 10.0, 11.0, 19.0, 20.0, 21.0, //
        ];
        let score = [0.1, 0.2, 0.3, 0.4];
        let cmap = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];

        let commands = draw_polygons(
            &coord,
            [2, 2, 2, 3],
            &score,
            [2, 2],
            &[[0, 1], [1, 0]],
            [2, 3],
            Some(&cmap),
            true,
        )
        .unwrap();

        assert_eq!(commands.len(), 6);
        assert_eq!(
            commands[0],
            PlotDrawCommand::Point {
                point: [3.0, 0.0],
                markersize: 1.6,
                color: [1.0, 0.0, 0.0],
            }
        );
        assert_eq!(
            commands[1],
            PlotDrawCommand::LineCollection {
                lines: vec![
                    [[13.0, 3.0], [3.0, 0.0]],
                    [[14.0, 4.0], [3.0, 0.0]],
                    [[15.0, 5.0], [3.0, 0.0]],
                ],
                linewidth: 0.4,
                color: [1.0, 0.0, 0.0],
            }
        );
        assert_eq!(
            commands[2],
            PlotDrawCommand::Line {
                points: vec![[13.0, 3.0], [14.0, 4.0], [15.0, 5.0], [13.0, 3.0]],
                linewidth: 0.6,
                color: [1.0, 0.0, 0.0],
                dashed: true,
            }
        );
    }

    #[test]
    fn draw_polygons_validates_shapes_and_indices() {
        let coord = vec![0.0; 2 * 2 * 2 * 3];
        let score = vec![1.0; 4];
        assert_eq!(
            draw_polygons(
                &coord,
                [2, 2, 1, 3],
                &score,
                [2, 2],
                &[],
                [1, 1],
                None,
                false
            )
            .unwrap_err(),
            PlotError::CoordShapeMismatch
        );
        assert_eq!(
            draw_polygons(
                &coord,
                [2, 2, 2, 3],
                &score,
                [2, 2],
                &[[2, 0]],
                [1, 1],
                None,
                false
            )
            .unwrap_err(),
            PlotError::PolygonIndexOutOfBounds
        );
    }

    #[test]
    fn draw_polygons_core_defaults_scores_and_requires_points_for_distances() {
        let polygons = vec![vec![[1.0, 2.0], [3.0, 4.0]]];
        let cmap = [[0.0, 0.0, 0.0], [0.2, 0.3, 0.4]];

        let commands =
            _draw_polygons(&polygons, None, None, Some([2, 2]), Some(&cmap), false).unwrap();
        assert_eq!(commands.len(), 1);
        assert_eq!(
            commands[0],
            PlotDrawCommand::Line {
                points: vec![[2.0, 1.0], [4.0, 3.0], [2.0, 1.0]],
                linewidth: 3.0,
                color: [0.2, 0.3, 0.4],
                dashed: true,
            }
        );
        assert_eq!(
            _draw_polygons(&polygons, None, None, None, Some(&cmap), true).unwrap_err(),
            PlotError::ShowDistRequiresPoints
        );
        assert_eq!(
            _draw_polygons(
                &polygons,
                Some(&[[0.0, 0.0]]),
                Some(&[]),
                None,
                Some(&cmap),
                false
            )
            .unwrap_err(),
            PlotError::ScoresLengthMismatch
        );
    }

    #[test]
    fn cmap_from_hls_converts_primary_hues_and_zeros_background() {
        let h = [0.0, 0.0, 1.0 / 3.0, 2.0 / 3.0];
        let l = [0.5, 0.5, 0.5, 0.5];
        let s = [1.0, 1.0, 1.0, 1.0];
        let cmap = cmap_from_hls(&h, &l, &s);

        assert_eq!(cmap[0], [0.0, 0.0, 0.0]);
        assert!((cmap[1][0] - 1.0).abs() < 1.0e-6);
        assert!(cmap[1][1].abs() < 1.0e-6);
        assert!(cmap[1][2].abs() < 1.0e-6);
        assert!(cmap[2][0].abs() < 1.0e-6);
        assert!((cmap[2][1] - 1.0).abs() < 1.0e-6);
        assert!(cmap[2][2].abs() < 1.0e-6);
        assert!(cmap[3][0].abs() < 1.0e-6);
        assert!(cmap[3][1].abs() < 1.0e-6);
        assert!((cmap[3][2] - 1.0).abs() < 1.0e-6);
    }

    #[test]
    fn random_hls_respects_scalar_and_range_inputs() {
        let (h, l, s) = random_hls(16, 0.33, (0.8, 1.0), [0.5, 0.8]);

        assert_eq!(h.len(), 16);
        assert_eq!(l.len(), 16);
        assert_eq!(s.len(), 16);
        assert!(h.iter().all(|value| (*value - 0.33).abs() < f32::EPSILON));
        assert!(l.iter().all(|value| *value >= 0.8 && *value <= 1.0));
        assert!(s.iter().all(|value| *value >= 0.5 && *value <= 0.8));
    }

    #[test]
    fn random_label_cmap_is_seeded_and_keeps_label_zero_black() {
        let a = random_label_cmap(8, (0.0, 1.0), (0.4, 1.0), (0.2, 0.8), Some(17));
        let b = random_label_cmap(8, (0.0, 1.0), (0.4, 1.0), (0.2, 0.8), Some(17));
        let c = random_label_cmap(8, (0.0, 1.0), (0.4, 1.0), (0.2, 0.8), Some(18));

        assert_eq!(a, b);
        assert_ne!(a, c);
        assert_eq!(a[0], [0.0, 0.0, 0.0]);
        assert!(
            a.iter()
                .flatten()
                .all(|value| *value >= 0.0 && *value <= 1.0)
        );
    }

    #[test]
    fn single_color_integer_cmap_colors_nonzero_labels_and_preserves_alpha() {
        assert_eq!(
            _single_color_integer_cmap(&[0, 1, 4], &[0.3, 0.4, 0.5]).unwrap(),
            vec![
                [0.0, 0.0, 0.0, 1.0],
                [0.3, 0.4, 0.5, 1.0],
                [0.3, 0.4, 0.5, 1.0]
            ]
        );
        assert_eq!(
            _single_color_integer_cmap(&[0, 1], &[0.3, 0.4, 0.5, 0.25]).unwrap(),
            vec![[0.0, 0.0, 0.0, 0.25], [0.3, 0.4, 0.5, 0.25]]
        );
        assert_eq!(
            _single_color_integer_cmap(&[1], &[0.3, 0.4]).unwrap_err(),
            PlotError::InvalidColorLength
        );
    }

    #[test]
    fn render_label_blends_labels_and_boundaries_like_plot_render() {
        let lbl = [0, 1, 1, 0];
        let img = [0.2, 0.2, 0.2, 0.2];
        let cmap = [[0.0, 0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 1.0]];
        let rendered = render_label(
            &lbl,
            [2, 2],
            Some(PlotImage::Gray {
                data: &img,
                shape: [2, 2],
            }),
            Some(&cmap),
            0.5,
            Some(1.0),
            false,
        )
        .unwrap();

        assert_eq!(rendered[0], [0.2, 0.2, 0.2, 1.0]);
        assert_eq!(rendered[1], [1.0, 0.0, 0.0, 1.0]);
        assert_eq!(rendered[2], [1.0, 0.0, 0.0, 1.0]);
        assert_eq!(rendered[3], [0.2, 0.2, 0.2, 1.0]);
        assert_eq!(
            render_label(&lbl[..3], [2, 2], None, Some(&cmap), 0.5, None, false).unwrap_err(),
            PlotError::LabelShapeMismatch
        );
    }

    #[test]
    fn render_label_pred_colors_true_positive_false_positive_and_false_negative_regions() {
        let y_true = [
            1, 1, 0, //
            0, 2, 2,
        ];
        let y_pred = [
            1, 1, 3, //
            3, 0, 0,
        ];
        let rendered = render_label_pred(
            &y_true,
            &y_pred,
            [2, 3],
            None,
            1.0,
            1.0,
            1.0,
            0.5,
            MatchingCriterion::Iou,
            false,
        )
        .unwrap();

        assert!(rendered[0][1] > rendered[0][0] && rendered[0][1] > rendered[0][2]);
        assert!(rendered[2][0] > rendered[2][1] && rendered[2][0] > rendered[2][2]);
        assert!(rendered[4][2] > rendered[4][0] && rendered[4][2] > rendered[4][1]);
    }

    #[test]
    fn match_labels_maps_predictions_to_reference_labels_and_assigns_unmatched_leftovers() {
        let y0 = [
            5, 5, 0, 0, //
            5, 5, 0, 0, //
            0, 0, 9, 9, //
            0, 0, 9, 9, //
        ];
        let y = [
            2, 2, 0, 7, //
            2, 2, 0, 0, //
            0, 0, 3, 3, //
            0, 0, 3, 3, //
        ];
        let matched = match_labels(&y0, &y).unwrap();
        assert_eq!(
            matched,
            vec![
                5, 5, 0, 1, //
                5, 5, 0, 0, //
                0, 0, 9, 9, //
                0, 0, 9, 9, //
            ]
        );
    }

    #[test]
    fn match_labels_returns_prediction_when_no_matching_found() {
        let y0 = [0, 0, 0, 0];
        let y = [0, 0, 2, 2];
        assert_eq!(match_labels(&y0, &y).unwrap(), y);
    }
}
