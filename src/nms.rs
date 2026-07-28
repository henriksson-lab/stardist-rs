use std::sync::OnceLock;

use crate::Rays;
use crate::geometry::{
    bounding_radius_inner_gravity, bounding_radius_inner_isotropic, bounding_radius_outer,
    bounding_radius_outer_gravity, bounding_radius_outer_isotropic, calculate_poly_offset_gravity,
    intersect_bbox, intersect_sphere_gravity, intersect_sphere_isotropic,
    overlap_render_polyhedron_precomputed, polyhedron_bbox, polyhedron_polyverts,
    polyhedron_volume, precompute_tetrahedron_planes, render_polyhedron,
};
use crate::utils::{_normalize_grid, GridError};

#[derive(Clone, Debug, PartialEq)]
pub struct NonMaximumSuppression2D {
    pub points: Vec<[f32; 2]>,
    pub prob: Vec<f32>,
    pub dist: Vec<f32>,
    pub n_rays: usize,
}

#[derive(Clone, Debug, PartialEq)]
pub struct NonMaximumSuppressionSparse2D {
    pub points: Vec<[f32; 2]>,
    pub prob: Vec<f32>,
    pub dist: Vec<f32>,
    pub n_rays: usize,
    pub indices: Vec<usize>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct NonMaximumSuppression3D {
    pub points: Vec<[f32; 3]>,
    pub prob: Vec<f32>,
    pub dist: Vec<f32>,
    pub n_rays: usize,
}

#[derive(Clone, Debug, PartialEq)]
pub struct NonMaximumSuppressionSparse3D {
    pub points: Vec<[f32; 3]>,
    pub prob: Vec<f32>,
    pub dist: Vec<f32>,
    pub n_rays: usize,
    pub indices: Vec<usize>,
}

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum NmsError {
    #[error("prob shape does not match provided shape")]
    ProbShapeMismatch,
    #[error("dist shape does not match prob shape and n_rays")]
    DistShapeMismatch,
    #[error("points must be shaped as n_polys by 2")]
    PointsShapeMismatch,
    #[error("points must be shaped as n_polys by 3")]
    Points3DShapeMismatch,
    #[error("scores must be shaped as n_polys")]
    ScoresShapeMismatch,
    #[error("need n_rays >= 3")]
    TooFewRays,
    #[error("rays count does not match dist shape")]
    RayCountMismatch,
    #[error(transparent)]
    Geometry(#[from] crate::geometry::GeometryError),
    #[error(transparent)]
    Grid(#[from] GridError),
}

pub fn _ind_prob_thresh(
    prob: &[f32],
    shape: &[usize],
    prob_thresh: f32,
    b: Option<&[[usize; 2]]>,
) -> Result<Vec<bool>, NmsError> {
    let len = shape.iter().product::<usize>();
    if prob.len() != len {
        return Err(NmsError::ProbShapeMismatch);
    }
    if let Some(b) = b {
        if b.len() != shape.len() {
            return Err(NmsError::ProbShapeMismatch);
        }
    }

    let mut ind_thresh = vec![false; prob.len()];
    for i in 0..prob.len() {
        ind_thresh[i] = prob[i] > prob_thresh;
    }
    if let Some(b) = b {
        let mut stride = vec![1usize; shape.len()];
        for i in (0..shape.len().saturating_sub(1)).rev() {
            stride[i] = stride[i + 1] * shape[i + 1];
        }
        for flat in 0..prob.len() {
            if !ind_thresh[flat] {
                continue;
            }
            let mut keep = true;
            for axis in 0..shape.len() {
                let coord = (flat / stride[axis]) % shape[axis];
                let low = b[axis][0];
                let high = shape[axis].saturating_sub(b[axis][1]);
                if coord < low || coord >= high {
                    keep = false;
                    break;
                }
            }
            ind_thresh[flat] = keep;
        }
    }
    Ok(ind_thresh)
}

pub fn _non_maximum_suppression_old(
    coord: &[f32],
    prob: &[f32],
    shape: [usize; 2],
    n_rays: usize,
    grid: [usize; 2],
    b: Option<[[usize; 2]; 2]>,
    nms_thresh: f32,
    prob_thresh: f32,
    _verbose: bool,
    _max_bbox_search: bool,
) -> Result<Vec<[usize; 2]>, NmsError> {
    if n_rays < 3 {
        return Err(NmsError::TooFewRays);
    }
    if prob.len() != shape[0] * shape[1] {
        return Err(NmsError::ProbShapeMismatch);
    }
    if coord.len() != shape[0] * shape[1] * 2 * n_rays {
        return Err(NmsError::DistShapeMismatch);
    }
    let _ = _normalize_grid::<2>(&grid)?;

    let mask = _ind_prob_thresh(prob, &shape, prob_thresh, b.as_ref().map(|b| b.as_slice()))?;
    let mut points = Vec::new();
    let mut polygons = Vec::<Vec<[f32; 2]>>::new();
    let mut scores = Vec::new();
    for y in 0..shape[0] {
        for x in 0..shape[1] {
            let flat = y * shape[1] + x;
            if mask[flat] {
                points.push([y, x]);
                scores.push(prob[flat]);
                let mut polygon = Vec::with_capacity(n_rays);
                for n in 0..n_rays {
                    let py = coord[((flat * 2) * n_rays) + n];
                    let px = coord[((flat * 2 + 1) * n_rays) + n];
                    polygon.push([px, py]);
                }
                polygons.push(polygon);
            }
        }
    }

    let mut ind: Vec<usize> = (0..scores.len()).collect();
    ind.sort_by(|&a, &b| {
        scores[b]
            .partial_cmp(&scores[a])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.cmp(&b))
    });

    let mut polygons_sorted = Vec::with_capacity(ind.len());
    for i in &ind {
        polygons_sorted.push(polygons[*i].clone());
    }

    let mut suppressed = vec![false; polygons_sorted.len()];
    let mut survivors_sorted = vec![false; polygons_sorted.len()];
    let mut bbox_x1 = vec![0.0; polygons_sorted.len()];
    let mut bbox_x2 = vec![0.0; polygons_sorted.len()];
    let mut bbox_y1 = vec![0.0; polygons_sorted.len()];
    let mut bbox_y2 = vec![0.0; polygons_sorted.len()];
    let mut areas = vec![0.0; polygons_sorted.len()];
    for i in 0..polygons_sorted.len() {
        for (j, p) in polygons_sorted[i].iter().enumerate() {
            if j == 0 {
                bbox_x1[i] = p[0];
                bbox_x2[i] = p[0];
                bbox_y1[i] = p[1];
                bbox_y2[i] = p[1];
            } else {
                bbox_x1[i] = bbox_x1[i].min(p[0]);
                bbox_x2[i] = bbox_x2[i].max(p[0]);
                bbox_y1[i] = bbox_y1[i].min(p[1]);
                bbox_y2[i] = bbox_y2[i].max(p[1]);
            }
        }
        areas[i] = area_from_path(&polygons_sorted[i]);
    }

    for i in 0..polygons_sorted.len() {
        if suppressed[i] {
            continue;
        }
        survivors_sorted[i] = true;
        for j in i + 1..polygons_sorted.len() {
            if suppressed[j] {
                continue;
            }
            if !bbox_intersect(
                bbox_x1[i], bbox_x2[i], bbox_y1[i], bbox_y2[i], bbox_x1[j], bbox_x2[j], bbox_y1[j],
                bbox_y2[j],
            ) {
                continue;
            }
            let area_inter = poly_intersection_area(&polygons_sorted[i], &polygons_sorted[j]);
            let overlap = area_inter / (areas[i] + 1.0e-10).min(areas[j] + 1.0e-10);
            if overlap > nms_thresh {
                suppressed[j] = true;
            }
        }
    }

    let mut survivors = vec![false; ind.len()];
    for sorted_i in 0..ind.len() {
        survivors[ind[sorted_i]] = survivors_sorted[sorted_i];
    }

    let mut retained_points = Vec::new();
    for i in 0..survivors.len() {
        if survivors[i] {
            retained_points.push(points[i]);
        }
    }
    Ok(retained_points)
}

pub fn non_maximum_suppression(
    dist: &[f32],
    prob: &[f32],
    shape: [usize; 2],
    n_rays: usize,
    grid: [usize; 2],
    b: Option<[[usize; 2]; 2]>,
    nms_thresh: f32,
    prob_thresh: f32,
    use_bbox: bool,
    use_kdtree: bool,
) -> Result<NonMaximumSuppression2D, NmsError> {
    if n_rays < 3 {
        return Err(NmsError::TooFewRays);
    }
    if prob.len() != shape[0] * shape[1] {
        return Err(NmsError::ProbShapeMismatch);
    }
    if dist.len() != prob.len() * n_rays {
        return Err(NmsError::DistShapeMismatch);
    }
    let grid = _normalize_grid::<2>(&grid)?;

    let mask = _ind_prob_thresh(prob, &shape, prob_thresh, b.as_ref().map(|b| b.as_slice()))?;
    let mut points = Vec::new();
    let mut disti = Vec::new();
    let mut scores = Vec::new();
    for y in 0..shape[0] {
        for x in 0..shape[1] {
            let flat = y * shape[1] + x;
            if mask[flat] {
                points.push([y as f32, x as f32]);
                disti.extend_from_slice(&dist[flat * n_rays..(flat + 1) * n_rays]);
                scores.push(prob[flat]);
            }
        }
    }

    let mut ind: Vec<usize> = (0..scores.len()).collect();
    ind.sort_by(|&a, &b| {
        scores[b]
            .partial_cmp(&scores[a])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.cmp(&b))
    });

    let mut points_sorted = Vec::with_capacity(ind.len());
    let mut dist_sorted = Vec::with_capacity(ind.len() * n_rays);
    let mut scores_sorted = Vec::with_capacity(ind.len());
    for i in ind {
        points_sorted.push([points[i][0] * grid[0] as f32, points[i][1] * grid[1] as f32]);
        dist_sorted.extend_from_slice(&disti[i * n_rays..(i + 1) * n_rays]);
        scores_sorted.push(scores[i]);
    }

    let inds = non_maximum_suppression_inds(
        &dist_sorted,
        &points_sorted,
        Some(&scores_sorted),
        n_rays,
        nms_thresh,
        use_bbox,
        use_kdtree,
    )?;

    let mut points_kept = Vec::new();
    let mut dist_kept = Vec::new();
    let mut scores_kept = Vec::new();
    for i in 0..inds.len() {
        if inds[i] {
            points_kept.push(points_sorted[i]);
            dist_kept.extend_from_slice(&dist_sorted[i * n_rays..(i + 1) * n_rays]);
            scores_kept.push(scores_sorted[i]);
        }
    }

    Ok(NonMaximumSuppression2D {
        points: points_kept,
        prob: scores_kept,
        dist: dist_kept,
        n_rays,
    })
}

pub fn non_maximum_suppression_sparse(
    dist: &[f32],
    prob: &[f32],
    points: &[[f32; 2]],
    n_rays: usize,
    _b: Option<[[usize; 2]; 2]>,
    nms_thresh: f32,
    use_bbox: bool,
    use_kdtree: bool,
) -> Result<NonMaximumSuppressionSparse2D, NmsError> {
    if n_rays < 3 {
        return Err(NmsError::TooFewRays);
    }
    if prob.len() != points.len() {
        return Err(NmsError::ProbShapeMismatch);
    }
    if dist.len() != points.len() * n_rays {
        return Err(NmsError::DistShapeMismatch);
    }

    let mut inds_original: Vec<usize> = (0..prob.len()).collect();
    let mut sorted: Vec<usize> = (0..prob.len()).collect();
    sorted.sort_by(|&a, &b| {
        prob[b]
            .partial_cmp(&prob[a])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.cmp(&b))
    });

    let mut probi = Vec::with_capacity(sorted.len());
    let mut disti = Vec::with_capacity(sorted.len() * n_rays);
    let mut pointsi = Vec::with_capacity(sorted.len());
    let mut sorted_original = Vec::with_capacity(sorted.len());
    for i in sorted {
        probi.push(prob[i]);
        disti.extend_from_slice(&dist[i * n_rays..(i + 1) * n_rays]);
        pointsi.push(points[i]);
        sorted_original.push(inds_original[i]);
    }
    inds_original = sorted_original;

    let inds = non_maximum_suppression_inds(
        &disti,
        &pointsi,
        Some(&probi),
        n_rays,
        nms_thresh,
        use_bbox,
        use_kdtree,
    )?;

    let mut points_kept = Vec::new();
    let mut prob_kept = Vec::new();
    let mut dist_kept = Vec::new();
    let mut indices_kept = Vec::new();
    for i in 0..inds.len() {
        if inds[i] {
            points_kept.push(pointsi[i]);
            prob_kept.push(probi[i]);
            dist_kept.extend_from_slice(&disti[i * n_rays..(i + 1) * n_rays]);
            indices_kept.push(inds_original[i]);
        }
    }

    Ok(NonMaximumSuppressionSparse2D {
        points: points_kept,
        prob: prob_kept,
        dist: dist_kept,
        n_rays,
        indices: indices_kept,
    })
}

pub fn non_maximum_suppression_3d(
    dist: &[f32],
    prob: &[f32],
    shape: [usize; 3],
    rays: &Rays,
    grid: [usize; 3],
    b: Option<[[usize; 2]; 3]>,
    nms_thresh: f32,
    prob_thresh: f32,
    use_bbox: bool,
    use_kdtree: bool,
    use_gravity: bool,
) -> Result<NonMaximumSuppression3D, NmsError> {
    if rays.vertices.len() < 3 {
        return Err(NmsError::TooFewRays);
    }
    if prob.len() != shape[0] * shape[1] * shape[2] {
        return Err(NmsError::ProbShapeMismatch);
    }
    if dist.len() != prob.len() * rays.vertices.len() {
        return Err(NmsError::DistShapeMismatch);
    }
    let grid = _normalize_grid::<3>(&grid)?;

    let ind_thresh = _ind_prob_thresh(prob, &shape, prob_thresh, b.as_ref().map(|b| b.as_slice()))?;
    let mut points = Vec::new();
    let mut probi = Vec::new();
    let mut disti = Vec::new();
    for z in 0..shape[0] {
        for y in 0..shape[1] {
            for x in 0..shape[2] {
                let flat = (z * shape[1] + y) * shape[2] + x;
                if ind_thresh[flat] {
                    points.push([z as f32, y as f32, x as f32]);
                    probi.push(prob[flat]);
                    disti.extend_from_slice(
                        &dist[flat * rays.vertices.len()..(flat + 1) * rays.vertices.len()],
                    );
                }
            }
        }
    }

    let mut sorted: Vec<usize> = (0..probi.len()).collect();
    sorted.sort_by(|&a, &b| {
        probi[b]
            .partial_cmp(&probi[a])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.cmp(&b))
    });

    let mut probi_sorted = Vec::with_capacity(sorted.len());
    let mut disti_sorted = Vec::with_capacity(sorted.len() * rays.vertices.len());
    let mut pointsi_sorted = Vec::with_capacity(sorted.len());
    for i in sorted {
        probi_sorted.push(probi[i]);
        disti_sorted
            .extend_from_slice(&disti[i * rays.vertices.len()..(i + 1) * rays.vertices.len()]);
        pointsi_sorted.push([
            points[i][0] * grid[0] as f32,
            points[i][1] * grid[1] as f32,
            points[i][2] * grid[2] as f32,
        ]);
    }

    let inds = non_maximum_suppression_3d_inds(
        &disti_sorted,
        &pointsi_sorted,
        rays,
        Some(&probi_sorted),
        nms_thresh,
        use_bbox,
        use_kdtree,
        use_gravity,
    )?;

    let mut points_kept = Vec::new();
    let mut prob_kept = Vec::new();
    let mut dist_kept = Vec::new();
    for i in 0..inds.len() {
        if inds[i] {
            points_kept.push(pointsi_sorted[i]);
            prob_kept.push(probi_sorted[i]);
            dist_kept.extend_from_slice(
                &disti_sorted[i * rays.vertices.len()..(i + 1) * rays.vertices.len()],
            );
        }
    }

    Ok(NonMaximumSuppression3D {
        points: points_kept,
        prob: prob_kept,
        dist: dist_kept,
        n_rays: rays.vertices.len(),
    })
}

pub fn non_maximum_suppression_3d_sparse(
    dist: &[f32],
    prob: &[f32],
    points: &[[f32; 3]],
    rays: &Rays,
    _b: Option<[[usize; 2]; 3]>,
    nms_thresh: f32,
    use_bbox: bool,
    use_kdtree: bool,
    use_gravity: bool,
) -> Result<NonMaximumSuppressionSparse3D, NmsError> {
    if rays.vertices.len() < 3 {
        return Err(NmsError::TooFewRays);
    }
    if prob.len() != points.len() {
        return Err(NmsError::ProbShapeMismatch);
    }
    if dist.len() != points.len() * rays.vertices.len() {
        return Err(NmsError::DistShapeMismatch);
    }

    let mut inds_original: Vec<usize> = (0..prob.len()).collect();
    let mut sorted: Vec<usize> = (0..prob.len()).collect();
    sorted.sort_by(|&a, &b| {
        prob[b]
            .partial_cmp(&prob[a])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.cmp(&b))
    });

    let mut probi = Vec::with_capacity(sorted.len());
    let mut disti = Vec::with_capacity(sorted.len() * rays.vertices.len());
    let mut pointsi = Vec::with_capacity(sorted.len());
    let mut sorted_original = Vec::with_capacity(sorted.len());
    for i in sorted {
        probi.push(prob[i]);
        disti.extend_from_slice(&dist[i * rays.vertices.len()..(i + 1) * rays.vertices.len()]);
        pointsi.push(points[i]);
        sorted_original.push(inds_original[i]);
    }
    inds_original = sorted_original;

    let inds = non_maximum_suppression_3d_inds(
        &disti,
        &pointsi,
        rays,
        Some(&probi),
        nms_thresh,
        use_bbox,
        use_kdtree,
        use_gravity,
    )?;

    let mut points_kept = Vec::new();
    let mut prob_kept = Vec::new();
    let mut dist_kept = Vec::new();
    let mut indices_kept = Vec::new();
    for i in 0..inds.len() {
        if inds[i] {
            points_kept.push(pointsi[i]);
            prob_kept.push(probi[i]);
            dist_kept
                .extend_from_slice(&disti[i * rays.vertices.len()..(i + 1) * rays.vertices.len()]);
            indices_kept.push(inds_original[i]);
        }
    }

    Ok(NonMaximumSuppressionSparse3D {
        points: points_kept,
        prob: prob_kept,
        dist: dist_kept,
        n_rays: rays.vertices.len(),
        indices: indices_kept,
    })
}

pub fn non_maximum_suppression_3d_inds(
    dist: &[f32],
    points: &[[f32; 3]],
    rays: &Rays,
    scores: Option<&[f32]>,
    thresh: f32,
    use_bbox: bool,
    _use_kdtree: bool,
    _use_gravity: bool,
) -> Result<Vec<bool>, NmsError> {
    let n_polys = points.len();
    let n_rays = rays.vertices.len();
    if n_rays < 3 {
        return Err(NmsError::TooFewRays);
    }
    if dist.len() != n_polys * n_rays {
        return Err(NmsError::DistShapeMismatch);
    }
    if let Some(scores) = scores {
        if scores.len() != n_polys {
            return Err(NmsError::ScoresShapeMismatch);
        }
    }

    if rays
        .faces
        .iter()
        .any(|face| face.iter().any(|i| *i >= n_rays))
    {
        return Err(NmsError::RayCountMismatch);
    }

    let mut volumes = vec![0.0f32; n_polys];
    let mut radius_outer = vec![0.0f32; n_polys];
    let mut radius_inner_isotropic = vec![0.0f32; n_polys];
    let mut radius_outer_isotropic = vec![0.0f32; n_polys];
    let mut radius_inner_gravity = vec![0.0f32; n_polys];
    let mut radius_outer_gravity = vec![0.0f32; n_polys];
    let mut poly_offset_gravity = vec![[0.0f32; 3]; n_polys];
    let mut bbox = vec![[0isize; 6]; n_polys];
    let mut suppressed = vec![false; n_polys];
    let mut anisotropy = [0.0f32; 3];
    let mut polyverts_cache: Vec<Vec<[f32; 3]>> = Vec::with_capacity(n_polys);
    let mut tetrahedra = Vec::with_capacity(n_polys);

    for i in 0..n_polys {
        let curr_dist = &dist[i * n_rays..(i + 1) * n_rays];
        volumes[i] = polyhedron_volume(curr_dist, &rays.vertices, &rays.faces)?;
        bbox[i] = polyhedron_bbox(curr_dist, &points[i], &rays.vertices)?;
        let curr_polyverts = polyhedron_polyverts(curr_dist, &points[i], &rays.vertices)?;
        tetrahedra.push(precompute_tetrahedron_planes(
            &points[i],
            &curr_polyverts,
            &rays.faces,
        ));
        polyverts_cache.push(curr_polyverts);
        anisotropy[0] += (bbox[i][1] - bbox[i][0]) as f32 / n_polys as f32;
        anisotropy[1] += (bbox[i][3] - bbox[i][2]) as f32 / n_polys as f32;
        anisotropy[2] += (bbox[i][5] - bbox[i][4]) as f32 / n_polys as f32;
    }

    let tmp = anisotropy[0].max(anisotropy[1]).max(anisotropy[2]);
    anisotropy[0] = tmp / anisotropy[0];
    anisotropy[1] = tmp / anisotropy[1];
    anisotropy[2] = tmp / anisotropy[2];

    let mut max_dist = 0.0f32;
    for i in 0..n_polys {
        let curr_dist = &dist[i * n_rays..(i + 1) * n_rays];
        radius_outer[i] = bounding_radius_outer(curr_dist);
        radius_outer_isotropic[i] =
            bounding_radius_outer_isotropic(curr_dist, &rays.vertices, &anisotropy)?;
        radius_inner_isotropic[i] =
            bounding_radius_inner_isotropic(curr_dist, &rays.vertices, &rays.faces, &anisotropy)?;
        poly_offset_gravity[i] =
            calculate_poly_offset_gravity(curr_dist, &rays.vertices, &points[i])?;
        radius_outer_gravity[i] = bounding_radius_outer_gravity(
            curr_dist,
            &rays.vertices,
            &anisotropy,
            &poly_offset_gravity[i],
        )?;
        radius_inner_gravity[i] = bounding_radius_inner_gravity(
            curr_dist,
            &rays.vertices,
            &rays.faces,
            &anisotropy,
            &poly_offset_gravity[i],
        )?;
        max_dist = max_dist.max(radius_outer[i]);
    }

    for i in 0..n_polys.saturating_sub(1) {
        if suppressed[i] {
            continue;
        }
        let curr_dist = &dist[i * n_rays..(i + 1) * n_rays];
        let nz = (bbox[i][1] - bbox[i][0] + 1).max(0) as usize;
        let ny = (bbox[i][3] - bbox[i][2] + 1).max(0) as usize;
        let nx = (bbox[i][5] - bbox[i][4] + 1).max(0) as usize;
        let curr_polyverts = &polyverts_cache[i];
        let curr_rendered = OnceLock::new();
        let j_start = i + 1;
        let remaining = n_polys.saturating_sub(j_start);
        if remaining == 0 {
            continue;
        }
        let workers = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1)
            .min(remaining);

        let suppressed_this_round = std::thread::scope(|scope| {
            let mut handles = Vec::with_capacity(workers);
            let suppressed_ref = &suppressed;
            let volumes_ref = &volumes;
            let radius_outer_ref = &radius_outer;
            let radius_inner_isotropic_ref = &radius_inner_isotropic;
            let radius_outer_isotropic_ref = &radius_outer_isotropic;
            let radius_inner_gravity_ref = &radius_inner_gravity;
            let radius_outer_gravity_ref = &radius_outer_gravity;
            let poly_offset_gravity_ref = &poly_offset_gravity;
            let bbox_ref = &bbox;
            let tetrahedra_ref = &tetrahedra;
            let curr_rendered_ref = &curr_rendered;
            let points_ref = points;
            let faces_ref = &rays.faces;
            for worker in 0..workers {
                let start = j_start + worker * remaining / workers;
                let end = j_start + (worker + 1) * remaining / workers;
                handles.push(scope.spawn(move || {
                    let mut local_suppressed = Vec::new();
                    for j in start..end {
                        if suppressed_ref[j] {
                            continue;
                        }
                        if _use_kdtree {
                            let dz = points_ref[i][0] - points_ref[j][0];
                            let dy = points_ref[i][1] - points_ref[j][1];
                            let dx = points_ref[i][2] - points_ref[j][2];
                            if dz * dz + dy * dy + dx * dx
                                > (max_dist + radius_outer_ref[i]).powi(2)
                            {
                                continue;
                            }
                        }

                        let a_min = volumes_ref[i].min(volumes_ref[j]);
                        let a_sphere_outer = if _use_gravity {
                            intersect_sphere_gravity(
                                radius_outer_gravity_ref[i],
                                &points_ref[i],
                                &poly_offset_gravity_ref[i],
                                radius_outer_gravity_ref[j],
                                &points_ref[j],
                                &poly_offset_gravity_ref[j],
                                &anisotropy,
                            )
                        } else {
                            intersect_sphere_isotropic(
                                radius_outer_isotropic_ref[i],
                                &points_ref[i],
                                radius_outer_isotropic_ref[j],
                                &points_ref[j],
                                &anisotropy,
                            )
                        };
                        let mut a_inter =
                            a_sphere_outer.min(intersect_bbox(&bbox_ref[i], &bbox_ref[j]));
                        let mut iou = 1.0f32.min(a_inter / (a_min + 1.0e-10));
                        if use_bbox && (a_inter < 1.0e-10 || iou <= thresh) {
                            continue;
                        }

                        a_inter = if _use_gravity {
                            intersect_sphere_gravity(
                                radius_inner_gravity_ref[i],
                                &points_ref[i],
                                &poly_offset_gravity_ref[i],
                                radius_inner_gravity_ref[j],
                                &points_ref[j],
                                &poly_offset_gravity_ref[j],
                                &anisotropy,
                            )
                        } else {
                            intersect_sphere_isotropic(
                                radius_inner_isotropic_ref[i],
                                &points_ref[i],
                                radius_inner_isotropic_ref[j],
                                &points_ref[j],
                                &anisotropy,
                            )
                        };
                        iou = 0.0f32.max(a_inter / (a_min + 1.0e-10));
                        if iou > thresh {
                            local_suppressed.push(j);
                            continue;
                        }

                        let rendered = curr_rendered_ref.get_or_init(|| {
                            render_polyhedron(
                                curr_dist,
                                &points_ref[i],
                                &bbox_ref[i],
                                curr_polyverts,
                                faces_ref,
                                nz,
                                ny,
                                nx,
                            )
                            .expect("validated polyhedron should render")
                        });
                        let a_inter_render = overlap_render_polyhedron_precomputed(
                            &bbox_ref[i],
                            &tetrahedra_ref[j],
                            rendered,
                            nz,
                            ny,
                            nx,
                            (a_min + 1.0e-10) * thresh,
                        ) as f32;
                        iou = a_inter_render / (a_min + 1.0e-10);
                        if iou > thresh {
                            local_suppressed.push(j);
                        }
                    }
                    local_suppressed
                }));
            }

            let mut merged = Vec::new();
            for handle in handles {
                merged.extend(handle.join().expect("nms worker panicked"));
            }
            merged
        });

        for j in suppressed_this_round {
            suppressed[j] = true;
        }
    }

    Ok(suppressed.into_iter().map(|v| !v).collect())
}

pub fn non_maximum_suppression_inds(
    dist: &[f32],
    points: &[[f32; 2]],
    scores: Option<&[f32]>,
    n_rays: usize,
    thresh: f32,
    use_bbox: bool,
    _use_kdtree: bool,
) -> Result<Vec<bool>, NmsError> {
    if n_rays < 3 {
        return Err(NmsError::TooFewRays);
    }
    let n_polys = points.len();
    if dist.len() != n_polys * n_rays {
        return Err(NmsError::DistShapeMismatch);
    }
    if let Some(scores) = scores {
        if scores.len() != n_polys {
            return Err(NmsError::ScoresShapeMismatch);
        }
    }

    let angle_pi = 2.0f32 * std::f32::consts::PI / n_rays as f32;
    let mut bbox_x1 = vec![0.0; n_polys];
    let mut bbox_x2 = vec![0.0; n_polys];
    let mut bbox_y1 = vec![0.0; n_polys];
    let mut bbox_y2 = vec![0.0; n_polys];
    let mut areas = vec![0.0; n_polys];
    let mut suppressed = vec![false; n_polys];
    let mut poly_paths = Vec::with_capacity(n_polys);

    for i in 0..n_polys {
        let py = points[i][0];
        let px = points[i][1];
        let mut clip = Vec::with_capacity(n_rays);
        for k in 0..n_rays {
            let d = dist[i * n_rays + k];
            let y = py + d * (angle_pi * k as f32).sin();
            let x = px + d * (angle_pi * k as f32).cos();
            if k == 0 {
                bbox_x1[i] = x;
                bbox_x2[i] = x;
                bbox_y1[i] = y;
                bbox_y2[i] = y;
            } else {
                bbox_x1[i] = bbox_x1[i].min(x);
                bbox_x2[i] = bbox_x2[i].max(x);
                bbox_y1[i] = bbox_y1[i].min(y);
                bbox_y2[i] = bbox_y2[i].max(y);
            }
            clip.push([x, y]);
        }
        areas[i] = area_from_path(&clip);
        poly_paths.push(clip);
    }

    for i in 0..n_polys.saturating_sub(1) {
        if suppressed[i] {
            continue;
        }
        for j in i + 1..n_polys {
            if suppressed[j] {
                continue;
            }
            if use_bbox
                && !bbox_intersect(
                    bbox_x1[i], bbox_x2[i], bbox_y1[i], bbox_y2[i], bbox_x1[j], bbox_x2[j],
                    bbox_y1[j], bbox_y2[j],
                )
            {
                continue;
            }
            let area_inter = poly_intersection_area(&poly_paths[i], &poly_paths[j]);
            let overlap = area_inter / (areas[i] + 1.0e-10).min(areas[j] + 1.0e-10);
            if overlap > thresh {
                suppressed[j] = true;
            }
        }
    }

    Ok(suppressed.into_iter().map(|v| !v).collect())
}

pub fn area_from_path(path: &[[f32; 2]]) -> f32 {
    let mut area = 0.0;
    let n = path.len();
    for i in 0..n {
        area += path[i][0] * path[(i + 1) % n][1] - path[i][1] * path[(i + 1) % n][0];
    }
    0.5 * area.abs()
}

pub fn bbox_intersect(
    bbox_a_x1: f32,
    bbox_a_x2: f32,
    bbox_a_y1: f32,
    bbox_a_y2: f32,
    bbox_b_x1: f32,
    bbox_b_x2: f32,
    bbox_b_y1: f32,
    bbox_b_y2: f32,
) -> bool {
    bbox_b_x1 <= bbox_a_x2
        && bbox_a_x1 <= bbox_b_x2
        && bbox_b_y1 <= bbox_a_y2
        && bbox_a_y1 <= bbox_b_y2
}

pub fn poly_intersection_area(poly_a_path: &[[f32; 2]], poly_b_path: &[[f32; 2]]) -> f32 {
    if poly_a_path.len() < 3 || poly_b_path.len() < 3 {
        return 0.0;
    }

    use geo_clipper::ClipperInt;
    use geo_types::{Coord, LineString, Polygon};

    let mut a = Vec::with_capacity(poly_a_path.len() + 1);
    let mut b = Vec::with_capacity(poly_b_path.len() + 1);
    for p in poly_a_path {
        a.push(Coord {
            x: p[0] as i64,
            y: p[1] as i64,
        });
    }
    for p in poly_b_path {
        b.push(Coord {
            x: p[0] as i64,
            y: p[1] as i64,
        });
    }
    a.push(a[0]);
    b.push(b[0]);

    let poly_a = Polygon::new(LineString(a), vec![]);
    let poly_b = Polygon::new(LineString(b), vec![]);
    let intersection = poly_a.intersection(&poly_b);
    let mut area = 0.0f32;
    for poly in intersection.0 {
        area += area_from_path_i64(poly.exterior().0.as_slice());
        for interior in poly.interiors() {
            area -= area_from_path_i64(interior.0.as_slice());
        }
    }
    area
}

fn area_from_path_i64(path: &[geo_types::Coord<i64>]) -> f32 {
    if path.len() < 3 {
        return 0.0;
    }
    let mut area = 0.0f32;
    let n = if path[0] == path[path.len() - 1] {
        path.len() - 1
    } else {
        path.len()
    };
    for i in 0..n {
        area += path[i].x as f32 * path[(i + 1) % n].y as f32
            - path[i].y as f32 * path[(i + 1) % n].x as f32;
    }
    0.5 * area.abs()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ind_prob_thresh_excludes_boundary_like_stardist() {
        let prob = vec![1.0; 25];
        let mask = _ind_prob_thresh(&prob, &[5, 5], 0.5, Some(&[[1, 1], [1, 1]])).unwrap();
        assert_eq!(mask.iter().filter(|v| **v).count(), 9);
        assert!(!mask[0]);
        assert!(mask[2 * 5 + 2]);
    }

    #[test]
    fn non_maximum_suppression_old_returns_original_mask_points() {
        let n_rays = 4;
        let shape = [4, 4];
        let mut prob = vec![0.0f32; shape[0] * shape[1]];
        prob[1 * shape[1] + 1] = 0.9;
        prob[2 * shape[1] + 2] = 0.8;
        let mut coord = vec![0.0f32; shape[0] * shape[1] * 2 * n_rays];
        for point in [[1usize, 1usize], [2usize, 2usize]] {
            let flat = point[0] * shape[1] + point[1];
            coord[(flat * 2) * n_rays] = point[0] as f32 - 0.4;
            coord[(flat * 2) * n_rays + 1] = point[0] as f32;
            coord[(flat * 2) * n_rays + 2] = point[0] as f32 + 0.4;
            coord[(flat * 2) * n_rays + 3] = point[0] as f32;
            coord[(flat * 2 + 1) * n_rays] = point[1] as f32;
            coord[(flat * 2 + 1) * n_rays + 1] = point[1] as f32 + 0.4;
            coord[(flat * 2 + 1) * n_rays + 2] = point[1] as f32;
            coord[(flat * 2 + 1) * n_rays + 3] = point[1] as f32 - 0.4;
        }
        let points = _non_maximum_suppression_old(
            &coord,
            &prob,
            shape,
            n_rays,
            [1, 1],
            None,
            0.5,
            0.5,
            false,
            true,
        )
        .unwrap();
        assert_eq!(points, vec![[1, 1], [2, 2]]);
    }

    #[test]
    fn non_maximum_suppression_old_suppresses_overlapping_lower_score_polygon() {
        let n_rays = 4;
        let shape = [3, 3];
        let mut prob = vec![0.0f32; shape[0] * shape[1]];
        prob[1 * shape[1] + 1] = 0.9;
        prob[1 * shape[1] + 2] = 0.8;
        let mut coord = vec![0.0f32; shape[0] * shape[1] * 2 * n_rays];
        for point in [[1usize, 1usize], [1usize, 2usize]] {
            let flat = point[0] * shape[1] + point[1];
            coord[(flat * 2) * n_rays] = 0.0;
            coord[(flat * 2) * n_rays + 1] = 1.0;
            coord[(flat * 2) * n_rays + 2] = 2.0;
            coord[(flat * 2) * n_rays + 3] = 1.0;
            coord[(flat * 2 + 1) * n_rays] = 1.0;
            coord[(flat * 2 + 1) * n_rays + 1] = 2.0;
            coord[(flat * 2 + 1) * n_rays + 2] = 1.0;
            coord[(flat * 2 + 1) * n_rays + 3] = 0.0;
        }
        let points = _non_maximum_suppression_old(
            &coord,
            &prob,
            shape,
            n_rays,
            [1, 1],
            None,
            0.5,
            0.5,
            false,
            true,
        )
        .unwrap();
        assert_eq!(points, vec![[1, 1]]);
    }

    #[test]
    fn area_from_path_matches_square_area() {
        let square = [[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0]];
        assert!((area_from_path(&square) - 4.0).abs() < 1e-6);
    }

    #[test]
    fn poly_intersection_area_matches_overlapping_squares() {
        let a = [[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0]];
        let b = [[1.0, 1.0], [3.0, 1.0], [3.0, 3.0], [1.0, 3.0]];
        assert!((poly_intersection_area(&a, &b) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn non_maximum_suppression_inds_suppresses_identical_lower_score_polygon() {
        let dist = vec![2.0; 16];
        let points = [[5.0, 5.0], [5.0, 5.0]];
        let scores = [0.9, 0.8];
        let inds = non_maximum_suppression_inds(&dist, &points, Some(&scores), 8, 0.5, true, false)
            .unwrap();
        assert_eq!(inds, vec![true, false]);
    }

    #[test]
    fn non_maximum_suppression_returns_sorted_candidates() {
        let n_rays = 8;
        let shape = [3, 3];
        let mut prob = vec![0.0; 9];
        prob[4] = 0.8;
        prob[8] = 0.9;
        let dist = vec![1.0; 9 * n_rays];
        let nms = non_maximum_suppression(
            &dist,
            &prob,
            shape,
            n_rays,
            [1, 1],
            None,
            0.5,
            0.5,
            true,
            false,
        )
        .unwrap();
        assert_eq!(nms.prob, vec![0.9, 0.8]);
        assert_eq!(nms.points, vec![[2.0, 2.0], [1.0, 1.0]]);
    }

    #[test]
    fn non_maximum_suppression_3d_inds_suppresses_identical_lower_score_polyhedron() {
        let rays = crate::RaysGoldenSpiral::new(8, None).unwrap().into_rays();
        let dist = vec![2.0; 2 * rays.vertices.len()];
        let points = [[5.0, 5.0, 5.0], [5.0, 5.0, 5.0]];
        let scores = [0.9, 0.8];
        let inds = non_maximum_suppression_3d_inds(
            &dist,
            &points,
            &rays,
            Some(&scores),
            0.5,
            true,
            false,
            false,
        )
        .unwrap();
        assert_eq!(inds, vec![true, false]);
    }

    #[test]
    fn non_maximum_suppression_3d_returns_sorted_candidates() {
        let rays = crate::RaysGoldenSpiral::new(8, None).unwrap().into_rays();
        let shape = [2, 2, 2];
        let mut prob = vec![0.0; 8];
        prob[0] = 0.7;
        prob[7] = 0.9;
        let dist = vec![1.0; prob.len() * rays.vertices.len()];
        let nms = non_maximum_suppression_3d(
            &dist,
            &prob,
            shape,
            &rays,
            [1, 2, 2],
            None,
            0.5,
            0.5,
            true,
            false,
            false,
        )
        .unwrap();
        assert_eq!(nms.prob, vec![0.9, 0.7]);
        assert_eq!(nms.points, vec![[1.0, 2.0, 2.0], [0.0, 0.0, 0.0]]);
    }

    #[test]
    fn non_maximum_suppression_3d_sparse_returns_original_indices() {
        let rays = crate::RaysGoldenSpiral::new(8, None).unwrap().into_rays();
        let prob = [0.6, 0.9, 0.7];
        let points = [[0.0, 0.0, 0.0], [10.0, 10.0, 10.0], [20.0, 20.0, 20.0]];
        let dist = vec![1.0; prob.len() * rays.vertices.len()];
        let nms = non_maximum_suppression_3d_sparse(
            &dist, &prob, &points, &rays, None, 0.5, true, false, false,
        )
        .unwrap();
        assert_eq!(nms.prob, vec![0.9, 0.7, 0.6]);
        assert_eq!(nms.indices, vec![1, 2, 0]);
    }
}
