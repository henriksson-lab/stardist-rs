use std::f32::consts::PI;
use std::path::Path;

use ndarray::{Array2, Array3, Array4, Array5};

use crate::Rays;
use crate::utils::{_normalize_grid, GridError};

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum GeometryError {
    #[error("need n_rays >= 3")]
    TooFewRays,
    #[error("label image length does not match shape")]
    ShapeMismatch,
    #[error("dist must be shaped as n_polys by n_rays")]
    DistShapeMismatch,
    #[error("points must be shaped as n_polys by 2")]
    PointsShapeMismatch,
    #[error("points must be shaped as n_polys by 3")]
    Points3DShapeMismatch,
    #[error("ray count does not match distance shape")]
    RayCountMismatch,
    #[error("distance array should be positive")]
    NonPositiveDistance,
    #[error("labels must be shaped as n_polys")]
    LabelsShapeMismatch,
    #[error("wrong obj export shapes")]
    ObjShapeMismatch,
    #[error("failed to write obj file")]
    ObjWriteFailed,
    #[error("unknown render mode")]
    UnknownRenderMode,
    #[error("prob must be shaped as n_polys")]
    ProbShapeMismatch,
    #[error("python star_dist only supports grid [1, 1]")]
    UnsupportedPythonGrid2D,
    #[error(transparent)]
    Grid(#[from] GridError),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PolyhedronRenderMode {
    Full,
    Kernel,
    Hull,
    Bbox,
    Debug,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CentroidMode {
    Absolute,
    Relative,
}

pub fn ray_angles(n_rays: usize) -> Vec<f32> {
    let mut angles = Vec::with_capacity(n_rays);
    for k in 0..n_rays {
        angles.push((2.0 * PI) * (k as f32) / (n_rays as f32));
    }
    angles
}

pub fn star_dist(
    labels: &[u16],
    shape: [usize; 2],
    n_rays: usize,
    grid: [usize; 2],
) -> Result<Array3<f32>, GeometryError> {
    if n_rays < 3 {
        return Err(GeometryError::TooFewRays);
    }
    if labels.len() != shape[0] * shape[1] {
        return Err(GeometryError::ShapeMismatch);
    }
    let grid = _normalize_grid::<2>(&grid)?;
    let height = shape[0];
    let width = shape[1];
    let out_height = (height - 1) / grid[0] + 1;
    let out_width = (width - 1) / grid[1] + 1;
    let mut dst = Array3::<f32>::zeros((out_height, out_width, n_rays));
    for i in 0..out_height {
        for j in 0..out_width {
            let value = labels[(i * grid[0]) * width + (j * grid[1])];
            if value == 0 {
                for k in 0..n_rays {
                    dst[[i, j, k]] = 0.0;
                }
            } else {
                let st_rays = (2.0 * PI) / (n_rays as f32);
                for k in 0..n_rays {
                    let phi = (k as f32) * st_rays;
                    let dy = phi.cos();
                    let dx = phi.sin();
                    let mut y = 0.0f32;
                    let mut x = 0.0f32;
                    loop {
                        x += dx;
                        y += dy;
                        let ii = ((i * grid[0]) as f32 + y).round() as isize;
                        let jj = ((j * grid[1]) as f32 + x).round() as isize;
                        if ii < 0
                            || ii >= height as isize
                            || jj < 0
                            || jj >= width as isize
                            || value != labels[(ii as usize) * width + (jj as usize)]
                        {
                            let t_corr = 0.5 / dx.abs().max(dy.abs());
                            x += (t_corr - 1.0) * dx;
                            y += (t_corr - 1.0) * dy;
                            dst[[i, j, k]] = (x * x + y * y).sqrt();
                            break;
                        }
                    }
                }
            }
        }
    }
    Ok(dst)
}

pub fn _py_star_dist(
    labels: &[u16],
    shape: [usize; 2],
    n_rays: usize,
    grid: [usize; 2],
) -> Result<Array3<f32>, GeometryError> {
    if grid != [1, 1] {
        return Err(GeometryError::UnsupportedPythonGrid2D);
    }
    star_dist(labels, shape, n_rays, grid)
}

pub fn _cpp_star_dist(
    labels: &[u16],
    shape: [usize; 2],
    n_rays: usize,
    grid: [usize; 2],
) -> Result<Array3<f32>, GeometryError> {
    star_dist(labels, shape, n_rays, grid)
}

pub fn _ocl_star_dist(
    labels: &[u16],
    shape: [usize; 2],
    n_rays: usize,
    grid: [usize; 2],
) -> Result<Array3<f32>, GeometryError> {
    star_dist(labels, shape, n_rays, grid)
}

pub fn dist_to_coord(
    dist: &[f32],
    points: &[[f32; 2]],
    n_rays: usize,
    scale_dist: [f32; 2],
) -> Result<Array3<f32>, GeometryError> {
    if n_rays < 3 {
        return Err(GeometryError::TooFewRays);
    }
    if dist.len() != points.len() * n_rays {
        return Err(GeometryError::DistShapeMismatch);
    }
    let mut coord = Array3::<f32>::zeros((points.len(), 2, n_rays));
    let phis = ray_angles(n_rays);
    for i in 0..points.len() {
        for k in 0..n_rays {
            coord[[i, 0, k]] = points[i][0] + dist[i * n_rays + k] * phis[k].sin() * scale_dist[0];
            coord[[i, 1, k]] = points[i][1] + dist[i * n_rays + k] * phis[k].cos() * scale_dist[1];
        }
    }
    Ok(coord)
}

pub fn _dist_to_coord_old(
    rhos: &[f32],
    shape: [usize; 4],
    grid: [usize; 2],
) -> Result<Array5<f32>, GeometryError> {
    let grid = _normalize_grid::<2>(&grid)?;
    let n_images = shape[0];
    let height = shape[1];
    let width = shape[2];
    let n_rays = shape[3];
    if rhos.len() != n_images * height * width * n_rays {
        return Err(GeometryError::DistShapeMismatch);
    }

    let mut coord = Array5::<f32>::zeros((n_images, height, width, 2, n_rays));
    for image in 0..n_images {
        for y in 0..height {
            for x in 0..width {
                for n in 0..n_rays {
                    coord[[image, y, x, 0, n]] = grid[0] as f32 * y as f32;
                    coord[[image, y, x, 1, n]] = grid[1] as f32 * x as f32;
                }
            }
        }
    }

    let phis = ray_angles(n_rays);
    for image in 0..n_images {
        for y in 0..height {
            for x in 0..width {
                for n in 0..n_rays {
                    let rho = rhos[((image * height + y) * width + x) * n_rays + n];
                    coord[[image, y, x, 0, n]] += rho * phis[n].sin();
                    coord[[image, y, x, 1, n]] += rho * phis[n].cos();
                }
            }
        }
    }

    Ok(coord)
}

pub fn polygons_to_label_coord(
    coord: &[f32],
    n_polys: usize,
    n_rays: usize,
    shape: [usize; 2],
    labels: Option<&[u32]>,
) -> Result<Array2<u32>, GeometryError> {
    if coord.len() != n_polys * 2 * n_rays {
        return Err(GeometryError::DistShapeMismatch);
    }
    if let Some(labels) = labels {
        if labels.len() != n_polys {
            return Err(GeometryError::PointsShapeMismatch);
        }
    }
    let mut out = Array2::<u32>::zeros((shape[0], shape[1]));
    for i in 0..n_polys {
        let mut min_y = f32::INFINITY;
        let mut max_y = f32::NEG_INFINITY;
        let mut min_x = f32::INFINITY;
        let mut max_x = f32::NEG_INFINITY;
        for k in 0..n_rays {
            let y = coord[(i * 2 * n_rays) + k];
            let x = coord[(i * 2 * n_rays) + n_rays + k];
            min_y = min_y.min(y);
            max_y = max_y.max(y);
            min_x = min_x.min(x);
            max_x = max_x.max(x);
        }
        let y0 = min_y.floor().max(0.0) as usize;
        let y1 = max_y.ceil().min((shape[0] - 1) as f32) as usize;
        let x0 = min_x.floor().max(0.0) as usize;
        let x1 = max_x.ceil().min((shape[1] - 1) as f32) as usize;
        for y in y0..=y1 {
            for x in x0..=x1 {
                let mut inside = false;
                let mut j = n_rays - 1;
                for k in 0..n_rays {
                    let yi = coord[(i * 2 * n_rays) + k];
                    let xi = coord[(i * 2 * n_rays) + n_rays + k];
                    let yj = coord[(i * 2 * n_rays) + j];
                    let xj = coord[(i * 2 * n_rays) + n_rays + j];
                    let py = y as f32;
                    let px = x as f32;
                    if ((yi > py) != (yj > py)) && (px < (xj - xi) * (py - yi) / (yj - yi) + xi) {
                        inside = !inside;
                    }
                    j = k;
                }
                if inside {
                    out[[y, x]] = labels.map(|v| v[i] + 1).unwrap_or((i + 1) as u32);
                }
            }
        }
    }
    Ok(out)
}

pub fn _polygons_to_label_old(
    coord: &[f32],
    coord_shape: [usize; 4],
    prob: &[f32],
    points: &[[usize; 2]],
    shape: Option<[usize; 2]>,
    thr: f32,
) -> Result<Array2<u32>, GeometryError> {
    let height = coord_shape[0];
    let width = coord_shape[1];
    let axis = coord_shape[2];
    let n_rays = coord_shape[3];
    if axis != 2 || coord.len() != height * width * 2 * n_rays {
        return Err(GeometryError::DistShapeMismatch);
    }
    if prob.len() != height * width {
        return Err(GeometryError::ProbShapeMismatch);
    }
    if points
        .iter()
        .any(|point| point[0] >= height || point[1] >= width)
    {
        return Err(GeometryError::PointsShapeMismatch);
    }

    let out_shape = shape.unwrap_or([height, width]);
    let mut lbl = Array2::<u32>::zeros((out_shape[0], out_shape[1]));
    let mut ind: Vec<usize> = (0..points.len()).collect();
    ind.sort_by(|&a, &b| {
        let pa = prob[points[a][0] * width + points[a][1]];
        let pb = prob[points[b][0] * width + points[b][1]];
        pa.partial_cmp(&pb)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.cmp(&b))
    });

    let mut i = 1u32;
    for point_index in ind {
        let p = points[point_index];
        if prob[p[0] * width + p[1]] < thr {
            continue;
        }

        let mut min_y = f32::INFINITY;
        let mut max_y = f32::NEG_INFINITY;
        let mut min_x = f32::INFINITY;
        let mut max_x = f32::NEG_INFINITY;
        for n in 0..n_rays {
            let y = coord[(((p[0] * width + p[1]) * 2) * n_rays) + n];
            let x = coord[(((p[0] * width + p[1]) * 2 + 1) * n_rays) + n];
            min_y = min_y.min(y);
            max_y = max_y.max(y);
            min_x = min_x.min(x);
            max_x = max_x.max(x);
        }

        let y0 = min_y.floor().max(0.0) as usize;
        let y1 = max_y.ceil().min((out_shape[0] - 1) as f32) as usize;
        let x0 = min_x.floor().max(0.0) as usize;
        let x1 = max_x.ceil().min((out_shape[1] - 1) as f32) as usize;
        for y in y0..=y1 {
            for x in x0..=x1 {
                let mut inside = false;
                let mut j = n_rays - 1;
                for n in 0..n_rays {
                    let yi = coord[(((p[0] * width + p[1]) * 2) * n_rays) + n];
                    let xi = coord[(((p[0] * width + p[1]) * 2 + 1) * n_rays) + n];
                    let yj = coord[(((p[0] * width + p[1]) * 2) * n_rays) + j];
                    let xj = coord[(((p[0] * width + p[1]) * 2 + 1) * n_rays) + j];
                    let py = y as f32;
                    let px = x as f32;
                    if ((yi > py) != (yj > py)) && (px < (xj - xi) * (py - yi) / (yj - yi) + xi) {
                        inside = !inside;
                    }
                    j = n;
                }
                if inside {
                    lbl[[y, x]] = i;
                }
            }
        }
        i += 1;
    }

    Ok(lbl)
}

pub fn polygons_to_label(
    dist: &[f32],
    points: &[[f32; 2]],
    shape: [usize; 2],
    prob: Option<&[f32]>,
    thr: f32,
    scale_dist: [f32; 2],
) -> Result<Array2<u32>, GeometryError> {
    if points.len()
        * points
            .first()
            .map(|_| dist.len() / points.len())
            .unwrap_or(0)
        != dist.len()
    {
        return Err(GeometryError::DistShapeMismatch);
    }
    if let Some(prob) = prob {
        if prob.len() != points.len() {
            return Err(GeometryError::ProbShapeMismatch);
        }
    }
    let n_rays = if points.is_empty() {
        0
    } else {
        dist.len() / points.len()
    };
    if n_rays < 3 && !points.is_empty() {
        return Err(GeometryError::TooFewRays);
    }

    let mut points_filtered = Vec::new();
    let mut dist_filtered = Vec::new();
    let mut prob_filtered = Vec::new();
    let mut ind = Vec::new();
    for i in 0..points.len() {
        let p = prob.map(|prob| prob[i]).unwrap_or(f32::INFINITY);
        if p > thr {
            points_filtered.push(points[i]);
            dist_filtered.extend_from_slice(&dist[i * n_rays..(i + 1) * n_rays]);
            prob_filtered.push(p);
            ind.push(i);
        }
    }

    let mut sorted: Vec<usize> = (0..prob_filtered.len()).collect();
    sorted.sort_by(|&a, &b| {
        prob_filtered[a]
            .partial_cmp(&prob_filtered[b])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.cmp(&b))
    });

    let mut points_sorted = Vec::with_capacity(sorted.len());
    let mut dist_sorted = Vec::with_capacity(sorted.len() * n_rays);
    let mut labels = Vec::with_capacity(sorted.len());
    for i in sorted {
        points_sorted.push(points_filtered[i]);
        dist_sorted.extend_from_slice(&dist_filtered[i * n_rays..(i + 1) * n_rays]);
        labels.push(ind[i] as u32);
    }

    if points_sorted.is_empty() {
        return Ok(Array2::<u32>::zeros((shape[0], shape[1])));
    }

    let coord = dist_to_coord(&dist_sorted, &points_sorted, n_rays, scale_dist)?;
    let coord_values: Vec<f32> = coord.iter().copied().collect();
    polygons_to_label_coord(
        &coord_values,
        points_sorted.len(),
        n_rays,
        shape,
        Some(&labels),
    )
}

pub fn relabel_image_stardist(
    lbl: &[u16],
    shape: [usize; 2],
    n_rays: usize,
    grid: [usize; 2],
) -> Result<Array2<u32>, GeometryError> {
    if lbl.len() != shape[0] * shape[1] {
        return Err(GeometryError::ShapeMismatch);
    }
    let dist_all = star_dist(lbl, shape, n_rays, grid)?;
    let mut max_label = 0u16;
    for value in lbl {
        max_label = max_label.max(*value);
    }

    let mut count = vec![0usize; max_label as usize + 1];
    let mut sum_y = vec![0usize; max_label as usize + 1];
    let mut sum_x = vec![0usize; max_label as usize + 1];
    for y in 0..shape[0] {
        for x in 0..shape[1] {
            let value = lbl[y * shape[1] + x] as usize;
            if value > 0 {
                count[value] += 1;
                sum_y[value] += y;
                sum_x[value] += x;
            }
        }
    }

    let mut points = Vec::<[f32; 2]>::new();
    let mut dist = Vec::<f32>::new();
    for label in 1..=max_label as usize {
        if count[label] == 0 {
            continue;
        }
        let y = sum_y[label] / count[label];
        let x = sum_x[label] / count[label];
        points.push([y as f32, x as f32]);
        for n in 0..n_rays {
            dist.push(dist_all[[y, x, n]]);
        }
    }

    if points.is_empty() {
        let dist = Vec::<f32>::new();
        let points = Vec::<[f32; 2]>::new();
        polygons_to_label(&dist, &points, shape, None, f32::NEG_INFINITY, [1.0, 1.0])
    } else {
        polygons_to_label(&dist, &points, shape, None, f32::NEG_INFINITY, [1.0, 1.0])
    }
}

pub fn star_dist3d(
    labels: &[u16],
    shape: [usize; 3],
    rays: &Rays,
    grid: [usize; 3],
) -> Result<Array4<f32>, GeometryError> {
    if labels.len() != shape[0] * shape[1] * shape[2] {
        return Err(GeometryError::ShapeMismatch);
    }
    let grid = _normalize_grid::<3>(&grid)?;
    let depth = shape[0];
    let height = shape[1];
    let width = shape[2];
    let n_rays = rays.vertices.len();
    let out_depth = (depth - 1) / grid[0] + 1;
    let out_height = (height - 1) / grid[1] + 1;
    let out_width = (width - 1) / grid[2] + 1;
    let mut dst = Array4::<f32>::zeros((out_depth, out_height, out_width, n_rays));
    for i in 0..out_depth {
        for j in 0..out_height {
            for k in 0..out_width {
                let value =
                    labels[((i * grid[0]) * height + (j * grid[1])) * width + (k * grid[2])];
                if value == 0 {
                    for n in 0..n_rays {
                        dst[[i, j, k, n]] = 0.0;
                    }
                } else {
                    for n in 0..n_rays {
                        let dz = rays.vertices[n][0];
                        let dy = rays.vertices[n][1];
                        let dx = rays.vertices[n][2];
                        let mut x = 0.0f32;
                        let mut y = 0.0f32;
                        let mut z = 0.0f32;
                        loop {
                            x += dx;
                            y += dy;
                            z += dz;
                            let ii = ((i * grid[0]) as f32 + z).round() as isize;
                            let jj = ((j * grid[1]) as f32 + y).round() as isize;
                            let kk = ((k * grid[2]) as f32 + x).round() as isize;
                            if ii < 0
                                || ii >= depth as isize
                                || jj < 0
                                || jj >= height as isize
                                || kk < 0
                                || kk >= width as isize
                                || value
                                    != labels[((ii as usize) * height + (jj as usize)) * width
                                        + (kk as usize)]
                            {
                                let x2 = x.round();
                                let y2 = y.round();
                                let z2 = z.round();
                                dst[[i, j, k, n]] = (x2 * x2 + y2 * y2 + z2 * z2).sqrt();
                                break;
                            }
                        }
                    }
                }
            }
        }
    }
    Ok(dst)
}

pub fn _py_star_dist3d(
    labels: &[u16],
    shape: [usize; 3],
    rays: &Rays,
    grid: [usize; 3],
) -> Result<Array4<f32>, GeometryError> {
    star_dist3d(labels, shape, rays, grid)
}

pub fn _cpp_star_dist3d(
    labels: &[u16],
    shape: [usize; 3],
    rays: &Rays,
    grid: [usize; 3],
) -> Result<Array4<f32>, GeometryError> {
    star_dist3d(labels, shape, rays, grid)
}

pub fn _ocl_star_dist3d(
    labels: &[u16],
    shape: [usize; 3],
    rays: &Rays,
    grid: [usize; 3],
) -> Result<Array4<f32>, GeometryError> {
    star_dist3d(labels, shape, rays, grid)
}

pub fn dist_to_coord3d(
    dist: &[f32],
    points: &[[f32; 3]],
    rays_vertices: &[[f32; 3]],
) -> Result<Array3<f32>, GeometryError> {
    let n_rays = rays_vertices.len();
    if n_rays == 0 || dist.len() != points.len() * n_rays {
        return Err(GeometryError::DistShapeMismatch);
    }
    let mut coord = Array3::<f32>::zeros((points.len(), n_rays, 3));
    for i in 0..points.len() {
        for n in 0..n_rays {
            coord[[i, n, 0]] = points[i][0] + dist[i * n_rays + n] * rays_vertices[n][0];
            coord[[i, n, 1]] = points[i][1] + dist[i * n_rays + n] * rays_vertices[n][1];
            coord[[i, n, 2]] = points[i][2] + dist[i * n_rays + n] * rays_vertices[n][2];
        }
    }
    Ok(coord)
}

pub fn inside_halfspace(
    z: f32,
    y: f32,
    x: f32,
    az: f32,
    ay: f32,
    ax: f32,
    bz: f32,
    by: f32,
    bx: f32,
    cz: f32,
    cy: f32,
    cx: f32,
) -> bool {
    let m00 = bz - az;
    let m01 = by - ay;
    let m02 = bx - ax;
    let m10 = cz - az;
    let m11 = cy - ay;
    let m12 = cx - ax;
    let m20 = z - az;
    let m21 = y - ay;
    let m22 = x - ax;
    let det = m00 * (m11 * m22 - m21 * m12) - m01 * (m10 * m22 - m12 * m20)
        + m02 * (m10 * m21 - m11 * m20);
    det >= 0.0
}

pub fn inside_tetrahedron(
    z: f32,
    y: f32,
    x: f32,
    rz: f32,
    ry: f32,
    rx: f32,
    az: f32,
    ay: f32,
    ax: f32,
    bz: f32,
    by: f32,
    bx: f32,
    cz: f32,
    cy: f32,
    cx: f32,
) -> bool {
    inside_halfspace(z, y, x, az, ay, ax, bz, by, bx, cz, cy, cx)
        && inside_halfspace(z, y, x, rz, ry, rx, bz, by, bx, az, ay, ax)
        && inside_halfspace(z, y, x, rz, ry, rx, cz, cy, cx, bz, by, bx)
        && inside_halfspace(z, y, x, rz, ry, rx, az, ay, ax, cz, cy, cx)
}

pub fn inside_polyhedron(
    z: f32,
    y: f32,
    x: f32,
    center: &[f32; 3],
    polyverts: &[[f32; 3]],
    faces: &[[usize; 3]],
) -> bool {
    for face in faces {
        let ia = face[0];
        let ib = face[1];
        let ic = face[2];
        if inside_tetrahedron(
            z,
            y,
            x,
            center[0],
            center[1],
            center[2],
            polyverts[ia][0],
            polyverts[ia][1],
            polyverts[ia][2],
            polyverts[ib][0],
            polyverts[ib][1],
            polyverts[ib][2],
            polyverts[ic][0],
            polyverts[ic][1],
            polyverts[ic][2],
        ) {
            return true;
        }
    }
    false
}

pub fn inside_polyhedron_kernel(
    z: f32,
    y: f32,
    x: f32,
    center: &[f32; 3],
    polyverts: &[[f32; 3]],
    faces: &[[usize; 3]],
) -> bool {
    let _ = center;
    for face in faces {
        let ia = face[0];
        let ib = face[1];
        let ic = face[2];
        if !inside_halfspace(
            z,
            y,
            x,
            polyverts[ia][0],
            polyverts[ia][1],
            polyverts[ia][2],
            polyverts[ib][0],
            polyverts[ib][1],
            polyverts[ib][2],
            polyverts[ic][0],
            polyverts[ic][1],
            polyverts[ic][2],
        ) {
            return false;
        }
    }
    true
}

pub fn tetrahedron_volume(
    rz: f32,
    ry: f32,
    rx: f32,
    az: f32,
    ay: f32,
    ax: f32,
    bz: f32,
    by: f32,
    bx: f32,
    cz: f32,
    cy: f32,
    cx: f32,
) -> f32 {
    let m00 = bz - az;
    let m01 = by - ay;
    let m02 = bx - ax;
    let m10 = cz - az;
    let m11 = cy - ay;
    let m12 = cx - ax;
    let m20 = rz - az;
    let m21 = ry - ay;
    let m22 = rx - ax;
    let det = m00 * (m11 * m22 - m21 * m12) - m01 * (m10 * m22 - m12 * m20)
        + m02 * (m10 * m21 - m11 * m20);
    det / 6.0
}

pub fn polyhedron_volume(
    dist: &[f32],
    verts: &[[f32; 3]],
    faces: &[[usize; 3]],
) -> Result<f32, GeometryError> {
    if dist.len() != verts.len() {
        return Err(GeometryError::RayCountMismatch);
    }
    if faces
        .iter()
        .any(|face| face.iter().any(|i| *i >= dist.len()))
    {
        return Err(GeometryError::RayCountMismatch);
    }
    let mut vol = 0.0f32;
    for face in faces {
        let ia = face[0];
        let ib = face[1];
        let ic = face[2];
        let az = dist[ia] * verts[ia][0];
        let ay = dist[ia] * verts[ia][1];
        let ax = dist[ia] * verts[ia][2];
        let bz = dist[ib] * verts[ib][0];
        let by = dist[ib] * verts[ib][1];
        let bx = dist[ib] * verts[ib][2];
        let cz = dist[ic] * verts[ic][0];
        let cy = dist[ic] * verts[ic][1];
        let cx = dist[ic] * verts[ic][2];
        vol += tetrahedron_volume(0.0, 0.0, 0.0, az, ay, ax, bz, by, bx, cz, cy, cx);
    }
    Ok(vol)
}

pub fn polyhedron_centroid(
    dist: &[f32],
    verts: &[[f32; 3]],
    faces: &[[usize; 3]],
) -> Result<[f32; 3], GeometryError> {
    if dist.len() != verts.len() {
        return Err(GeometryError::RayCountMismatch);
    }
    if faces
        .iter()
        .any(|face| face.iter().any(|i| *i >= dist.len()))
    {
        return Err(GeometryError::RayCountMismatch);
    }
    let mut vol = 0.0f32;
    let mut rz = 0.0f32;
    let mut ry = 0.0f32;
    let mut rx = 0.0f32;
    for face in faces {
        let ia = face[0];
        let ib = face[1];
        let ic = face[2];
        let az = dist[ia] * verts[ia][0];
        let ay = dist[ia] * verts[ia][1];
        let ax = dist[ia] * verts[ia][2];
        let bz = dist[ib] * verts[ib][0];
        let by = dist[ib] * verts[ib][1];
        let bx = dist[ib] * verts[ib][2];
        let cz = dist[ic] * verts[ic][0];
        let cy = dist[ic] * verts[ic][1];
        let cx = dist[ic] * verts[ic][2];
        let curr_vol = tetrahedron_volume(0.0, 0.0, 0.0, az, ay, ax, bz, by, bx, cz, cy, cx);
        rz += 0.25 * (az + bz + cz) * curr_vol;
        ry += 0.25 * (ay + by + cy) * curr_vol;
        rx += 0.25 * (ax + bx + cx) * curr_vol;
        vol += curr_vol;
    }
    Ok(if vol > 1.0e-10 {
        [rz / vol, ry / vol, rx / vol]
    } else {
        [0.0, 0.0, 0.0]
    })
}

pub fn bounding_radius_outer(dist: &[f32]) -> f32 {
    let mut r = 0.0f32;
    for d in dist {
        r = r.max(*d);
    }
    r
}

pub fn bounding_radius_inner(
    dist: &[f32],
    verts: &[[f32; 3]],
    faces: &[[usize; 3]],
) -> Result<f32, GeometryError> {
    if dist.len() != verts.len() {
        return Err(GeometryError::RayCountMismatch);
    }
    if faces
        .iter()
        .any(|face| face.iter().any(|i| *i >= dist.len()))
    {
        return Err(GeometryError::RayCountMismatch);
    }
    let mut r = f32::INFINITY;
    for face in faces {
        let ia = face[0];
        let ib = face[1];
        let ic = face[2];
        let az = dist[ia] * verts[ia][0];
        let ay = dist[ia] * verts[ia][1];
        let ax = dist[ia] * verts[ia][2];
        let bz = dist[ib] * verts[ib][0];
        let by = dist[ib] * verts[ib][1];
        let bx = dist[ib] * verts[ib][2];
        let cz = dist[ic] * verts[ic][0];
        let cy = dist[ic] * verts[ic][1];
        let cx = dist[ic] * verts[ic][2];
        let pz = bz - az;
        let py = by - ay;
        let px = bx - ax;
        let qz = cz - az;
        let qy = cy - ay;
        let qx = cx - ax;
        let mut nz = px * qy - py * qx;
        let mut ny = -(px * qz - pz * qx);
        let mut nx = py * qz - pz * qy;
        let normz = 1.0 / ((nz * nz + ny * ny + nx * nx).sqrt() + 1.0e-10);
        nz *= normz;
        ny *= normz;
        nx *= normz;
        let distance = az * nz + ay * ny + ax * nx;
        r = r.min(distance);
    }
    Ok(r)
}

pub fn bounding_radius_outer_isotropic(
    dist: &[f32],
    verts: &[[f32; 3]],
    aniso: &[f32; 3],
) -> Result<f32, GeometryError> {
    if dist.len() != verts.len() {
        return Err(GeometryError::RayCountMismatch);
    }
    let mut r_squared_max = 0.0f32;
    for i in 0..dist.len() {
        let z = aniso[0] * dist[i] * verts[i][0];
        let y = aniso[1] * dist[i] * verts[i][1];
        let x = aniso[2] * dist[i] * verts[i][2];
        let r_squared = z * z + y * y + x * x;
        r_squared_max = r_squared_max.max(r_squared);
    }
    Ok(r_squared_max.sqrt())
}

pub fn bounding_radius_inner_isotropic(
    dist: &[f32],
    verts: &[[f32; 3]],
    faces: &[[usize; 3]],
    aniso: &[f32; 3],
) -> Result<f32, GeometryError> {
    if dist.len() != verts.len() {
        return Err(GeometryError::RayCountMismatch);
    }
    if faces
        .iter()
        .any(|face| face.iter().any(|i| *i >= dist.len()))
    {
        return Err(GeometryError::RayCountMismatch);
    }
    let mut r_min = f32::INFINITY;
    for face in faces {
        let ia = face[0];
        let ib = face[1];
        let ic = face[2];
        let az = aniso[0] * dist[ia] * verts[ia][0];
        let ay = aniso[1] * dist[ia] * verts[ia][1];
        let ax = aniso[2] * dist[ia] * verts[ia][2];
        let bz = aniso[0] * dist[ib] * verts[ib][0];
        let by = aniso[1] * dist[ib] * verts[ib][1];
        let bx = aniso[2] * dist[ib] * verts[ib][2];
        let cz = aniso[0] * dist[ic] * verts[ic][0];
        let cy = aniso[1] * dist[ic] * verts[ic][1];
        let cx = aniso[2] * dist[ic] * verts[ic][2];
        let pz = bz - az;
        let py = by - ay;
        let px = bx - ax;
        let qz = cz - az;
        let qy = cy - ay;
        let qx = cx - ax;
        let mut nz = px * qy - py * qx;
        let mut ny = pz * qx - px * qz;
        let mut nx = py * qz - pz * qy;
        let normz = 1.0 / ((nz * nz + ny * ny + nx * nx).sqrt() + 1.0e-10);
        nz *= normz;
        ny *= normz;
        nx *= normz;
        let radius = az * nz + ay * ny + ax * nx;
        r_min = r_min.min(radius);
    }
    Ok(r_min)
}

pub fn calculate_poly_offset_gravity(
    dist: &[f32],
    verts: &[[f32; 3]],
    point: &[f32; 3],
) -> Result<[f32; 3], GeometryError> {
    let _ = point;
    if dist.len() != verts.len() {
        return Err(GeometryError::RayCountMismatch);
    }
    let mut pz = 0.0f32;
    let mut py = 0.0f32;
    let mut px = 0.0f32;
    let mut weight = 0.0f32;
    for i in 0..dist.len() {
        let r = dist[i];
        let z = r * verts[i][0];
        let y = r * verts[i][1];
        let x = r * verts[i][2];
        pz += r * z;
        py += r * y;
        px += r * x;
        weight += r;
    }
    Ok([pz / weight, py / weight, px / weight])
}

pub fn bounding_radius_outer_gravity(
    dist: &[f32],
    verts: &[[f32; 3]],
    aniso: &[f32; 3],
    offset: &[f32; 3],
) -> Result<f32, GeometryError> {
    if dist.len() != verts.len() {
        return Err(GeometryError::RayCountMismatch);
    }
    let mut r_squared_max = 0.0f32;
    for i in 0..dist.len() {
        let r = dist[i];
        let z = aniso[0] * (r * verts[i][0] - offset[0]);
        let y = aniso[1] * (r * verts[i][1] - offset[1]);
        let x = aniso[2] * (r * verts[i][2] - offset[2]);
        let r_squared = z * z + y * y + x * x;
        r_squared_max = r_squared_max.max(r_squared);
    }
    Ok(r_squared_max.sqrt())
}

pub fn bounding_radius_inner_gravity(
    dist: &[f32],
    verts: &[[f32; 3]],
    faces: &[[usize; 3]],
    aniso: &[f32; 3],
    offset: &[f32; 3],
) -> Result<f32, GeometryError> {
    if dist.len() != verts.len() {
        return Err(GeometryError::RayCountMismatch);
    }
    if faces
        .iter()
        .any(|face| face.iter().any(|i| *i >= dist.len()))
    {
        return Err(GeometryError::RayCountMismatch);
    }
    let mut r_min = f32::INFINITY;
    let pz = aniso[0] * offset[0];
    let py = aniso[1] * offset[1];
    let px = aniso[2] * offset[2];
    for face in faces {
        let ia = face[0];
        let ib = face[1];
        let ic = face[2];
        let az = aniso[0] * dist[ia] * verts[ia][0];
        let ay = aniso[1] * dist[ia] * verts[ia][1];
        let ax = aniso[2] * dist[ia] * verts[ia][2];
        let bz = aniso[0] * dist[ib] * verts[ib][0];
        let by = aniso[1] * dist[ib] * verts[ib][1];
        let bx = aniso[2] * dist[ib] * verts[ib][2];
        let cz = aniso[0] * dist[ic] * verts[ic][0];
        let cy = aniso[1] * dist[ic] * verts[ic][1];
        let cx = aniso[2] * dist[ic] * verts[ic][2];
        let abz = bz - az;
        let aby = by - ay;
        let abx = bx - ax;
        let acz = cz - az;
        let acy = cy - ay;
        let acx = cx - ax;
        let apz = pz - az;
        let apy = py - ay;
        let apx = px - ax;
        let d1 = abz * apz + aby * apy + abx * apx;
        let d2 = acz * apz + acy * apy + acx * apx;
        if d1 <= 0.0 && d2 <= 0.0 {
            let radius = ((az - pz).powi(2) + (ay - py).powi(2) + (ax - px).powi(2)).sqrt();
            r_min = r_min.min(radius);
            continue;
        }

        let bpz = pz - bz;
        let bpy = py - by;
        let bpx = px - bx;
        let d3 = abz * bpz + aby * bpy + abx * bpx;
        let d4 = acz * bpz + acy * bpy + acx * bpx;
        if d3 >= 0.0 && d4 <= d3 {
            let radius = ((bz - pz).powi(2) + (by - py).powi(2) + (bx - px).powi(2)).sqrt();
            r_min = r_min.min(radius);
            continue;
        }

        let cpz = pz - cz;
        let cpy = py - cy;
        let cpx = px - cx;
        let d5 = abz * cpz + aby * cpy + abx * cpx;
        let d6 = acz * cpz + acy * cpy + acx * cpx;
        if d6 >= 0.0 && d5 <= d6 {
            let radius = ((cz - pz).powi(2) + (cy - py).powi(2) + (cx - px).powi(2)).sqrt();
            r_min = r_min.min(radius);
            continue;
        }

        let vc = d1 * d4 - d3 * d2;
        if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
            let v = d1 / (d1 - d3);
            let sz = az + v * abz;
            let sy = ay + v * aby;
            let sx = ax + v * abx;
            let radius = ((sz - pz).powi(2) + (sy - py).powi(2) + (sx - px).powi(2)).sqrt();
            r_min = r_min.min(radius);
            continue;
        }

        let vb = d5 * d2 - d1 * d6;
        if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
            let v = d2 / (d2 - d6);
            let sz = az + v * acz;
            let sy = ay + v * acy;
            let sx = ax + v * acx;
            let radius = ((sz - pz).powi(2) + (sy - py).powi(2) + (sx - px).powi(2)).sqrt();
            r_min = r_min.min(radius);
            continue;
        }

        let va = d3 * d6 - d5 * d4;
        if va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0 {
            let v = (d4 - d3) / ((d4 - d3) + (d5 - d6));
            let sz = bz + v * (cz - bz);
            let sy = by + v * (cy - by);
            let sx = bx + v * (cx - bx);
            let radius = ((sz - pz).powi(2) + (sy - py).powi(2) + (sx - px).powi(2)).sqrt();
            r_min = r_min.min(radius);
            continue;
        }

        let denom = 1.0 / (va + vb + vc);
        let v = vb * denom;
        let w = vc * denom;
        let sz = az + v * abz + w * acz;
        let sy = ay + v * aby + w * acy;
        let sx = ax + v * abx + w * acx;
        let radius = ((sz - pz).powi(2) + (sy - py).powi(2) + (sx - px).powi(2)).sqrt();
        r_min = r_min.min(radius);
    }
    Ok(r_min)
}

pub fn intersect_sphere_gravity(
    r1: f32,
    p1: &[f32; 3],
    offset1: &[f32; 3],
    r2: f32,
    p2: &[f32; 3],
    offset2: &[f32; 3],
    anisotropy: &[f32; 3],
) -> f32 {
    let dz = anisotropy[0] * (p1[0] - p2[0] + offset1[0] - offset2[0]);
    let dy = anisotropy[1] * (p1[1] - p2[1] + offset1[1] - offset2[1]);
    let dx = anisotropy[2] * (p1[2] - p2[2] + offset1[2] - offset2[2]);
    let d = (dz * dz + dy * dy + dx * dx).sqrt();
    let rmin = r1.min(r2);
    let rmax = r1.max(r2);
    if d > r1 + r2 {
        return 0.0;
    }
    if rmax >= d + rmin - 1.0e-10 {
        return PI * 4.0 / 3.0 * rmin * rmin * rmin;
    }
    let t = (r1 + r2 - d) / 2.0 / d;
    let h1 = (r2 - r1 + d) * t;
    let h2 = (r1 - r2 + d) * t;
    let v1 = PI / 3.0 * h1 * h1 * (3.0 * r1 - h1);
    let v2 = PI / 3.0 * h2 * h2 * (3.0 * r2 - h2);
    (v1 + v2) / (anisotropy[0] * anisotropy[1] * anisotropy[2])
}

pub fn intersect_sphere(r1: f32, p1: &[f32; 3], r2: f32, p2: &[f32; 3]) -> f32 {
    let dz = p1[0] - p2[0];
    let dy = p1[1] - p2[1];
    let dx = p1[2] - p2[2];
    let d = (dz * dz + dy * dy + dx * dx).sqrt();
    if d > r1 + r2 {
        return 0.0;
    }
    let rmin = r1.min(r2);
    let rmax = r1.max(r2);
    if rmax > d + rmin {
        return PI * 4.0 / 3.0 * rmin * rmin * rmin;
    }
    let t = (r1 + r2 - d) / 2.0 / d;
    let h1 = (r2 - r1 + d) * t;
    let h2 = (r1 - r2 + d) * t;
    let v1 = PI / 3.0 * h1 * h1 * (3.0 * r1 - h1);
    let v2 = PI / 3.0 * h2 * h2 * (3.0 * r2 - h2);
    v1 + v2
}

pub fn intersect_sphere_isotropic(
    r1: f32,
    p1: &[f32; 3],
    r2: f32,
    p2: &[f32; 3],
    anisotropy: &[f32; 3],
) -> f32 {
    let dz = anisotropy[0] * (p1[0] - p2[0]);
    let dy = anisotropy[1] * (p1[1] - p2[1]);
    let dx = anisotropy[2] * (p1[2] - p2[2]);
    let d = (dz * dz + dy * dy + dx * dx).sqrt();
    let rmin = r1.min(r2);
    let rmax = r1.max(r2);
    if d > r1 + r2 {
        return 0.0;
    }
    if rmax >= d + rmin - 1.0e-10 {
        return PI * 4.0 / 3.0 * rmin * rmin * rmin;
    }
    let t = (r1 + r2 - d) / 2.0 / d;
    let h1 = (r2 - r1 + d) * t;
    let h2 = (r1 - r2 + d) * t;
    let v1 = PI / 3.0 * h1 * h1 * (3.0 * r1 - h1);
    let v2 = PI / 3.0 * h2 * h2 * (3.0 * r2 - h2);
    (v1 + v2) / (anisotropy[0] * anisotropy[1] * anisotropy[2])
}

pub fn intersect_bbox(box1: &[isize; 6], box2: &[isize; 6]) -> f32 {
    let wz = 0isize.max(box1[1].min(box2[1]) - box1[0].max(box2[0])) as f32;
    let wy = 0isize.max(box1[3].min(box2[3]) - box1[2].max(box2[2])) as f32;
    let wx = 0isize.max(box1[5].min(box2[5]) - box1[4].max(box2[4])) as f32;
    wx * wy * wz
}

pub fn polyhedron_bbox(
    dist: &[f32],
    center: &[f32; 3],
    verts: &[[f32; 3]],
) -> Result<[isize; 6], GeometryError> {
    if dist.len() != verts.len() {
        return Err(GeometryError::RayCountMismatch);
    }
    let mut z1 = isize::MAX;
    let mut z2 = -1isize;
    let mut y1 = isize::MAX;
    let mut y2 = -1isize;
    let mut x1 = isize::MAX;
    let mut x2 = -1isize;
    for j in 0..dist.len() {
        let z = center[0] + dist[j] * verts[j][0];
        let y = center[1] + dist[j] * verts[j][1];
        let x = center[2] + dist[j] * verts[j][2];
        z1 = z1.min(z.round() as isize);
        z2 = z2.max(z.round() as isize);
        y1 = y1.min(y.round() as isize);
        y2 = y2.max(y.round() as isize);
        x1 = x1.min(x.round() as isize);
        x2 = x2.max(x.round() as isize);
    }
    Ok([z1, z2, y1, y2, x1, x2])
}

pub fn polyhedron_polyverts(
    dist: &[f32],
    center: &[f32; 3],
    verts: &[[f32; 3]],
) -> Result<Vec<[f32; 3]>, GeometryError> {
    if dist.len() != verts.len() {
        return Err(GeometryError::RayCountMismatch);
    }
    let mut polyverts = Vec::with_capacity(dist.len());
    for j in 0..dist.len() {
        let z = center[0] + dist[j] * verts[j][0];
        let y = center[1] + dist[j] * verts[j][1];
        let x = center[2] + dist[j] * verts[j][2];
        polyverts.push([z, y, x]);
    }
    Ok(polyverts)
}

pub fn render_polyhedron(
    dist: &[f32],
    center: &[f32; 3],
    bbox: &[isize; 6],
    polyverts: &[[f32; 3]],
    faces: &[[usize; 3]],
    nz: usize,
    ny: usize,
    nx: usize,
) -> Result<Vec<bool>, GeometryError> {
    if dist.len() != polyverts.len() {
        return Err(GeometryError::RayCountMismatch);
    }
    if faces
        .iter()
        .any(|face| face.iter().any(|i| *i >= dist.len()))
    {
        return Err(GeometryError::RayCountMismatch);
    }
    let mut rendered = vec![false; nz * ny * nx];
    for z in 0..nz {
        for y in 0..ny {
            for x in 0..nx {
                rendered[x + y * nx + z * nx * ny] = inside_polyhedron(
                    z as f32 + bbox[0] as f32,
                    y as f32 + bbox[2] as f32,
                    x as f32 + bbox[4] as f32,
                    center,
                    polyverts,
                    faces,
                );
            }
        }
    }
    Ok(rendered)
}

pub fn overlap_render_polyhedron(
    dist: &[f32],
    center: &[f32; 3],
    bbox: &[isize; 6],
    polyverts: &[[f32; 3]],
    faces: &[[usize; 3]],
    rendered: &[bool],
    nz: usize,
    ny: usize,
    nx: usize,
    overlap_maximal: f32,
) -> Result<usize, GeometryError> {
    if dist.len() != polyverts.len() || rendered.len() != nz * ny * nx {
        return Err(GeometryError::RayCountMismatch);
    }
    if faces
        .iter()
        .any(|face| face.iter().any(|i| *i >= dist.len()))
    {
        return Err(GeometryError::RayCountMismatch);
    }
    let mut res = 0usize;
    for z in 0..nz {
        for y in 0..ny {
            for x in 0..nx {
                let pz = z as f32 + bbox[0] as f32;
                let py = y as f32 + bbox[2] as f32;
                let px = x as f32 + bbox[4] as f32;
                if rendered[x + y * nx + z * nx * ny]
                    && inside_polyhedron(pz, py, px, center, polyverts, faces)
                {
                    res += 1;
                }
                if (res as f32) > overlap_maximal {
                    return Ok(res);
                }
            }
        }
    }
    Ok(res)
}

pub fn overlap_render_polyhedron_kernel(
    dist: &[f32],
    center: &[f32; 3],
    bbox: &[isize; 6],
    polyverts: &[[f32; 3]],
    faces: &[[usize; 3]],
    rendered: &[bool],
    nz: usize,
    ny: usize,
    nx: usize,
) -> Result<usize, GeometryError> {
    if dist.len() != polyverts.len() || rendered.len() != nz * ny * nx {
        return Err(GeometryError::RayCountMismatch);
    }
    if faces
        .iter()
        .any(|face| face.iter().any(|i| *i >= dist.len()))
    {
        return Err(GeometryError::RayCountMismatch);
    }
    let mut res = 0usize;
    for z in 0..nz {
        for y in 0..ny {
            for x in 0..nx {
                let pz = z as f32 + bbox[0] as f32;
                let py = y as f32 + bbox[2] as f32;
                let px = x as f32 + bbox[4] as f32;
                if rendered[x + y * nx + z * nx * ny]
                    && inside_polyhedron_kernel(pz, py, px, center, polyverts, faces)
                {
                    res += 1;
                }
            }
        }
    }
    Ok(res)
}

pub fn build_halfspace(a: &[f32; 3], b: &[f32; 3], c: &[f32; 3]) -> [f64; 4] {
    let az = a[0];
    let ay = a[1];
    let ax = a[2];
    let bz = b[0];
    let by = b[1];
    let bx = b[2];
    let cz = c[0];
    let cy = c[1];
    let cx = c[2];
    let pz = bz - az;
    let py = by - ay;
    let px = bx - ax;
    let qz = cz - az;
    let qy = cy - ay;
    let qx = cx - ax;
    let nz = -(py * qx - px * qy);
    let ny = -(px * qz - pz * qx);
    let nx = -(pz * qy - py * qz);
    [
        nz as f64,
        ny as f64,
        nx as f64,
        -(az * nz + ay * ny + ax * nx) as f64,
    ]
}

pub fn halfspaces_convex(polyverts: &[[f32; 3]]) -> Vec<[f64; 4]> {
    let n_rays = polyverts.len();
    let eps = 1.0e-5f64;
    let mut halfspaces = Vec::new();
    for ia in 0..n_rays.saturating_sub(2) {
        for ib in ia + 1..n_rays.saturating_sub(1) {
            for ic in ib + 1..n_rays {
                let hs = build_halfspace(&polyverts[ia], &polyverts[ib], &polyverts[ic]);
                let norm = (hs[0] * hs[0] + hs[1] * hs[1] + hs[2] * hs[2]).sqrt();
                if norm <= 1.0e-12 {
                    continue;
                }
                let mut pos = false;
                let mut neg = false;
                for (ip, p) in polyverts.iter().enumerate() {
                    if ip == ia || ip == ib || ip == ic {
                        continue;
                    }
                    let side =
                        hs[0] * p[0] as f64 + hs[1] * p[1] as f64 + hs[2] * p[2] as f64 + hs[3];
                    if side > eps {
                        pos = true;
                    } else if side < -eps {
                        neg = true;
                    }
                    if pos && neg {
                        break;
                    }
                }
                if !(pos && neg) {
                    let mut normalized = if pos {
                        [-hs[0] / norm, -hs[1] / norm, -hs[2] / norm, -hs[3] / norm]
                    } else {
                        [hs[0] / norm, hs[1] / norm, hs[2] / norm, hs[3] / norm]
                    };
                    if normalized[0].abs() < 1.0e-12 {
                        normalized[0] = 0.0;
                    }
                    if normalized[1].abs() < 1.0e-12 {
                        normalized[1] = 0.0;
                    }
                    if normalized[2].abs() < 1.0e-12 {
                        normalized[2] = 0.0;
                    }
                    if normalized[3].abs() < 1.0e-12 {
                        normalized[3] = 0.0;
                    }
                    if !halfspaces.iter().any(|other: &[f64; 4]| {
                        (other[0] - normalized[0]).abs() < 1.0e-7
                            && (other[1] - normalized[1]).abs() < 1.0e-7
                            && (other[2] - normalized[2]).abs() < 1.0e-7
                            && (other[3] - normalized[3]).abs() < 1.0e-7
                    }) {
                        halfspaces.push(normalized);
                    };
                }
            }
        }
    }
    halfspaces
}

pub fn halfspaces_kernel(polyverts: &[[f32; 3]], faces: &[[usize; 3]]) -> Vec<[f64; 4]> {
    let mut halfspaces = Vec::with_capacity(faces.len());
    for face in faces {
        let ia = face[0];
        let ib = face[1];
        let ic = face[2];
        halfspaces.push(build_halfspace(
            &polyverts[ia],
            &polyverts[ib],
            &polyverts[ic],
        ));
    }
    halfspaces
}

pub fn point_in_halfspaces(z: f32, y: f32, x: f32, halfspaces: &[[f64; 4]]) -> bool {
    for hs in halfspaces {
        if hs[0] * z as f64 + hs[1] * y as f64 + hs[2] * x as f64 + hs[3] > 0.0 {
            return false;
        }
    }
    true
}

pub fn qhull_volume_halfspace_intersection(
    halfspaces: &[[f64; 4]],
    interior_point: &[f64; 3],
    err_value: f32,
) -> f32 {
    if halfspaces.len() < 4 {
        return err_value;
    }

    let mut active_halfspaces = Vec::<[f64; 4]>::new();
    for hs in halfspaces {
        let norm = (hs[0] * hs[0] + hs[1] * hs[1] + hs[2] * hs[2]).sqrt();
        if norm <= 1.0e-12 {
            continue;
        }
        let normalized = [hs[0] / norm, hs[1] / norm, hs[2] / norm, hs[3] / norm];
        if !active_halfspaces.iter().any(|other| {
            (other[0] - normalized[0]).abs() < 1.0e-7
                && (other[1] - normalized[1]).abs() < 1.0e-7
                && (other[2] - normalized[2]).abs() < 1.0e-7
                && (other[3] - normalized[3]).abs() < 1.0e-7
        }) {
            active_halfspaces.push(normalized);
        }
    }
    if active_halfspaces.len() < 4 {
        return err_value;
    }

    for hs in &active_halfspaces {
        if hs[0] * interior_point[0] + hs[1] * interior_point[1] + hs[2] * interior_point[2] + hs[3]
            > 1.0e-7
        {
            return err_value;
        }
    }

    let mut intersections = Vec::<[f64; 3]>::new();
    for i in 0..active_halfspaces.len().saturating_sub(2) {
        for j in i + 1..active_halfspaces.len().saturating_sub(1) {
            for k in j + 1..active_halfspaces.len() {
                let a = [
                    [
                        active_halfspaces[i][0],
                        active_halfspaces[i][1],
                        active_halfspaces[i][2],
                    ],
                    [
                        active_halfspaces[j][0],
                        active_halfspaces[j][1],
                        active_halfspaces[j][2],
                    ],
                    [
                        active_halfspaces[k][0],
                        active_halfspaces[k][1],
                        active_halfspaces[k][2],
                    ],
                ];
                let b = [
                    -active_halfspaces[i][3],
                    -active_halfspaces[j][3],
                    -active_halfspaces[k][3],
                ];
                let det = a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1])
                    - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
                    + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0]);
                if det.abs() <= 1.0e-10 {
                    continue;
                }
                let dz = b[0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1])
                    - a[0][1] * (b[1] * a[2][2] - a[1][2] * b[2])
                    + a[0][2] * (b[1] * a[2][1] - a[1][1] * b[2]);
                let dy = a[0][0] * (b[1] * a[2][2] - a[1][2] * b[2])
                    - b[0] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
                    + a[0][2] * (a[1][0] * b[2] - b[1] * a[2][0]);
                let dx = a[0][0] * (a[1][1] * b[2] - b[1] * a[2][1])
                    - a[0][1] * (a[1][0] * b[2] - b[1] * a[2][0])
                    + b[0] * (a[1][0] * a[2][1] - a[1][1] * a[2][0]);
                let p = [dz / det, dy / det, dx / det];
                let mut feasible = true;
                for hs in &active_halfspaces {
                    if hs[0] * p[0] + hs[1] * p[1] + hs[2] * p[2] + hs[3] > 1.0e-7 {
                        feasible = false;
                        break;
                    }
                }
                if feasible
                    && !intersections.iter().any(|q| {
                        (q[0] - p[0]).abs() < 1.0e-6
                            && (q[1] - p[1]).abs() < 1.0e-6
                            && (q[2] - p[2]).abs() < 1.0e-6
                    })
                {
                    intersections.push(p);
                }
            }
        }
    }

    if intersections.len() < 4 {
        return err_value;
    }

    let mut vol = 0.0f64;
    for hs in &active_halfspaces {
        let mut face_points = Vec::<[f64; 3]>::new();
        for p in &intersections {
            let side = hs[0] * p[0] + hs[1] * p[1] + hs[2] * p[2] + hs[3];
            if side.abs() <= 1.0e-6 {
                face_points.push(*p);
            }
        }
        if face_points.len() < 3 {
            continue;
        }

        let mut center = [0.0f64; 3];
        for p in &face_points {
            center[0] += p[0] / face_points.len() as f64;
            center[1] += p[1] / face_points.len() as f64;
            center[2] += p[2] / face_points.len() as f64;
        }

        let normal_norm = (hs[0] * hs[0] + hs[1] * hs[1] + hs[2] * hs[2]).sqrt();
        if normal_norm <= 1.0e-12 {
            continue;
        }
        let normal = [
            hs[0] / normal_norm,
            hs[1] / normal_norm,
            hs[2] / normal_norm,
        ];
        let axis = if normal[0].abs() <= normal[1].abs() && normal[0].abs() <= normal[2].abs() {
            [1.0, 0.0, 0.0]
        } else if normal[1].abs() <= normal[2].abs() {
            [0.0, 1.0, 0.0]
        } else {
            [0.0, 0.0, 1.0]
        };
        let mut u = [
            normal[1] * axis[2] - normal[2] * axis[1],
            normal[2] * axis[0] - normal[0] * axis[2],
            normal[0] * axis[1] - normal[1] * axis[0],
        ];
        let u_norm = (u[0] * u[0] + u[1] * u[1] + u[2] * u[2]).sqrt();
        u[0] /= u_norm;
        u[1] /= u_norm;
        u[2] /= u_norm;
        let v = [
            normal[1] * u[2] - normal[2] * u[1],
            normal[2] * u[0] - normal[0] * u[2],
            normal[0] * u[1] - normal[1] * u[0],
        ];

        face_points.sort_by(|a, b| {
            let da = [a[0] - center[0], a[1] - center[1], a[2] - center[2]];
            let db = [b[0] - center[0], b[1] - center[1], b[2] - center[2]];
            let aa = (da[0] * v[0] + da[1] * v[1] + da[2] * v[2])
                .atan2(da[0] * u[0] + da[1] * u[1] + da[2] * u[2]);
            let ab = (db[0] * v[0] + db[1] * v[1] + db[2] * v[2])
                .atan2(db[0] * u[0] + db[1] * u[1] + db[2] * u[2]);
            aa.partial_cmp(&ab).unwrap_or(std::cmp::Ordering::Equal)
        });

        for i in 0..face_points.len() {
            let mut b = face_points[i];
            let mut c = face_points[(i + 1) % face_points.len()];
            let cb = [b[0] - center[0], b[1] - center[1], b[2] - center[2]];
            let cc = [c[0] - center[0], c[1] - center[1], c[2] - center[2]];
            let tri_normal = [
                cb[1] * cc[2] - cb[2] * cc[1],
                cb[2] * cc[0] - cb[0] * cc[2],
                cb[0] * cc[1] - cb[1] * cc[0],
            ];
            if tri_normal[0] * normal[0] + tri_normal[1] * normal[1] + tri_normal[2] * normal[2]
                < 0.0
            {
                std::mem::swap(&mut b, &mut c);
            }
            let a = center;
            let det = a[0] * (b[1] * c[2] - b[2] * c[1]) - a[1] * (b[0] * c[2] - b[2] * c[0])
                + a[2] * (b[0] * c[1] - b[1] * c[0]);
            vol += det / 6.0;
        }
    }
    vol.abs() as f32
}

pub fn qhull_overlap_kernel(
    polyverts1: &[[f32; 3]],
    center1: &[f32; 3],
    polyverts2: &[[f32; 3]],
    center2: &[f32; 3],
    faces: &[[usize; 3]],
    n_step: usize,
) -> f32 {
    let mut halfspaces = Vec::new();
    let step = n_step.max(1);
    for i in (0..faces.len()).step_by(step) {
        let ia = faces[i][0];
        let ib = faces[i][1];
        let ic = faces[i][2];
        halfspaces.push(build_halfspace(
            &polyverts1[ia],
            &polyverts1[ib],
            &polyverts1[ic],
        ));
        halfspaces.push(build_halfspace(
            &polyverts2[ia],
            &polyverts2[ib],
            &polyverts2[ic],
        ));
    }

    let interior_point = [
        0.5 * (center1[0] as f64 + center2[0] as f64),
        0.5 * (center1[1] as f64 + center2[1] as f64),
        0.5 * (center1[2] as f64 + center2[2] as f64),
    ];
    qhull_volume_halfspace_intersection(&halfspaces, &interior_point, 0.0)
}

pub fn qhull_overlap_convex_hulls(
    polyverts1: &[[f32; 3]],
    center1: &[f32; 3],
    polyverts2: &[[f32; 3]],
    center2: &[f32; 3],
    faces: &[[usize; 3]],
) -> f32 {
    let _ = faces;
    let mut halfspaces = halfspaces_convex(polyverts1);
    halfspaces.extend(halfspaces_convex(polyverts2));
    let interior_point = [
        0.5 * (center1[0] as f64 + center2[0] as f64),
        0.5 * (center1[1] as f64 + center2[1] as f64),
        0.5 * (center1[2] as f64 + center2[2] as f64),
    ];
    qhull_volume_halfspace_intersection(&halfspaces, &interior_point, 1.0e10)
}

pub fn dist_to_volume(
    dist: &[f32],
    shape: [usize; 3],
    rays: &Rays,
) -> Result<Array3<f32>, GeometryError> {
    let n_rays = rays.vertices.len();
    if n_rays == 0 || dist.len() != shape[0] * shape[1] * shape[2] * n_rays {
        return Err(GeometryError::DistShapeMismatch);
    }
    if rays
        .faces
        .iter()
        .any(|face| face.iter().any(|i| *i >= n_rays))
    {
        return Err(GeometryError::RayCountMismatch);
    }

    let mut result = Array3::<f32>::zeros((shape[0], shape[1], shape[2]));
    for k in 0..shape[0] {
        for j in 0..shape[1] {
            for i in 0..shape[2] {
                let ind = n_rays * (i + shape[2] * (j + k * shape[1]));
                let curr_dist = &dist[ind..ind + n_rays];
                result[[k, j, i]] = polyhedron_volume(curr_dist, &rays.vertices, &rays.faces)?;
            }
        }
    }
    Ok(result)
}

pub fn dist_to_centroid(
    dist: &[f32],
    shape: [usize; 3],
    rays: &Rays,
    mode: CentroidMode,
) -> Result<Array4<f32>, GeometryError> {
    let n_rays = rays.vertices.len();
    if n_rays == 0 || dist.len() != shape[0] * shape[1] * shape[2] * n_rays {
        return Err(GeometryError::DistShapeMismatch);
    }
    if rays
        .faces
        .iter()
        .any(|face| face.iter().any(|i| *i >= n_rays))
    {
        return Err(GeometryError::RayCountMismatch);
    }

    let mut result = Array4::<f32>::zeros((shape[0], shape[1], shape[2], 3));
    for k in 0..shape[0] {
        for j in 0..shape[1] {
            for i in 0..shape[2] {
                let ind = n_rays * (i + shape[2] * (j + k * shape[1]));
                let curr_dist = &dist[ind..ind + n_rays];
                let centroid = polyhedron_centroid(curr_dist, &rays.vertices, &rays.faces)?;
                let absolute = mode == CentroidMode::Absolute;
                result[[k, j, i, 0]] = centroid[0] + if absolute { k as f32 } else { 0.0 };
                result[[k, j, i, 1]] = centroid[1] + if absolute { j as f32 } else { 0.0 };
                result[[k, j, i, 2]] = centroid[2] + if absolute { i as f32 } else { 0.0 };
            }
        }
    }
    Ok(result)
}

pub fn relabel_image_stardist3d(
    lbl: &[u16],
    shape: [usize; 3],
    rays: &Rays,
    grid: [usize; 3],
    verbose: bool,
    mode: PolyhedronRenderMode,
    overlap_label: Option<u32>,
) -> Result<Array3<u32>, GeometryError> {
    let _ = verbose;
    if lbl.len() != shape[0] * shape[1] * shape[2] {
        return Err(GeometryError::ShapeMismatch);
    }
    let dist_all = star_dist3d(lbl, shape, rays, grid)?;
    let mut max_label = 0u16;
    for value in lbl {
        max_label = max_label.max(*value);
    }

    let mut count = vec![0usize; max_label as usize + 1];
    let mut sum_z = vec![0usize; max_label as usize + 1];
    let mut sum_y = vec![0usize; max_label as usize + 1];
    let mut sum_x = vec![0usize; max_label as usize + 1];
    for z in 0..shape[0] {
        for y in 0..shape[1] {
            for x in 0..shape[2] {
                let value = lbl[(z * shape[1] + y) * shape[2] + x] as usize;
                if value > 0 {
                    count[value] += 1;
                    sum_z[value] += z;
                    sum_y[value] += y;
                    sum_x[value] += x;
                }
            }
        }
    }

    let n_rays = rays.vertices.len();
    let mut points = Vec::<[f32; 3]>::new();
    let mut labs = Vec::<u32>::new();
    let mut dist = Vec::<f32>::new();
    for label in 1..=max_label as usize {
        if count[label] == 0 {
            continue;
        }
        let z = sum_z[label] / count[label];
        let y = sum_y[label] / count[label];
        let x = sum_x[label] / count[label];
        points.push([z as f32, y as f32, x as f32]);
        labs.push(label as u32);
        for n in 0..n_rays {
            dist.push(dist_all[[z, y, x, n]].max(1.0e-3));
        }
    }

    polyhedron_to_label(
        &dist,
        &points,
        rays,
        shape,
        None,
        f32::NEG_INFINITY,
        Some(&labs),
        mode,
        overlap_label,
    )
}

pub fn export_to_obj_file3d(
    dist: &[f32],
    points: &[[f32; 3]],
    rays_vertices: &[[f32; 3]],
    rays_faces: &[[usize; 3]],
    fname: Option<&Path>,
    scale: [f32; 3],
    single_mesh: bool,
    uv_map: bool,
    name: &str,
) -> Result<String, GeometryError> {
    let n_rays = rays_vertices.len();
    if n_rays == 0
        || dist.len() != points.len() * n_rays
        || rays_faces
            .iter()
            .any(|face| face.iter().any(|i| *i >= n_rays))
    {
        return Err(GeometryError::ObjShapeMismatch);
    }

    let coord = dist_to_coord3d(dist, points, rays_vertices)?;
    let min_scale = scale[0].min(scale[1]).min(scale[2]);
    let decimals = (1.0f32.max(1.0 - min_scale.log10())) as usize;
    let mut scaled_verts = Vec::with_capacity(rays_vertices.len());
    for vertex in rays_vertices {
        let mut scaled = [
            scale[0] * vertex[0],
            scale[1] * vertex[1],
            scale[2] * vertex[2],
        ];
        let norm = (scaled[0] * scaled[0] + scaled[1] * scaled[1] + scaled[2] * scaled[2]).sqrt();
        scaled[0] /= norm;
        scaled[1] /= norm;
        scaled[2] /= norm;
        scaled_verts.push(scaled);
    }

    let mut obj_str = String::new();
    let mut face_offset = 1usize;
    for i in 0..points.len() {
        if i == 0 || !single_mesh {
            obj_str.push_str(&format!("o {name}_{i}\n"));
        }

        for n in 0..n_rays {
            let z = coord[[i, n, 0]] * scale[0];
            let y = coord[[i, n, 1]] * scale[1];
            let x = coord[[i, n, 2]] * scale[2];
            obj_str.push_str(&format!("v {x:.decimals$} {y:.decimals$} {z:.decimals$}\n"));
        }

        if uv_map {
            for vertex in &scaled_verts {
                let vz = vertex[0];
                let vy = vertex[1];
                let vx = vertex[2];
                let u = 1.0 - (0.5 + 0.5 * vz.atan2(vx) / PI);
                let v = 1.0 - (0.5 - vy.asin() / PI);
                obj_str.push_str(&format!("vt {u:.4} {v:.4}\n"));
            }
        }

        for face in rays_faces {
            let a = face[0] + face_offset;
            let b = face[1] + face_offset;
            let c = face[2] + face_offset;
            obj_str.push_str(&format!("f {a}/{a} {b}/{b} {c}/{c}\n"));
        }

        face_offset += n_rays;
    }

    if let Some(fname) = fname {
        std::fs::write(fname, &obj_str).map_err(|_| GeometryError::ObjWriteFailed)?;
    }

    Ok(obj_str)
}

pub fn polyhedron_to_label(
    dist: &[f32],
    points: &[[f32; 3]],
    rays: &Rays,
    shape: [usize; 3],
    prob: Option<&[f32]>,
    thr: f32,
    labels: Option<&[u32]>,
    mode: PolyhedronRenderMode,
    overlap_label: Option<u32>,
) -> Result<Array3<u32>, GeometryError> {
    if points.is_empty() {
        return Ok(Array3::<u32>::zeros((shape[0], shape[1], shape[2])));
    }
    let n_rays = rays.vertices.len();
    if n_rays == 0 || dist.len() != points.len() * n_rays {
        return Err(GeometryError::DistShapeMismatch);
    }
    if dist.iter().any(|v| *v <= 0.0) {
        return Err(GeometryError::NonPositiveDistance);
    }
    if rays
        .faces
        .iter()
        .any(|face| face.iter().any(|i| *i >= n_rays))
    {
        return Err(GeometryError::RayCountMismatch);
    }
    if let Some(prob) = prob {
        if prob.len() != points.len() {
            return Err(GeometryError::ProbShapeMismatch);
        }
    }
    if let Some(labels) = labels {
        if labels.len() != points.len() {
            return Err(GeometryError::LabelsShapeMismatch);
        }
    }

    let mut ind = Vec::new();
    for i in 0..points.len() {
        let p = prob.map(|prob| prob[i]).unwrap_or(1.0);
        if p >= thr {
            ind.push(i);
        }
    }
    if ind.is_empty() {
        return Ok(Array3::<u32>::zeros((shape[0], shape[1], shape[2])));
    }

    ind.sort_by(|&a, &b| {
        let pa = prob.map(|prob| prob[a]).unwrap_or(1.0);
        let pb = prob.map(|prob| prob[b]).unwrap_or(1.0);
        pb.partial_cmp(&pa)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.cmp(&b))
    });

    let mut result = Array3::<u32>::zeros((shape[0], shape[1], shape[2]));
    for i in ind {
        let curr_dist = &dist[i * n_rays..(i + 1) * n_rays];
        let curr_center = points[i];
        let label = labels.map(|labels| labels[i]).unwrap_or((i + 1) as u32);
        let polyverts = polyhedron_polyverts(curr_dist, &curr_center, &rays.vertices)?;
        let bbox = polyhedron_bbox(curr_dist, &curr_center, &rays.vertices)?;
        let hs_convex = halfspaces_convex(&polyverts);
        let hs_kernel = halfspaces_kernel(&polyverts, &rays.faces);

        let z0 = bbox[0].max(0) as usize;
        let z1 = bbox[1].min(shape[0].saturating_sub(1) as isize);
        let y0 = bbox[2].max(0) as usize;
        let y1 = bbox[3].min(shape[1].saturating_sub(1) as isize);
        let x0 = bbox[4].max(0) as usize;
        let x1 = bbox[5].min(shape[2].saturating_sub(1) as isize);
        if z1 < z0 as isize || y1 < y0 as isize || x1 < x0 as isize {
            continue;
        }

        for z in z0..=z1 as usize {
            for y in y0..=y1 as usize {
                for x in x0..=x1 as usize {
                    let inside = match mode {
                        PolyhedronRenderMode::Bbox => true,
                        PolyhedronRenderMode::Full => {
                            point_in_halfspaces(z as f32, y as f32, x as f32, &hs_kernel)
                                || (point_in_halfspaces(z as f32, y as f32, x as f32, &hs_convex)
                                    && inside_polyhedron(
                                        z as f32,
                                        y as f32,
                                        x as f32,
                                        &curr_center,
                                        &polyverts,
                                        &rays.faces,
                                    ))
                        }
                        PolyhedronRenderMode::Kernel => {
                            point_in_halfspaces(z as f32, y as f32, x as f32, &hs_kernel)
                        }
                        PolyhedronRenderMode::Hull => {
                            point_in_halfspaces(z as f32, y as f32, x as f32, &hs_convex)
                        }
                        PolyhedronRenderMode::Debug => {
                            inside_polyhedron_kernel(
                                z as f32,
                                y as f32,
                                x as f32,
                                &curr_center,
                                &polyverts,
                                &rays.faces,
                            ) && !inside_polyhedron(
                                z as f32,
                                y as f32,
                                x as f32,
                                &curr_center,
                                &polyverts,
                                &rays.faces,
                            )
                        }
                    };

                    if inside {
                        let current = result[[z, y, x]];
                        result[[z, y, x]] = if current == 0 {
                            label
                        } else {
                            overlap_label.unwrap_or(current)
                        };
                    }
                }
            }
        }
    }

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ray_angles_match_stardist_order() {
        let angles = ray_angles(4);
        assert_eq!(angles[0], 0.0);
        assert!((angles[1] - PI / 2.0).abs() < 1e-6);
        assert!((angles[2] - PI).abs() < 1e-6);
        assert!((angles[3] - 3.0 * PI / 2.0).abs() < 1e-6);
    }

    #[test]
    fn star_dist_returns_zero_for_background() {
        let dist = star_dist(&[0; 9], [3, 3], 8, [1, 1]).unwrap();
        assert_eq!(dist.shape(), &[3, 3, 8]);
        assert!(dist.iter().all(|v| *v == 0.0));
    }

    #[test]
    fn star_dist_center_pixel_has_unit_boundary_distance() {
        let labels = [0, 0, 0, 0, 1, 0, 0, 0, 0];
        let dist = star_dist(&labels, [3, 3], 4, [1, 1]).unwrap();
        assert!((dist[[1, 1, 0]] - 0.5).abs() < 1e-6);
        assert!((dist[[1, 1, 1]] - 0.5).abs() < 1e-6);
        assert!((dist[[1, 1, 2]] - 0.5).abs() < 1e-6);
        assert!((dist[[1, 1, 3]] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn star_dist_2d_wrappers_match_native_path() {
        let labels = [1, 1, 0, 1, 1, 0, 0, 0, 0];
        let native = star_dist(&labels, [3, 3], 8, [1, 1]).unwrap();
        assert_eq!(_py_star_dist(&labels, [3, 3], 8, [1, 1]).unwrap(), native);
        assert_eq!(_cpp_star_dist(&labels, [3, 3], 8, [1, 1]).unwrap(), native);
        assert_eq!(_ocl_star_dist(&labels, [3, 3], 8, [1, 1]).unwrap(), native);
    }

    #[test]
    fn py_star_dist_rejects_non_unit_grid_like_python_reference() {
        let err = _py_star_dist(&[1; 9], [3, 3], 8, [2, 1]).unwrap_err();
        assert_eq!(err, GeometryError::UnsupportedPythonGrid2D);
    }

    #[test]
    fn dist_to_coord_matches_stardist_axis_order() {
        let coord = dist_to_coord(&[1.0, 1.0, 1.0, 1.0], &[[10.0, 20.0]], 4, [1.0, 1.0]).unwrap();
        assert!((coord[[0, 0, 0]] - 10.0).abs() < 1e-6);
        assert!((coord[[0, 1, 0]] - 21.0).abs() < 1e-6);
        assert!((coord[[0, 0, 1]] - 11.0).abs() < 1e-6);
        assert!((coord[[0, 1, 1]] - 20.0).abs() < 1e-6);
    }

    #[test]
    fn dist_to_coord_old_matches_grid_and_ray_layout() {
        let coord = _dist_to_coord_old(&[1.0, 1.0, 1.0, 1.0], [1, 1, 1, 4], [2, 4]).unwrap();
        assert_eq!(coord.shape(), &[1, 1, 1, 2, 4]);
        assert!((coord[[0, 0, 0, 0, 0]] - 0.0).abs() < 1e-6);
        assert!((coord[[0, 0, 0, 1, 0]] - 1.0).abs() < 1e-6);
        assert!((coord[[0, 0, 0, 0, 1]] - 1.0).abs() < 1e-6);
        assert!((coord[[0, 0, 0, 1, 1]] - 0.0).abs() < 1e-6);
    }

    #[test]
    fn polygons_to_label_old_sorts_by_increasing_probability_and_thresholds() {
        let n_rays = 4;
        let mut coord = vec![0.0f32; 3 * 3 * 2 * n_rays];
        let points = [[1usize, 1usize], [2usize, 2usize]];
        for point in points {
            let base = (point[0] * 3 + point[1]) * 2 * n_rays;
            coord[base] = point[0] as f32 - 0.5;
            coord[base + 1] = point[0] as f32;
            coord[base + 2] = point[0] as f32 + 0.5;
            coord[base + 3] = point[0] as f32;
            coord[base + n_rays] = point[1] as f32;
            coord[base + n_rays + 1] = point[1] as f32 + 0.5;
            coord[base + n_rays + 2] = point[1] as f32;
            coord[base + n_rays + 3] = point[1] as f32 - 0.5;
        }
        let mut prob = vec![0.0f32; 9];
        prob[4] = 0.8;
        prob[8] = 0.4;
        let labels =
            _polygons_to_label_old(&coord, [3, 3, 2, n_rays], &prob, &points, None, 0.5).unwrap();
        assert_eq!(labels[[1, 1]], 1);
        assert_eq!(labels[[2, 2]], 0);
    }

    #[test]
    fn polygons_to_label_filters_and_sorts_like_stardist() {
        let dist = vec![1.0; 8];
        let points = [[2.0, 2.0], [4.0, 4.0]];
        let prob = [0.9, 0.4];
        let labels =
            polygons_to_label(&dist, &points, [7, 7], Some(&prob), 0.5, [1.0, 1.0]).unwrap();
        assert!(labels.iter().any(|v| *v == 1));
        assert!(!labels.iter().any(|v| *v == 2));
    }

    #[test]
    fn relabel_image_stardist_returns_background_for_empty_label_image() {
        let labels = relabel_image_stardist(&[0; 9], [3, 3], 8, [1, 1]).unwrap();
        assert_eq!(labels.shape(), &[3, 3]);
        assert!(labels.iter().all(|v| *v == 0));
    }

    #[test]
    fn relabel_image_stardist_relabels_regions_with_star_representation() {
        let lbl = [0, 0, 0, 0, 1, 0, 0, 0, 0];
        let labels = relabel_image_stardist(&lbl, [3, 3], 8, [1, 1]).unwrap();
        assert_eq!(labels.shape(), &[3, 3]);
        assert_eq!(labels[[1, 1]], 1);
    }

    #[test]
    fn star_dist3d_returns_zero_for_background() {
        let rays = crate::RaysGoldenSpiral::new(4, None).unwrap().into_rays();
        let dist = star_dist3d(&[0; 27], [3, 3, 3], &rays, [1, 1, 1]).unwrap();
        assert_eq!(dist.shape(), &[3, 3, 3, 4]);
        assert!(dist.iter().all(|v| *v == 0.0));
    }

    #[test]
    fn star_dist3d_center_voxel_matches_cpp_rounding() {
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
        let mut labels = vec![0u16; 27];
        labels[(1 * 3 + 1) * 3 + 1] = 1;
        let dist = star_dist3d(&labels, [3, 3, 3], &rays, [1, 1, 1]).unwrap();
        for n in 0..6 {
            assert!((dist[[1, 1, 1, n]] - 1.0).abs() < 1e-6);
        }
    }

    #[test]
    fn star_dist_3d_wrappers_match_native_path() {
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
        let mut labels = vec![0u16; 27];
        labels[(1 * 3 + 1) * 3 + 1] = 1;
        let native = star_dist3d(&labels, [3, 3, 3], &rays, [1, 1, 1]).unwrap();
        assert_eq!(
            _py_star_dist3d(&labels, [3, 3, 3], &rays, [1, 1, 1]).unwrap(),
            native
        );
        assert_eq!(
            _cpp_star_dist3d(&labels, [3, 3, 3], &rays, [1, 1, 1]).unwrap(),
            native
        );
        assert_eq!(
            _ocl_star_dist3d(&labels, [3, 3, 3], &rays, [1, 1, 1]).unwrap(),
            native
        );
    }

    #[test]
    fn dist_to_coord3d_matches_stardist_broadcast_formula() {
        let coord = dist_to_coord3d(
            &[2.0, 3.0],
            &[[10.0, 20.0, 30.0]],
            &[[1.0, 0.0, 0.0], [0.0, -1.0, 0.5]],
        )
        .unwrap();
        assert_eq!(coord.shape(), &[1, 2, 3]);
        assert!((coord[[0, 0, 0]] - 12.0).abs() < 1e-6);
        assert!((coord[[0, 0, 1]] - 20.0).abs() < 1e-6);
        assert!((coord[[0, 0, 2]] - 30.0).abs() < 1e-6);
        assert!((coord[[0, 1, 0]] - 10.0).abs() < 1e-6);
        assert!((coord[[0, 1, 1]] - 17.0).abs() < 1e-6);
        assert!((coord[[0, 1, 2]] - 31.5).abs() < 1e-6);
    }

    #[test]
    fn relabel_image_stardist3d_preserves_original_region_labels() {
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
            faces: vec![
                [0, 2, 4],
                [0, 5, 2],
                [0, 4, 3],
                [0, 3, 5],
                [1, 4, 2],
                [1, 2, 5],
                [1, 3, 4],
                [1, 5, 3],
            ],
        };
        let mut lbl = vec![0u16; 27];
        lbl[(1 * 3 + 1) * 3 + 1] = 7;
        let labels = relabel_image_stardist3d(
            &lbl,
            [3, 3, 3],
            &rays,
            [1, 1, 1],
            false,
            PolyhedronRenderMode::Bbox,
            None,
        )
        .unwrap();
        assert_eq!(labels.shape(), &[3, 3, 3]);
        assert_eq!(labels[[1, 1, 1]], 7);
    }

    #[test]
    fn export_to_obj_file3d_matches_python_axis_and_face_order() {
        let dist = vec![1.0, 2.0, 3.0, 4.0];
        let points = [[10.0, 20.0, 30.0]];
        let rays_vertices = [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [-1.0, -1.0, -1.0],
        ];
        let rays_faces = [[0, 1, 2], [0, 3, 1], [0, 2, 3], [1, 3, 2]];
        let obj = export_to_obj_file3d(
            &dist,
            &points,
            &rays_vertices,
            &rays_faces,
            None,
            [1.0, 1.0, 1.0],
            true,
            false,
            "poly",
        )
        .unwrap();
        assert!(obj.starts_with("o poly_0\n"));
        assert!(obj.contains("v 30.0 20.0 11.0\n"));
        assert!(obj.contains("v 26.0 16.0 6.0\n"));
        assert!(obj.contains("f 1/1 2/2 3/3\n"));
    }

    #[test]
    fn export_to_obj_file3d_offsets_faces_and_writes_uvs() {
        let dist = vec![1.0; 8];
        let points = [[0.0, 0.0, 0.0], [10.0, 10.0, 10.0]];
        let rays_vertices = [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [-1.0, -1.0, -1.0],
        ];
        let rays_faces = [[0, 1, 2], [0, 3, 1], [0, 2, 3], [1, 3, 2]];
        let obj = export_to_obj_file3d(
            &dist,
            &points,
            &rays_vertices,
            &rays_faces,
            None,
            [1.0, 2.0, 1.0],
            false,
            true,
            "cell",
        )
        .unwrap();
        assert!(obj.contains("o cell_0\n"));
        assert!(obj.contains("o cell_1\n"));
        assert!(obj.contains("vt "));
        assert!(obj.contains("f 5/5 6/6 7/7\n"));
    }

    #[test]
    fn dist_to_volume_matches_rays_volume_spot_check() {
        let rays = crate::RaysGoldenSpiral::new(10, None).unwrap().into_rays();
        let dist = vec![1.0; rays.vertices.len()];
        let volume = dist_to_volume(&dist, [1, 1, 1], &rays).unwrap();
        assert_eq!(volume.shape(), &[1, 1, 1]);
        assert!((volume[[0, 0, 0]] - 2.0228531).abs() < 1e-5);
    }

    #[test]
    fn polyhedron_volume_matches_rays_volume() {
        let rays = crate::RaysGoldenSpiral::new(10, None).unwrap().into_rays();
        let dist = vec![1.0; rays.vertices.len()];
        let translated = polyhedron_volume(&dist, &rays.vertices, &rays.faces).unwrap();
        let existing = rays.volume(Some(&dist)).unwrap();
        assert!((translated - existing).abs() < 1e-6);
    }

    #[test]
    fn polyhedron_bbox_and_polyverts_match_native_axis_order() {
        let verts = vec![
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
        ];
        let dist = vec![1.0; verts.len()];
        let center = [2.0, 3.0, 4.0];
        let bbox = polyhedron_bbox(&dist, &center, &verts).unwrap();
        let polyverts = polyhedron_polyverts(&dist, &center, &verts).unwrap();
        assert_eq!(bbox, [1, 3, 2, 4, 3, 5]);
        assert_eq!(polyverts[0], [3.0, 3.0, 4.0]);
        assert_eq!(polyverts[5], [2.0, 3.0, 3.0]);
    }

    #[test]
    fn native_polyhedron_inside_and_render_functions_agree() {
        let rays = crate::RaysGoldenSpiral::new(10, None).unwrap().into_rays();
        let dist = vec![1.0; rays.vertices.len()];
        let center = [3.0, 3.0, 3.0];
        let bbox = polyhedron_bbox(&dist, &center, &rays.vertices).unwrap();
        let polyverts = polyhedron_polyverts(&dist, &center, &rays.vertices).unwrap();
        assert!(inside_polyhedron(
            3.0,
            3.0,
            3.0,
            &center,
            &polyverts,
            &rays.faces
        ));
        assert!(!inside_polyhedron(
            6.0,
            3.0,
            3.0,
            &center,
            &polyverts,
            &rays.faces
        ));

        let nz = (bbox[1] - bbox[0] + 1) as usize;
        let ny = (bbox[3] - bbox[2] + 1) as usize;
        let nx = (bbox[5] - bbox[4] + 1) as usize;
        let center_index = (center[2] as isize - bbox[4]) as usize
            + (center[1] as isize - bbox[2]) as usize * nx
            + (center[0] as isize - bbox[0]) as usize * nx * ny;

        let rendered =
            render_polyhedron(&dist, &center, &bbox, &polyverts, &rays.faces, nz, ny, nx).unwrap();
        assert!(rendered[center_index]);
        let overlap = overlap_render_polyhedron(
            &dist,
            &center,
            &bbox,
            &polyverts,
            &rays.faces,
            &rendered,
            nz,
            ny,
            nx,
            f32::INFINITY,
        )
        .unwrap();
        assert_eq!(overlap, rendered.iter().filter(|v| **v).count());
    }

    #[test]
    fn native_nms_radius_primitives_return_expected_bounds() {
        let rays = crate::RaysGoldenSpiral::new(10, None).unwrap().into_rays();
        let dist = vec![2.0; rays.vertices.len()];
        let aniso = [2.0, 1.0, 1.0];
        let offset =
            calculate_poly_offset_gravity(&dist, &rays.vertices, &[0.0, 0.0, 0.0]).unwrap();
        let outer = bounding_radius_outer(&dist);
        let inner = bounding_radius_inner(&dist, &rays.vertices, &rays.faces).unwrap();
        let outer_iso = bounding_radius_outer_isotropic(&dist, &rays.vertices, &aniso).unwrap();
        let inner_iso =
            bounding_radius_inner_isotropic(&dist, &rays.vertices, &rays.faces, &aniso).unwrap();
        let outer_gravity =
            bounding_radius_outer_gravity(&dist, &rays.vertices, &aniso, &offset).unwrap();
        let inner_gravity =
            bounding_radius_inner_gravity(&dist, &rays.vertices, &rays.faces, &aniso, &offset)
                .unwrap();
        assert_eq!(outer, 2.0);
        assert!(inner > 0.0 && inner <= outer);
        assert!(outer_iso >= outer);
        assert!(inner_iso > 0.0 && inner_iso <= outer_iso);
        assert!(outer_gravity > 0.0);
        assert!(inner_gravity > 0.0 && inner_gravity <= outer_gravity);
    }

    #[test]
    fn native_sphere_and_bbox_intersections_match_closed_forms() {
        let p0 = [0.0, 0.0, 0.0];
        let p1 = [5.0, 0.0, 0.0];
        let p2 = [1.0, 0.0, 0.0];
        let aniso = [1.0, 1.0, 1.0];
        assert_eq!(intersect_sphere(1.0, &p0, 1.0, &p1), 0.0);
        assert!((intersect_sphere(3.0, &p0, 1.0, &p2) - 4.0 * PI / 3.0).abs() < 1.0e-6);
        assert!(
            (intersect_sphere_isotropic(3.0, &p0, 1.0, &p2, &aniso) - 4.0 * PI / 3.0).abs()
                < 1.0e-6
        );
        assert!(
            (intersect_sphere_gravity(
                3.0,
                &p0,
                &[0.0, 0.0, 0.0],
                1.0,
                &p2,
                &[0.0, 0.0, 0.0],
                &aniso
            ) - 4.0 * PI / 3.0)
                .abs()
                < 1.0e-6
        );
        assert_eq!(
            intersect_bbox(&[0, 3, 0, 3, 0, 3], &[2, 5, 2, 5, 2, 5]),
            1.0
        );
        assert_eq!(
            intersect_bbox(&[0, 1, 0, 1, 0, 1], &[2, 3, 2, 3, 2, 3]),
            0.0
        );
    }

    #[test]
    fn qhull_volume_halfspace_intersection_matches_unit_cube() {
        let halfspaces = [
            [-1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, -1.0],
            [0.0, -1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, -1.0],
            [0.0, 0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0, -1.0],
        ];
        let volume = qhull_volume_halfspace_intersection(&halfspaces, &[0.5, 0.5, 0.5], -1.0);
        assert!((volume - 1.0).abs() < 1.0e-5);
    }

    #[test]
    fn halfspaces_convex_returns_unique_outward_cube_facets() {
        let polyverts = [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 1.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
            [1.0, 1.0, 1.0],
        ];
        let halfspaces = halfspaces_convex(&polyverts);
        assert_eq!(halfspaces.len(), 6);
        assert!(point_in_halfspaces(0.5, 0.5, 0.5, &halfspaces));
        assert!(!point_in_halfspaces(1.5, 0.5, 0.5, &halfspaces));
        for vertex in &polyverts {
            assert!(point_in_halfspaces(
                vertex[0],
                vertex[1],
                vertex[2],
                &halfspaces
            ));
        }
        let volume = qhull_volume_halfspace_intersection(&halfspaces, &[0.5, 0.5, 0.5], -1.0);
        assert!((volume - 1.0).abs() < 1.0e-5);
    }

    #[test]
    fn qhull_overlap_functions_bound_identical_polyhedron_volume() {
        let rays = crate::RaysGoldenSpiral::new(10, None).unwrap().into_rays();
        let dist = vec![1.0; rays.vertices.len()];
        let center = [3.0, 3.0, 3.0];
        let polyverts = polyhedron_polyverts(&dist, &center, &rays.vertices).unwrap();
        let volume = polyhedron_volume(&dist, &rays.vertices, &rays.faces).unwrap();
        let kernel = qhull_overlap_kernel(&polyverts, &center, &polyverts, &center, &rays.faces, 1);
        let convex =
            qhull_overlap_convex_hulls(&polyverts, &center, &polyverts, &center, &rays.faces);
        assert!(kernel >= 0.0);
        assert!(kernel <= volume + 1.0e-4);
        assert!((convex - volume).abs() < 1.0e-4);
    }

    #[test]
    fn dist_to_centroid_absolute_adds_voxel_coordinate() {
        let rays = crate::Rays {
            name: "Rays_Explicit".to_string(),
            kwargs: crate::RaysKwargs {
                n: 4,
                anisotropy: None,
                ..Default::default()
            },
            vertices: vec![
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [-1.0, -1.0, -1.0],
            ],
            faces: vec![[0, 1, 2], [0, 3, 1], [0, 2, 3], [1, 3, 2]],
        };
        let dist = vec![1.0; 2 * rays.vertices.len()];
        let relative = dist_to_centroid(&dist, [1, 1, 2], &rays, CentroidMode::Relative).unwrap();
        let absolute = dist_to_centroid(&dist, [1, 1, 2], &rays, CentroidMode::Absolute).unwrap();
        assert_eq!(relative.shape(), &[1, 1, 2, 3]);
        assert_eq!(absolute.shape(), &[1, 1, 2, 3]);
        assert!((absolute[[0, 0, 1, 0]] - relative[[0, 0, 1, 0]]).abs() < 1e-6);
        assert!((absolute[[0, 0, 1, 1]] - relative[[0, 0, 1, 1]]).abs() < 1e-6);
        assert!((absolute[[0, 0, 1, 2]] - relative[[0, 0, 1, 2]] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn polyhedron_to_label_returns_empty_background_for_no_points() {
        let rays = crate::RaysGoldenSpiral::new(4, None).unwrap().into_rays();
        let labels = polyhedron_to_label(
            &[],
            &[],
            &rays,
            [5, 5, 5],
            None,
            f32::NEG_INFINITY,
            None,
            PolyhedronRenderMode::Bbox,
            None,
        )
        .unwrap();
        assert_eq!(labels.shape(), &[5, 5, 5]);
        assert!(labels.iter().all(|v| *v == 0));
    }

    #[test]
    fn polyhedron_to_label_bbox_filters_and_sorts_by_probability() {
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
            faces: vec![
                [0, 2, 4],
                [0, 5, 2],
                [0, 4, 3],
                [0, 3, 5],
                [1, 4, 2],
                [1, 2, 5],
                [1, 3, 4],
                [1, 5, 3],
            ],
        };
        let dist = vec![1.0; 12];
        let points = [[2.0, 2.0, 2.0], [4.0, 4.0, 4.0]];
        let prob = [0.4, 0.9];
        let labels = [10, 20];
        let out = polyhedron_to_label(
            &dist,
            &points,
            &rays,
            [7, 7, 7],
            Some(&prob),
            0.5,
            Some(&labels),
            PolyhedronRenderMode::Bbox,
            None,
        )
        .unwrap();
        assert_eq!(out[[4, 4, 4]], 20);
        assert_eq!(out[[2, 2, 2]], 0);
    }

    #[test]
    fn polyhedron_to_label_uses_overlap_label_when_requested() {
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
            faces: vec![
                [0, 2, 4],
                [0, 5, 2],
                [0, 4, 3],
                [0, 3, 5],
                [1, 4, 2],
                [1, 2, 5],
                [1, 3, 4],
                [1, 5, 3],
            ],
        };
        let dist = vec![1.0; 12];
        let points = [[2.0, 2.0, 2.0], [2.0, 2.0, 2.0]];
        let out = polyhedron_to_label(
            &dist,
            &points,
            &rays,
            [5, 5, 5],
            None,
            f32::NEG_INFINITY,
            None,
            PolyhedronRenderMode::Bbox,
            Some(99),
        )
        .unwrap();
        assert_eq!(out[[2, 2, 2]], 99);
    }
}
