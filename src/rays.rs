use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq)]
pub struct Rays {
    pub name: String,
    pub kwargs: RaysKwargs,
    pub vertices: Vec<[f32; 3]>,
    pub faces: Vec<[usize; 3]>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct RaysJson {
    pub name: String,
    pub kwargs: RaysKwargs,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct RaysKwargs {
    #[serde(default)]
    pub n: usize,
    #[serde(default)]
    pub anisotropy: Option<[f32; 3]>,
    #[serde(default)]
    pub vertices0: Vec<[f32; 3]>,
    #[serde(default)]
    pub faces0: Vec<[usize; 3]>,
    #[serde(default)]
    pub n_rays_x: usize,
    #[serde(default)]
    pub n_rays_z: usize,
    #[serde(default)]
    pub n_level: usize,
}

impl Default for RaysKwargs {
    fn default() -> Self {
        Self {
            n: 0,
            anisotropy: None,
            vertices0: Vec::new(),
            faces0: Vec::new(),
            n_rays_x: 0,
            n_rays_z: 0,
            n_level: 0,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct RaysExplicit {
    pub vertices0: Vec<[f32; 3]>,
    pub faces0: Vec<[usize; 3]>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RaysCartesian {
    pub n_rays_x: usize,
    pub n_rays_z: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RaysSubDivide {
    pub n_level: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RaysTetra {
    pub n_level: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RaysOcto {
    pub n_level: usize,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RaysGoldenSpiral {
    pub n: usize,
    pub anisotropy: Option<[f32; 3]>,
}

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum RaysError {
    #[error("At least 4 points have to be given!")]
    TooFewPoints,
    #[error("unknown rays class {0}")]
    UnknownRaysClass(String),
    #[error("last dimension of dist should have length len(rays.vertices)")]
    DistLengthMismatch,
}

impl Rays {
    pub fn setup_vertices_faces(&self) -> (Vec<[f32; 3]>, Vec<[usize; 3]>) {
        (self.vertices.clone(), self.faces.clone())
    }

    pub fn vertices(&self) -> Vec<[f32; 3]> {
        self.vertices.clone()
    }

    pub fn faces(&self) -> Vec<[usize; 3]> {
        self.faces.clone()
    }

    pub fn get(&self, i: usize) -> Option<[f32; 3]> {
        self.vertices.get(i).copied()
    }

    pub fn len(&self) -> usize {
        self.vertices.len()
    }

    pub fn repr(&self) -> String {
        match self.name.as_str() {
            "Rays_Cartesian" => format!(
                "Rays_Cartesian_n_rays_x_{}_n_rays_z_{}",
                self.kwargs.n_rays_x, self.kwargs.n_rays_z
            ),
            "Rays_Tetra" => format!("Rays_Tetra_n_level_{}", self.kwargs.n_level),
            "Rays_Octo" => format!("Rays_Octo_n_level_{}", self.kwargs.n_level),
            "Rays_GoldenSpiral" => {
                if let Some(anisotropy) = self.kwargs.anisotropy {
                    format!(
                        "Rays_GoldenSpiral_anisotropy_{:.2}_{:.2}_{:.2}_n_{}",
                        anisotropy[0], anisotropy[1], anisotropy[2], self.kwargs.n
                    )
                } else {
                    format!("Rays_GoldenSpiral_anisotropy_None_n_{}", self.kwargs.n)
                }
            }
            "Rays_Explicit" => {
                let mut vertices = String::new();
                for (i, vertex) in self.kwargs.vertices0.iter().enumerate() {
                    if i > 0 {
                        vertices.push('_');
                    }
                    vertices.push_str(&format!(
                        "{:.2}_{:.2}_{:.2}",
                        vertex[0], vertex[1], vertex[2]
                    ));
                }
                let mut faces = String::new();
                for (i, face) in self.kwargs.faces0.iter().enumerate() {
                    if i > 0 {
                        faces.push('_');
                    }
                    faces.push_str(&format!("{}_{}_{}", face[0], face[1], face[2]));
                }
                format!("Rays_Explicit_faces0_{faces}_vertices0_{vertices}")
            }
            _ => self.name.clone(),
        }
    }

    pub fn to_json(&self) -> RaysJson {
        RaysJson {
            name: self.name.clone(),
            kwargs: self.kwargs.clone(),
        }
    }

    pub fn dist_loss_weights(&self, anisotropy: Option<[f32; 3]>) -> Vec<f32> {
        let anisotropy = anisotropy.unwrap_or([1.0, 1.0, 1.0]);
        let mut weights = Vec::with_capacity(self.vertices.len());
        for vertex in &self.vertices {
            weights.push(
                ((vertex[0] * anisotropy[0]).powi(2)
                    + (vertex[1] * anisotropy[1]).powi(2)
                    + (vertex[2] * anisotropy[2]).powi(2))
                .sqrt(),
            );
        }
        weights
    }

    pub fn volume(&self, dist: Option<&[f32]>) -> Result<f32, RaysError> {
        let n_rays = self.vertices.len();
        if let Some(dist) = dist {
            if dist.len() != n_rays {
                return Err(RaysError::DistLengthMismatch);
            }
        }
        let mut d_sum = 0.0f32;
        for face in &self.faces {
            let d0 = dist.map(|dist| dist[face[0]]).unwrap_or(1.0);
            let d1 = dist.map(|dist| dist[face[1]]).unwrap_or(1.0);
            let d2 = dist.map(|dist| dist[face[2]]).unwrap_or(1.0);
            let v0 = [
                d0 * self.vertices[face[0]][0],
                d0 * self.vertices[face[0]][1],
                d0 * self.vertices[face[0]][2],
            ];
            let v1 = [
                d1 * self.vertices[face[1]][0],
                d1 * self.vertices[face[1]][1],
                d1 * self.vertices[face[1]][2],
            ];
            let v2 = [
                d2 * self.vertices[face[2]][0],
                d2 * self.vertices[face[2]][1],
                d2 * self.vertices[face[2]][2],
            ];
            d_sum += v0[0] * (v1[1] * v2[2] - v1[2] * v2[1])
                - v0[1] * (v1[0] * v2[2] - v1[2] * v2[0])
                + v0[2] * (v1[0] * v2[1] - v1[1] * v2[0]);
        }
        Ok(-1.0 / 6.0 * d_sum)
    }

    pub fn surface(&self, dist: &[f32]) -> Result<f32, RaysError> {
        if dist.len() != self.vertices.len() {
            return Err(RaysError::DistLengthMismatch);
        }
        let mut d_sum = 0.0f32;
        for face in &self.faces {
            let v0 = [
                dist[face[0]] * self.vertices[face[0]][0],
                dist[face[0]] * self.vertices[face[0]][1],
                dist[face[0]] * self.vertices[face[0]][2],
            ];
            let v1 = [
                dist[face[1]] * self.vertices[face[1]][0],
                dist[face[1]] * self.vertices[face[1]][1],
                dist[face[1]] * self.vertices[face[1]][2],
            ];
            let v2 = [
                dist[face[2]] * self.vertices[face[2]][0],
                dist[face[2]] * self.vertices[face[2]][1],
                dist[face[2]] * self.vertices[face[2]][2],
            ];
            let pa = [v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2]];
            let pb = [v2[0] - v0[0], v2[1] - v0[1], v2[2] - v0[2]];
            let cross = [
                pa[1] * pb[2] - pa[2] * pb[1],
                pa[2] * pb[0] - pa[0] * pb[2],
                pa[0] * pb[1] - pa[1] * pb[0],
            ];
            d_sum += 0.5 * (cross[0].powi(2) + cross[1].powi(2) + cross[2].powi(2)).sqrt();
        }
        Ok(d_sum)
    }

    pub fn copy(&self, scale: [f32; 3]) -> Self {
        let mut res = self.clone();
        for vertex in &mut res.vertices {
            vertex[0] *= scale[0];
            vertex[1] *= scale[1];
            vertex[2] *= scale[2];
        }
        res
    }
}

impl RaysExplicit {
    pub fn new(vertices0: Vec<[f32; 3]>, faces0: Vec<[usize; 3]>) -> Self {
        Self { vertices0, faces0 }
    }

    pub fn setup_vertices_faces(&self) -> (Vec<[f32; 3]>, Vec<[usize; 3]>) {
        (self.vertices0.clone(), self.faces0.clone())
    }

    pub fn into_rays(self) -> Rays {
        let (vertices, faces) = self.setup_vertices_faces();
        Rays {
            name: "Rays_Explicit".to_string(),
            kwargs: RaysKwargs {
                n: 0,
                anisotropy: None,
                vertices0: self.vertices0,
                faces0: self.faces0,
                n_rays_x: 0,
                n_rays_z: 0,
                n_level: 0,
            },
            vertices,
            faces,
        }
    }
}

impl RaysCartesian {
    pub fn new(n_rays_x: usize, n_rays_z: usize) -> Self {
        Self { n_rays_x, n_rays_z }
    }

    pub fn setup_vertices_faces(&self) -> (Vec<[f32; 3]>, Vec<[usize; 3]>) {
        let n_rays_x = self.n_rays_x;
        let n_rays_z = self.n_rays_z;
        let dphi = 2.0f32 * std::f32::consts::PI / n_rays_x as f32;
        let dtheta = std::f32::consts::PI / n_rays_z as f32;

        let mut verts = Vec::with_capacity(n_rays_x * n_rays_z);
        for mz in 0..n_rays_z {
            for mx in 0..n_rays_x {
                let phi = mx as f32 * dphi;
                let mut theta = mz as f32 * dtheta;
                if mz == 0 {
                    theta = 1.0e-12;
                }
                if mz == n_rays_z - 1 {
                    theta = std::f32::consts::PI - 1.0e-12;
                }
                let mut dx = phi.cos() * theta.sin();
                let mut dy = phi.sin() * theta.sin();
                let dz = theta.cos();
                if mz == 0 || mz == n_rays_z - 1 {
                    dx += 1.0e-12;
                    dy += 1.0e-12;
                }
                verts.push([dz, dy, dx]);
            }
        }

        let mut faces = Vec::with_capacity((n_rays_z.saturating_sub(1)) * n_rays_x * 2);
        for mz in 0..n_rays_z.saturating_sub(1) {
            for mx in 0..n_rays_x {
                faces.push([
                    mz * n_rays_x + mx,
                    (mz + 1) * n_rays_x + (mx + 1) % n_rays_x,
                    mz * n_rays_x + (mx + 1) % n_rays_x,
                ]);
                faces.push([
                    mz * n_rays_x + mx,
                    (mz + 1) * n_rays_x + mx,
                    (mz + 1) * n_rays_x + (mx + 1) % n_rays_x,
                ]);
            }
        }

        (verts, faces)
    }

    pub fn into_rays(self) -> Rays {
        let (vertices, faces) = self.setup_vertices_faces();
        Rays {
            name: "Rays_Cartesian".to_string(),
            kwargs: RaysKwargs {
                n: 0,
                anisotropy: None,
                vertices0: Vec::new(),
                faces0: Vec::new(),
                n_rays_x: self.n_rays_x,
                n_rays_z: self.n_rays_z,
                n_level: 0,
            },
            vertices,
            faces,
        }
    }
}

impl RaysSubDivide {
    pub fn new(n_level: usize) -> Self {
        Self { n_level }
    }

    pub fn base_polyhedron(&self) -> (Vec<[f32; 3]>, Vec<[usize; 3]>) {
        unimplemented!("Rays_SubDivide.base_polyhedron is abstract")
    }

    pub fn setup_vertices_faces(&self) -> (Vec<[f32; 3]>, Vec<[usize; 3]>) {
        let (verts0, faces0) = self.base_polyhedron();
        self._recursive_split(verts0, faces0, self.n_level)
    }

    pub fn _recursive_split(
        &self,
        verts: Vec<[f32; 3]>,
        faces: Vec<[usize; 3]>,
        n_level: usize,
    ) -> (Vec<[f32; 3]>, Vec<[usize; 3]>) {
        if n_level <= 1 {
            (verts, faces)
        } else {
            let (verts, faces) = RaysSubDivide::split(&verts, &faces);
            self._recursive_split(verts, faces, n_level - 1)
        }
    }

    pub fn split(verts0: &[[f32; 3]], faces0: &[[usize; 3]]) -> (Vec<[f32; 3]>, Vec<[usize; 3]>) {
        let mut split_edges = Vec::<((usize, usize), usize)>::new();
        let mut verts = verts0.to_vec();
        let mut faces = Vec::with_capacity(faces0.len() * 4);

        for face in faces0 {
            let v1 = face[0];
            let v2 = face[1];
            let v3 = face[2];

            let edge = if v1 < v2 { (v1, v2) } else { (v2, v1) };
            let ind1 = if let Some((_, ind)) = split_edges.iter().find(|(e, _)| *e == edge) {
                *ind
            } else {
                let mut v = [
                    0.5 * (verts[v1][0] + verts[v2][0]),
                    0.5 * (verts[v1][1] + verts[v2][1]),
                    0.5 * (verts[v1][2] + verts[v2][2]),
                ];
                let norm = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
                v[0] /= norm;
                v[1] /= norm;
                v[2] /= norm;
                verts.push(v);
                let ind = verts.len() - 1;
                split_edges.push((edge, ind));
                ind
            };

            let edge = if v2 < v3 { (v2, v3) } else { (v3, v2) };
            let ind2 = if let Some((_, ind)) = split_edges.iter().find(|(e, _)| *e == edge) {
                *ind
            } else {
                let mut v = [
                    0.5 * (verts[v2][0] + verts[v3][0]),
                    0.5 * (verts[v2][1] + verts[v3][1]),
                    0.5 * (verts[v2][2] + verts[v3][2]),
                ];
                let norm = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
                v[0] /= norm;
                v[1] /= norm;
                v[2] /= norm;
                verts.push(v);
                let ind = verts.len() - 1;
                split_edges.push((edge, ind));
                ind
            };

            let edge = if v3 < v1 { (v3, v1) } else { (v1, v3) };
            let ind3 = if let Some((_, ind)) = split_edges.iter().find(|(e, _)| *e == edge) {
                *ind
            } else {
                let mut v = [
                    0.5 * (verts[v3][0] + verts[v1][0]),
                    0.5 * (verts[v3][1] + verts[v1][1]),
                    0.5 * (verts[v3][2] + verts[v1][2]),
                ];
                let norm = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
                v[0] /= norm;
                v[1] /= norm;
                v[2] /= norm;
                verts.push(v);
                let ind = verts.len() - 1;
                split_edges.push((edge, ind));
                ind
            };

            faces.push([v1, ind1, ind3]);
            faces.push([v2, ind2, ind1]);
            faces.push([v3, ind3, ind2]);
            faces.push([ind1, ind2, ind3]);
        }

        (verts, faces)
    }
}

impl RaysTetra {
    pub fn new(n_level: usize) -> Self {
        Self { n_level }
    }

    pub fn base_polyhedron(&self) -> (Vec<[f32; 3]>, Vec<[usize; 3]>) {
        let verts = vec![
            [(8.0f32 / 9.0).sqrt(), 0.0, -1.0 / 3.0],
            [-(2.0f32 / 9.0).sqrt(), (2.0f32 / 3.0).sqrt(), -1.0 / 3.0],
            [-(2.0f32 / 9.0).sqrt(), -(2.0f32 / 3.0).sqrt(), -1.0 / 3.0],
            [0.0, 0.0, 1.0],
        ];
        let faces = vec![[0, 1, 2], [0, 3, 1], [0, 2, 3], [1, 3, 2]];
        (verts, faces)
    }

    pub fn setup_vertices_faces(&self) -> (Vec<[f32; 3]>, Vec<[usize; 3]>) {
        let (verts0, faces0) = self.base_polyhedron();
        RaysSubDivide::new(self.n_level)._recursive_split(verts0, faces0, self.n_level)
    }

    pub fn into_rays(self) -> Rays {
        let (vertices, faces) = self.setup_vertices_faces();
        Rays {
            name: "Rays_Tetra".to_string(),
            kwargs: RaysKwargs {
                n: 0,
                anisotropy: None,
                vertices0: Vec::new(),
                faces0: Vec::new(),
                n_rays_x: 0,
                n_rays_z: 0,
                n_level: self.n_level,
            },
            vertices,
            faces,
        }
    }
}

impl RaysOcto {
    pub fn new(n_level: usize) -> Self {
        Self { n_level }
    }

    pub fn base_polyhedron(&self) -> (Vec<[f32; 3]>, Vec<[usize; 3]>) {
        let verts = vec![
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, -1.0],
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
        ];
        let faces = vec![
            [0, 1, 4],
            [0, 5, 1],
            [1, 2, 4],
            [1, 5, 2],
            [2, 3, 4],
            [2, 5, 3],
            [3, 0, 4],
            [3, 5, 0],
        ];
        (verts, faces)
    }

    pub fn setup_vertices_faces(&self) -> (Vec<[f32; 3]>, Vec<[usize; 3]>) {
        let (verts0, faces0) = self.base_polyhedron();
        RaysSubDivide::new(self.n_level)._recursive_split(verts0, faces0, self.n_level)
    }

    pub fn into_rays(self) -> Rays {
        let (vertices, faces) = self.setup_vertices_faces();
        Rays {
            name: "Rays_Octo".to_string(),
            kwargs: RaysKwargs {
                n: 0,
                anisotropy: None,
                vertices0: Vec::new(),
                faces0: Vec::new(),
                n_rays_x: 0,
                n_rays_z: 0,
                n_level: self.n_level,
            },
            vertices,
            faces,
        }
    }
}

impl RaysGoldenSpiral {
    pub fn new(n: usize, anisotropy: Option<[f32; 3]>) -> Result<Self, RaysError> {
        if n < 4 {
            return Err(RaysError::TooFewPoints);
        }
        Ok(Self { n, anisotropy })
    }

    pub fn setup_vertices_faces(&self) -> (Vec<[f32; 3]>, Vec<[usize; 3]>) {
        let n = self.n;
        let anisotropy = self.anisotropy.unwrap_or([1.0, 1.0, 1.0]);
        let g = (3.0f32 - 5.0f32.sqrt()) * std::f32::consts::PI;
        let mut verts = Vec::with_capacity(n);
        for i in 0..n {
            let phi = g * i as f32;
            let z = if n == 1 {
                -1.0
            } else {
                -1.0 + 2.0 * i as f32 / (n as f32 - 1.0)
            };
            let rho = (1.0 - z.powi(2)).sqrt();
            verts.push([
                z / anisotropy[0],
                rho * phi.sin() / anisotropy[1],
                rho * phi.cos() / anisotropy[2],
            ]);
        }

        let mut faces = Vec::<[usize; 3]>::new();
        let eps = 1.0e-5f32;
        for a in 0..n.saturating_sub(2) {
            for b in a + 1..n.saturating_sub(1) {
                for c in b + 1..n {
                    let va = verts[a];
                    let vb = verts[b];
                    let vc = verts[c];
                    let ab = [vb[0] - va[0], vb[1] - va[1], vb[2] - va[2]];
                    let ac = [vc[0] - va[0], vc[1] - va[1], vc[2] - va[2]];
                    let normal = [
                        ab[1] * ac[2] - ab[2] * ac[1],
                        ab[2] * ac[0] - ab[0] * ac[2],
                        ab[0] * ac[1] - ab[1] * ac[0],
                    ];
                    let norm = (normal[0].powi(2) + normal[1].powi(2) + normal[2].powi(2)).sqrt();
                    if norm <= eps {
                        continue;
                    }
                    let mut pos = false;
                    let mut neg = false;
                    for (p, vp) in verts.iter().enumerate() {
                        if p == a || p == b || p == c {
                            continue;
                        }
                        let ap = [vp[0] - va[0], vp[1] - va[1], vp[2] - va[2]];
                        let side = normal[0] * ap[0] + normal[1] * ap[1] + normal[2] * ap[2];
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
                        faces.push([a, b, c]);
                    }
                }
            }
        }

        let faces = reorder_faces(&verts, &faces);
        for vert in &mut verts {
            let norm = (vert[0].powi(2) + vert[1].powi(2) + vert[2].powi(2)).sqrt();
            vert[0] /= norm;
            vert[1] /= norm;
            vert[2] /= norm;
        }

        (verts, faces)
    }

    pub fn into_rays(self) -> Rays {
        let (vertices, faces) = self.setup_vertices_faces();
        Rays {
            name: "Rays_GoldenSpiral".to_string(),
            kwargs: RaysKwargs {
                n: self.n,
                anisotropy: self.anisotropy,
                vertices0: Vec::new(),
                faces0: Vec::new(),
                n_rays_x: 0,
                n_rays_z: 0,
                n_level: 0,
            },
            vertices,
            faces,
        }
    }
}

pub fn rays_from_json(d: &RaysJson) -> Result<Rays, RaysError> {
    if d.name == "Rays_GoldenSpiral" {
        Ok(RaysGoldenSpiral::new(d.kwargs.n, d.kwargs.anisotropy)?.into_rays())
    } else if d.name == "Rays_Explicit" {
        Ok(RaysExplicit::new(d.kwargs.vertices0.clone(), d.kwargs.faces0.clone()).into_rays())
    } else if d.name == "Rays_Cartesian" {
        Ok(RaysCartesian::new(d.kwargs.n_rays_x, d.kwargs.n_rays_z).into_rays())
    } else if d.name == "Rays_Tetra" {
        Ok(RaysTetra::new(d.kwargs.n_level).into_rays())
    } else if d.name == "Rays_Octo" {
        Ok(RaysOcto::new(d.kwargs.n_level).into_rays())
    } else {
        Err(RaysError::UnknownRaysClass(d.name.clone()))
    }
}

pub fn reorder_faces(verts: &[[f32; 3]], faces: &[[usize; 3]]) -> Vec<[usize; 3]> {
    let mut reordered = Vec::with_capacity(faces.len());
    for face in faces {
        let a = verts[face[0]];
        let b = verts[face[1]];
        let c = verts[face[2]];
        let det = a[0] * (b[1] * c[2] - b[2] * c[1]) - a[1] * (b[0] * c[2] - b[2] * c[0])
            + a[2] * (b[0] * c[1] - b[1] * c[0]);
        if det > 0.0 {
            reordered.push([face[2], face[1], face[0]]);
        } else {
            reordered.push(*face);
        }
    }
    reordered
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn golden_spiral_rejects_too_few_points() {
        assert_eq!(
            RaysGoldenSpiral::new(3, None).unwrap_err(),
            RaysError::TooFewPoints
        );
    }

    #[test]
    fn explicit_rays_round_trip_through_json() {
        let rays = RaysExplicit::new(
            vec![
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [-1.0, -1.0, -1.0],
            ],
            vec![[0, 1, 2], [0, 3, 1], [0, 2, 3], [1, 3, 2]],
        )
        .into_rays();
        let rebuilt = rays_from_json(&rays.to_json()).unwrap();
        assert_eq!(rebuilt.name, "Rays_Explicit");
        assert_eq!(rebuilt.vertices, rays.vertices);
        assert_eq!(rebuilt.faces, rays.faces);
    }

    #[test]
    fn cartesian_rays_match_python_counts_and_poles() {
        let rays = RaysCartesian::new(4, 3).into_rays();
        assert_eq!(rays.vertices.len(), 12);
        assert_eq!(rays.faces.len(), 16);
        assert!((rays.vertices[0][0] - 1.0).abs() < 1.0e-6);
        assert!(rays.vertices[0][1] > 0.0);
        assert!(rays.vertices[0][2] > 0.0);
        assert!((rays.vertices[8][0] + 1.0).abs() < 1.0e-6);
    }

    #[test]
    fn rays_base_sequence_and_repr_methods_match_python_shape() {
        let cartesian = RaysCartesian::new(4, 3).into_rays();
        assert_eq!(cartesian.len(), 12);
        assert_eq!(cartesian.get(0), Some(cartesian.vertices[0]));
        assert_eq!(cartesian.get(12), None);
        assert_eq!(cartesian.repr(), "Rays_Cartesian_n_rays_x_4_n_rays_z_3");
        let mut vertices_copy = cartesian.vertices();
        vertices_copy[0] = [9.0, 9.0, 9.0];
        assert_ne!(vertices_copy[0], cartesian.vertices[0]);
        let mut faces_copy = cartesian.faces();
        faces_copy[0] = [9, 9, 9];
        assert_ne!(faces_copy[0], cartesian.faces[0]);
        let default_weights = cartesian.dist_loss_weights(None);
        assert!(
            default_weights
                .iter()
                .all(|weight| (weight - 1.0).abs() < 1e-6)
        );
        let anisotropic_weights = cartesian.dist_loss_weights(Some([2.0, 1.0, 1.0]));
        assert!((anisotropic_weights[0] - 2.0).abs() < 1e-6);

        let tetra = RaysTetra::new(2).into_rays();
        assert_eq!(tetra.repr(), "Rays_Tetra_n_level_2");

        let golden = RaysGoldenSpiral::new(4, Some([2.0, 1.0, 0.5]))
            .unwrap()
            .into_rays();
        assert_eq!(
            golden.repr(),
            "Rays_GoldenSpiral_anisotropy_2.00_1.00_0.50_n_4"
        );
    }

    #[test]
    fn subdivide_split_matches_original_counts() {
        let tetra = RaysTetra::new(2).into_rays();
        let octo = RaysOcto::new(2).into_rays();
        assert_eq!(tetra.vertices.len(), 10);
        assert_eq!(tetra.faces.len(), 16);
        assert_eq!(octo.vertices.len(), 18);
        assert_eq!(octo.faces.len(), 32);
        assert!(
            tetra
                .vertices
                .iter()
                .chain(octo.vertices.iter())
                .all(
                    |v| ((v[0].powi(2) + v[1].powi(2) + v[2].powi(2)).sqrt() - 1.0).abs() < 1.0e-6
                )
        );
    }

    #[test]
    fn subdivide_base_polyhedron_overrides_match_python_base_counts() {
        let (tetra_vertices, tetra_faces) = RaysTetra::new(1).base_polyhedron();
        assert_eq!(tetra_vertices.len(), 4);
        assert_eq!(
            tetra_faces,
            vec![[0, 1, 2], [0, 3, 1], [0, 2, 3], [1, 3, 2]]
        );

        let (octo_vertices, octo_faces) = RaysOcto::new(1).base_polyhedron();
        assert_eq!(octo_vertices.len(), 6);
        assert_eq!(
            octo_faces,
            vec![
                [0, 1, 4],
                [0, 5, 1],
                [1, 2, 4],
                [1, 5, 2],
                [2, 3, 4],
                [2, 5, 3],
                [3, 0, 4],
                [3, 5, 0],
            ]
        );
    }

    #[test]
    fn reorder_faces_flips_positive_determinants_like_python() {
        let verts = vec![[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let faces = vec![[0, 1, 2], [2, 1, 0]];
        assert_eq!(reorder_faces(&verts, &faces), vec![[2, 1, 0], [2, 1, 0]]);
    }

    #[test]
    fn rays_json_rebuilds_tetra_octo_and_cartesian() {
        let tetra = rays_from_json(&RaysJson {
            name: "Rays_Tetra".to_string(),
            kwargs: RaysKwargs {
                n: 0,
                anisotropy: None,
                vertices0: Vec::new(),
                faces0: Vec::new(),
                n_rays_x: 0,
                n_rays_z: 0,
                n_level: 1,
            },
        })
        .unwrap();
        let octo = rays_from_json(&RaysJson {
            name: "Rays_Octo".to_string(),
            kwargs: RaysKwargs {
                n: 0,
                anisotropy: None,
                vertices0: Vec::new(),
                faces0: Vec::new(),
                n_rays_x: 0,
                n_rays_z: 0,
                n_level: 1,
            },
        })
        .unwrap();
        let cartesian = rays_from_json(&RaysJson {
            name: "Rays_Cartesian".to_string(),
            kwargs: RaysKwargs {
                n: 0,
                anisotropy: None,
                vertices0: Vec::new(),
                faces0: Vec::new(),
                n_rays_x: 4,
                n_rays_z: 3,
                n_level: 0,
            },
        })
        .unwrap();
        assert_eq!(tetra.vertices.len(), 4);
        assert_eq!(octo.vertices.len(), 6);
        assert_eq!(cartesian.vertices.len(), 12);
    }

    #[test]
    fn golden_spiral_matches_first_python_vertices_without_anisotropy() {
        let rays = RaysGoldenSpiral::new(4, None).unwrap().into_rays();
        assert_eq!(rays.vertices.len(), 4);
        assert_eq!(rays.faces.len(), 4);
        let expected = [
            [-1.0, 0.0, 0.0],
            [-0.3333334, 0.6368584, -0.6951980],
            [0.3333334, -0.9391991, 0.0824258],
        ];
        for (actual, expected) in rays.vertices.iter().take(3).zip(expected) {
            for i in 0..3 {
                assert!((actual[i] - expected[i]).abs() < 1e-5);
            }
        }
    }

    #[test]
    fn golden_spiral_face_counts_match_convex_hull_for_small_sets() {
        for n in [4, 6, 10, 96] {
            let rays = RaysGoldenSpiral::new(n, if n == 96 { Some([2.0, 1.0, 1.0]) } else { None })
                .unwrap()
                .into_rays();
            assert_eq!(rays.vertices.len(), n);
            assert_eq!(rays.faces.len(), 2 * n - 4);
        }
    }

    #[test]
    fn golden_spiral_volume_and_surface_match_python_spot_checks() {
        let rays = RaysGoldenSpiral::new(10, None).unwrap().into_rays();
        let dist = vec![1.0; rays.vertices.len()];
        assert!((rays.volume(Some(&dist)).unwrap() - 2.0228531).abs() < 1e-5);
        assert!((rays.surface(&dist).unwrap() - 8.6687736).abs() < 1e-5);
    }

    #[test]
    fn rays_json_round_trip_rebuilds_golden_spiral() {
        let rays = RaysGoldenSpiral::new(6, Some([2.0, 1.0, 1.0]))
            .unwrap()
            .into_rays();
        let json = rays.to_json();
        let rebuilt = rays_from_json(&json).unwrap();
        assert_eq!(rebuilt.name, "Rays_GoldenSpiral");
        assert_eq!(rebuilt.kwargs.n, 6);
        assert_eq!(rebuilt.kwargs.anisotropy, Some([2.0, 1.0, 1.0]));
        assert_eq!(rebuilt.vertices.len(), 6);
        assert_eq!(rebuilt.faces.len(), 8);
    }
}
