use std::hint::black_box;
use std::time::Instant;

use stardist_rs::nms::poly_intersection_area;

fn radial_polygon(cx: f32, cy: f32, base_radius: f32, phase: f32, n_rays: usize) -> Vec<[f32; 2]> {
    let mut points = Vec::with_capacity(n_rays);
    for ray in 0..n_rays {
        let angle = (ray as f32) * std::f32::consts::TAU / (n_rays as f32);
        let wobble = 1.0 + 0.18 * (angle * 3.0 + phase).sin() + 0.08 * (angle * 7.0).cos();
        let radius = base_radius * wobble.max(0.25);
        points.push([cx + radius * angle.cos(), cy + radius * angle.sin()]);
    }
    points
}

fn polygon_pairs(count: usize, n_rays: usize) -> Vec<(Vec<[f32; 2]>, Vec<[f32; 2]>)> {
    let mut pairs = Vec::with_capacity(count);
    for i in 0..count {
        let x = ((i * 37) % 256) as f32 + 64.0;
        let y = ((i * 53) % 256) as f32 + 64.0;
        let radius = 10.0 + ((i * 17) % 16) as f32;
        let dx = ((i * 11) % 9) as f32 - 4.0;
        let dy = ((i * 13) % 9) as f32 - 4.0;
        pairs.push((
            radial_polygon(x, y, radius, i as f32 * 0.17, n_rays),
            radial_polygon(x + dx, y + dy, radius * 0.93, i as f32 * 0.19 + 0.4, n_rays),
        ));
    }
    pairs
}

fn time_backend<F>(
    name: &str,
    pairs: &[(Vec<[f32; 2]>, Vec<[f32; 2]>)],
    repeats: usize,
    f: F,
) -> (f64, f32)
where
    F: Fn(&[[f32; 2]], &[[f32; 2]]) -> f32,
{
    let start = Instant::now();
    let mut total = 0.0f32;
    for _ in 0..repeats {
        for (a, b) in pairs {
            total += black_box(f(black_box(a), black_box(b)));
        }
    }
    let seconds = start.elapsed().as_secs_f64();
    println!("{name}: {seconds:.6} s total_area={total:.3}");
    (seconds, total)
}

fn main() {
    let args = std::env::args().collect::<Vec<_>>();
    let pairs = args.get(1).and_then(|v| v.parse().ok()).unwrap_or(20_000);
    let repeats = args.get(2).and_then(|v| v.parse().ok()).unwrap_or(5);
    let n_rays = args.get(3).and_then(|v| v.parse().ok()).unwrap_or(32);

    let pairs = polygon_pairs(pairs, n_rays);
    let _ = time_backend("clipper", &pairs, repeats, poly_intersection_area);
}
