use std::fs::File;
use std::path::Path;

use serde_json::Value;
use stardist_rs::{Config2D, StarDist2D, StarDistThresholds};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model_dir = Path::new("assets/models/examples/2D_demo");
    let config = Config2D::from_json_file(model_dir.join("config.json"))?;
    let thresholds = load_thresholds(model_dir.join("thresholds.json"))?;

    let mut model = StarDist2D::new(config);
    model.set_thresholds(thresholds)?;

    println!("axes: {}", model.config.axes);
    println!("input channels: {}", model.config.n_channel_in);
    println!("rays: {}", model.config.n_rays);
    println!("grid: {:?}", model.config.grid);
    println!(
        "thresholds: prob={:.3}, nms={:.3}",
        model.thresholds().prob,
        model.thresholds().nms
    );

    Ok(())
}

fn load_thresholds(
    path: impl AsRef<Path>,
) -> Result<StarDistThresholds, Box<dyn std::error::Error>> {
    let value: Value = serde_json::from_reader(File::open(path)?)?;
    let prob = value.get("prob").and_then(Value::as_f64).map(|v| v as f32);
    let nms = value.get("nms").and_then(Value::as_f64).map(|v| v as f32);

    Ok(StarDistThresholds::from_options(prob, nms)?)
}
