use stardist_rs::{_create_stardist_dependencies, _get_stardist_metadata};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let outdir = std::env::temp_dir().join(format!(
        "stardist-rs-bioimageio-example-{}",
        std::process::id()
    ));

    let dependencies = _create_stardist_dependencies(
        &outdir,
        "2.13.1",
        "0.9.2",
        &["bioimageio.core>=0.5".to_string()],
    )?;
    let metadata = _get_stardist_metadata(
        &outdir,
        2,
        "StarDist nuclei segmentation model",
        "Martin Weigert, Uwe Schmidt",
        "https://github.com/stardist/stardist",
        "BSD-3-Clause",
        Some(dependencies),
    )?;

    println!("description: {}", metadata.description);
    println!("authors: {}", metadata.authors.len());
    println!(
        "tags include 2d: {}",
        metadata.tags.iter().any(|tag| tag == "2d")
    );
    println!("documentation: {}", metadata.documentation.display());
    println!(
        "dependencies: {}",
        metadata.dependencies.unwrap_or_default()
    );

    Ok(())
}
