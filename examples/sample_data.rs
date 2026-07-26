use stardist_rs::{data::abspath, sample_points, test_image_he_2d, test_image_nuclei_2d};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let nuclei_path = abspath("images/img2d.tif");
    let nuclei = test_image_nuclei_2d(true)?;
    let histology = test_image_he_2d()?;

    let mask = nuclei
        .mask
        .as_ref()
        .expect("return_mask=true should load the nuclei mask");
    let foreground: Vec<bool> = mask.data.iter().map(|label| *label > 0).collect();
    let points = sample_points(5, &foreground, &mask.shape, None, Some(4), 7)?;

    println!("nuclei path: {}", nuclei_path.display());
    println!("nuclei image shape: {:?}", nuclei.image.shape);
    println!("nuclei mask shape: {:?}", mask.shape);
    println!("histology RGB shape: {:?}", histology.shape);
    println!("sampled foreground points: {:?}", points);

    Ok(())
}
