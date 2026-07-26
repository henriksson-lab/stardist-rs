use std::fs::File;
use std::io::BufReader;
use std::path::{Path, PathBuf};

use image::GenericImageView;
use thiserror::Error;
use tiff::decoder::{Decoder, DecodingResult};

#[derive(Debug, Error)]
pub enum DataError {
    #[error("failed to decode image")]
    ImageDecode(#[from] image::ImageError),
    #[error("failed to decode tiff")]
    TiffDecode(#[from] tiff::TiffError),
    #[error("failed to open image file")]
    Io(#[from] std::io::Error),
    #[error("expected 16-bit unsigned grayscale tiff data")]
    UnsupportedTiffData,
    #[error("decoded image dimensions do not match pixel data")]
    ShapeMismatch,
}

#[derive(Clone, Debug, PartialEq)]
pub struct GrayU16Image {
    pub data: Vec<u16>,
    pub shape: Vec<usize>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct RgbU8Image {
    pub data: Vec<u8>,
    pub shape: [usize; 3],
}

#[derive(Clone, Debug, PartialEq)]
pub struct TestImageNuclei2D {
    pub image: GrayU16Image,
    pub mask: Option<GrayU16Image>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct TestImageNuclei3D {
    pub image: GrayU16Image,
    pub mask: Option<GrayU16Image>,
}

pub fn abspath(path: impl AsRef<Path>) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("assets")
        .join("data")
        .join(path)
}

pub fn test_image_nuclei_2d(return_mask: bool) -> Result<TestImageNuclei2D, DataError> {
    let img_path = abspath("images/img2d.tif");
    let mask_path = abspath("images/mask2d.tif");

    let mut img_decoder = Decoder::new(BufReader::new(File::open(img_path)?))?;
    let (img_width, img_height) = img_decoder.dimensions()?;
    let image = match img_decoder.read_image()? {
        DecodingResult::U16(data) => {
            if data.len() != img_width as usize * img_height as usize {
                return Err(DataError::ShapeMismatch);
            }
            GrayU16Image {
                data,
                shape: vec![img_height as usize, img_width as usize],
            }
        }
        _ => return Err(DataError::UnsupportedTiffData),
    };

    let mut mask_decoder = Decoder::new(BufReader::new(File::open(mask_path)?))?;
    let (mask_width, mask_height) = mask_decoder.dimensions()?;
    let mask = match mask_decoder.read_image()? {
        DecodingResult::U16(data) => {
            if data.len() != mask_width as usize * mask_height as usize {
                return Err(DataError::ShapeMismatch);
            }
            GrayU16Image {
                data,
                shape: vec![mask_height as usize, mask_width as usize],
            }
        }
        _ => return Err(DataError::UnsupportedTiffData),
    };

    Ok(TestImageNuclei2D {
        image,
        mask: if return_mask { Some(mask) } else { None },
    })
}

pub fn test_image_he_2d() -> Result<RgbU8Image, DataError> {
    let img = image::open(abspath("images/histo.jpg"))?;
    let (width, height) = img.dimensions();
    let rgb = img.into_rgb8();
    let data = rgb.into_raw();
    if data.len() != width as usize * height as usize * 3 {
        return Err(DataError::ShapeMismatch);
    }
    Ok(RgbU8Image {
        data,
        shape: [height as usize, width as usize, 3],
    })
}

pub fn test_image_nuclei_3d(return_mask: bool) -> Result<TestImageNuclei3D, DataError> {
    let img_path = abspath("images/img3d.tif");
    let mask_path = abspath("images/mask3d.tif");

    let mut img_decoder = Decoder::new(BufReader::new(File::open(img_path)?))?;
    let mut image_data = Vec::<u16>::new();
    let mut image_shape = Vec::<usize>::new();
    loop {
        let (width, height) = img_decoder.dimensions()?;
        let plane = match img_decoder.read_image()? {
            DecodingResult::U16(data) => data,
            _ => return Err(DataError::UnsupportedTiffData),
        };
        if plane.len() != width as usize * height as usize {
            return Err(DataError::ShapeMismatch);
        }
        if image_shape.is_empty() {
            image_shape = vec![0, height as usize, width as usize];
        } else if image_shape[1] != height as usize || image_shape[2] != width as usize {
            return Err(DataError::ShapeMismatch);
        }
        image_shape[0] += 1;
        image_data.extend(plane);
        if img_decoder.more_images() {
            img_decoder.next_image()?;
        } else {
            break;
        }
    }
    let image = GrayU16Image {
        data: image_data,
        shape: image_shape,
    };

    let mut mask_decoder = Decoder::new(BufReader::new(File::open(mask_path)?))?;
    let mut mask_data = Vec::<u16>::new();
    let mut mask_shape = Vec::<usize>::new();
    loop {
        let (width, height) = mask_decoder.dimensions()?;
        let plane = match mask_decoder.read_image()? {
            DecodingResult::U16(data) => data,
            _ => return Err(DataError::UnsupportedTiffData),
        };
        if plane.len() != width as usize * height as usize {
            return Err(DataError::ShapeMismatch);
        }
        if mask_shape.is_empty() {
            mask_shape = vec![0, height as usize, width as usize];
        } else if mask_shape[1] != height as usize || mask_shape[2] != width as usize {
            return Err(DataError::ShapeMismatch);
        }
        mask_shape[0] += 1;
        mask_data.extend(plane);
        if mask_decoder.more_images() {
            mask_decoder.next_image()?;
        } else {
            break;
        }
    }
    let mask = GrayU16Image {
        data: mask_data,
        shape: mask_shape,
    };

    Ok(TestImageNuclei3D {
        image,
        mask: if return_mask { Some(mask) } else { None },
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn abspath_points_to_vendored_data_file() {
        let path = abspath("images/img2d.tif");
        assert!(path.is_absolute());
        assert!(path.ends_with("assets/data/images/img2d.tif"));
        assert!(path.exists());
    }

    #[test]
    fn test_image_nuclei_2d_loads_image_and_optional_mask_like_python() {
        let image_only = test_image_nuclei_2d(false).unwrap();
        assert_eq!(image_only.image.shape, vec![512, 512]);
        assert_eq!(image_only.image.data.len(), 512 * 512);
        assert!(image_only.image.data.iter().any(|value| *value > 0));
        assert!(image_only.mask.is_none());

        let with_mask = test_image_nuclei_2d(true).unwrap();
        assert_eq!(with_mask.image.shape, vec![512, 512]);
        let mask = with_mask.mask.unwrap();
        assert_eq!(mask.shape, vec![512, 512]);
        assert_eq!(mask.data.len(), 512 * 512);
        assert!(mask.data.iter().any(|value| *value > 0));
    }

    #[test]
    fn test_image_he_2d_loads_rgb_image_like_python() {
        let img = test_image_he_2d().unwrap();
        assert_eq!(img.shape, [300, 500, 3]);
        assert_eq!(img.data.len(), 300 * 500 * 3);
        assert!(img.data.iter().any(|value| *value > 0));
    }

    #[test]
    fn test_image_nuclei_3d_loads_volume_and_optional_mask_like_python() {
        let image_only = test_image_nuclei_3d(false).unwrap();
        assert_eq!(image_only.image.shape, vec![31, 61, 57]);
        assert_eq!(image_only.image.data.len(), 31 * 61 * 57);
        assert!(image_only.image.data.iter().any(|value| *value > 0));
        assert!(image_only.mask.is_none());

        let with_mask = test_image_nuclei_3d(true).unwrap();
        assert_eq!(with_mask.image.shape, vec![31, 61, 57]);
        let mask = with_mask.mask.unwrap();
        assert_eq!(mask.shape, vec![31, 61, 57]);
        assert_eq!(mask.data.len(), 31 * 61 * 57);
        assert!(mask.data.iter().any(|value| *value > 0));
    }
}
