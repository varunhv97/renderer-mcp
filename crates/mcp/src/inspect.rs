use crate::output::MAX_OUTPUT_PATH_BYTES;
use image::{ImageFormat, ImageReader};
use sha2::{Digest, Sha256};
use std::fs::File;
use std::io::Read;
use std::path::PathBuf;

const MAX_INSPECT_FILE_BYTES: u64 = 64 * 1024 * 1024;

pub(crate) fn inspect_image(arguments: &serde_json::Value) -> Result<serde_json::Value, String> {
    let path = PathBuf::from(
        arguments
            .get("path")
            .and_then(serde_json::Value::as_str)
            .ok_or("path is required")?,
    );
    if path.as_os_str().len() > MAX_OUTPUT_PATH_BYTES {
        return Err("path exceeds 4 KiB".into());
    }
    let reader = ImageReader::open(&path)
        .map_err(|error| format!("could not open image: {error}"))?
        .with_guessed_format()
        .map_err(|error| format!("could not detect image format: {error}"))?;
    let format = reader.format().ok_or("could not detect image format")?;
    let (width, height) = reader
        .into_dimensions()
        .map_err(|error| format!("could not inspect image: {error}"))?;
    let sha256 = hash_image_file(&path)?;
    let metadata = serde_json::json!({ "path": path, "mime_type": mime_type_for_format(format)?, "width": width, "height": height, "sha256": sha256 });
    Ok(serde_json::json!({ "content": [{ "type": "text", "text": metadata.to_string() }] }))
}

fn hash_image_file(path: &std::path::Path) -> Result<String, String> {
    let file = File::open(path).map_err(|error| format!("could not read image: {error}"))?;
    if file
        .metadata()
        .map_err(|error| format!("could not inspect image size: {error}"))?
        .len()
        > MAX_INSPECT_FILE_BYTES
    {
        return Err("image exceeds 64 MiB inspection limit".into());
    }
    let mut reader = std::io::BufReader::new(file);
    let mut hasher = Sha256::new();
    let mut buffer = [0_u8; 64 * 1024];
    loop {
        let count = reader
            .read(&mut buffer)
            .map_err(|error| format!("could not read image: {error}"))?;
        if count == 0 {
            break;
        }
        hasher.update(&buffer[..count]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

pub(crate) fn mime_type_for_path(path: &std::path::Path) -> Result<&'static str, String> {
    let reader = ImageReader::open(path)
        .map_err(|error| format!("could not open rendered output: {error}"))?
        .with_guessed_format()
        .map_err(|error| format!("could not detect rendered output format: {error}"))?;
    mime_type_for_format(
        reader
            .format()
            .ok_or("could not detect rendered output format")?,
    )
}

fn mime_type_for_format(format: ImageFormat) -> Result<&'static str, String> {
    match format {
        ImageFormat::Png => Ok("image/png"),
        ImageFormat::Gif => Ok("image/gif"),
        ImageFormat::Jpeg => Ok("image/jpeg"),
        ImageFormat::WebP => Ok("image/webp"),
        _ => Err("unsupported image format".into()),
    }
}

#[cfg(test)]
mod tests {

    use crate::handlers::expected_revision;

    use crate::inspect::MAX_INSPECT_FILE_BYTES;
    use crate::inspect::hash_image_file;
    use crate::inspect::inspect_image;
    use crate::inspect::mime_type_for_format;

    use crate::output::validate_output_path;

    use image::ImageFormat;

    #[test]
    fn validates_output_paths_and_inspects_exact_file_bytes() {
        assert!(validate_output_path(std::path::Path::new("result.png"), "png").is_ok());
        assert!(validate_output_path(std::path::Path::new("result.gif"), "png").is_err());
        assert_eq!(mime_type_for_format(ImageFormat::Gif), Ok("image/gif"));
        assert!(mime_type_for_format(ImageFormat::Bmp).is_err());

        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("inspect.png");
        image::RgbaImage::from_pixel(3, 2, image::Rgba([1, 2, 3, 4]))
            .save_with_format(&path, ImageFormat::Png)
            .unwrap();
        let response = inspect_image(&serde_json::json!({ "path": path })).unwrap();
        let metadata: serde_json::Value =
            serde_json::from_str(response["content"][0]["text"].as_str().unwrap()).unwrap();
        assert_eq!(metadata["mime_type"], "image/png");
        assert_eq!(metadata["width"], 3);
        assert_eq!(metadata["height"], 2);
        assert_eq!(metadata["sha256"].as_str().unwrap().len(), 64);

        assert_eq!(
            expected_revision(&serde_json::json!({ "expected_revision": 4 })),
            Ok(Some(4))
        );
        assert_eq!(expected_revision(&serde_json::json!({})), Ok(None));
        for value in [
            serde_json::json!(0),
            serde_json::json!(-1),
            serde_json::json!(1.5),
            serde_json::json!("4"),
        ] {
            assert!(expected_revision(&serde_json::json!({ "expected_revision": value })).is_err());
        }

        let large_path = directory.path().join("large.png");
        std::fs::File::create(&large_path)
            .unwrap()
            .set_len(MAX_INSPECT_FILE_BYTES + 1)
            .unwrap();
        assert_eq!(
            hash_image_file(&large_path),
            Err("image exceeds 64 MiB inspection limit".into())
        );
    }
}
