use crate::error::RenderError;
use renderer_schema::Color;
use sha2::{Digest, Sha256};
use std::{fs, path::Path};

pub(crate) fn to_wgpu_color(color: Color) -> wgpu::Color {
    wgpu::Color {
        r: color[0] as f64,
        g: color[1] as f64,
        b: color[2] as f64,
        a: color[3] as f64,
    }
}

pub(crate) fn align_to(value: u32, alignment: u32) -> u32 {
    value.div_ceil(alignment) * alignment
}

pub(crate) fn hash_file(path: &Path) -> Result<String, RenderError> {
    Ok(format!(
        "{:x}",
        Sha256::digest(fs::read(path).map_err(RenderError::OutputRead)?)
    ))
}

pub(crate) fn ensure_png_output_path(output: &Path) -> Result<(), RenderError> {
    if output
        .extension()
        .and_then(|extension| extension.to_str())
        .is_some_and(|extension| extension.eq_ignore_ascii_case("png"))
    {
        Ok(())
    } else {
        Err(RenderError::InvalidPngOutputPath(output.into()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::*;
    use std::path::Path;

    #[test]
    fn aligns_copy_rows() {
        assert_eq!(align_to(256, 256), 256);
        assert_eq!(align_to(260, 256), 512);
    }

    #[test]
    fn accepts_only_png_render_paths() {
        assert!(ensure_png_output_path(Path::new("scene.png")).is_ok());
        assert!(ensure_png_output_path(Path::new("scene.PNG")).is_ok());
        assert!(matches!(
            ensure_png_output_path(Path::new("scene.gif")),
            Err(RenderError::InvalidPngOutputPath(_))
        ));
        assert!(matches!(
            ensure_png_output_path(Path::new("scene")),
            Err(RenderError::InvalidPngOutputPath(_))
        ));
    }
}
