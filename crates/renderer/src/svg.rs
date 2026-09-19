use crate::{error::RenderError, limits::SVG_RASTER_TIME_BUDGET};
use resvg::{tiny_skia, usvg};
use std::{path::Path, sync::Arc};

/// Extension-based SVG detection, mirroring [`ensure_png_output_path`]'s
/// case-insensitive extension check.
pub(crate) fn has_svg_extension(path: &Path) -> bool {
    path.extension()
        .and_then(|extension| extension.to_str())
        .is_some_and(|extension| extension.eq_ignore_ascii_case("svg"))
}

/// Cheap content sniff so a `.svg`-named file that is not actually SVG (or is
/// empty/binary garbage) fails with a clear diagnostic instead of being
/// handed to the XML parser. Mirrors the spirit of `image`'s own
/// magic-byte format guessing for raster assets.
pub(crate) fn looks_like_svg(bytes: &[u8]) -> bool {
    let prefix_len = bytes.len().min(4096);
    let Ok(prefix) = std::str::from_utf8(&bytes[..prefix_len]) else {
        return false;
    };
    let trimmed = prefix.trim_start_matches('\u{feff}').trim_start();
    trimmed.starts_with("<?xml") || trimmed.starts_with("<svg") || trimmed.contains("<svg")
}

/// Rasterizes an SVG document to an `image::RgbaImage` of exactly
/// `width`x`height` pixels.
///
/// Security posture (see `AGENTS.md`: "Resolve assets locally only; do not
/// introduce implicit remote asset fetching"):
///
/// - `usvg` never performs network I/O of any kind (confirmed by reading the
///   `usvg` 0.48 source: `ImageHrefResolver`'s doc comment states it plainly,
///   and there is no HTTP client anywhere in its dependency tree).
/// - The only way an SVG can reach outside this call is via `<image
///   xlink:href="...">` (or a `<style>`/font-family reference, neither of
///   which `usvg` resolves from the filesystem at all). `usvg`'s *default*
///   string-href resolver treats the href as a filesystem path relative to
///   `Options::resources_dir` -- but critically, `PathBuf::join` treats an
///   *absolute* href as replacing the base entirely, so a default-configured
///   `resources_dir` does NOT stop `<image href="/etc/passwd">` (or a `..`
///   traversal) from escaping the asset root.
///
///   To close that off completely rather than merely "scope it", the
///   `resolve_string` resolver below is replaced with one that returns
///   `None` unconditionally: embedded `<image href="...">` references to
///   *any* local file path are refused, full stop. Only self-contained
///   `data:` URIs (handled by `resolve_data`, which never touches the
///   filesystem) are honored for embedded images. This is strictly more
///   restrictive than scoping to the asset root, so there is no residual
///   path-escape risk from embedded image hrefs.
/// - `fontdb` is left empty (the `system-fonts` cargo feature is disabled and
///   `load_system_fonts()` is never called), so text glyph lookups cannot
///   read arbitrary font files from the host either; SVGs with `<text>` will
///   render without glyphs rather than pulling in system state.
/// - Residual risk: `usvg` cannot be configured to refuse XML parsing
///   entirely (that is the whole point of this function), and a
///   sufficiently adversarial-but-under-the-node-limit document (e.g. many
///   chained blur filters) could still be CPU-expensive to rasterize. That
///   residual is bounded by `SVG_RASTER_TIME_BUDGET` below, on top of
///   `usvg`'s own 1,000,000-node parse limit and the existing
///   `MAX_ASSET_BYTES`/`MAX_IMAGE_RASTER_PIXELS`-derived output-size caps
///   this function's caller already enforces.
pub(crate) fn rasterize_svg(
    data: &[u8],
    asset_root: &Path,
    source: &str,
    width: u32,
    height: u32,
) -> Result<image::RgbaImage, RenderError> {
    let data = Arc::new(data.to_vec());
    let asset_root = asset_root.to_path_buf();
    let source_label = source.to_string();
    let (sender, receiver) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let result = rasterize_svg_blocking(&data, &asset_root, &source_label, width, height);
        // The receiver may already be gone if we hit the timeout below; that
        // is fine, the render result is simply dropped.
        let _ = sender.send(result);
    });
    match receiver.recv_timeout(SVG_RASTER_TIME_BUDGET) {
        Ok(result) => result,
        Err(_) => Err(RenderError::Asset(format!(
            "svg '{source}' exceeded the {}s rasterization time budget",
            SVG_RASTER_TIME_BUDGET.as_secs()
        ))),
    }
}

fn rasterize_svg_blocking(
    data: &[u8],
    asset_root: &Path,
    source: &str,
    width: u32,
    height: u32,
) -> Result<image::RgbaImage, RenderError> {
    let image_href_resolver = usvg::ImageHrefResolver {
        resolve_data: usvg::ImageHrefResolver::default_data_resolver(),
        resolve_string: Box::new(|_href: &str, _options: &usvg::Options| {
            // Deliberately refuse every filesystem-path-shaped `href`: see
            // the security-posture comment on `rasterize_svg` above.
            None
        }),
    };
    // `..Default::default()` leaves `fontdb` at `usvg::Options::default()`'s
    // empty `fontdb::Database`: with the `system-fonts` cargo feature
    // disabled and `load_system_fonts()` never called, no host font files
    // are ever read (see the security-posture comment above).
    let options = usvg::Options {
        resources_dir: Some(asset_root.to_path_buf()),
        image_href_resolver,
        ..Default::default()
    };
    let tree = usvg::Tree::from_data(data, &options)
        .map_err(|error| RenderError::Asset(format!("could not parse svg '{source}': {error}")))?;

    let mut pixmap = tiny_skia::Pixmap::new(width, height).ok_or_else(|| {
        RenderError::Asset(format!(
            "svg '{source}' has an invalid raster target size {width}x{height}"
        ))
    })?;
    let tree_size = tree.size();
    let scale_x = if tree_size.width() > 0.0 {
        width as f32 / tree_size.width()
    } else {
        1.0
    };
    let scale_y = if tree_size.height() > 0.0 {
        height as f32 / tree_size.height()
    } else {
        1.0
    };
    let transform = tiny_skia::Transform::from_scale(scale_x, scale_y);
    resvg::render(&tree, transform, &mut pixmap.as_mut());

    // `Pixmap` stores premultiplied alpha internally; the rest of this
    // renderer's textured-quad pipeline (and the `image` crate decode path
    // above) works in straight alpha, so demultiply on the way out.
    let rgba = pixmap.take_demultiplied();
    image::RgbaImage::from_raw(width, height, rgba).ok_or_else(|| {
        RenderError::Asset(format!(
            "failed to assemble rasterized buffer for svg '{source}'"
        ))
    })
}

#[cfg(test)]
mod tests {
    use crate::assets::*;
    use crate::error::*;
    use std::fs;

    #[test]
    fn rasterizes_svg_assets_to_declared_dimensions_without_a_gpu() {
        let directory = tempfile::tempdir().unwrap();
        fs::write(
            directory.path().join("badge.svg"),
            br##"<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 10 10">
<rect width="10" height="10" fill="#ff0000"/>
</svg>"##,
        )
        .unwrap();

        // Rasterizing directly at a non-square declared size (rather than
        // decoding at some source resolution and bilinearly rescaling)
        // should still produce an exact widthxheight buffer with crisp,
        // uniform color -- there is nothing to blur since the whole
        // viewBox is one flat rect.
        let image = load_image(directory.path(), "badge.svg", 40, 20).unwrap();
        assert_eq!(image.width(), 40);
        assert_eq!(image.height(), 20);
        for pixel in image.pixels() {
            assert_eq!(pixel.0, [255, 0, 0, 255]);
        }
    }

    #[test]
    fn rejects_svg_assets_that_escape_the_asset_root() {
        let directory = tempfile::tempdir().unwrap();
        let nested = directory.path().join("nested");
        fs::create_dir(&nested).unwrap();

        // Same two escape shapes already covered for raster images by the
        // `resolve_asset` assertions in
        // `rasterizes_text_images_and_constrained_assets_without_a_gpu`
        // above: `..` traversal and an absolute path. `load_image` routes
        // every source (SVG included) through the exact same
        // `resolve_asset` call, so both are rejected before the file is
        // ever opened.
        assert!(matches!(
            load_image(&nested, "../escape.svg", 8, 8),
            Err(RenderError::Asset(_))
        ));
        assert!(matches!(
            load_image(directory.path(), "/absolute-escape.svg", 8, 8),
            Err(RenderError::Asset(_))
        ));
    }

    #[test]
    fn rejects_malformed_and_oversized_svg_assets_without_panicking() {
        let directory = tempfile::tempdir().unwrap();

        fs::write(
            directory.path().join("malformed.svg"),
            b"<svg><unterminated",
        )
        .unwrap();
        assert!(matches!(
            load_image(directory.path(), "malformed.svg", 8, 8),
            Err(RenderError::Asset(_))
        ));

        // Named `.svg` but the content sniff should refuse to hand this to
        // the XML parser at all.
        fs::write(
            directory.path().join("not-svg.svg"),
            b"this has an .svg extension but is not svg content",
        )
        .unwrap();
        assert!(matches!(
            load_image(directory.path(), "not-svg.svg", 8, 8),
            Err(RenderError::Asset(_))
        ));

        // Same 16 MiB cap the raster (`image` crate) path already enforces
        // via `MAX_ASSET_BYTES`, applied before the file is ever parsed.
        let mut oversized = b"<svg xmlns=\"http://www.w3.org/2000/svg\">".to_vec();
        oversized.resize(17 * 1024 * 1024, b' ');
        oversized.extend_from_slice(b"</svg>");
        fs::write(directory.path().join("oversized.svg"), &oversized).unwrap();
        assert!(matches!(
            load_image(directory.path(), "oversized.svg", 8, 8),
            Err(RenderError::Asset(_))
        ));
    }

    #[test]
    fn refuses_to_follow_embedded_svg_image_references_outside_the_asset_root() {
        let directory = tempfile::tempdir().unwrap();

        // A file outside the configured asset root that a hostile SVG will
        // try to pull in via an absolute-path `<image href>`.
        let secret = directory.path().join("secret.png");
        image::RgbaImage::from_pixel(2, 2, image::Rgba([255, 0, 0, 255]))
            .save(&secret)
            .unwrap();

        let asset_root = directory.path().join("assets");
        fs::create_dir(&asset_root).unwrap();
        fs::write(
            asset_root.join("evil.svg"),
            format!(
                r##"<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" viewBox="0 0 4 4">
<rect width="4" height="4" fill="#00ff00"/>
<image xlink:href="{}" width="4" height="4"/>
</svg>"##,
                secret.display()
            ),
        )
        .unwrap();

        let image = load_image(&asset_root, "evil.svg", 4, 4).unwrap();
        // The embedded absolute-path href must be refused outright (see the
        // security-posture comment on `rasterize_svg`): only the green
        // background rect should ever be visible, never the referenced
        // file's red pixels.
        for pixel in image.pixels() {
            assert_eq!(pixel.0, [0, 255, 0, 255]);
        }
    }
}
