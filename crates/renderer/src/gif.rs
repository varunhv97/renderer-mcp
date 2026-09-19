use crate::error::RenderError;
use std::{
    borrow::Cow,
    collections::{BTreeSet, HashMap},
    fs::File,
    io::Write,
    path::Path,
};

/// Passed to `color_quant::NeuQuant` as its `samplefac` (range `[1, 30]`,
/// lower is higher quality and slower). Matches the speed the old per-frame
/// `image`-crate path used: speed 1 (that crate's default) explicitly
/// prioritizes quantization quality "at any cost", and with MSAA-anti-aliased
/// edges introducing far more distinct colors than hard edges did, that cost
/// became real (~2x slower GIF encoding on a moderately complex scene,
/// measured locally); speed 10 recovers most of that with no visible quality
/// difference (checked side-by-side on several test scenes).
pub(crate) const GIF_QUANTIZATION_SPEED: i32 = 10;

/// Upper bound on how many pixels [`encode_gif_with_shared_palette`] trains
/// its shared `NeuQuant` network on, via [`subsample_for_palette_training`].
///
/// `color_quant::NeuQuant`'s training cost (`NeuQuant::learn`) scales
/// linearly with however many pixels it's handed, divided by `samplefac`
/// (`GIF_QUANTIZATION_SPEED`) -- it has no notion of "this is already
/// enough". Handing it every pixel of a long animation (as an earlier
/// version of this function did) trains on `frame_count` times more data
/// than a single frame would, costing roughly as much in total as the old
/// per-frame quantization it was meant to replace -- measured directly: an
/// early version of this change made a 120-frame export *slower*
/// (6.36s vs. 5.5s) than the per-frame baseline it was supposed to improve
/// on. Training on a bounded, evenly-strided sample instead keeps that cost
/// roughly constant regardless of frame count or canvas size. Every actual
/// output pixel is still mapped to the trained palette afterwards via
/// `NeuQuant::index_of`, a cheap nearest-neighbor lookup rather than a
/// training update (see `search_netindex`'s early-exit search), so output
/// quality doesn't suffer from the palette itself only having been trained
/// on a sample.
pub(crate) const PALETTE_TRAINING_PIXEL_BUDGET: usize = 200_000;

/// Returns an evenly strided subset of `pixels` (RGBA, 4 bytes per pixel)
/// with at most `budget` pixels, spread across the whole buffer so a long
/// animation's later frames contribute samples too, not just its first one.
/// See [`PALETTE_TRAINING_PIXEL_BUDGET`] for why this exists.
pub(crate) fn subsample_for_palette_training(pixels: &[u8], budget: usize) -> Vec<u8> {
    let total_pixels = pixels.len() / 4;
    if total_pixels <= budget {
        return pixels.to_vec();
    }
    let stride = total_pixels.div_ceil(budget);
    pixels
        .as_chunks::<4>()
        .0
        .iter()
        .step_by(stride)
        .flat_map(|pixel| pixel.iter().copied())
        .collect()
}

/// Encodes `pixels` -- every frame of one animated scene's RGBA output,
/// concatenated back to back into one flat, contiguous buffer of
/// `frame_count * width * height * 4` bytes -- as a single GIF at `output`,
/// sharing one color palette across every frame instead of, as the `image`
/// crate's high-level `GifEncoder` does, quantizing each frame independently.
///
/// Profiling `render_gif_with_asset_root` (see `architecture.md`) found
/// per-frame NeuQuant training, not GPU compositing, was the dominant cost of
/// an animated export: isolating frame count from scene complexity showed a
/// consistent ~43ms of marginal cost per frame even after GPU render targets
/// stopped being reallocated every frame (`FrameTargets`), matching
/// `color_quant`'s own training cost far more than a few-millisecond GPU
/// composite + readback. Training one shared network once, on every frame's
/// pixels at once, instead of `frame_count` independent times, turns that
/// into the dominant one-time cost of the whole export rather than a
/// per-frame recurring one.
///
/// Mirrors `gif::Frame::from_rgba_speed`'s own semantics, just applied across
/// every frame at once instead of to one frame at a time: any non-zero alpha
/// becomes fully opaque (the GIF format has no partial transparency), the
/// first fully-transparent pixel encountered pins the one RGBA value every
/// other fully-transparent pixel -- in any frame -- is normalized to (the
/// whole animation can mark only one palette index as "the" transparent
/// color, so every frame has to agree on which color that is), and an exact,
/// lossless palette is used whenever the whole animation fits in 256 colors,
/// falling back to NeuQuant only when it genuinely doesn't.
pub(crate) fn encode_gif_with_shared_palette(
    output: &Path,
    width: u32,
    height: u32,
    pixels: &mut [u8],
    frame_count: u32,
    fps: u32,
    speed: i32,
) -> Result<(), RenderError> {
    let gif_width = u16::try_from(width).unwrap_or(u16::MAX);
    let gif_height = u16::try_from(height).unwrap_or(u16::MAX);
    let frame_len = (width as usize) * (height as usize) * 4;

    let mut transparent_color: Option<[u8; 4]> = None;
    for pixel in pixels.as_chunks_mut::<4>().0 {
        if pixel[3] != 0 {
            pixel[3] = 0xFF;
            continue;
        }
        match transparent_color {
            Some([r, g, b, a]) => {
                pixel[0] = r;
                pixel[1] = g;
                pixel[2] = b;
                pixel[3] = a;
            }
            None => transparent_color = Some([pixel[0], pixel[1], pixel[2], pixel[3]]),
        }
    }

    let file = File::create(output).map_err(RenderError::OutputDirectory)?;
    let delay_cs = (100 / fps.max(1)).clamp(1, u32::from(u16::MAX)) as u16;

    // As with the per-frame path this replaces, prefer an exact palette when
    // the whole animation uses 256 colors or fewer -- lossless, and cheap to
    // build -- and only reach for NeuQuant when it genuinely doesn't fit.
    let mut colors: BTreeSet<(u8, u8, u8, u8)> = BTreeSet::new();
    let mut exact = true;
    for pixel in pixels.as_chunks::<4>().0 {
        if colors.insert((pixel[0], pixel[1], pixel[2], pixel[3])) && colors.len() > 256 {
            exact = false;
            break;
        }
    }

    if exact {
        let colors: Vec<(u8, u8, u8, u8)> = colors.into_iter().collect();
        let palette: Vec<u8> = colors.iter().flat_map(|&(r, g, b, _a)| [r, g, b]).collect();
        let index_of: HashMap<(u8, u8, u8, u8), u8> = colors.into_iter().zip(0u8..).collect();
        let transparent_index = transparent_color.map(|[r, g, b, a]| index_of[&(r, g, b, a)]);
        let mut encoder = gif::Encoder::new(file, gif_width, gif_height, &palette)
            .map_err(RenderError::GifEncoding)?;
        encoder
            .set_repeat(gif::Repeat::Infinite)
            .map_err(RenderError::GifEncoding)?;
        write_gif_frames(
            &mut encoder,
            pixels,
            frame_len,
            frame_count,
            gif_width,
            gif_height,
            delay_cs,
            transparent_index,
            |pixel| index_of[&(pixel[0], pixel[1], pixel[2], pixel[3])],
        )
    } else {
        let training_sample = subsample_for_palette_training(pixels, PALETTE_TRAINING_PIXEL_BUDGET);
        let quant = color_quant::NeuQuant::new(speed, 256, &training_sample);
        let palette = quant.color_map_rgb();
        let transparent_index = transparent_color.map(|color| quant.index_of(&color) as u8);
        let mut encoder = gif::Encoder::new(file, gif_width, gif_height, &palette)
            .map_err(RenderError::GifEncoding)?;
        encoder
            .set_repeat(gif::Repeat::Infinite)
            .map_err(RenderError::GifEncoding)?;
        write_gif_frames(
            &mut encoder,
            pixels,
            frame_len,
            frame_count,
            gif_width,
            gif_height,
            delay_cs,
            transparent_index,
            |pixel| quant.index_of(pixel) as u8,
        )
    }
}

/// Writes every `frame_len`-byte slice of `pixels` as one GIF frame, mapping
/// each pixel to a palette index via `index_of` (either an exact-palette
/// hash lookup or a shared `NeuQuant` network -- see
/// `encode_gif_with_shared_palette`). `palette: None` on every frame so each
/// one is decoded against the encoder's global palette instead of carrying
/// its own local one.
// Every parameter is a genuinely distinct piece of the frame/encoding
// context (dimensions, timing, transparency, the pixel data itself, and how
// to quantize it); a params struct for this one internal helper would only
// add indirection, not clarity -- same call as `add_rect_analytic` above.
#[allow(clippy::too_many_arguments)]
pub(crate) fn write_gif_frames<W: Write>(
    encoder: &mut gif::Encoder<W>,
    pixels: &[u8],
    frame_len: usize,
    frame_count: u32,
    width: u16,
    height: u16,
    delay_cs: u16,
    transparent: Option<u8>,
    index_of: impl Fn(&[u8]) -> u8,
) -> Result<(), RenderError> {
    for frame_pixels in pixels.chunks_exact(frame_len).take(frame_count as usize) {
        let indices: Vec<u8> = frame_pixels
            .as_chunks::<4>()
            .0
            .iter()
            .map(|pixel| index_of(pixel))
            .collect();
        let frame = gif::Frame {
            delay: delay_cs,
            // Every frame here is a full, opaque-or-transparent-by-alpha
            // composite of the whole canvas (never a sparse/partial update),
            // so `Keep` and `Background` are behaviorally identical; picked
            // to match what the `image` crate's own encoder used to set.
            dispose: gif::DisposalMethod::Background,
            transparent,
            width,
            height,
            palette: None,
            buffer: Cow::Owned(indices),
            ..gif::Frame::default()
        };
        encoder
            .write_frame(&frame)
            .map_err(RenderError::GifEncoding)?;
    }
    Ok(())
}
