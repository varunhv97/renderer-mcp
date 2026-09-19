// -- GIF decoding -----------------------------------------------------------

use crate::error::TerminalError;
use image::codecs::gif::GifDecoder;
use image::{AnimationDecoder, ImageFormat, RgbaImage};
use std::fs;
use std::io::Cursor;
use std::path::Path;

pub(crate) fn decode_gif_frames(path: &Path) -> Result<Vec<(RgbaImage, u32)>, TerminalError> {
    let file = fs::File::open(path).map_err(|source| TerminalError::Read {
        path: path.to_path_buf(),
        source,
    })?;
    let decoder =
        GifDecoder::new(std::io::BufReader::new(file)).map_err(|source| TerminalError::Image {
            path: path.to_path_buf(),
            source,
        })?;
    let frames = decoder
        .into_frames()
        .collect_frames()
        .map_err(|source| TerminalError::Image {
            path: path.to_path_buf(),
            source,
        })?;
    Ok(frames
        .into_iter()
        .map(|frame| {
            let (numerator, denominator) = frame.delay().numer_denom_ms();
            let delay_ms = numerator.checked_div(denominator).unwrap_or(numerator);
            (frame.into_buffer(), delay_ms.max(1))
        })
        .collect())
}

fn encode_frame_png(path: &Path, image: &RgbaImage) -> Result<Vec<u8>, TerminalError> {
    let mut bytes = Vec::new();
    let mut cursor = Cursor::new(&mut bytes);
    image::DynamicImage::ImageRgba8(image.clone())
        .write_to(&mut cursor, ImageFormat::Png)
        .map_err(|source| TerminalError::Image {
            path: path.to_path_buf(),
            source,
        })?;
    Ok(bytes)
}

pub(crate) fn encode_frames_as_png(
    path: &Path,
    frames: &[(RgbaImage, u32)],
) -> Result<Vec<(Vec<u8>, u32)>, TerminalError> {
    frames
        .iter()
        .map(|(image, delay)| Ok((encode_frame_png(path, image)?, *delay)))
        .collect()
}
