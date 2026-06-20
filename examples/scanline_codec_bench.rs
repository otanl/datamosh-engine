use std::time::Instant;

use datamosh::{
    ScanlineCodec, ScanlineCodecConfig, ScanlineGlitchParams, load_scanline_signal_preset,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let width = parse_arg(&args, 0, 640_usize)?;
    let height = parse_arg(&args, 1, 360_usize)?;
    let frames = parse_arg(&args, 2, 120_usize)?;
    let preset = args
        .get(3)
        .map(String::as_str)
        .unwrap_or("composite-collapse");

    let frame_len = width
        .checked_mul(height)
        .and_then(|pixels| pixels.checked_mul(3))
        .ok_or("frame dimensions overflow")?;
    let mut input = vec![0_u8; frame_len];
    let mut output = vec![0_u8; frame_len];
    let mut params = ScanlineGlitchParams::default();
    load_scanline_signal_preset(preset, &mut params)?;
    let mut codec = ScanlineCodec::new(ScanlineCodecConfig::new(width, height))?;

    let started = Instant::now();
    let mut checksum = 0_u64;
    for frame in 0..frames {
        fill_frame(&mut input, width, height, frame);
        codec.process_rgb_frame(&input, &params, &mut output)?;
        checksum = checksum.wrapping_add(
            output
                .iter()
                .step_by((frame_len / 1024).max(1))
                .map(|value| *value as u64)
                .sum::<u64>(),
        );
    }

    let elapsed = started.elapsed().as_secs_f64();
    let stats = codec.stats();
    let ratio = if stats.encoded_bytes == 0 {
        0.0
    } else {
        stats.raw_bytes as f64 / stats.encoded_bytes as f64
    };
    println!(
        "{width}x{height} preset={preset} frames={frames} elapsed={elapsed:.3}s fps={:.2} ms/frame={:.2} compression={ratio:.2}:1 checksum={checksum}",
        frames as f64 / elapsed,
        elapsed * 1000.0 / frames as f64,
    );
    Ok(())
}

fn parse_arg<T>(args: &[String], index: usize, default: T) -> Result<T, T::Err>
where
    T: std::str::FromStr,
{
    args.get(index).map_or(Ok(default), |value| value.parse())
}

fn fill_frame(frame: &mut [u8], width: usize, height: usize, frame_index: usize) {
    let bar_x = (frame_index * 3) % width.max(1);
    let bar_y = (frame_index * 2) % height.max(1);
    for y in 0..height {
        for x in 0..width {
            let offset = (y * width + x) * 3;
            let moving = x.abs_diff(bar_x) < 24 || y.abs_diff(bar_y) < 12;
            frame[offset] = ((x + frame_index) & 0xff) as u8;
            frame[offset + 1] = ((y * 2 + frame_index * 3) & 0xff) as u8;
            frame[offset + 2] = if moving { 240 } else { ((x ^ y) & 0xff) as u8 };
        }
    }
}
