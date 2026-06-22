use std::time::Instant;

use datamosh::{WaveletCodec, WaveletCodecConfig, WaveletGlitchParams, load_wavelet_preset};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let width: usize = args.first().map(String::as_str).unwrap_or("640").parse()?;
    let height: usize = args.get(1).map(String::as_str).unwrap_or("360").parse()?;
    let frames: usize = args.get(2).map(String::as_str).unwrap_or("120").parse()?;
    let preset = args
        .get(3)
        .map(String::as_str)
        .unwrap_or("hierarchy-collapse");
    let estimate_rate = args.get(4).is_some_and(|value| value == "rate");

    let config = WaveletCodecConfig::new(width, height);
    let mut codec = WaveletCodec::new(config)?;
    codec.set_rate_estimation(estimate_rate);
    let mut params = WaveletGlitchParams::default();
    load_wavelet_preset(preset, &mut params)?;

    let mut input = vec![0_u8; width * height * 3];
    let mut output = vec![0_u8; input.len()];
    let start = Instant::now();
    for frame in 0..frames {
        for y in 0..height {
            for x in 0..width {
                let offset = (y * width + x) * 3;
                input[offset] = ((x + frame * 3) & 0xff) as u8;
                input[offset + 1] = ((y * 2 + frame * 5) & 0xff) as u8;
                input[offset + 2] = (((x ^ y) + frame * 7) & 0xff) as u8;
            }
        }
        codec.process_rgb_frame(&input, &params, &mut output)?;
    }
    let elapsed = start.elapsed().as_secs_f64();
    let checksum = output
        .iter()
        .enumerate()
        .fold(0_u64, |hash, (index, value)| {
            hash.wrapping_mul(1_099_511_628_211)
                .wrapping_add(*value as u64 + index as u64)
        });
    let stats = codec.stats();
    let rate = if estimate_rate {
        format!(
            " estimated-compression={:.2}:1",
            stats.raw_bytes as f64 / stats.estimated_bytes.max(1) as f64
        )
    } else {
        String::new()
    };
    println!(
        "{width}x{height} preset={preset} frames={frames} elapsed={elapsed:.3}s fps={:.2} ms/frame={:.2}{rate} checksum={checksum}",
        frames as f64 / elapsed,
        elapsed * 1000.0 / frames as f64,
    );
    Ok(())
}
