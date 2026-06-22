// Streams raw RGB24 frames from stdin to stdout through `MoshEngine`, with the
// codec backend and preset selectable on the command line. Unlike the `raw-mosh`
// CLI subcommand (motion codec only), this reaches every engine backend and is
// handy for generating side-by-side demo clips without a plugin host:
//
//   ffmpeg ... -f rawvideo -pix_fmt rgb24 - \
//     | cargo run --release --example raw_engine -- <w> <h> <backend 1|2|3|4> <preset>
//     | ffmpeg -f rawvideo -pixel_format rgb24 -video_size <w>x<h> -i - out.mp4
//
// backend: 1 = raw_mosh_v1 (MSH0 motion), 2 = scanline_signal_v1 (SCN0),
//          3 = dct_transform_v1 (DCT0), 4 = wavelet_pyramid_v1 (WVT0).

use std::io::{self, BufReader, BufWriter, Read, Write};

use datamosh::{MoshEngine, MoshEngineBackend};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let width: usize = args
        .first()
        .ok_or("usage: raw_engine <width> <height> <backend 1|2|3|4> <preset>")?
        .parse()?;
    let height: usize = args.get(1).ok_or("missing <height>")?.parse()?;
    let backend_id: u32 = args.get(2).map_or(Ok(1), |s| s.parse())?;
    let preset = args.get(3).map(String::as_str).unwrap_or("classic");

    let backend = MoshEngineBackend::parse_id(backend_id)
        .ok_or("backend must be 1 (motion), 2 (scanline), 3 (dct), or 4 (wavelet)")?;
    let frame_len = width
        .checked_mul(height)
        .and_then(|p| p.checked_mul(3))
        .ok_or("frame dimensions overflow")?;

    let mut engine = MoshEngine::with_backend(backend, width, height)?;
    engine.set_preset(preset)?;

    let mut input = vec![0_u8; frame_len];
    let mut output = vec![0_u8; frame_len];
    let mut reader = BufReader::new(io::stdin().lock());
    let mut writer = BufWriter::new(io::stdout().lock());

    loop {
        match reader.read_exact(&mut input) {
            Ok(()) => {}
            Err(err) if err.kind() == io::ErrorKind::UnexpectedEof => break,
            Err(err) => return Err(err.into()),
        }
        engine.process_rgb24(&input, &mut output)?;
        writer.write_all(&output)?;
    }
    writer.flush()?;
    Ok(())
}
