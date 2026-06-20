# datamosh-engine

Realtime datamoshing core built around a Rust stream filter, FFmpeg-compatible elementary streams, and an experimental custom motion codec.

![Three datamosh codec backends glitching the same Mandelbrot zoom — motion (MSH0), scanline (SCN0), DCT (DCT0)](assets/datamosh.gif)

*The same Mandelbrot zoom pushed through all three custom codec backends: **MSH0 motion** (`classic` — stale motion vectors with residuals dropped, so it melts), **SCN0 scanline/signal** (`predictor-ghost` — analog-style line tearing and ghosting), and **DCT0 transform** (`bleed` — JPEG-style block DC smear). Three distinct glitch families on identical input, made with a pure FFmpeg → engine → FFmpeg pipe — no plugin host. The two CUDA TOPs are GPU-native re-implementations of the motion and DCT paths, kept at parity by hand — same look, on the GPU.*

<details>
<summary>Regenerate this GIF</summary>

The full pipeline (render each backend, label, tile 1×3, palette-optimize) lives in [`scripts/make-demo-gif.sh`](scripts/make-demo-gif.sh):

```bash
bash scripts/make-demo-gif.sh        # writes assets/datamosh.gif
```

Its core is one FFmpeg → engine → FFmpeg pipe per backend (`raw_engine` arg 3 selects the backend: 1 = motion, 2 = scanline, 3 = dct):

```bash
cargo build --release --example raw_engine
ffmpeg -f lavfi -i "mandelbrot=size=854x480:rate=30" -t 8 -vf format=rgb24 -f rawvideo -pix_fmt rgb24 - \
  | target/release/examples/raw_engine 854 480 1 classic \
  | ffmpeg -f rawvideo -pixel_format rgb24 -video_size 854x480 -framerate 30 -i - -pix_fmt yuv420p out.mp4
```

Swap the backend id and preset (`datamosh raw-mosh --help` lists the motion presets; `touchdesigner/README.md` lists the scanline/DCT patterns).
</details>

The current tool reads compressed elementary streams from stdin and writes modified compressed streams to stdout. It supports H.264 Annex B, H.265/HEVC Annex B, MPEG-4 Part 2/ASP `.m4v` including Xvid/DivX-style streams, MPEG-1 Video, and MPEG-2 Video elementary streams. It drops selected keyframes, damages predicted frame payloads, truncates payloads, scrambles bytes, rotates payload regions, splices/grows with previous or donor payload data, injects previous units, replaces units, XORs with previous or donor units, rewrites MPEG frame type headers, and repeats predicted units. It also includes an experimental raw RGB24 `MoshCodec` that encodes frames into block motion vectors plus residuals, then decodes them through controllable glitch parameters. The custom MSH0 bitstream path can also damage serialized motion/residual data before decode, including residual payload byte-slip to simulate decoder state desync, transform-coefficient corruption for a frequency-domain glitch path, and residual codebook/dictionary misreads. Two further custom codec backends sit alongside it: an SCN0 predictive scanline/signal codec, and a DCT0 intra transform codec (8x8 DCT, 4:2:0 chroma, and a real canonical-Huffman entropy bitstream — JPEG/DV-style), each with its own coefficient-domain and serialized-stream glitch families.

The CLI is now only a test harness around the library API. The intended direction is to embed the Rust core in a plugin, with TouchDesigner or another host handling capture, UI, audio analysis, and output routing.

## Requirements

- Rust toolchain
- FFmpeg and FFplay on `PATH`

## Build

```powershell
cargo build --release
```

## Performance Check

The dependency-free benchmark example exercises the same `MoshEngine` path used by plugin hosts:

```powershell
cargo run --release --example raw_mosh_bench -- 640 360 120 codebook
cargo run --release --example raw_mosh_bench -- 320 180 120 melt
```

It reports frames per second, milliseconds per frame, and a deterministic checksum. Use the checksum to catch output changes while optimizing the codec; timing is only comparable when runs use the same machine, preset, resolution, and build profile.

Frames of at least 200,000 pixels use a shared Rayon worker pool for motion search and non-overlap decode. The pool defaults to half the available logical CPUs, capped at 16 threads to leave capacity for a plugin host. Set `DATAMOSH_THREADS` before the first codec frame to override the worker count.

## Raw-Mosh Demo GUI

For quick visual checks on Windows, use the lightweight preset launcher:

```powershell
.\scripts\raw-mosh-demo.cmd
```

It starts the existing OBS Virtual Camera pipeline and opens ffplay as the preview window. The launcher is only a test harness for the custom raw motion codec; it does not add a new rendering path. The default demo list is curated to `drift`, `bank`, `plane`, `vector`, `residue`, `entropy`, `coeff`, `codebook`, `melt`, and `unstable`. Enable `Show all presets` to expose the more redundant comparison presets such as `scan`, `pixel`, `grain`, and `classic`.

While running with `Realtime control` enabled, selecting another preset sends a UDP control message to `raw-mosh` and keeps the video pipe alive. The macro sliders send `controls intensity motion residual temporal bitstream`; the parameter slider sends `set <id> <value>` for the currently selected preset group. `Reset glitch` sends `reset-glitch`, clearing dirty reference history and residual codebook state without changing the current parameters. Changing resolution, upscale, preview port, or extra startup arguments still requires `Apply` because those belong to the FFmpeg/process pipeline. With `Keep ffplay` enabled, preview transport uses local UDP/MPEG-TS after `raw-mosh`; the raw-mosh glitch path itself is unchanged.

The raw-mosh control socket is also usable outside the GUI:

```powershell
.\target\release\datamosh-cli.exe raw-mosh --width 480 --height 270 --preset drift --control-port 24000

# Example UDP messages:
# preset codebook
# controls 0.8 1.0 0.6 1.0 1.0
# set codebook_replace_every 4
# reset-glitch
```

The `entropy` preset is intentionally milder by default. Use `--preset entropy-hard` or extra args such as `--bs-entropy-slip-every 1 --bs-entropy-windows 10 --bs-entropy-resync-bytes 8192` to restore the harsher byte-slip behavior.

The `coeff` preset damages transformed residual coefficient tiles before normal decode. Use `--preset coeff-hard` or extra args such as `--bs-coeff-shift 5 --bs-coeff-sign-flip-every 7 --bs-coeff-zero-high 9 --bs-coeff-quant 8` for a stronger frequency-domain break.

The `codebook` preset stores clean residual tiles in a decoder dictionary and later decodes selected tiles from older or shifted dictionary slots. Use `--preset codebook-hard` or extra args such as `--bs-codebook-every 4 --bs-codebook-slots 128 --bs-codebook-stride -31 --bs-codebook-shuffle-every 2` for more frequent texture intrusion.

The GUI can also switch to an FFmpeg test pattern when OBS is not running. To inspect the presets without opening the GUI:

```powershell
.\scripts\raw-mosh-demo.cmd -ListPresets
.\scripts\raw-mosh-demo.cmd -PrintDefaultCommand
.\scripts\raw-mosh-demo.cmd -PrintPersistentCommands
.\scripts\raw-mosh-demo.cmd -SmokeGui
```

## Library API

Use `DatamoshStream` when input arrives in arbitrary chunks. This is the path a TouchDesigner plugin or FFI wrapper should use.

```rust
use datamosh::{Codec, Config, DatamoshStream, FrameTypeRewrite};

let mut stream = DatamoshStream::new(Config {
    codec: Codec::Mpeg2,
    drop_idr_after: 1,
    recover_every: 8,
    xor_slice_every: 5,
    xor_amount: 32,
    shift_slice_address_every: 1,
    shift_slice_address_by: 3,
    donor_splice_slice_every: 5,
    donor_splice_amount: 24,
    rewrite_frame_type_every: 6,
    rewrite_frame_type_to: FrameTypeRewrite::B,
    splice_slice_every: 7,
    splice_amount: 24,
    quiet: true,
    ..Config::default()
});

let mut output = Vec::new();
stream.process_chunk(input_bytes, &mut output)?;
stream.finish(&mut output)?;
```

If you have a second compressed stream, feed it into the donor side before or during primary processing. The filter stores predicted compressed units from the donor stream and can splice, grow, XOR, or replace primary predicted units with donor payloads.

```rust
use datamosh::load_donor_stream;

let mut stream = DatamoshStream::new(Config {
    codec: Codec::H264,
    donor_splice_slice_every: 5,
    donor_splice_amount: 32,
    quiet: true,
    ..Config::default()
});

load_donor_stream(&mut stream, donor_h264_bytes)?;
stream.process_chunk(primary_h264_bytes, &mut output)?;
```

For already separated compressed units, use `MoshFilter::process_unit(...)` directly.

For plugin-style use of the custom internal codec path, prefer `MoshEngine`. It owns the codec, preset state, realtime controls, parameters, and scratch buffers behind one stable object. It does not try to be a storage-efficient video codec; it creates a low-latency, editable motion representation that can be damaged safely.

```rust
use datamosh::{MoshEngine, RawMoshControls};

let mut engine = MoshEngine::new(width, height)?;
engine.set_preset("codebook")?;
engine.set_parameter("codebook_replace_every", audio_hit_period)?;
engine.set_controls(RawMoshControls {
    intensity: audio_envelope,
    motion: 0.8,
    residual: 1.0,
    temporal: 0.6,
    bitstream: 1.0,
});

engine.process_rgba8(input_rgba8, output_rgba8)?;

if beat_reset {
    engine.reset_glitch();
}
```

The C ABI exposes the same boundary for a C++ host. Treat the returned pointer as opaque:

```c
uint32_t backend = datamosh_mosh_engine_default_backend(); /* raw_mosh_v1 */
void* engine = datamosh_mosh_engine_new_with_backend(backend, width, height);
datamosh_mosh_engine_set_preset(engine, "codebook");
datamosh_mosh_engine_set_controls(engine, intensity, motion, residual, temporal, bitstream);
datamosh_mosh_engine_set_parameter(engine, "codebook_replace_every", 4.0f);
datamosh_mosh_engine_process_rgba8(engine, input, input_len, output, output_len);
datamosh_mosh_engine_reset_glitch(engine);
datamosh_mosh_engine_free(engine);
```

The C ABI is intentionally backend-based. It currently exposes:

- `DATAMOSH_BACKEND_RAW_MOSH_V1` (`1`): block motion vectors, residuals, and MSH0 corruption.
- `DATAMOSH_BACKEND_SCANLINE_SIGNAL_V1` (`2`): SCN0 scanline predictive signal codec.
- `DATAMOSH_BACKEND_DCT_TRANSFORM_V1` (`3`): DCT0 intra transform codec.

SCN0 is not an analog-look post effect. Each line is encoded with a sync word, horizontal or temporal luma prediction, quantized/RLE luma residuals, and a phase-multiplexed chroma carrier. SCN0 v7 sends the frame as even and odd scan fields, while decode restores progressive line placement. The receiver carries horizontal clock phase, burst phase, field parity, and expected line address across transmitted scanlines. Field starts are strong anchors; additional resync markers are inserted adaptively according to accumulated compressed payload, bounded to 6-20 transmitted lines.

The dedicated TOP menu is curated to `clean`, `timebase-tear`, `clock-skew`, `sync-dropout`, `chroma-sequence`, `burst-seed-loss`, `carrier-xor`, `predictor-ghost`, `rle-runaway`, `plane-crosswire`, and `composite-collapse`. Lower-level aliases remain available through the Rust/C API for experiments but are intentionally omitted from the operator menu.

```rust
use datamosh::{MoshEngine, MoshEngineBackend};

let mut engine =
    MoshEngine::with_backend(MoshEngineBackend::ScanlineSignalV1, width, height)?;
engine.set_preset("composite-collapse")?;
engine.process_rgba8(input_rgba8, output_rgba8)?;
```

Benchmark the standalone codec path with:

```powershell
cargo run --release --example scanline_codec_bench -- 640 360 120 composite-collapse
```

On the current development machine, SCN0 v7 measured about 7.7-8.0 ms/frame at 640x360 and about 29.5 ms/frame at 1280x720, with roughly 2.21:1 RGB24 compression. The current sequential CPU implementation is not yet a reliable 1080p30 path.

DCT0 (`DATAMOSH_BACKEND_DCT_TRANSFORM_V1`, `3`) is an intra transform codec in the JPEG/DV family: per-frame 8x8-block DCT (a fast even/odd transform), JPEG-style quantization, and 4:2:0 chroma subsampling. Its glitches corrupt the transform domain — quantization, DC/AC coefficients, the differential DC predictor (block-by-block colour smear), zig-zag order, block transpose/remap, and Cb/Cr swap — and an optional `persistence` knob adds temporal feedback through the real pipeline. DCT0 also has a real entropy/bitstream stage (`DTE0`): the quantized coefficients are losslessly coded with DC DPCM + AC run-length + per-frame canonical Huffman, and the `desync`/`shred`/`truncate` presets damage that byte stream so the decoder loses sync (the cascading "broken JPEG" slide). It is block-parallel (Rayon) and, unlike SCN0, also has a GPU-native CUDA reimplementation.

```rust
use datamosh::{MoshEngine, MoshEngineBackend};

let mut engine = MoshEngine::with_backend(MoshEngineBackend::DctTransformV1, width, height)?;
engine.set_preset("dc-smear")?; // or "false-color", "desync", "composite", ...
engine.process_rgba8(input_rgba8, output_rgba8)?;
```

```powershell
cargo run --release --example dct_codec_bench -- 640 360 120 composite
```

Build and run the standalone C++ smoke host with:

```powershell
cargo build --release
.\scripts\build-cpp-smoke.cmd -Run
```

Build the experimental TouchDesigner TOP bridge with:

```powershell
cargo build --release
.\scripts\build-td-top.cmd
```

This writes `target\release\DatamoshTOP.dll` for the motion/residual codec. Build the dedicated SCN0 operator with:

```powershell
.\scripts\build-td-scanline-top.cmd
```

This writes `target\release\ScanlineSignalTOP.dll`. Keep `target\release\datamosh.dll` beside either TOP DLL. The two operators deliberately have separate pattern menus and parameter pages; codec-specific controls are not relabeled or reused across incompatible representations.

The TouchDesigner TOP DLL is built with MSVC. Do not use a MinGW/g++ build for the TOP itself; TouchDesigner's C++ plugin boundary depends on MSVC-compatible C++ class/vtable layout.

For an NVIDIA GPU-native path that avoids texture download and upload, build the separate CUDA TOP:

```powershell
.\scripts\build-td-cuda-top.cmd
.\scripts\build-cuda-smoke.cmd -Run
```

This writes `target\release\DatamoshCudaTOP.dll`. It is a separate TouchDesigner operator because a TOP's CPU/CUDA execution mode is fixed when its plugin DLL is loaded. The CUDA TOP keeps motion vectors, residuals, clean references, dirty references, and residual history on the GPU and does not depend on `datamosh.dll`. It accepts BGRA/RGBA 8-bit, RGBA16 fixed/float, and RGBA32 float inputs and outputs BGRA8, including common Movie File In TOP formats. GPU-only patterns include incorrect row-pitch decode, residual scale mismatch, packet-tile loss, and history-slot weaving. The CPU TOP remains the compatibility fallback and currently exposes the broader parameter set.

All the TOPs use `Clean` at pattern index `0`. The CPU TOPs expose a numeric `Pattern` menu, macro controls, codec-specific override pages, and an `Audio` page that reads a referenced CHOP for realtime macro control and reset triggers. The same `Pattern` parameter displays names in the UI and accepts zero-based CHOP/Python index control. `DatamoshTOP` is fixed to backend `1`, `ScanlineSignalTOP` to backend `2`, and `DatamoshDctTOP` to backend `3`. Keep `Use Overrides` off when auditioning patterns. See `touchdesigner/README.md` for the consolidated operator, pattern-index, parameter-page, and migration reference.

The DCT0 codec has both a CPU TOP (`DatamoshDctTOP`, built with `.\scripts\build-td-dct-top.cmd`) and a GPU-native CUDA TOP (`DatamoshDctCudaTOP`, built with `.\scripts\build-td-dct-cuda-top.cmd`). The CUDA TOP is a separate hand-maintained reimplementation; the parity check (`.\scripts\build-dct-parity-check.cmd`, then `target\release\dct_parity_check.exe`) verifies it still matches the CPU codec per preset.

For lower-level experiments, use `MoshCodec` directly. By default, `MoshCodec` keeps a clean encoder history and a dirty decoder history: frame differences are encoded against the original input stream, while decode can pull active blocks from already-glitched reconstructed frames. Set `reference_mode` to `Feedback` for the older recursive style where glitches also become the next encoder reference.

```rust
use datamosh::{
    apply_raw_mosh_controls, load_raw_mosh_preset, raw_mosh_parameter_infos_for_preset,
    raw_mosh_preset_infos, set_raw_mosh_parameter, MoshBitstreamParams, MoshCodec,
    MoshCodecConfig, MoshGlitchParams, RawMoshControls,
};

let mut config = MoshCodecConfig::new(width, height);
let mut params = MoshGlitchParams::default();
let mut bitstream = MoshBitstreamParams::default();

load_raw_mosh_preset("codebook", &mut config, &mut params, &mut bitstream)?;
let _presets = raw_mosh_preset_infos();
let _ui_parameters = raw_mosh_parameter_infos_for_preset("codebook")?;
let mut codec = MoshCodec::new(config)?;

set_raw_mosh_parameter(
    &mut config,
    &mut params,
    &mut bitstream,
    "codebook_replace_every",
    audio_hit_period,
)?;

apply_raw_mosh_controls(
    &mut params,
    &mut bitstream,
    RawMoshControls {
        intensity: audio_envelope,
        motion: 0.8,
        residual: 1.0,
        temporal: 0.6,
        bitstream: 1.0,
    },
);

codec.process_rgb_frame_bitstream(input_rgb24, &params, &bitstream, output_rgb24)?;
```

`RawMoshControls` is intended for host/plugin control surfaces. Build a fresh preset state with `load_raw_mosh_preset(...)`, then apply normalized control values before processing each frame or parameter block. `intensity` is the master fade; the other fields damp motion-vector, residual, temporal-reference, and MSH0 bitstream corruption independently. For explicit plugin UI, call `raw_mosh_preset_infos()` and `raw_mosh_parameter_infos_for_preset(...)`, then write values with `set_raw_mosh_parameter(...)`. Each `RawMoshParameterInfo` exposes `id`, `kind`, `min`, `max`, `default`, and `is_realtime()`; `block_size`, `search_radius`, and `search_step` require recreating the codec after change.

## Preview The Custom Raw Motion Codec

`raw-mosh` reads and writes raw `rgb24` frames. Start at 640x360 for motion-vector presets. For heavier codec-state presets, process at 480x270 and upscale the output with `--upscale 2` or `--output-width 1280 --output-height 720`. The glitch is still generated at the internal codec resolution; the output scaling is only a display stage. Use `--scale-mode nearest` for crisp damaged pixels, or `--scale-mode linear` when the preview should be less pixelated. Use `--preset melt` or `--preset classic` for datamosh-like motion smear. The default `--reference-mode split` keeps motion/residual analysis tied to the source frames, while the decoder can still smear active blocks from dirty reconstructed frames.

The `melt` preset is not a post blur. It changes the decoder path: active pixels are gated from residual/motion scores, dirty reconstructed references can latch when activity drops, and the decoder hard-switches between clean and dirty references instead of alpha-blending them. It also uses motion-vector predictor desync, sample-level dirty reference-slot misreads, per-pixel motion-vector field interpolation, and small dirty reference address desync, so one bad internal state can smear through later blocks without making the whole frame a uniform block grid.

Motion-only melt, closest to classic interframe datamoshing:

```powershell
cmd /c "ffmpeg -f dshow -rtbufsize 256M -video_size 1280x720 -framerate 30 -pixel_format yuv420p -i video=""OBS Virtual Camera"" -an -vf scale=640:360,format=rgb24 -f rawvideo - | .\target\release\datamosh-cli.exe raw-mosh --width 640 --height 360 --preset melt --block-size 16 --search-radius 8 --search-step 4 | ffplay -fflags nobuffer -flags low_delay -framedrop -f rawvideo -pixel_format rgb24 -video_size 640x360 -framerate 30 -"
```

Pixel-grain dirty reference tearing:

```powershell
cmd /c "ffmpeg -hide_banner -loglevel error -f dshow -rtbufsize 64M -video_size 1280x720 -framerate 30 -pixel_format yuv420p -i video=""OBS Virtual Camera"" -an -vf scale=640:360:flags=fast_bilinear,format=rgb24 -f rawvideo - | .\target\release\datamosh-cli.exe raw-mosh --width 640 --height 360 --preset pixel --history 16 --quiet | ffplay -hide_banner -loglevel error -fflags nobuffer -flags low_delay -framedrop -f rawvideo -pixel_format rgb24 -video_size 640x360 -framerate 30 -"
```

Medium-grain tearing between `melt` and `pixel`:

```powershell
cmd /c "ffmpeg -hide_banner -loglevel error -f dshow -rtbufsize 64M -video_size 1280x720 -framerate 30 -pixel_format yuv420p -i video=""OBS Virtual Camera"" -an -vf scale=640:360:flags=fast_bilinear,format=rgb24 -f rawvideo - | .\target\release\datamosh-cli.exe raw-mosh --width 640 --height 360 --preset grain --history 16 --quiet | ffplay -hide_banner -loglevel error -fflags nobuffer -flags low_delay -framedrop -f rawvideo -pixel_format rgb24 -video_size 640x360 -framerate 30 -"
```

Residual stream desync, a different direction from motion smearing. This preset uses a cheaper internal encode path; 480x270 internal processing with 2x display upscale is the safer live starting point.

```powershell
cmd /c "ffmpeg -hide_banner -loglevel fatal -f dshow -rtbufsize 256M -video_size 1280x720 -framerate 30 -pixel_format yuv420p -i video=""OBS Virtual Camera"" -an -vf scale=480:270:flags=fast_bilinear,format=rgb24 -f rawvideo - | .\target\release\datamosh-cli.exe raw-mosh --width 480 --height 270 --preset residue --history 16 --upscale 2 --scale-mode nearest --quiet | ffplay -hide_banner -loglevel fatal -fflags nobuffer -flags low_delay -framedrop -f rawvideo -pixel_format rgb24 -video_size 960x540 -framerate 30 -"
```

Scanline reference-history desync:

```powershell
cmd /c "ffmpeg -hide_banner -loglevel fatal -f dshow -rtbufsize 256M -video_size 1280x720 -framerate 30 -pixel_format yuv420p -i video=""OBS Virtual Camera"" -an -vf scale=480:270:flags=fast_bilinear,format=rgb24 -f rawvideo - | .\target\release\datamosh-cli.exe raw-mosh --width 480 --height 270 --preset scan --history 16 --upscale 2 --scale-mode nearest --quiet | ffplay -hide_banner -loglevel fatal -fflags nobuffer -flags low_delay -framedrop -f rawvideo -pixel_format rgb24 -video_size 960x540 -framerate 30 -"
```

Temporal slice drift. Horizontal bands read different decoded reference ages, and the band phase drifts over time:

```powershell
cmd /c "ffmpeg -hide_banner -loglevel fatal -f dshow -rtbufsize 256M -video_size 1280x720 -framerate 30 -pixel_format yuv420p -i video=""OBS Virtual Camera"" -an -vf scale=480:270:flags=fast_bilinear,format=rgb24 -f rawvideo - | .\target\release\datamosh-cli.exe raw-mosh --width 480 --height 270 --preset drift --history 16 --upscale 2 --scale-mode nearest --quiet | ffplay -hide_banner -loglevel fatal -fflags nobuffer -flags low_delay -framedrop -f rawvideo -pixel_format rgb24 -video_size 960x540 -framerate 30 -"
```

Residual bank swap. Motion/reference stays relatively readable, while the residual stream is decoded from wrong cells:

```powershell
cmd /c "ffmpeg -hide_banner -loglevel fatal -f dshow -rtbufsize 256M -video_size 1280x720 -framerate 30 -pixel_format yuv420p -i video=""OBS Virtual Camera"" -an -vf scale=480:270:flags=fast_bilinear,format=rgb24 -f rawvideo - | .\target\release\datamosh-cli.exe raw-mosh --width 480 --height 270 --preset bank --history 16 --upscale 2 --scale-mode nearest --quiet | ffplay -hide_banner -loglevel fatal -fflags nobuffer -flags low_delay -framedrop -f rawvideo -pixel_format rgb24 -video_size 960x540 -framerate 30 -"
```

Channel plane desync. RGB reference planes read different channels and different decoded reference ages:

```powershell
cmd /c "ffmpeg -hide_banner -loglevel fatal -f dshow -rtbufsize 256M -video_size 1280x720 -framerate 30 -pixel_format yuv420p -i video=""OBS Virtual Camera"" -an -vf scale=480:270:flags=fast_bilinear,format=rgb24 -f rawvideo - | .\target\release\datamosh-cli.exe raw-mosh --width 480 --height 270 --preset plane --history 16 --upscale 2 --scale-mode nearest --quiet | ffplay -hide_banner -loglevel fatal -fflags nobuffer -flags low_delay -framedrop -f rawvideo -pixel_format rgb24 -video_size 960x540 -framerate 30 -"
```

Motion-vector bank desync. The decoder applies vectors from the wrong block banks, so the motion field itself tears:

```powershell
cmd /c "ffmpeg -hide_banner -loglevel fatal -f dshow -rtbufsize 256M -video_size 1280x720 -framerate 30 -pixel_format yuv420p -i video=""OBS Virtual Camera"" -an -vf scale=480:270:flags=fast_bilinear,format=rgb24 -f rawvideo - | .\target\release\datamosh-cli.exe raw-mosh --width 480 --height 270 --preset vector --history 16 --upscale 2 --scale-mode nearest --quiet | ffplay -hide_banner -loglevel fatal -fflags nobuffer -flags low_delay -framedrop -f rawvideo -pixel_format rgb24 -video_size 960x540 -framerate 30 -"
```

Same 480x270 internal stream, but displayed at 1280x720:

```powershell
cmd /c "ffmpeg -hide_banner -loglevel fatal -f dshow -rtbufsize 256M -video_size 1280x720 -framerate 30 -pixel_format yuv420p -i video=""OBS Virtual Camera"" -an -vf scale=480:270:flags=fast_bilinear,format=rgb24 -f rawvideo - | .\target\release\datamosh-cli.exe raw-mosh --width 480 --height 270 --preset residue --history 16 --output-width 1280 --output-height 720 --scale-mode linear --quiet | ffplay -hide_banner -loglevel fatal -fflags nobuffer -flags low_delay -framedrop -f rawvideo -pixel_format rgb24 -video_size 1280x720 -framerate 30 -"
```

Classic motion smear:

```powershell
cmd /c "ffmpeg -f dshow -rtbufsize 256M -video_size 1280x720 -framerate 30 -pixel_format yuv420p -i video=""OBS Virtual Camera"" -an -vf scale=640:360,format=rgb24 -f rawvideo - | .\target\release\datamosh-cli.exe raw-mosh --width 640 --height 360 --preset classic --block-size 16 --search-radius 8 --search-step 4 | ffplay -fflags nobuffer -flags low_delay -framedrop -f rawvideo -pixel_format rgb24 -video_size 640x360 -framerate 30 -"
```

More subtle and readable:

```powershell
cmd /c "ffmpeg -f dshow -rtbufsize 256M -video_size 1280x720 -framerate 30 -pixel_format yuv420p -i video=""OBS Virtual Camera"" -an -vf scale=640:360,format=rgb24 -f rawvideo - | .\target\release\datamosh-cli.exe raw-mosh --width 640 --height 360 --preset subtle --block-size 16 --search-radius 8 --search-step 4 | ffplay -fflags nobuffer -flags low_delay -framedrop -f rawvideo -pixel_format rgb24 -video_size 640x360 -framerate 30 -"
```

More visible motion drift with extra internal-codec artifacts:

```powershell
cmd /c "ffmpeg -f dshow -rtbufsize 256M -video_size 1280x720 -framerate 30 -pixel_format yuv420p -i video=""OBS Virtual Camera"" -an -vf scale=640:360,format=rgb24 -f rawvideo - | .\target\release\datamosh-cli.exe raw-mosh --width 640 --height 360 --preset balanced --block-size 16 --search-radius 8 --search-step 4 | ffplay -fflags nobuffer -flags low_delay -framedrop -f rawvideo -pixel_format rgb24 -video_size 640x360 -framerate 30 -"
```

Manual motion-area datamosh tuning:

```powershell
cmd /c "ffmpeg -f dshow -rtbufsize 256M -video_size 1280x720 -framerate 30 -pixel_format yuv420p -i video=""OBS Virtual Camera"" -an -vf scale=640:360,format=rgb24 -f rawvideo - | .\target\release\datamosh-cli.exe raw-mosh --width 640 --height 360 --activity-mode active --activity-threshold 12 --residual-keep 0.2 --reference-lag 4 --mv-jitter 1 --mv-quant 2 --block-size 16 --search-radius 8 --search-step 4 | ffplay -fflags nobuffer -flags low_delay -framedrop -f rawvideo -pixel_format rgb24 -video_size 640x360 -framerate 30 -"
```

Stickier, more broken previous-frame retention:

```powershell
cmd /c "ffmpeg -f dshow -rtbufsize 64M -video_size 1280x720 -framerate 30 -pixel_format yuv420p -i video=""OBS Virtual Camera"" -an -vf scale=640:360:flags=fast_bilinear,format=rgb24 -f rawvideo - | .\target\release\datamosh-cli.exe raw-mosh --width 640 --height 360 --history 16 --activity-mode active --activity-threshold 10 --activity-softness 0 --reference-bleed 0.22 --reference-latch-frames 8 --reference-slots 10 --reference-slot-shuffle-every 7 --residual-keep 0.0 --reference-lag 12 --overlap 0 --motion-diffusion 0 --mv-field-interpolation 0.85 --sample-address-desync 1.2 --mv-predictor-desync-every 7 --mv-predictor-desync-x 3 --mv-predictor-desync-y -2 --mv-jitter 1 --mv-quant 2 --block-size 16 --search-radius 8 --search-step 4 --quiet | ffplay -hide_banner -loglevel error -fflags nobuffer -flags low_delay -framedrop -f rawvideo -pixel_format rgb24 -video_size 640x360 -framerate 30 -"
```

More unpredictable codec-state corruption:

```powershell
cmd /c "ffmpeg -hide_banner -loglevel error -f dshow -rtbufsize 64M -video_size 1280x720 -framerate 30 -pixel_format yuv420p -i video=""OBS Virtual Camera"" -an -vf scale=640:360:flags=fast_bilinear,format=rgb24 -f rawvideo - | .\target\release\datamosh-cli.exe raw-mosh --width 640 --height 360 --preset unstable --history 16 --block-size 16 --search-radius 8 --search-step 4 --quiet | ffplay -hide_banner -loglevel error -fflags nobuffer -flags low_delay -framedrop -f rawvideo -pixel_format rgb24 -video_size 640x360 -framerate 30 -"
```

Recursive feedback variant, more likely to over-melt or collapse:

```powershell
cmd /c "ffmpeg -f dshow -rtbufsize 256M -video_size 1280x720 -framerate 30 -pixel_format yuv420p -i video=""OBS Virtual Camera"" -an -vf scale=640:360,format=rgb24 -f rawvideo - | .\target\release\datamosh-cli.exe raw-mosh --width 640 --height 360 --preset melt --reference-mode feedback --block-size 16 --search-radius 8 --search-step 4 | ffplay -fflags nobuffer -flags low_delay -framedrop -f rawvideo -pixel_format rgb24 -video_size 640x360 -framerate 30 -"
```

Dedicated MSH0 bitstream mutation path:

```powershell
cmd /c "ffmpeg -f dshow -rtbufsize 256M -video_size 1280x720 -framerate 30 -pixel_format yuv420p -i video=""OBS Virtual Camera"" -an -vf scale=640:360,format=rgb24 -f rawvideo - | .\target\release\datamosh-cli.exe raw-mosh --width 640 --height 360 --preset classic --bitstream --bs-residual-zero-every 3 --bs-mv-sign-flip-every 7 --bs-block-shift-every 11 --bs-block-shift-x 1 --block-size 16 --search-radius 8 --search-step 4 | ffplay -fflags nobuffer -flags low_delay -framedrop -f rawvideo -pixel_format rgb24 -video_size 640x360 -framerate 30 -"
```

The original aggressive example is now available as `destroy`:

```powershell
cmd /c "ffmpeg -f dshow -rtbufsize 256M -video_size 1280x720 -framerate 30 -pixel_format yuv420p -i video=""OBS Virtual Camera"" -an -vf scale=640:360,format=rgb24 -f rawvideo - | .\target\release\datamosh-cli.exe raw-mosh --width 640 --height 360 --preset destroy --block-size 16 --search-radius 8 --search-step 4 | ffplay -fflags nobuffer -flags low_delay -framedrop -f rawvideo -pixel_format rgb24 -video_size 640x360 -framerate 30 -"
```

## Preview A File In Realtime With H.264

On Windows PowerShell, run binary H.264 pipelines through `cmd /c`. Plain PowerShell pipes can corrupt the byte stream.

```powershell
cmd /c "ffmpeg -re -stream_loop -1 -i input.mp4 -an -c:v libx264 -preset ultrafast -tune zerolatency -g 24 -keyint_min 24 -bf 0 -x264-params scenecut=0 -f h264 - | .\target\release\datamosh-cli.exe filter --codec h264 --drop-keyframe-after 1 --recover-every 8 | ffplay -fflags nobuffer -flags low_delay -framedrop -f h264 -"
```

The encode settings are chosen for low latency and predictable GOPs:

- `-g 24 -keyint_min 24`: insert regular IDR frames
- `-bf 0`: disable B-frames, keeping the stream simpler for live glitching
- `-tune zerolatency`: reduce encoder buffering
- `-f h264`: write raw Annex B H.264 for the Rust filter

## Preview A File In Realtime With H.265/HEVC

HEVC is common in MP4 workflows. This does not mutate the MP4 container; it mutates the HEVC elementary stream that can be carried inside MP4. Decoder recovery can be harsher than H.264/MPEG-2, so start with lighter payload operations.

```powershell
cmd /c "ffmpeg -re -stream_loop -1 -i input.mp4 -an -c:v libx265 -preset ultrafast -tune zerolatency -x265-params keyint=24:min-keyint=24:scenecut=0:bframes=0 -f hevc - | .\target\release\datamosh-cli.exe filter --codec hevc --drop-keyframe-after 1 --recover-every 8 --rotate-slice-every 5 --rotate-amount 12 --splice-slice-every 11 --splice-amount 16 | ffplay -fflags nobuffer -flags low_delay -framedrop -f hevc -"
```

## Preview A File In Realtime With MPEG-4 Part 2 / ASP

MPEG-4 Part 2 / ASP is the classic Xvid/DivX datamosh family. `--codec mpeg4`, `--codec xvid`, `--codec divx`, and `--codec mpeg4-asp` all select the same parser.

```powershell
cmd /c "ffmpeg -re -stream_loop -1 -i input.mp4 -an -c:v mpeg4 -q:v 4 -g 24 -bf 0 -f m4v - | .\target\release\datamosh-cli.exe filter --codec mpeg4 --drop-keyframe-after 1 --recover-every 8 | ffplay -fflags nobuffer -flags low_delay -framedrop -f m4v -"
```

If `libxvid` is available, this variant gives a related but different MPEG-4 Part 2 bitstream:

```powershell
cmd /c "ffmpeg -re -stream_loop -1 -i input.mp4 -an -c:v libxvid -q:v 4 -g 24 -bf 0 -f m4v - | .\target\release\datamosh-cli.exe filter --codec mpeg4 --drop-keyframe-after 1 --recover-every 8 --damage-slice-every 5 --damage-amount 2 | ffplay -fflags nobuffer -flags low_delay -framedrop -f m4v -"
```

Frame type rewrite is a bitstream-header lie: the payload stays encoded as one frame class, while the VOP type bits are changed before decode.

```powershell
cmd /c "ffmpeg -re -stream_loop -1 -i input.mp4 -an -c:v mpeg4 -q:v 4 -g 24 -bf 0 -f m4v - | .\target\release\datamosh-cli.exe filter --codec mpeg4 --drop-keyframe-after 99 --rewrite-frame-type-every 3 --rewrite-frame-type-to b | ffplay -fflags nobuffer -flags low_delay -framedrop -f m4v -"
```

## Preview With A Donor Stream

The donor path mixes compressed payloads from a second elementary stream into the primary stream. For a repeatable file test, first make a donor elementary stream:

```powershell
ffmpeg -y -i donor.mp4 -an -c:v libx264 -preset ultrafast -tune zerolatency -g 24 -keyint_min 24 -bf 0 -x264-params scenecut=0 -f h264 donor.h264
```

Then play the primary stream while using donor payloads:

```powershell
cmd /c "ffmpeg -re -stream_loop -1 -i input.mp4 -an -c:v libx264 -preset ultrafast -tune zerolatency -g 24 -keyint_min 24 -bf 0 -x264-params scenecut=0 -f h264 - | .\target\release\datamosh-cli.exe filter --codec h264 --donor-file donor.h264 --drop-keyframe-after 1 --recover-every 8 --donor-splice-slice-every 5 --donor-splice-amount 32 | ffplay -fflags nobuffer -flags low_delay -framedrop -f h264 -"
```

For MPEG-2, donor mixing tends to be more forgiving and more visibly structural:

```powershell
ffmpeg -y -i donor.mp4 -an -c:v mpeg2video -q:v 4 -g 24 -bf 0 -f mpeg2video donor.m2v
cmd /c "ffmpeg -re -stream_loop -1 -i input.mp4 -an -c:v mpeg2video -q:v 4 -g 24 -bf 0 -f mpeg2video - | .\target\release\datamosh-cli.exe filter --codec mpeg2 --donor-file donor.m2v --drop-keyframe-after 1 --recover-every 8 --donor-splice-slice-every 5 --donor-splice-amount 32 --donor-xor-slice-every 9 --donor-xor-amount 24 | ffplay -fflags nobuffer -flags low_delay -framedrop -f mpegvideo -"
```

## Preview A File In Realtime With MPEG-2 Video

MPEG-2 Video is useful for realtime pure glitch experiments because picture and slice start codes are easy to manipulate, and decoder recovery tends to be forgiving.

```powershell
cmd /c "ffmpeg -re -stream_loop -1 -i input.mp4 -an -c:v mpeg2video -q:v 4 -g 24 -bf 0 -f mpeg2video - | .\target\release\datamosh-cli.exe filter --codec mpeg2 --drop-keyframe-after 1 --recover-every 8 --xor-slice-every 5 --xor-amount 32 | ffplay -fflags nobuffer -flags low_delay -framedrop -f mpegvideo -"
```

MPEG-1/2 also supports structural slice address mutations. These change which macroblock row a slice claims to belong to, which is closer to bitstream surgery than a normal visual effect.

```powershell
cmd /c "ffmpeg -re -stream_loop -1 -i input.mp4 -an -c:v mpeg2video -q:v 4 -g 24 -bf 0 -f mpeg2video - | .\target\release\datamosh-cli.exe filter --codec mpeg2 --drop-keyframe-after 1 --recover-every 8 --shift-slice-address-every 1 --shift-slice-address-by 3 | ffplay -fflags nobuffer -flags low_delay -framedrop -f mpegvideo -"
```

Partial I-picture slice deletion is useful when full keyframe removal is too uniform. It lets only selected MPEG-1/2 slice rows refresh, leaving other rows contaminated by prior prediction.

```powershell
cmd /c "ffmpeg -re -stream_loop -1 -i input.mp4 -an -c:v mpeg2video -q:v 4 -g 24 -bf 0 -f mpeg2video - | .\target\release\datamosh-cli.exe filter --codec mpeg2 --drop-keyframe-after 99 --drop-mpeg-slice-address-every 2 --drop-mpeg-slice-address-phase 0 --drop-mpeg-slice-address-mode key --splice-slice-every 7 --splice-amount 24 | ffplay -fflags nobuffer -flags low_delay -framedrop -f mpegvideo -"
```

MPEG-1/2 picture type rewrite changes the picture coding type bits while leaving the coded slices alone.

```powershell
cmd /c "ffmpeg -re -stream_loop -1 -i input.mp4 -an -c:v mpeg2video -q:v 4 -g 24 -bf 0 -f mpeg2video - | .\target\release\datamosh-cli.exe filter --codec mpeg2 --drop-keyframe-after 99 --rewrite-frame-type-every 3 --rewrite-frame-type-to b --shift-slice-address-every 4 --shift-slice-address-by 1 | ffplay -fflags nobuffer -flags low_delay -framedrop -f mpegvideo -"
```

## Preview A File In Realtime With MPEG-1 Video

MPEG-1 Video is another old GOP codec and uses the same picture/slice mutation path as MPEG-2. It tends to look rougher and more retro.

```powershell
cmd /c "ffmpeg -re -stream_loop -1 -i input.mp4 -an -c:v mpeg1video -q:v 4 -g 24 -bf 0 -f mpeg1video - | .\target\release\datamosh-cli.exe filter --codec mpeg1 --drop-keyframe-after 1 --recover-every 8 --splice-slice-every 7 --splice-amount 24 --rotate-slice-every 5 --rotate-amount 12 | ffplay -fflags nobuffer -flags low_delay -framedrop -f mpegvideo -"
```

## Preview A Windows Camera

List DirectShow devices:

```powershell
ffmpeg -list_devices true -f dshow -i dummy
```

Then replace `DEVICE NAME`:

```powershell
cmd /c "ffmpeg -f dshow -i video=""DEVICE NAME"" -an -c:v libx264 -preset ultrafast -tune zerolatency -g 24 -keyint_min 24 -bf 0 -x264-params scenecut=0 -f h264 - | .\target\release\datamosh-cli.exe filter --codec h264 --drop-keyframe-after 1 --recover-every 8 | ffplay -fflags nobuffer -flags low_delay -framedrop -f h264 -"
```

## CLI

```text
datamosh-cli filter [options]
datamosh-cli raw-mosh --width <px> --height <px> [raw options]
datamosh-cli [options]

Options:
  --codec <h264|hevc|mpeg4|mpeg1|mpeg2>
                              Input elementary stream codec. Default: h264
  --drop-keyframe-after <n>  Pass the first n keyframes, then drop keyframes. Default: 1
  --drop-idr-after <n>       Alias for --drop-keyframe-after.
  --recover-every <n>        After initial drops, pass every nth later keyframe. 0 disables recovery.
  --drop-slice-every <n>     Drop every nth predicted slice/VOP. 0 disables dropping.
  --damage-slice-every <n>   Corrupt every nth predicted payload. 0 disables payload damage.
  --damage-amount <n>        Bytes to flip in each damaged slice. Default: 4
  --truncate-slice-every <n> Truncate every nth predicted slice/VOP. 0 disables truncation.
  --truncate-amount <n>      Tail bytes to remove from truncated slices/VOPs. Default: 16
  --scramble-slice-every <n> Byte-scramble every nth predicted slice/VOP. 0 disables scrambling.
  --scramble-amount <n>      Payload bytes involved in each scramble. Default: 16
  --rotate-slice-every <n>   Rotate bytes inside every nth predicted payload. 0 disables rotation.
  --rotate-amount <n>        Payload bytes involved in each rotate. Default: 8
  --splice-slice-every <n>   Copy previous payload bytes into every nth payload. 0 disables splice.
  --splice-amount <n>        Payload bytes copied by splice. Default: 32
  --grow-slice-every <n>     Insert previous payload bytes into every nth unit. 0 disables grow.
  --grow-amount <n>          Payload bytes inserted by grow. Default: 8
  --donor-file <path>        Load a second elementary stream and use its predicted payloads as donors.
  --donor-bank-size <n>      Donor predicted unit history size. 0 disables donor storage. Default: 16
  --donor-splice-slice-every <n>
                              Copy donor payload bytes into every nth payload. 0 disables.
  --donor-splice-amount <n>  Donor payload bytes copied by splice. Default: 32
  --donor-grow-slice-every <n>
                              Insert donor payload bytes into every nth unit. 0 disables.
  --donor-grow-amount <n>    Donor payload bytes inserted by grow. Default: 8
  --donor-xor-slice-every <n>
                              XOR every nth predicted unit with a donor unit. 0 disables.
  --donor-xor-amount <n>     Donor payload bytes involved in each XOR. Default: 16
  --donor-replace-slice-every <n>
                              Replace every nth predicted unit with a donor unit. 0 disables.
  --rewrite-frame-type-every <n>
                              Rewrite every nth MPEG-4/MPEG-1/2 frame type header. 0 disables.
  --rewrite-frame-type-to <i|p|b|s|d>
                              Target frame type for MPEG frame type rewrite. Default: p
  --shift-slice-address-every <n>
                              Shift every nth MPEG-1/2 slice address. 0 disables address shift.
  --shift-slice-address-by <n>
                              Signed MPEG-1/2 slice address offset. Default: 1
  --drop-mpeg-slice-address-every <n>
                              Drop MPEG-1/2 slices whose address matches this period. 0 disables.
  --drop-mpeg-slice-address-phase <n>
                              Address phase for MPEG-1/2 partial slice drop. Default: 0
  --drop-mpeg-slice-address-mode <all|key|predicted>
                              Picture type scope for MPEG-1/2 partial slice drop. Default: all
  --xor-slice-every <n>      XOR every nth predicted unit with the previous unit. 0 disables XOR.
  --xor-amount <n>           Payload bytes involved in each XOR. Default: 16
  --echo-slice-every <n>     Inject previous predicted unit before every nth unit. 0 disables echo.
  --echo-count <n>           Previous-unit copies to inject on echo. Default: 1
  --replace-slice-every <n>  Replace every nth predicted unit with the previous unit. 0 disables replace.
  --repeat-slice-every <n>   Repeat every nth predicted slice/VOP. 0 disables repeats.
  --repeat-count <n>         Extra copies to write when repeating. Default: 1
  --drop-headers-after-first Drop repeated SPS/PPS headers after the first pair for harsher glitches.
  --quiet                    Do not print realtime stats to stderr.
```

`raw-mosh` options:

```text
--width <n>                  Raw RGB24 frame width. Required.
--height <n>                 Raw RGB24 frame height. Required.
--output-width <n>           Raw RGB24 output width after display scaling.
--output-height <n>          Raw RGB24 output height after display scaling.
--upscale <n>                Integer output scale factor. Cannot be combined with output size.
--scale-mode <nearest|linear>
                              Display scaling mode. Default: nearest.
--preset <clean|subtle|classic|melt|grain|pixel|residue|scan|drift|bank|plane|vector|entropy|coeff|codebook|unstable|balanced|destroy>
                              Load a parameter preset. Later options can override it.
--block-size <n>             Motion block size. Default: 16
--search-radius <n>          Motion search radius. Default: 8
--search-step <n>            Motion search step. Default: 4
--keyframe-every <n>         Internal I-frame period. 0 means first only.
--history <n>                Reference history length. Default: 8
--reference-mode <split|feedback>
                              split keeps encoder input history clean; feedback re-encodes glitches.
--mv-scale <n>               Scale both motion-vector axes.
--mv-scale-x <n>             Scale horizontal vectors.
--mv-scale-y <n>             Scale vertical vectors.
--mv-jitter <n>              Deterministic signed vector jitter.
--mv-quant <n>               Quantize vectors to a pixel grid.
--reference-lag <n>          Decode from an older reconstructed frame.
--residual-keep <n>          Residual multiplier. 1 reconstructs, 0 pure motion smear.
--residual-invert-every <n>  Invert residual every nth block.
--residual-address-shift-x <n>
                              Read residual samples from a horizontally shifted address.
--residual-address-shift-y <n>
                              Read residual samples from a vertically shifted address.
--residual-address-jitter <n>
                              Add deterministic residual address jitter up to n pixels.
--residual-channel-shift <n> Rotate residual channel reads.
--temporal-slice-height <n>
                              Horizontal stripe height for temporal slice drift. 0 disables.
--temporal-slice-lag-span <n>
                              Reference history span used by temporal slice drift. 0 disables.
--temporal-slice-drift <n>   Signed lag drift applied each latch bucket.
--residual-bank-size <n>     Residual cell size for residual-bank misreads. 0 disables.
--residual-bank-stride <n>   Signed residual-bank offset used when reading residuals.
--residual-bank-shuffle-every <n>
                              Randomize selected residual-bank offsets every nth bank. 0 disables.
--reference-channel-shift <n>
                              Rotate reference channel reads.
--reference-channel-lag-span <n>
                              Reference history span for per-channel plane desync. 0 disables.
--reference-channel-lag-stride <n>
                              Signed per-channel reference lag offset.
--mv-bank-size <n>           Motion-vector bank cell size in blocks. 0 disables.
--mv-bank-stride <n>         Signed motion-vector bank offset.
--mv-bank-shuffle-every <n>
                              Randomize selected motion-vector bank offsets every nth bank. 0 disables.
--block-remap-every <n>      Borrow another block's vector every nth block.
--block-remap-stride <n>     Vector-bank offset for block remap.
--channel-shift <n>          Shift G/B reference sampling.
--activity-mode <all|active|static>
                              Apply glitch everywhere, only changed/moving blocks, or only static blocks.
--activity-threshold <n>     Difference threshold for activity gating. Try 8-24.
--activity-softness <n>      Soft transition range above the activity threshold. 0 is binary.
--reference-bleed <n>        Minimum hard-switch chance to dirty references, 0.0-1.0.
--reference-latch-frames <n> Keep dirty-reference switch decisions stable for this many frames.
--reference-slots <n>        Dirty decoded reference slots available for sample-level misreads.
--reference-slot-shuffle-every <n>
                              Use a wrong dirty reference slot on distributed sample cells.
--reference-scanline-height <n>
                              Stripe height for scanline reference-history desync. 0 disables.
--reference-scanline-lag-span <n>
                              Reference history span used by scanline desync. 0 disables.
--overlap <n>                Overlapped block-compensation radius in pixels. 0 disables.
--motion-diffusion <n>       Blend motion vectors toward neighboring vectors, 0.0-1.0.
--mv-field-interpolation <n> Interpolate decoded motion vectors per pixel, 0.0-1.0.
--sample-address-desync <n>  Corrupt dirty reference sample addresses by up to n pixels.
--pixel-grain <n>            Pixel-cell size for sample-level glitches. 0 uses tuned defaults;
                              1 hashes every pixel independently.
--pixel-grain-x <n>          Horizontal pixel-cell size. Overrides --pixel-grain on X.
--pixel-grain-y <n>          Vertical pixel-cell size. Overrides --pixel-grain on Y.
--mv-predictor-desync-every <n>
                              Desync the motion-vector predictor on every nth block.
--mv-predictor-desync-x <n>  Horizontal predictor desync delta.
--mv-predictor-desync-y <n>  Vertical predictor desync delta.
--wrap-motion                Wrap out-of-range motion sampling.
--clamp-motion               Clamp out-of-range motion sampling. Default.
--bitstream                  Serialize to MSH0 packet bytes and decode through the bitstream path.
--bs-mv-sign-flip-every <n>  Flip serialized motion-vector signs every nth block.
--bs-mv-delta-every <n>      Add a serialized motion-vector delta every nth block.
--bs-mv-delta-x <n>          Horizontal delta for --bs-mv-delta-every.
--bs-mv-delta-y <n>          Vertical delta for --bs-mv-delta-every.
--bs-block-shift-every <n>   Shift serialized block destination addresses every nth block.
--bs-block-shift-x <n>       Horizontal block address shift.
--bs-block-shift-y <n>       Vertical block address shift.
--bs-residual-zero-every <n> Zero serialized residual samples every nth block.
--bs-residual-xor-every <n>  XOR serialized residual bytes every nth block.
--bs-residual-xor-mask <n>   Byte mask for residual XOR. Default: 255
--bs-entropy-slip-every <n>  Byte-slip residual payload windows every nth P frame.
--bs-entropy-slip-by <n>     Signed byte rotation for entropy slip windows. Default: 1
--bs-entropy-resync-bytes <n>
                              Residual payload window length before simulated resync.
--bs-entropy-windows <n>     Number of entropy slip windows per affected frame. Default: 1
--bs-coeff-every <n>         Transform residual payload tiles every nth P frame.
--bs-coeff-block-size <n>    Hadamard residual transform tile size: 4, 8, or 16.
--bs-coeff-shift <n>         Signed coefficient rotation excluding DC.
--bs-coeff-sign-flip-every <n>
                              Flip every nth transformed residual coefficient.
--bs-coeff-zero-high <n>     Zero coefficients where x+y is at least n.
--bs-coeff-quant <n>         Quantize transformed residual coefficients.
--bs-codebook-every <n>      Replace every nth residual tile with a codebook tile.
--bs-codebook-tile-size <n>  Residual codebook tile size: 4, 8, or 16.
--bs-codebook-slots <n>      Maximum residual tiles kept in the decoder codebook.
--bs-codebook-stride <n>     Signed codebook slot offset used for replacement.
--bs-codebook-update-every <n>
                              Store every nth clean residual tile in the codebook.
--bs-codebook-shuffle-every <n>
                              Randomize selected codebook slot reads.
```

Examples:

```powershell
# Strong continuous mosh, no planned recovery.
.\target\release\datamosh-cli.exe filter --drop-keyframe-after 1 --drop-headers-after-first

# Let the decoder recover every tenth later IDR unit.
.\target\release\datamosh-cli.exe filter --drop-keyframe-after 1 --recover-every 10

# Add compression damage to moving areas while still recovering periodically.
.\target\release\datamosh-cli.exe filter --drop-keyframe-after 1 --recover-every 8 --damage-slice-every 5 --damage-amount 2

# Choppy stutter/echo by repeating predicted slices.
.\target\release\datamosh-cli.exe filter --drop-keyframe-after 1 --recover-every 8 --repeat-slice-every 6 --repeat-count 2

# Harsh breakup by dropping predicted slices.
.\target\release\datamosh-cli.exe filter --drop-keyframe-after 1 --recover-every 8 --drop-slice-every 4

# Bitstream truncation, rougher than bit flipping.
.\target\release\datamosh-cli.exe filter --drop-keyframe-after 1 --recover-every 8 --truncate-slice-every 7 --truncate-amount 8

# Byte-order scrambling inside predicted compressed payloads.
.\target\release\datamosh-cli.exe filter --drop-keyframe-after 1 --recover-every 8 --scramble-slice-every 5 --scramble-amount 12

# Payload phase shift. Often less explosive than scramble.
.\target\release\datamosh-cli.exe filter --drop-keyframe-after 1 --recover-every 8 --rotate-slice-every 4 --rotate-amount 16

# Copy previous compressed payload bytes into the current unit.
.\target\release\datamosh-cli.exe filter --drop-keyframe-after 1 --recover-every 8 --splice-slice-every 7 --splice-amount 24

# Insert previous compressed payload bytes, changing the unit length.
.\target\release\datamosh-cli.exe filter --drop-keyframe-after 1 --recover-every 8 --grow-slice-every 9 --grow-amount 8

# XOR with the previous predicted payload.
.\target\release\datamosh-cli.exe filter --drop-keyframe-after 1 --recover-every 8 --xor-slice-every 5 --xor-amount 32

# Inject previous compressed units for a harsher temporal echo.
.\target\release\datamosh-cli.exe filter --drop-keyframe-after 1 --recover-every 8 --echo-slice-every 6 --echo-count 1

# Replace predicted units with the previous compressed unit.
.\target\release\datamosh-cli.exe filter --drop-keyframe-after 1 --recover-every 8 --replace-slice-every 7

# HEVC/MP4-ish stream path, usually best with lighter mutation.
.\target\release\datamosh-cli.exe filter --codec hevc --drop-keyframe-after 1 --recover-every 8 --rotate-slice-every 5 --rotate-amount 12 --splice-slice-every 11 --splice-amount 16

# MPEG-4 ASP aliases.
.\target\release\datamosh-cli.exe filter --codec xvid --drop-keyframe-after 1 --recover-every 8
.\target\release\datamosh-cli.exe filter --codec divx --drop-keyframe-after 1 --recover-every 8

# MPEG-1 Video, rough old-GOP path.
.\target\release\datamosh-cli.exe filter --codec mpeg1 --drop-keyframe-after 1 --recover-every 8 --splice-slice-every 7 --splice-amount 24

# MPEG-2 slice row remap.
.\target\release\datamosh-cli.exe filter --codec mpeg2 --drop-keyframe-after 1 --recover-every 8 --shift-slice-address-every 1 --shift-slice-address-by 3

# MPEG-2 partial I-picture refresh.
.\target\release\datamosh-cli.exe filter --codec mpeg2 --drop-keyframe-after 99 --drop-mpeg-slice-address-every 2 --drop-mpeg-slice-address-phase 0 --drop-mpeg-slice-address-mode key

# Use compressed payloads from a second elementary stream.
.\target\release\datamosh-cli.exe filter --codec h264 --donor-file donor.h264 --drop-keyframe-after 1 --recover-every 8 --donor-splice-slice-every 5 --donor-splice-amount 32

# MPEG frame-type lie. Works on MPEG-4 ASP and MPEG-1/2 paths.
.\target\release\datamosh-cli.exe filter --codec mpeg2 --drop-keyframe-after 99 --rewrite-frame-type-every 3 --rewrite-frame-type-to b
```

## Webcam Recipes

List your camera's exact device name first:

```powershell
ffmpeg -list_devices true -f dshow -i dummy
```

Then substitute that name for `<your camera>` in the recipes below. Change only the `datamosh-cli.exe filter ...` portion to alter the visual character.

```powershell
# Melting motion smear (H.264).
cmd /c "ffmpeg -f dshow -video_size 1280x720 -framerate 30 -pixel_format yuv420p -i video=""<your camera>"" -an -c:v libx264 -preset ultrafast -tune zerolatency -g 24 -keyint_min 24 -bf 0 -x264-params scenecut=0 -f h264 - | .\target\release\datamosh-cli.exe filter --codec h264 --drop-keyframe-after 1 --recover-every 8 | ffplay -fflags nobuffer -flags low_delay -framedrop -f h264 -"

# Sparkly block damage (H.264).
cmd /c "ffmpeg -f dshow -video_size 1280x720 -framerate 30 -pixel_format yuv420p -i video=""<your camera>"" -an -c:v libx264 -preset ultrafast -tune zerolatency -g 24 -keyint_min 24 -bf 0 -x264-params scenecut=0 -f h264 - | .\target\release\datamosh-cli.exe filter --codec h264 --drop-keyframe-after 1 --recover-every 8 --damage-slice-every 5 --damage-amount 2 | ffplay -fflags nobuffer -flags low_delay -framedrop -f h264 -"

# Choppy frame drag (H.264).
cmd /c "ffmpeg -f dshow -video_size 1280x720 -framerate 30 -pixel_format yuv420p -i video=""<your camera>"" -an -c:v libx264 -preset ultrafast -tune zerolatency -g 24 -keyint_min 24 -bf 0 -x264-params scenecut=0 -f h264 - | .\target\release\datamosh-cli.exe filter --codec h264 --drop-keyframe-after 1 --recover-every 8 --repeat-slice-every 6 --repeat-count 2 | ffplay -fflags nobuffer -flags low_delay -framedrop -f h264 -"

# Unstable breakup (H.264).
cmd /c "ffmpeg -f dshow -video_size 1280x720 -framerate 30 -pixel_format yuv420p -i video=""<your camera>"" -an -c:v libx264 -preset ultrafast -tune zerolatency -g 24 -keyint_min 24 -bf 0 -x264-params scenecut=0 -f h264 - | .\target\release\datamosh-cli.exe filter --codec h264 --drop-keyframe-after 1 --recover-every 8 --drop-slice-every 4 --damage-slice-every 7 --damage-amount 2 | ffplay -fflags nobuffer -flags low_delay -framedrop -f h264 -"

# HEVC / MP4-ish smear.
cmd /c "ffmpeg -f dshow -video_size 1280x720 -framerate 30 -pixel_format yuv420p -i video=""<your camera>"" -an -c:v libx265 -preset ultrafast -tune zerolatency -x265-params keyint=24:min-keyint=24:scenecut=0:bframes=0 -f hevc - | .\target\release\datamosh-cli.exe filter --codec hevc --drop-keyframe-after 1 --recover-every 8 --rotate-slice-every 5 --rotate-amount 12 --splice-slice-every 11 --splice-amount 16 | ffplay -fflags nobuffer -flags low_delay -framedrop -f hevc -"

# MPEG-4 Visual classic smear.
cmd /c "ffmpeg -f dshow -video_size 1280x720 -framerate 30 -pixel_format yuv420p -i video=""<your camera>"" -an -c:v mpeg4 -q:v 4 -g 24 -bf 0 -f m4v - | .\target\release\datamosh-cli.exe filter --codec mpeg4 --drop-keyframe-after 1 --recover-every 8 | ffplay -fflags nobuffer -flags low_delay -framedrop -f m4v -"

# MPEG-2 XOR smear.
cmd /c "ffmpeg -f dshow -video_size 1280x720 -framerate 30 -pixel_format yuv420p -i video=""<your camera>"" -an -c:v mpeg2video -q:v 4 -g 24 -bf 0 -f mpeg2video - | .\target\release\datamosh-cli.exe filter --codec mpeg2 --drop-keyframe-after 1 --recover-every 8 --xor-slice-every 5 --xor-amount 32 | ffplay -fflags nobuffer -flags low_delay -framedrop -f mpegvideo -"

# MPEG-1 rough retro smear.
cmd /c "ffmpeg -f dshow -video_size 1280x720 -framerate 30 -pixel_format yuv420p -i video=""<your camera>"" -an -c:v mpeg1video -q:v 4 -g 24 -bf 0 -f mpeg1video - | .\target\release\datamosh-cli.exe filter --codec mpeg1 --drop-keyframe-after 1 --recover-every 8 --splice-slice-every 7 --splice-amount 24 --rotate-slice-every 5 --rotate-amount 12 | ffplay -fflags nobuffer -flags low_delay -framedrop -f mpegvideo -"
```

## Encoder Variants

The Rust filter only cares about the elementary stream format selected by `--codec`; the FFmpeg encoder before the pipe can vary.

```powershell
# H.264 CPU, lowest friction.
-c:v libx264 -preset ultrafast -tune zerolatency -g 24 -keyint_min 24 -bf 0 -x264-params scenecut=0 -f h264 -

# H.264 NVIDIA, if NVENC is available.
-c:v h264_nvenc -preset p1 -tune ull -g 24 -bf 0 -forced-idr 1 -f h264 -

# H.264 AMD, if AMF is available.
-c:v h264_amf -usage ultralowlatency -quality speed -g 24 -bf 0 -f h264 -

# HEVC CPU.
-c:v libx265 -preset ultrafast -tune zerolatency -x265-params keyint=24:min-keyint=24:scenecut=0:bframes=0 -f hevc -

# HEVC NVIDIA, if NVENC is available.
-c:v hevc_nvenc -preset p1 -tune ull -g 24 -bf 0 -f hevc -

# HEVC AMD, if AMF is available.
-c:v hevc_amf -usage ultralowlatency -quality speed -g 24 -bf 0 -f hevc -

# MPEG-4 Visual built into FFmpeg.
-c:v mpeg4 -q:v 4 -g 24 -bf 0 -f m4v -

# MPEG-4 Visual through Xvid.
-c:v libxvid -q:v 4 -g 24 -bf 0 -f m4v -

# MPEG-1 Video.
-c:v mpeg1video -q:v 4 -g 24 -bf 0 -f mpeg1video -

# MPEG-2 Video.
-c:v mpeg2video -q:v 4 -g 24 -bf 0 -f mpeg2video -
```

## TouchDesigner Direction

Do not treat the CLI as the final integration shape. The target shape should be:

- TouchDesigner C++ plugin handles TOP/CHOP/plugin lifecycle
- Rust `MoshEngine` handles custom motion/residual codec state
- The C ABI owns one opaque `MoshEngine` pointer per plugin instance
- TouchDesigner TOP frames call `datamosh_mosh_engine_process_rgba8`
- TouchDesigner parameters or CHOP values call `set_controls`, `set_parameter`, and `reset_glitch`

The first bridge lives in `touchdesigner/DatamoshTOP`. It is a CPU-memory TOP that downloads input as `RGBA8Fixed`, processes the previous downloaded frame through `datamosh_mosh_engine_process_rgba8`, and uploads `RGBA8Fixed` output. This avoids FFmpeg/ffplay in the plugin path and keeps the codec state inside Rust. The compressed elementary-stream filter remains useful for FFmpeg experiments, but plugin hosts should use `MoshEngine` directly.

## Architecture Direction

The project is a Cargo workspace of three crates:

- `Cargo.toml` + `src/`: the `datamosh` core crate (workspace root) — custom codecs, `MoshEngine`, and the C ABI exports; builds `datamosh.dll`. It no longer contains the FFmpeg filter, so the plugin library stays lean.
- `crates/datamosh-streamfilter/`: the FFmpeg elementary-stream filter (`Codec`, `Config`, `MoshFilter`, `DatamoshStream`, `run_stream`) as a standalone crate with no dependency on the core.
- `crates/datamosh-cli/`: the `datamosh-cli` test-harness binary; it depends on both other crates and dispatches `raw-mosh` (core) versus the elementary-stream filter (streamfilter).

- `src/lib.rs`: the core crate root — `MoshEngine` and the C ABI exports.
- `src/mosh_codec.rs`: custom raw motion/residual codec.
- `src/scanline_codec.rs`: SCN0 predictive scanline/signal codec.
- `src/dct_codec.rs`: DCT0 intra transform codec (8x8 DCT, quantization, 4:2:0 chroma).
- `src/dct_bitstream.rs`: DCT0 `DTE0` entropy/bitstream stage (canonical Huffman) and its byte corruption.
- `include/datamosh_ffi.h`: C/C++ ABI declarations and backend constants.
- `examples/cpp_smoke/main.cpp`: tiny dynamic-load C++ host for ABI checks.
- `touchdesigner/DatamoshTOP/DatamoshTOP.cpp`: experimental TouchDesigner TOP bridge.
- `target/release/datamosh.dll`: C ABI dynamic library on Windows.
- `target/release/datamosh.lib`: static/import library output on Windows.
- `target/release/DatamoshTOP.dll`: TouchDesigner C++ TOP plugin output.

## License

MIT — see [LICENSE](LICENSE).
