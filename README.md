# datamosh-engine

Realtime datamoshing in Rust: a low-latency codec core you can damage frame-by-frame, plus an FFmpeg-compatible elementary-stream corruptor. Built to be embedded in a host (e.g. TouchDesigner) through a C ABI.

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

## What's inside

Two independent glitch paths that share almost no code:

- **Custom codecs** (the focus) — tiny video codecs you encode into and then *damage on decode*. Three backends, each a different glitch family, all driven through one `MoshEngine` object and exposed over a C ABI for plugin hosts:
  - **MSH0** (backend `1`) — block motion vectors + residuals; the classic interframe "melt".
  - **SCN0** (backend `2`) — a predictive scanline/signal codec; analog-style sync, line, and chroma tearing.
  - **DCT0** (backend `3`) — an intra 8×8-DCT transform codec (JPEG/DV family); block/coefficient corruption plus a real canonical-Huffman entropy bitstream for "broken JPEG" decoder desync.
- **FFmpeg stream filter** — reads a compressed elementary stream on stdin, corrupts *predicted-frame* payloads, and writes the damaged stream to stdout. Supports H.264, HEVC, MPEG-4 Part 2/ASP (Xvid/DivX), MPEG-1, and MPEG-2.

The `datamosh-cli` binary is only a test harness around the library API. The intended deployment is the Rust core embedded in a host that owns capture, UI, audio, and output.

## Requirements & build

- Rust toolchain
- FFmpeg + FFplay on `PATH` (only for the stream filter and the preview pipes)

```powershell
cargo build --release   # -> target/release/{datamosh-cli.exe, datamosh.dll, datamosh.lib}
```

## The custom codecs

All three backends run through `MoshEngine` (one engine per host node) and take/return raw `rgb24` or `rgba8`. This is the path plugin hosts use; it is not a storage codec, just a low-latency, editable representation that can be damaged safely.

```rust
use datamosh::{MoshEngine, MoshEngineBackend, RawMoshControls};

let mut engine = MoshEngine::with_backend(MoshEngineBackend::DctTransformV1, width, height)?;
engine.set_preset("bleed")?;
engine.set_controls(RawMoshControls { intensity: audio_envelope, ..Default::default() });
engine.process_rgba8(input_rgba8, output_rgba8)?;   // or process_rgb24(...)
if beat { engine.reset_glitch(); }
```

| Backend | id | Preset menu |
|---|----|----|
| **MSH0** motion (`raw_mosh_v1`) | `1` | clean, subtle, classic, melt, grain, pixel, residue, scan, drift, bank, plane, vector, entropy, coeff, codebook, unstable, balanced, destroy |
| **SCN0** scanline (`scanline_signal_v1`) | `2` | clean, timebase-tear, clock-skew, sync-dropout, chroma-sequence, burst-seed-loss, carrier-xor, predictor-ghost, rle-runaway, plane-crosswire, composite-collapse |
| **DCT0** transform (`dct_transform_v1`) | `3` | clean, blocks, dc-smear, bleed, blur, ring, scramble, block-slip, echo, flow, false-color, composite, desync, shred, truncate |

- **MSH0** searches block motion (luma SAD), then decodes through glitch knobs. `reference_mode` is the key switch: `split` keeps the encoder's history clean (classic moshing); `feedback` re-encodes the glitched output so damage compounds.
- **SCN0** is a real signal codec, not an analog-look post effect: each line carries a sync word, luma prediction, quantized/RLE residuals, and a phase-multiplexed chroma carrier. Glitches damage the serialized line clocks, addresses, sync, and predictor state before decode.
- **DCT0** is intra-only (per-frame 8×8 DCT + JPEG quantization + 4:2:0). Glitches corrupt the transform domain (quantization, DC/AC coefficients, the differential DC predictor, zig-zag order, block transpose/remap, chroma swap). The optional `DTE0` entropy stage (`desync`/`shred`/`truncate`) losslessly codes the coefficients (DC DPCM + AC run-length + canonical Huffman) and then damages that byte stream — the cascading "broken JPEG" slide. It is block-parallel (Rayon).

Benchmark any backend (reports fps + a deterministic checksum; use the checksum to catch output drift while optimizing):

```powershell
cargo run --release --example raw_mosh_bench       -- 640 360 120 codebook
cargo run --release --example scanline_codec_bench -- 640 360 120 composite-collapse
cargo run --release --example dct_codec_bench      -- 640 360 120 composite
```

Frames of at least 200,000 pixels use a shared Rayon pool (defaults to half the logical CPUs, capped at 16); set `DATAMOSH_THREADS` before the first frame to override.

### Preview the motion codec live (`raw-mosh`)

`raw-mosh` pipes raw `rgb24` in and out (MSH0 only). General shape — change only the middle command:

```powershell
cmd /c "ffmpeg -f dshow -video_size 1280x720 -framerate 30 -pixel_format yuv420p -i video=""<your camera>"" -an -vf scale=640:360,format=rgb24 -f rawvideo - | .\target\release\datamosh-cli.exe raw-mosh --width 640 --height 360 --preset melt | ffplay -fflags nobuffer -flags low_delay -framedrop -f rawvideo -pixel_format rgb24 -video_size 640x360 -framerate 30 -"
```

Swap `--preset` for any motion preset in the table above (`melt`/`classic` are the datamosh-like smears). Heavier codec-state presets (`residue`, `drift`, `bank`, `plane`, `vector`, `unstable`) are best processed at 480×270 with `--upscale 2` added. `raw-mosh` also accepts live UDP control on `--control-port` so you can retune without restarting the pipe.

<details>
<summary>Preset launcher GUI &amp; harder variants</summary>

`.\scripts\raw-mosh-demo.cmd` is a Windows preset launcher that drives an OBS Virtual Camera (or an FFmpeg test pattern) into `raw-mosh` and opens ffplay as the preview. With `Realtime control` on, the sliders send `controls intensity motion residual temporal bitstream` and `set <id> <value>` over UDP, and `Reset glitch` sends `reset-glitch`. Inspect presets without the GUI via `-ListPresets`, `-PrintDefaultCommand`, `-SmokeGui`.

The `entropy`, `coeff`, and `codebook` presets are mild by default; use the `-hard` variants (`--preset coeff-hard`, etc.) or the `--bs-*` flags from `raw-mosh --help` for stronger breakup. `--reference-mode feedback` makes `melt` more likely to fully collapse.
</details>

## Embedding (C ABI)

The C ABI mirrors `MoshEngine` behind one opaque pointer per host node:

```c
void* e = datamosh_mosh_engine_new_with_backend(1 /* raw_mosh_v1 */, width, height);
datamosh_mosh_engine_set_preset(e, "codebook");
datamosh_mosh_engine_set_controls(e, intensity, motion, residual, temporal, bitstream);
datamosh_mosh_engine_set_parameter(e, "codebook_replace_every", 4.0f);
datamosh_mosh_engine_process_rgba8(e, input, input_len, output, output_len);
datamosh_mosh_engine_reset_glitch(e);
datamosh_mosh_engine_free(e);
```

Backend constants: `1` = `RAW_MOSH_V1`, `2` = `SCANLINE_SIGNAL_V1`, `3` = `DCT_TRANSFORM_V1`. Add future codecs as new backend IDs, not new entry points. Full declarations are in [`include/datamosh_ffi.h`](include/datamosh_ffi.h); a dynamic-load smoke host builds with `.\scripts\build-cpp-smoke.cmd -Run`.

For the FFmpeg stream filter as a library (e.g. wrapping it elsewhere), use `DatamoshStream` from the `datamosh-streamfilter` crate — feed it bytes with `process_chunk(...)` and, optionally, a second elementary stream as a donor with `load_donor_stream(...)`.

## FFmpeg stream filter (CLI)

Corrupt a compressed elementary stream in realtime. On Windows, wrap binary pipelines in `cmd /c` — plain PowerShell pipes corrupt the byte stream. The filter only touches the elementary stream selected by `--codec`; the FFmpeg encoder before the pipe can vary.

```powershell
cmd /c "ffmpeg -re -stream_loop -1 -i input.mp4 -an <ENCODER FLAGS> - | .\target\release\datamosh-cli.exe filter --codec <CODEC> --drop-keyframe-after 1 --recover-every 8 | ffplay -fflags nobuffer -flags low_delay -framedrop -f <FMT> -"
```

| `--codec` | `<ENCODER FLAGS>` | ffplay `-f <FMT>` |
|---|---|---|
| `h264` | `-c:v libx264 -preset ultrafast -tune zerolatency -g 24 -keyint_min 24 -bf 0 -x264-params scenecut=0 -f h264` | `h264` |
| `hevc` | `-c:v libx265 -preset ultrafast -tune zerolatency -x265-params keyint=24:min-keyint=24:scenecut=0:bframes=0 -f hevc` | `hevc` |
| `mpeg4` (also `xvid`/`divx`) | `-c:v mpeg4 -q:v 4 -g 24 -bf 0 -f m4v` | `m4v` |
| `mpeg2` | `-c:v mpeg2video -q:v 4 -g 24 -bf 0 -f mpeg2video` | `mpegvideo` |
| `mpeg1` | `-c:v mpeg1video -q:v 4 -g 24 -bf 0 -f mpeg1video` | `mpegvideo` |

Mutations combine freely (`datamosh-cli filter --help` lists them all):

- `--drop-keyframe-after 1 --recover-every 8` — classic keyframe-drop mosh with periodic recovery
- `--damage-slice-every 5 --damage-amount 2` — sparkly block damage
- `--repeat-slice-every 6 --repeat-count 2` — choppy frame drag; `--drop-slice-every 4` — harsher breakup
- `--truncate-slice-every 7` / `--scramble-slice-every 5` / `--rotate-slice-every 4` / `--xor-slice-every 5` — byte-level payload damage
- `--donor-file donor.h264 --donor-splice-slice-every 5 --donor-splice-amount 32` — splice payloads from a second stream
- MPEG-1/2 structural surgery: `--shift-slice-address-by 3`, `--drop-mpeg-slice-address-every 2`, `--rewrite-frame-type-to b`

To capture from a webcam instead of a file, list devices with `ffmpeg -list_devices true -f dshow -i dummy`, then use `-f dshow -i video="<your camera>"` as the FFmpeg input.

<details>
<summary>Full <code>filter</code> option reference &amp; encoder variants</summary>

```text
datamosh-cli filter [options]            # corrupt a compressed elementary stream
datamosh-cli raw-mosh --width <px> --height <px> [raw options]
datamosh-cli [options]                   # defaults to the filter

  --codec <h264|hevc|mpeg4|mpeg1|mpeg2>   Input elementary stream codec. Default: h264
  --drop-keyframe-after <n>   Pass the first n keyframes, then drop keyframes. Default: 1
  --recover-every <n>         After initial drops, pass every nth later keyframe. 0 disables recovery.
  --drop-slice-every <n>      Drop every nth predicted slice/VOP.
  --damage-slice-every <n>    Corrupt every nth predicted payload.   --damage-amount <n>   Default: 4
  --truncate-slice-every <n>  Truncate every nth predicted slice/VOP. --truncate-amount <n> Default: 16
  --scramble-slice-every <n>  Byte-scramble every nth payload.        --scramble-amount <n> Default: 16
  --rotate-slice-every <n>    Rotate bytes inside every nth payload.  --rotate-amount <n>   Default: 8
  --splice-slice-every <n>    Copy previous payload bytes into every nth payload. --splice-amount <n>
  --grow-slice-every <n>      Insert previous payload bytes into every nth unit.  --grow-amount <n>
  --xor-slice-every <n>       XOR every nth predicted unit with the previous unit. --xor-amount <n>
  --echo-slice-every <n>      Inject previous predicted unit before every nth unit. --echo-count <n>
  --replace-slice-every <n>   Replace every nth predicted unit with the previous unit.
  --repeat-slice-every <n>    Repeat every nth predicted slice/VOP.   --repeat-count <n>    Default: 1
  --donor-file <path>         Load a second elementary stream; use its predicted payloads as donors.
  --donor-splice-slice-every / --donor-grow-slice-every / --donor-xor-slice-every / --donor-replace-slice-every <n>
                              Donor-sourced splice/grow/XOR/replace (with matching --donor-*-amount).
  --rewrite-frame-type-every <n> --rewrite-frame-type-to <i|p|b|s|d>   MPEG-4/MPEG-1/2 frame-type lie.
  --shift-slice-address-every <n> --shift-slice-address-by <n>         MPEG-1/2 slice-address remap.
  --drop-mpeg-slice-address-every <n> [--drop-mpeg-slice-address-phase/-mode]  MPEG-1/2 partial slice drop.
  --drop-headers-after-first  Drop repeated SPS/PPS headers after the first pair.
  --quiet                     Do not print realtime stats to stderr.
```

`raw-mosh` adds `--width`/`--height` (required), `--output-width`/`--output-height`/`--upscale`/`--scale-mode` (display scaling), `--preset`, `--reference-mode <split|feedback>`, `--control-port`, motion-search tuning (`--block-size`/`--search-radius`/`--search-step`/`--history`), and a large set of per-stage glitch knobs (`--mv-*`, `--residual-*`, `--reference-*`, `--temporal-slice-*`, `--*-bank-*`, `--bitstream` + `--bs-*`). Run `datamosh-cli raw-mosh --help` for the complete list.

FFmpeg encoder before the pipe can use hardware encoders, e.g. `-c:v h264_nvenc -preset p1 -tune ull -g 24 -bf 0 -forced-idr 1 -f h264` (NVIDIA) or `-c:v h264_amf -usage ultralowlatency -quality speed -g 24 -bf 0 -f h264` (AMD); HEVC has `hevc_nvenc`/`hevc_amf` equivalents.
</details>

## TouchDesigner

The intended deployment shape: a TD C++ TOP owns the operator lifecycle, parameters, and IO; the Rust `MoshEngine` owns codec state behind the C ABI.

- CPU TOPs `DatamoshTOP` / `ScanlineSignalTOP` / `DatamoshDctTOP` (backends 1/2/3) load `datamosh.dll` beside them. They download input as `RGBA8Fixed`, process the previous frame (≈1-frame latency, avoids a same-frame GPU readback stall), and upload `RGBA8Fixed`.
- Two **CUDA TOPs** (`DatamoshCudaTOP`, `DatamoshDctCudaTOP`) are separate GPU-native re-implementations of the motion and DCT paths that keep all state on-GPU and do **not** link `datamosh.dll`. Parity with the CPU codecs is maintained by hand; the DCT path has an automated parity guard (`.\scripts\build-dct-parity-check.cmd`).
- Build the plugins with `.\scripts\build-td-*.cmd` (MSVC required — a MinGW build crashes the host). Stage them for the demo with `.\scripts\build-td-plugins.cmd`.

A ready-to-open demo project and the full per-operator pattern/parameter reference live in **[`touchdesigner/README.md`](touchdesigner/README.md)** and **[`touchdesigner/demo/`](touchdesigner/demo/)**.

## Architecture

A Cargo workspace of three crates:

- **`datamosh`** (workspace root, `src/`) — the custom codecs, `MoshEngine`, and the C ABI exports; builds `datamosh.dll`. It does not depend on the stream filter, so the plugin library stays lean.
- **`crates/datamosh-streamfilter/`** — the FFmpeg elementary-stream filter (`Codec`, `Config`, `MoshFilter`, `DatamoshStream`), standalone.
- **`crates/datamosh-cli/`** — the `datamosh-cli` test-harness binary; depends on both and dispatches `raw-mosh` (core) vs the filter (streamfilter).

Key files: `src/mosh_codec.rs` (MSH0), `src/scanline_codec.rs` (SCN0), `src/dct_codec.rs` + `src/dct_bitstream.rs` (DCT0 / DTE0), `include/datamosh_ffi.h` (C ABI), `touchdesigner/` (the TD bridges).

## License

MIT — see [LICENSE](LICENSE).
