#!/usr/bin/env bash
# Regenerate assets/datamosh.gif — the README hero.
#
# Runs the same Mandelbrot zoom through all three engine backends (motion / scanline /
# DCT) via the `raw_engine` example, labels each clip, tiles them into a 1x3 comparison
# strip, and palette-optimizes the result into a looping GIF. Pure FFmpeg + the engine,
# no plugin host. Requires ffmpeg on PATH and a Rust toolchain.
#
# Font: defaults to Consolas on Windows; override for other systems, e.g.
#   FONT=/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf bash scripts/make-demo-gif.sh
set -euo pipefail
cd "$(dirname "$0")/.."

W=854; H=480
SRC="mandelbrot=size=${W}x${H}:rate=30"
OUT=assets/datamosh.gif
mkdir -p target assets

cargo build --release --example raw_engine
EX=target/release/examples/raw_engine.exe
[ -x "$EX" ] || EX=target/release/examples/raw_engine   # non-Windows binary name

# backend ids: 1 = raw_mosh_v1 (MSH0 motion), 2 = scanline_signal_v1 (SCN0), 3 = dct_transform_v1 (DCT0)
render() { # <backend> <preset> <name>
  ffmpeg -y -v error -f lavfi -i "$SRC" -t 8 -vf format=rgb24 -f rawvideo -pix_fmt rgb24 - \
    | "$EX" "$W" "$H" "$1" "$2" \
    | ffmpeg -y -v error -f rawvideo -pixel_format rgb24 -video_size "${W}x${H}" -framerate 30 -i - \
        -frames:v 240 -pix_fmt yuv420p "target/$3.mp4"
}
render 1 classic         mot
render 2 predictor-ghost scn
render 3 bleed           dct

FONT="${FONT:-/c/Windows/Fonts/consolab.ttf}"
cp "$FONT" target/font.ttf
FF="fontfile=target/font.ttf"
lbl() { echo "drawtext=$FF:text='$1':x=12:y=10:fontsize=22:fontcolor=white:box=1:boxcolor=black@0.45:boxborderw=8"; }

ffmpeg -y -v error -i target/mot.mp4 -i target/scn.mp4 -i target/dct.mp4 -filter_complex \
"[0:v]scale=380:-2,$(lbl 'MOTION | MSH0')[a];\
[1:v]scale=380:-2,$(lbl 'SCANLINE | SCN0')[b];\
[2:v]scale=380:-2,$(lbl 'DCT | DCT0')[c];\
[a][b][c]hstack=inputs=3[v]" -map "[v]" -frames:v 240 -pix_fmt yuv420p target/montage.mp4

# MP4 -> palette -> optimized looping GIF
ffmpeg -y -v error -t 6 -i target/montage.mp4 \
  -vf "fps=10,scale=900:-1:flags=lanczos,palettegen=stats_mode=diff" target/pal.png
ffmpeg -y -v error -t 6 -i target/montage.mp4 -i target/pal.png \
  -lavfi "fps=10,scale=900:-1:flags=lanczos[x];[x][1:v]paletteuse=dither=none" "$OUT"

echo "wrote $OUT ($(( $(stat -c %s "$OUT") / 1024 )) KB)"
