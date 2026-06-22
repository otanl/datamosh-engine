# Datamosh — TouchDesigner demo

A quick-start project that runs the datamosh codec TOPs on a video input in realtime.

## Setup

1. Build the core library and the plugin DLLs and stage them into this folder's `Plugins/`:

   ```powershell
   .\scripts\build-td-plugins.cmd          # add -SkipCuda if you have no NVIDIA GPU / CUDA toolkit
   ```

   This builds `datamosh.dll` plus the TouchDesigner TOP plugins and copies the ones TD needs
   into `touchdesigner/demo/Plugins/`. (You can also copy them by hand from `target/release/`:
   `datamosh.dll`, `DatamoshTOP.dll`, `DatamoshScanlineTOP.dll`, `DatamoshDctTOP.dll`, and — with an
   NVIDIA GPU — `DatamoshCudaTOP.dll`, `DatamoshDctCudaTOP.dll`.)

2. Open `datamosh-demo.toe` in TouchDesigner. TouchDesigner loads Custom Operators from the
   `Plugins/` folder next to the `.toe`, so the datamosh TOPs resolve automatically — no global
   install needed.

## What the project shows

The network is a **side-by-side comparison** of the three CPU codec backends running live on
one moving source:

```
source (Movie File In, media/sample.mp4)
  → xform (slow pan + rotate — injects motion so the motion codec has vectors to chew on)
      → motion1   (DatamoshTOP        · MSH0 · Pattern "melt")
      → scanline1 (DatamoshScanlineTOP  · SCN0 · Pattern "predictor-ghost")
      → dct1      (DatamoshDctTOP      · DCT0 · Pattern "bleed")
  → grid (Layout TOP, 2×2, with text labels) → out1
```

- `source` plays `media/sample.mp4`. Swap in a `Video Device In TOP` for a live webcam (list
  devices with `ffmpeg -list_devices true -f dshow -i dummy`).
- `xform` adds a gentle continuous pan/rotate. Datamoshing lives on **motion** — a frozen test
  pattern barely moshes, so the transform gives the motion codec real motion vectors to smear.
- Each datamosh Custom TOP exposes a `Pattern` menu plus the shared macro sliders (`Intensity`,
  `Motion`, `Residual`, `Temporal`, `Bitstream`); keep `Use Overrides` off while auditioning
  patterns. `out1` shows the labelled 2×2 grid (Original / Motion / Scanline / DCT).
- **To use a single codec** instead of the comparison, just wire that codec (e.g. `motion1`)
  straight into `out1`, or drop in `Datamosh.tox`.
- **Audio-reactive** is wired into the parameters but left off: the motion TOP's `Audio` page
  already maps a control channel to `Intensity`/`Temporal` and a reset on transients
  (`Audio Gain`, `Reset Threshold`). Attach a CHOP (e.g. `Audio Device In → Analyze RMS → Lag`)
  to the `Control CHOP` field and switch `Audio Enable` on to make sound drive the glitch.

The three CPU TOPs (motion / scanline / DCT) and the two CUDA TOPs each expose their own
`Pattern` menu — see `../README.md` for the per-operator pattern/parameter reference.

## Layout

```
touchdesigner/demo/
  datamosh-demo.toe   the project (open this)
  Datamosh.tox        optional reusable component
  media/sample.mp4    small test clip (committed)
  Plugins/            built TOP DLLs land here (git-ignored — rebuild with the script above)
  README.md           this file
```

## Notes

- The CPU TOPs need `datamosh.dll` beside them in `Plugins/`; the CUDA TOPs do not (the build
  script copies `datamosh.dll` in regardless, which is harmless).
- Media is referenced by **relative path** — keep clips under `media/` so the project stays
  portable.
- `Plugins/*.dll` are build artifacts (platform- and TD-version-specific) and are intentionally
  not committed; regenerate them with `build-td-plugins.cmd`.
