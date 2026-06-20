# Datamosh — TouchDesigner demo

A quick-start project that runs the datamosh codec TOPs on a video input in realtime.

## Setup

1. Build the core library and the plugin DLLs and stage them into this folder's `Plugins/`:

   ```powershell
   .\scripts\build-td-plugins.cmd          # add -SkipCuda if you have no NVIDIA GPU / CUDA toolkit
   ```

   This builds `datamosh.dll` plus the TouchDesigner TOP plugins and copies the ones TD needs
   into `touchdesigner/demo/Plugins/`. (You can also copy them by hand from `target/release/`:
   `datamosh.dll`, `DatamoshTOP.dll`, `ScanlineSignalTOP.dll`, `DatamoshDctTOP.dll`, and — with an
   NVIDIA GPU — `DatamoshCudaTOP.dll`, `DatamoshDctCudaTOP.dll`.)

2. Open `datamosh-demo.toe` in TouchDesigner. TouchDesigner loads Custom Operators from the
   `Plugins/` folder next to the `.toe`, so the datamosh TOPs resolve automatically — no global
   install needed.

## What the project shows

- A `Movie File In TOP` plays `media/sample.mp4`. Swap in a `Video Device In TOP` for a live
  webcam (list devices with `ffmpeg -list_devices true -f dshow -i dummy`).
- A datamosh Custom TOP applies the glitch. Its `Pattern` menu and the macro sliders
  (`Intensity`, `Structure`, `Persist`/`Residual`, `DC`/`Temporal`, `Quant`/`Bitstream`) are
  the main controls; keep `Use Overrides` off while auditioning patterns.
- Optional audio-reactive chain: `Audio Device In CHOP → Audio Spectrum/Analyze → Lag/Math →`
  a CHOP referenced by the TOP's `Audio` page, so sound drives the macros and reset.

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
