# Datamosh Wavelet TOP

CPU TouchDesigner TOP for the `wavelet_pyramid_v1` (`WVT0`) backend.

Build:

```powershell
.\scripts\build-td-wavelet-top.cmd -ReleaseRust
```

Place `DatamoshWaveletTOP.dll` and `datamosh.dll` in the same plugin directory.
The operator type is `Datamoshwavelet`.

`WVT0` converts RGB through reversible YCoCg-R, applies a multilevel integer Haar
lifting transform, quantizes each scale independently, and decodes independently
addressable subband packets. Its patterns damage scale/orientation routing,
bitplanes, packet loss concealment, temporal subband history, and inverse lifting
state rather than applying an RGB image filter.
