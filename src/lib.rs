use std::ffi::CStr;
use std::io;
use std::os::raw::c_char;
use std::panic::{self, AssertUnwindSafe};
use std::ptr;
use std::slice;

pub mod dct_bitstream;

pub mod dct_codec;

pub mod mosh_codec;

pub mod scanline_codec;
pub use dct_bitstream::{
    DctBitstreamMutationStats, DctBitstreamParams, apply_dct_bitstream_controls,
    decode_dct_bitstream, encode_dct_bitstream, load_dct_bitstream_preset, mutate_dct_bitstream,
    set_dct_bitstream_parameter,
};
pub use dct_codec::{
    DctCodec, DctCodecConfig, DctCodecStats, DctGlitchParams, DctMutationStats,
    apply_dct_transform_controls, load_dct_transform_preset, set_dct_transform_parameter,
};
pub use mosh_codec::{
    ActivityMode, MoshBitstreamMutationStats, MoshBitstreamParams, MoshCodec, MoshCodecConfig,
    MoshCodecStats, MoshGlitchParams, MoshPacket, MoshPacketKind, MoshReferenceMode, MotionBlock,
    decode_packet_bitstream, encode_packet_bitstream, mutate_packet_bitstream,
};
pub use scanline_codec::{
    ScanlineCodec, ScanlineCodecConfig, ScanlineCodecStats, ScanlineGlitchParams,
    ScanlineMutationStats, mutate_scanline_bitstream,
};

pub const RAW_RGB_CHANNELS: usize = 3;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RawMoshControls {
    pub intensity: f32,
    pub motion: f32,
    pub residual: f32,
    pub temporal: f32,
    pub bitstream: f32,
}

pub const RAW_MOSH_CONTROL_MAX: f32 = 2.0;

const RAW_MOSH_COMBINED_CONTROL_MAX: f32 = RAW_MOSH_CONTROL_MAX * RAW_MOSH_CONTROL_MAX;

impl Default for RawMoshControls {
    fn default() -> Self {
        Self {
            intensity: 1.0,
            motion: 1.0,
            residual: 1.0,
            temporal: 1.0,
            bitstream: 1.0,
        }
    }
}

pub fn load_raw_mosh_preset(
    name: &str,
    config: &mut MoshCodecConfig,
    params: &mut MoshGlitchParams,
    bitstream: &mut MoshBitstreamParams,
) -> Result<(), String> {
    apply_raw_mosh_preset(name, config, params, bitstream)
}

pub fn apply_raw_mosh_controls(
    params: &mut MoshGlitchParams,
    bitstream: &mut MoshBitstreamParams,
    controls: RawMoshControls,
) {
    let motion = control_amount(controls.intensity, controls.motion);
    let residual = control_amount(controls.intensity, controls.residual);
    let temporal = control_amount(controls.intensity, controls.temporal);
    let bitstream_amount = control_amount(controls.intensity, controls.bitstream);

    params.mv_scale_x = 1.0 + (params.mv_scale_x - 1.0) * motion;
    params.mv_scale_y = 1.0 + (params.mv_scale_y - 1.0) * motion;
    params.mv_jitter = scale_i16(params.mv_jitter, motion);
    params.block_remap_every = scale_event_interval(params.block_remap_every, motion);
    params.block_remap_stride = scale_i32(params.block_remap_stride, motion);
    params.channel_shift = scale_i16(params.channel_shift, motion);
    params.motion_diffusion *= motion;
    params.mv_field_interpolation *= motion;
    params.sample_address_desync *= motion;
    params.mv_predictor_desync_every =
        scale_event_interval(params.mv_predictor_desync_every, motion);
    params.mv_predictor_desync_x = scale_i16(params.mv_predictor_desync_x, motion);
    params.mv_predictor_desync_y = scale_i16(params.mv_predictor_desync_y, motion);
    params.mv_bank_stride = scale_i32(params.mv_bank_stride, motion);
    params.mv_bank_shuffle_every = scale_event_interval(params.mv_bank_shuffle_every, motion);

    params.residual_keep = 1.0 + (params.residual_keep - 1.0) * residual;
    params.residual_invert_every = scale_event_interval(params.residual_invert_every, residual);
    params.residual_address_shift_x = scale_i16(params.residual_address_shift_x, residual);
    params.residual_address_shift_y = scale_i16(params.residual_address_shift_y, residual);
    params.residual_address_jitter = scale_i16(params.residual_address_jitter, residual);
    params.residual_channel_shift = scale_i16(params.residual_channel_shift, residual);
    params.residual_bank_stride = scale_i32(params.residual_bank_stride, residual);
    params.residual_bank_shuffle_every =
        scale_event_interval(params.residual_bank_shuffle_every, residual);

    params.reference_lag = 1 + scale_usize(params.reference_lag.saturating_sub(1), temporal);
    params.reference_bleed *= temporal;
    params.reference_latch_frames =
        1 + scale_u64(params.reference_latch_frames.saturating_sub(1), temporal);
    params.reference_slot_count =
        1 + scale_usize(params.reference_slot_count.saturating_sub(1), temporal);
    params.reference_slot_shuffle_every =
        scale_event_interval(params.reference_slot_shuffle_every, temporal);
    params.reference_scanline_lag_span = scale_usize(params.reference_scanline_lag_span, temporal);
    params.temporal_slice_lag_span = scale_usize(params.temporal_slice_lag_span, temporal);
    params.temporal_slice_drift = scale_i16(params.temporal_slice_drift, temporal);
    params.reference_channel_lag_span = scale_usize(params.reference_channel_lag_span, temporal);
    params.reference_channel_lag_stride = scale_i16(params.reference_channel_lag_stride, temporal);

    if bitstream_amount <= 0.0 {
        *bitstream = MoshBitstreamParams::default();
        return;
    }

    bitstream.mv_sign_flip_every =
        scale_event_interval(bitstream.mv_sign_flip_every, bitstream_amount);
    bitstream.mv_delta_every = scale_event_interval(bitstream.mv_delta_every, bitstream_amount);
    bitstream.mv_delta_x = scale_i16(bitstream.mv_delta_x, bitstream_amount);
    bitstream.mv_delta_y = scale_i16(bitstream.mv_delta_y, bitstream_amount);
    bitstream.block_address_shift_every =
        scale_event_interval(bitstream.block_address_shift_every, bitstream_amount);
    bitstream.block_address_shift_x = scale_i16(bitstream.block_address_shift_x, bitstream_amount);
    bitstream.block_address_shift_y = scale_i16(bitstream.block_address_shift_y, bitstream_amount);
    bitstream.residual_zero_every =
        scale_event_interval(bitstream.residual_zero_every, bitstream_amount);
    bitstream.residual_xor_every =
        scale_event_interval(bitstream.residual_xor_every, bitstream_amount);
    bitstream.entropy_slip_every =
        scale_event_interval(bitstream.entropy_slip_every, bitstream_amount);
    bitstream.entropy_slip_bytes = scale_i16(bitstream.entropy_slip_bytes, bitstream_amount);
    bitstream.entropy_slip_windows = scale_usize(bitstream.entropy_slip_windows, bitstream_amount);
    bitstream.coeff_glitch_every =
        scale_event_interval(bitstream.coeff_glitch_every, bitstream_amount);
    bitstream.coeff_shift = scale_i16(bitstream.coeff_shift, bitstream_amount);
    bitstream.coeff_sign_flip_every =
        scale_event_interval(bitstream.coeff_sign_flip_every, bitstream_amount);
    bitstream.coeff_quant =
        scale_i16(bitstream.coeff_quant.saturating_sub(1), bitstream_amount).saturating_add(1);
    bitstream.codebook_replace_every =
        scale_event_interval(bitstream.codebook_replace_every, bitstream_amount);
    bitstream.codebook_stride = scale_i32(bitstream.codebook_stride, bitstream_amount);
    bitstream.codebook_shuffle_every =
        scale_event_interval(bitstream.codebook_shuffle_every, bitstream_amount);
    bitstream.enabled = bitstream.has_mutations();
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum RawMoshPresetGroup {
    Motion,
    Reference,
    Residual,
    Bitstream,
    Hybrid,
    Other,
}

impl RawMoshPresetGroup {
    pub fn name(self) -> &'static str {
        match self {
            Self::Motion => "motion",
            Self::Reference => "reference",
            Self::Residual => "residual",
            Self::Bitstream => "bitstream",
            Self::Hybrid => "hybrid",
            Self::Other => "other",
        }
    }
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum RawMoshParameterKind {
    Float,
    Integer,
    Bool,
}

#[derive(Debug, Clone, Copy)]
pub struct RawMoshPresetInfo {
    pub name: &'static str,
    pub group: RawMoshPresetGroup,
    pub title: &'static str,
    pub description: &'static str,
}

#[derive(Debug, Clone, Copy)]
pub struct RawMoshParameterInfo {
    pub id: &'static str,
    pub group: RawMoshPresetGroup,
    pub kind: RawMoshParameterKind,
    pub label: &'static str,
    pub description: &'static str,
    pub min: f32,
    pub max: f32,
    pub default: f32,
}

impl RawMoshParameterInfo {
    pub fn is_realtime(self) -> bool {
        !self.requires_codec_rebuild()
    }

    pub fn requires_codec_rebuild(self) -> bool {
        matches!(self.id, "block_size" | "search_radius" | "search_step")
    }
}

pub fn raw_mosh_preset_infos() -> &'static [RawMoshPresetInfo] {
    RAW_MOSH_PRESET_INFOS
}

pub fn raw_mosh_parameter_infos() -> &'static [RawMoshParameterInfo] {
    RAW_MOSH_PARAMETER_INFOS
}

pub fn raw_mosh_parameter_infos_for_group(group: RawMoshPresetGroup) -> Vec<RawMoshParameterInfo> {
    RAW_MOSH_PARAMETER_INFOS
        .iter()
        .copied()
        .filter(|parameter| parameter.group == group)
        .collect()
}

pub fn raw_mosh_parameter_infos_for_preset(
    preset: &str,
) -> Result<Vec<RawMoshParameterInfo>, String> {
    let group = raw_mosh_preset_group(preset)?;
    Ok(RAW_MOSH_PARAMETER_INFOS
        .iter()
        .copied()
        .filter(|parameter| {
            if group == RawMoshPresetGroup::Hybrid {
                true
            } else {
                parameter.group == group || parameter.group == RawMoshPresetGroup::Hybrid
            }
        })
        .collect())
}

pub fn raw_mosh_preset_group(preset: &str) -> Result<RawMoshPresetGroup, String> {
    RAW_MOSH_PRESET_INFOS
        .iter()
        .find(|info| info.name == preset)
        .map(|info| info.group)
        .ok_or_else(|| format!("unknown raw-mosh preset `{preset}`"))
}

pub fn set_raw_mosh_parameter(
    config: &mut MoshCodecConfig,
    params: &mut MoshGlitchParams,
    bitstream: &mut MoshBitstreamParams,
    id: &str,
    value: f32,
) -> Result<(), String> {
    let finite = if value.is_finite() { value } else { 0.0 };
    match id {
        "mv_scale" => {
            let value = finite.clamp(0.0, 2.0);
            params.mv_scale_x = value;
            params.mv_scale_y = value;
        }
        "mv_jitter" => params.mv_jitter = finite.round().clamp(0.0, 16.0) as i16,
        "mv_field_interpolation" => params.mv_field_interpolation = finite.clamp(0.0, 1.0),
        "sample_address_desync" => params.sample_address_desync = finite.clamp(0.0, 4.0),
        "mv_bank_stride" => params.mv_bank_stride = finite.round().clamp(-64.0, 64.0) as i32,
        "reference_lag" => params.reference_lag = finite.round().clamp(1.0, 32.0) as usize,
        "reference_bleed" => params.reference_bleed = finite.clamp(0.0, 1.0),
        "reference_latch_frames" => {
            params.reference_latch_frames = finite.round().clamp(1.0, 64.0) as u64;
        }
        "temporal_slice_drift" => {
            params.temporal_slice_drift = finite.round().clamp(-16.0, 16.0) as i16;
        }
        "reference_channel_lag_span" => {
            params.reference_channel_lag_span = finite.round().clamp(0.0, 32.0) as usize;
        }
        "residual_keep" => params.residual_keep = finite.clamp(-2.0, 2.0),
        "residual_address_jitter" => {
            params.residual_address_jitter = finite.round().clamp(0.0, 32.0) as i16;
        }
        "residual_channel_shift" => {
            params.residual_channel_shift = finite.round().clamp(-4.0, 4.0) as i16;
        }
        "residual_bank_stride" => {
            params.residual_bank_stride = finite.round().clamp(-64.0, 64.0) as i32;
        }
        "entropy_slip_every" => {
            bitstream.entropy_slip_every = finite.round().clamp(0.0, 64.0) as u64;
        }
        "entropy_slip_windows" => {
            bitstream.entropy_slip_windows = finite.round().clamp(0.0, 64.0) as usize;
        }
        "entropy_resync_bytes" => {
            bitstream.entropy_resync_bytes = finite.round().clamp(0.0, 65_536.0) as usize;
        }
        "coeff_shift" => bitstream.coeff_shift = finite.round().clamp(-32.0, 32.0) as i16,
        "coeff_quant" => bitstream.coeff_quant = finite.round().clamp(1.0, 64.0) as i16,
        "codebook_replace_every" => {
            bitstream.codebook_replace_every = finite.round().clamp(0.0, 64.0) as u64;
        }
        "codebook_stride" => {
            bitstream.codebook_stride = finite.round().clamp(-128.0, 128.0) as i32;
        }
        "codebook_shuffle_every" => {
            bitstream.codebook_shuffle_every = finite.round().clamp(0.0, 64.0) as u64;
        }
        "block_size" => config.block_size = finite.round().clamp(4.0, 64.0) as usize,
        "search_radius" => config.search_radius = finite.round().clamp(0.0, 64.0) as i16,
        "search_step" => config.search_step = finite.round().clamp(1.0, 16.0) as i16,
        "bitstream_enabled" => bitstream.enabled = finite >= 0.5,
        _ => return Err(format!("unknown raw-mosh parameter `{id}`")),
    }

    if id != "bitstream_enabled" {
        bitstream.enabled = bitstream.enabled || bitstream.has_mutations();
    }
    Ok(())
}

pub fn raw_mosh_parameter_requires_rebuild(id: &str) -> bool {
    RAW_MOSH_PARAMETER_INFOS
        .iter()
        .find(|parameter| parameter.id == id)
        .map(|parameter| parameter.requires_codec_rebuild())
        .unwrap_or(false)
}

#[repr(u32)]
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum MoshEngineBackend {
    RawMoshV1 = 1,
    ScanlineSignalV1 = 2,
    DctTransformV1 = 3,
}

impl MoshEngineBackend {
    pub fn parse_id(id: u32) -> Option<Self> {
        match id {
            1 => Some(Self::RawMoshV1),
            2 => Some(Self::ScanlineSignalV1),
            3 => Some(Self::DctTransformV1),
            _ => None,
        }
    }

    pub fn id(self) -> u32 {
        self as u32
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::RawMoshV1 => "raw_mosh_v1",
            Self::ScanlineSignalV1 => "scanline_signal_v1",
            Self::DctTransformV1 => "dct_transform_v1",
        }
    }
}

pub struct MoshEngine {
    backend: MoshEngineBackend,
    config: MoshCodecConfig,
    codec: Option<MoshCodec>,
    params: MoshGlitchParams,
    bitstream: MoshBitstreamParams,
    scanline_config: ScanlineCodecConfig,
    scanline_codec: Option<ScanlineCodec>,
    scanline_params: ScanlineGlitchParams,
    scanline_stats: MoshCodecStats,
    scanline_fresh: bool,
    dct_config: DctCodecConfig,
    dct_codec: Option<DctCodec>,
    dct_params: DctGlitchParams,
    dct_bitstream: DctBitstreamParams,
    dct_stats: MoshCodecStats,
    controls: RawMoshControls,
    rgb_input: Vec<u8>,
    rgb_output: Vec<u8>,
    last_bitstream_stats: MoshBitstreamMutationStats,
}

impl MoshEngine {
    pub fn new(width: usize, height: usize) -> io::Result<Self> {
        Self::with_backend(MoshEngineBackend::RawMoshV1, width, height)
    }

    pub fn with_backend(
        backend: MoshEngineBackend,
        width: usize,
        height: usize,
    ) -> io::Result<Self> {
        Self::with_backend_config(backend, MoshCodecConfig::new(width, height))
    }

    pub fn with_config(config: MoshCodecConfig) -> io::Result<Self> {
        Self::with_backend_config(MoshEngineBackend::RawMoshV1, config)
    }

    pub fn with_backend_config(
        backend: MoshEngineBackend,
        config: MoshCodecConfig,
    ) -> io::Result<Self> {
        let frame_len = config.frame_len().unwrap_or(0);
        let scanline_config = ScanlineCodecConfig::new(config.width, config.height);
        let dct_config = DctCodecConfig::new(config.width, config.height);
        let (codec, scanline_codec) = match backend {
            MoshEngineBackend::RawMoshV1 => (Some(MoshCodec::new(config)?), None),
            MoshEngineBackend::ScanlineSignalV1 => {
                (None, Some(ScanlineCodec::new(scanline_config)?))
            }
            MoshEngineBackend::DctTransformV1 => (None, None),
        };
        let dct_codec = match backend {
            MoshEngineBackend::DctTransformV1 => Some(DctCodec::new(dct_config)?),
            _ => None,
        };
        Ok(Self {
            backend,
            config,
            codec,
            params: MoshGlitchParams::default(),
            bitstream: MoshBitstreamParams::default(),
            scanline_config,
            scanline_codec,
            scanline_params: ScanlineGlitchParams::default(),
            scanline_stats: MoshCodecStats::default(),
            scanline_fresh: true,
            dct_config,
            dct_codec,
            dct_params: DctGlitchParams::default(),
            dct_bitstream: DctBitstreamParams::default(),
            dct_stats: MoshCodecStats::default(),
            controls: RawMoshControls::default(),
            rgb_input: Vec::with_capacity(frame_len),
            rgb_output: Vec::with_capacity(frame_len),
            last_bitstream_stats: MoshBitstreamMutationStats::default(),
        })
    }

    pub fn backend(&self) -> MoshEngineBackend {
        self.backend
    }

    pub fn config(&self) -> &MoshCodecConfig {
        &self.config
    }

    pub fn params(&self) -> &MoshGlitchParams {
        &self.params
    }

    pub fn bitstream_params(&self) -> &MoshBitstreamParams {
        &self.bitstream
    }

    pub fn scanline_config(&self) -> Option<&ScanlineCodecConfig> {
        (self.backend == MoshEngineBackend::ScanlineSignalV1).then_some(&self.scanline_config)
    }

    pub fn scanline_params(&self) -> Option<&ScanlineGlitchParams> {
        (self.backend == MoshEngineBackend::ScanlineSignalV1).then_some(&self.scanline_params)
    }

    pub fn dct_config(&self) -> Option<&DctCodecConfig> {
        (self.backend == MoshEngineBackend::DctTransformV1).then_some(&self.dct_config)
    }

    pub fn dct_params(&self) -> Option<&DctGlitchParams> {
        (self.backend == MoshEngineBackend::DctTransformV1).then_some(&self.dct_params)
    }

    pub fn controls(&self) -> RawMoshControls {
        self.controls
    }

    pub fn stats(&self) -> &MoshCodecStats {
        match self.backend {
            MoshEngineBackend::RawMoshV1 => self
                .codec
                .as_ref()
                .map(MoshCodec::stats)
                .unwrap_or(&self.scanline_stats),
            MoshEngineBackend::ScanlineSignalV1 => &self.scanline_stats,
            MoshEngineBackend::DctTransformV1 => &self.dct_stats,
        }
    }

    pub fn last_bitstream_stats(&self) -> &MoshBitstreamMutationStats {
        &self.last_bitstream_stats
    }

    pub fn set_preset(&mut self, name: &str) -> Result<(), String> {
        match self.backend {
            MoshEngineBackend::RawMoshV1 => {
                apply_raw_mosh_preset(
                    name,
                    &mut self.config,
                    &mut self.params,
                    &mut self.bitstream,
                )?;
                self.rebuild_codec()
            }
            MoshEngineBackend::ScanlineSignalV1 => {
                load_scanline_signal_preset(name, &mut self.scanline_params)?;
                if let Some(codec) = &mut self.scanline_codec {
                    codec.reset_glitch_state();
                    self.scanline_fresh = true;
                }
                Ok(())
            }
            MoshEngineBackend::DctTransformV1 => {
                load_dct_transform_preset(name, &mut self.dct_params)?;
                load_dct_bitstream_preset(name, &mut self.dct_bitstream);
                if let Some(codec) = &mut self.dct_codec {
                    codec.reset_glitch_state();
                }
                Ok(())
            }
        }
    }

    pub fn set_controls(&mut self, controls: RawMoshControls) {
        self.controls = normalized_raw_mosh_controls(controls);
    }

    pub fn reset_controls(&mut self) {
        self.controls = RawMoshControls::default();
    }

    pub fn set_parameter(&mut self, id: &str, value: f32) -> Result<(), String> {
        match self.backend {
            MoshEngineBackend::RawMoshV1 => {
                set_raw_mosh_parameter(
                    &mut self.config,
                    &mut self.params,
                    &mut self.bitstream,
                    id,
                    value,
                )?;
                if raw_mosh_parameter_requires_rebuild(id) {
                    self.rebuild_codec()?;
                }
            }
            MoshEngineBackend::ScanlineSignalV1 => {
                set_scanline_signal_parameter(&mut self.scanline_params, id, value)?;
            }
            MoshEngineBackend::DctTransformV1 => {
                if id == "quality" {
                    let quality = if value.is_finite() {
                        value.round().clamp(1.0, 100.0) as u8
                    } else {
                        DctCodecConfig::default().quality
                    };
                    if quality != self.dct_config.quality {
                        self.dct_config.quality = quality;
                        self.rebuild_codec()?;
                    }
                // Coefficient parameters first; fall back to the entropy-bitstream params.
                } else if set_dct_transform_parameter(&mut self.dct_params, id, value).is_err() {
                    set_dct_bitstream_parameter(&mut self.dct_bitstream, id, value)?;
                }
            }
        }
        Ok(())
    }

    pub fn reset_glitch(&mut self) {
        if let Some(codec) = &mut self.codec {
            codec.reset_glitch_state();
        }
        if let Some(codec) = &mut self.scanline_codec {
            codec.reset_glitch_state();
            self.scanline_fresh = true;
        }
        if let Some(codec) = &mut self.dct_codec {
            codec.reset_glitch_state();
        }
        self.last_bitstream_stats = MoshBitstreamMutationStats::default();
    }

    pub fn process_rgb24(
        &mut self,
        input: &[u8],
        output: &mut [u8],
    ) -> io::Result<MoshBitstreamMutationStats> {
        let frame_len = self.config.frame_len().ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                "frame dimensions overflow addressable memory",
            )
        })?;
        if input.len() != frame_len {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("input frame must be {frame_len} bytes of rgb24"),
            ));
        }
        if output.len() != frame_len {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("output frame must be {frame_len} bytes of rgb24"),
            ));
        }

        let stats = match self.backend {
            MoshEngineBackend::RawMoshV1 => {
                let (frame_params, frame_bitstream) = self.effective_frame_settings();
                let codec = self.codec.as_mut().expect("raw backend owns raw codec");
                if frame_bitstream.enabled || frame_bitstream.has_mutations() {
                    codec.process_rgb_frame_bitstream(
                        input,
                        &frame_params,
                        &frame_bitstream,
                        output,
                    )?
                } else {
                    codec.process_rgb_frame(input, &frame_params, output)?;
                    MoshBitstreamMutationStats::default()
                }
            }
            MoshEngineBackend::ScanlineSignalV1 => {
                let mut frame_params = self.scanline_params;
                apply_scanline_signal_controls(&mut frame_params, self.controls);
                self.scanline_codec
                    .as_mut()
                    .expect("scanline backend owns scanline codec")
                    .process_rgb_frame(input, &frame_params, output)?;
                self.update_scanline_compat_stats();
                MoshBitstreamMutationStats::default()
            }
            MoshEngineBackend::DctTransformV1 => {
                let mut frame_params = self.dct_params;
                apply_dct_transform_controls(&mut frame_params, self.controls);
                let mut frame_bitstream = self.dct_bitstream;
                apply_dct_bitstream_controls(&mut frame_bitstream, self.controls);
                let codec = self.dct_codec.as_mut().expect("dct backend owns dct codec");
                if frame_bitstream.has_mutations() {
                    codec.process_rgb_frame_bitstream(
                        input,
                        &frame_params,
                        &frame_bitstream,
                        output,
                    )?;
                } else {
                    codec.process_rgb_frame(input, &frame_params, output)?;
                }
                self.update_dct_compat_stats();
                MoshBitstreamMutationStats::default()
            }
        };
        self.last_bitstream_stats = stats;
        Ok(stats)
    }

    pub fn process_rgba8(
        &mut self,
        input: &[u8],
        output: &mut [u8],
    ) -> io::Result<MoshBitstreamMutationStats> {
        let pixels = self
            .config
            .width
            .checked_mul(self.config.height)
            .ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "frame dimensions overflow addressable memory",
                )
            })?;
        let rgba_len = pixels.checked_mul(4).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                "rgba frame dimensions overflow addressable memory",
            )
        })?;
        if input.len() != rgba_len {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("input frame must be {rgba_len} bytes of rgba8"),
            ));
        }
        if output.len() != rgba_len {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("output frame must be {rgba_len} bytes of rgba8"),
            ));
        }

        self.rgb_input.resize(pixels * RAW_RGB_CHANNELS, 0);
        self.rgb_output.resize(pixels * RAW_RGB_CHANNELS, 0);

        for index in 0..pixels {
            let rgba = index * 4;
            let rgb = index * RAW_RGB_CHANNELS;
            self.rgb_input[rgb..rgb + RAW_RGB_CHANNELS].copy_from_slice(&input[rgba..rgba + 3]);
        }

        let stats = self.process_rgb24_buffered()?;

        for index in 0..pixels {
            let rgba = index * 4;
            let rgb = index * RAW_RGB_CHANNELS;
            output[rgba..rgba + 3].copy_from_slice(&self.rgb_output[rgb..rgb + RAW_RGB_CHANNELS]);
            output[rgba + 3] = input[rgba + 3];
        }

        Ok(stats)
    }

    fn process_rgb24_buffered(&mut self) -> io::Result<MoshBitstreamMutationStats> {
        let stats = match self.backend {
            MoshEngineBackend::RawMoshV1 => {
                let (frame_params, frame_bitstream) = self.effective_frame_settings();
                let codec = self.codec.as_mut().expect("raw backend owns raw codec");
                if frame_bitstream.enabled || frame_bitstream.has_mutations() {
                    codec.process_rgb_frame_bitstream(
                        &self.rgb_input,
                        &frame_params,
                        &frame_bitstream,
                        &mut self.rgb_output,
                    )?
                } else {
                    codec.process_rgb_frame(
                        &self.rgb_input,
                        &frame_params,
                        &mut self.rgb_output,
                    )?;
                    MoshBitstreamMutationStats::default()
                }
            }
            MoshEngineBackend::ScanlineSignalV1 => {
                let mut frame_params = self.scanline_params;
                apply_scanline_signal_controls(&mut frame_params, self.controls);
                self.scanline_codec
                    .as_mut()
                    .expect("scanline backend owns scanline codec")
                    .process_rgb_frame(&self.rgb_input, &frame_params, &mut self.rgb_output)?;
                self.update_scanline_compat_stats();
                MoshBitstreamMutationStats::default()
            }
            MoshEngineBackend::DctTransformV1 => {
                let mut frame_params = self.dct_params;
                apply_dct_transform_controls(&mut frame_params, self.controls);
                let mut frame_bitstream = self.dct_bitstream;
                apply_dct_bitstream_controls(&mut frame_bitstream, self.controls);
                if frame_bitstream.has_mutations() {
                    self.dct_codec
                        .as_mut()
                        .expect("dct backend owns dct codec")
                        .process_rgb_frame_bitstream(
                            &self.rgb_input,
                            &frame_params,
                            &frame_bitstream,
                            &mut self.rgb_output,
                        )?;
                } else {
                    self.dct_codec
                        .as_mut()
                        .expect("dct backend owns dct codec")
                        .process_rgb_frame(&self.rgb_input, &frame_params, &mut self.rgb_output)?;
                }
                self.update_dct_compat_stats();
                MoshBitstreamMutationStats::default()
            }
        };
        self.last_bitstream_stats = stats;
        Ok(stats)
    }

    fn effective_frame_settings(&self) -> (MoshGlitchParams, MoshBitstreamParams) {
        let mut frame_params = self.params;
        let mut frame_bitstream = self.bitstream;
        apply_raw_mosh_controls(&mut frame_params, &mut frame_bitstream, self.controls);
        (frame_params, frame_bitstream)
    }

    fn rebuild_codec(&mut self) -> Result<(), String> {
        match self.backend {
            MoshEngineBackend::RawMoshV1 => {
                self.codec = Some(MoshCodec::new(self.config).map_err(|error| error.to_string())?);
            }
            MoshEngineBackend::ScanlineSignalV1 => {
                self.scanline_codec = Some(
                    ScanlineCodec::new(self.scanline_config).map_err(|error| error.to_string())?,
                );
                self.scanline_fresh = true;
            }
            MoshEngineBackend::DctTransformV1 => {
                self.dct_codec =
                    Some(DctCodec::new(self.dct_config).map_err(|error| error.to_string())?);
            }
        }
        self.last_bitstream_stats = MoshBitstreamMutationStats::default();
        Ok(())
    }

    fn update_scanline_compat_stats(&mut self) {
        self.scanline_stats.frames_in += 1;
        self.scanline_stats.blocks_encoded += self.scanline_config.height as u64;
        if self.scanline_fresh {
            self.scanline_stats.keyframes += 1;
            self.scanline_fresh = false;
        } else {
            self.scanline_stats.predicted_frames += 1;
        }
    }

    fn update_dct_compat_stats(&mut self) {
        self.dct_stats.frames_in += 1;
        let luma_blocks = self.dct_config.width.div_ceil(8) * self.dct_config.height.div_ceil(8);
        let chroma_blocks = self.dct_config.width.div_ceil(2).div_ceil(8)
            * self.dct_config.height.div_ceil(2).div_ceil(8);
        self.dct_stats.blocks_encoded += (luma_blocks + 2 * chroma_blocks) as u64;
        self.dct_stats.keyframes += 1;
    }
}

fn normalized_raw_mosh_controls(controls: RawMoshControls) -> RawMoshControls {
    RawMoshControls {
        intensity: finite_control(controls.intensity),
        motion: finite_control(controls.motion),
        residual: finite_control(controls.residual),
        temporal: finite_control(controls.temporal),
        bitstream: finite_control(controls.bitstream),
    }
}

fn finite_control(value: f32) -> f32 {
    if value.is_finite() {
        value.clamp(0.0, RAW_MOSH_CONTROL_MAX)
    } else {
        0.0
    }
}

pub fn load_scanline_signal_preset(
    name: &str,
    params: &mut ScanlineGlitchParams,
) -> Result<(), String> {
    *params = ScanlineGlitchParams::default();
    match name {
        "clean" => {}
        "subtle" => {
            params.line_shift = 2;
            params.line_shift_every = 13;
            params.burst_loss_every = 23;
        }
        "classic" | "tear" | "timebase-tear" => {
            params.line_shift = 14;
            params.line_shift_every = 6;
        }
        "vector" | "clock-skew" => {
            params.line_shift = 2;
            params.line_shift_every = 1;
        }
        "drift" | "vertical-roll" => {
            params.line_index_offset = 8;
            params.line_index_every = 1;
        }
        "sync" | "sync-dropout" => {
            params.sync_loss_every = 9;
            params.field_sync_loss_every = 2;
            params.field_parity_flip_every = 4;
            params.predictor_lag = 3;
        }
        "plane" | "phase" | "phase-walk" => {
            params.phase_drift = 1;
            params.chroma_sequence_offset = 1;
            params.chroma_sequence_every = 5;
        }
        "carrier" | "carrier-slip" => {
            params.phase_offset = 1;
            params.chroma_sequence_offset = 1;
            params.chroma_sequence_every = 3;
            params.chroma_payload_slip = 3;
            params.chroma_payload_slip_every = 5;
        }
        "chroma-grid" => {
            params.chroma_group_delta = 1;
            params.phase_offset = 1;
            params.chroma_seed_loss_every = 9;
        }
        "chroma-sequence" => {
            params.chroma_sequence_offset = 1;
            params.chroma_sequence_every = 1;
        }
        "burst-seed-loss" => {
            params.chroma_seed_loss_every = 4;
            params.burst_loss_every = 7;
        }
        "carrier-xor" => {
            params.chroma_xor_mask = 0x60;
            params.chroma_xor_every = 3;
            params.carrier_sign_flip_every = 11;
        }
        "melt" | "ghost" | "predictor-ghost" => {
            params.predictor_lag = 5;
            params.predictor_flip_every = 8;
        }
        "line-crosstalk" => {
            params.predictor_lag = 3;
            params.predictor_line_offset = 8;
            params.predictor_line_offset_every = 3;
        }
        "residue" | "rle-runaway" => {
            params.luma_run_delta = 5;
            params.luma_run_delta_every = 7;
            params.luma_payload_slip = 1;
            params.luma_payload_slip_every = 17;
        }
        "entropy" | "packet-length" => {
            params.packet_length_delta = 3;
            params.packet_length_delta_every = 13;
        }
        "plane-crosswire" => {
            params.payload_swap_every = 11;
        }
        "coeff" | "quantizer-pump" => {
            params.quant_offset = 4;
            params.quant_offset_every = 5;
        }
        "codebook" | "weave" | "history-weave" => {
            params.predictor_lag = 5;
            params.history_line_weave = 4;
            params.history_line_weave_every = 5;
            params.predictor_flip_every = 13;
        }
        "unstable" | "balanced" | "destroy" | "composite" | "composite-collapse" => {
            params.line_shift = 12;
            params.line_shift_every = 6;
            params.line_shift_drift = 1;
            params.line_index_offset = 1;
            params.line_index_every = 13;
            params.line_index_stride = 1;
            params.sync_loss_every = 31;
            params.field_sync_loss_every = 2;
            params.field_parity_flip_every = 3;
            params.phase_offset = 1;
            params.burst_loss_every = 11;
            params.predictor_flip_every = 9;
            params.predictor_lag = 4;
            params.predictor_line_offset = 5;
            params.predictor_line_offset_every = 12;
            params.quant_offset = 3;
            params.quant_offset_every = 7;
            params.luma_payload_slip = 2;
            params.luma_payload_slip_every = 8;
            params.chroma_payload_slip = -2;
            params.chroma_payload_slip_every = 9;
            params.chroma_sequence_offset = 1;
            params.chroma_sequence_every = 7;
            params.chroma_seed_loss_every = 19;
            params.chroma_xor_mask = 0x20;
            params.chroma_xor_every = 23;
            params.luma_run_delta = 3;
            params.luma_run_delta_every = 17;
            params.packet_length_delta = 2;
            params.packet_length_delta_every = 29;
            params.payload_swap_every = 37;
            params.history_line_weave = 3;
            params.history_line_weave_every = 10;
        }
        _ => {
            return Err(format!(
                "unknown scanline-signal preset `{name}`; expected clean, subtle, timebase-tear, clock-skew, vertical-roll, sync-dropout, phase-walk, carrier-slip, chroma-grid, chroma-sequence, burst-seed-loss, carrier-xor, predictor-ghost, line-crosstalk, rle-runaway, packet-length, plane-crosswire, quantizer-pump, history-weave, or composite-collapse"
            ));
        }
    }
    Ok(())
}

pub fn apply_scanline_signal_controls(
    params: &mut ScanlineGlitchParams,
    controls: RawMoshControls,
) {
    let motion = control_amount(controls.intensity, controls.motion);
    let residual = control_amount(controls.intensity, controls.residual);
    let temporal = control_amount(controls.intensity, controls.temporal);
    let bitstream = control_amount(controls.intensity, controls.bitstream);

    params.line_shift = scale_i16(params.line_shift, motion);
    params.line_shift_every = scale_event_interval(params.line_shift_every, motion);
    params.line_shift_drift = scale_i16(params.line_shift_drift, motion);
    params.line_index_offset = scale_i16(params.line_index_offset, motion);
    params.line_index_every = scale_event_interval(params.line_index_every, motion);
    params.line_index_stride = scale_i16(params.line_index_stride, motion);

    params.quant_offset = scale_i8(params.quant_offset, residual);
    params.quant_offset_every = scale_event_interval(params.quant_offset_every, residual);
    params.carrier_sign_flip_every = scale_event_interval(params.carrier_sign_flip_every, residual);
    params.chroma_group_delta = scale_i8(params.chroma_group_delta, residual);
    params.phase_offset = scale_i8(params.phase_offset, residual);
    params.phase_drift = scale_i8(params.phase_drift, residual);
    params.burst_loss_every = scale_event_interval(params.burst_loss_every, residual);
    params.chroma_sequence_offset = scale_i8(params.chroma_sequence_offset, residual);
    params.chroma_sequence_every = scale_event_interval(params.chroma_sequence_every, residual);
    params.chroma_seed_loss_every = scale_event_interval(params.chroma_seed_loss_every, residual);
    params.chroma_xor_mask = scale_u8(params.chroma_xor_mask, residual);
    params.chroma_xor_every = scale_event_interval(params.chroma_xor_every, residual);
    params.chroma_payload_slip = scale_i16(params.chroma_payload_slip, residual);
    params.chroma_payload_slip_every =
        scale_event_interval(params.chroma_payload_slip_every, residual);

    params.predictor_lag = 1 + scale_usize(params.predictor_lag.saturating_sub(1), temporal);
    params.predictor_flip_every = scale_event_interval(params.predictor_flip_every, temporal);
    params.predictor_line_offset = scale_i16(params.predictor_line_offset, temporal);
    params.predictor_line_offset_every =
        scale_event_interval(params.predictor_line_offset_every, temporal);
    params.history_line_weave = scale_usize(params.history_line_weave, temporal);
    params.history_line_weave_every =
        scale_event_interval(params.history_line_weave_every, temporal);

    params.sync_loss_every = scale_event_interval(params.sync_loss_every, bitstream);
    params.field_sync_loss_every = scale_event_interval(params.field_sync_loss_every, bitstream);
    params.field_parity_flip_every =
        scale_event_interval(params.field_parity_flip_every, bitstream);
    params.luma_payload_slip = scale_i16(params.luma_payload_slip, bitstream);
    params.luma_payload_slip_every =
        scale_event_interval(params.luma_payload_slip_every, bitstream);
    params.luma_run_delta = scale_i8(params.luma_run_delta, bitstream);
    params.luma_run_delta_every = scale_event_interval(params.luma_run_delta_every, bitstream);
    params.packet_length_delta = scale_i16(params.packet_length_delta, bitstream);
    params.packet_length_delta_every =
        scale_event_interval(params.packet_length_delta_every, bitstream);
    params.payload_swap_every = scale_event_interval(params.payload_swap_every, bitstream);
}

pub fn set_scanline_signal_parameter(
    params: &mut ScanlineGlitchParams,
    id: &str,
    value: f32,
) -> Result<(), String> {
    let finite = if value.is_finite() { value } else { 0.0 };
    match id {
        "line_shift" | "mv_jitter" => {
            params.line_shift = finite.round().clamp(-256.0, 256.0) as i16;
            if params.line_shift_every == 0 {
                params.line_shift_every = 1;
            }
        }
        "line_shift_every" => {
            params.line_shift_every = finite.round().clamp(0.0, 512.0) as u64;
        }
        "line_shift_drift" => {
            params.line_shift_drift = finite.round().clamp(-256.0, 256.0) as i16;
            if params.line_shift_every == 0 {
                params.line_shift_every = 1;
            }
        }
        "line_index_offset" | "temporal_slice_drift" => {
            params.line_index_offset = finite.round().clamp(-256.0, 256.0) as i16;
            if params.line_index_every == 0 {
                params.line_index_every = 1;
            }
        }
        "line_index_every" | "codebook_shuffle_every" => {
            params.line_index_every = finite.round().clamp(0.0, 512.0) as u64;
        }
        "line_index_stride" => {
            params.line_index_stride = finite.round().clamp(-64.0, 64.0) as i16;
            if params.line_index_every == 0 {
                params.line_index_every = 1;
            }
        }
        "sync_loss_every" => {
            params.sync_loss_every = finite.round().clamp(0.0, 512.0) as u64;
        }
        "field_sync_loss_every" => {
            params.field_sync_loss_every = finite.round().clamp(0.0, 512.0) as u64;
        }
        "field_parity_flip_every" => {
            params.field_parity_flip_every = finite.round().clamp(0.0, 512.0) as u64;
        }
        "phase_offset" | "residual_channel_shift" => {
            params.phase_offset = finite.round().clamp(-16.0, 16.0) as i8;
        }
        "phase_drift" | "mv_field_interpolation" => {
            params.phase_drift = finite.round().clamp(-16.0, 16.0) as i8;
        }
        "burst_loss_every" => {
            params.burst_loss_every = finite.round().clamp(0.0, 512.0) as u64;
        }
        "predictor_flip_every" => {
            params.predictor_flip_every = finite.round().clamp(0.0, 512.0) as u64;
        }
        "predictor_lag" | "reference_lag" => {
            params.predictor_lag = finite.round().clamp(1.0, 64.0) as usize;
        }
        "predictor_line_offset" => {
            params.predictor_line_offset = finite.round().clamp(-256.0, 256.0) as i16;
            if params.predictor_line_offset_every == 0 {
                params.predictor_line_offset_every = 1;
            }
        }
        "predictor_line_offset_every" => {
            params.predictor_line_offset_every = finite.round().clamp(0.0, 512.0) as u64;
        }
        "quant_offset" | "coeff_shift" => {
            params.quant_offset = finite.round().clamp(-64.0, 64.0) as i8;
            if params.quant_offset_every == 0 {
                params.quant_offset_every = 1;
            }
        }
        "quant_offset_every" => {
            params.quant_offset_every = finite.round().clamp(0.0, 512.0) as u64;
        }
        "luma_payload_slip" | "sample_address_desync" | "residual_address_jitter" => {
            params.luma_payload_slip = finite.round().clamp(-256.0, 256.0) as i16;
            if params.luma_payload_slip_every == 0 {
                params.luma_payload_slip_every = 1;
            }
        }
        "luma_payload_slip_every" | "entropy_slip_every" => {
            params.luma_payload_slip_every = finite.round().clamp(0.0, 512.0) as u64;
        }
        "chroma_payload_slip" => {
            params.chroma_payload_slip = finite.round().clamp(-256.0, 256.0) as i16;
            if params.chroma_payload_slip_every == 0 {
                params.chroma_payload_slip_every = 1;
            }
        }
        "chroma_payload_slip_every" => {
            params.chroma_payload_slip_every = finite.round().clamp(0.0, 512.0) as u64;
        }
        "carrier_sign_flip_every" => {
            params.carrier_sign_flip_every = finite.round().clamp(0.0, 512.0) as u64;
        }
        "chroma_group_delta" => {
            params.chroma_group_delta = finite.round().clamp(-6.0, 6.0) as i8;
        }
        "chroma_sequence_offset" => {
            params.chroma_sequence_offset = finite.round().clamp(-4.0, 4.0) as i8;
            if params.chroma_sequence_every == 0 {
                params.chroma_sequence_every = 1;
            }
        }
        "chroma_sequence_every" => {
            params.chroma_sequence_every = finite.round().clamp(0.0, 512.0) as u64;
        }
        "chroma_seed_loss_every" => {
            params.chroma_seed_loss_every = finite.round().clamp(0.0, 512.0) as u64;
        }
        "chroma_xor_mask" => {
            params.chroma_xor_mask = finite.round().clamp(0.0, 255.0) as u8;
        }
        "chroma_xor_every" => {
            params.chroma_xor_every = finite.round().clamp(0.0, 512.0) as u64;
        }
        "luma_run_delta" => {
            params.luma_run_delta = finite.round().clamp(-127.0, 127.0) as i8;
            if params.luma_run_delta_every == 0 {
                params.luma_run_delta_every = 1;
            }
        }
        "luma_run_delta_every" => {
            params.luma_run_delta_every = finite.round().clamp(0.0, 512.0) as u64;
        }
        "packet_length_delta" => {
            params.packet_length_delta = finite.round().clamp(-1024.0, 1024.0) as i16;
            if params.packet_length_delta_every == 0 {
                params.packet_length_delta_every = 1;
            }
        }
        "packet_length_delta_every" => {
            params.packet_length_delta_every = finite.round().clamp(0.0, 512.0) as u64;
        }
        "payload_swap_every" => {
            params.payload_swap_every = finite.round().clamp(0.0, 512.0) as u64;
        }
        "history_line_weave" | "codebook_stride" => {
            params.history_line_weave = finite.abs().round().clamp(0.0, 64.0) as usize;
            if params.history_line_weave_every == 0 && params.history_line_weave != 0 {
                params.history_line_weave_every = 1;
            }
        }
        "history_line_weave_every" | "codebook_replace_every" => {
            params.history_line_weave_every = finite.round().clamp(0.0, 512.0) as u64;
        }
        "mv_scale" => {
            params.line_shift = ((finite - 1.0) * 32.0).round().clamp(-256.0, 256.0) as i16;
            if params.line_shift != 0 && params.line_shift_every == 0 {
                params.line_shift_every = 1;
            }
        }
        "reference_bleed" => {
            params.history_line_weave = (finite.clamp(0.0, 1.0) * 8.0).round() as usize;
            params.history_line_weave_every = 3;
        }
        "reference_latch_frames" => {
            params.history_line_weave = finite.round().clamp(0.0, 64.0) as usize;
            params.history_line_weave_every = 1;
        }
        "residual_keep" => {
            params.quant_offset = ((finite - 1.0) * 8.0).round().clamp(-64.0, 64.0) as i8;
            params.quant_offset_every = 1;
        }
        "entropy_slip_windows" => {
            params.luma_payload_slip = finite.round().clamp(-256.0, 256.0) as i16;
        }
        "coeff_quant" => {
            params.quant_offset = finite.round().clamp(-64.0, 64.0) as i8;
            params.quant_offset_every = 1;
        }
        _ => return Err(format!("unknown scanline-signal parameter `{id}`")),
    }
    Ok(())
}

pub const DATAMOSH_STATUS_OK: i32 = 0;

pub const DATAMOSH_STATUS_NULL_POINTER: i32 = -1;

pub const DATAMOSH_STATUS_INVALID_UTF8: i32 = -2;

pub const DATAMOSH_STATUS_INVALID_ARGUMENT: i32 = -3;

pub const DATAMOSH_STATUS_PROCESS_ERROR: i32 = -4;

pub const DATAMOSH_STATUS_PANIC: i32 = -255;

pub const DATAMOSH_BACKEND_RAW_MOSH_V1: u32 = 1;

pub const DATAMOSH_BACKEND_SCANLINE_SIGNAL_V1: u32 = 2;

pub const DATAMOSH_BACKEND_DCT_TRANSFORM_V1: u32 = 3;

fn ffi_status(function: impl FnOnce() -> Result<(), i32>) -> i32 {
    match panic::catch_unwind(AssertUnwindSafe(function)) {
        Ok(Ok(())) => DATAMOSH_STATUS_OK,
        Ok(Err(status)) => status,
        Err(_) => DATAMOSH_STATUS_PANIC,
    }
}

fn ffi_engine_mut<'a>(engine: *mut MoshEngine) -> Result<&'a mut MoshEngine, i32> {
    if engine.is_null() {
        Err(DATAMOSH_STATUS_NULL_POINTER)
    } else {
        Ok(unsafe { &mut *engine })
    }
}

fn ffi_cstr<'a>(value: *const c_char) -> Result<&'a str, i32> {
    if value.is_null() {
        return Err(DATAMOSH_STATUS_NULL_POINTER);
    }
    unsafe { CStr::from_ptr(value) }
        .to_str()
        .map_err(|_| DATAMOSH_STATUS_INVALID_UTF8)
}

fn ffi_input_slice<'a>(data: *const u8, len: usize) -> Result<&'a [u8], i32> {
    if data.is_null() {
        Err(DATAMOSH_STATUS_NULL_POINTER)
    } else {
        Ok(unsafe { slice::from_raw_parts(data, len) })
    }
}

fn ffi_output_slice<'a>(data: *mut u8, len: usize) -> Result<&'a mut [u8], i32> {
    if data.is_null() {
        Err(DATAMOSH_STATUS_NULL_POINTER)
    } else {
        Ok(unsafe { slice::from_raw_parts_mut(data, len) })
    }
}

fn ffi_process_status(result: io::Result<MoshBitstreamMutationStats>) -> Result<(), i32> {
    result.map(|_| ()).map_err(|error| match error.kind() {
        io::ErrorKind::InvalidInput => DATAMOSH_STATUS_INVALID_ARGUMENT,
        _ => DATAMOSH_STATUS_PROCESS_ERROR,
    })
}

#[unsafe(no_mangle)]
pub extern "C" fn datamosh_status_message(status: i32) -> *const c_char {
    let message: &'static [u8] = match status {
        DATAMOSH_STATUS_OK => b"ok\0",
        DATAMOSH_STATUS_NULL_POINTER => b"null pointer\0",
        DATAMOSH_STATUS_INVALID_UTF8 => b"invalid utf-8\0",
        DATAMOSH_STATUS_INVALID_ARGUMENT => b"invalid argument\0",
        DATAMOSH_STATUS_PROCESS_ERROR => b"process error\0",
        DATAMOSH_STATUS_PANIC => b"panic\0",
        _ => b"unknown status\0",
    };
    message.as_ptr().cast()
}

#[unsafe(no_mangle)]
pub extern "C" fn datamosh_mosh_engine_backend_count() -> usize {
    3
}

#[unsafe(no_mangle)]
pub extern "C" fn datamosh_mosh_engine_default_backend() -> u32 {
    MoshEngineBackend::RawMoshV1.id()
}

#[unsafe(no_mangle)]
pub extern "C" fn datamosh_mosh_engine_backend_name(backend: u32) -> *const c_char {
    let name: &'static [u8] = match MoshEngineBackend::parse_id(backend) {
        Some(MoshEngineBackend::RawMoshV1) => b"raw_mosh_v1\0",
        Some(MoshEngineBackend::ScanlineSignalV1) => b"scanline_signal_v1\0",
        Some(MoshEngineBackend::DctTransformV1) => b"dct_transform_v1\0",
        None => b"unknown\0",
    };
    name.as_ptr().cast()
}

#[unsafe(no_mangle)]
pub extern "C" fn datamosh_mosh_engine_new(width: usize, height: usize) -> *mut MoshEngine {
    datamosh_mosh_engine_new_with_backend(MoshEngineBackend::RawMoshV1.id(), width, height)
}

#[unsafe(no_mangle)]
pub extern "C" fn datamosh_mosh_engine_new_with_backend(
    backend: u32,
    width: usize,
    height: usize,
) -> *mut MoshEngine {
    let Some(backend) = MoshEngineBackend::parse_id(backend) else {
        return ptr::null_mut();
    };
    match panic::catch_unwind(AssertUnwindSafe(|| {
        MoshEngine::with_backend(backend, width, height)
    })) {
        Ok(Ok(engine)) => Box::into_raw(Box::new(engine)),
        _ => ptr::null_mut(),
    }
}

#[unsafe(no_mangle)]
#[allow(clippy::not_unsafe_ptr_arg_deref)]
pub extern "C" fn datamosh_mosh_engine_backend(engine: *const MoshEngine) -> u32 {
    if engine.is_null() {
        return 0;
    }
    unsafe { &*engine }.backend().id()
}

#[unsafe(no_mangle)]
#[allow(clippy::not_unsafe_ptr_arg_deref)]
pub extern "C" fn datamosh_mosh_engine_free(engine: *mut MoshEngine) {
    if !engine.is_null() {
        drop(unsafe { Box::from_raw(engine) });
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn datamosh_mosh_engine_set_preset(
    engine: *mut MoshEngine,
    preset: *const c_char,
) -> i32 {
    ffi_status(|| {
        let engine = ffi_engine_mut(engine)?;
        let preset = ffi_cstr(preset)?;
        engine
            .set_preset(preset)
            .map_err(|_| DATAMOSH_STATUS_INVALID_ARGUMENT)
    })
}

#[unsafe(no_mangle)]
pub extern "C" fn datamosh_mosh_engine_set_controls(
    engine: *mut MoshEngine,
    intensity: f32,
    motion: f32,
    residual: f32,
    temporal: f32,
    bitstream: f32,
) -> i32 {
    ffi_status(|| {
        let engine = ffi_engine_mut(engine)?;
        engine.set_controls(RawMoshControls {
            intensity,
            motion,
            residual,
            temporal,
            bitstream,
        });
        Ok(())
    })
}

#[unsafe(no_mangle)]
pub extern "C" fn datamosh_mosh_engine_reset_controls(engine: *mut MoshEngine) -> i32 {
    ffi_status(|| {
        let engine = ffi_engine_mut(engine)?;
        engine.reset_controls();
        Ok(())
    })
}

#[unsafe(no_mangle)]
pub extern "C" fn datamosh_mosh_engine_set_parameter(
    engine: *mut MoshEngine,
    id: *const c_char,
    value: f32,
) -> i32 {
    ffi_status(|| {
        let engine = ffi_engine_mut(engine)?;
        let id = ffi_cstr(id)?;
        engine
            .set_parameter(id, value)
            .map_err(|_| DATAMOSH_STATUS_INVALID_ARGUMENT)
    })
}

#[unsafe(no_mangle)]
pub extern "C" fn datamosh_mosh_engine_reset_glitch(engine: *mut MoshEngine) -> i32 {
    ffi_status(|| {
        let engine = ffi_engine_mut(engine)?;
        engine.reset_glitch();
        Ok(())
    })
}

#[unsafe(no_mangle)]
pub extern "C" fn datamosh_mosh_engine_process_rgb24(
    engine: *mut MoshEngine,
    input: *const u8,
    input_len: usize,
    output: *mut u8,
    output_len: usize,
) -> i32 {
    ffi_status(|| {
        let engine = ffi_engine_mut(engine)?;
        if input.is_null() || output.is_null() {
            return Err(DATAMOSH_STATUS_NULL_POINTER);
        }
        if input == output as *const u8 {
            let input_copy = ffi_input_slice(input, input_len)?.to_vec();
            let output = ffi_output_slice(output, output_len)?;
            return ffi_process_status(engine.process_rgb24(&input_copy, output));
        }

        let input = ffi_input_slice(input, input_len)?;
        let output = ffi_output_slice(output, output_len)?;
        ffi_process_status(engine.process_rgb24(input, output))
    })
}

#[unsafe(no_mangle)]
pub extern "C" fn datamosh_mosh_engine_process_rgba8(
    engine: *mut MoshEngine,
    input: *const u8,
    input_len: usize,
    output: *mut u8,
    output_len: usize,
) -> i32 {
    ffi_status(|| {
        let engine = ffi_engine_mut(engine)?;
        if input.is_null() || output.is_null() {
            return Err(DATAMOSH_STATUS_NULL_POINTER);
        }
        if input == output as *const u8 {
            let input_copy = ffi_input_slice(input, input_len)?.to_vec();
            let output = ffi_output_slice(output, output_len)?;
            return ffi_process_status(engine.process_rgba8(&input_copy, output));
        }

        let input = ffi_input_slice(input, input_len)?;
        let output = ffi_output_slice(output, output_len)?;
        ffi_process_status(engine.process_rgba8(input, output))
    })
}

const RAW_MOSH_PRESET_INFOS: &[RawMoshPresetInfo] = &[
    RawMoshPresetInfo {
        name: "clean",
        group: RawMoshPresetGroup::Hybrid,
        title: "Clean decode",
        description: "Codec reconstruction without intentional state corruption.",
    },
    RawMoshPresetInfo {
        name: "subtle",
        group: RawMoshPresetGroup::Motion,
        title: "Subtle motion smear",
        description: "Lower-strength motion smear for audio-reactive ranges.",
    },
    RawMoshPresetInfo {
        name: "melt",
        group: RawMoshPresetGroup::Motion,
        title: "Motion melt",
        description: "Classic active-area dirty-reference smear.",
    },
    RawMoshPresetInfo {
        name: "classic",
        group: RawMoshPresetGroup::Motion,
        title: "Classic motion smear",
        description: "Readable baseline for comparing custom-codec variants.",
    },
    RawMoshPresetInfo {
        name: "vector",
        group: RawMoshPresetGroup::Motion,
        title: "Motion-vector bank desync",
        description: "Motion vectors come from wrong block banks.",
    },
    RawMoshPresetInfo {
        name: "drift",
        group: RawMoshPresetGroup::Reference,
        title: "Temporal slice drift",
        description: "Horizontal bands read different decoded reference ages.",
    },
    RawMoshPresetInfo {
        name: "plane",
        group: RawMoshPresetGroup::Reference,
        title: "Channel plane desync",
        description: "RGB planes read different channels and reference ages.",
    },
    RawMoshPresetInfo {
        name: "scan",
        group: RawMoshPresetGroup::Reference,
        title: "Scanline history desync",
        description: "Thin horizontal reference-history misreads.",
    },
    RawMoshPresetInfo {
        name: "residue",
        group: RawMoshPresetGroup::Residual,
        title: "Residual stream desync",
        description: "Residual address and channel corruption.",
    },
    RawMoshPresetInfo {
        name: "bank",
        group: RawMoshPresetGroup::Residual,
        title: "Residual bank swap",
        description: "Residual samples are decoded from wrong cells.",
    },
    RawMoshPresetInfo {
        name: "pixel",
        group: RawMoshPresetGroup::Residual,
        title: "Pixel dirty reference tearing",
        description: "Fine dirty-reference misreads.",
    },
    RawMoshPresetInfo {
        name: "grain",
        group: RawMoshPresetGroup::Residual,
        title: "Medium grain tearing",
        description: "Between melt and pixel.",
    },
    RawMoshPresetInfo {
        name: "entropy",
        group: RawMoshPresetGroup::Bitstream,
        title: "Entropy byte-slip",
        description: "Residual payload bytes slip before decode.",
    },
    RawMoshPresetInfo {
        name: "entropy-hard",
        group: RawMoshPresetGroup::Bitstream,
        title: "Hard entropy byte-slip",
        description: "Harsher residual payload byte-slip.",
    },
    RawMoshPresetInfo {
        name: "coeff",
        group: RawMoshPresetGroup::Bitstream,
        title: "Transform coefficient drift",
        description: "Residual coefficient tiles are damaged before inverse transform.",
    },
    RawMoshPresetInfo {
        name: "coeff-hard",
        group: RawMoshPresetGroup::Bitstream,
        title: "Hard coefficient drift",
        description: "Stronger transform coefficient corruption.",
    },
    RawMoshPresetInfo {
        name: "codebook",
        group: RawMoshPresetGroup::Bitstream,
        title: "Residual codebook leak",
        description: "Residual tiles are decoded from older dictionary slots.",
    },
    RawMoshPresetInfo {
        name: "codebook-hard",
        group: RawMoshPresetGroup::Bitstream,
        title: "Hard residual codebook leak",
        description: "Stronger residual dictionary slot replacement.",
    },
    RawMoshPresetInfo {
        name: "unstable",
        group: RawMoshPresetGroup::Hybrid,
        title: "Codec state collapse",
        description: "Chaotic dirty-reference and predictor desync.",
    },
    RawMoshPresetInfo {
        name: "balanced",
        group: RawMoshPresetGroup::Hybrid,
        title: "Balanced codec corruption",
        description: "Mid-strength mix of motion, reference, and residual damage.",
    },
    RawMoshPresetInfo {
        name: "destroy",
        group: RawMoshPresetGroup::Hybrid,
        title: "Destroy",
        description: "High-strength all-frame codec-state damage.",
    },
];

const RAW_MOSH_PARAMETER_INFOS: &[RawMoshParameterInfo] = &[
    RawMoshParameterInfo {
        id: "mv_scale",
        group: RawMoshPresetGroup::Motion,
        kind: RawMoshParameterKind::Float,
        label: "Motion scale",
        description: "Scales both motion-vector axes.",
        min: 0.0,
        max: 2.0,
        default: 1.0,
    },
    RawMoshParameterInfo {
        id: "mv_jitter",
        group: RawMoshPresetGroup::Motion,
        kind: RawMoshParameterKind::Integer,
        label: "Vector jitter",
        description: "Deterministic signed jitter added to motion vectors.",
        min: 0.0,
        max: 16.0,
        default: 0.0,
    },
    RawMoshParameterInfo {
        id: "mv_field_interpolation",
        group: RawMoshPresetGroup::Motion,
        kind: RawMoshParameterKind::Float,
        label: "Vector field interpolation",
        description: "Interpolates decoded motion vectors per pixel.",
        min: 0.0,
        max: 1.0,
        default: 0.0,
    },
    RawMoshParameterInfo {
        id: "sample_address_desync",
        group: RawMoshPresetGroup::Motion,
        kind: RawMoshParameterKind::Float,
        label: "Sample address desync",
        description: "Corrupts dirty reference sample addresses.",
        min: 0.0,
        max: 4.0,
        default: 0.0,
    },
    RawMoshParameterInfo {
        id: "mv_bank_stride",
        group: RawMoshPresetGroup::Motion,
        kind: RawMoshParameterKind::Integer,
        label: "Vector bank stride",
        description: "Signed motion-vector bank offset.",
        min: -64.0,
        max: 64.0,
        default: 0.0,
    },
    RawMoshParameterInfo {
        id: "reference_lag",
        group: RawMoshPresetGroup::Reference,
        kind: RawMoshParameterKind::Integer,
        label: "Reference lag",
        description: "Decode from an older reconstructed frame.",
        min: 1.0,
        max: 32.0,
        default: 1.0,
    },
    RawMoshParameterInfo {
        id: "reference_bleed",
        group: RawMoshPresetGroup::Reference,
        kind: RawMoshParameterKind::Float,
        label: "Reference bleed",
        description: "Minimum hard-switch chance to dirty references.",
        min: 0.0,
        max: 1.0,
        default: 0.0,
    },
    RawMoshParameterInfo {
        id: "reference_latch_frames",
        group: RawMoshPresetGroup::Reference,
        kind: RawMoshParameterKind::Integer,
        label: "Reference latch",
        description: "Keeps dirty-reference switch decisions stable.",
        min: 1.0,
        max: 64.0,
        default: 1.0,
    },
    RawMoshParameterInfo {
        id: "temporal_slice_drift",
        group: RawMoshPresetGroup::Reference,
        kind: RawMoshParameterKind::Integer,
        label: "Temporal drift",
        description: "Signed lag drift applied per latch bucket.",
        min: -16.0,
        max: 16.0,
        default: 0.0,
    },
    RawMoshParameterInfo {
        id: "reference_channel_lag_span",
        group: RawMoshPresetGroup::Reference,
        kind: RawMoshParameterKind::Integer,
        label: "Channel lag span",
        description: "Reference history span for per-channel plane desync.",
        min: 0.0,
        max: 32.0,
        default: 0.0,
    },
    RawMoshParameterInfo {
        id: "residual_keep",
        group: RawMoshPresetGroup::Residual,
        kind: RawMoshParameterKind::Float,
        label: "Residual gain",
        description: "Residual multiplier. 1 reconstructs, 0 smears motion.",
        min: -2.0,
        max: 2.0,
        default: 1.0,
    },
    RawMoshParameterInfo {
        id: "residual_address_jitter",
        group: RawMoshPresetGroup::Residual,
        kind: RawMoshParameterKind::Integer,
        label: "Residual address jitter",
        description: "Deterministic residual address jitter in pixels.",
        min: 0.0,
        max: 32.0,
        default: 0.0,
    },
    RawMoshParameterInfo {
        id: "residual_channel_shift",
        group: RawMoshPresetGroup::Residual,
        kind: RawMoshParameterKind::Integer,
        label: "Residual channel shift",
        description: "Rotates residual channel reads.",
        min: -4.0,
        max: 4.0,
        default: 0.0,
    },
    RawMoshParameterInfo {
        id: "residual_bank_stride",
        group: RawMoshPresetGroup::Residual,
        kind: RawMoshParameterKind::Integer,
        label: "Residual bank stride",
        description: "Signed residual-bank offset.",
        min: -64.0,
        max: 64.0,
        default: 0.0,
    },
    RawMoshParameterInfo {
        id: "entropy_slip_every",
        group: RawMoshPresetGroup::Bitstream,
        kind: RawMoshParameterKind::Integer,
        label: "Entropy slip period",
        description: "Byte-slip residual payload windows every nth P frame.",
        min: 0.0,
        max: 64.0,
        default: 0.0,
    },
    RawMoshParameterInfo {
        id: "entropy_slip_windows",
        group: RawMoshPresetGroup::Bitstream,
        kind: RawMoshParameterKind::Integer,
        label: "Entropy windows",
        description: "Number of byte-slip windows per affected frame.",
        min: 0.0,
        max: 64.0,
        default: 1.0,
    },
    RawMoshParameterInfo {
        id: "entropy_resync_bytes",
        group: RawMoshPresetGroup::Bitstream,
        kind: RawMoshParameterKind::Integer,
        label: "Entropy resync bytes",
        description: "Window length before simulated decoder resync.",
        min: 0.0,
        max: 65_536.0,
        default: 0.0,
    },
    RawMoshParameterInfo {
        id: "coeff_shift",
        group: RawMoshPresetGroup::Bitstream,
        kind: RawMoshParameterKind::Integer,
        label: "Coefficient shift",
        description: "Signed coefficient rotation excluding DC.",
        min: -32.0,
        max: 32.0,
        default: 0.0,
    },
    RawMoshParameterInfo {
        id: "coeff_quant",
        group: RawMoshPresetGroup::Bitstream,
        kind: RawMoshParameterKind::Integer,
        label: "Coefficient quant",
        description: "Quantizes transformed residual coefficients.",
        min: 1.0,
        max: 64.0,
        default: 1.0,
    },
    RawMoshParameterInfo {
        id: "codebook_replace_every",
        group: RawMoshPresetGroup::Bitstream,
        kind: RawMoshParameterKind::Integer,
        label: "Codebook period",
        description: "Replaces every nth residual tile with a codebook tile.",
        min: 0.0,
        max: 64.0,
        default: 0.0,
    },
    RawMoshParameterInfo {
        id: "codebook_stride",
        group: RawMoshPresetGroup::Bitstream,
        kind: RawMoshParameterKind::Integer,
        label: "Codebook stride",
        description: "Signed dictionary slot offset for replacement.",
        min: -128.0,
        max: 128.0,
        default: 1.0,
    },
    RawMoshParameterInfo {
        id: "codebook_shuffle_every",
        group: RawMoshPresetGroup::Bitstream,
        kind: RawMoshParameterKind::Integer,
        label: "Codebook shuffle",
        description: "Randomizes selected codebook slot reads.",
        min: 0.0,
        max: 64.0,
        default: 0.0,
    },
    RawMoshParameterInfo {
        id: "block_size",
        group: RawMoshPresetGroup::Hybrid,
        kind: RawMoshParameterKind::Integer,
        label: "Block size",
        description: "Motion block size. Recreate the codec after changing.",
        min: 4.0,
        max: 64.0,
        default: 16.0,
    },
    RawMoshParameterInfo {
        id: "search_radius",
        group: RawMoshPresetGroup::Hybrid,
        kind: RawMoshParameterKind::Integer,
        label: "Search radius",
        description: "Motion search radius. Recreate the codec after changing.",
        min: 0.0,
        max: 64.0,
        default: 8.0,
    },
    RawMoshParameterInfo {
        id: "search_step",
        group: RawMoshPresetGroup::Hybrid,
        kind: RawMoshParameterKind::Integer,
        label: "Search step",
        description: "Motion search step. Recreate the codec after changing.",
        min: 1.0,
        max: 16.0,
        default: 4.0,
    },
    RawMoshParameterInfo {
        id: "bitstream_enabled",
        group: RawMoshPresetGroup::Hybrid,
        kind: RawMoshParameterKind::Bool,
        label: "Bitstream enabled",
        description: "Forces the MSH0 bitstream mutation path on or off.",
        min: 0.0,
        max: 1.0,
        default: 0.0,
    },
];

pub fn apply_raw_mosh_preset(
    name: &str,
    config: &mut MoshCodecConfig,
    params: &mut MoshGlitchParams,
    bitstream: &mut MoshBitstreamParams,
) -> Result<(), String> {
    // Presets own these codec-search fields and all corruption state. Resetting
    // them here makes a preset deterministic regardless of the previous preset.
    config.block_size = 16;
    config.search_radius = 8;
    config.search_step = 4;
    *params = MoshGlitchParams::default();
    *bitstream = MoshBitstreamParams::default();

    match name {
        "clean" | "none" | "off" => {}
        "subtle" => {
            params.mv_scale_x = 1.06;
            params.mv_scale_y = 1.03;
            params.mv_jitter = 1;
            params.mv_quant = 2;
            params.reference_lag = 2;
            params.residual_keep = 0.75;
            params.residual_invert_every = 0;
            params.block_remap_every = 0;
            params.block_remap_stride = 0;
            params.channel_shift = 0;
            params.wrap_motion = false;
            params.activity_mode = ActivityMode::Active;
            params.activity_threshold = 10;
            params.activity_softness = 14;
            params.reference_bleed = 0.08;
            params.reference_latch_frames = 4;
            params.reference_slot_count = 1;
            params.reference_slot_shuffle_every = 0;
            params.overlap = 4;
            params.motion_diffusion = 0.12;
            params.mv_field_interpolation = 0.25;
            params.sample_address_desync = 0.25;
            params.glitch_cell_size = 0;
            params.glitch_cell_width = 0;
            params.glitch_cell_height = 0;
            params.mv_predictor_desync_every = 0;
            params.mv_predictor_desync_x = 0;
            params.mv_predictor_desync_y = 0;
        }
        "classic" | "mosh" => {
            params.mv_scale_x = 1.0;
            params.mv_scale_y = 1.0;
            params.mv_jitter = 1;
            params.mv_quant = 2;
            params.reference_lag = 3;
            params.residual_keep = 0.22;
            params.residual_invert_every = 0;
            params.block_remap_every = 0;
            params.block_remap_stride = 0;
            params.channel_shift = 0;
            params.wrap_motion = false;
            params.activity_mode = ActivityMode::Active;
            params.activity_threshold = 12;
            params.activity_softness = 0;
            params.reference_bleed = 0.12;
            params.reference_latch_frames = 5;
            params.reference_slot_count = 4;
            params.reference_slot_shuffle_every = 11;
            params.overlap = 0;
            params.motion_diffusion = 0.0;
            params.mv_field_interpolation = 0.7;
            params.sample_address_desync = 0.65;
            params.glitch_cell_size = 0;
            params.glitch_cell_width = 0;
            params.glitch_cell_height = 0;
            params.mv_predictor_desync_every = 17;
            params.mv_predictor_desync_x = 1;
            params.mv_predictor_desync_y = 0;
        }
        "melt" | "motion-melt" => {
            params.mv_scale_x = 1.0;
            params.mv_scale_y = 1.0;
            params.mv_jitter = 1;
            params.mv_quant = 2;
            params.reference_lag = 8;
            params.residual_keep = 0.0;
            params.residual_invert_every = 0;
            params.block_remap_every = 0;
            params.block_remap_stride = 0;
            params.channel_shift = 0;
            params.wrap_motion = false;
            params.activity_mode = ActivityMode::Active;
            params.activity_threshold = 10;
            params.activity_softness = 0;
            params.reference_bleed = 0.16;
            params.reference_latch_frames = 6;
            params.reference_slot_count = 6;
            params.reference_slot_shuffle_every = 9;
            params.overlap = 0;
            params.motion_diffusion = 0.0;
            params.mv_field_interpolation = 0.85;
            params.sample_address_desync = 1.15;
            params.glitch_cell_size = 0;
            params.glitch_cell_width = 0;
            params.glitch_cell_height = 0;
            params.mv_predictor_desync_every = 11;
            params.mv_predictor_desync_x = 2;
            params.mv_predictor_desync_y = -1;
        }
        "grain" | "medium" | "meso" => {
            params.mv_scale_x = 1.0;
            params.mv_scale_y = 1.0;
            params.mv_jitter = 1;
            params.mv_quant = 1;
            params.reference_lag = 8;
            params.residual_keep = 0.0;
            params.residual_invert_every = 0;
            params.block_remap_every = 0;
            params.block_remap_stride = 0;
            params.channel_shift = 0;
            params.wrap_motion = false;
            params.activity_mode = ActivityMode::Active;
            params.activity_threshold = 10;
            params.activity_softness = 0;
            params.reference_bleed = 0.15;
            params.reference_latch_frames = 6;
            params.reference_slot_count = 7;
            params.reference_slot_shuffle_every = 6;
            params.overlap = 0;
            params.motion_diffusion = 0.0;
            params.mv_field_interpolation = 0.88;
            params.sample_address_desync = 1.3;
            params.glitch_cell_size = 0;
            params.glitch_cell_width = 3;
            params.glitch_cell_height = 2;
            params.mv_predictor_desync_every = 10;
            params.mv_predictor_desync_x = 2;
            params.mv_predictor_desync_y = -1;
        }
        "residue" | "residual" => {
            config.block_size = 32;
            config.search_radius = 0;
            config.search_step = 1;
            params.mv_scale_x = 1.0;
            params.mv_scale_y = 1.0;
            params.mv_jitter = 0;
            params.mv_quant = 1;
            params.reference_lag = 1;
            params.residual_keep = 1.35;
            params.residual_invert_every = 0;
            params.residual_address_shift_x = 2;
            params.residual_address_shift_y = 0;
            params.residual_address_jitter = 5;
            params.residual_channel_shift = 1;
            params.block_remap_every = 0;
            params.block_remap_stride = 0;
            params.channel_shift = 0;
            params.wrap_motion = false;
            params.activity_mode = ActivityMode::All;
            params.activity_threshold = 0;
            params.activity_softness = 0;
            params.reference_bleed = 0.0;
            params.reference_latch_frames = 4;
            params.reference_slot_count = 1;
            params.reference_slot_shuffle_every = 0;
            params.overlap = 0;
            params.motion_diffusion = 0.0;
            params.mv_field_interpolation = 0.0;
            params.sample_address_desync = 0.0;
            params.glitch_cell_size = 0;
            params.glitch_cell_width = 3;
            params.glitch_cell_height = 1;
            params.mv_predictor_desync_every = 0;
            params.mv_predictor_desync_x = 0;
            params.mv_predictor_desync_y = 0;
        }
        "scan" | "scanline" => {
            config.block_size = 32;
            config.search_radius = 0;
            config.search_step = 1;
            params.mv_scale_x = 1.0;
            params.mv_scale_y = 1.0;
            params.mv_jitter = 0;
            params.mv_quant = 1;
            params.reference_lag = 1;
            params.residual_keep = 0.65;
            params.residual_invert_every = 0;
            params.block_remap_every = 0;
            params.block_remap_stride = 0;
            params.channel_shift = 0;
            params.wrap_motion = false;
            params.activity_mode = ActivityMode::All;
            params.activity_threshold = 0;
            params.activity_softness = 0;
            params.reference_bleed = 0.0;
            params.reference_latch_frames = 2;
            params.reference_slot_count = 1;
            params.reference_slot_shuffle_every = 0;
            params.reference_scanline_height = 2;
            params.reference_scanline_lag_span = 4;
            params.overlap = 0;
            params.motion_diffusion = 0.0;
            params.mv_field_interpolation = 0.0;
            params.sample_address_desync = 0.0;
            params.glitch_cell_size = 0;
            params.glitch_cell_width = 0;
            params.glitch_cell_height = 0;
            params.mv_predictor_desync_every = 0;
            params.mv_predictor_desync_x = 0;
            params.mv_predictor_desync_y = 0;
        }
        "drift" | "slice-drift" | "temporal" => {
            config.block_size = 32;
            config.search_radius = 0;
            config.search_step = 1;
            params.mv_scale_x = 1.0;
            params.mv_scale_y = 1.0;
            params.mv_jitter = 0;
            params.mv_quant = 1;
            params.reference_lag = 1;
            params.residual_keep = 0.75;
            params.residual_invert_every = 0;
            params.block_remap_every = 0;
            params.block_remap_stride = 0;
            params.channel_shift = 0;
            params.wrap_motion = false;
            params.activity_mode = ActivityMode::All;
            params.activity_threshold = 0;
            params.activity_softness = 0;
            params.reference_bleed = 0.0;
            params.reference_latch_frames = 2;
            params.reference_slot_count = 1;
            params.reference_slot_shuffle_every = 0;
            params.temporal_slice_height = 12;
            params.temporal_slice_lag_span = 8;
            params.temporal_slice_drift = 1;
            params.overlap = 0;
            params.motion_diffusion = 0.0;
            params.mv_field_interpolation = 0.0;
            params.sample_address_desync = 0.0;
            params.glitch_cell_size = 0;
            params.glitch_cell_width = 0;
            params.glitch_cell_height = 0;
            params.mv_predictor_desync_every = 0;
            params.mv_predictor_desync_x = 0;
            params.mv_predictor_desync_y = 0;
        }
        "bank" | "residual-bank" | "swap" => {
            config.block_size = 32;
            config.search_radius = 0;
            config.search_step = 1;
            params.mv_scale_x = 1.0;
            params.mv_scale_y = 1.0;
            params.mv_jitter = 0;
            params.mv_quant = 1;
            params.reference_lag = 1;
            params.residual_keep = 1.2;
            params.residual_invert_every = 0;
            params.residual_channel_shift = 1;
            params.residual_bank_size = 24;
            params.residual_bank_stride = 5;
            params.residual_bank_shuffle_every = 3;
            params.block_remap_every = 0;
            params.block_remap_stride = 0;
            params.channel_shift = 0;
            params.wrap_motion = false;
            params.activity_mode = ActivityMode::All;
            params.activity_threshold = 0;
            params.activity_softness = 0;
            params.reference_bleed = 0.0;
            params.reference_latch_frames = 3;
            params.reference_slot_count = 1;
            params.reference_slot_shuffle_every = 0;
            params.overlap = 0;
            params.motion_diffusion = 0.0;
            params.mv_field_interpolation = 0.0;
            params.sample_address_desync = 0.0;
            params.glitch_cell_size = 0;
            params.glitch_cell_width = 0;
            params.glitch_cell_height = 0;
            params.mv_predictor_desync_every = 0;
            params.mv_predictor_desync_x = 0;
            params.mv_predictor_desync_y = 0;
        }
        "plane" | "channel-plane" | "rgb-plane" => {
            config.block_size = 32;
            config.search_radius = 0;
            config.search_step = 1;
            params.mv_scale_x = 1.0;
            params.mv_scale_y = 1.0;
            params.mv_jitter = 0;
            params.mv_quant = 1;
            params.reference_lag = 1;
            params.residual_keep = 0.85;
            params.residual_invert_every = 0;
            params.residual_channel_shift = -1;
            params.reference_channel_shift = 1;
            params.reference_channel_lag_span = 6;
            params.reference_channel_lag_stride = 1;
            params.block_remap_every = 0;
            params.block_remap_stride = 0;
            params.channel_shift = 0;
            params.wrap_motion = false;
            params.activity_mode = ActivityMode::All;
            params.activity_threshold = 0;
            params.activity_softness = 0;
            params.reference_bleed = 0.0;
            params.reference_latch_frames = 3;
            params.reference_slot_count = 1;
            params.reference_slot_shuffle_every = 0;
            params.overlap = 0;
            params.motion_diffusion = 0.0;
            params.mv_field_interpolation = 0.0;
            params.sample_address_desync = 0.0;
            params.glitch_cell_size = 0;
            params.glitch_cell_width = 0;
            params.glitch_cell_height = 0;
            params.mv_predictor_desync_every = 0;
            params.mv_predictor_desync_x = 0;
            params.mv_predictor_desync_y = 0;
        }
        "vector" | "vector-bank" | "field" => {
            config.block_size = 24;
            config.search_radius = 8;
            config.search_step = 4;
            params.mv_scale_x = 1.0;
            params.mv_scale_y = 1.0;
            params.mv_jitter = 1;
            params.mv_quant = 2;
            params.reference_lag = 5;
            params.residual_keep = 0.16;
            params.residual_invert_every = 0;
            params.mv_bank_size = 2;
            params.mv_bank_stride = 3;
            params.mv_bank_shuffle_every = 4;
            params.block_remap_every = 0;
            params.block_remap_stride = 0;
            params.channel_shift = 0;
            params.wrap_motion = false;
            params.activity_mode = ActivityMode::Active;
            params.activity_threshold = 8;
            params.activity_softness = 6;
            params.reference_bleed = 0.12;
            params.reference_latch_frames = 4;
            params.reference_slot_count = 4;
            params.reference_slot_shuffle_every = 0;
            params.overlap = 0;
            params.motion_diffusion = 0.0;
            params.mv_field_interpolation = 0.82;
            params.sample_address_desync = 0.45;
            params.glitch_cell_size = 0;
            params.glitch_cell_width = 0;
            params.glitch_cell_height = 0;
            params.mv_predictor_desync_every = 13;
            params.mv_predictor_desync_x = 1;
            params.mv_predictor_desync_y = -1;
        }
        "entropy" | "byte-slip" | "stream-slip" => {
            config.block_size = 16;
            config.search_radius = 4;
            config.search_step = 4;
            params.mv_scale_x = 1.0;
            params.mv_scale_y = 1.0;
            params.mv_jitter = 0;
            params.mv_quant = 1;
            params.reference_lag = 1;
            params.residual_keep = 1.0;
            params.residual_invert_every = 0;
            params.block_remap_every = 0;
            params.block_remap_stride = 0;
            params.channel_shift = 0;
            params.wrap_motion = false;
            params.activity_mode = ActivityMode::All;
            params.activity_threshold = 0;
            params.activity_softness = 0;
            params.reference_bleed = 0.0;
            params.reference_latch_frames = 3;
            params.reference_slot_count = 1;
            params.reference_slot_shuffle_every = 0;
            params.overlap = 0;
            params.motion_diffusion = 0.0;
            params.mv_field_interpolation = 0.0;
            params.sample_address_desync = 0.0;
            params.glitch_cell_size = 0;
            params.glitch_cell_width = 0;
            params.glitch_cell_height = 0;
            params.mv_predictor_desync_every = 0;
            params.mv_predictor_desync_x = 0;
            params.mv_predictor_desync_y = 0;
            bitstream.enabled = true;
            bitstream.entropy_slip_every = 2;
            bitstream.entropy_slip_bytes = 1;
            bitstream.entropy_resync_bytes = 4096;
            bitstream.entropy_slip_windows = 4;
        }
        "entropy-hard" | "byte-slip-hard" | "stream-slip-hard" => {
            config.block_size = 16;
            config.search_radius = 4;
            config.search_step = 4;
            params.mv_scale_x = 1.0;
            params.mv_scale_y = 1.0;
            params.mv_jitter = 0;
            params.mv_quant = 1;
            params.reference_lag = 1;
            params.residual_keep = 1.0;
            params.residual_invert_every = 0;
            params.block_remap_every = 0;
            params.block_remap_stride = 0;
            params.channel_shift = 0;
            params.wrap_motion = false;
            params.activity_mode = ActivityMode::All;
            params.activity_threshold = 0;
            params.activity_softness = 0;
            params.reference_bleed = 0.0;
            params.reference_latch_frames = 3;
            params.reference_slot_count = 1;
            params.reference_slot_shuffle_every = 0;
            params.overlap = 0;
            params.motion_diffusion = 0.0;
            params.mv_field_interpolation = 0.0;
            params.sample_address_desync = 0.0;
            params.glitch_cell_size = 0;
            params.glitch_cell_width = 0;
            params.glitch_cell_height = 0;
            params.mv_predictor_desync_every = 0;
            params.mv_predictor_desync_x = 0;
            params.mv_predictor_desync_y = 0;
            bitstream.enabled = true;
            bitstream.entropy_slip_every = 1;
            bitstream.entropy_slip_bytes = 1;
            bitstream.entropy_resync_bytes = 8192;
            bitstream.entropy_slip_windows = 10;
        }
        "coeff" | "coefficient" | "transform" | "frequency" => {
            config.block_size = 16;
            config.search_radius = 4;
            config.search_step = 4;
            params.mv_scale_x = 1.0;
            params.mv_scale_y = 1.0;
            params.mv_jitter = 0;
            params.mv_quant = 1;
            params.reference_lag = 1;
            params.residual_keep = 1.0;
            params.residual_invert_every = 0;
            params.block_remap_every = 0;
            params.block_remap_stride = 0;
            params.channel_shift = 0;
            params.wrap_motion = false;
            params.activity_mode = ActivityMode::All;
            params.activity_threshold = 0;
            params.activity_softness = 0;
            params.reference_bleed = 0.0;
            params.reference_latch_frames = 2;
            params.reference_slot_count = 1;
            params.reference_slot_shuffle_every = 0;
            params.overlap = 0;
            params.motion_diffusion = 0.0;
            params.mv_field_interpolation = 0.0;
            params.sample_address_desync = 0.0;
            params.glitch_cell_size = 0;
            params.glitch_cell_width = 0;
            params.glitch_cell_height = 0;
            params.mv_predictor_desync_every = 0;
            params.mv_predictor_desync_x = 0;
            params.mv_predictor_desync_y = 0;
            bitstream.enabled = true;
            bitstream.coeff_glitch_every = 1;
            bitstream.coeff_block_size = 8;
            bitstream.coeff_shift = 2;
            bitstream.coeff_sign_flip_every = 19;
            bitstream.coeff_zero_high = 12;
            bitstream.coeff_quant = 4;
        }
        "coeff-hard" | "coefficient-hard" | "transform-hard" | "frequency-hard" => {
            config.block_size = 16;
            config.search_radius = 4;
            config.search_step = 4;
            params.mv_scale_x = 1.0;
            params.mv_scale_y = 1.0;
            params.mv_jitter = 0;
            params.mv_quant = 1;
            params.reference_lag = 1;
            params.residual_keep = 1.0;
            params.residual_invert_every = 0;
            params.block_remap_every = 0;
            params.block_remap_stride = 0;
            params.channel_shift = 0;
            params.wrap_motion = false;
            params.activity_mode = ActivityMode::All;
            params.activity_threshold = 0;
            params.activity_softness = 0;
            params.reference_bleed = 0.0;
            params.reference_latch_frames = 2;
            params.reference_slot_count = 1;
            params.reference_slot_shuffle_every = 0;
            params.overlap = 0;
            params.motion_diffusion = 0.0;
            params.mv_field_interpolation = 0.0;
            params.sample_address_desync = 0.0;
            params.glitch_cell_size = 0;
            params.glitch_cell_width = 0;
            params.glitch_cell_height = 0;
            params.mv_predictor_desync_every = 0;
            params.mv_predictor_desync_x = 0;
            params.mv_predictor_desync_y = 0;
            bitstream.enabled = true;
            bitstream.coeff_glitch_every = 1;
            bitstream.coeff_block_size = 8;
            bitstream.coeff_shift = 5;
            bitstream.coeff_sign_flip_every = 7;
            bitstream.coeff_zero_high = 9;
            bitstream.coeff_quant = 8;
        }
        "codebook" | "dictionary" | "dict" => {
            config.block_size = 16;
            config.search_radius = 4;
            config.search_step = 4;
            params.mv_scale_x = 1.0;
            params.mv_scale_y = 1.0;
            params.mv_jitter = 0;
            params.mv_quant = 1;
            params.reference_lag = 1;
            params.residual_keep = 1.0;
            params.residual_invert_every = 0;
            params.block_remap_every = 0;
            params.block_remap_stride = 0;
            params.channel_shift = 0;
            params.wrap_motion = false;
            params.activity_mode = ActivityMode::All;
            params.activity_threshold = 0;
            params.activity_softness = 0;
            params.reference_bleed = 0.0;
            params.reference_latch_frames = 4;
            params.reference_slot_count = 1;
            params.reference_slot_shuffle_every = 0;
            params.overlap = 0;
            params.motion_diffusion = 0.0;
            params.mv_field_interpolation = 0.0;
            params.sample_address_desync = 0.0;
            params.glitch_cell_size = 0;
            params.glitch_cell_width = 0;
            params.glitch_cell_height = 0;
            params.mv_predictor_desync_every = 0;
            params.mv_predictor_desync_x = 0;
            params.mv_predictor_desync_y = 0;
            bitstream.enabled = true;
            bitstream.codebook_replace_every = 9;
            bitstream.codebook_tile_size = 8;
            bitstream.codebook_slots = 96;
            bitstream.codebook_stride = -17;
            bitstream.codebook_update_every = 3;
            bitstream.codebook_shuffle_every = 5;
        }
        "codebook-hard" | "dictionary-hard" | "dict-hard" => {
            config.block_size = 16;
            config.search_radius = 4;
            config.search_step = 4;
            params.mv_scale_x = 1.0;
            params.mv_scale_y = 1.0;
            params.mv_jitter = 0;
            params.mv_quant = 1;
            params.reference_lag = 1;
            params.residual_keep = 1.0;
            params.residual_invert_every = 0;
            params.block_remap_every = 0;
            params.block_remap_stride = 0;
            params.channel_shift = 0;
            params.wrap_motion = false;
            params.activity_mode = ActivityMode::All;
            params.activity_threshold = 0;
            params.activity_softness = 0;
            params.reference_bleed = 0.0;
            params.reference_latch_frames = 4;
            params.reference_slot_count = 1;
            params.reference_slot_shuffle_every = 0;
            params.overlap = 0;
            params.motion_diffusion = 0.0;
            params.mv_field_interpolation = 0.0;
            params.sample_address_desync = 0.0;
            params.glitch_cell_size = 0;
            params.glitch_cell_width = 0;
            params.glitch_cell_height = 0;
            params.mv_predictor_desync_every = 0;
            params.mv_predictor_desync_x = 0;
            params.mv_predictor_desync_y = 0;
            bitstream.enabled = true;
            bitstream.codebook_replace_every = 4;
            bitstream.codebook_tile_size = 8;
            bitstream.codebook_slots = 128;
            bitstream.codebook_stride = -31;
            bitstream.codebook_update_every = 2;
            bitstream.codebook_shuffle_every = 2;
        }
        "pixel" | "pixel-melt" | "micro" => {
            params.mv_scale_x = 1.0;
            params.mv_scale_y = 1.0;
            params.mv_jitter = 1;
            params.mv_quant = 1;
            params.reference_lag = 8;
            params.residual_keep = 0.0;
            params.residual_invert_every = 0;
            params.block_remap_every = 0;
            params.block_remap_stride = 0;
            params.channel_shift = 0;
            params.wrap_motion = false;
            params.activity_mode = ActivityMode::Active;
            params.activity_threshold = 10;
            params.activity_softness = 0;
            params.reference_bleed = 0.14;
            params.reference_latch_frames = 5;
            params.reference_slot_count = 8;
            params.reference_slot_shuffle_every = 5;
            params.overlap = 0;
            params.motion_diffusion = 0.0;
            params.mv_field_interpolation = 0.9;
            params.sample_address_desync = 1.25;
            params.glitch_cell_size = 1;
            params.glitch_cell_width = 1;
            params.glitch_cell_height = 1;
            params.mv_predictor_desync_every = 9;
            params.mv_predictor_desync_x = 2;
            params.mv_predictor_desync_y = -1;
        }
        "unstable" | "desync" => {
            params.mv_scale_x = 1.0;
            params.mv_scale_y = 1.0;
            params.mv_jitter = 2;
            params.mv_quant = 2;
            params.reference_lag = 10;
            params.residual_keep = 0.0;
            params.residual_invert_every = 0;
            params.block_remap_every = 0;
            params.block_remap_stride = 0;
            params.channel_shift = 0;
            params.wrap_motion = false;
            params.activity_mode = ActivityMode::Active;
            params.activity_threshold = 8;
            params.activity_softness = 0;
            params.reference_bleed = 0.24;
            params.reference_latch_frames = 9;
            params.reference_slot_count = 10;
            params.reference_slot_shuffle_every = 7;
            params.overlap = 0;
            params.motion_diffusion = 0.0;
            params.mv_field_interpolation = 0.85;
            params.sample_address_desync = 1.8;
            params.glitch_cell_size = 0;
            params.glitch_cell_width = 0;
            params.glitch_cell_height = 0;
            params.mv_predictor_desync_every = 7;
            params.mv_predictor_desync_x = 3;
            params.mv_predictor_desync_y = -2;
        }
        "balanced" => {
            params.mv_scale_x = 1.22;
            params.mv_scale_y = 1.12;
            params.mv_jitter = 2;
            params.mv_quant = 3;
            params.reference_lag = 3;
            params.residual_keep = 0.45;
            params.residual_invert_every = 0;
            params.block_remap_every = 9;
            params.block_remap_stride = 5;
            params.channel_shift = 1;
            params.wrap_motion = false;
            params.activity_mode = ActivityMode::Active;
            params.activity_threshold = 8;
            params.activity_softness = 8;
            params.reference_bleed = 0.12;
            params.reference_latch_frames = 4;
            params.reference_slot_count = 4;
            params.reference_slot_shuffle_every = 9;
            params.overlap = 4;
            params.motion_diffusion = 0.2;
            params.mv_field_interpolation = 0.55;
            params.sample_address_desync = 0.9;
            params.glitch_cell_size = 0;
            params.glitch_cell_width = 0;
            params.glitch_cell_height = 0;
            params.mv_predictor_desync_every = 11;
            params.mv_predictor_desync_x = 1;
            params.mv_predictor_desync_y = 1;
        }
        "destroy" => {
            params.mv_scale_x = 1.8;
            params.mv_scale_y = 1.8;
            params.mv_jitter = 4;
            params.mv_quant = 1;
            params.reference_lag = 3;
            params.residual_keep = 0.0;
            params.residual_invert_every = 0;
            params.block_remap_every = 5;
            params.block_remap_stride = 17;
            params.channel_shift = 3;
            params.wrap_motion = true;
            params.activity_mode = ActivityMode::All;
            params.activity_threshold = 0;
            params.activity_softness = 0;
            params.reference_bleed = 0.0;
            params.reference_latch_frames = 1;
            params.reference_slot_count = 1;
            params.reference_slot_shuffle_every = 0;
            params.overlap = 0;
            params.motion_diffusion = 0.0;
            params.mv_field_interpolation = 0.0;
            params.sample_address_desync = 0.0;
            params.glitch_cell_size = 0;
            params.glitch_cell_width = 0;
            params.glitch_cell_height = 0;
            params.mv_predictor_desync_every = 0;
            params.mv_predictor_desync_x = 0;
            params.mv_predictor_desync_y = 0;
        }
        _ => {
            return Err(format!(
                "unknown raw-mosh preset `{name}`; expected clean, subtle, classic, melt, grain, pixel, residue, scan, drift, bank, plane, vector, entropy, coeff, codebook, unstable, balanced, or destroy"
            ));
        }
    }

    Ok(())
}

fn control_amount(master: f32, channel: f32) -> f32 {
    if master.is_finite() && channel.is_finite() {
        (master * channel).clamp(0.0, RAW_MOSH_COMBINED_CONTROL_MAX)
    } else {
        0.0
    }
}

fn scale_i16(value: i16, amount: f32) -> i16 {
    (value as f32 * amount)
        .round()
        .clamp(i16::MIN as f32, i16::MAX as f32) as i16
}

fn scale_i8(value: i8, amount: f32) -> i8 {
    (value as f32 * amount)
        .round()
        .clamp(i8::MIN as f32, i8::MAX as f32) as i8
}

fn scale_u8(value: u8, amount: f32) -> u8 {
    (value as f32 * amount)
        .round()
        .clamp(u8::MIN as f32, u8::MAX as f32) as u8
}

fn scale_i32(value: i32, amount: f32) -> i32 {
    (value as f32 * amount)
        .round()
        .clamp(i32::MIN as f32, i32::MAX as f32) as i32
}

fn scale_u64(value: u64, amount: f32) -> u64 {
    (value as f32 * amount).round().max(0.0) as u64
}

fn scale_usize(value: usize, amount: f32) -> usize {
    (value as f32 * amount).round().max(0.0) as usize
}

fn scale_event_interval(interval: u64, amount: f32) -> u64 {
    if interval == 0 {
        return 0;
    }
    if amount <= 0.0 {
        return 0;
    }
    (interval as f32 / amount).round().max(1.0) as u64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn raw_mosh_preset_switch_does_not_inherit_previous_state() {
        let mut config = MoshCodecConfig::new(64, 48);
        let mut params = MoshGlitchParams::default();
        let mut bitstream = MoshBitstreamParams::default();

        load_raw_mosh_preset("residue", &mut config, &mut params, &mut bitstream).unwrap();
        load_raw_mosh_preset("melt", &mut config, &mut params, &mut bitstream).unwrap();

        assert_eq!(config.block_size, 16);
        assert_eq!(config.search_radius, 8);
        assert_eq!(config.search_step, 4);
        assert_eq!(params.residual_address_jitter, 0);
        assert_eq!(params.residual_channel_shift, 0);
        assert!(!bitstream.enabled);
        assert!(!bitstream.has_mutations());
    }

    #[test]
    fn public_raw_mosh_preset_loader_exposes_custom_codec_presets() {
        let mut config = MoshCodecConfig::new(64, 48);
        let mut params = MoshGlitchParams::default();
        let mut bitstream = MoshBitstreamParams::default();

        load_raw_mosh_preset("codebook", &mut config, &mut params, &mut bitstream).unwrap();

        assert_eq!(config.block_size, 16);
        assert_eq!(params.residual_keep, 1.0);
        assert!(bitstream.enabled);
        assert_eq!(bitstream.codebook_replace_every, 9);
    }

    #[test]
    fn raw_mosh_controls_can_fade_preset_to_neutral() {
        let mut config = MoshCodecConfig::new(64, 48);
        let mut params = MoshGlitchParams::default();
        let mut bitstream = MoshBitstreamParams::default();

        load_raw_mosh_preset("coeff-hard", &mut config, &mut params, &mut bitstream).unwrap();
        apply_raw_mosh_controls(
            &mut params,
            &mut bitstream,
            RawMoshControls {
                intensity: 0.0,
                ..RawMoshControls::default()
            },
        );

        assert_eq!(params.reference_lag, 1);
        assert_eq!(params.residual_keep, 1.0);
        assert_eq!(params.mv_jitter, 0);
        assert!(!bitstream.enabled);
        assert!(!bitstream.has_mutations());
    }

    #[test]
    fn raw_mosh_parameter_schema_filters_by_preset_group() {
        let params = raw_mosh_parameter_infos_for_preset("codebook").unwrap();

        assert!(
            params
                .iter()
                .any(|parameter| parameter.id == "codebook_replace_every")
        );
        assert!(params.iter().any(|parameter| parameter.id == "block_size"));
        assert!(!params.iter().any(|parameter| parameter.id == "mv_scale"));
        assert!(params.iter().any(|parameter| parameter.is_realtime()));
        assert!(
            params
                .iter()
                .any(|parameter| parameter.requires_codec_rebuild())
        );
    }

    #[test]
    fn raw_mosh_parameter_schema_exposes_hybrid_presets_as_all_groups() {
        let params = raw_mosh_parameter_infos_for_preset("unstable").unwrap();

        assert!(params.iter().any(|parameter| parameter.id == "mv_scale"));
        assert!(
            params
                .iter()
                .any(|parameter| parameter.id == "reference_lag")
        );
        assert!(
            params
                .iter()
                .any(|parameter| parameter.id == "residual_keep")
        );
        assert!(
            params
                .iter()
                .any(|parameter| parameter.id == "entropy_slip_every")
        );
        assert!(params.iter().any(|parameter| parameter.id == "block_size"));
    }

    #[test]
    fn raw_mosh_parameter_setter_updates_direct_fields() {
        let mut config = MoshCodecConfig::new(64, 48);
        let mut params = MoshGlitchParams::default();
        let mut bitstream = MoshBitstreamParams::default();

        set_raw_mosh_parameter(&mut config, &mut params, &mut bitstream, "mv_scale", 1.5).unwrap();
        set_raw_mosh_parameter(
            &mut config,
            &mut params,
            &mut bitstream,
            "codebook_replace_every",
            4.0,
        )
        .unwrap();
        set_raw_mosh_parameter(&mut config, &mut params, &mut bitstream, "block_size", 24.0)
            .unwrap();

        assert_eq!(params.mv_scale_x, 1.5);
        assert_eq!(params.mv_scale_y, 1.5);
        assert_eq!(bitstream.codebook_replace_every, 4);
        assert!(bitstream.enabled);
        assert_eq!(config.block_size, 24);
        assert!(
            set_raw_mosh_parameter(&mut config, &mut params, &mut bitstream, "unknown", 1.0,)
                .is_err()
        );
    }

    #[test]
    fn mosh_engine_processes_rgba8_and_preserves_alpha() {
        let mut engine = MoshEngine::new(2, 2).unwrap();
        let input = vec![
            10, 20, 30, 1, 40, 50, 60, 2, 70, 80, 90, 3, 100, 110, 120, 4,
        ];
        let mut output = vec![0; input.len()];

        engine.process_rgba8(&input, &mut output).unwrap();

        assert_eq!(output, input);
        assert_eq!(engine.stats().keyframes, 1);
    }

    #[test]
    fn mosh_engine_updates_controls_parameters_and_reset() {
        let mut engine = MoshEngine::with_backend(MoshEngineBackend::RawMoshV1, 64, 48).unwrap();

        assert_eq!(engine.backend(), MoshEngineBackend::RawMoshV1);
        engine.set_preset("codebook").unwrap();
        engine.set_parameter("codebook_replace_every", 4.0).unwrap();
        engine.set_controls(RawMoshControls {
            intensity: 2.0,
            motion: -1.0,
            residual: 0.5,
            temporal: f32::NAN,
            bitstream: 0.25,
        });

        assert!(engine.bitstream_params().enabled);
        assert_eq!(engine.bitstream_params().codebook_replace_every, 4);
        assert_eq!(engine.controls().intensity, 2.0);
        assert_eq!(engine.controls().motion, 0.0);
        assert_eq!(engine.controls().temporal, 0.0);
        assert_eq!(engine.controls().bitstream, 0.25);

        engine.reset_controls();
        assert_eq!(engine.controls(), RawMoshControls::default());

        let first = vec![0_u8; 64 * 48 * RAW_RGB_CHANNELS];
        let second = vec![16_u8; 64 * 48 * RAW_RGB_CHANNELS];
        let mut output = vec![0_u8; first.len()];
        engine.process_rgb24(&first, &mut output).unwrap();
        engine.process_rgb24(&second, &mut output).unwrap();
        engine.reset_glitch();
        engine.process_rgb24(&second, &mut output).unwrap();

        assert_eq!(engine.stats().keyframes, 2);
    }

    #[test]
    fn raw_mosh_controls_support_authored_range_overdrive() {
        let mut config = MoshCodecConfig::new(64, 48);
        let mut params = MoshGlitchParams::default();
        let mut bitstream = MoshBitstreamParams::default();

        load_raw_mosh_preset("melt", &mut config, &mut params, &mut bitstream).unwrap();
        apply_raw_mosh_controls(
            &mut params,
            &mut bitstream,
            RawMoshControls {
                intensity: 2.0,
                ..RawMoshControls::default()
            },
        );

        assert_eq!(params.reference_lag, 15);
        assert_eq!(params.mv_jitter, 2);
    }

    #[test]
    fn scanline_controls_saturate_small_integer_fields_during_overdrive() {
        let mut params = ScanlineGlitchParams {
            quant_offset: 100,
            chroma_xor_mask: 200,
            luma_run_delta: -100,
            ..ScanlineGlitchParams::default()
        };

        apply_scanline_signal_controls(
            &mut params,
            RawMoshControls {
                intensity: 2.0,
                ..RawMoshControls::default()
            },
        );

        assert_eq!(params.quant_offset, i8::MAX);
        assert_eq!(params.chroma_xor_mask, u8::MAX);
        assert_eq!(params.luma_run_delta, i8::MIN);
    }

    #[test]
    fn c_abi_mosh_engine_processes_rgb24() {
        assert_eq!(datamosh_mosh_engine_backend_count(), 3);
        assert_eq!(
            datamosh_mosh_engine_default_backend(),
            DATAMOSH_BACKEND_RAW_MOSH_V1
        );
        assert_eq!(
            unsafe {
                CStr::from_ptr(datamosh_mosh_engine_backend_name(
                    DATAMOSH_BACKEND_SCANLINE_SIGNAL_V1,
                ))
            }
            .to_str()
            .unwrap(),
            "scanline_signal_v1"
        );

        let engine = datamosh_mosh_engine_new_with_backend(DATAMOSH_BACKEND_RAW_MOSH_V1, 2, 2);
        assert!(!engine.is_null());
        assert_eq!(
            datamosh_mosh_engine_backend(engine),
            DATAMOSH_BACKEND_RAW_MOSH_V1
        );
        assert!(datamosh_mosh_engine_new_with_backend(999, 2, 2).is_null());

        let preset = std::ffi::CString::new("classic").unwrap();
        assert_eq!(
            datamosh_mosh_engine_set_preset(engine, preset.as_ptr()),
            DATAMOSH_STATUS_OK
        );

        let parameter = std::ffi::CString::new("residual_keep").unwrap();
        assert_eq!(
            datamosh_mosh_engine_set_parameter(engine, parameter.as_ptr(), 0.5),
            DATAMOSH_STATUS_OK
        );
        assert_eq!(
            datamosh_mosh_engine_set_controls(engine, 1.0, 1.0, 1.0, 1.0, 1.0),
            DATAMOSH_STATUS_OK
        );

        let input = vec![10_u8; 2 * 2 * RAW_RGB_CHANNELS];
        let mut output = vec![0_u8; input.len()];
        assert_eq!(
            datamosh_mosh_engine_process_rgb24(
                engine,
                input.as_ptr(),
                input.len(),
                output.as_mut_ptr(),
                output.len(),
            ),
            DATAMOSH_STATUS_OK
        );
        assert_eq!(output, input);

        assert_eq!(
            datamosh_mosh_engine_reset_glitch(engine),
            DATAMOSH_STATUS_OK
        );
        assert_eq!(
            datamosh_mosh_engine_process_rgb24(
                engine,
                input.as_ptr(),
                input.len().saturating_sub(1),
                output.as_mut_ptr(),
                output.len(),
            ),
            DATAMOSH_STATUS_INVALID_ARGUMENT
        );

        datamosh_mosh_engine_free(engine);
    }

    #[test]
    fn scanline_signal_backend_processes_existing_pattern_names() {
        let mut engine =
            MoshEngine::with_backend(MoshEngineBackend::ScanlineSignalV1, 32, 16).unwrap();
        assert_eq!(engine.backend(), MoshEngineBackend::ScanlineSignalV1);
        assert!(engine.scanline_config().is_some());
        engine.set_preset("plane").unwrap();
        engine.set_parameter("phase_offset", 2.0).unwrap();

        let mut input = vec![0_u8; 32 * 16 * RAW_RGB_CHANNELS];
        for (index, pixel) in input.chunks_exact_mut(RAW_RGB_CHANNELS).enumerate() {
            pixel[0] = (index & 0xff) as u8;
            pixel[1] = ((index * 3) & 0xff) as u8;
            pixel[2] = ((index * 7) & 0xff) as u8;
        }
        let mut output = vec![0_u8; input.len()];
        engine.process_rgb24(&input, &mut output).unwrap();

        assert_ne!(output, input);
        assert_eq!(engine.stats().frames_in, 1);
        assert_eq!(engine.stats().keyframes, 1);
    }

    #[test]
    fn dct_backend_honors_master_bypass_and_intra_stats() {
        let width = 32;
        let height = 16;
        let mut engine =
            MoshEngine::with_backend(MoshEngineBackend::DctTransformV1, width, height).unwrap();
        engine.set_preset("composite").unwrap();
        engine.set_controls(RawMoshControls {
            intensity: 0.0,
            ..RawMoshControls::default()
        });

        let mut input = vec![0_u8; width * height * RAW_RGB_CHANNELS];
        for (index, pixel) in input.chunks_exact_mut(RAW_RGB_CHANNELS).enumerate() {
            pixel[0] = (index & 0xff) as u8;
            pixel[1] = ((index * 3) & 0xff) as u8;
            pixel[2] = ((index * 7) & 0xff) as u8;
        }
        let mut bypass = vec![0_u8; input.len()];
        engine.process_rgb24(&input, &mut bypass).unwrap();

        let mut clean =
            MoshEngine::with_backend(MoshEngineBackend::DctTransformV1, width, height).unwrap();
        let mut clean_output = vec![0_u8; input.len()];
        clean.process_rgb24(&input, &mut clean_output).unwrap();

        assert_eq!(bypass, clean_output);
        assert_eq!(engine.stats().frames_in, 1);
        assert_eq!(engine.stats().keyframes, 1);
        assert_eq!(engine.stats().predicted_frames, 0);
        assert_eq!(engine.stats().blocks_encoded, 12);

        engine.process_rgb24(&input, &mut bypass).unwrap();
        assert_eq!(engine.stats().keyframes, 2);
        assert_eq!(engine.stats().predicted_frames, 0);
    }

    #[test]
    fn dct_entropy_preset_is_disabled_by_master_zero() {
        let width = 64;
        let height = 32;
        let mut engine =
            MoshEngine::with_backend(MoshEngineBackend::DctTransformV1, width, height).unwrap();
        engine.set_preset("desync").unwrap();
        engine.set_controls(RawMoshControls {
            intensity: 0.0,
            ..RawMoshControls::default()
        });

        let input = vec![96_u8; width * height * RAW_RGB_CHANNELS];
        let mut output = vec![0_u8; input.len()];
        engine.process_rgb24(&input, &mut output).unwrap();

        let mut clean =
            MoshEngine::with_backend(MoshEngineBackend::DctTransformV1, width, height).unwrap();
        let mut clean_output = vec![0_u8; input.len()];
        clean.process_rgb24(&input, &mut clean_output).unwrap();
        assert_eq!(output, clean_output);
    }

    #[test]
    fn dct_quality_parameter_rebuilds_the_codec() {
        let mut engine =
            MoshEngine::with_backend(MoshEngineBackend::DctTransformV1, 32, 16).unwrap();
        assert_eq!(engine.dct_config().unwrap().quality, 50);

        engine.set_parameter("quality", 85.0).unwrap();
        assert_eq!(engine.dct_config().unwrap().quality, 85);

        engine.set_parameter("quality", 0.0).unwrap();
        assert_eq!(engine.dct_config().unwrap().quality, 1);
    }

    #[test]
    fn scanline_signal_presets_are_available_and_visually_distinct() {
        use std::collections::HashSet;

        const PRESETS: &[&str] = &[
            "clean",
            "subtle",
            "timebase-tear",
            "clock-skew",
            "vertical-roll",
            "sync-dropout",
            "phase-walk",
            "carrier-slip",
            "chroma-grid",
            "chroma-sequence",
            "burst-seed-loss",
            "carrier-xor",
            "predictor-ghost",
            "line-crosstalk",
            "rle-runaway",
            "packet-length",
            "plane-crosswire",
            "quantizer-pump",
            "history-weave",
            "composite-collapse",
        ];

        let width = 48;
        let height = 24;
        let mut checksums = HashSet::new();
        for preset in PRESETS {
            let mut engine =
                MoshEngine::with_backend(MoshEngineBackend::ScanlineSignalV1, width, height)
                    .unwrap();
            engine.set_preset(preset).unwrap();
            let mut input = vec![0_u8; width * height * RAW_RGB_CHANNELS];
            let mut output = vec![0_u8; input.len()];
            for frame in 0..4 {
                for y in 0..height {
                    for x in 0..width {
                        let offset = (y * width + x) * RAW_RGB_CHANNELS;
                        input[offset] = ((x * 5 + frame * 11) & 0xff) as u8;
                        input[offset + 1] = ((y * 9 + frame * 7) & 0xff) as u8;
                        input[offset + 2] = (((x ^ y) * 13 + frame * 17) & 0xff) as u8;
                    }
                }
                engine.process_rgb24(&input, &mut output).unwrap();
            }
            let checksum = output
                .iter()
                .enumerate()
                .fold(0_u64, |hash, (index, value)| {
                    hash.wrapping_mul(1_099_511_628_211)
                        .wrapping_add(*value as u64 + index as u64)
                });
            checksums.insert(checksum);
        }

        assert!(
            checksums.len() >= 17,
            "expected at least 17 distinct SCN0 preset outputs, got {}",
            checksums.len()
        );
    }
}
