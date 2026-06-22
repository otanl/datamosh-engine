//! WVT0 - multiresolution wavelet packet codec.
//!
//! Frames are converted with the reversible YCoCg-R transform, decomposed with an
//! integer Haar lifting transform, quantized by scale, and stored as independently
//! addressable subband packets. Glitches operate on those packets or on the inverse
//! lifting state; no RGB post-processing is involved.

use std::collections::VecDeque;
use std::io;

use rayon::prelude::*;

use crate::mosh_codec::codec_thread_pool;
use crate::{RAW_MOSH_COMBINED_CONTROL_MAX, RawMoshControls};

const CHANNELS: usize = 3;
const PARALLEL_TRANSFORM_PIXELS: usize = 65_536;

type ForwardRgbRows<'a> = (((&'a [u8], &'a mut [i32]), &'a mut [i32]), &'a mut [i32]);
type InverseRgbRows<'a> = (((&'a mut [u8], &'a [i32]), &'a [i32]), &'a [i32]);

#[derive(Debug, Clone, Copy)]
pub struct WaveletCodecConfig {
    pub width: usize,
    pub height: usize,
    pub levels: usize,
    pub quality: u8,
    pub history_len: usize,
}

impl WaveletCodecConfig {
    pub fn new(width: usize, height: usize) -> Self {
        let mut config = Self {
            width,
            height,
            ..Self::default()
        };
        config.levels = config.levels.min(config.max_levels().max(1));
        config
    }

    pub fn frame_len(&self) -> Option<usize> {
        self.width.checked_mul(self.height)?.checked_mul(CHANNELS)
    }

    pub fn max_levels(&self) -> usize {
        let mut width = self.width;
        let mut height = self.height;
        let mut levels = 0;
        while width > 1 && height > 1 {
            levels += 1;
            width = width.div_ceil(2);
            height = height.div_ceil(2);
        }
        levels
    }

    fn validate(&self) -> io::Result<()> {
        if self.width == 0 || self.height == 0 {
            return Err(invalid_input("width and height must be greater than zero"));
        }
        if self.levels == 0 || self.levels > self.max_levels() {
            return Err(invalid_input(format!(
                "levels must be in 1..={} for this resolution",
                self.max_levels()
            )));
        }
        if self.quality == 0 || self.quality > 100 {
            return Err(invalid_input("quality must be in 1..=100"));
        }
        if self.history_len == 0 {
            return Err(invalid_input("history_len must be greater than zero"));
        }
        self.frame_len()
            .ok_or_else(|| invalid_input("frame dimensions overflow addressable memory"))?;
        Ok(())
    }
}

impl Default for WaveletCodecConfig {
    fn default() -> Self {
        Self {
            width: 0,
            height: 0,
            levels: 3,
            quality: 82,
            history_len: 12,
        }
    }
}

#[repr(u8)]
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum WaveletOrientation {
    Approximation = 0,
    Horizontal = 1,
    Vertical = 2,
    Diagonal = 3,
}

impl WaveletOrientation {
    fn rotated(self, amount: i8) -> Self {
        if self == Self::Approximation {
            return self;
        }
        let index = (self as i8 - 1 + amount).rem_euclid(3);
        match index {
            0 => Self::Horizontal,
            1 => Self::Vertical,
            _ => Self::Diagonal,
        }
    }
}

#[derive(Debug, Clone)]
pub struct WaveletBand {
    pub channel: u8,
    pub level: u8,
    pub orientation: WaveletOrientation,
    pub width: usize,
    pub height: usize,
    pub quant_step: i32,
    pub coefficients: Vec<i16>,
}

#[derive(Debug, Clone)]
pub struct WaveletPacket {
    pub width: usize,
    pub height: usize,
    pub levels: usize,
    pub bands: Vec<WaveletBand>,
    pub estimated_bytes: usize,
}

#[derive(Debug, Clone, Copy)]
pub struct WaveletGlitchParams {
    pub packet_shift: i16,
    pub packet_shift_every: u64,
    pub orientation_rotate: i8,
    pub orientation_rotate_every: u64,
    pub level_fold: i8,
    pub level_fold_every: u64,
    pub channel_route: i8,
    pub channel_route_every: u64,
    pub packet_loss_every: u64,
    pub packet_loss_conceal: bool,
    pub bitplane_clear: u8,
    pub bitplane_clear_every: u64,
    pub bitplane_xor: u8,
    pub bitplane_xor_every: u64,
    pub sign_flip_every: u64,
    pub history_lag: usize,
    pub history_band_every: u64,
    pub lowpass_history_lag: usize,
    pub lifting_bias: i16,
    pub lifting_bias_every: u64,
}

impl Default for WaveletGlitchParams {
    fn default() -> Self {
        Self {
            packet_shift: 0,
            packet_shift_every: 0,
            orientation_rotate: 0,
            orientation_rotate_every: 0,
            level_fold: 0,
            level_fold_every: 0,
            channel_route: 0,
            channel_route_every: 0,
            packet_loss_every: 0,
            packet_loss_conceal: true,
            bitplane_clear: 0,
            bitplane_clear_every: 0,
            bitplane_xor: 0,
            bitplane_xor_every: 0,
            sign_flip_every: 0,
            history_lag: 1,
            history_band_every: 0,
            lowpass_history_lag: 0,
            lifting_bias: 0,
            lifting_bias_every: 0,
        }
    }
}

impl WaveletGlitchParams {
    pub fn has_mutations(&self) -> bool {
        (self.packet_shift != 0 && self.packet_shift_every != 0)
            || (self.orientation_rotate != 0 && self.orientation_rotate_every != 0)
            || (self.level_fold != 0 && self.level_fold_every != 0)
            || (self.channel_route != 0 && self.channel_route_every != 0)
            || self.packet_loss_every != 0
            || (self.bitplane_clear != 0 && self.bitplane_clear_every != 0)
            || (self.bitplane_xor != 0 && self.bitplane_xor_every != 0)
            || self.sign_flip_every != 0
            || self.history_band_every != 0
            || self.lowpass_history_lag != 0
            || (self.lifting_bias != 0 && self.lifting_bias_every != 0)
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct WaveletCodecStats {
    pub frames_in: u64,
    pub bands_encoded: u64,
    pub coefficients_encoded: u64,
    pub raw_bytes: u64,
    pub estimated_bytes: u64,
    pub damaged_bands: u64,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct WaveletMutationStats {
    pub packets_shifted: u64,
    pub orientations_rotated: u64,
    pub levels_folded: u64,
    pub channels_routed: u64,
    pub packets_lost: u64,
    pub packets_concealed: u64,
    pub history_packets_used: u64,
    pub bitplanes_cleared: u64,
    pub bitplanes_xored: u64,
    pub signs_flipped: u64,
    pub lifting_samples_biased: u64,
}

pub struct WaveletCodec {
    config: WaveletCodecConfig,
    stats: WaveletCodecStats,
    history: VecDeque<WaveletPacket>,
    recycled_packet: Option<WaveletPacket>,
    rate_estimation: bool,
    frame_index: u64,
    planes: [Vec<i32>; CHANNELS],
    transform_buffers: [Vec<i32>; CHANNELS],
}

impl WaveletCodec {
    pub fn new(config: WaveletCodecConfig) -> io::Result<Self> {
        config.validate()?;
        let pixels = config.width * config.height;
        Ok(Self {
            config,
            stats: WaveletCodecStats::default(),
            history: VecDeque::with_capacity(config.history_len),
            recycled_packet: None,
            rate_estimation: false,
            frame_index: 0,
            planes: std::array::from_fn(|_| vec![0; pixels]),
            transform_buffers: std::array::from_fn(|_| vec![0; pixels]),
        })
    }

    pub fn config(&self) -> &WaveletCodecConfig {
        &self.config
    }

    pub fn stats(&self) -> &WaveletCodecStats {
        &self.stats
    }

    pub fn set_rate_estimation(&mut self, enabled: bool) {
        self.rate_estimation = enabled;
    }

    pub fn reset_glitch_state(&mut self) {
        if self.recycled_packet.is_none() {
            self.recycled_packet = self.history.pop_front();
        }
        self.history.clear();
        self.frame_index = 0;
    }

    pub fn encode_rgb_frame(&mut self, input: &[u8]) -> io::Result<WaveletPacket> {
        self.encode_rgb_frame_reusing(input, None)
    }

    fn encode_rgb_frame_reusing(
        &mut self,
        input: &[u8],
        reusable: Option<WaveletPacket>,
    ) -> io::Result<WaveletPacket> {
        self.validate_frame(input, "input")?;
        forward_rgb_haar(
            input,
            &mut self.planes,
            &mut self.transform_buffers,
            self.config.width,
            self.config.height,
            self.config.levels,
        );

        let mut packet = reusable
            .filter(|packet| packet_matches_config(packet, self.config))
            .unwrap_or_else(|| new_wavelet_packet(self.config));
        let encoded_band_bytes = if let Some(pool) = codec_thread_pool() {
            pool.install(|| {
                packet
                    .bands
                    .par_iter_mut()
                    .map(|band| {
                        fill_band_coefficients(
                            &self.planes[band.channel as usize],
                            self.config.width,
                            self.config.height,
                            band,
                            self.rate_estimation,
                        )
                    })
                    .sum::<usize>()
            })
        } else {
            packet
                .bands
                .iter_mut()
                .map(|band| {
                    fill_band_coefficients(
                        &self.planes[band.channel as usize],
                        self.config.width,
                        self.config.height,
                        band,
                        self.rate_estimation,
                    )
                })
                .sum::<usize>()
        };
        packet.estimated_bytes = if self.rate_estimation {
            16 + packet.bands.len() * 16 + encoded_band_bytes
        } else {
            0
        };
        Ok(packet)
    }

    pub fn decode_packet(
        &mut self,
        packet: &WaveletPacket,
        params: &WaveletGlitchParams,
        output: &mut [u8],
    ) -> io::Result<WaveletMutationStats> {
        self.validate_frame(output, "output")?;
        if packet.width != self.config.width
            || packet.height != self.config.height
            || packet.levels != self.config.levels
        {
            return Err(invalid_input("WVT0 packet configuration mismatch"));
        }

        let history = &self.history;
        let stats = if let Some(pool) = codec_thread_pool() {
            pool.install(|| {
                self.planes
                    .par_iter_mut()
                    .enumerate()
                    .map(|(channel, plane)| {
                        decode_channel_bands(
                            packet,
                            history,
                            channel as u8,
                            plane,
                            self.config,
                            params,
                            self.frame_index,
                        )
                    })
                    .reduce(WaveletMutationStats::default, WaveletMutationStats::merge)
            })
        } else {
            self.planes
                .iter_mut()
                .enumerate()
                .map(|(channel, plane)| {
                    decode_channel_bands(
                        packet,
                        history,
                        channel as u8,
                        plane,
                        self.config,
                        params,
                        self.frame_index,
                    )
                })
                .fold(WaveletMutationStats::default(), WaveletMutationStats::merge)
        };

        inverse_haar_to_rgb(
            &mut self.planes,
            &mut self.transform_buffers,
            self.config.width,
            self.config.height,
            self.config.levels,
            output,
        );
        Ok(stats)
    }

    pub fn process_rgb_frame(
        &mut self,
        input: &[u8],
        params: &WaveletGlitchParams,
        output: &mut [u8],
    ) -> io::Result<WaveletMutationStats> {
        let reusable = self.recycled_packet.take();
        let packet = self.encode_rgb_frame_reusing(input, reusable)?;
        let mutation = self.decode_packet(&packet, params, output)?;

        self.stats.frames_in += 1;
        self.stats.bands_encoded += packet.bands.len() as u64;
        self.stats.coefficients_encoded +=
            (self.config.width * self.config.height * CHANNELS) as u64;
        self.stats.raw_bytes += input.len() as u64;
        self.stats.estimated_bytes += packet.estimated_bytes as u64;
        self.stats.damaged_bands += mutation.damaged_bands();

        self.history.push_front(packet);
        if self.history.len() > self.config.history_len {
            self.recycled_packet = self.history.pop_back();
        }
        self.frame_index = self.frame_index.wrapping_add(1);
        Ok(mutation)
    }

    fn validate_frame(&self, frame: &[u8], name: &str) -> io::Result<()> {
        let expected = self
            .config
            .frame_len()
            .ok_or_else(|| invalid_input("frame dimensions overflow addressable memory"))?;
        if frame.len() != expected {
            return Err(invalid_input(format!(
                "{name} frame must be {expected} bytes of rgb24"
            )));
        }
        Ok(())
    }
}

fn history_packet(history: &VecDeque<WaveletPacket>, lag: usize) -> Option<&WaveletPacket> {
    history.get(lag.saturating_sub(1))
}

impl WaveletMutationStats {
    pub fn damaged_bands(self) -> u64 {
        self.packets_shifted
            + self.orientations_rotated
            + self.levels_folded
            + self.channels_routed
            + self.packets_lost
            + self.history_packets_used
    }

    fn merge(self, other: Self) -> Self {
        Self {
            packets_shifted: self.packets_shifted + other.packets_shifted,
            orientations_rotated: self.orientations_rotated + other.orientations_rotated,
            levels_folded: self.levels_folded + other.levels_folded,
            channels_routed: self.channels_routed + other.channels_routed,
            packets_lost: self.packets_lost + other.packets_lost,
            packets_concealed: self.packets_concealed + other.packets_concealed,
            history_packets_used: self.history_packets_used + other.history_packets_used,
            bitplanes_cleared: self.bitplanes_cleared + other.bitplanes_cleared,
            bitplanes_xored: self.bitplanes_xored + other.bitplanes_xored,
            signs_flipped: self.signs_flipped + other.signs_flipped,
            lifting_samples_biased: self.lifting_samples_biased + other.lifting_samples_biased,
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn decode_channel_bands(
    packet: &WaveletPacket,
    history: &VecDeque<WaveletPacket>,
    channel: u8,
    plane: &mut [i32],
    config: WaveletCodecConfig,
    params: &WaveletGlitchParams,
    frame_index: u64,
) -> WaveletMutationStats {
    plane.fill(0);
    let mut stats = WaveletMutationStats::default();

    for (band_index, destination) in packet
        .bands
        .iter()
        .enumerate()
        .filter(|(_, band)| band.channel == channel)
    {
        let ordinal = band_index as u64 + frame_index;
        let mut source = destination;

        if destination.orientation == WaveletOrientation::Approximation
            && params.lowpass_history_lag != 0
        {
            if let Some(history_packet) = history_packet(history, params.lowpass_history_lag) {
                if let Some(history_band) = find_matching_band(history_packet, destination) {
                    source = history_band;
                    stats.history_packets_used += 1;
                }
            }
        } else if event(ordinal, params.history_band_every) {
            if let Some(history_packet) = history_packet(history, params.history_lag) {
                if let Some(history_band) = find_matching_band(history_packet, destination) {
                    source = history_band;
                    stats.history_packets_used += 1;
                }
            }
        }

        if event(ordinal, params.orientation_rotate_every) && params.orientation_rotate != 0 {
            if let Some(remapped) = find_band(
                packet,
                destination.channel,
                destination.level,
                destination.orientation.rotated(params.orientation_rotate),
            ) {
                source = remapped;
                stats.orientations_rotated += 1;
            }
        }

        if event(ordinal, params.level_fold_every) && params.level_fold != 0 {
            let level = wrap_level(
                destination.level as i16 + params.level_fold as i16,
                config.levels,
            );
            if let Some(remapped) =
                find_band(packet, destination.channel, level, destination.orientation)
            {
                source = remapped;
                stats.levels_folded += 1;
            }
        }

        if event(ordinal, params.channel_route_every) && params.channel_route != 0 {
            let source_channel =
                (destination.channel as i16 + params.channel_route as i16).rem_euclid(3) as u8;
            if let Some(remapped) = find_band(
                packet,
                source_channel,
                destination.level,
                destination.orientation,
            ) {
                source = remapped;
                stats.channels_routed += 1;
            }
        }

        if event(ordinal, params.packet_shift_every) && params.packet_shift != 0 {
            let source_index = (band_index as i64 + params.packet_shift as i64)
                .rem_euclid(packet.bands.len() as i64) as usize;
            source = &packet.bands[source_index];
            stats.packets_shifted += 1;
        }

        if event(ordinal, params.packet_loss_every) {
            stats.packets_lost += 1;
            let concealed = if params.packet_loss_conceal {
                history_packet(history, params.history_lag)
                    .and_then(|history_packet| find_matching_band(history_packet, destination))
            } else {
                None
            };
            if let Some(history_band) = concealed {
                source = history_band;
                stats.packets_concealed += 1;
            } else {
                continue;
            }
        }

        write_band_to_plane(
            plane,
            config.width,
            config.height,
            destination,
            source,
            params,
            ordinal,
            &mut stats,
        );
    }
    stats
}

pub fn load_wavelet_preset(name: &str, params: &mut WaveletGlitchParams) -> Result<(), String> {
    *params = WaveletGlitchParams::default();
    match name {
        "clean" => {}
        "subband-slip" => {
            params.packet_shift = 2;
            params.packet_shift_every = 5;
        }
        "orientation-cross" => {
            params.orientation_rotate = 1;
            params.orientation_rotate_every = 3;
        }
        "scale-fold" => {
            params.level_fold = 1;
            params.level_fold_every = 4;
        }
        "bitplane-rain" => {
            params.bitplane_clear = 2;
            params.bitplane_clear_every = 3;
            params.bitplane_xor = 4;
            params.bitplane_xor_every = 11;
        }
        "lowpass-ghost" => {
            params.lowpass_history_lag = 5;
        }
        "temporal-weave" => {
            params.history_lag = 6;
            params.history_band_every = 3;
        }
        "packet-loss" => {
            params.history_lag = 3;
            params.packet_loss_every = 5;
            params.packet_loss_conceal = true;
        }
        "lifting-drift" => {
            params.lifting_bias = 12;
            params.lifting_bias_every = 7;
        }
        "chroma-pyramid" => {
            params.channel_route = 1;
            params.channel_route_every = 4;
            params.level_fold = -1;
            params.level_fold_every = 7;
        }
        "hierarchy-collapse" => {
            params.packet_shift = 5;
            params.packet_shift_every = 4;
            params.orientation_rotate = 1;
            params.orientation_rotate_every = 5;
            params.level_fold = 1;
            params.level_fold_every = 6;
            params.bitplane_clear = 2;
            params.bitplane_clear_every = 3;
            params.history_lag = 4;
            params.history_band_every = 5;
            params.packet_loss_every = 9;
            params.packet_loss_conceal = true;
            params.lifting_bias = 8;
            params.lifting_bias_every = 11;
        }
        _ => {
            return Err(format!(
                "unknown wavelet preset `{name}`; expected clean, subband-slip, orientation-cross, scale-fold, bitplane-rain, lowpass-ghost, temporal-weave, packet-loss, lifting-drift, chroma-pyramid, or hierarchy-collapse"
            ));
        }
    }
    Ok(())
}

pub fn apply_wavelet_controls(params: &mut WaveletGlitchParams, controls: RawMoshControls) {
    let structure = control_amount(controls.intensity, controls.motion);
    let coefficients = control_amount(controls.intensity, controls.residual);
    let temporal = control_amount(controls.intensity, controls.temporal);
    let packets = control_amount(controls.intensity, controls.bitstream);

    params.packet_shift = scale_i16(params.packet_shift, structure);
    params.packet_shift_every = scale_interval(params.packet_shift_every, structure);
    params.orientation_rotate = scale_i8(params.orientation_rotate, structure);
    params.orientation_rotate_every = scale_interval(params.orientation_rotate_every, structure);
    params.level_fold = scale_i8(params.level_fold, structure);
    params.level_fold_every = scale_interval(params.level_fold_every, structure);

    params.bitplane_clear = scale_u8(params.bitplane_clear, coefficients);
    params.bitplane_clear_every = scale_interval(params.bitplane_clear_every, coefficients);
    params.bitplane_xor = scale_u8(params.bitplane_xor, coefficients);
    params.bitplane_xor_every = scale_interval(params.bitplane_xor_every, coefficients);
    params.sign_flip_every = scale_interval(params.sign_flip_every, coefficients);
    params.lifting_bias = scale_i16(params.lifting_bias, coefficients);
    params.lifting_bias_every = scale_interval(params.lifting_bias_every, coefficients);

    params.history_lag = 1 + scale_usize(params.history_lag.saturating_sub(1), temporal);
    params.history_band_every = scale_interval(params.history_band_every, temporal);
    params.lowpass_history_lag = scale_usize(params.lowpass_history_lag, temporal);

    params.channel_route = scale_i8(params.channel_route, packets);
    params.channel_route_every = scale_interval(params.channel_route_every, packets);
    params.packet_loss_every = scale_interval(params.packet_loss_every, packets);
}

pub fn set_wavelet_parameter(
    params: &mut WaveletGlitchParams,
    id: &str,
    value: f32,
) -> Result<(), String> {
    let finite = if value.is_finite() { value } else { 0.0 };
    match id {
        "packet_shift" => params.packet_shift = clamp_i16(finite),
        "packet_shift_every" => params.packet_shift_every = clamp_u64(finite),
        "orientation_rotate" => params.orientation_rotate = clamp_i8(finite),
        "orientation_rotate_every" => params.orientation_rotate_every = clamp_u64(finite),
        "level_fold" => params.level_fold = clamp_i8(finite),
        "level_fold_every" => params.level_fold_every = clamp_u64(finite),
        "channel_route" => params.channel_route = clamp_i8(finite),
        "channel_route_every" => params.channel_route_every = clamp_u64(finite),
        "packet_loss_every" => params.packet_loss_every = clamp_u64(finite),
        "packet_loss_conceal" => params.packet_loss_conceal = finite >= 0.5,
        "bitplane_clear" => params.bitplane_clear = finite.round().clamp(0.0, 30.0) as u8,
        "bitplane_clear_every" => params.bitplane_clear_every = clamp_u64(finite),
        "bitplane_xor" => params.bitplane_xor = finite.round().clamp(0.0, 30.0) as u8,
        "bitplane_xor_every" => params.bitplane_xor_every = clamp_u64(finite),
        "sign_flip_every" => params.sign_flip_every = clamp_u64(finite),
        "history_lag" => params.history_lag = finite.round().max(1.0) as usize,
        "history_band_every" => params.history_band_every = clamp_u64(finite),
        "lowpass_history_lag" => params.lowpass_history_lag = finite.round().max(0.0) as usize,
        "lifting_bias" => params.lifting_bias = clamp_i16(finite),
        "lifting_bias_every" => params.lifting_bias_every = clamp_u64(finite),
        _ => return Err(format!("unknown wavelet parameter `{id}`")),
    }
    Ok(())
}

fn new_wavelet_packet(config: WaveletCodecConfig) -> WaveletPacket {
    let mut bands = Vec::with_capacity(CHANNELS * (1 + config.levels * 3));
    let dimensions = level_dimensions(config.width, config.height, config.levels);
    let (deep_width, deep_height) = dimensions[config.levels];

    for channel in 0..CHANNELS {
        push_empty_band(
            &mut bands,
            config,
            channel,
            config.levels,
            WaveletOrientation::Approximation,
            deep_width,
            deep_height,
        );
        for level in 1..=config.levels {
            let (width, height) = dimensions[level - 1];
            let (low_width, low_height) = dimensions[level];
            let high_width = width / 2;
            let high_height = height / 2;
            push_empty_band(
                &mut bands,
                config,
                channel,
                level,
                WaveletOrientation::Horizontal,
                high_width,
                low_height,
            );
            push_empty_band(
                &mut bands,
                config,
                channel,
                level,
                WaveletOrientation::Vertical,
                low_width,
                high_height,
            );
            push_empty_band(
                &mut bands,
                config,
                channel,
                level,
                WaveletOrientation::Diagonal,
                high_width,
                high_height,
            );
        }
    }

    WaveletPacket {
        width: config.width,
        height: config.height,
        levels: config.levels,
        bands,
        estimated_bytes: 0,
    }
}

fn push_empty_band(
    bands: &mut Vec<WaveletBand>,
    config: WaveletCodecConfig,
    channel: usize,
    level: usize,
    orientation: WaveletOrientation,
    width: usize,
    height: usize,
) {
    bands.push(WaveletBand {
        channel: channel as u8,
        level: level as u8,
        orientation,
        width,
        height,
        quant_step: quant_step(config, channel, level, orientation),
        coefficients: vec![0; width * height],
    });
}

fn packet_matches_config(packet: &WaveletPacket, config: WaveletCodecConfig) -> bool {
    packet.width == config.width
        && packet.height == config.height
        && packet.levels == config.levels
        && packet.bands.len() == CHANNELS * (1 + config.levels * 3)
}

fn fill_band_coefficients(
    plane: &[i32],
    plane_width: usize,
    plane_height: usize,
    band: &mut WaveletBand,
    estimate_rate: bool,
) -> usize {
    let coefficient_count = band.width * band.height;
    band.coefficients.resize(coefficient_count, 0);
    let (x, y) = band_origin(
        plane_width,
        plane_height,
        band.level as usize,
        band.orientation,
    );
    let mut encoded_bytes = 0;
    let mut zero_run = 0_u64;
    let mut destination = 0;
    for row in 0..band.height {
        let source = (y + row) * plane_width + x;
        for column in 0..band.width {
            let coefficient = round_div(plane[source + column], band.quant_step)
                .clamp(i16::MIN as i32, i16::MAX as i32) as i16;
            band.coefficients[destination] = coefficient;
            destination += 1;
            if estimate_rate {
                if coefficient == 0 {
                    zero_run += 1;
                } else {
                    encoded_bytes += varint_len(zero_run);
                    encoded_bytes += varint_len(zigzag_i32(coefficient as i32));
                    zero_run = 0;
                }
            }
        }
    }
    if estimate_rate {
        encoded_bytes + varint_len(zero_run)
    } else {
        0
    }
}

#[allow(clippy::too_many_arguments)]
fn write_band_to_plane(
    plane: &mut [i32],
    plane_width: usize,
    plane_height: usize,
    destination: &WaveletBand,
    source: &WaveletBand,
    params: &WaveletGlitchParams,
    ordinal: u64,
    stats: &mut WaveletMutationStats,
) {
    let (x, y) = band_origin(
        plane_width,
        plane_height,
        destination.level as usize,
        destination.orientation,
    );
    if destination.width == 0 || destination.height == 0 || source.coefficients.is_empty() {
        return;
    }

    let coefficient_mutations = (params.bitplane_clear != 0 && params.bitplane_clear_every != 0)
        || (params.bitplane_xor != 0 && params.bitplane_xor_every != 0)
        || params.sign_flip_every != 0
        || (destination.orientation != WaveletOrientation::Approximation
            && params.lifting_bias != 0
            && params.lifting_bias_every != 0);
    let decode_row = |row: usize, target: &mut [i32]| {
        let mut row_stats = WaveletMutationStats::default();
        let source_row = if source.height == destination.height {
            row
        } else {
            row * source.height / destination.height
        };
        let same_width = source.width == destination.width;
        if !coefficient_mutations {
            for column in 0..destination.width {
                let source_column = if same_width {
                    column
                } else {
                    column * source.width / destination.width
                };
                let source_index = source_row * source.width + source_column;
                target[x + column] = (source.coefficients[source_index] as i32)
                    .saturating_mul(destination.quant_step);
            }
            return row_stats;
        }

        let row_ordinal = ordinal.wrapping_mul(1_000_003) + (row * destination.width) as u64;
        let mut bitplane_clear = EventCursor::new(row_ordinal, params.bitplane_clear_every);
        let mut bitplane_xor = EventCursor::new(row_ordinal, params.bitplane_xor_every);
        let mut sign_flip = EventCursor::new(row_ordinal, params.sign_flip_every);
        let mut lifting_bias = EventCursor::new(row_ordinal, params.lifting_bias_every);
        for column in 0..destination.width {
            let source_column = if same_width {
                column
            } else {
                column * source.width / destination.width
            };
            let source_index = source_row * source.width + source_column;
            let mut value = source.coefficients[source_index] as i32;

            if params.bitplane_clear != 0 && bitplane_clear.hit() {
                value = clear_low_bits(value, params.bitplane_clear);
                row_stats.bitplanes_cleared += 1;
            }
            if params.bitplane_xor != 0 && bitplane_xor.hit() {
                value = xor_bitplane(value, params.bitplane_xor);
                row_stats.bitplanes_xored += 1;
            }
            if sign_flip.hit() {
                value = value.saturating_neg();
                row_stats.signs_flipped += 1;
            }

            let mut reconstructed = value.saturating_mul(destination.quant_step);
            if destination.orientation != WaveletOrientation::Approximation
                && params.lifting_bias != 0
                && lifting_bias.hit()
            {
                reconstructed = reconstructed.saturating_add(params.lifting_bias as i32);
                row_stats.lifting_samples_biased += 1;
            }
            target[x + column] = reconstructed;
        }
        row_stats
    };

    let area = destination.width.saturating_mul(destination.height);
    let row_stats = if area >= 16_384 && codec_thread_pool().is_some() {
        plane
            .par_chunks_mut(plane_width)
            .skip(y)
            .take(destination.height)
            .enumerate()
            .map(|(row, target)| decode_row(row, target))
            .reduce(WaveletMutationStats::default, WaveletMutationStats::merge)
    } else {
        plane
            .chunks_mut(plane_width)
            .skip(y)
            .take(destination.height)
            .enumerate()
            .map(|(row, target)| decode_row(row, target))
            .fold(WaveletMutationStats::default(), WaveletMutationStats::merge)
    };
    *stats = stats.merge(row_stats);
    debug_assert!(y + destination.height <= plane_height);
    debug_assert!(x + destination.width <= plane_width);
}

struct EventCursor {
    every: u64,
    remaining: u64,
}

impl EventCursor {
    fn new(ordinal: u64, every: u64) -> Self {
        let remaining = if every == 0 {
            u64::MAX
        } else {
            (every - ordinal % every) % every
        };
        Self { every, remaining }
    }

    fn hit(&mut self) -> bool {
        if self.every == 0 {
            return false;
        }
        if self.remaining == 0 {
            self.remaining = self.every - 1;
            true
        } else {
            self.remaining -= 1;
            false
        }
    }
}

fn find_matching_band<'a>(
    packet: &'a WaveletPacket,
    band: &WaveletBand,
) -> Option<&'a WaveletBand> {
    find_band(packet, band.channel, band.level, band.orientation)
}

fn find_band(
    packet: &WaveletPacket,
    channel: u8,
    level: u8,
    orientation: WaveletOrientation,
) -> Option<&WaveletBand> {
    packet.bands.iter().find(|band| {
        band.channel == channel && band.level == level && band.orientation == orientation
    })
}

fn band_origin(
    width: usize,
    height: usize,
    level: usize,
    orientation: WaveletOrientation,
) -> (usize, usize) {
    if orientation == WaveletOrientation::Approximation {
        return (0, 0);
    }
    let dimensions = level_dimensions(width, height, level);
    let (low_width, low_height) = dimensions[level];
    match orientation {
        WaveletOrientation::Approximation => (0, 0),
        WaveletOrientation::Horizontal => (low_width, 0),
        WaveletOrientation::Vertical => (0, low_height),
        WaveletOrientation::Diagonal => (low_width, low_height),
    }
}

fn level_dimensions(width: usize, height: usize, levels: usize) -> Vec<(usize, usize)> {
    let mut dimensions = Vec::with_capacity(levels + 1);
    let (mut width, mut height) = (width, height);
    dimensions.push((width, height));
    for _ in 0..levels {
        width = width.div_ceil(2);
        height = height.div_ceil(2);
        dimensions.push((width, height));
    }
    dimensions
}

fn quant_step(
    config: WaveletCodecConfig,
    channel: usize,
    level: usize,
    orientation: WaveletOrientation,
) -> i32 {
    if config.quality == 100 {
        return 1;
    }
    let quality_loss = 101 - config.quality as i32;
    let base = 1 + quality_loss * quality_loss / 180;
    let chroma = if channel == 0 { 1 } else { 2 };
    if orientation == WaveletOrientation::Approximation {
        (base / 3).max(1) * chroma
    } else {
        let fine_scale = (config.levels - level + 1) as i32;
        (base * fine_scale * chroma).max(1)
    }
}

fn forward_rgb_haar(
    input: &[u8],
    planes: &mut [Vec<i32>; CHANNELS],
    buffers: &mut [Vec<i32>; CHANNELS],
    width: usize,
    height: usize,
    levels: usize,
) {
    let [y_buffer, co_buffer, cg_buffer] = buffers;
    let input_stride = width * CHANNELS;
    let encode_row = |(((rgb, y_row), co_row), cg_row): ForwardRgbRows<'_>| {
        forward_rgb_row(
            rgb,
            &mut y_row[..width],
            &mut co_row[..width],
            &mut cg_row[..width],
        );
    };
    if let Some(pool) = codec_thread_pool() {
        pool.install(|| {
            input
                .par_chunks(input_stride)
                .zip(y_buffer.par_chunks_mut(width))
                .zip(co_buffer.par_chunks_mut(width))
                .zip(cg_buffer.par_chunks_mut(width))
                .for_each(encode_row);
        });
    } else {
        input
            .chunks(input_stride)
            .zip(y_buffer.chunks_mut(width))
            .zip(co_buffer.chunks_mut(width))
            .zip(cg_buffer.chunks_mut(width))
            .for_each(encode_row);
    }

    let transform_channel = |(plane, buffer): (&mut Vec<i32>, &mut Vec<i32>)| {
        forward_haar_after_horizontal(plane, buffer, width, height, levels);
    };
    if let Some(pool) = codec_thread_pool() {
        pool.install(|| {
            planes
                .par_iter_mut()
                .zip(buffers.par_iter_mut())
                .for_each(transform_channel);
        });
    } else {
        planes
            .iter_mut()
            .zip(buffers.iter_mut())
            .for_each(transform_channel);
    }
}

fn forward_haar_after_horizontal(
    plane: &mut [i32],
    buffer: &mut [i32],
    width: usize,
    height: usize,
    levels: usize,
) {
    forward_vertical(buffer, plane, width, width, height);
    let (mut active_width, mut active_height) = (width.div_ceil(2), height.div_ceil(2));
    for _ in 1..levels {
        let parallel = active_width.saturating_mul(active_height) >= PARALLEL_TRANSFORM_PIXELS
            && codec_thread_pool().is_some();
        let transform_row = |(source, target): (&[i32], &mut [i32])| {
            forward_haar_1d_to(&source[..active_width], &mut target[..active_width]);
        };
        if parallel {
            plane
                .par_chunks(width)
                .zip(buffer.par_chunks_mut(width))
                .take(active_height)
                .for_each(transform_row);
        } else {
            plane
                .chunks(width)
                .zip(buffer.chunks_mut(width))
                .take(active_height)
                .for_each(transform_row);
        }
        forward_vertical(buffer, plane, width, active_width, active_height);
        active_width = active_width.div_ceil(2);
        active_height = active_height.div_ceil(2);
    }
}

fn forward_vertical(
    input: &[i32],
    output: &mut [i32],
    stride: usize,
    active_width: usize,
    active_height: usize,
) {
    let transform_column_row = |(row, target): (usize, &mut [i32])| {
        forward_haar_column_row(input, target, row, stride, active_width, active_height);
    };
    if active_width.saturating_mul(active_height) >= PARALLEL_TRANSFORM_PIXELS
        && codec_thread_pool().is_some()
    {
        output
            .par_chunks_mut(stride)
            .take(active_height)
            .enumerate()
            .for_each(transform_column_row);
    } else {
        output
            .chunks_mut(stride)
            .take(active_height)
            .enumerate()
            .for_each(transform_column_row);
    }
}

fn inverse_haar_to_rgb(
    planes: &mut [Vec<i32>; CHANNELS],
    buffers: &mut [Vec<i32>; CHANNELS],
    width: usize,
    height: usize,
    levels: usize,
    output: &mut [u8],
) {
    let inverse_channel = |(plane, buffer): (&mut Vec<i32>, &mut Vec<i32>)| {
        inverse_haar_before_horizontal(plane, buffer, width, height, levels);
    };
    if let Some(pool) = codec_thread_pool() {
        pool.install(|| {
            planes
                .par_iter_mut()
                .zip(buffers.par_iter_mut())
                .for_each(inverse_channel);
        });
    } else {
        planes
            .iter_mut()
            .zip(buffers.iter_mut())
            .for_each(inverse_channel);
    }

    let [y_buffer, co_buffer, cg_buffer] = buffers;
    let output_stride = width * CHANNELS;
    let decode_row = |(((rgb, y_row), co_row), cg_row): InverseRgbRows<'_>| {
        inverse_rgb_row(&y_row[..width], &co_row[..width], &cg_row[..width], rgb);
    };
    if let Some(pool) = codec_thread_pool() {
        pool.install(|| {
            output
                .par_chunks_mut(output_stride)
                .zip(y_buffer.par_chunks(width))
                .zip(co_buffer.par_chunks(width))
                .zip(cg_buffer.par_chunks(width))
                .for_each(decode_row);
        });
    } else {
        output
            .chunks_mut(output_stride)
            .zip(y_buffer.chunks(width))
            .zip(co_buffer.chunks(width))
            .zip(cg_buffer.chunks(width))
            .for_each(decode_row);
    }
}

fn inverse_haar_before_horizontal(
    plane: &mut [i32],
    buffer: &mut [i32],
    width: usize,
    height: usize,
    levels: usize,
) {
    let dimensions = level_dimensions(width, height, levels);
    for level in (1..=levels).rev() {
        let (active_width, active_height) = dimensions[level - 1];
        inverse_vertical(plane, buffer, width, active_width, active_height);
        if level == 1 {
            break;
        }
        let inverse_row = |(source, target): (&[i32], &mut [i32])| {
            inverse_haar_1d_to(&source[..active_width], &mut target[..active_width]);
        };
        if active_width.saturating_mul(active_height) >= PARALLEL_TRANSFORM_PIXELS
            && codec_thread_pool().is_some()
        {
            buffer
                .par_chunks(width)
                .zip(plane.par_chunks_mut(width))
                .take(active_height)
                .for_each(inverse_row);
        } else {
            buffer
                .chunks(width)
                .zip(plane.chunks_mut(width))
                .take(active_height)
                .for_each(inverse_row);
        }
    }
}

fn inverse_vertical(
    input: &[i32],
    output: &mut [i32],
    stride: usize,
    active_width: usize,
    active_height: usize,
) {
    let inverse_column_row = |(row, target): (usize, &mut [i32])| {
        inverse_haar_column_row(input, target, row, stride, active_width, active_height);
    };
    if active_width.saturating_mul(active_height) >= PARALLEL_TRANSFORM_PIXELS
        && codec_thread_pool().is_some()
    {
        output
            .par_chunks_mut(stride)
            .take(active_height)
            .enumerate()
            .for_each(inverse_column_row);
    } else {
        output
            .chunks_mut(stride)
            .take(active_height)
            .enumerate()
            .for_each(inverse_column_row);
    }
}

fn forward_haar_1d_to(input: &[i32], output: &mut [i32]) {
    let len = input.len();
    debug_assert!(output.len() >= len);
    if len == 0 {
        return;
    }
    if len == 1 {
        output[0] = input[0];
        return;
    }
    let low_len = len.div_ceil(2);
    let pairs = len / 2;
    for index in 0..pairs {
        let even = input[index * 2];
        let odd = input[index * 2 + 1];
        let high = odd - even;
        let low = even + (high >> 1);
        output[index] = low;
        output[low_len + index] = high;
    }
    if len & 1 != 0 {
        output[low_len - 1] = input[len - 1];
    }
}

fn inverse_haar_1d_to(input: &[i32], output: &mut [i32]) {
    let len = input.len();
    debug_assert!(output.len() >= len);
    if len == 0 {
        return;
    }
    if len == 1 {
        output[0] = input[0];
        return;
    }
    let low_len = len.div_ceil(2);
    let pairs = len / 2;
    for index in 0..pairs {
        let low = input[index];
        let high = input[low_len + index];
        let even = low - (high >> 1);
        let odd = high + even;
        output[index * 2] = even;
        output[index * 2 + 1] = odd;
    }
    if len & 1 != 0 {
        output[len - 1] = input[low_len - 1];
    }
}

fn forward_rgb_row(rgb: &[u8], y: &mut [i32], co: &mut [i32], cg: &mut [i32]) {
    let width = y.len();
    let low_len = width.div_ceil(2);
    let pairs = width / 2;
    for pair in 0..pairs {
        let even = rgb_to_ycocg_pixel(&rgb[pair * 6..pair * 6 + 3]);
        let odd = rgb_to_ycocg_pixel(&rgb[pair * 6 + 3..pair * 6 + 6]);
        let (y_low, y_high) = forward_haar_pair(even.0, odd.0);
        let (co_low, co_high) = forward_haar_pair(even.1, odd.1);
        let (cg_low, cg_high) = forward_haar_pair(even.2, odd.2);
        y[pair] = y_low;
        y[low_len + pair] = y_high;
        co[pair] = co_low;
        co[low_len + pair] = co_high;
        cg[pair] = cg_low;
        cg[low_len + pair] = cg_high;
    }
    if width & 1 != 0 {
        let last = rgb_to_ycocg_pixel(&rgb[(width - 1) * CHANNELS..width * CHANNELS]);
        y[low_len - 1] = last.0;
        co[low_len - 1] = last.1;
        cg[low_len - 1] = last.2;
    }
}

fn inverse_rgb_row(y: &[i32], co: &[i32], cg: &[i32], output: &mut [u8]) {
    for x in 0..y.len() {
        let y_value = inverse_haar_sample(y, x);
        let co_value = inverse_haar_sample(co, x);
        let cg_value = inverse_haar_sample(cg, x);
        let temporary = y_value - (cg_value >> 1);
        let green = cg_value + temporary;
        let blue = temporary - (co_value >> 1);
        let red = co_value + blue;
        let offset = x * CHANNELS;
        output[offset] = clamp_u8_value(red);
        output[offset + 1] = clamp_u8_value(green);
        output[offset + 2] = clamp_u8_value(blue);
    }
}

fn rgb_to_ycocg_pixel(rgb: &[u8]) -> (i32, i32, i32) {
    let red = rgb[0] as i32;
    let green = rgb[1] as i32;
    let blue = rgb[2] as i32;
    let co = red - blue;
    let temporary = blue + (co >> 1);
    let cg = green - temporary;
    let y = temporary + (cg >> 1);
    (y, co, cg)
}

fn forward_haar_pair(even: i32, odd: i32) -> (i32, i32) {
    let high = odd - even;
    (even + (high >> 1), high)
}

fn inverse_haar_sample(input: &[i32], index: usize) -> i32 {
    let len = input.len();
    let low_len = len.div_ceil(2);
    if len & 1 != 0 && index == len - 1 {
        return input[low_len - 1];
    }
    let pair = index / 2;
    let low = input[pair];
    let high = input[low_len + pair];
    let even = low - (high >> 1);
    if index & 1 == 0 { even } else { high + even }
}

fn forward_haar_column_row(
    input: &[i32],
    output: &mut [i32],
    output_row: usize,
    stride: usize,
    active_width: usize,
    active_height: usize,
) {
    let low_len = active_height.div_ceil(2);
    let pairs = active_height / 2;
    if output_row < pairs {
        let even_row = output_row * 2;
        let odd_row = even_row + 1;
        for column in 0..active_width {
            let even = input[even_row * stride + column];
            let odd = input[odd_row * stride + column];
            let high = odd - even;
            output[column] = even + (high >> 1);
        }
    } else if output_row < low_len {
        let source_row = active_height - 1;
        output[..active_width]
            .copy_from_slice(&input[source_row * stride..source_row * stride + active_width]);
    } else {
        let pair = output_row - low_len;
        let even_row = pair * 2;
        let odd_row = even_row + 1;
        for column in 0..active_width {
            output[column] = input[odd_row * stride + column] - input[even_row * stride + column];
        }
    }
}

fn inverse_haar_column_row(
    input: &[i32],
    output: &mut [i32],
    output_row: usize,
    stride: usize,
    active_width: usize,
    active_height: usize,
) {
    let low_len = active_height.div_ceil(2);
    let pairs = active_height / 2;
    if output_row >= pairs * 2 {
        output[..active_width]
            .copy_from_slice(&input[(low_len - 1) * stride..(low_len - 1) * stride + active_width]);
        return;
    }

    let pair = output_row / 2;
    let low_row = pair;
    let high_row = low_len + pair;
    if output_row & 1 == 0 {
        for column in 0..active_width {
            let low = input[low_row * stride + column];
            let high = input[high_row * stride + column];
            output[column] = low - (high >> 1);
        }
    } else {
        for column in 0..active_width {
            let low = input[low_row * stride + column];
            let high = input[high_row * stride + column];
            let even = low - (high >> 1);
            output[column] = high + even;
        }
    }
}

fn event(ordinal: u64, every: u64) -> bool {
    every != 0 && ordinal % every == 0
}

fn wrap_level(level: i16, levels: usize) -> u8 {
    (level - 1).rem_euclid(levels as i16) as u8 + 1
}

fn clear_low_bits(value: i32, bits: u8) -> i32 {
    let bits = bits.min(30);
    if bits == 0 {
        return value;
    }
    let sign = value.signum();
    let magnitude = value.unsigned_abs();
    let cleared = (magnitude >> bits) << bits;
    (cleared as i32).saturating_mul(sign)
}

fn xor_bitplane(value: i32, bitplane: u8) -> i32 {
    let bit = bitplane.saturating_sub(1).min(30);
    let sign = if value < 0 { -1 } else { 1 };
    let magnitude = value.unsigned_abs() ^ (1_u32 << bit);
    (magnitude as i32).saturating_mul(sign)
}

fn round_div(value: i32, divisor: i32) -> i32 {
    if divisor <= 1 {
        return value;
    }
    if value >= 0 {
        (value + divisor / 2) / divisor
    } else {
        (value - divisor / 2) / divisor
    }
}

fn control_amount(master: f32, channel: f32) -> f32 {
    if master.is_finite() && channel.is_finite() {
        (master * channel).clamp(0.0, RAW_MOSH_COMBINED_CONTROL_MAX)
    } else {
        0.0
    }
}

fn scale_interval(interval: u64, amount: f32) -> u64 {
    if interval == 0 || amount <= 0.0 {
        0
    } else {
        (interval as f32 / amount).round().max(1.0) as u64
    }
}

fn scale_i16(value: i16, amount: f32) -> i16 {
    clamp_i16(value as f32 * amount)
}

fn scale_i8(value: i8, amount: f32) -> i8 {
    clamp_i8(value as f32 * amount)
}

fn scale_u8(value: u8, amount: f32) -> u8 {
    (value as f32 * amount)
        .round()
        .clamp(u8::MIN as f32, u8::MAX as f32) as u8
}

fn scale_usize(value: usize, amount: f32) -> usize {
    (value as f32 * amount).round().max(0.0) as usize
}

fn clamp_i16(value: f32) -> i16 {
    value.round().clamp(i16::MIN as f32, i16::MAX as f32) as i16
}

fn clamp_i8(value: f32) -> i8 {
    value.round().clamp(i8::MIN as f32, i8::MAX as f32) as i8
}

fn clamp_u64(value: f32) -> u64 {
    value.round().max(0.0) as u64
}

fn clamp_u8_value(value: i32) -> u8 {
    value.clamp(0, 255) as u8
}

fn zigzag_i32(value: i32) -> u64 {
    ((value << 1) ^ (value >> 31)) as u32 as u64
}

fn varint_len(mut value: u64) -> usize {
    let mut len = 1;
    while value >= 0x80 {
        value >>= 7;
        len += 1;
    }
    len
}

fn invalid_input(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, message.into())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    fn test_frame(width: usize, height: usize, frame: usize) -> Vec<u8> {
        let mut input = vec![0; width * height * CHANNELS];
        for y in 0..height {
            for x in 0..width {
                let offset = (y * width + x) * CHANNELS;
                input[offset] = ((x * 17 + frame * 13) & 0xff) as u8;
                input[offset + 1] = ((y * 23 + frame * 7) & 0xff) as u8;
                input[offset + 2] = (((x ^ y) * 19 + frame * 29) & 0xff) as u8;
            }
        }
        input
    }

    #[test]
    fn clean_quality_100_is_reversible_for_odd_dimensions() {
        let config = WaveletCodecConfig {
            width: 17,
            height: 13,
            levels: 3,
            quality: 100,
            history_len: 4,
        };
        let mut codec = WaveletCodec::new(config).unwrap();
        let input = test_frame(config.width, config.height, 0);
        let mut output = vec![0; input.len()];
        codec
            .process_rgb_frame(&input, &WaveletGlitchParams::default(), &mut output)
            .unwrap();
        assert_eq!(output, input);
    }

    #[test]
    fn small_config_clamps_default_decomposition_depth() {
        let config = WaveletCodecConfig::new(2, 2);
        assert_eq!(config.levels, 1);
        WaveletCodec::new(config).unwrap();
    }

    #[test]
    fn packet_contains_one_approximation_and_three_details_per_level() {
        let config = WaveletCodecConfig {
            width: 32,
            height: 24,
            levels: 3,
            quality: 82,
            history_len: 4,
        };
        let mut codec = WaveletCodec::new(config).unwrap();
        codec.set_rate_estimation(true);
        let packet = codec
            .encode_rgb_frame(&test_frame(config.width, config.height, 0))
            .unwrap();
        assert_eq!(packet.bands.len(), CHANNELS * (1 + config.levels * 3));
        assert!(packet.estimated_bytes > 0);
        assert!(
            packet
                .bands
                .iter()
                .any(|band| band.orientation == WaveletOrientation::Approximation)
        );
    }

    #[test]
    fn presets_produce_distinct_codec_outputs() {
        const PRESETS: &[&str] = &[
            "clean",
            "subband-slip",
            "orientation-cross",
            "scale-fold",
            "bitplane-rain",
            "lowpass-ghost",
            "temporal-weave",
            "packet-loss",
            "lifting-drift",
            "chroma-pyramid",
            "hierarchy-collapse",
        ];
        let config = WaveletCodecConfig {
            width: 48,
            height: 32,
            levels: 3,
            quality: 82,
            history_len: 8,
        };
        let mut checksums = HashSet::new();
        for preset in PRESETS {
            let mut codec = WaveletCodec::new(config).unwrap();
            let mut params = WaveletGlitchParams::default();
            load_wavelet_preset(preset, &mut params).unwrap();
            let mut output = vec![0; config.frame_len().unwrap()];
            for frame in 0..8 {
                codec
                    .process_rgb_frame(
                        &test_frame(config.width, config.height, frame),
                        &params,
                        &mut output,
                    )
                    .unwrap();
            }
            let checksum = output.iter().fold(0_u64, |hash, value| {
                hash.wrapping_mul(1_099_511_628_211)
                    .wrapping_add(*value as u64)
            });
            checksums.insert(checksum);
        }
        assert!(
            checksums.len() >= 9,
            "expected at least 9 distinct outputs, got {}",
            checksums.len()
        );
    }

    #[test]
    fn zero_intensity_disables_wavelet_mutations() {
        let mut params = WaveletGlitchParams::default();
        load_wavelet_preset("hierarchy-collapse", &mut params).unwrap();
        apply_wavelet_controls(
            &mut params,
            RawMoshControls {
                intensity: 0.0,
                ..RawMoshControls::default()
            },
        );
        assert!(!params.has_mutations());
    }

    #[test]
    fn reset_discards_temporal_subband_history() {
        let config = WaveletCodecConfig {
            width: 32,
            height: 24,
            levels: 3,
            quality: 82,
            history_len: 8,
        };
        let mut codec = WaveletCodec::new(config).unwrap();
        let mut params = WaveletGlitchParams::default();
        load_wavelet_preset("lowpass-ghost", &mut params).unwrap();
        let mut output = vec![0; config.frame_len().unwrap()];
        for frame in 0..6 {
            codec
                .process_rgb_frame(
                    &test_frame(config.width, config.height, frame),
                    &params,
                    &mut output,
                )
                .unwrap();
        }
        codec.reset_glitch_state();
        let input = test_frame(config.width, config.height, 9);
        codec
            .process_rgb_frame(&input, &params, &mut output)
            .unwrap();

        let mut fresh = WaveletCodec::new(config).unwrap();
        let mut expected = vec![0; output.len()];
        fresh
            .process_rgb_frame(&input, &params, &mut expected)
            .unwrap();
        assert_eq!(output, expected);
    }
}
