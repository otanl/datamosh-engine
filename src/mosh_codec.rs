use std::collections::VecDeque;
use std::io;
use std::sync::OnceLock;

use rayon::prelude::*;
use rayon::{ThreadPool, ThreadPoolBuilder};

const CHANNELS: usize = 3;
const BITSTREAM_MAGIC: &[u8; 4] = b"MSH0";
const BITSTREAM_VERSION: u8 = 1;
const BITSTREAM_HEADER_LEN: usize = 28;
const BITSTREAM_BLOCK_LEN: usize = 12;
const REFERENCE_SWITCH_CELL_X: usize = 5;
const REFERENCE_SWITCH_CELL_Y: usize = 2;
const REFERENCE_SLOT_CELL_X: usize = 7;
const REFERENCE_SLOT_CELL_Y: usize = 3;
const SAMPLE_ADDRESS_CELL_X: usize = 9;
const SAMPLE_ADDRESS_CELL_Y: usize = 1;
const PARALLEL_FRAME_PIXELS: usize = 200_000;

#[derive(Debug, Clone, Copy)]
pub struct MoshCodecConfig {
    pub width: usize,
    pub height: usize,
    pub block_size: usize,
    pub search_radius: i16,
    pub search_step: i16,
    pub keyframe_interval: u64,
    pub history_len: usize,
    pub reference_mode: MoshReferenceMode,
}

impl MoshCodecConfig {
    pub fn new(width: usize, height: usize) -> Self {
        Self {
            width,
            height,
            ..Self::default()
        }
    }

    pub fn frame_len(&self) -> Option<usize> {
        self.width.checked_mul(self.height)?.checked_mul(CHANNELS)
    }

    fn validate(&self) -> io::Result<()> {
        if self.width == 0 || self.height == 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "width and height must be greater than zero",
            ));
        }
        if self.block_size == 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "block_size must be greater than zero",
            ));
        }
        if self.search_step <= 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "search_step must be greater than zero",
            ));
        }
        if self.history_len == 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "history_len must be greater than zero",
            ));
        }
        self.frame_len().ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                "frame dimensions overflow addressable memory",
            )
        })?;
        Ok(())
    }
}

impl Default for MoshCodecConfig {
    fn default() -> Self {
        Self {
            width: 0,
            height: 0,
            block_size: 16,
            search_radius: 8,
            search_step: 4,
            keyframe_interval: 0,
            history_len: 8,
            reference_mode: MoshReferenceMode::Split,
        }
    }
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum MoshReferenceMode {
    Split,
    Feedback,
}

impl Default for MoshReferenceMode {
    fn default() -> Self {
        Self::Split
    }
}

impl MoshReferenceMode {
    pub fn parse(value: &str) -> Result<Self, String> {
        match value {
            "split" | "clean" | "classic" => Ok(Self::Split),
            "feedback" | "dirty" | "recursive" => Ok(Self::Feedback),
            _ => Err(format!(
                "unsupported reference mode `{value}`; expected split or feedback"
            )),
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::Split => "split",
            Self::Feedback => "feedback",
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct MoshGlitchParams {
    pub mv_scale_x: f32,
    pub mv_scale_y: f32,
    pub mv_jitter: i16,
    pub mv_quant: i16,
    pub reference_lag: usize,
    pub residual_keep: f32,
    pub residual_invert_every: u64,
    pub residual_address_shift_x: i16,
    pub residual_address_shift_y: i16,
    pub residual_address_jitter: i16,
    pub residual_channel_shift: i16,
    pub block_remap_every: u64,
    pub block_remap_stride: i32,
    pub channel_shift: i16,
    pub wrap_motion: bool,
    pub activity_mode: ActivityMode,
    pub activity_threshold: u16,
    pub activity_softness: u16,
    pub reference_bleed: f32,
    pub reference_latch_frames: u64,
    pub reference_slot_count: usize,
    pub reference_slot_shuffle_every: u64,
    pub reference_scanline_height: usize,
    pub reference_scanline_lag_span: usize,
    pub temporal_slice_height: usize,
    pub temporal_slice_lag_span: usize,
    pub temporal_slice_drift: i16,
    pub residual_bank_size: usize,
    pub residual_bank_stride: i32,
    pub residual_bank_shuffle_every: u64,
    pub reference_channel_shift: i16,
    pub reference_channel_lag_span: usize,
    pub reference_channel_lag_stride: i16,
    pub mv_bank_size: usize,
    pub mv_bank_stride: i32,
    pub mv_bank_shuffle_every: u64,
    pub overlap: usize,
    pub motion_diffusion: f32,
    pub mv_field_interpolation: f32,
    pub sample_address_desync: f32,
    pub glitch_cell_size: usize,
    pub glitch_cell_width: usize,
    pub glitch_cell_height: usize,
    pub mv_predictor_desync_every: u64,
    pub mv_predictor_desync_x: i16,
    pub mv_predictor_desync_y: i16,
}

#[derive(Debug, Clone, Copy)]
pub struct MoshBitstreamParams {
    pub enabled: bool,
    pub mv_sign_flip_every: u64,
    pub mv_delta_every: u64,
    pub mv_delta_x: i16,
    pub mv_delta_y: i16,
    pub block_address_shift_every: u64,
    pub block_address_shift_x: i16,
    pub block_address_shift_y: i16,
    pub residual_zero_every: u64,
    pub residual_xor_every: u64,
    pub residual_xor_mask: u8,
    pub entropy_slip_every: u64,
    pub entropy_slip_bytes: i16,
    pub entropy_resync_bytes: usize,
    pub entropy_slip_windows: usize,
    pub coeff_glitch_every: u64,
    pub coeff_block_size: usize,
    pub coeff_shift: i16,
    pub coeff_sign_flip_every: u64,
    pub coeff_zero_high: usize,
    pub coeff_quant: i16,
    pub codebook_replace_every: u64,
    pub codebook_tile_size: usize,
    pub codebook_slots: usize,
    pub codebook_stride: i32,
    pub codebook_update_every: u64,
    pub codebook_shuffle_every: u64,
}

impl Default for MoshBitstreamParams {
    fn default() -> Self {
        Self {
            enabled: false,
            mv_sign_flip_every: 0,
            mv_delta_every: 0,
            mv_delta_x: 0,
            mv_delta_y: 0,
            block_address_shift_every: 0,
            block_address_shift_x: 0,
            block_address_shift_y: 0,
            residual_zero_every: 0,
            residual_xor_every: 0,
            residual_xor_mask: 0xff,
            entropy_slip_every: 0,
            entropy_slip_bytes: 1,
            entropy_resync_bytes: 0,
            entropy_slip_windows: 1,
            coeff_glitch_every: 0,
            coeff_block_size: 8,
            coeff_shift: 0,
            coeff_sign_flip_every: 0,
            coeff_zero_high: 0,
            coeff_quant: 1,
            codebook_replace_every: 0,
            codebook_tile_size: 8,
            codebook_slots: 64,
            codebook_stride: 1,
            codebook_update_every: 1,
            codebook_shuffle_every: 0,
        }
    }
}

impl MoshBitstreamParams {
    pub fn has_mutations(&self) -> bool {
        self.mv_sign_flip_every != 0
            || self.mv_delta_every != 0
            || self.block_address_shift_every != 0
            || self.residual_zero_every != 0
            || self.residual_xor_every != 0
            || self.entropy_slip_every != 0
            || self.coeff_glitch_every != 0
            || self.codebook_replace_every != 0
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct MoshBitstreamMutationStats {
    pub mv_sign_flipped: u64,
    pub mv_delta_applied: u64,
    pub block_addresses_shifted: u64,
    pub residual_blocks_zeroed: u64,
    pub residual_blocks_xored: u64,
    pub entropy_slips: u64,
    pub coeff_blocks: u64,
    pub codebook_tiles: u64,
}

impl Default for MoshGlitchParams {
    fn default() -> Self {
        Self {
            mv_scale_x: 1.0,
            mv_scale_y: 1.0,
            mv_jitter: 0,
            mv_quant: 1,
            reference_lag: 1,
            residual_keep: 1.0,
            residual_invert_every: 0,
            residual_address_shift_x: 0,
            residual_address_shift_y: 0,
            residual_address_jitter: 0,
            residual_channel_shift: 0,
            block_remap_every: 0,
            block_remap_stride: 0,
            channel_shift: 0,
            wrap_motion: false,
            activity_mode: ActivityMode::All,
            activity_threshold: 12,
            activity_softness: 0,
            reference_bleed: 0.0,
            reference_latch_frames: 1,
            reference_slot_count: 1,
            reference_slot_shuffle_every: 0,
            reference_scanline_height: 0,
            reference_scanline_lag_span: 0,
            temporal_slice_height: 0,
            temporal_slice_lag_span: 0,
            temporal_slice_drift: 0,
            residual_bank_size: 0,
            residual_bank_stride: 0,
            residual_bank_shuffle_every: 0,
            reference_channel_shift: 0,
            reference_channel_lag_span: 0,
            reference_channel_lag_stride: 0,
            mv_bank_size: 0,
            mv_bank_stride: 0,
            mv_bank_shuffle_every: 0,
            overlap: 0,
            motion_diffusion: 0.0,
            mv_field_interpolation: 0.0,
            sample_address_desync: 0.0,
            glitch_cell_size: 0,
            glitch_cell_width: 0,
            glitch_cell_height: 0,
            mv_predictor_desync_every: 0,
            mv_predictor_desync_x: 0,
            mv_predictor_desync_y: 0,
        }
    }
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum ActivityMode {
    All,
    Active,
    Static,
}

impl ActivityMode {
    pub fn parse(value: &str) -> Result<Self, String> {
        match value {
            "all" => Ok(Self::All),
            "active" | "motion" | "diff" | "difference" => Ok(Self::Active),
            "static" | "still" | "inactive" => Ok(Self::Static),
            _ => Err(format!(
                "unsupported activity mode `{value}`; expected all, active, or static"
            )),
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::All => "all",
            Self::Active => "active",
            Self::Static => "static",
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct MoshCodecStats {
    pub frames_in: u64,
    pub keyframes: u64,
    pub predicted_frames: u64,
    pub blocks_encoded: u64,
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum MoshPacketKind {
    I,
    P,
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub struct MotionBlock {
    pub x: usize,
    pub y: usize,
    pub w: usize,
    pub h: usize,
    pub dx: i16,
    pub dy: i16,
}

#[derive(Clone, Copy)]
struct DecodeBlockState {
    block_mix: f32,
    dirty_dx: i16,
    dirty_dy: i16,
    invert_residual: bool,
}

#[derive(Debug, Clone)]
pub struct MoshPacket {
    pub kind: MoshPacketKind,
    pub width: usize,
    pub height: usize,
    pub blocks: Vec<MotionBlock>,
    pub residual: Vec<i16>,
    pub keyframe: Vec<u8>,
}

impl Default for MoshPacket {
    fn default() -> Self {
        Self {
            kind: MoshPacketKind::I,
            width: 0,
            height: 0,
            blocks: Vec::new(),
            residual: Vec::new(),
            keyframe: Vec::new(),
        }
    }
}

pub struct MoshCodec {
    config: MoshCodecConfig,
    stats: MoshCodecStats,
    encoder_history: VecDeque<Vec<u8>>,
    decoder_history: VecDeque<Vec<u8>>,
    residual_codebook: VecDeque<Vec<i16>>,
    packet_scratch: MoshPacket,
    decoded_packet_scratch: MoshPacket,
    bitstream_scratch: Vec<u8>,
    clean_residual_scratch: Vec<i16>,
    codebook_index_scratch: Vec<usize>,
}

impl MoshCodec {
    pub fn new(config: MoshCodecConfig) -> io::Result<Self> {
        config.validate()?;
        Ok(Self {
            config,
            stats: MoshCodecStats::default(),
            encoder_history: VecDeque::new(),
            decoder_history: VecDeque::new(),
            residual_codebook: VecDeque::new(),
            packet_scratch: MoshPacket::default(),
            decoded_packet_scratch: MoshPacket::default(),
            bitstream_scratch: Vec::new(),
            clean_residual_scratch: Vec::new(),
            codebook_index_scratch: Vec::new(),
        })
    }

    pub fn config(&self) -> &MoshCodecConfig {
        &self.config
    }

    pub fn stats(&self) -> &MoshCodecStats {
        &self.stats
    }

    pub fn reset_glitch_state(&mut self) {
        self.encoder_history.clear();
        self.decoder_history.clear();
        self.residual_codebook.clear();
    }

    pub fn process_rgb_frame(
        &mut self,
        input: &[u8],
        params: &MoshGlitchParams,
        output: &mut [u8],
    ) -> io::Result<()> {
        let mut packet = std::mem::take(&mut self.packet_scratch);
        let result = (|| {
            self.encode_rgb_packet_into(input, &mut packet)?;
            self.decode_rgb_packet(&packet, params, output)?;
            self.push_processed_frame(input, output);
            Ok(())
        })();
        self.packet_scratch = packet;
        result
    }

    pub fn process_rgb_frame_bitstream(
        &mut self,
        input: &[u8],
        params: &MoshGlitchParams,
        bitstream_params: &MoshBitstreamParams,
        output: &mut [u8],
    ) -> io::Result<MoshBitstreamMutationStats> {
        let mut packet = std::mem::take(&mut self.packet_scratch);
        let mut decoded_packet = std::mem::take(&mut self.decoded_packet_scratch);
        let mut bitstream = std::mem::take(&mut self.bitstream_scratch);
        let result = (|| {
            self.encode_rgb_packet_into(input, &mut packet)?;
            let mut stats = MoshBitstreamMutationStats::default();
            stats.codebook_tiles +=
                self.mutate_packet_with_residual_codebook(&mut packet, bitstream_params)?;
            encode_packet_bitstream_into(&packet, &mut bitstream)?;
            let bitstream_stats =
                mutate_packet_bitstream(&mut bitstream, bitstream_params, self.stats.frames_in)?;
            stats.mv_sign_flipped += bitstream_stats.mv_sign_flipped;
            stats.mv_delta_applied += bitstream_stats.mv_delta_applied;
            stats.block_addresses_shifted += bitstream_stats.block_addresses_shifted;
            stats.residual_blocks_zeroed += bitstream_stats.residual_blocks_zeroed;
            stats.residual_blocks_xored += bitstream_stats.residual_blocks_xored;
            stats.entropy_slips += bitstream_stats.entropy_slips;
            stats.coeff_blocks += bitstream_stats.coeff_blocks;
            stats.codebook_tiles += bitstream_stats.codebook_tiles;
            decode_packet_bitstream_into(&bitstream, &mut decoded_packet)?;
            self.decode_rgb_packet(&decoded_packet, params, output)?;
            self.push_processed_frame(input, output);
            Ok(stats)
        })();
        self.packet_scratch = packet;
        self.decoded_packet_scratch = decoded_packet;
        self.bitstream_scratch = bitstream;
        result
    }

    pub fn encode_rgb_packet(&mut self, input: &[u8]) -> io::Result<MoshPacket> {
        let mut packet = MoshPacket::default();
        self.encode_rgb_packet_into(input, &mut packet)?;
        Ok(packet)
    }

    fn encode_rgb_packet_into(&mut self, input: &[u8], packet: &mut MoshPacket) -> io::Result<()> {
        let frame_len = self.config.frame_len().expect("validated frame dimensions");
        if input.len() != frame_len {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("input frame must be {frame_len} bytes of rgb24"),
            ));
        }

        self.stats.frames_in += 1;
        let should_keyframe = self.encoder_history.is_empty()
            || (self.config.keyframe_interval != 0
                && (self.stats.frames_in - 1) % self.config.keyframe_interval == 0);

        packet.width = self.config.width;
        packet.height = self.config.height;
        packet.blocks.clear();
        packet.residual.clear();
        packet.keyframe.clear();

        if should_keyframe {
            self.stats.keyframes += 1;
            packet.kind = MoshPacketKind::I;
            packet.keyframe.extend_from_slice(input);
            return Ok(());
        }

        let reference = self
            .encoder_reference_frame(1)
            .expect("encoder history is not empty");
        self.motion_blocks_into(input, reference, &mut packet.blocks);
        self.residual_into(input, reference, &packet.blocks, &mut packet.residual);
        self.stats.predicted_frames += 1;
        self.stats.blocks_encoded += packet.blocks.len() as u64;
        packet.kind = MoshPacketKind::P;
        Ok(())
    }

    pub fn decode_rgb_packet(
        &self,
        packet: &MoshPacket,
        params: &MoshGlitchParams,
        output: &mut [u8],
    ) -> io::Result<()> {
        let frame_len = self.config.frame_len().expect("validated frame dimensions");
        if output.len() != frame_len {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("output frame must be {frame_len} bytes of rgb24"),
            ));
        }
        if packet.width != self.config.width || packet.height != self.config.height {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "packet dimensions do not match codec dimensions",
            ));
        }

        match packet.kind {
            MoshPacketKind::I => {
                if packet.keyframe.len() != frame_len {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidInput,
                        "I packet keyframe has the wrong length",
                    ));
                }
                output.copy_from_slice(&packet.keyframe);
            }
            MoshPacketKind::P => {
                if packet.residual.len() != frame_len {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidInput,
                        "P packet residual has the wrong length",
                    ));
                }
                let glitch_reference = self
                    .decoder_reference_frame(params.reference_lag)
                    .ok_or_else(|| {
                        io::Error::new(io::ErrorKind::InvalidInput, "no reference frame available")
                    })?;
                let clean_reference = self.encoder_reference_frame(1).unwrap_or(glitch_reference);
                self.decode_predicted_packet(
                    packet,
                    glitch_reference,
                    clean_reference,
                    params,
                    output,
                );
            }
        }

        Ok(())
    }

    fn mutate_packet_with_residual_codebook(
        &mut self,
        packet: &mut MoshPacket,
        params: &MoshBitstreamParams,
    ) -> io::Result<u64> {
        if packet.kind != MoshPacketKind::P
            || params.codebook_slots == 0
            || packet.residual.is_empty()
        {
            return Ok(0);
        }

        let tile_size = coefficient_block_size(params.codebook_tile_size);
        if packet.width < tile_size || packet.height < tile_size {
            return Ok(0);
        }

        let tile_len = tile_size * tile_size * CHANNELS;
        let mut clean_residual = std::mem::take(&mut self.clean_residual_scratch);
        let mut eligible = std::mem::take(&mut self.codebook_index_scratch);
        let result = (|| {
            clean_residual.clear();
            clean_residual.extend_from_slice(&packet.residual);
            eligible.clear();
            eligible.extend(
                self.residual_codebook
                    .iter()
                    .enumerate()
                    .filter_map(|(index, tile)| (tile.len() == tile_len).then_some(index)),
            );

            let mut replaced = 0;
            let mut tile_index = 0_u64;

            if params.codebook_replace_every != 0 && !eligible.is_empty() {
                for y in (0..=packet.height - tile_size).step_by(tile_size) {
                    for x in (0..=packet.width - tile_size).step_by(tile_size) {
                        let ordinal = tile_index + 1;
                        if is_every_u64(params.codebook_replace_every, ordinal) {
                            let source = codebook_source_index(
                                &eligible,
                                tile_index,
                                self.stats.frames_in,
                                params,
                            );
                            let tile = self.residual_codebook.get(source).ok_or_else(|| {
                                invalid_data("MSH0 residual codebook index escaped")
                            })?;
                            write_residual_tile(
                                &mut packet.residual,
                                packet.width,
                                x,
                                y,
                                tile_size,
                                tile,
                            )?;
                            replaced += 1;
                        }
                        tile_index += 1;
                    }
                }
            }

            self.update_residual_codebook(packet, &clean_residual, tile_size, params)?;
            Ok(replaced)
        })();
        self.clean_residual_scratch = clean_residual;
        self.codebook_index_scratch = eligible;
        result
    }

    fn update_residual_codebook(
        &mut self,
        packet: &MoshPacket,
        residual: &[i16],
        tile_size: usize,
        params: &MoshBitstreamParams,
    ) -> io::Result<()> {
        let update_every = params.codebook_update_every.max(1);
        let mut tile_index = 0_u64;

        for y in (0..=packet.height - tile_size).step_by(tile_size) {
            for x in (0..=packet.width - tile_size).step_by(tile_size) {
                let ordinal = tile_index + 1;
                if is_every_u64(update_every, ordinal) {
                    let tile = read_residual_tile(residual, packet.width, x, y, tile_size)?;
                    self.residual_codebook.push_back(tile);
                    while self.residual_codebook.len() > params.codebook_slots {
                        self.residual_codebook.pop_front();
                    }
                }
                tile_index += 1;
            }
        }

        Ok(())
    }

    fn push_processed_frame(&mut self, input: &[u8], output: &[u8]) {
        match self.config.reference_mode {
            MoshReferenceMode::Split => {
                push_bounded_history_copy(
                    &mut self.encoder_history,
                    self.config.history_len,
                    input,
                );
            }
            MoshReferenceMode::Feedback => {
                push_bounded_history_copy(
                    &mut self.encoder_history,
                    self.config.history_len,
                    output,
                );
            }
        }
        push_bounded_history_copy(&mut self.decoder_history, self.config.history_len, output);
    }

    #[cfg(test)]
    fn push_decoder_history(&mut self, frame: Vec<u8>) {
        while self.decoder_history.len() >= self.config.history_len {
            self.decoder_history.pop_front();
        }
        self.decoder_history.push_back(frame);
    }

    fn encoder_reference_frame(&self, lag: usize) -> Option<&[u8]> {
        history_frame(&self.encoder_history, lag)
    }

    fn decoder_reference_frame(&self, lag: usize) -> Option<&[u8]> {
        history_frame(&self.decoder_history, lag)
    }

    fn motion_blocks_into(&self, input: &[u8], reference: &[u8], blocks: &mut Vec<MotionBlock>) {
        blocks.clear();
        let block = self.config.block_size;
        let width = self.config.width;
        let height = self.config.height;
        let columns = width.div_ceil(block);
        let rows = height.div_ceil(block);
        blocks.reserve(columns.saturating_mul(rows));

        for y in (0..height).step_by(block) {
            for x in (0..width).step_by(block) {
                let w = block.min(width - x);
                let h = block.min(height - y);
                blocks.push(MotionBlock {
                    x,
                    y,
                    w,
                    h,
                    dx: 0,
                    dy: 0,
                });
            }
        }

        if width.saturating_mul(height) >= PARALLEL_FRAME_PIXELS {
            if let Some(pool) = codec_thread_pool() {
                pool.install(|| {
                    blocks.par_iter_mut().for_each(|motion_block| {
                        let (dx, dy) = self.find_motion(
                            input,
                            reference,
                            motion_block.x,
                            motion_block.y,
                            motion_block.w,
                            motion_block.h,
                        );
                        motion_block.dx = dx;
                        motion_block.dy = dy;
                    });
                });
                return;
            }
        }

        for motion_block in blocks {
            let (dx, dy) = self.find_motion(
                input,
                reference,
                motion_block.x,
                motion_block.y,
                motion_block.w,
                motion_block.h,
            );
            motion_block.dx = dx;
            motion_block.dy = dy;
        }
    }

    fn find_motion(
        &self,
        input: &[u8],
        reference: &[u8],
        x: usize,
        y: usize,
        w: usize,
        h: usize,
    ) -> (i16, i16) {
        let radius = self.config.search_radius.max(0);
        let step = self.config.search_step.max(1);
        let mut best = (0, 0);
        let mut best_error = u64::MAX;

        let mut dy = -radius;
        while dy <= radius {
            let mut dx = -radius;
            while dx <= radius {
                let error = self.block_error(input, reference, x, y, w, h, dx, dy);
                if error < best_error
                    || (error == best_error && vector_cost(dx, dy) < vector_cost(best.0, best.1))
                {
                    best_error = error;
                    best = (dx, dy);
                }
                dx = dx.saturating_add(step);
                if step == 0 {
                    break;
                }
            }
            dy = dy.saturating_add(step);
            if step == 0 {
                break;
            }
        }

        best
    }

    fn block_error(
        &self,
        input: &[u8],
        reference: &[u8],
        x: usize,
        y: usize,
        w: usize,
        h: usize,
        dx: i16,
        dy: i16,
    ) -> u64 {
        let sample_step = (self.config.block_size / 4).max(2);
        let mut error = 0_u64;

        for by in (0..h).step_by(sample_step) {
            for bx in (0..w).step_by(sample_step) {
                let src_x = clamp_coord(x as isize + bx as isize + dx as isize, self.config.width);
                let src_y = clamp_coord(y as isize + by as isize + dy as isize, self.config.height);
                let input_index = rgb_index(self.config.width, x + bx, y + by);
                let ref_index = rgb_index(self.config.width, src_x, src_y);
                let input_y = luma(
                    input[input_index],
                    input[input_index + 1],
                    input[input_index + 2],
                );
                let ref_y = luma(
                    reference[ref_index],
                    reference[ref_index + 1],
                    reference[ref_index + 2],
                );
                error += input_y.abs_diff(ref_y) as u64;
            }
        }

        error
    }

    fn residual_into(
        &self,
        input: &[u8],
        reference: &[u8],
        blocks: &[MotionBlock],
        residual: &mut Vec<i16>,
    ) {
        let frame_len = self.config.frame_len().expect("validated frame dimensions");
        residual.resize(frame_len, 0);

        for block in blocks {
            for by in 0..block.h {
                for bx in 0..block.w {
                    let dst_x = block.x + bx;
                    let dst_y = block.y + by;
                    let src_x = clamp_coord(dst_x as isize + block.dx as isize, self.config.width);
                    let src_y = clamp_coord(dst_y as isize + block.dy as isize, self.config.height);
                    let dst = rgb_index(self.config.width, dst_x, dst_y);
                    let src = rgb_index(self.config.width, src_x, src_y);

                    for channel in 0..CHANNELS {
                        residual[dst + channel] =
                            input[dst + channel] as i16 - reference[src + channel] as i16;
                    }
                }
            }
        }
    }

    fn decode_predicted_packet(
        &self,
        packet: &MoshPacket,
        glitch_reference: &[u8],
        clean_reference: &[u8],
        params: &MoshGlitchParams,
        output: &mut [u8],
    ) {
        let frame_len = self.config.frame_len().expect("validated frame dimensions");
        let overlap = params.overlap.min(self.config.block_size);
        let dirty_vectors = self.dirty_motion_vectors(packet, params);
        let block_states: Vec<_> = packet
            .blocks
            .iter()
            .enumerate()
            .map(|(block_index, block)| {
                let block_mix = activity_mix(block_activity_score(packet, block), params);
                let (dirty_dx, dirty_dy) = dirty_vectors
                    .get(block_index)
                    .copied()
                    .unwrap_or((block.dx, block.dy));
                DecodeBlockState {
                    block_mix,
                    dirty_dx,
                    dirty_dy,
                    invert_residual: block_mix > 0.0
                        && params.residual_invert_every != 0
                        && ((block_index as u64 + 1) % params.residual_invert_every == 0),
                }
            })
            .collect();

        if overlap == 0 {
            if self.config.width.saturating_mul(self.config.height) < PARALLEL_FRAME_PIXELS {
                for (block_index, block) in packet.blocks.iter().enumerate() {
                    let state = block_states[block_index];
                    let x_end = (block.x + block.w).min(self.config.width);
                    let y_end = (block.y + block.h).min(self.config.height);
                    for dst_y in block.y..y_end {
                        for dst_x in block.x..x_end {
                            let Some(values) = self.decode_predicted_pixel(
                                packet,
                                glitch_reference,
                                clean_reference,
                                params,
                                &dirty_vectors,
                                block_index,
                                block,
                                state,
                                dst_x,
                                dst_y,
                                true,
                            ) else {
                                continue;
                            };
                            let dst = rgb_index(self.config.width, dst_x, dst_y);
                            for channel in 0..CHANNELS {
                                output[dst + channel] =
                                    values[channel].round().clamp(0.0, 255.0) as u8;
                            }
                        }
                    }
                }
                return;
            }

            let row_stride = self.config.width * CHANNELS;
            let decode_row = |(dst_y, output_row): (usize, &mut [u8])| {
                for (block_index, block) in packet.blocks.iter().enumerate() {
                    if dst_y < block.y || dst_y >= block.y + block.h {
                        continue;
                    }
                    let state = block_states[block_index];
                    let x_end = (block.x + block.w).min(self.config.width);
                    for dst_x in block.x..x_end {
                        let Some(values) = self.decode_predicted_pixel(
                            packet,
                            glitch_reference,
                            clean_reference,
                            params,
                            &dirty_vectors,
                            block_index,
                            block,
                            state,
                            dst_x,
                            dst_y,
                            true,
                        ) else {
                            continue;
                        };
                        let dst = dst_x * CHANNELS;
                        for channel in 0..CHANNELS {
                            output_row[dst + channel] =
                                values[channel].round().clamp(0.0, 255.0) as u8;
                        }
                    }
                }
            };

            if let Some(pool) = codec_thread_pool() {
                pool.install(|| {
                    output
                        .par_chunks_mut(row_stride)
                        .enumerate()
                        .for_each(decode_row);
                });
            } else {
                output
                    .chunks_mut(row_stride)
                    .enumerate()
                    .for_each(decode_row);
            }
            return;
        }

        let mut accum = vec![0.0_f32; frame_len];
        let mut weights = vec![0.0_f32; self.config.width * self.config.height];
        for (block_index, block) in packet.blocks.iter().enumerate() {
            let state = block_states[block_index];
            let x_start = block.x.saturating_sub(overlap);
            let y_start = block.y.saturating_sub(overlap);
            let x_end = (block.x + block.w + overlap).min(self.config.width);
            let y_end = (block.y + block.h + overlap).min(self.config.height);

            for dst_y in y_start..y_end {
                for dst_x in x_start..x_end {
                    let inside_core = block_contains(block, dst_x, dst_y);
                    let weight = overlap_weight(block, dst_x, dst_y, overlap);
                    if weight <= 0.0 {
                        continue;
                    }
                    let Some(values) = self.decode_predicted_pixel(
                        packet,
                        glitch_reference,
                        clean_reference,
                        params,
                        &dirty_vectors,
                        block_index,
                        block,
                        state,
                        dst_x,
                        dst_y,
                        inside_core,
                    ) else {
                        continue;
                    };
                    let dst = rgb_index(self.config.width, dst_x, dst_y);
                    let pixel = dst_y * self.config.width + dst_x;
                    for channel in 0..CHANNELS {
                        accum[dst + channel] += values[channel] * weight;
                    }
                    weights[pixel] += weight;
                }
            }
        }

        for y in 0..self.config.height {
            for x in 0..self.config.width {
                let pixel = y * self.config.width + x;
                let weight = weights[pixel];
                let dst = rgb_index(self.config.width, x, y);
                if weight <= 0.0 {
                    output[dst..dst + CHANNELS].fill(0);
                    continue;
                }
                for channel in 0..CHANNELS {
                    output[dst + channel] =
                        (accum[dst + channel] / weight).round().clamp(0.0, 255.0) as u8;
                }
            }
        }
    }

    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    fn decode_predicted_pixel(
        &self,
        packet: &MoshPacket,
        glitch_reference: &[u8],
        clean_reference: &[u8],
        params: &MoshGlitchParams,
        dirty_vectors: &[(i16, i16)],
        block_index: usize,
        block: &MotionBlock,
        state: DecodeBlockState,
        dst_x: usize,
        dst_y: usize,
        inside_core: bool,
    ) -> Option<[f32; CHANNELS]> {
        let activity = if inside_core {
            activity_mix(pixel_activity_score(packet, block, dst_x, dst_y), params)
        } else {
            state.block_mix
        };
        let dirty_chance = reference_mix(activity, params);
        if !inside_core && dirty_chance <= 0.0 {
            return None;
        }
        let use_dirty = reference_switch(
            dirty_chance,
            block_index,
            dst_x,
            dst_y,
            self.stats.frames_in,
            params,
        );
        let (sample_dirty_dx, sample_dirty_dy) = if use_dirty {
            let (field_dx, field_dy) = interpolated_motion_vector(
                dirty_vectors,
                block_index,
                state.dirty_dx,
                state.dirty_dy,
                packet.width,
                self.config.height,
                self.config.block_size,
                dst_x,
                dst_y,
                params.mv_field_interpolation,
            );
            let (address_dx, address_dy) =
                sample_address_offset(block_index, dst_x, dst_y, self.stats.frames_in, params);
            (field_dx + address_dx, field_dy + address_dy)
        } else {
            (state.dirty_dx as f32, state.dirty_dy as f32)
        };
        let (residual_x, residual_y) = residual_address(
            packet,
            dst_x,
            dst_y,
            block_index,
            self.stats.frames_in,
            params,
        );
        let residual_channel_shift =
            residual_channel_shift(block_index, self.stats.frames_in, params);
        let dirty_reference = if use_dirty {
            Some(
                self.dirty_reference_for_sample(block_index, dst_x, dst_y, params)
                    .unwrap_or(glitch_reference),
            )
        } else {
            None
        };
        let mut values = [0.0; CHANNELS];

        for channel in 0..CHANNELS {
            values[channel] = if use_dirty {
                let channel_reference = self
                    .dirty_reference_for_channel_sample(block_index, dst_x, dst_y, channel, params)
                    .unwrap_or_else(|| dirty_reference.expect("set when use_dirty is true"));
                self.predicted_sample(
                    channel_reference,
                    packet,
                    dst_x,
                    dst_y,
                    channel,
                    sample_dirty_dx,
                    sample_dirty_dy,
                    residual_x,
                    residual_y,
                    residual_channel_shift,
                    params.residual_keep.clamp(-2.0, 2.0),
                    state.invert_residual,
                    params.channel_shift,
                    params.wrap_motion,
                    params.reference_channel_shift,
                )
            } else {
                self.predicted_sample(
                    clean_reference,
                    packet,
                    dst_x,
                    dst_y,
                    channel,
                    block.dx as f32,
                    block.dy as f32,
                    residual_x,
                    residual_y,
                    residual_channel_shift,
                    1.0,
                    false,
                    0,
                    false,
                    0,
                )
            };
        }

        Some(values)
    }

    fn glitched_motion_vector(
        &self,
        packet: &MoshPacket,
        block_index: usize,
        vector_block: &MotionBlock,
        params: &MoshGlitchParams,
    ) -> (i16, i16) {
        let mut dx = scaled_vector(vector_block.dx, params.mv_scale_x);
        let mut dy = scaled_vector(vector_block.dy, params.mv_scale_y);

        if params.motion_diffusion != 0.0 {
            let (diffused_dx, diffused_dy) =
                self.diffused_motion_vector(packet, block_index, dx, dy, params);
            dx = diffused_dx;
            dy = diffused_dy;
        }

        if params.mv_quant > 1 {
            dx = quantize_i16(dx, params.mv_quant);
            dy = quantize_i16(dy, params.mv_quant);
        }

        if params.mv_jitter != 0 {
            dx = dx.saturating_add(jitter(block_index, 0, params.mv_jitter));
            dy = dy.saturating_add(jitter(block_index, 1, params.mv_jitter));
        }

        (dx, dy)
    }

    fn dirty_motion_vectors(
        &self,
        packet: &MoshPacket,
        params: &MoshGlitchParams,
    ) -> Vec<(i16, i16)> {
        if params.mv_predictor_desync_every == 0 {
            return packet
                .blocks
                .iter()
                .enumerate()
                .map(|(block_index, block)| {
                    let vector_block = vector_bank_block(
                        packet,
                        block_index,
                        self.config.block_size,
                        self.stats.frames_in,
                        params,
                    )
                    .or_else(|| remapped_block(&packet.blocks, block_index, params))
                    .unwrap_or(block);
                    self.glitched_motion_vector(packet, block_index, vector_block, params)
                })
                .collect();
        }

        let columns = blocks_per_row(packet.width, self.config.block_size);
        let mut decoded = Vec::with_capacity(packet.blocks.len());

        for (block_index, block) in packet.blocks.iter().enumerate() {
            let vector_block = vector_bank_block(
                packet,
                block_index,
                self.config.block_size,
                self.stats.frames_in,
                params,
            )
            .or_else(|| remapped_block(&packet.blocks, block_index, params))
            .unwrap_or(block);
            let clean_predictor = predictor_from_packet(packet, block_index, columns);
            let dirty_predictor = predictor_from_decoded_vectors(&decoded, block_index, columns)
                .unwrap_or(clean_predictor);
            let delta_x = vector_block.dx.saturating_sub(clean_predictor.0);
            let delta_y = vector_block.dy.saturating_sub(clean_predictor.1);
            let mut dx = dirty_predictor.0.saturating_add(delta_x);
            let mut dy = dirty_predictor.1.saturating_add(delta_y);

            if is_every_u64(params.mv_predictor_desync_every, block_index as u64 + 1) {
                dx = dx.saturating_add(params.mv_predictor_desync_x);
                dy = dy.saturating_add(params.mv_predictor_desync_y);
            }

            dx = scaled_vector(dx, params.mv_scale_x);
            dy = scaled_vector(dy, params.mv_scale_y);

            if params.motion_diffusion != 0.0 {
                let (diffused_dx, diffused_dy) =
                    self.diffused_motion_vector(packet, block_index, dx, dy, params);
                dx = diffused_dx;
                dy = diffused_dy;
            }

            if params.mv_quant > 1 {
                dx = quantize_i16(dx, params.mv_quant);
                dy = quantize_i16(dy, params.mv_quant);
            }

            if params.mv_jitter != 0 {
                dx = dx.saturating_add(jitter(block_index, 0, params.mv_jitter));
                dy = dy.saturating_add(jitter(block_index, 1, params.mv_jitter));
            }

            decoded.push((dx, dy));
        }

        decoded
    }

    #[cfg(test)]
    fn dirty_reference_for_block(
        &self,
        block_index: usize,
        params: &MoshGlitchParams,
    ) -> Option<&[u8]> {
        if params.reference_slot_count <= 1 || params.reference_slot_shuffle_every == 0 {
            return self.decoder_reference_frame(params.reference_lag);
        }
        if !is_every_u64(params.reference_slot_shuffle_every, block_index as u64 + 1) {
            return self.decoder_reference_frame(params.reference_lag);
        }

        let limit = params
            .reference_slot_count
            .max(1)
            .min(self.decoder_history.len())
            .max(1);
        let latch = params.reference_latch_frames.max(1);
        let frame_bucket = self.stats.frames_in / latch;
        let seed = (block_index as u64)
            .wrapping_mul(0xd6e8_feb8_6659_fd93)
            .wrapping_add(frame_bucket.wrapping_mul(0xa076_1d64_78bd_642f));
        let offset = (hash_u64(seed) as usize) % limit;
        let base = params.reference_lag.max(1).saturating_sub(1);
        let lag = 1 + ((base + offset) % limit);
        self.decoder_reference_frame(lag)
    }

    fn dirty_reference_for_sample(
        &self,
        block_index: usize,
        x: usize,
        y: usize,
        params: &MoshGlitchParams,
    ) -> Option<&[u8]> {
        if let Some(reference) = self.temporal_slice_reference_for_sample(y, params) {
            return Some(reference);
        }

        if let Some(reference) = self.scanline_reference_for_sample(x, y, params) {
            return Some(reference);
        }

        if params.reference_slot_count <= 1 || params.reference_slot_shuffle_every == 0 {
            return self.decoder_reference_frame(params.reference_lag);
        }

        let latch = params.reference_latch_frames.max(1);
        let frame_bucket = self.stats.frames_in / latch;
        let seed = sample_cell_seed(
            block_index,
            x,
            y,
            frame_bucket,
            glitch_cell_x(params, REFERENCE_SLOT_CELL_X),
            glitch_cell_y(params, REFERENCE_SLOT_CELL_Y),
        );
        if hash_u64(seed) % params.reference_slot_shuffle_every != 0 {
            return self.decoder_reference_frame(params.reference_lag);
        }

        let limit = params
            .reference_slot_count
            .max(1)
            .min(self.decoder_history.len())
            .max(1);
        let offset = (hash_u64(seed ^ 0x1d8e_4e27_c47d_124f) as usize) % limit;
        let base = params.reference_lag.max(1).saturating_sub(1);
        let lag = 1 + ((base + offset) % limit);
        self.decoder_reference_frame(lag)
    }

    fn dirty_reference_for_channel_sample(
        &self,
        block_index: usize,
        x: usize,
        y: usize,
        channel: usize,
        params: &MoshGlitchParams,
    ) -> Option<&[u8]> {
        if params.reference_channel_lag_span == 0 {
            return None;
        }

        let limit = params
            .reference_channel_lag_span
            .min(self.decoder_history.len())
            .max(1);
        let latch = params.reference_latch_frames.max(1);
        let frame_bucket = self.stats.frames_in / latch;
        let stride = params.reference_channel_lag_stride as i64;
        let seed = sample_cell_seed(
            block_index,
            x,
            y,
            frame_bucket,
            glitch_cell_x(params, REFERENCE_SLOT_CELL_X),
            glitch_cell_y(params, REFERENCE_SLOT_CELL_Y),
        );
        let spatial_wobble = if params.reference_slot_shuffle_every == 0 {
            0
        } else {
            (hash_u64(seed) % limit as u64) as i64
        };
        let offset = (channel as i64 * stride + frame_bucket as i64 + spatial_wobble)
            .rem_euclid(limit as i64) as usize;
        let base = params.reference_lag.max(1).saturating_sub(1);
        let lag = 1 + ((base + offset) % limit);
        self.decoder_reference_frame(lag)
    }

    fn temporal_slice_reference_for_sample(
        &self,
        y: usize,
        params: &MoshGlitchParams,
    ) -> Option<&[u8]> {
        if params.temporal_slice_height == 0 || params.temporal_slice_lag_span == 0 {
            return None;
        }

        let limit = params
            .temporal_slice_lag_span
            .min(self.decoder_history.len())
            .max(1);
        let latch = params.reference_latch_frames.max(1);
        let frame_bucket = self.stats.frames_in / latch;
        let slice = y / params.temporal_slice_height.max(1);
        let drift = (frame_bucket as i64).wrapping_mul(params.temporal_slice_drift as i64);
        let offset = (slice as i64 + drift).rem_euclid(limit as i64) as usize;
        let base = params.reference_lag.max(1).saturating_sub(1);
        let lag = 1 + ((base + offset) % limit);
        self.decoder_reference_frame(lag)
    }

    fn scanline_reference_for_sample(
        &self,
        x: usize,
        y: usize,
        params: &MoshGlitchParams,
    ) -> Option<&[u8]> {
        if params.reference_scanline_height == 0 || params.reference_scanline_lag_span == 0 {
            return None;
        }

        let limit = params
            .reference_scanline_lag_span
            .min(self.decoder_history.len())
            .max(1);
        let latch = params.reference_latch_frames.max(1);
        let frame_bucket = self.stats.frames_in / latch;
        let row = y / params.reference_scanline_height.max(1);
        let seed = (row as u64)
            .wrapping_mul(0xd6e8_feb8_6659_fd93)
            .wrapping_add(((x / 32) as u64).wrapping_mul(0x94d0_49bb_1331_11eb))
            .wrapping_add(frame_bucket.wrapping_mul(0xa076_1d64_78bd_642f));
        let offset = (hash_u64(seed) as usize) % limit;
        let base = params.reference_lag.max(1).saturating_sub(1);
        let lag = 1 + ((base + offset) % limit);
        self.decoder_reference_frame(lag)
    }

    fn diffused_motion_vector(
        &self,
        packet: &MoshPacket,
        block_index: usize,
        dx: i16,
        dy: i16,
        params: &MoshGlitchParams,
    ) -> (i16, i16) {
        let strength = params.motion_diffusion.clamp(0.0, 1.0);
        if strength <= 0.0 || packet.blocks.is_empty() {
            return (dx, dy);
        }

        let columns = blocks_per_row(packet.width, self.config.block_size);
        let row = block_index / columns;
        let column = block_index % columns;
        let mut sum_x = 0.0_f32;
        let mut sum_y = 0.0_f32;
        let mut count = 0.0_f32;

        for neighbor_row in row.saturating_sub(1)..=(row + 1) {
            for neighbor_column in column.saturating_sub(1)..=(column + 1) {
                if neighbor_column >= columns {
                    continue;
                }
                let neighbor_index = neighbor_row * columns + neighbor_column;
                let Some(neighbor) = packet.blocks.get(neighbor_index) else {
                    continue;
                };
                sum_x += scaled_vector(neighbor.dx, params.mv_scale_x) as f32;
                sum_y += scaled_vector(neighbor.dy, params.mv_scale_y) as f32;
                count += 1.0;
            }
        }

        if count <= 0.0 {
            return (dx, dy);
        }

        let avg_x = sum_x / count;
        let avg_y = sum_y / count;
        (
            (dx as f32 + (avg_x - dx as f32) * strength)
                .round()
                .clamp(i16::MIN as f32, i16::MAX as f32) as i16,
            (dy as f32 + (avg_y - dy as f32) * strength)
                .round()
                .clamp(i16::MIN as f32, i16::MAX as f32) as i16,
        )
    }

    fn predicted_sample(
        &self,
        reference: &[u8],
        packet: &MoshPacket,
        dst_x: usize,
        dst_y: usize,
        channel: usize,
        dx: f32,
        dy: f32,
        residual_x: usize,
        residual_y: usize,
        residual_channel_shift: i16,
        residual_keep: f32,
        invert_residual: bool,
        channel_shift: i16,
        wrap_motion: bool,
        reference_channel_shift: i16,
    ) -> f32 {
        let channel_offset = if channel == 0 {
            0
        } else {
            channel_shift as isize * channel as isize
        };
        let reference_channel = shifted_channel(channel, reference_channel_shift);
        let sample_x = dst_x as f32 + dx + channel_offset as f32;
        let sample_y = dst_y as f32 + dy;
        let residual = residual_sample(
            packet,
            residual_x,
            residual_y,
            channel,
            residual_channel_shift,
            residual_keep,
            invert_residual,
        );
        sample_reference(
            reference,
            self.config.width,
            self.config.height,
            sample_x,
            sample_y,
            reference_channel,
            wrap_motion,
        ) + residual
    }
}

fn remapped_block<'a>(
    blocks: &'a [MotionBlock],
    block_index: usize,
    params: &MoshGlitchParams,
) -> Option<&'a MotionBlock> {
    if params.block_remap_every == 0
        || params.block_remap_stride == 0
        || (block_index as u64 + 1) % params.block_remap_every != 0
    {
        return None;
    }

    let len = blocks.len() as i32;
    let index = (block_index as i32 + params.block_remap_stride).rem_euclid(len) as usize;
    blocks.get(index)
}

fn vector_bank_block<'a>(
    packet: &'a MoshPacket,
    block_index: usize,
    block_size: usize,
    frame_index: u64,
    params: &MoshGlitchParams,
) -> Option<&'a MotionBlock> {
    if params.mv_bank_size == 0 || (params.mv_bank_stride == 0 && params.mv_bank_shuffle_every == 0)
    {
        return None;
    }

    let columns = blocks_per_row(packet.width, block_size);
    let rows = packet.height.div_ceil(block_size).max(1);
    let bank_size = params.mv_bank_size.max(1);
    let banks_x = columns.div_ceil(bank_size).max(1);
    let banks_y = rows.div_ceil(bank_size).max(1);
    let bank_count = banks_x.checked_mul(banks_y)?;
    if bank_count <= 1 {
        return None;
    }

    let column = block_index % columns;
    let row = block_index / columns;
    let bank_x = (column / bank_size).min(banks_x - 1);
    let bank_y = (row / bank_size).min(banks_y - 1);
    let local_x = column % bank_size;
    let local_y = row % bank_size;
    let bank_index = bank_y * banks_x + bank_x;
    let latch = params.reference_latch_frames.max(1);
    let frame_bucket = frame_index / latch;
    let seed = (block_index as u64)
        .wrapping_mul(0xd6e8_feb8_6659_fd93)
        .wrapping_add((bank_index as u64).wrapping_mul(0x94d0_49bb_1331_11eb))
        .wrapping_add(frame_bucket.wrapping_mul(0xa076_1d64_78bd_642f));

    let stride = if params.mv_bank_shuffle_every != 0
        && hash_u64(seed) % params.mv_bank_shuffle_every == 0
    {
        1 + (hash_u64(seed ^ 0xbf58_476d_1ce4_e5b9) as usize % (bank_count - 1)) as i32
    } else {
        params.mv_bank_stride
    };
    let source_bank = (bank_index as i64 + stride as i64).rem_euclid(bank_count as i64) as usize;
    let source_bank_x = source_bank % banks_x;
    let source_bank_y = source_bank / banks_x;
    let source_column = (source_bank_x * bank_size + local_x).min(columns - 1);
    let source_row = (source_bank_y * bank_size + local_y).min(rows - 1);
    let source_index = source_row * columns + source_column;

    packet.blocks.get(source_index)
}

fn blocks_per_row(width: usize, block_size: usize) -> usize {
    width.div_ceil(block_size).max(1)
}

fn block_contains(block: &MotionBlock, x: usize, y: usize) -> bool {
    x >= block.x && x < block.x + block.w && y >= block.y && y < block.y + block.h
}

fn predictor_from_packet(packet: &MoshPacket, block_index: usize, columns: usize) -> (i16, i16) {
    let mut sum_x = 0_i32;
    let mut sum_y = 0_i32;
    let mut count = 0_i32;

    if block_index % columns != 0 {
        let left = packet.blocks[block_index - 1];
        sum_x += left.dx as i32;
        sum_y += left.dy as i32;
        count += 1;
    }
    if block_index >= columns {
        let top = packet.blocks[block_index - columns];
        sum_x += top.dx as i32;
        sum_y += top.dy as i32;
        count += 1;
    }

    if count == 0 {
        (0, 0)
    } else {
        ((sum_x / count) as i16, (sum_y / count) as i16)
    }
}

fn predictor_from_decoded_vectors(
    decoded: &[(i16, i16)],
    block_index: usize,
    columns: usize,
) -> Option<(i16, i16)> {
    let mut sum_x = 0_i32;
    let mut sum_y = 0_i32;
    let mut count = 0_i32;

    if block_index % columns != 0 {
        let left = *decoded.get(block_index - 1)?;
        sum_x += left.0 as i32;
        sum_y += left.1 as i32;
        count += 1;
    }
    if block_index >= columns {
        let top = *decoded.get(block_index - columns)?;
        sum_x += top.0 as i32;
        sum_y += top.1 as i32;
        count += 1;
    }

    if count == 0 {
        Some((0, 0))
    } else {
        Some(((sum_x / count) as i16, (sum_y / count) as i16))
    }
}

fn interpolated_motion_vector(
    vectors: &[(i16, i16)],
    block_index: usize,
    base_dx: i16,
    base_dy: i16,
    width: usize,
    height: usize,
    block_size: usize,
    x: usize,
    y: usize,
    strength: f32,
) -> (f32, f32) {
    let strength = strength.clamp(0.0, 1.0);
    if strength <= 0.0 || vectors.is_empty() {
        return (base_dx as f32, base_dy as f32);
    }

    let columns = blocks_per_row(width, block_size);
    let rows = height.div_ceil(block_size).max(1);
    let column = (x / block_size).min(columns.saturating_sub(1));
    let row = (y / block_size).min(rows.saturating_sub(1));
    let next_column = (column + 1).min(columns.saturating_sub(1));
    let next_row = (row + 1).min(rows.saturating_sub(1));
    let fx = (x % block_size) as f32 / block_size as f32;
    let fy = (y % block_size) as f32 / block_size as f32;

    let v00 = vector_at(vectors, columns, row, column, block_index, base_dx, base_dy);
    let v10 = vector_at(
        vectors,
        columns,
        row,
        next_column,
        block_index,
        base_dx,
        base_dy,
    );
    let v01 = vector_at(
        vectors,
        columns,
        next_row,
        column,
        block_index,
        base_dx,
        base_dy,
    );
    let v11 = vector_at(
        vectors,
        columns,
        next_row,
        next_column,
        block_index,
        base_dx,
        base_dy,
    );

    let top_x = lerp(v00.0 as f32, v10.0 as f32, fx);
    let top_y = lerp(v00.1 as f32, v10.1 as f32, fx);
    let bottom_x = lerp(v01.0 as f32, v11.0 as f32, fx);
    let bottom_y = lerp(v01.1 as f32, v11.1 as f32, fx);
    let field_x = lerp(top_x, bottom_x, fy);
    let field_y = lerp(top_y, bottom_y, fy);

    (
        lerp(base_dx as f32, field_x, strength),
        lerp(base_dy as f32, field_y, strength),
    )
}

fn vector_at(
    vectors: &[(i16, i16)],
    columns: usize,
    row: usize,
    column: usize,
    fallback_index: usize,
    fallback_dx: i16,
    fallback_dy: i16,
) -> (i16, i16) {
    vectors
        .get(row * columns + column)
        .copied()
        .or_else(|| vectors.get(fallback_index).copied())
        .unwrap_or((fallback_dx, fallback_dy))
}

fn overlap_weight(block: &MotionBlock, x: usize, y: usize, overlap: usize) -> f32 {
    if overlap == 0 {
        return if block_contains(block, x, y) {
            1.0
        } else {
            0.0
        };
    }
    axis_overlap_weight(x, block.x, block.w, overlap)
        * axis_overlap_weight(y, block.y, block.h, overlap)
}

fn axis_overlap_weight(value: usize, start: usize, len: usize, overlap: usize) -> f32 {
    let end = start + len;
    if value < start {
        let distance = start - value;
        1.0 - distance as f32 / (overlap as f32 + 1.0)
    } else if value >= end {
        let distance = value - end + 1;
        1.0 - distance as f32 / (overlap as f32 + 1.0)
    } else {
        1.0
    }
    .clamp(0.0, 1.0)
}

pub fn encode_packet_bitstream(packet: &MoshPacket) -> io::Result<Vec<u8>> {
    let mut output = Vec::new();
    encode_packet_bitstream_into(packet, &mut output)?;
    Ok(output)
}

fn encode_packet_bitstream_into(packet: &MoshPacket, output: &mut Vec<u8>) -> io::Result<()> {
    if packet.width > u32::MAX as usize || packet.height > u32::MAX as usize {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "packet dimensions are too large for MSH0 bitstream",
        ));
    }
    if packet.blocks.len() > u32::MAX as usize
        || packet.residual.len() > u32::MAX as usize
        || packet.keyframe.len() > u32::MAX as usize
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "packet sections are too large for MSH0 bitstream",
        ));
    }

    let required = BITSTREAM_HEADER_LEN
        + packet.blocks.len() * BITSTREAM_BLOCK_LEN
        + packet.residual.len() * 2
        + packet.keyframe.len();
    output.clear();
    output.reserve(required);
    output.extend_from_slice(BITSTREAM_MAGIC);
    output.push(BITSTREAM_VERSION);
    output.push(match packet.kind {
        MoshPacketKind::I => 0,
        MoshPacketKind::P => 1,
    });
    output.extend_from_slice(&0_u16.to_le_bytes());
    write_u32(output, packet.width as u32);
    write_u32(output, packet.height as u32);
    write_u32(output, packet.blocks.len() as u32);
    write_u32(output, packet.residual.len() as u32);
    write_u32(output, packet.keyframe.len() as u32);

    for block in &packet.blocks {
        write_u16_checked(output, block.x, "block x")?;
        write_u16_checked(output, block.y, "block y")?;
        write_u16_checked(output, block.w, "block width")?;
        write_u16_checked(output, block.h, "block height")?;
        output.extend_from_slice(&block.dx.to_le_bytes());
        output.extend_from_slice(&block.dy.to_le_bytes());
    }

    for sample in &packet.residual {
        output.extend_from_slice(&sample.to_le_bytes());
    }
    output.extend_from_slice(&packet.keyframe);
    Ok(())
}

pub fn decode_packet_bitstream(bytes: &[u8]) -> io::Result<MoshPacket> {
    let mut packet = MoshPacket::default();
    decode_packet_bitstream_into(bytes, &mut packet)?;
    Ok(packet)
}

fn decode_packet_bitstream_into(bytes: &[u8], packet: &mut MoshPacket) -> io::Result<()> {
    let header = BitstreamHeader::read(bytes)?;
    let block_start = BITSTREAM_HEADER_LEN;
    let residual_start = header.residual_offset()?;
    let keyframe_start = header.keyframe_offset()?;
    let end = keyframe_start
        .checked_add(header.keyframe_len)
        .ok_or_else(|| invalid_data("MSH0 packet length overflow"))?;
    if bytes.len() < end {
        return Err(invalid_data("MSH0 packet is truncated"));
    }

    packet.kind = header.kind;
    packet.width = header.width;
    packet.height = header.height;
    packet.blocks.clear();
    packet.residual.clear();
    packet.keyframe.clear();
    packet.blocks.reserve(header.block_count);
    packet.residual.reserve(header.residual_len);
    packet.keyframe.reserve(header.keyframe_len);

    for index in 0..header.block_count {
        let offset = block_start + index * BITSTREAM_BLOCK_LEN;
        let x = read_u16_at(bytes, offset)? as usize;
        let y = read_u16_at(bytes, offset + 2)? as usize;
        let w = read_u16_at(bytes, offset + 4)? as usize;
        let h = read_u16_at(bytes, offset + 6)? as usize;
        let dx = read_i16_at(bytes, offset + 8)?;
        let dy = read_i16_at(bytes, offset + 10)?;
        if let Some(block) = sanitize_block(header.width, header.height, x, y, w, h, dx, dy) {
            packet.blocks.push(block);
        }
    }

    let residual_bytes = &bytes[residual_start..keyframe_start];
    for chunk in residual_bytes.chunks_exact(2) {
        packet
            .residual
            .push(i16::from_le_bytes([chunk[0], chunk[1]]));
    }

    packet
        .keyframe
        .extend_from_slice(&bytes[keyframe_start..end]);
    Ok(())
}

pub fn mutate_packet_bitstream(
    bytes: &mut [u8],
    params: &MoshBitstreamParams,
    seed: u64,
) -> io::Result<MoshBitstreamMutationStats> {
    if !params.enabled && !params.has_mutations() {
        return Ok(MoshBitstreamMutationStats::default());
    }

    let header = BitstreamHeader::read(bytes)?;
    if header.kind != MoshPacketKind::P {
        return Ok(MoshBitstreamMutationStats::default());
    }

    let residual_start = header.residual_offset()?;
    let residual_end = header.keyframe_offset()?;
    let mut stats = MoshBitstreamMutationStats::default();

    stats.entropy_slips +=
        mutate_entropy_slip_windows(bytes, residual_start, residual_end, params, seed)?;
    stats.coeff_blocks +=
        mutate_residual_coefficients(bytes, residual_start, &header, params, seed)?;

    for block_index in 0..header.block_count {
        let ordinal = block_index as u64 + 1;
        let block_offset = BITSTREAM_HEADER_LEN + block_index * BITSTREAM_BLOCK_LEN;

        if is_every_u64(params.mv_sign_flip_every, ordinal) {
            let dx = read_i16_at(bytes, block_offset + 8)?.saturating_neg();
            let dy = read_i16_at(bytes, block_offset + 10)?.saturating_neg();
            write_i16_at(bytes, block_offset + 8, dx)?;
            write_i16_at(bytes, block_offset + 10, dy)?;
            stats.mv_sign_flipped += 1;
        }

        if is_every_u64(params.mv_delta_every, ordinal) {
            let dx = read_i16_at(bytes, block_offset + 8)?.saturating_add(params.mv_delta_x);
            let dy = read_i16_at(bytes, block_offset + 10)?.saturating_add(params.mv_delta_y);
            write_i16_at(bytes, block_offset + 8, dx)?;
            write_i16_at(bytes, block_offset + 10, dy)?;
            stats.mv_delta_applied += 1;
        }

        if is_every_u64(params.block_address_shift_every, ordinal) {
            let x = read_u16_at(bytes, block_offset)? as usize;
            let y = read_u16_at(bytes, block_offset + 2)? as usize;
            let w = read_u16_at(bytes, block_offset + 4)? as usize;
            let h = read_u16_at(bytes, block_offset + 6)? as usize;
            let max_x = header.width.saturating_sub(w.max(1));
            let max_y = header.height.saturating_sub(h.max(1));
            let shifted_x = shift_index(x.min(max_x), params.block_address_shift_x, max_x + 1);
            let shifted_y = shift_index(y.min(max_y), params.block_address_shift_y, max_y + 1);
            write_u16_at(bytes, block_offset, shifted_x as u16)?;
            write_u16_at(bytes, block_offset + 2, shifted_y as u16)?;
            stats.block_addresses_shifted += 1;
        }

        if is_every_u64(params.residual_zero_every, ordinal) {
            let block = read_block_for_mutation(bytes, &header, block_offset)?;
            mutate_residual_block(bytes, residual_start, header.width, &block, |sample| {
                sample[0] = 0;
                sample[1] = 0;
            })?;
            stats.residual_blocks_zeroed += 1;
        }

        if is_every_u64(params.residual_xor_every, ordinal) {
            let block = read_block_for_mutation(bytes, &header, block_offset)?;
            let mask = params
                .residual_xor_mask
                .wrapping_add((seed as u8).wrapping_mul(17))
                .wrapping_add(block_index as u8);
            mutate_residual_block(bytes, residual_start, header.width, &block, |sample| {
                sample[0] ^= mask;
                sample[1] ^= mask.rotate_left(1);
            })?;
            stats.residual_blocks_xored += 1;
        }
    }

    Ok(stats)
}

fn mutate_residual_coefficients(
    bytes: &mut [u8],
    residual_start: usize,
    header: &BitstreamHeader,
    params: &MoshBitstreamParams,
    seed: u64,
) -> io::Result<u64> {
    if !is_every_u64(params.coeff_glitch_every, seed) {
        return Ok(0);
    }

    let size = coefficient_block_size(params.coeff_block_size);
    if header.width < size || header.height < size {
        return Ok(0);
    }

    let mut tile = vec![0_i32; size * size];
    let mut column = vec![0_i32; size];
    let mut mutated = 0;
    let mut tile_index = 0_u64;

    for y in (0..=header.height - size).step_by(size) {
        for x in (0..=header.width - size).step_by(size) {
            for channel in 0..CHANNELS {
                for ty in 0..size {
                    for tx in 0..size {
                        let offset = residual_sample_byte_offset(
                            residual_start,
                            header.width,
                            x + tx,
                            y + ty,
                            channel,
                        )?;
                        tile[ty * size + tx] = read_i16_at(bytes, offset)? as i32;
                    }
                }

                hadamard2d_with_column(&mut tile, size, &mut column);
                mutate_coeff_tile(&mut tile, size, tile_index, channel, seed, params);
                hadamard2d_with_column(&mut tile, size, &mut column);

                let denominator = (size * size) as i32;
                for ty in 0..size {
                    for tx in 0..size {
                        let offset = residual_sample_byte_offset(
                            residual_start,
                            header.width,
                            x + tx,
                            y + ty,
                            channel,
                        )?;
                        let value = round_div_i32(tile[ty * size + tx], denominator)
                            .clamp(i16::MIN as i32, i16::MAX as i32)
                            as i16;
                        write_i16_at(bytes, offset, value)?;
                    }
                }

                mutated += 1;
            }
            tile_index += 1;
        }
    }

    Ok(mutated)
}

fn mutate_coeff_tile(
    coeffs: &mut [i32],
    size: usize,
    tile_index: u64,
    channel: usize,
    seed: u64,
    params: &MoshBitstreamParams,
) {
    if coeffs.len() <= 1 {
        return;
    }

    if params.coeff_shift != 0 {
        let jitter = (hash_u64(
            seed ^ tile_index.wrapping_mul(0x9e37_79b9_7f4a_7c15)
                ^ (channel as u64).wrapping_mul(0xa076_1d64_78bd_642f),
        ) % 3) as i16
            - 1;
        rotate_i32_signed(&mut coeffs[1..], params.coeff_shift.saturating_add(jitter));
    }

    if params.coeff_sign_flip_every != 0 {
        let coeff_count = coeffs.len() as u64;
        for (index, coeff) in coeffs.iter_mut().enumerate().skip(1) {
            let ordinal = tile_index
                .wrapping_mul(coeff_count)
                .wrapping_add(index as u64)
                .wrapping_add((channel as u64).wrapping_mul(17))
                .wrapping_add(1);
            if is_every_u64(params.coeff_sign_flip_every, ordinal) {
                *coeff = coeff.saturating_neg();
            }
        }
    }

    if params.coeff_zero_high != 0 {
        for y in 0..size {
            for x in 0..size {
                if x == 0 && y == 0 {
                    continue;
                }
                if x + y >= params.coeff_zero_high {
                    coeffs[y * size + x] = 0;
                }
            }
        }
    }

    let quant = params.coeff_quant.max(1) as i32;
    if quant > 1 {
        for coeff in coeffs.iter_mut().skip(1) {
            *coeff = round_div_i32(*coeff, quant) * quant;
        }
    }
}

fn coefficient_block_size(value: usize) -> usize {
    match value {
        0..=4 => 4,
        5..=8 => 8,
        _ => 16,
    }
}

fn codebook_source_index(
    eligible: &[usize],
    tile_index: u64,
    frame_index: u64,
    params: &MoshBitstreamParams,
) -> usize {
    let seed = frame_index
        .wrapping_mul(0xa076_1d64_78bd_642f)
        .wrapping_add(tile_index.wrapping_mul(0xe703_7ed1_a0b4_28db));
    if params.codebook_shuffle_every != 0 && hash_u64(seed) % params.codebook_shuffle_every == 0 {
        return eligible[hash_u64(seed ^ 0x8ebc_6af0_9c88_c6e3) as usize % eligible.len()];
    }

    let index =
        (tile_index as i64 + params.codebook_stride as i64).rem_euclid(eligible.len() as i64);
    eligible[index as usize]
}

fn read_residual_tile(
    residual: &[i16],
    width: usize,
    x: usize,
    y: usize,
    tile_size: usize,
) -> io::Result<Vec<i16>> {
    let mut tile = Vec::with_capacity(tile_size * tile_size * CHANNELS);
    for ty in 0..tile_size {
        for tx in 0..tile_size {
            let index = residual_sample_index(width, x + tx, y + ty, 0)?;
            let pixel = residual
                .get(index..index + CHANNELS)
                .ok_or_else(|| invalid_data("MSH0 residual tile read exceeds section"))?;
            tile.extend_from_slice(pixel);
        }
    }
    Ok(tile)
}

fn write_residual_tile(
    residual: &mut [i16],
    width: usize,
    x: usize,
    y: usize,
    tile_size: usize,
    tile: &[i16],
) -> io::Result<()> {
    if tile.len() != tile_size * tile_size * CHANNELS {
        return Err(invalid_data("MSH0 residual codebook tile size mismatch"));
    }

    let mut src = 0;
    for ty in 0..tile_size {
        for tx in 0..tile_size {
            let index = residual_sample_index(width, x + tx, y + ty, 0)?;
            let pixel = residual
                .get_mut(index..index + CHANNELS)
                .ok_or_else(|| invalid_data("MSH0 residual tile write exceeds section"))?;
            pixel.copy_from_slice(&tile[src..src + CHANNELS]);
            src += CHANNELS;
        }
    }

    Ok(())
}

fn residual_sample_index(width: usize, x: usize, y: usize, channel: usize) -> io::Result<usize> {
    y.checked_mul(width)
        .and_then(|row| row.checked_add(x))
        .and_then(|pixel| pixel.checked_mul(CHANNELS))
        .and_then(|sample| sample.checked_add(channel))
        .ok_or_else(|| invalid_data("MSH0 residual sample index overflow"))
}

fn residual_sample_byte_offset(
    residual_start: usize,
    width: usize,
    x: usize,
    y: usize,
    channel: usize,
) -> io::Result<usize> {
    residual_start
        .checked_add(
            y.checked_mul(width)
                .and_then(|row| row.checked_add(x))
                .and_then(|pixel| pixel.checked_mul(CHANNELS))
                .and_then(|sample| sample.checked_add(channel))
                .and_then(|sample| sample.checked_mul(2))
                .ok_or_else(|| invalid_data("MSH0 residual sample index overflow"))?,
        )
        .ok_or_else(|| invalid_data("MSH0 residual byte offset overflow"))
}

fn hadamard2d_with_column(values: &mut [i32], size: usize, column: &mut [i32]) {
    for row in 0..size {
        let start = row * size;
        fwht_1d(&mut values[start..start + size]);
    }

    for x in 0..size {
        for y in 0..size {
            column[y] = values[y * size + x];
        }
        fwht_1d(column);
        for y in 0..size {
            values[y * size + x] = column[y];
        }
    }
}

fn fwht_1d(values: &mut [i32]) {
    let mut step = 1;
    while step < values.len() {
        let stride = step * 2;
        for base in (0..values.len()).step_by(stride) {
            for offset in 0..step {
                let a = values[base + offset];
                let b = values[base + offset + step];
                values[base + offset] = a.saturating_add(b);
                values[base + offset + step] = a.saturating_sub(b);
            }
        }
        step = stride;
    }
}

fn rotate_i32_signed(values: &mut [i32], amount: i16) {
    if values.len() <= 1 || amount == 0 {
        return;
    }

    let shift = amount.unsigned_abs() as usize % values.len();
    if shift == 0 {
        return;
    }
    if amount > 0 {
        values.rotate_right(shift);
    } else {
        values.rotate_left(shift);
    }
}

fn round_div_i32(value: i32, denominator: i32) -> i32 {
    if denominator <= 1 {
        return value;
    }
    if value >= 0 {
        (value + denominator / 2) / denominator
    } else {
        (value - denominator / 2) / denominator
    }
}

fn mutate_entropy_slip_windows(
    bytes: &mut [u8],
    residual_start: usize,
    residual_end: usize,
    params: &MoshBitstreamParams,
    seed: u64,
) -> io::Result<u64> {
    if !is_every_u64(params.entropy_slip_every, seed)
        || params.entropy_slip_bytes == 0
        || params.entropy_slip_windows == 0
    {
        return Ok(0);
    }

    let residual_len = residual_end
        .checked_sub(residual_start)
        .ok_or_else(|| invalid_data("MSH0 residual section is inverted"))?;
    if residual_len < 2 {
        return Ok(0);
    }

    let window_len = if params.entropy_resync_bytes == 0 {
        residual_len
    } else {
        params.entropy_resync_bytes.clamp(2, residual_len)
    };
    let max_start = residual_len.saturating_sub(window_len);
    let window_count = params.entropy_slip_windows;
    let mut mutated = 0;

    for window_index in 0..window_count {
        let seed = seed
            .wrapping_mul(0xa076_1d64_78bd_642f)
            .wrapping_add((window_index as u64).wrapping_mul(0xe703_7ed1_a0b4_28db));
        let local_start = if max_start == 0 {
            0
        } else {
            hash_u64(seed) as usize % (max_start + 1)
        };
        let start = residual_start
            .checked_add(local_start)
            .ok_or_else(|| invalid_data("MSH0 entropy slip start overflow"))?;
        let end = start
            .checked_add(window_len)
            .ok_or_else(|| invalid_data("MSH0 entropy slip end overflow"))?;
        let window = bytes
            .get_mut(start..end)
            .ok_or_else(|| invalid_data("MSH0 entropy slip window exceeds residual section"))?;
        rotate_bytes_signed(window, params.entropy_slip_bytes);
        mutated += 1;
    }

    Ok(mutated)
}

#[derive(Debug, Clone, Copy)]
struct BitstreamHeader {
    kind: MoshPacketKind,
    width: usize,
    height: usize,
    block_count: usize,
    residual_len: usize,
    keyframe_len: usize,
}

impl BitstreamHeader {
    fn read(bytes: &[u8]) -> io::Result<Self> {
        if bytes.len() < BITSTREAM_HEADER_LEN {
            return Err(invalid_data("MSH0 header is truncated"));
        }
        if &bytes[..4] != BITSTREAM_MAGIC {
            return Err(invalid_data("MSH0 magic is missing"));
        }
        if bytes[4] != BITSTREAM_VERSION {
            return Err(invalid_data("unsupported MSH0 version"));
        }

        let kind = match bytes[5] {
            0 => MoshPacketKind::I,
            _ => MoshPacketKind::P,
        };
        let width = read_u32_at(bytes, 8)? as usize;
        let height = read_u32_at(bytes, 12)? as usize;
        let block_count = read_u32_at(bytes, 16)? as usize;
        let residual_len = read_u32_at(bytes, 20)? as usize;
        let keyframe_len = read_u32_at(bytes, 24)? as usize;

        if width == 0 || height == 0 {
            return Err(invalid_data("MSH0 dimensions must be non-zero"));
        }
        let max_blocks = width
            .checked_mul(height)
            .and_then(|pixels| pixels.checked_add(1))
            .ok_or_else(|| invalid_data("MSH0 dimensions overflow"))?;
        if block_count > max_blocks {
            return Err(invalid_data("MSH0 block count is implausibly large"));
        }

        let block_bytes = block_count
            .checked_mul(BITSTREAM_BLOCK_LEN)
            .ok_or_else(|| invalid_data("MSH0 block table length overflow"))?;
        let residual_bytes = residual_len
            .checked_mul(2)
            .ok_or_else(|| invalid_data("MSH0 residual length overflow"))?;
        let min_len = BITSTREAM_HEADER_LEN
            .checked_add(block_bytes)
            .and_then(|len| len.checked_add(residual_bytes))
            .and_then(|len| len.checked_add(keyframe_len))
            .ok_or_else(|| invalid_data("MSH0 packet length overflow"))?;
        if bytes.len() < min_len {
            return Err(invalid_data("MSH0 packet is shorter than section lengths"));
        }

        Ok(Self {
            kind,
            width,
            height,
            block_count,
            residual_len,
            keyframe_len,
        })
    }

    fn residual_offset(self) -> io::Result<usize> {
        self.block_count
            .checked_mul(BITSTREAM_BLOCK_LEN)
            .and_then(|len| BITSTREAM_HEADER_LEN.checked_add(len))
            .ok_or_else(|| invalid_data("MSH0 residual offset overflow"))
    }

    fn keyframe_offset(self) -> io::Result<usize> {
        self.residual_len
            .checked_mul(2)
            .and_then(|len| self.residual_offset().ok()?.checked_add(len))
            .ok_or_else(|| invalid_data("MSH0 keyframe offset overflow"))
    }
}

fn sanitize_block(
    width: usize,
    height: usize,
    x: usize,
    y: usize,
    w: usize,
    h: usize,
    dx: i16,
    dy: i16,
) -> Option<MotionBlock> {
    if width == 0 || height == 0 {
        return None;
    }
    let x = x.min(width - 1);
    let y = y.min(height - 1);
    let w = w.max(1).min(width - x);
    let h = h.max(1).min(height - y);
    Some(MotionBlock { x, y, w, h, dx, dy })
}

fn read_block_for_mutation(
    bytes: &[u8],
    header: &BitstreamHeader,
    offset: usize,
) -> io::Result<MotionBlock> {
    let x = read_u16_at(bytes, offset)? as usize;
    let y = read_u16_at(bytes, offset + 2)? as usize;
    let w = read_u16_at(bytes, offset + 4)? as usize;
    let h = read_u16_at(bytes, offset + 6)? as usize;
    let dx = read_i16_at(bytes, offset + 8)?;
    let dy = read_i16_at(bytes, offset + 10)?;
    sanitize_block(header.width, header.height, x, y, w, h, dx, dy)
        .ok_or_else(|| invalid_data("MSH0 block cannot be sanitized"))
}

fn mutate_residual_block(
    bytes: &mut [u8],
    residual_start: usize,
    width: usize,
    block: &MotionBlock,
    mut mutate: impl FnMut(&mut [u8]),
) -> io::Result<()> {
    for by in 0..block.h {
        for bx in 0..block.w {
            let pixel = (block.y + by)
                .checked_mul(width)
                .and_then(|row| row.checked_add(block.x + bx))
                .ok_or_else(|| invalid_data("MSH0 residual pixel index overflow"))?;
            for channel in 0..CHANNELS {
                let sample_offset = residual_start
                    .checked_add(
                        pixel
                            .checked_mul(CHANNELS)
                            .and_then(|index| index.checked_add(channel))
                            .and_then(|index| index.checked_mul(2))
                            .ok_or_else(|| invalid_data("MSH0 residual sample index overflow"))?,
                    )
                    .ok_or_else(|| invalid_data("MSH0 residual byte offset overflow"))?;
                let sample = bytes
                    .get_mut(sample_offset..sample_offset + 2)
                    .ok_or_else(|| invalid_data("MSH0 residual block exceeds section"))?;
                mutate(sample);
            }
        }
    }

    Ok(())
}

fn write_u16_checked(output: &mut Vec<u8>, value: usize, name: &str) -> io::Result<()> {
    if value > u16::MAX as usize {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("{name} is too large for MSH0 block table"),
        ));
    }
    output.extend_from_slice(&(value as u16).to_le_bytes());
    Ok(())
}

fn write_u32(output: &mut Vec<u8>, value: u32) {
    output.extend_from_slice(&value.to_le_bytes());
}

fn read_u16_at(bytes: &[u8], offset: usize) -> io::Result<u16> {
    let data = bytes
        .get(offset..offset + 2)
        .ok_or_else(|| invalid_data("MSH0 u16 read out of range"))?;
    Ok(u16::from_le_bytes([data[0], data[1]]))
}

fn read_u32_at(bytes: &[u8], offset: usize) -> io::Result<u32> {
    let data = bytes
        .get(offset..offset + 4)
        .ok_or_else(|| invalid_data("MSH0 u32 read out of range"))?;
    Ok(u32::from_le_bytes([data[0], data[1], data[2], data[3]]))
}

fn read_i16_at(bytes: &[u8], offset: usize) -> io::Result<i16> {
    let data = bytes
        .get(offset..offset + 2)
        .ok_or_else(|| invalid_data("MSH0 i16 read out of range"))?;
    Ok(i16::from_le_bytes([data[0], data[1]]))
}

fn write_u16_at(bytes: &mut [u8], offset: usize, value: u16) -> io::Result<()> {
    let data = bytes
        .get_mut(offset..offset + 2)
        .ok_or_else(|| invalid_data("MSH0 u16 write out of range"))?;
    data.copy_from_slice(&value.to_le_bytes());
    Ok(())
}

fn write_i16_at(bytes: &mut [u8], offset: usize, value: i16) -> io::Result<()> {
    let data = bytes
        .get_mut(offset..offset + 2)
        .ok_or_else(|| invalid_data("MSH0 i16 write out of range"))?;
    data.copy_from_slice(&value.to_le_bytes());
    Ok(())
}

fn shift_index(value: usize, delta: i16, span: usize) -> usize {
    if span == 0 {
        return 0;
    }
    (value as isize + delta as isize).rem_euclid(span as isize) as usize
}

fn rotate_bytes_signed(bytes: &mut [u8], amount: i16) {
    if bytes.len() <= 1 || amount == 0 {
        return;
    }

    let shift = amount.unsigned_abs() as usize % bytes.len();
    if shift == 0 {
        return;
    }
    if amount > 0 {
        bytes.rotate_right(shift);
    } else {
        bytes.rotate_left(shift);
    }
}

fn is_every_u64(interval: u64, count: u64) -> bool {
    interval != 0 && count % interval == 0
}

pub(crate) fn codec_thread_pool() -> Option<&'static ThreadPool> {
    static POOL: OnceLock<Option<ThreadPool>> = OnceLock::new();
    POOL.get_or_init(|| {
        let available = std::thread::available_parallelism()
            .map(usize::from)
            .unwrap_or(1);
        let default_threads = (available / 2).clamp(1, 16);
        let threads = std::env::var("DATAMOSH_THREADS")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(default_threads)
            .clamp(1, available);
        if threads <= 1 {
            None
        } else {
            ThreadPoolBuilder::new().num_threads(threads).build().ok()
        }
    })
    .as_ref()
}

fn push_bounded_history_copy(history: &mut VecDeque<Vec<u8>>, limit: usize, frame: &[u8]) {
    let mut buffer = if history.len() >= limit {
        history.pop_front().unwrap_or_default()
    } else {
        Vec::with_capacity(frame.len())
    };
    buffer.clear();
    buffer.extend_from_slice(frame);
    history.push_back(buffer);
}

fn history_frame(history: &VecDeque<Vec<u8>>, lag: usize) -> Option<&[u8]> {
    if history.is_empty() {
        return None;
    }
    let lag = lag.max(1).min(history.len());
    history.get(history.len() - lag).map(Vec::as_slice)
}

fn invalid_data(message: &'static str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message)
}

fn block_activity_score(packet: &MoshPacket, block: &MotionBlock) -> u32 {
    let mut residual_sum = 0_u64;
    let mut samples = 0_u64;

    for by in 0..block.h {
        for bx in 0..block.w {
            let index = rgb_index(packet.width, block.x + bx, block.y + by);
            for channel in 0..CHANNELS {
                residual_sum += packet.residual[index + channel].unsigned_abs() as u64;
                samples += 1;
            }
        }
    }

    let residual_avg = if samples == 0 {
        0
    } else {
        residual_sum / samples
    };
    let motion = block.dx.unsigned_abs() as u32 + block.dy.unsigned_abs() as u32;
    residual_avg as u32 + motion.saturating_mul(4)
}

fn pixel_activity_score(packet: &MoshPacket, block: &MotionBlock, x: usize, y: usize) -> u32 {
    let index = rgb_index(packet.width, x, y);
    let residual = (packet.residual[index].unsigned_abs() as u32
        + packet.residual[index + 1].unsigned_abs() as u32
        + packet.residual[index + 2].unsigned_abs() as u32)
        / CHANNELS as u32;
    residual + vector_cost(block.dx, block.dy).saturating_mul(2)
}

fn activity_mix(score: u32, params: &MoshGlitchParams) -> f32 {
    match params.activity_mode {
        ActivityMode::All => 1.0,
        ActivityMode::Active => {
            score_to_mix(score, params.activity_threshold, params.activity_softness)
        }
        ActivityMode::Static => {
            1.0 - score_to_mix(score, params.activity_threshold, params.activity_softness)
        }
    }
}

fn reference_mix(activity: f32, params: &MoshGlitchParams) -> f32 {
    let bleed = params.reference_bleed.clamp(0.0, 1.0);
    bleed + activity.clamp(0.0, 1.0) * (1.0 - bleed)
}

fn residual_sample(
    packet: &MoshPacket,
    x: usize,
    y: usize,
    channel: usize,
    channel_shift: i16,
    residual_keep: f32,
    invert: bool,
) -> f32 {
    if residual_keep == 0.0 {
        return 0.0;
    }

    let channel = shifted_residual_channel(channel, channel_shift);
    let index = rgb_index(packet.width, x, y) + channel;
    let mut residual = packet.residual[index] as f32 * residual_keep;
    if invert {
        residual = -residual;
    }
    residual
}

fn residual_address(
    packet: &MoshPacket,
    x: usize,
    y: usize,
    block_index: usize,
    frame_index: u64,
    params: &MoshGlitchParams,
) -> (usize, usize) {
    let (source_x, source_y) =
        residual_bank_address(packet, x, y, block_index, frame_index, params).unwrap_or((x, y));
    let mut shift_x = params.residual_address_shift_x as isize;
    let mut shift_y = params.residual_address_shift_y as isize;
    let jitter = params.residual_address_jitter.max(0) as isize;
    if jitter != 0 {
        let latch = params.reference_latch_frames.max(1);
        let frame_bucket = frame_index / latch;
        let seed = sample_cell_seed(
            block_index,
            x,
            y,
            frame_bucket,
            glitch_cell_x(params, 1),
            glitch_cell_y(params, 1),
        );
        shift_x += (signed_hash_unit(seed) * jitter as f32).round() as isize;
        shift_y +=
            (signed_hash_unit(seed ^ 0xe703_7ed1_a0b4_28db) * jitter as f32).round() as isize;
    }

    (
        clamp_coord(source_x as isize + shift_x, packet.width),
        clamp_coord(source_y as isize + shift_y, packet.height),
    )
}

fn residual_bank_address(
    packet: &MoshPacket,
    x: usize,
    y: usize,
    block_index: usize,
    frame_index: u64,
    params: &MoshGlitchParams,
) -> Option<(usize, usize)> {
    if params.residual_bank_size == 0
        || (params.residual_bank_stride == 0 && params.residual_bank_shuffle_every == 0)
    {
        return None;
    }

    let bank_size = params.residual_bank_size.max(1);
    let banks_x = packet.width.div_ceil(bank_size).max(1);
    let banks_y = packet.height.div_ceil(bank_size).max(1);
    let bank_count = banks_x.checked_mul(banks_y)?;
    if bank_count <= 1 {
        return None;
    }

    let bank_x = (x / bank_size).min(banks_x - 1);
    let bank_y = (y / bank_size).min(banks_y - 1);
    let local_x = x % bank_size;
    let local_y = y % bank_size;
    let bank_index = bank_y * banks_x + bank_x;
    let latch = params.reference_latch_frames.max(1);
    let frame_bucket = frame_index / latch;
    let seed = (block_index as u64)
        .wrapping_mul(0x9e37_79b9_7f4a_7c15)
        .wrapping_add((bank_index as u64).wrapping_mul(0x94d0_49bb_1331_11eb))
        .wrapping_add(frame_bucket.wrapping_mul(0xbf58_476d_1ce4_e5b9));

    let stride = if params.residual_bank_shuffle_every != 0
        && hash_u64(seed) % params.residual_bank_shuffle_every == 0
    {
        1 + (hash_u64(seed ^ 0xa076_1d64_78bd_642f) as usize % (bank_count - 1)) as i32
    } else {
        params.residual_bank_stride
    };
    let source_bank = (bank_index as i64 + stride as i64).rem_euclid(bank_count as i64) as usize;
    let source_bank_x = source_bank % banks_x;
    let source_bank_y = source_bank / banks_x;
    let source_x = (source_bank_x * bank_size + local_x).min(packet.width - 1);
    let source_y = (source_bank_y * bank_size + local_y).min(packet.height - 1);

    Some((source_x, source_y))
}

fn residual_channel_shift(block_index: usize, frame_index: u64, params: &MoshGlitchParams) -> i16 {
    let mut shift = params.residual_channel_shift as i32;
    if shift == 0 {
        return 0;
    }
    if params.reference_latch_frames > 1 {
        let frame_bucket = frame_index / params.reference_latch_frames.max(1);
        let seed = (block_index as u64)
            .wrapping_mul(0x9e37_79b9_7f4a_7c15)
            .wrapping_add(frame_bucket.wrapping_mul(0xbf58_476d_1ce4_e5b9));
        shift += (hash_u64(seed) % CHANNELS as u64) as i32;
    }
    shift.rem_euclid(CHANNELS as i32) as i16
}

fn shifted_residual_channel(channel: usize, shift: i16) -> usize {
    shifted_channel(channel, shift)
}

fn shifted_channel(channel: usize, shift: i16) -> usize {
    (channel as i32 + shift as i32).rem_euclid(CHANNELS as i32) as usize
}

fn reference_switch(
    chance: f32,
    block_index: usize,
    x: usize,
    y: usize,
    frame_index: u64,
    params: &MoshGlitchParams,
) -> bool {
    let chance = chance.clamp(0.0, 1.0);
    if chance <= 0.0 {
        return false;
    }
    if chance >= 1.0 {
        return true;
    }

    let latch = params.reference_latch_frames.max(1);
    let frame_bucket = frame_index / latch;
    let block_seed = sample_cell_seed(
        block_index,
        x,
        y,
        frame_bucket,
        glitch_cell_x(params, REFERENCE_SWITCH_CELL_X),
        glitch_cell_y(params, REFERENCE_SWITCH_CELL_Y),
    );
    hash_unit(block_seed) < chance
}

fn sample_address_offset(
    block_index: usize,
    x: usize,
    y: usize,
    frame_index: u64,
    params: &MoshGlitchParams,
) -> (f32, f32) {
    let amount = params.sample_address_desync.clamp(0.0, 16.0);
    if amount <= 0.0 {
        return (0.0, 0.0);
    }

    let latch = params.reference_latch_frames.max(1);
    let frame_bucket = frame_index / latch;
    let seed = sample_cell_seed(
        block_index,
        x,
        y,
        frame_bucket,
        glitch_cell_x(params, SAMPLE_ADDRESS_CELL_X),
        glitch_cell_y(params, SAMPLE_ADDRESS_CELL_Y),
    );
    (
        signed_hash_unit(seed) * amount,
        signed_hash_unit(seed ^ 0xa076_1d64_78bd_642f) * amount * 0.45,
    )
}

fn glitch_cell_x(params: &MoshGlitchParams, default: usize) -> usize {
    if params.glitch_cell_width == 0 {
        glitch_cell(params, default)
    } else {
        params.glitch_cell_width
    }
}

fn glitch_cell_y(params: &MoshGlitchParams, default: usize) -> usize {
    if params.glitch_cell_height == 0 {
        glitch_cell(params, default)
    } else {
        params.glitch_cell_height
    }
}

fn glitch_cell(params: &MoshGlitchParams, default: usize) -> usize {
    if params.glitch_cell_size == 0 {
        default.max(1)
    } else {
        params.glitch_cell_size
    }
}

fn sample_cell_seed(
    block_index: usize,
    x: usize,
    y: usize,
    frame_bucket: u64,
    cell_x: usize,
    cell_y: usize,
) -> u64 {
    (block_index as u64)
        .wrapping_mul(0x9e37_79b9_7f4a_7c15)
        .wrapping_add(frame_bucket.wrapping_mul(0xbf58_476d_1ce4_e5b9))
        .wrapping_add(((x / cell_x.max(1)) as u64).wrapping_mul(0x94d0_49bb_1331_11eb))
        .wrapping_add(((y / cell_y.max(1)) as u64).wrapping_mul(0x2545_f491_4f6c_dd1d))
}

fn hash_unit(mut value: u64) -> f32 {
    value = hash_u64(value);
    ((value >> 40) as f32) / ((1_u32 << 24) as f32)
}

fn signed_hash_unit(value: u64) -> f32 {
    hash_unit(value) * 2.0 - 1.0
}

fn hash_u64(mut value: u64) -> u64 {
    value ^= value >> 30;
    value = value.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value ^= value >> 27;
    value = value.wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^= value >> 31;
    value
}

fn score_to_mix(score: u32, threshold: u16, softness: u16) -> f32 {
    if softness == 0 {
        return if score >= threshold as u32 { 1.0 } else { 0.0 };
    }
    let x = ((score as f32 - threshold as f32) / softness as f32).clamp(0.0, 1.0);
    x * x * (3.0 - 2.0 * x)
}

fn scaled_vector(value: i16, scale: f32) -> i16 {
    (value as f32 * scale)
        .round()
        .clamp(i16::MIN as f32, i16::MAX as f32) as i16
}

fn vector_cost(dx: i16, dy: i16) -> u32 {
    dx.unsigned_abs() as u32 + dy.unsigned_abs() as u32
}

fn quantize_i16(value: i16, quantum: i16) -> i16 {
    let quantum = quantum.max(1);
    ((value as f32 / quantum as f32).round() as i16).saturating_mul(quantum)
}

fn jitter(index: usize, axis: u64, amount: i16) -> i16 {
    let span = amount as i32 * 2 + 1;
    let value = (index as u64)
        .wrapping_mul(1_103_515_245)
        .wrapping_add(axis.wrapping_mul(2_654_435_761))
        .wrapping_add(12_345);
    (value % span as u64) as i16 - amount
}

fn sample_reference(
    reference: &[u8],
    width: usize,
    height: usize,
    x: f32,
    y: f32,
    channel: usize,
    wrap: bool,
) -> f32 {
    let x0_raw = x.floor();
    let y0_raw = y.floor();
    let fx = x - x0_raw;
    let fy = y - y0_raw;
    let x0 = sample_coord(x0_raw as isize, width, wrap);
    let x1 = sample_coord(x0_raw as isize + 1, width, wrap);
    let y0 = sample_coord(y0_raw as isize, height, wrap);
    let y1 = sample_coord(y0_raw as isize + 1, height, wrap);

    let p00 = reference[rgb_index(width, x0, y0) + channel] as f32;
    let p10 = reference[rgb_index(width, x1, y0) + channel] as f32;
    let p01 = reference[rgb_index(width, x0, y1) + channel] as f32;
    let p11 = reference[rgb_index(width, x1, y1) + channel] as f32;
    let top = lerp(p00, p10, fx);
    let bottom = lerp(p01, p11, fx);
    lerp(top, bottom, fy)
}

fn sample_coord(value: isize, max: usize, wrap: bool) -> usize {
    if wrap {
        wrap_coord(value, max)
    } else {
        clamp_coord(value, max)
    }
}

fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t.clamp(0.0, 1.0)
}

fn rgb_index(width: usize, x: usize, y: usize) -> usize {
    (y * width + x) * CHANNELS
}

fn luma(r: u8, g: u8, b: u8) -> u8 {
    ((77_u16 * r as u16 + 150_u16 * g as u16 + 29_u16 * b as u16) >> 8) as u8
}

fn clamp_coord(value: isize, max: usize) -> usize {
    value.clamp(0, max.saturating_sub(1) as isize) as usize
}

fn wrap_coord(value: isize, max: usize) -> usize {
    value.rem_euclid(max as isize) as usize
}

#[cfg(test)]
mod tests {
    use super::*;

    fn codec(width: usize, height: usize) -> MoshCodec {
        MoshCodec::new(MoshCodecConfig {
            width,
            height,
            block_size: 2,
            search_radius: 2,
            search_step: 1,
            keyframe_interval: 0,
            history_len: 4,
            ..MoshCodecConfig::default()
        })
        .unwrap()
    }

    #[test]
    fn keyframe_outputs_input_frame() {
        let mut codec = codec(2, 2);
        let input = vec![10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120];
        let mut output = vec![0; input.len()];

        codec
            .process_rgb_frame(&input, &MoshGlitchParams::default(), &mut output)
            .unwrap();

        assert_eq!(output, input);
        assert_eq!(codec.stats().keyframes, 1);
    }

    #[test]
    fn default_predicted_frame_reconstructs_input_with_residual() {
        let mut codec = codec(4, 2);
        let first = vec![
            0, 0, 0, 10, 0, 0, 20, 0, 0, 30, 0, 0, 40, 0, 0, 50, 0, 0, 60, 0, 0, 70, 0, 0,
        ];
        let second = vec![
            10, 0, 0, 20, 0, 0, 30, 0, 0, 40, 0, 0, 50, 0, 0, 60, 0, 0, 70, 0, 0, 80, 0, 0,
        ];
        let mut output = vec![0; first.len()];

        codec
            .process_rgb_frame(&first, &MoshGlitchParams::default(), &mut output)
            .unwrap();
        codec
            .process_rgb_frame(&second, &MoshGlitchParams::default(), &mut output)
            .unwrap();

        assert_eq!(output, second);
        assert_eq!(codec.stats().predicted_frames, 1);
    }

    #[test]
    fn reset_glitch_state_makes_next_frame_a_fresh_keyframe() {
        let mut codec = codec(4, 2);
        let first = vec![
            0, 0, 0, 10, 0, 0, 20, 0, 0, 30, 0, 0, 40, 0, 0, 50, 0, 0, 60, 0, 0, 70, 0, 0,
        ];
        let second = vec![
            10, 0, 0, 20, 0, 0, 30, 0, 0, 40, 0, 0, 50, 0, 0, 60, 0, 0, 70, 0, 0, 80, 0, 0,
        ];
        let third = vec![
            20, 0, 0, 30, 0, 0, 40, 0, 0, 50, 0, 0, 60, 0, 0, 70, 0, 0, 80, 0, 0, 90, 0, 0,
        ];
        let mut output = vec![0; first.len()];

        codec
            .process_rgb_frame(&first, &MoshGlitchParams::default(), &mut output)
            .unwrap();
        codec
            .process_rgb_frame(&second, &MoshGlitchParams::default(), &mut output)
            .unwrap();

        codec.reset_glitch_state();
        codec
            .process_rgb_frame(&third, &MoshGlitchParams::default(), &mut output)
            .unwrap();

        assert_eq!(output, third);
        assert_eq!(codec.stats().keyframes, 2);
        assert_eq!(codec.stats().predicted_frames, 1);
    }

    #[test]
    fn dropping_residual_creates_motion_smeared_prediction() {
        let mut codec = codec(4, 2);
        let first = vec![
            0, 0, 0, 10, 0, 0, 20, 0, 0, 30, 0, 0, 40, 0, 0, 50, 0, 0, 60, 0, 0, 70, 0, 0,
        ];
        let second = vec![
            10, 0, 0, 20, 0, 0, 30, 0, 0, 40, 0, 0, 50, 0, 0, 60, 0, 0, 70, 0, 0, 80, 0, 0,
        ];
        let mut output = vec![0; first.len()];

        codec
            .process_rgb_frame(&first, &MoshGlitchParams::default(), &mut output)
            .unwrap();
        codec
            .process_rgb_frame(
                &second,
                &MoshGlitchParams {
                    residual_keep: 0.0,
                    ..MoshGlitchParams::default()
                },
                &mut output,
            )
            .unwrap();

        assert_ne!(output, second);
    }

    #[test]
    fn block_remap_changes_prediction_path() {
        let params = MoshGlitchParams {
            residual_keep: 0.0,
            block_remap_every: 1,
            block_remap_stride: 1,
            ..MoshGlitchParams::default()
        };
        let mut codec = codec(4, 4);
        let first: Vec<u8> = (0..48).map(|v| (v * 3) as u8).collect();
        let second: Vec<u8> = (0..48).map(|v| (255 - v * 3) as u8).collect();
        let mut output_a = vec![0; first.len()];
        let mut output_b = vec![0; first.len()];

        codec
            .process_rgb_frame(&first, &MoshGlitchParams::default(), &mut output_a)
            .unwrap();
        let packet = codec.encode_rgb_packet(&second).unwrap();
        codec
            .decode_rgb_packet(
                &packet,
                &MoshGlitchParams {
                    residual_keep: 0.0,
                    ..MoshGlitchParams::default()
                },
                &mut output_a,
            )
            .unwrap();
        codec
            .decode_rgb_packet(&packet, &params, &mut output_b)
            .unwrap();

        assert_ne!(output_a, output_b);
    }

    #[test]
    fn active_activity_mode_glitches_only_changed_blocks() {
        let mut codec = codec(4, 4);
        let first = vec![10_u8; 4 * 4 * 3];
        let mut second = first.clone();
        for y in 0..2 {
            for x in 0..2 {
                let index = rgb_index(4, x, y);
                second[index] = 200;
                second[index + 1] = 200;
                second[index + 2] = 200;
            }
        }
        let mut output = vec![0; first.len()];

        codec
            .process_rgb_frame(&first, &MoshGlitchParams::default(), &mut output)
            .unwrap();
        codec
            .process_rgb_frame(
                &second,
                &MoshGlitchParams {
                    residual_keep: 0.0,
                    activity_mode: ActivityMode::Active,
                    activity_threshold: 12,
                    ..MoshGlitchParams::default()
                },
                &mut output,
            )
            .unwrap();

        assert_ne!(&output[0..12], &second[0..12]);
        let unchanged_index = rgb_index(4, 3, 3);
        assert_eq!(
            &output[unchanged_index..unchanged_index + 3],
            &second[unchanged_index..unchanged_index + 3]
        );
    }

    #[test]
    fn soft_activity_hard_switches_without_alpha_blending() {
        let mut codec = codec(4, 4);
        let first = vec![10_u8; 4 * 4 * 3];
        let mut second = first.clone();
        for y in 0..2 {
            for x in 0..2 {
                let index = rgb_index(4, x, y);
                second[index] = 200;
                second[index + 1] = 200;
                second[index + 2] = 200;
            }
        }
        let mut output = vec![0; first.len()];

        codec
            .process_rgb_frame(&first, &MoshGlitchParams::default(), &mut output)
            .unwrap();
        codec
            .process_rgb_frame(
                &second,
                &MoshGlitchParams {
                    residual_keep: 0.0,
                    activity_mode: ActivityMode::Active,
                    activity_threshold: 12,
                    activity_softness: 200,
                    ..MoshGlitchParams::default()
                },
                &mut output,
            )
            .unwrap();

        assert!(output[0] == 10 || output[0] == 200);
        let unchanged_index = rgb_index(4, 3, 3);
        assert_eq!(output[unchanged_index], 10);
    }

    #[test]
    fn overlap_blends_neighbor_block_predictions_at_boundaries() {
        let mut codec = codec(4, 2);
        let first = vec![
            0, 0, 0, 50, 0, 0, 100, 0, 0, 150, 0, 0, 0, 0, 0, 50, 0, 0, 100, 0, 0, 150, 0, 0,
        ];
        let packet = MoshPacket {
            kind: MoshPacketKind::P,
            width: 4,
            height: 2,
            blocks: vec![
                MotionBlock {
                    x: 0,
                    y: 0,
                    w: 2,
                    h: 2,
                    dx: 1,
                    dy: 0,
                },
                MotionBlock {
                    x: 2,
                    y: 0,
                    w: 2,
                    h: 2,
                    dx: 0,
                    dy: 0,
                },
            ],
            residual: vec![0; first.len()],
            keyframe: Vec::new(),
        };
        let mut output_plain = vec![0; first.len()];
        let mut output_overlap = vec![0; first.len()];

        codec
            .process_rgb_frame(&first, &MoshGlitchParams::default(), &mut output_plain)
            .unwrap();
        codec
            .decode_rgb_packet(
                &packet,
                &MoshGlitchParams {
                    residual_keep: 0.0,
                    activity_mode: ActivityMode::All,
                    mv_scale_x: 2.0,
                    ..MoshGlitchParams::default()
                },
                &mut output_plain,
            )
            .unwrap();
        codec
            .decode_rgb_packet(
                &packet,
                &MoshGlitchParams {
                    residual_keep: 0.0,
                    activity_mode: ActivityMode::All,
                    mv_scale_x: 2.0,
                    overlap: 1,
                    ..MoshGlitchParams::default()
                },
                &mut output_overlap,
            )
            .unwrap();

        let boundary_index = rgb_index(4, 2, 0);
        assert!(output_overlap[boundary_index] > output_plain[boundary_index]);
    }

    #[test]
    fn split_reference_mode_keeps_encoder_history_clean_after_glitch() {
        let mut codec = codec(4, 4);
        let first = vec![10_u8; 4 * 4 * 3];
        let mut second = first.clone();
        for y in 0..2 {
            for x in 0..2 {
                let index = rgb_index(4, x, y);
                second[index] = 200;
                second[index + 1] = 200;
                second[index + 2] = 200;
            }
        }
        let params = MoshGlitchParams {
            residual_keep: 0.0,
            activity_mode: ActivityMode::Active,
            activity_threshold: 12,
            ..MoshGlitchParams::default()
        };
        let mut output = vec![0; first.len()];

        codec
            .process_rgb_frame(&first, &MoshGlitchParams::default(), &mut output)
            .unwrap();
        codec
            .process_rgb_frame(&second, &params, &mut output)
            .unwrap();
        assert_ne!(&output[0..12], &second[0..12]);

        codec
            .process_rgb_frame(&second, &params, &mut output)
            .unwrap();

        assert_eq!(output, second);
    }

    #[test]
    fn reference_bleed_keeps_dirty_prediction_after_motion_stops() {
        let mut codec = codec(4, 4);
        let first = vec![10_u8; 4 * 4 * 3];
        let mut second = first.clone();
        for y in 0..2 {
            for x in 0..2 {
                let index = rgb_index(4, x, y);
                second[index] = 200;
                second[index + 1] = 200;
                second[index + 2] = 200;
            }
        }
        let params = MoshGlitchParams {
            residual_keep: 0.0,
            reference_bleed: 1.0,
            activity_mode: ActivityMode::Active,
            activity_threshold: 12,
            ..MoshGlitchParams::default()
        };
        let mut output = vec![0; first.len()];

        codec
            .process_rgb_frame(&first, &MoshGlitchParams::default(), &mut output)
            .unwrap();
        codec
            .process_rgb_frame(&second, &params, &mut output)
            .unwrap();
        codec
            .process_rgb_frame(&second, &params, &mut output)
            .unwrap();

        assert_eq!(output[0], 10);
        let unchanged_index = rgb_index(4, 3, 3);
        assert_eq!(output[unchanged_index], 10);
    }

    #[test]
    fn reference_switch_is_hard_and_latched() {
        let params = MoshGlitchParams {
            reference_latch_frames: 8,
            ..MoshGlitchParams::default()
        };

        assert!(!reference_switch(0.0, 3, 16, 16, 10, &params));
        assert!(reference_switch(1.0, 3, 16, 16, 10, &params));
        assert_eq!(
            reference_switch(0.5, 3, 16, 16, 10, &params),
            reference_switch(0.5, 3, 16, 16, 15, &params)
        );
    }

    #[test]
    fn motion_predictor_desync_propagates_to_later_blocks() {
        let codec = codec(6, 2);
        let packet = MoshPacket {
            kind: MoshPacketKind::P,
            width: 6,
            height: 2,
            blocks: vec![
                MotionBlock {
                    x: 0,
                    y: 0,
                    w: 2,
                    h: 2,
                    dx: 0,
                    dy: 0,
                },
                MotionBlock {
                    x: 2,
                    y: 0,
                    w: 2,
                    h: 2,
                    dx: 0,
                    dy: 0,
                },
                MotionBlock {
                    x: 4,
                    y: 0,
                    w: 2,
                    h: 2,
                    dx: 0,
                    dy: 0,
                },
            ],
            residual: vec![0; 6 * 2 * 3],
            keyframe: Vec::new(),
        };

        let vectors = codec.dirty_motion_vectors(
            &packet,
            &MoshGlitchParams {
                mv_predictor_desync_every: 1,
                mv_predictor_desync_x: 2,
                ..MoshGlitchParams::default()
            },
        );

        assert_eq!(vectors[0].0, 2);
        assert!(vectors[2].0 > vectors[0].0);
    }

    #[test]
    fn vector_bank_block_reads_motion_vector_from_another_bank() {
        let packet = MoshPacket {
            kind: MoshPacketKind::P,
            width: 4,
            height: 4,
            blocks: vec![
                MotionBlock {
                    x: 0,
                    y: 0,
                    w: 2,
                    h: 2,
                    dx: 1,
                    dy: 0,
                },
                MotionBlock {
                    x: 2,
                    y: 0,
                    w: 2,
                    h: 2,
                    dx: 7,
                    dy: 0,
                },
                MotionBlock {
                    x: 0,
                    y: 2,
                    w: 2,
                    h: 2,
                    dx: 2,
                    dy: 0,
                },
                MotionBlock {
                    x: 2,
                    y: 2,
                    w: 2,
                    h: 2,
                    dx: 3,
                    dy: 0,
                },
            ],
            residual: vec![0; 4 * 4 * CHANNELS],
            keyframe: Vec::new(),
        };
        let params = MoshGlitchParams {
            mv_bank_size: 1,
            mv_bank_stride: 1,
            ..MoshGlitchParams::default()
        };

        let block = vector_bank_block(&packet, 0, 2, 1, &params).unwrap();

        assert_eq!(block.dx, 7);
    }

    #[test]
    fn motion_field_interpolation_changes_vector_inside_block() {
        let vectors = vec![(0, 0), (8, 0), (0, 8), (8, 8)];
        let near_origin = interpolated_motion_vector(&vectors, 0, 0, 0, 4, 4, 2, 0, 0, 1.0);
        let near_corner = interpolated_motion_vector(&vectors, 0, 0, 0, 4, 4, 2, 1, 1, 1.0);

        assert_eq!(near_origin, (0.0, 0.0));
        assert!(near_corner.0 > near_origin.0);
        assert!(near_corner.1 > near_origin.1);
    }

    #[test]
    fn subpixel_reference_sampling_interpolates_reference_pixels() {
        let reference = vec![0, 0, 0, 100, 0, 0, 0, 0, 0, 100, 0, 0];
        let sample = sample_reference(&reference, 2, 2, 0.5, 0.5, 0, false);

        assert_eq!(sample, 50.0);
    }

    #[test]
    fn sample_address_desync_is_sub_block_and_latched() {
        let params = MoshGlitchParams {
            sample_address_desync: 1.5,
            reference_latch_frames: 8,
            ..MoshGlitchParams::default()
        };

        let first = sample_address_offset(3, 2, 4, 10, &params);
        let same_bucket = sample_address_offset(3, 2, 4, 15, &params);
        let next_cell = sample_address_offset(3, 11, 4, 10, &params);

        assert_eq!(first, same_bucket);
        assert_ne!(first, next_cell);
        assert!(first.0.abs() <= 1.5);
        assert!(first.1.abs() <= 1.5 * 0.45);
    }

    #[test]
    fn pixel_grain_changes_sample_address_per_pixel() {
        let params = MoshGlitchParams {
            sample_address_desync: 1.5,
            glitch_cell_size: 1,
            reference_latch_frames: 8,
            ..MoshGlitchParams::default()
        };

        let first = sample_address_offset(3, 2, 4, 10, &params);
        let next_pixel = sample_address_offset(3, 3, 4, 10, &params);

        assert_ne!(first, next_pixel);
    }

    #[test]
    fn rectangular_grain_changes_sample_address_by_axis_cells() {
        let params = MoshGlitchParams {
            sample_address_desync: 1.5,
            glitch_cell_width: 4,
            glitch_cell_height: 2,
            reference_latch_frames: 8,
            ..MoshGlitchParams::default()
        };

        let first = sample_address_offset(3, 2, 4, 10, &params);
        let same_cell = sample_address_offset(3, 3, 5, 10, &params);
        let next_x_cell = sample_address_offset(3, 4, 4, 10, &params);
        let next_y_cell = sample_address_offset(3, 2, 6, 10, &params);

        assert_eq!(first, same_cell);
        assert_ne!(first, next_x_cell);
        assert_ne!(first, next_y_cell);
    }

    #[test]
    fn residual_address_shift_reads_neighbor_residual() {
        let packet = MoshPacket {
            kind: MoshPacketKind::P,
            width: 2,
            height: 1,
            blocks: Vec::new(),
            residual: vec![10, 0, 0, 80, 0, 0],
            keyframe: Vec::new(),
        };
        let params = MoshGlitchParams {
            residual_address_shift_x: 1,
            ..MoshGlitchParams::default()
        };

        let (x, y) = residual_address(&packet, 0, 0, 0, 1, &params);
        let shift = residual_channel_shift(0, 1, &params);
        let residual = residual_sample(&packet, x, y, 0, shift, 1.0, false);

        assert_eq!(residual, 80.0);
    }

    #[test]
    fn residual_channel_shift_reads_neighbor_channel() {
        let packet = MoshPacket {
            kind: MoshPacketKind::P,
            width: 1,
            height: 1,
            blocks: Vec::new(),
            residual: vec![10, 40, 90],
            keyframe: Vec::new(),
        };
        let params = MoshGlitchParams {
            residual_channel_shift: 1,
            ..MoshGlitchParams::default()
        };

        let shift = residual_channel_shift(0, 1, &params);
        let residual = residual_sample(&packet, 0, 0, 0, shift, 1.0, false);

        assert_eq!(residual, 40.0);
    }

    #[test]
    fn reference_channel_shift_reads_neighbor_reference_channel() {
        let codec = codec(1, 1);
        let packet = MoshPacket {
            kind: MoshPacketKind::P,
            width: 1,
            height: 1,
            blocks: Vec::new(),
            residual: vec![0, 0, 0],
            keyframe: Vec::new(),
        };
        let reference = vec![10, 40, 90];

        let sample = codec.predicted_sample(
            &reference, &packet, 0, 0, 0, 0.0, 0.0, 0, 0, 0, 1.0, false, 0, false, 1,
        );

        assert_eq!(sample, 40.0);
    }

    #[test]
    fn residual_bank_address_reads_from_another_residual_cell() {
        let packet = MoshPacket {
            kind: MoshPacketKind::P,
            width: 4,
            height: 2,
            blocks: Vec::new(),
            residual: vec![
                10, 0, 0, 20, 0, 0, 80, 0, 0, 90, 0, 0, 30, 0, 0, 40, 0, 0, 100, 0, 0, 110, 0, 0,
            ],
            keyframe: Vec::new(),
        };
        let params = MoshGlitchParams {
            residual_bank_size: 2,
            residual_bank_stride: 1,
            ..MoshGlitchParams::default()
        };

        let (x, y) = residual_address(&packet, 0, 0, 0, 1, &params);
        let residual = residual_sample(&packet, x, y, 0, 0, 1.0, false);

        assert_eq!((x, y), (2, 0));
        assert_eq!(residual, 80.0);
    }

    #[test]
    fn reference_slot_corruption_can_select_older_dirty_history() {
        let mut codec = codec(2, 2);
        let frame_len = 2 * 2 * 3;
        for value in [10_u8, 20, 30, 40] {
            codec.push_decoder_history(vec![value; frame_len]);
        }
        let params = MoshGlitchParams {
            reference_lag: 1,
            reference_slot_count: 4,
            reference_slot_shuffle_every: 1,
            reference_latch_frames: 3,
            ..MoshGlitchParams::default()
        };

        let selected_older = (0..64).any(|block_index| {
            codec
                .dirty_reference_for_block(block_index, &params)
                .is_some_and(|frame| frame[0] != 40)
        });

        assert!(selected_older);
    }

    #[test]
    fn reference_channel_lag_selects_different_history_per_channel() {
        let mut codec = codec(2, 2);
        let frame_len = 2 * 2 * 3;
        for value in [10_u8, 20, 30, 40] {
            codec.push_decoder_history(vec![value; frame_len]);
        }
        let params = MoshGlitchParams {
            reference_lag: 1,
            reference_channel_lag_span: 4,
            reference_channel_lag_stride: 1,
            reference_latch_frames: 1,
            ..MoshGlitchParams::default()
        };

        let red_ref = codec
            .dirty_reference_for_channel_sample(0, 0, 0, 0, &params)
            .unwrap()[0];
        let green_ref = codec
            .dirty_reference_for_channel_sample(0, 0, 0, 1, &params)
            .unwrap()[0];
        let blue_ref = codec
            .dirty_reference_for_channel_sample(0, 0, 0, 2, &params)
            .unwrap()[0];

        assert_eq!(red_ref, 40);
        assert_eq!(green_ref, 30);
        assert_eq!(blue_ref, 20);
    }

    #[test]
    fn scanline_reference_can_select_older_dirty_history() {
        let mut codec = codec(4, 4);
        let frame_len = 4 * 4 * 3;
        for value in [10_u8, 20, 30, 40] {
            codec.push_decoder_history(vec![value; frame_len]);
        }
        let params = MoshGlitchParams {
            reference_lag: 1,
            reference_scanline_height: 1,
            reference_scanline_lag_span: 4,
            reference_latch_frames: 2,
            ..MoshGlitchParams::default()
        };

        let selected_older = (0..64).any(|y| {
            codec
                .scanline_reference_for_sample(0, y, &params)
                .is_some_and(|frame| frame[0] != 40)
        });

        assert!(selected_older);
    }

    #[test]
    fn temporal_slice_drift_selects_different_history_lags_by_row_and_time() {
        let mut codec = codec(4, 4);
        let frame_len = 4 * 4 * 3;
        for value in [10_u8, 20, 30, 40] {
            codec.push_decoder_history(vec![value; frame_len]);
        }
        let params = MoshGlitchParams {
            reference_lag: 1,
            temporal_slice_height: 1,
            temporal_slice_lag_span: 4,
            temporal_slice_drift: 1,
            reference_latch_frames: 1,
            ..MoshGlitchParams::default()
        };

        let first_row = codec
            .temporal_slice_reference_for_sample(0, &params)
            .unwrap()[0];
        let second_row = codec
            .temporal_slice_reference_for_sample(1, &params)
            .unwrap()[0];
        codec.stats.frames_in = 1;
        let drifted_first_row = codec
            .temporal_slice_reference_for_sample(0, &params)
            .unwrap()[0];

        assert_eq!(first_row, 40);
        assert_ne!(second_row, first_row);
        assert_eq!(drifted_first_row, second_row);
    }

    #[test]
    fn reference_slot_corruption_can_select_older_dirty_history_per_sample() {
        let mut codec = codec(4, 4);
        let frame_len = 4 * 4 * 3;
        for value in [10_u8, 20, 30, 40] {
            codec.push_decoder_history(vec![value; frame_len]);
        }
        let params = MoshGlitchParams {
            reference_lag: 1,
            reference_slot_count: 4,
            reference_slot_shuffle_every: 1,
            reference_latch_frames: 3,
            ..MoshGlitchParams::default()
        };

        let selected_older = (0..4).any(|y| {
            (0..4).any(|x| {
                codec
                    .dirty_reference_for_sample(0, x, y, &params)
                    .is_some_and(|frame| frame[0] != 40)
            })
        });

        assert!(selected_older);
    }

    #[test]
    fn feedback_reference_mode_can_recurse_previous_glitch_into_next_encode() {
        let mut codec = MoshCodec::new(MoshCodecConfig {
            width: 4,
            height: 4,
            block_size: 2,
            search_radius: 2,
            search_step: 1,
            keyframe_interval: 0,
            history_len: 4,
            reference_mode: MoshReferenceMode::Feedback,
        })
        .unwrap();
        let first = vec![10_u8; 4 * 4 * 3];
        let mut second = first.clone();
        for y in 0..2 {
            for x in 0..2 {
                let index = rgb_index(4, x, y);
                second[index] = 200;
                second[index + 1] = 200;
                second[index + 2] = 200;
            }
        }
        let params = MoshGlitchParams {
            residual_keep: 0.0,
            activity_mode: ActivityMode::Active,
            activity_threshold: 12,
            ..MoshGlitchParams::default()
        };
        let mut output = vec![0; first.len()];

        codec
            .process_rgb_frame(&first, &MoshGlitchParams::default(), &mut output)
            .unwrap();
        codec
            .process_rgb_frame(&second, &params, &mut output)
            .unwrap();
        codec
            .process_rgb_frame(&second, &params, &mut output)
            .unwrap();

        assert_ne!(&output[0..12], &second[0..12]);
    }

    #[test]
    fn packet_bitstream_round_trips_predicted_packet() {
        let mut codec = codec(4, 2);
        let first = vec![
            0, 0, 0, 10, 0, 0, 20, 0, 0, 30, 0, 0, 40, 0, 0, 50, 0, 0, 60, 0, 0, 70, 0, 0,
        ];
        let second = vec![
            10, 0, 0, 20, 0, 0, 30, 0, 0, 40, 0, 0, 50, 0, 0, 60, 0, 0, 70, 0, 0, 80, 0, 0,
        ];
        let mut output = vec![0; first.len()];

        codec
            .process_rgb_frame(&first, &MoshGlitchParams::default(), &mut output)
            .unwrap();
        let packet = codec.encode_rgb_packet(&second).unwrap();
        let bitstream = encode_packet_bitstream(&packet).unwrap();
        let decoded = decode_packet_bitstream(&bitstream).unwrap();
        codec
            .decode_rgb_packet(&decoded, &MoshGlitchParams::default(), &mut output)
            .unwrap();

        assert_eq!(decoded.kind, MoshPacketKind::P);
        assert_eq!(decoded.blocks, packet.blocks);
        assert_eq!(output, second);
    }

    #[test]
    fn bitstream_residual_zero_mutation_changes_decoded_frame() {
        let mut codec = codec(4, 2);
        let first = vec![
            0, 0, 0, 10, 0, 0, 20, 0, 0, 30, 0, 0, 40, 0, 0, 50, 0, 0, 60, 0, 0, 70, 0, 0,
        ];
        let second = vec![
            10, 0, 0, 20, 0, 0, 30, 0, 0, 40, 0, 0, 50, 0, 0, 60, 0, 0, 70, 0, 0, 80, 0, 0,
        ];
        let mut output = vec![0; first.len()];

        codec
            .process_rgb_frame(&first, &MoshGlitchParams::default(), &mut output)
            .unwrap();
        let stats = codec
            .process_rgb_frame_bitstream(
                &second,
                &MoshGlitchParams::default(),
                &MoshBitstreamParams {
                    enabled: true,
                    residual_zero_every: 1,
                    ..MoshBitstreamParams::default()
                },
                &mut output,
            )
            .unwrap();

        assert_ne!(output, second);
        assert!(stats.residual_blocks_zeroed > 0);
    }

    #[test]
    fn bitstream_codebook_reuses_older_residual_tiles() {
        let mut codec = codec(8, 8);
        let first = vec![0_u8; 8 * 8 * CHANNELS];
        let second: Vec<u8> = (0..8 * 8 * CHANNELS)
            .map(|index| (index * 3 % 251) as u8)
            .collect();
        let third: Vec<u8> = (0..8 * 8 * CHANNELS)
            .map(|index| (index * 7 % 251) as u8)
            .collect();
        let params = MoshBitstreamParams {
            enabled: true,
            codebook_replace_every: 1,
            codebook_tile_size: 4,
            codebook_slots: 8,
            codebook_stride: 0,
            codebook_update_every: 1,
            ..MoshBitstreamParams::default()
        };
        let mut output = vec![0; first.len()];

        codec
            .process_rgb_frame_bitstream(&first, &MoshGlitchParams::default(), &params, &mut output)
            .unwrap();
        let warmup = codec
            .process_rgb_frame_bitstream(
                &second,
                &MoshGlitchParams::default(),
                &params,
                &mut output,
            )
            .unwrap();
        let stats = codec
            .process_rgb_frame_bitstream(&third, &MoshGlitchParams::default(), &params, &mut output)
            .unwrap();

        assert_eq!(warmup.codebook_tiles, 0);
        assert!(stats.codebook_tiles > 0);
        assert_ne!(output, third);
    }

    #[test]
    fn bitstream_mv_sign_flip_mutates_block_table_bytes() {
        let mut codec = codec(4, 2);
        let first = vec![
            0, 0, 0, 10, 0, 0, 20, 0, 0, 30, 0, 0, 40, 0, 0, 50, 0, 0, 60, 0, 0, 70, 0, 0,
        ];
        let second = vec![
            10, 0, 0, 20, 0, 0, 30, 0, 0, 40, 0, 0, 50, 0, 0, 60, 0, 0, 70, 0, 0, 80, 0, 0,
        ];
        let mut output = vec![0; first.len()];

        codec
            .process_rgb_frame(&first, &MoshGlitchParams::default(), &mut output)
            .unwrap();
        let packet = codec.encode_rgb_packet(&second).unwrap();
        let original_dx = packet.blocks[0].dx;
        let mut bitstream = encode_packet_bitstream(&packet).unwrap();
        let stats = mutate_packet_bitstream(
            &mut bitstream,
            &MoshBitstreamParams {
                enabled: true,
                mv_sign_flip_every: 1,
                ..MoshBitstreamParams::default()
            },
            1,
        )
        .unwrap();
        let decoded = decode_packet_bitstream(&bitstream).unwrap();

        assert_eq!(decoded.blocks[0].dx, original_dx.saturating_neg());
        assert!(stats.mv_sign_flipped > 0);
    }

    #[test]
    fn bitstream_entropy_slip_mutates_residual_payload_bytes() {
        let packet = MoshPacket {
            kind: MoshPacketKind::P,
            width: 4,
            height: 2,
            blocks: vec![MotionBlock {
                x: 0,
                y: 0,
                w: 4,
                h: 2,
                dx: 0,
                dy: 0,
            }],
            residual: (0..24).map(|value| (value * 17 - 80) as i16).collect(),
            keyframe: Vec::new(),
        };
        let mut bitstream = encode_packet_bitstream(&packet).unwrap();
        let stats = mutate_packet_bitstream(
            &mut bitstream,
            &MoshBitstreamParams {
                enabled: true,
                entropy_slip_every: 1,
                entropy_slip_bytes: 1,
                entropy_resync_bytes: 12,
                entropy_slip_windows: 2,
                ..MoshBitstreamParams::default()
            },
            1,
        )
        .unwrap();
        let decoded = decode_packet_bitstream(&bitstream).unwrap();

        assert_eq!(stats.entropy_slips, 2);
        assert_eq!(decoded.blocks, packet.blocks);
        assert_ne!(decoded.residual, packet.residual);
    }

    #[test]
    fn bitstream_coeff_glitch_mutates_residual_coefficients() {
        let packet = MoshPacket {
            kind: MoshPacketKind::P,
            width: 4,
            height: 4,
            blocks: vec![MotionBlock {
                x: 0,
                y: 0,
                w: 4,
                h: 4,
                dx: 0,
                dy: 0,
            }],
            residual: (0..48).map(|value| (value * 9 - 120) as i16).collect(),
            keyframe: Vec::new(),
        };
        let mut bitstream = encode_packet_bitstream(&packet).unwrap();
        let stats = mutate_packet_bitstream(
            &mut bitstream,
            &MoshBitstreamParams {
                enabled: true,
                coeff_glitch_every: 1,
                coeff_block_size: 4,
                coeff_shift: 1,
                coeff_sign_flip_every: 3,
                coeff_zero_high: 5,
                coeff_quant: 4,
                ..MoshBitstreamParams::default()
            },
            1,
        )
        .unwrap();
        let decoded = decode_packet_bitstream(&bitstream).unwrap();

        assert_eq!(stats.coeff_blocks, 3);
        assert_eq!(decoded.blocks, packet.blocks);
        assert_ne!(decoded.residual, packet.residual);
    }
}
