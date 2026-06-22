use std::collections::VecDeque;
use std::io;

use rayon::prelude::*;

use crate::mosh_codec::codec_thread_pool;

const CHANNELS: usize = 3;
const PARALLEL_FRAME_PIXELS: usize = 200_000;
const MAGIC: &[u8; 4] = b"SCN0";
const VERSION: u8 = 7;
const HEADER_LEN: usize = 16;
const LINE_HEADER_LEN: usize = 14;
const LINE_SYNC: [u8; 2] = [0xa5, 0x5a];
const CHROMA_QUANT: i16 = 2;
const RESYNC_MIN_LINES: usize = 6;
const RESYNC_MAX_LINES: usize = 20;
const RESYNC_PAYLOAD_LINES: usize = 10;

#[derive(Debug, Clone, Copy)]
pub struct ScanlineCodecConfig {
    pub width: usize,
    pub height: usize,
    pub luma_quant: u8,
    pub chroma_group: usize,
    pub history_len: usize,
}

impl ScanlineCodecConfig {
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
            return Err(invalid_input("width and height must be greater than zero"));
        }
        if self.width > u16::MAX as usize || self.height > u16::MAX as usize {
            return Err(invalid_input("SCN0 dimensions must fit in 16 bits"));
        }
        if self.luma_quant == 0 {
            return Err(invalid_input("luma_quant must be greater than zero"));
        }
        if self.chroma_group == 0 || !self.chroma_group.is_power_of_two() {
            return Err(invalid_input(
                "chroma_group must be a non-zero power of two",
            ));
        }
        if self.history_len == 0 {
            return Err(invalid_input("history_len must be greater than zero"));
        }
        self.frame_len()
            .ok_or_else(|| invalid_input("frame dimensions overflow addressable memory"))?;
        Ok(())
    }
}

impl Default for ScanlineCodecConfig {
    fn default() -> Self {
        Self {
            width: 0,
            height: 0,
            luma_quant: 2,
            chroma_group: 2,
            history_len: 8,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct ScanlineGlitchParams {
    pub line_shift: i16,
    pub line_shift_every: u64,
    pub line_shift_drift: i16,
    pub line_index_offset: i16,
    pub line_index_every: u64,
    pub line_index_stride: i16,
    pub sync_loss_every: u64,
    pub field_sync_loss_every: u64,
    pub field_parity_flip_every: u64,
    pub phase_offset: i8,
    pub phase_drift: i8,
    pub burst_loss_every: u64,
    pub predictor_flip_every: u64,
    pub predictor_lag: usize,
    pub predictor_line_offset: i16,
    pub predictor_line_offset_every: u64,
    pub quant_offset: i8,
    pub quant_offset_every: u64,
    pub luma_payload_slip: i16,
    pub luma_payload_slip_every: u64,
    pub chroma_payload_slip: i16,
    pub chroma_payload_slip_every: u64,
    pub carrier_sign_flip_every: u64,
    pub chroma_group_delta: i8,
    pub chroma_sequence_offset: i8,
    pub chroma_sequence_every: u64,
    pub chroma_seed_loss_every: u64,
    pub chroma_xor_mask: u8,
    pub chroma_xor_every: u64,
    pub luma_run_delta: i8,
    pub luma_run_delta_every: u64,
    pub packet_length_delta: i16,
    pub packet_length_delta_every: u64,
    pub payload_swap_every: u64,
    pub history_line_weave: usize,
    pub history_line_weave_every: u64,
}

impl Default for ScanlineGlitchParams {
    fn default() -> Self {
        Self {
            line_shift: 0,
            line_shift_every: 0,
            line_shift_drift: 0,
            line_index_offset: 0,
            line_index_every: 0,
            line_index_stride: 0,
            sync_loss_every: 0,
            field_sync_loss_every: 0,
            field_parity_flip_every: 0,
            phase_offset: 0,
            phase_drift: 0,
            burst_loss_every: 0,
            predictor_flip_every: 0,
            predictor_lag: 1,
            predictor_line_offset: 0,
            predictor_line_offset_every: 0,
            quant_offset: 0,
            quant_offset_every: 0,
            luma_payload_slip: 0,
            luma_payload_slip_every: 0,
            chroma_payload_slip: 0,
            chroma_payload_slip_every: 0,
            carrier_sign_flip_every: 0,
            chroma_group_delta: 0,
            chroma_sequence_offset: 0,
            chroma_sequence_every: 0,
            chroma_seed_loss_every: 0,
            chroma_xor_mask: 0,
            chroma_xor_every: 0,
            luma_run_delta: 0,
            luma_run_delta_every: 0,
            packet_length_delta: 0,
            packet_length_delta_every: 0,
            payload_swap_every: 0,
            history_line_weave: 0,
            history_line_weave_every: 0,
        }
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct ScanlineCodecStats {
    pub frames_in: u64,
    pub lines_encoded: u64,
    pub raw_bytes: u64,
    pub encoded_bytes: u64,
    pub damaged_lines: u64,
    pub concealed_lines: u64,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct ScanlineMutationStats {
    pub sync_words_lost: u64,
    pub field_syncs_lost: u64,
    pub field_parities_flipped: u64,
    pub line_starts_shifted: u64,
    pub line_indices_shifted: u64,
    pub predictor_lines_shifted: u64,
    pub phases_shifted: u64,
    pub predictors_flipped: u64,
    pub quantizers_shifted: u64,
    pub luma_payloads_slipped: u64,
    pub chroma_payloads_slipped: u64,
    pub carriers_sign_flipped: u64,
    pub chroma_groups_shifted: u64,
    pub chroma_sequences_shifted: u64,
    pub chroma_seeds_lost: u64,
    pub chroma_samples_xored: u64,
    pub luma_runs_shifted: u64,
    pub packet_lengths_shifted: u64,
    pub payloads_swapped: u64,
}

#[derive(Clone, Copy, Eq, PartialEq)]
enum Predictor {
    Horizontal = 0,
    Temporal = 1,
}

impl Predictor {
    fn from_flags(flags: u8) -> Self {
        if flags & 1 == 0 {
            Self::Horizontal
        } else {
            Self::Temporal
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct LineRecord {
    header: usize,
    luma: usize,
    luma_len: usize,
    chroma: usize,
    chroma_len: usize,
}

#[derive(Debug, Clone, Copy, Default)]
struct SignalDecoderState {
    expected_line: usize,
    last_transmitted_line: Option<usize>,
    horizontal_phase: i32,
    last_timebase_error: i16,
    burst_phase_error: i16,
    locked: bool,
}

#[derive(Debug, Clone, Copy)]
struct ReceiverLineState {
    output_line: usize,
    horizontal_shift: usize,
    chroma_phase: u8,
}

pub struct ScanlineCodec {
    config: ScanlineCodecConfig,
    stats: ScanlineCodecStats,
    encoder_history: VecDeque<Vec<u8>>,
    decoder_history: VecDeque<Vec<u8>>,
    clean_ycbcr: Vec<u8>,
    decoded_ycbcr: Vec<u8>,
    bitstream: Vec<u8>,
    line_present: Vec<bool>,
    decoder_state: SignalDecoderState,
}

impl ScanlineCodec {
    pub fn new(config: ScanlineCodecConfig) -> io::Result<Self> {
        config.validate()?;
        let frame_len = config.frame_len().expect("validated frame dimensions");
        Ok(Self {
            config,
            stats: ScanlineCodecStats::default(),
            encoder_history: VecDeque::new(),
            decoder_history: VecDeque::new(),
            clean_ycbcr: vec![0; frame_len],
            decoded_ycbcr: vec![0; frame_len],
            bitstream: Vec::with_capacity(frame_len),
            line_present: vec![false; config.height],
            decoder_state: SignalDecoderState::default(),
        })
    }

    pub fn config(&self) -> &ScanlineCodecConfig {
        &self.config
    }

    pub fn stats(&self) -> &ScanlineCodecStats {
        &self.stats
    }

    pub fn reset_glitch_state(&mut self) {
        self.encoder_history.clear();
        self.decoder_history.clear();
        self.decoded_ycbcr.fill(0);
        self.decoder_state = SignalDecoderState::default();
    }

    pub fn process_rgb_frame(
        &mut self,
        input: &[u8],
        params: &ScanlineGlitchParams,
        output: &mut [u8],
    ) -> io::Result<ScanlineMutationStats> {
        let frame_len = self.config.frame_len().expect("validated frame dimensions");
        if input.len() != frame_len || output.len() != frame_len {
            return Err(invalid_input(format!(
                "input and output frames must be {frame_len} bytes of rgb24"
            )));
        }

        rgb_to_ycbcr(input, &mut self.clean_ycbcr);
        self.encode_frame()?;
        let mutation_stats =
            mutate_scanline_bitstream(&mut self.bitstream, params, self.stats.frames_in)?;
        self.decode_frame(params)?;
        ycbcr_to_rgb(&self.decoded_ycbcr, output);

        self.push_history();
        self.stats.frames_in += 1;
        self.stats.raw_bytes = self.stats.raw_bytes.saturating_add(frame_len as u64);
        self.stats.encoded_bytes = self
            .stats
            .encoded_bytes
            .saturating_add(self.bitstream.len() as u64);
        self.stats.damaged_lines = self.stats.damaged_lines.saturating_add(
            mutation_stats.sync_words_lost
                + mutation_stats.field_syncs_lost
                + mutation_stats.field_parities_flipped
                + mutation_stats.line_starts_shifted
                + mutation_stats.line_indices_shifted
                + mutation_stats.predictor_lines_shifted
                + mutation_stats.phases_shifted
                + mutation_stats.predictors_flipped
                + mutation_stats.quantizers_shifted
                + mutation_stats.luma_payloads_slipped
                + mutation_stats.chroma_payloads_slipped
                + mutation_stats.carriers_sign_flipped
                + mutation_stats.chroma_groups_shifted
                + mutation_stats.chroma_sequences_shifted
                + mutation_stats.chroma_seeds_lost
                + mutation_stats.chroma_samples_xored
                + mutation_stats.luma_runs_shifted
                + mutation_stats.packet_lengths_shifted
                + mutation_stats.payloads_swapped,
        );
        Ok(mutation_stats)
    }

    fn encode_frame(&mut self) -> io::Result<()> {
        self.bitstream.clear();
        self.bitstream.extend_from_slice(MAGIC);
        self.bitstream.push(VERSION);
        self.bitstream.push(CHROMA_QUANT as u8);
        push_u16(&mut self.bitstream, self.config.width as u16);
        push_u16(&mut self.bitstream, self.config.height as u16);
        push_u32(&mut self.bitstream, self.stats.frames_in as u32);
        self.bitstream
            .push(self.config.chroma_group.trailing_zeros() as u8);
        self.bitstream.push(2);

        // Per-line payloads are independent: horizontal prediction stays within a
        // line and temporal prediction only reads the previous frame (read-only).
        // Compute them in parallel, then assemble the bitstream serially in
        // field-scan order. The serial pass keeps the resync counters and byte
        // layout identical to the single-threaded encoder, so the SCN0 output is
        // byte-for-byte unchanged.
        let height = self.config.height;
        let width = self.config.width;
        let frames_in = self.stats.frames_in;
        let reference = self.encoder_history.back().map(Vec::as_slice);
        let clean = self.clean_ycbcr.as_slice();
        let config = &self.config;
        let encode_lines = || -> Vec<EncodedLine> {
            (0..height)
                .into_par_iter()
                .map(|y| encode_line(clean, reference, config, frames_in, y))
                .collect()
        };
        let lines: Vec<EncodedLine> = if width.saturating_mul(height) >= PARALLEL_FRAME_PIXELS {
            match codec_thread_pool() {
                Some(pool) => pool.install(encode_lines),
                None => (0..height)
                    .map(|y| encode_line(clean, reference, config, frames_in, y))
                    .collect(),
            }
        } else {
            (0..height)
                .map(|y| encode_line(clean, reference, config, frames_in, y))
                .collect()
        };

        let mut lines_since_resync = 0_usize;
        let mut bytes_since_resync = 0_usize;
        let resync_payload_target = self.config.width.saturating_mul(RESYNC_PAYLOAD_LINES);
        for y in field_scan_order(self.config.height) {
            let line = &lines[y];
            let line_payload_len = line
                .luma_payload
                .len()
                .saturating_add(line.chroma_payload.len());
            let field_start = is_field_start(y);
            let adaptive_resync = lines_since_resync >= RESYNC_MAX_LINES
                || (lines_since_resync >= RESYNC_MIN_LINES
                    && bytes_since_resync.saturating_add(line_payload_len)
                        >= resync_payload_target);
            let resync = field_start || adaptive_resync;

            if line.luma_payload.len() > u16::MAX as usize
                || line.chroma_payload.len() > u16::MAX as usize
            {
                return Err(invalid_input("SCN0 line payload exceeds 16-bit length"));
            }

            self.bitstream.extend_from_slice(&LINE_SYNC);
            push_u16(&mut self.bitstream, y as u16);
            let flags = (line.predictor as u8)
                | ((line.phase & 3) << 1)
                | if line.chroma_packed { 0x40 } else { 0 }
                | if resync { 0x80 } else { 0 };
            self.bitstream.push(flags);
            self.bitstream.push(self.config.luma_quant);
            push_i16(&mut self.bitstream, 0);
            push_i16(&mut self.bitstream, 0);
            push_u16(&mut self.bitstream, line.luma_payload.len() as u16);
            push_u16(&mut self.bitstream, line.chroma_payload.len() as u16);
            self.bitstream.extend_from_slice(&line.luma_payload);
            self.bitstream.extend_from_slice(&line.chroma_payload);
            self.stats.lines_encoded += 1;
            if resync {
                lines_since_resync = 1;
                bytes_since_resync = line_payload_len;
            } else {
                lines_since_resync += 1;
                bytes_since_resync = bytes_since_resync.saturating_add(line_payload_len);
            }
        }
        Ok(())
    }

    fn decode_frame(&mut self, params: &ScanlineGlitchParams) -> io::Result<()> {
        validate_stream_header(&self.bitstream, &self.config)?;
        self.line_present.fill(false);
        let chroma_group_log2 = self.bitstream[14].min(6);
        let chroma_group = 1_usize << chroma_group_log2;
        let chroma_quant = self.bitstream[5].max(1) as i16;

        if let Some(reference) = history_reference(&self.decoder_history, params.predictor_lag) {
            self.decoded_ycbcr.copy_from_slice(reference);
        } else {
            self.decoded_ycbcr.fill(0);
            for pixel in self.decoded_ycbcr.chunks_exact_mut(CHANNELS) {
                pixel[1] = 128;
                pixel[2] = 128;
            }
        }

        // The receiver is a serial state machine (timebase/phase/parity accumulate
        // line-to-line), so resolve all receiver states serially first. The heavy
        // per-pixel decode is independent per OUTPUT row, so group each output line's
        // records (in transmission order) and decode the rows in parallel. Applying a
        // row's records in order reproduces the single-threaded "last write wins"
        // result exactly, so the decoded frame is byte-for-byte unchanged.
        let height = self.config.height;
        let records = scan_line_records(&self.bitstream);
        let mut records_by_row: Vec<Vec<(LineRecord, ReceiverLineState)>> =
            vec![Vec::new(); height];
        for record in records {
            let receiver = self.receiver_line_state(record)?;
            records_by_row[receiver.output_line].push((record, receiver));
        }

        let mut concealed = 0_u64;
        for (row, present) in self.line_present.iter_mut().enumerate() {
            *present = !records_by_row[row].is_empty();
            if !*present {
                concealed += 1;
            }
        }
        self.stats.concealed_lines = self.stats.concealed_lines.saturating_add(concealed);

        let width = self.config.width;
        let row_stride = width * CHANNELS;
        let parallel = width.saturating_mul(height) >= PARALLEL_FRAME_PIXELS;
        let decoded_ycbcr = &mut self.decoded_ycbcr;
        let bitstream = &self.bitstream;
        let decoder_history = &self.decoder_history;
        let config = &self.config;
        let records_by_row = &records_by_row;
        let predictor_reference =
            history_reference(decoder_history, params.predictor_lag).map(Vec::as_slice);
        let weave_reference =
            history_reference(decoder_history, params.history_line_weave).map(Vec::as_slice);
        let decode_row = |row: usize, slice: &mut [u8]| {
            let mut scratch: Vec<i8> = Vec::new();
            for (record, receiver) in &records_by_row[row] {
                decode_line_into(
                    slice,
                    &mut scratch,
                    bitstream,
                    predictor_reference,
                    weave_reference,
                    config,
                    params,
                    chroma_group,
                    chroma_quant,
                    *record,
                    *receiver,
                );
            }
        };

        if parallel {
            if let Some(pool) = codec_thread_pool() {
                pool.install(|| {
                    decoded_ycbcr
                        .par_chunks_mut(row_stride)
                        .enumerate()
                        .for_each(|(row, slice)| decode_row(row, slice));
                });
                return Ok(());
            }
        }
        decoded_ycbcr
            .chunks_mut(row_stride)
            .enumerate()
            .for_each(|(row, slice)| decode_row(row, slice));
        Ok(())
    }

    fn receiver_line_state(&mut self, record: LineRecord) -> io::Result<ReceiverLineState> {
        let transmitted_line = read_u16(&self.bitstream, record.header + 2)? as usize;
        if transmitted_line >= self.config.height {
            return Err(invalid_data("SCN0 line address exceeds frame height"));
        }
        let flags = self.bitstream[record.header + 4];
        let resync = flags & 0x80 != 0;
        let transmitted_phase = (flags >> 1) & 3;
        let timing_error = read_i16(&self.bitstream, record.header + 6)?;

        if resync {
            self.decoder_state.expected_line = transmitted_line;
            self.decoder_state.horizontal_phase = 0;
            self.decoder_state.last_timebase_error = 0;
            self.decoder_state.burst_phase_error = 0;
            self.decoder_state.locked = true;
        } else if let Some(last_line) = self.decoder_state.last_transmitted_line {
            let expected_transmitted = next_field_scan_line(last_line, self.config.height);
            let missing =
                field_scan_distance(expected_transmitted, transmitted_line, self.config.height);
            if missing != 0 {
                let missing = missing.min(RESYNC_MAX_LINES - 1);
                self.decoder_state.locked = false;
                self.decoder_state.horizontal_phase +=
                    missing as i32 * (self.decoder_state.last_timebase_error as i32 + 2);
                self.decoder_state.burst_phase_error =
                    (self.decoder_state.burst_phase_error + missing as i16).rem_euclid(4);
            }
        }

        let output_line = if resync || self.decoder_state.locked {
            transmitted_line
        } else {
            self.decoder_state.expected_line
        };
        self.decoder_state.expected_line = next_field_scan_line(output_line, self.config.height);
        self.decoder_state.last_transmitted_line = Some(transmitted_line);

        self.decoder_state.horizontal_phase += timing_error as i32;
        if timing_error != 0 {
            self.decoder_state.last_timebase_error = timing_error;
        } else {
            self.decoder_state.last_timebase_error =
                self.decoder_state.last_timebase_error.saturating_mul(3) / 4;
        }

        let expected_phase = ((self.stats.frames_in as usize + transmitted_line) & 3) as u8;
        let phase_error = quarter_phase_delta(transmitted_phase, expected_phase);
        self.decoder_state.burst_phase_error =
            (self.decoder_state.burst_phase_error + phase_error).rem_euclid(4);

        Ok(ReceiverLineState {
            output_line,
            horizontal_shift: signed_mod(self.decoder_state.horizontal_phase, self.config.width),
            chroma_phase: (expected_phase as i16 + self.decoder_state.burst_phase_error)
                .rem_euclid(4) as u8,
        })
    }

    fn push_history(&mut self) {
        recycle_history_frame(
            &mut self.encoder_history,
            &mut self.clean_ycbcr,
            self.config.history_len,
        );
        recycle_history_frame(
            &mut self.decoder_history,
            &mut self.decoded_ycbcr,
            self.config.history_len,
        );
    }
}

pub fn mutate_scanline_bitstream(
    bitstream: &mut [u8],
    params: &ScanlineGlitchParams,
    frame_index: u64,
) -> io::Result<ScanlineMutationStats> {
    if bitstream.len() < HEADER_LEN || &bitstream[..4] != MAGIC {
        return Err(invalid_data("invalid SCN0 stream"));
    }
    let records = scan_line_records(bitstream);
    let mut stats = ScanlineMutationStats::default();
    let mut burst_phase = params.phase_offset as i16;
    let mut field_marker_index = 0_u64;

    if params.chroma_group_delta != 0 {
        bitstream[14] = (bitstream[14] as i16 + params.chroma_group_delta as i16).clamp(0, 6) as u8;
        stats.chroma_groups_shifted = 1;
    }

    for (record_index, record) in records.iter().copied().enumerate() {
        let ordinal = record_index as u64 + 1;
        let line = read_u16(bitstream, record.header + 2)? as usize;
        let field_start = is_field_start(line);
        if field_start {
            field_marker_index += 1;
            if is_every(params.field_sync_loss_every, field_marker_index) {
                bitstream[record.header + 4] &= !0x80;
                stats.field_syncs_lost += 1;
            }
            if bitstream.len() > 8
                && read_u16(bitstream, 8)? > 1
                && is_every(params.field_parity_flip_every, field_marker_index)
            {
                write_u16(bitstream, record.header + 2, (line ^ 1) as u16)?;
                stats.field_parities_flipped += 1;
            }
        }
        if params.burst_loss_every != 0 && is_every(params.burst_loss_every, ordinal) {
            burst_phase += if ((frame_index + ordinal) & 1) == 0 {
                1
            } else {
                -1
            };
        }
        let phase_delta = burst_phase + params.phase_drift as i16 * record_index as i16;
        if phase_delta != 0 {
            let flags = bitstream[record.header + 4];
            let phase = ((flags >> 1) & 3) as i16;
            let shifted = (phase + phase_delta).rem_euclid(4) as u8;
            bitstream[record.header + 4] = (flags & !0x06) | (shifted << 1);
            stats.phases_shifted += 1;
        }

        if is_every(params.predictor_flip_every, ordinal) {
            bitstream[record.header + 4] ^= 1;
            stats.predictors_flipped += 1;
        }
        if is_every(params.payload_swap_every, ordinal) {
            bitstream[record.header + 4] ^= 0x08;
            stats.payloads_swapped += 1;
        }
        if params.chroma_sequence_offset != 0 && is_every(params.chroma_sequence_every, ordinal) {
            let flags = bitstream[record.header + 4];
            let current = ((flags >> 4) & 3) as i16;
            let shifted = (current + params.chroma_sequence_offset as i16).rem_euclid(4) as u8;
            bitstream[record.header + 4] = (flags & !0x30) | (shifted << 4);
            stats.chroma_sequences_shifted += 1;
        }
        if params.quant_offset != 0 && is_every(params.quant_offset_every, ordinal) {
            bitstream[record.header + 5] = (bitstream[record.header + 5] as i16
                + params.quant_offset as i16)
                .clamp(1, 255) as u8;
            stats.quantizers_shifted += 1;
        }
        if (params.line_shift != 0 || params.line_shift_drift != 0)
            && is_every(params.line_shift_every, ordinal)
        {
            let shift =
                params.line_shift as i32 + params.line_shift_drift as i32 * record_index as i32;
            write_i16(
                bitstream,
                record.header + 6,
                shift.clamp(i16::MIN as i32, i16::MAX as i32) as i16,
            )?;
            stats.line_starts_shifted += 1;
        }
        if (params.line_index_offset != 0 || params.line_index_stride != 0)
            && is_every(params.line_index_every, ordinal)
        {
            let line = read_u16(bitstream, record.header + 2)? as i32;
            let height = read_u16(bitstream, 8)? as i32;
            let shifted = (line
                + params.line_index_offset as i32
                + params.line_index_stride as i32 * record_index as i32)
                .rem_euclid(height) as u16;
            write_u16(bitstream, record.header + 2, shifted)?;
            stats.line_indices_shifted += 1;
        }
        if params.predictor_line_offset != 0
            && is_every(params.predictor_line_offset_every, ordinal)
        {
            write_i16(bitstream, record.header + 8, params.predictor_line_offset)?;
            stats.predictor_lines_shifted += 1;
        }
        if params.luma_run_delta != 0 && is_every(params.luma_run_delta_every, ordinal) {
            mutate_rle_run_lengths(
                &mut bitstream[record.luma..record.luma + record.luma_len],
                params.luma_run_delta,
            );
            stats.luma_runs_shifted += 1;
        }
        if params.luma_payload_slip != 0 && is_every(params.luma_payload_slip_every, ordinal) {
            rotate_bytes(
                &mut bitstream[record.luma..record.luma + record.luma_len],
                params.luma_payload_slip,
            );
            stats.luma_payloads_slipped += 1;
        }
        if params.chroma_payload_slip != 0 && is_every(params.chroma_payload_slip_every, ordinal) {
            rotate_bytes(
                &mut bitstream[record.chroma..record.chroma + record.chroma_len],
                params.chroma_payload_slip,
            );
            stats.chroma_payloads_slipped += 1;
        }
        if is_every(params.chroma_seed_loss_every, ordinal) {
            if let Some(seed) = bitstream.get_mut(record.chroma..record.chroma + 2) {
                seed.fill(0);
                stats.chroma_seeds_lost += 1;
            }
        }
        if params.chroma_xor_mask != 0 && is_every(params.chroma_xor_every, ordinal) {
            for value in &mut bitstream[record.chroma..record.chroma + record.chroma_len] {
                *value ^= params.chroma_xor_mask;
            }
            stats.chroma_samples_xored += 1;
        }
        if is_every(params.carrier_sign_flip_every, ordinal) {
            for value in &mut bitstream[record.chroma..record.chroma + record.chroma_len] {
                *value = (*value as i8).wrapping_neg() as u8;
            }
            stats.carriers_sign_flipped += 1;
        }
        if is_every(params.sync_loss_every, ordinal) {
            bitstream[record.header] = 0;
            bitstream[record.header + 1] = 0;
            stats.sync_words_lost += 1;
        }
        if params.packet_length_delta != 0 && is_every(params.packet_length_delta_every, ordinal) {
            let luma_len = read_u16(bitstream, record.header + 10)? as i32;
            let shifted = (luma_len + params.packet_length_delta as i32).clamp(0, u16::MAX as i32);
            write_u16(bitstream, record.header + 10, shifted as u16)?;
            stats.packet_lengths_shifted += 1;
        }
    }
    Ok(stats)
}

// Decode one transmitted line into a single output-row slice (row-relative writes),
// using a caller-provided scratch buffer so rows can be decoded in parallel. Mirrors
// the former ScanlineCodec::decode_line exactly. Infallible: scan_line_records already
// guarantees the header is complete and payload ranges are in-bounds.
#[allow(clippy::too_many_arguments)]
fn decode_line_into(
    row: &mut [u8],
    luma_residual: &mut Vec<i8>,
    bitstream: &[u8],
    predictor_reference: Option<&[u8]>,
    weave_reference: Option<&[u8]>,
    config: &ScanlineCodecConfig,
    params: &ScanlineGlitchParams,
    chroma_group: usize,
    chroma_quant: i16,
    record: LineRecord,
    receiver: ReceiverLineState,
) {
    let width = config.width;
    let flags = bitstream[record.header + 4];
    let predictor = Predictor::from_flags(flags);
    let payload_swap = flags & 0x08 != 0;
    let sequence_offset = (flags >> 4) & 3;
    let chroma_packed = flags & 0x40 != 0;
    let quant = bitstream[record.header + 5].max(1) as i16;
    let predictor_line_offset = read_i16(bitstream, record.header + 8).unwrap_or(0);
    let line_index = receiver.output_line;

    luma_residual.clear();
    let luma_range = if payload_swap {
        record.chroma..record.chroma + record.chroma_len
    } else {
        record.luma..record.luma + record.luma_len
    };
    decode_zero_rle(&bitstream[luma_range], width, luma_residual);

    let reference = predictor_reference;
    let shift = receiver.horizontal_shift;
    let mut left = 0_i16;
    let predictor_y = signed_mod(
        line_index as i32 + predictor_line_offset as i32,
        config.height,
    );
    for x in 0..width {
        let residual = luma_residual.get(x).copied().unwrap_or(0) as i16;
        let source_offset = pixel_offset(width, x, predictor_y);
        let predicted = match predictor {
            Predictor::Horizontal => left,
            Predictor::Temporal => reference.map_or(0, |frame| frame[source_offset] as i16),
        };
        let decoded = (predicted + residual * quant).clamp(0, 255) as u8;
        left = decoded as i16;
        let output_x = (x + shift) % width;
        row[output_x * CHANNELS] = decoded;
    }

    let chroma = if payload_swap {
        &bitstream[record.luma..record.luma + record.luma_len]
    } else {
        &bitstream[record.chroma..record.chroma + record.chroma_len]
    };
    let groups = width.div_ceil(chroma_group);
    let Some((&seed_a, &seed_b)) = chroma.first().zip(chroma.get(1)) else {
        return;
    };
    let mut carrier_a = seed_a as i8;
    let mut carrier_b = seed_b as i8;
    let sequence_phase = receiver.chroma_phase.wrapping_add(sequence_offset) & 3;
    luma_residual.clear();
    if chroma_packed {
        decode_chroma_deltas(
            chroma.get(2..).unwrap_or_default(),
            groups.saturating_sub(1),
            luma_residual,
        );
    } else {
        luma_residual.extend(
            chroma
                .get(2..)
                .unwrap_or_default()
                .iter()
                .take(groups.saturating_sub(1))
                .map(|value| *value as i8),
        );
        luma_residual.resize(groups.saturating_sub(1), 0);
    }
    for group_index in 0..groups {
        if group_index != 0 {
            let delta = luma_residual.get(group_index - 1).copied().unwrap_or(0);
            if ((sequence_phase as usize + group_index) & 1) == 0 {
                carrier_a = (carrier_a as i16 + delta as i16 * chroma_quant).clamp(-128, 127) as i8;
            } else {
                carrier_b = (carrier_b as i16 + delta as i16 * chroma_quant).clamp(-128, 127) as i8;
            }
        }
        let (cb, cr) = inverse_rotate_chroma(carrier_a, carrier_b, sequence_phase);
        let start_x = group_index * chroma_group;
        let end_x = (start_x + chroma_group).min(width);
        for x in start_x..end_x {
            let output_x = (x + shift) % width;
            row[output_x * CHANNELS + 1] = (cb as i16 + 128).clamp(0, 255) as u8;
            row[output_x * CHANNELS + 2] = (cr as i16 + 128).clamp(0, 255) as u8;
        }
    }

    if params.history_line_weave != 0
        && is_every(params.history_line_weave_every, line_index as u64 + 1)
    {
        if let Some(history) = weave_reference {
            let row_start = line_index * width * CHANNELS;
            let row_end = row_start + width * CHANNELS;
            row.copy_from_slice(&history[row_start..row_end]);
        }
    }
}

fn scan_line_records(bitstream: &[u8]) -> Vec<LineRecord> {
    let mut records = Vec::new();
    let mut cursor = HEADER_LEN;
    while cursor + LINE_HEADER_LEN <= bitstream.len() {
        if bitstream[cursor..cursor + 2] != LINE_SYNC {
            cursor += 1;
            continue;
        }
        let Ok(luma_len) = read_u16(bitstream, cursor + 10) else {
            break;
        };
        let Ok(chroma_len) = read_u16(bitstream, cursor + 12) else {
            break;
        };
        let luma = cursor + LINE_HEADER_LEN;
        let chroma = luma.saturating_add(luma_len as usize);
        let end = chroma.saturating_add(chroma_len as usize);
        if end > bitstream.len() {
            cursor += 2;
            continue;
        }
        records.push(LineRecord {
            header: cursor,
            luma,
            luma_len: luma_len as usize,
            chroma,
            chroma_len: chroma_len as usize,
        });
        cursor = end;
    }
    records
}

fn validate_stream_header(bitstream: &[u8], config: &ScanlineCodecConfig) -> io::Result<()> {
    if bitstream.len() < HEADER_LEN || &bitstream[..4] != MAGIC {
        return Err(invalid_data("invalid SCN0 stream header"));
    }
    if bitstream[4] != VERSION {
        return Err(invalid_data("unsupported SCN0 stream version"));
    }
    let width = read_u16(bitstream, 6)? as usize;
    let height = read_u16(bitstream, 8)? as usize;
    if width != config.width || height != config.height {
        return Err(invalid_data("SCN0 stream dimensions do not match codec"));
    }
    Ok(())
}

fn encode_zero_rle(values: &[i8], output: &mut Vec<u8>) {
    output.clear();
    let mut index = 0;
    while index < values.len() {
        if values[index] == 0 {
            let mut run = 1;
            while index + run < values.len() && values[index + run] == 0 && run < 128 {
                run += 1;
            }
            output.push(0x80 | (run as u8 - 1));
            index += run;
            continue;
        }

        let start = index;
        index += 1;
        while index < values.len() && values[index] != 0 && index - start < 128 {
            index += 1;
        }
        output.push((index - start - 1) as u8);
        output.extend(values[start..index].iter().map(|value| *value as u8));
    }
}

fn decode_zero_rle(input: &[u8], expected: usize, output: &mut Vec<i8>) {
    output.clear();
    let mut cursor = 0;
    while cursor < input.len() && output.len() < expected {
        let token = input[cursor];
        cursor += 1;
        let run = (token as usize & 0x7f) + 1;
        if token & 0x80 != 0 {
            output.resize((output.len() + run).min(expected), 0);
        } else {
            let available = run.min(input.len().saturating_sub(cursor));
            let needed = available.min(expected - output.len());
            output.extend(
                input[cursor..cursor + needed]
                    .iter()
                    .map(|value| *value as i8),
            );
            cursor += available;
            if available < run {
                break;
            }
        }
    }
    output.resize(expected, 0);
}

fn mutate_rle_run_lengths(payload: &mut [u8], delta: i8) {
    let mut cursor = 0;
    while cursor < payload.len() {
        let token = payload[cursor];
        let original_run = (token & 0x7f) as usize + 1;
        let shifted_run = (original_run as i16 + delta as i16).clamp(1, 128) as u8;
        payload[cursor] = (token & 0x80) | (shifted_run - 1);
        cursor += 1;
        if token & 0x80 == 0 {
            cursor = cursor.saturating_add(original_run);
        }
    }
}

fn encode_chroma_deltas(values: &[i8], output: &mut Vec<u8>) {
    output.clear();
    let mut index = 0;
    while index < values.len() {
        if values[index] == 0 {
            let mut run = 1;
            while index + run < values.len() && values[index + run] == 0 && run < 64 {
                run += 1;
            }
            if run >= 2 {
                output.push(0x80 | (run as u8 - 1));
                index += run;
                continue;
            }
        }

        if index + 1 < values.len()
            && (-4..=3).contains(&values[index])
            && (-4..=3).contains(&values[index + 1])
        {
            let first = (values[index] + 4) as u8;
            let second = (values[index + 1] + 4) as u8;
            output.push((first << 4) | second);
            index += 2;
            continue;
        }

        output.push(0xc0);
        output.push(values[index] as u8);
        index += 1;
    }
}

fn decode_chroma_deltas(input: &[u8], expected: usize, output: &mut Vec<i8>) {
    output.clear();
    let mut cursor = 0;
    while cursor < input.len() && output.len() < expected {
        let token = input[cursor];
        cursor += 1;
        match token {
            0x00..=0x77 => {
                output.push(((token >> 4) as i8) - 4);
                if output.len() < expected {
                    output.push(((token & 0x0f) as i8) - 4);
                }
            }
            0x80..=0xbf => {
                let run = (token as usize & 0x3f) + 1;
                output.resize((output.len() + run).min(expected), 0);
            }
            0xc0 => {
                let Some(value) = input.get(cursor) else {
                    break;
                };
                output.push(*value as i8);
                cursor += 1;
            }
            _ => {
                output.push(0);
            }
        }
    }
    output.resize(expected, 0);
}

struct EncodedLine {
    predictor: Predictor,
    phase: u8,
    chroma_packed: bool,
    luma_payload: Vec<u8>,
    chroma_payload: Vec<u8>,
}

// Free-function encode of a single scanline into local buffers, so the per-line
// work can run in parallel (rayon) without sharing the codec's scratch buffers.
// Logic mirrors choose_predictor + encode_luma_line + encode_chroma_line exactly.
fn encode_line(
    clean: &[u8],
    reference: Option<&[u8]>,
    config: &ScanlineCodecConfig,
    frames_in: u64,
    y: usize,
) -> EncodedLine {
    let width = config.width;
    let phase = ((frames_in as usize + y) & 3) as u8;

    let predictor = match reference {
        Some(reference) => {
            let mut horizontal_cost = 0_u64;
            let mut temporal_cost = 0_u64;
            let mut left = 0_i16;
            for x in 0..width {
                let offset = pixel_offset(width, x, y);
                let current = clean[offset] as i16;
                horizontal_cost += current.abs_diff(left) as u64;
                temporal_cost += current.abs_diff(reference[offset] as i16) as u64;
                left = current;
            }
            if temporal_cost < horizontal_cost {
                Predictor::Temporal
            } else {
                Predictor::Horizontal
            }
        }
        None => Predictor::Horizontal,
    };

    let quant = config.luma_quant as i16;
    let mut luma_residual: Vec<i8> = Vec::with_capacity(width);
    let mut left = 0_i16;
    for x in 0..width {
        let offset = pixel_offset(width, x, y);
        let current = clean[offset] as i16;
        let predicted = match predictor {
            Predictor::Horizontal => left,
            Predictor::Temporal => reference.map_or(0, |frame| frame[offset] as i16),
        };
        let residual = div_round(current - predicted, quant).clamp(-128, 127) as i8;
        luma_residual.push(residual);
        left = (predicted + residual as i16 * quant).clamp(0, 255);
    }
    let mut luma_payload = Vec::new();
    encode_zero_rle(&luma_residual, &mut luma_payload);

    let (chroma_payload, chroma_packed) = encode_chroma_line_local(clean, config, y, phase);

    EncodedLine {
        predictor,
        phase,
        chroma_packed,
        luma_payload,
        chroma_payload,
    }
}

fn encode_chroma_line_local(
    clean: &[u8],
    config: &ScanlineCodecConfig,
    y: usize,
    phase: u8,
) -> (Vec<u8>, bool) {
    let width = config.width;
    let group = config.chroma_group;
    let mut chroma_payload: Vec<u8> = Vec::new();
    let mut chroma_residual: Vec<i8> = Vec::new();
    let mut carrier_a = 0_i8;
    let mut carrier_b = 0_i8;
    for (group_index, start_x) in (0..width).step_by(group).enumerate() {
        let end_x = (start_x + group).min(width);
        let count = (end_x - start_x) as i32;
        let mut cb = 0_i32;
        let mut cr = 0_i32;
        for x in start_x..end_x {
            let offset = pixel_offset(width, x, y);
            cb += clean[offset + 1] as i32 - 128;
            cr += clean[offset + 2] as i32 - 128;
        }
        let cb = (cb / count).clamp(-128, 127) as i8;
        let cr = (cr / count).clamp(-128, 127) as i8;
        let (a, b) = rotate_chroma(cb, cr, phase);
        if group_index == 0 {
            chroma_payload.push(a as u8);
            chroma_payload.push(b as u8);
            carrier_a = a;
            carrier_b = b;
        } else {
            let residual = if ((phase as usize + group_index) & 1) == 0 {
                let residual =
                    div_round(a as i16 - carrier_a as i16, CHROMA_QUANT).clamp(-128, 127) as i8;
                carrier_a =
                    (carrier_a as i16 + residual as i16 * CHROMA_QUANT).clamp(-128, 127) as i8;
                residual
            } else {
                let residual =
                    div_round(b as i16 - carrier_b as i16, CHROMA_QUANT).clamp(-128, 127) as i8;
                carrier_b =
                    (carrier_b as i16 + residual as i16 * CHROMA_QUANT).clamp(-128, 127) as i8;
                residual
            };
            chroma_residual.push(residual);
        }
    }
    let mut chroma_rle = Vec::new();
    encode_chroma_deltas(&chroma_residual, &mut chroma_rle);
    if chroma_rle.len() < chroma_residual.len() {
        chroma_payload.extend_from_slice(&chroma_rle);
        (chroma_payload, true)
    } else {
        chroma_payload.extend(chroma_residual.iter().map(|value| *value as u8));
        (chroma_payload, false)
    }
}

#[inline]
fn px_rgb_to_ycbcr(rgb: &[u8], ycbcr: &mut [u8]) {
    let r = rgb[0] as i32;
    let g = rgb[1] as i32;
    let b = rgb[2] as i32;
    let y = (77 * r + 150 * g + 29 * b + 128) >> 8;
    let cb = ((-43 * r - 85 * g + 128 * b + 128) >> 8) + 128;
    let cr = ((128 * r - 107 * g - 21 * b + 128) >> 8) + 128;
    ycbcr[0] = y.clamp(0, 255) as u8;
    ycbcr[1] = cb.clamp(0, 255) as u8;
    ycbcr[2] = cr.clamp(0, 255) as u8;
}

fn rgb_to_ycbcr(input: &[u8], output: &mut [u8]) {
    if output.len() / CHANNELS >= PARALLEL_FRAME_PIXELS {
        if let Some(pool) = codec_thread_pool() {
            pool.install(|| {
                input
                    .par_chunks_exact(CHANNELS)
                    .zip(output.par_chunks_exact_mut(CHANNELS))
                    .for_each(|(rgb, ycbcr)| px_rgb_to_ycbcr(rgb, ycbcr));
            });
            return;
        }
    }
    input
        .chunks_exact(CHANNELS)
        .zip(output.chunks_exact_mut(CHANNELS))
        .for_each(|(rgb, ycbcr)| px_rgb_to_ycbcr(rgb, ycbcr));
}

#[inline]
fn px_ycbcr_to_rgb(ycbcr: &[u8], rgb: &mut [u8]) {
    let y = ycbcr[0] as i32;
    let cb = ycbcr[1] as i32 - 128;
    let cr = ycbcr[2] as i32 - 128;
    rgb[0] = (y + ((359 * cr + 128) >> 8)).clamp(0, 255) as u8;
    rgb[1] = (y - ((88 * cb + 183 * cr + 128) >> 8)).clamp(0, 255) as u8;
    rgb[2] = (y + ((454 * cb + 128) >> 8)).clamp(0, 255) as u8;
}

fn ycbcr_to_rgb(input: &[u8], output: &mut [u8]) {
    if output.len() / CHANNELS >= PARALLEL_FRAME_PIXELS {
        if let Some(pool) = codec_thread_pool() {
            pool.install(|| {
                input
                    .par_chunks_exact(CHANNELS)
                    .zip(output.par_chunks_exact_mut(CHANNELS))
                    .for_each(|(ycbcr, rgb)| px_ycbcr_to_rgb(ycbcr, rgb));
            });
            return;
        }
    }
    input
        .chunks_exact(CHANNELS)
        .zip(output.chunks_exact_mut(CHANNELS))
        .for_each(|(ycbcr, rgb)| px_ycbcr_to_rgb(ycbcr, rgb));
}

fn rotate_chroma(cb: i8, cr: i8, phase: u8) -> (i8, i8) {
    match phase & 3 {
        0 => (cb, cr),
        1 => (cr, cb.saturating_neg()),
        2 => (cb.saturating_neg(), cr.saturating_neg()),
        _ => (cr.saturating_neg(), cb),
    }
}

fn inverse_rotate_chroma(a: i8, b: i8, phase: u8) -> (i8, i8) {
    match phase & 3 {
        0 => (a, b),
        1 => (b.saturating_neg(), a),
        2 => (a.saturating_neg(), b.saturating_neg()),
        _ => (b, a.saturating_neg()),
    }
}

fn history_reference(history: &VecDeque<Vec<u8>>, lag: usize) -> Option<&Vec<u8>> {
    if history.is_empty() {
        return None;
    }
    let lag = lag.max(1).min(history.len());
    history.get(history.len() - lag)
}

fn recycle_history_frame(
    history: &mut VecDeque<Vec<u8>>,
    current: &mut Vec<u8>,
    history_len: usize,
) {
    let mut replacement = if history.len() >= history_len {
        history
            .pop_front()
            .expect("full history contains a recyclable frame")
    } else {
        vec![0; current.len()]
    };
    std::mem::swap(current, &mut replacement);
    history.push_back(replacement);
}

fn pixel_offset(width: usize, x: usize, y: usize) -> usize {
    (y * width + x) * CHANNELS
}

fn div_round(value: i16, divisor: i16) -> i16 {
    if value >= 0 {
        (value + divisor / 2) / divisor
    } else {
        (value - divisor / 2) / divisor
    }
}

fn rotate_bytes(bytes: &mut [u8], amount: i16) {
    if bytes.len() < 2 || amount == 0 {
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

fn signed_mod(value: i32, modulus: usize) -> usize {
    value.rem_euclid(modulus as i32) as usize
}

fn field_scan_order(height: usize) -> impl Iterator<Item = usize> {
    (0..height).step_by(2).chain((1..height).step_by(2))
}

fn is_field_start(line: usize) -> bool {
    line <= 1
}

fn next_field_scan_line(line: usize, height: usize) -> usize {
    let next = line + 2;
    if next < height {
        next
    } else if line & 1 == 0 && height > 1 {
        1
    } else {
        0
    }
}

fn field_scan_position(line: usize, height: usize) -> usize {
    let even_lines = height.div_ceil(2);
    if line & 1 == 0 {
        line / 2
    } else {
        even_lines + line / 2
    }
}

fn field_scan_distance(from: usize, to: usize, height: usize) -> usize {
    let from = field_scan_position(from, height);
    let to = field_scan_position(to, height);
    (to + height - from) % height
}

fn quarter_phase_delta(actual: u8, expected: u8) -> i16 {
    let delta = (actual as i16 - expected as i16).rem_euclid(4);
    if delta > 2 { delta - 4 } else { delta }
}

fn is_every(period: u64, ordinal: u64) -> bool {
    period != 0 && ordinal % period == 0
}

fn push_u16(output: &mut Vec<u8>, value: u16) {
    output.extend_from_slice(&value.to_le_bytes());
}

fn push_u32(output: &mut Vec<u8>, value: u32) {
    output.extend_from_slice(&value.to_le_bytes());
}

fn push_i16(output: &mut Vec<u8>, value: i16) {
    output.extend_from_slice(&value.to_le_bytes());
}

fn read_u16(input: &[u8], offset: usize) -> io::Result<u16> {
    let bytes = input
        .get(offset..offset + 2)
        .ok_or_else(|| invalid_data("truncated SCN0 integer"))?;
    Ok(u16::from_le_bytes([bytes[0], bytes[1]]))
}

fn write_u16(output: &mut [u8], offset: usize, value: u16) -> io::Result<()> {
    let bytes = output
        .get_mut(offset..offset + 2)
        .ok_or_else(|| invalid_data("truncated SCN0 integer"))?;
    bytes.copy_from_slice(&value.to_le_bytes());
    Ok(())
}

fn read_i16(input: &[u8], offset: usize) -> io::Result<i16> {
    let bytes = input
        .get(offset..offset + 2)
        .ok_or_else(|| invalid_data("truncated SCN0 integer"))?;
    Ok(i16::from_le_bytes([bytes[0], bytes[1]]))
}

fn write_i16(output: &mut [u8], offset: usize, value: i16) -> io::Result<()> {
    let bytes = output
        .get_mut(offset..offset + 2)
        .ok_or_else(|| invalid_data("truncated SCN0 integer"))?;
    bytes.copy_from_slice(&value.to_le_bytes());
    Ok(())
}

fn invalid_input(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, message.into())
}

fn invalid_data(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message.into())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn frame(width: usize, height: usize, phase: usize) -> Vec<u8> {
        let mut frame = vec![0; width * height * CHANNELS];
        for y in 0..height {
            for x in 0..width {
                let offset = pixel_offset(width, x, y);
                frame[offset] = ((x * 13 + phase * 7) & 0xff) as u8;
                frame[offset + 1] = ((y * 19 + phase * 3) & 0xff) as u8;
                frame[offset + 2] = (((x + y) * 11 + phase * 5) & 0xff) as u8;
            }
        }
        frame
    }

    #[test]
    fn zero_rle_round_trips() {
        let values = [0, 0, 1, -2, 0, 3, 4, 0, 0, 0];
        let mut encoded = Vec::new();
        let mut decoded = Vec::new();
        encode_zero_rle(&values, &mut encoded);
        decode_zero_rle(&encoded, values.len(), &mut decoded);
        assert_eq!(decoded, values);
    }

    #[test]
    fn chroma_phase_rotation_round_trips() {
        for phase in 0..4 {
            let encoded = rotate_chroma(48, -37, phase);
            assert_eq!(
                inverse_rotate_chroma(encoded.0, encoded.1, phase),
                (48, -37)
            );
        }
    }

    #[test]
    fn chroma_delta_codes_round_trip_and_pack_small_values() {
        let values = [0, 0, 1, -2, 3, -4, 12, 0, 0, 0, -90];
        let mut encoded = Vec::new();
        let mut decoded = Vec::new();
        encode_chroma_deltas(&values, &mut encoded);
        decode_chroma_deltas(&encoded, values.len(), &mut decoded);
        assert_eq!(decoded, values);
        assert!(encoded.len() < values.len());
    }

    #[test]
    fn clean_codec_preserves_structure_and_compresses_flat_frames() {
        let width = 64;
        let height = 32;
        let mut codec = ScanlineCodec::new(ScanlineCodecConfig::new(width, height)).unwrap();
        let input = vec![96; width * height * CHANNELS];
        let mut output = vec![0; input.len()];
        codec
            .process_rgb_frame(&input, &ScanlineGlitchParams::default(), &mut output)
            .unwrap();

        let max_error = input
            .iter()
            .zip(&output)
            .map(|(left, right)| left.abs_diff(*right))
            .max()
            .unwrap();
        assert!(max_error <= 3);
        assert!(codec.stats().encoded_bytes < codec.stats().raw_bytes);
    }

    #[test]
    fn clean_codec_keeps_color_quantization_error_bounded() {
        let width = 64;
        let height = 32;
        let input = frame(width, height, 9);
        let mut output = vec![0; input.len()];
        let mut codec = ScanlineCodec::new(ScanlineCodecConfig::new(width, height)).unwrap();
        codec
            .process_rgb_frame(&input, &ScanlineGlitchParams::default(), &mut output)
            .unwrap();

        let total_error = input
            .iter()
            .zip(&output)
            .map(|(left, right)| left.abs_diff(*right) as u64)
            .sum::<u64>();
        let mean_error = total_error as f64 / input.len() as f64;
        assert!(mean_error < 20.0, "mean RGB error was {mean_error:.2}");
    }

    #[test]
    fn serialized_phase_damage_changes_output_without_touching_rgb() {
        let width = 48;
        let height = 24;
        let input = frame(width, height, 0);
        let mut clean_codec = ScanlineCodec::new(ScanlineCodecConfig::new(width, height)).unwrap();
        let mut dirty_codec = ScanlineCodec::new(ScanlineCodecConfig::new(width, height)).unwrap();
        let mut clean = vec![0; input.len()];
        let mut dirty = vec![0; input.len()];

        clean_codec
            .process_rgb_frame(&input, &ScanlineGlitchParams::default(), &mut clean)
            .unwrap();
        let params = ScanlineGlitchParams {
            phase_offset: 1,
            ..ScanlineGlitchParams::default()
        };
        let stats = dirty_codec
            .process_rgb_frame(&input, &params, &mut dirty)
            .unwrap();

        assert_ne!(clean, dirty);
        assert_eq!(stats.phases_shifted, height as u64);
    }

    #[test]
    fn chroma_sequence_damage_changes_color_more_than_luma() {
        let width = 64;
        let height = 32;
        let input = frame(width, height, 5);
        let mut clean_codec = ScanlineCodec::new(ScanlineCodecConfig::new(width, height)).unwrap();
        let mut dirty_codec = ScanlineCodec::new(ScanlineCodecConfig::new(width, height)).unwrap();
        let mut clean = vec![0; input.len()];
        let mut dirty = vec![0; input.len()];
        let mut clean_ycbcr = vec![0; input.len()];
        let mut dirty_ycbcr = vec![0; input.len()];

        clean_codec
            .process_rgb_frame(&input, &ScanlineGlitchParams::default(), &mut clean)
            .unwrap();
        let params = ScanlineGlitchParams {
            chroma_sequence_offset: 1,
            chroma_sequence_every: 1,
            ..ScanlineGlitchParams::default()
        };
        dirty_codec
            .process_rgb_frame(&input, &params, &mut dirty)
            .unwrap();
        rgb_to_ycbcr(&clean, &mut clean_ycbcr);
        rgb_to_ycbcr(&dirty, &mut dirty_ycbcr);

        let mut luma_error = 0_u64;
        let mut chroma_error = 0_u64;
        for (clean_pixel, dirty_pixel) in clean_ycbcr
            .chunks_exact(CHANNELS)
            .zip(dirty_ycbcr.chunks_exact(CHANNELS))
        {
            luma_error += clean_pixel[0].abs_diff(dirty_pixel[0]) as u64;
            chroma_error += clean_pixel[1].abs_diff(dirty_pixel[1]) as u64;
            chroma_error += clean_pixel[2].abs_diff(dirty_pixel[2]) as u64;
        }
        assert!(chroma_error > luma_error * 4);
    }

    #[test]
    fn timebase_error_propagates_until_resync_marker() {
        let width = 64;
        let height = 32;
        let input = frame(width, height, 4);
        let mut clean_codec = ScanlineCodec::new(ScanlineCodecConfig::new(width, height)).unwrap();
        let mut dirty_codec = ScanlineCodec::new(ScanlineCodecConfig::new(width, height)).unwrap();
        let mut clean = vec![0; input.len()];
        let mut dirty = vec![0; input.len()];
        clean_codec
            .process_rgb_frame(&input, &ScanlineGlitchParams::default(), &mut clean)
            .unwrap();
        rgb_to_ycbcr(&input, &mut dirty_codec.clean_ycbcr);
        dirty_codec.encode_frame().unwrap();
        let records = scan_line_records(&dirty_codec.bitstream);
        let target = first_record_with_recovery(&dirty_codec.bitstream, &records);
        let recovery = next_resync_index(&dirty_codec.bitstream, &records, target).unwrap();
        write_i16(&mut dirty_codec.bitstream, records[target].header + 6, 11).unwrap();
        dirty_codec
            .decode_frame(&ScanlineGlitchParams::default())
            .unwrap();
        ycbcr_to_rgb(&dirty_codec.decoded_ycbcr, &mut dirty);

        let target_line = record_line(&dirty_codec.bitstream, records[target]);
        let following_line = record_line(&dirty_codec.bitstream, records[target + 1]);
        let recovery_line = record_line(&dirty_codec.bitstream, records[recovery]);
        assert!(row_error(&clean, &dirty, width, target_line) > 0);
        assert!(row_error(&clean, &dirty, width, following_line) > 0);
        assert_eq!(row_error(&clean, &dirty, width, recovery_line), 0);
    }

    #[test]
    fn sync_loss_misaddresses_following_lines_then_recovers() {
        let width = 48;
        let height = 32;
        let input = frame(width, height, 7);
        let mut clean_codec = ScanlineCodec::new(ScanlineCodecConfig::new(width, height)).unwrap();
        let mut dirty_codec = ScanlineCodec::new(ScanlineCodecConfig::new(width, height)).unwrap();
        let mut clean = vec![0; input.len()];
        let mut dirty = vec![0; input.len()];
        clean_codec
            .process_rgb_frame(&input, &ScanlineGlitchParams::default(), &mut clean)
            .unwrap();
        rgb_to_ycbcr(&input, &mut dirty_codec.clean_ycbcr);
        dirty_codec.encode_frame().unwrap();
        let records = scan_line_records(&dirty_codec.bitstream);
        let target = first_record_with_recovery(&dirty_codec.bitstream, &records);
        let recovery = next_resync_index(&dirty_codec.bitstream, &records, target).unwrap();
        dirty_codec.bitstream[records[target].header] = 0;
        dirty_codec.bitstream[records[target].header + 1] = 0;
        dirty_codec
            .decode_frame(&ScanlineGlitchParams::default())
            .unwrap();
        ycbcr_to_rgb(&dirty_codec.decoded_ycbcr, &mut dirty);

        let lost_line = record_line(&dirty_codec.bitstream, records[target]);
        let following_line = record_line(&dirty_codec.bitstream, records[target + 1]);
        let recovery_line = record_line(&dirty_codec.bitstream, records[recovery]);
        assert!(row_error(&clean, &dirty, width, lost_line) > 0);
        assert!(row_error(&clean, &dirty, width, following_line) > 0);
        assert_eq!(row_error(&clean, &dirty, width, recovery_line), 0);
    }

    #[test]
    fn burst_phase_error_propagates_until_resync_marker() {
        let width = 48;
        let height = 32;
        let input = frame(width, height, 11);
        let mut clean_codec = ScanlineCodec::new(ScanlineCodecConfig::new(width, height)).unwrap();
        let mut dirty_codec = ScanlineCodec::new(ScanlineCodecConfig::new(width, height)).unwrap();
        let mut clean = vec![0; input.len()];
        let mut dirty = vec![0; input.len()];
        clean_codec
            .process_rgb_frame(&input, &ScanlineGlitchParams::default(), &mut clean)
            .unwrap();

        rgb_to_ycbcr(&input, &mut dirty_codec.clean_ycbcr);
        dirty_codec.encode_frame().unwrap();
        let records = scan_line_records(&dirty_codec.bitstream);
        let target = first_record_with_recovery(&dirty_codec.bitstream, &records);
        let recovery = next_resync_index(&dirty_codec.bitstream, &records, target).unwrap();
        let record = records[target];
        let flags = dirty_codec.bitstream[record.header + 4];
        let phase = ((flags >> 1) & 3).wrapping_add(1) & 3;
        dirty_codec.bitstream[record.header + 4] = (flags & !0x06) | (phase << 1);
        dirty_codec
            .decode_frame(&ScanlineGlitchParams::default())
            .unwrap();
        ycbcr_to_rgb(&dirty_codec.decoded_ycbcr, &mut dirty);

        let target_line = record_line(&dirty_codec.bitstream, records[target]);
        let following_line = record_line(&dirty_codec.bitstream, records[target + 1]);
        let recovery_line = record_line(&dirty_codec.bitstream, records[recovery]);
        assert!(row_error(&clean, &dirty, width, target_line) > 0);
        assert!(row_error(&clean, &dirty, width, following_line) > 0);
        assert_eq!(row_error(&clean, &dirty, width, recovery_line), 0);
    }

    #[test]
    fn field_order_and_adaptive_resync_markers_are_encoded() {
        let width = 64;
        let height = 48;
        let input = frame(width, height, 2);
        let mut codec = ScanlineCodec::new(ScanlineCodecConfig::new(width, height)).unwrap();
        rgb_to_ycbcr(&input, &mut codec.clean_ycbcr);
        codec.encode_frame().unwrap();
        let records = scan_line_records(&codec.bitstream);
        let lines: Vec<_> = records
            .iter()
            .map(|record| record_line(&codec.bitstream, *record))
            .collect();
        assert_eq!(lines, field_scan_order(height).collect::<Vec<_>>());

        let marker_indices: Vec<_> = records
            .iter()
            .enumerate()
            .filter_map(|(index, record)| {
                (codec.bitstream[record.header + 4] & 0x80 != 0).then_some(index)
            })
            .collect();
        assert!(marker_indices.len() > 2);
        assert!(marker_indices.iter().any(|index| {
            let line = record_line(&codec.bitstream, records[*index]);
            !is_field_start(line)
        }));
        for pair in marker_indices.windows(2) {
            assert!(pair[1] - pair[0] <= RESYNC_MAX_LINES);
        }
        assert!(codec.bitstream[records[0].header + 4] & 0x80 != 0);
        let odd_field = lines.iter().position(|line| *line == 1).unwrap();
        assert!(codec.bitstream[records[odd_field].header + 4] & 0x80 != 0);
    }

    #[test]
    fn field_sync_loss_carries_timebase_error_into_next_field() {
        let width = 64;
        let height = 32;
        let input = frame(width, height, 6);
        let mut clean_codec = ScanlineCodec::new(ScanlineCodecConfig::new(width, height)).unwrap();
        let mut dirty_codec = ScanlineCodec::new(ScanlineCodecConfig::new(width, height)).unwrap();
        let mut clean = vec![0; input.len()];
        let mut dirty = vec![0; input.len()];
        clean_codec
            .process_rgb_frame(&input, &ScanlineGlitchParams::default(), &mut clean)
            .unwrap();

        rgb_to_ycbcr(&input, &mut dirty_codec.clean_ycbcr);
        dirty_codec.encode_frame().unwrap();
        let records = scan_line_records(&dirty_codec.bitstream);
        let odd_field = records
            .iter()
            .position(|record| record_line(&dirty_codec.bitstream, *record) == 1)
            .unwrap();
        write_i16(
            &mut dirty_codec.bitstream,
            records[odd_field - 1].header + 6,
            9,
        )
        .unwrap();
        dirty_codec.bitstream[records[odd_field].header + 4] &= !0x80;
        dirty_codec
            .decode_frame(&ScanlineGlitchParams::default())
            .unwrap();
        ycbcr_to_rgb(&dirty_codec.decoded_ycbcr, &mut dirty);

        assert!(row_error(&clean, &dirty, width, 1) > 0);
        let recovery = next_resync_index(&dirty_codec.bitstream, &records, odd_field).unwrap();
        let recovery_line = record_line(&dirty_codec.bitstream, records[recovery]);
        assert_eq!(row_error(&clean, &dirty, width, recovery_line), 0);
    }

    #[test]
    fn sync_loss_conceals_lines_from_decoder_history() {
        let width = 32;
        let height = 16;
        let mut codec = ScanlineCodec::new(ScanlineCodecConfig::new(width, height)).unwrap();
        let first = frame(width, height, 0);
        let second = frame(width, height, 7);
        let mut output = vec![0; first.len()];
        codec
            .process_rgb_frame(&first, &ScanlineGlitchParams::default(), &mut output)
            .unwrap();
        let params = ScanlineGlitchParams {
            sync_loss_every: 2,
            ..ScanlineGlitchParams::default()
        };
        let stats = codec
            .process_rgb_frame(&second, &params, &mut output)
            .unwrap();

        assert_eq!(stats.sync_words_lost, (height / 2) as u64);
        assert!(codec.stats().concealed_lines >= (height / 2) as u64);
    }

    #[test]
    fn reset_makes_temporal_predictor_start_cleanly() {
        let width = 32;
        let height = 16;
        let mut codec = ScanlineCodec::new(ScanlineCodecConfig::new(width, height)).unwrap();
        let input = frame(width, height, 3);
        let mut output = vec![0; input.len()];
        codec
            .process_rgb_frame(&input, &ScanlineGlitchParams::default(), &mut output)
            .unwrap();
        codec.reset_glitch_state();
        codec
            .process_rgb_frame(&input, &ScanlineGlitchParams::default(), &mut output)
            .unwrap();
        assert_eq!(codec.encoder_history.len(), 1);
        assert_eq!(codec.decoder_history.len(), 1);
    }

    fn record_line(bitstream: &[u8], record: LineRecord) -> usize {
        read_u16(bitstream, record.header + 2).unwrap() as usize
    }

    fn next_resync_index(bitstream: &[u8], records: &[LineRecord], after: usize) -> Option<usize> {
        records
            .iter()
            .enumerate()
            .skip(after + 1)
            .find_map(|(index, record)| (bitstream[record.header + 4] & 0x80 != 0).then_some(index))
    }

    fn first_record_with_recovery(bitstream: &[u8], records: &[LineRecord]) -> usize {
        (1..records.len().saturating_sub(2))
            .find(|index| {
                bitstream[records[*index].header + 4] & 0x80 == 0
                    && next_resync_index(bitstream, records, *index)
                        .is_some_and(|recovery| recovery > *index + 1)
            })
            .expect("encoded stream should contain a non-marker line before a later marker")
    }

    fn row_error(left: &[u8], right: &[u8], width: usize, row: usize) -> u64 {
        let start = row * width * CHANNELS;
        let end = start + width * CHANNELS;
        left[start..end]
            .iter()
            .zip(&right[start..end])
            .map(|(a, b)| a.abs_diff(*b) as u64)
            .sum()
    }
}
