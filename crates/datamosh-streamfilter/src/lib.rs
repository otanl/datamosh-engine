// FFmpeg / Annex B elementary-stream datamosh filter.
//
// Split out of the original `datamosh` crate during the workspace refactor.
// Depends only on `std`.

use std::collections::VecDeque;
use std::io::{self, Read, Write};
use std::time::{Duration, Instant};

pub const REPORT_INTERVAL: Duration = Duration::from_secs(1);

const READ_CHUNK_SIZE: usize = 64 * 1024;

const MAX_PREFIXLESS_BUFFER: usize = 1024 * 1024;

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum Codec {
    H264,
    Hevc,
    Mpeg4,
    Mpeg1,
    Mpeg2,
}

impl Codec {
    pub fn parse(value: &str) -> Result<Self, String> {
        match value {
            "h264" | "avc" => Ok(Self::H264),
            "hevc" | "h265" | "h.265" => Ok(Self::Hevc),
            "mpeg4" | "mpeg4-asp" | "mpeg4asp" | "asp" | "m4v" | "xvid" | "divx" => Ok(Self::Mpeg4),
            "mpeg1" | "mpeg1video" | "mpg" => Ok(Self::Mpeg1),
            "mpeg2" | "mpeg2video" | "mpv" => Ok(Self::Mpeg2),
            _ => Err(format!(
                "unsupported codec `{value}`; expected h264, hevc, mpeg4, mpeg1, or mpeg2"
            )),
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::H264 => "h264",
            Self::Hevc => "hevc",
            Self::Mpeg4 => "mpeg4",
            Self::Mpeg1 => "mpeg1",
            Self::Mpeg2 => "mpeg2",
        }
    }
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum MpegSliceDropMode {
    All,
    Key,
    Predicted,
}

impl MpegSliceDropMode {
    pub fn parse(value: &str) -> Result<Self, String> {
        match value {
            "all" => Ok(Self::All),
            "key" | "i" | "intra" => Ok(Self::Key),
            "predicted" | "inter" | "p" | "b" => Ok(Self::Predicted),
            _ => Err(format!(
                "unsupported MPEG slice drop mode `{value}`; expected all, key, or predicted"
            )),
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::All => "all",
            Self::Key => "key",
            Self::Predicted => "predicted",
        }
    }
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum FrameTypeRewrite {
    I,
    P,
    B,
    S,
    D,
}

impl FrameTypeRewrite {
    pub fn parse(value: &str) -> Result<Self, String> {
        match value {
            "i" | "key" | "intra" => Ok(Self::I),
            "p" | "predicted" => Ok(Self::P),
            "b" | "bidirectional" => Ok(Self::B),
            "s" | "sprite" => Ok(Self::S),
            "d" | "dc" => Ok(Self::D),
            _ => Err(format!(
                "unsupported frame type rewrite `{value}`; expected i, p, b, s, or d"
            )),
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::I => "i",
            Self::P => "p",
            Self::B => "b",
            Self::S => "s",
            Self::D => "d",
        }
    }
}

#[derive(Debug, Clone)]
pub struct Config {
    pub codec: Codec,
    pub drop_idr_after: u64,
    pub recover_every: u64,
    pub drop_slice_every: u64,
    pub damage_slice_every: u64,
    pub damage_amount: usize,
    pub truncate_slice_every: u64,
    pub truncate_amount: usize,
    pub scramble_slice_every: u64,
    pub scramble_amount: usize,
    pub rotate_slice_every: u64,
    pub rotate_amount: usize,
    pub splice_slice_every: u64,
    pub splice_amount: usize,
    pub grow_slice_every: u64,
    pub grow_amount: usize,
    pub donor_bank_size: usize,
    pub donor_splice_slice_every: u64,
    pub donor_splice_amount: usize,
    pub donor_grow_slice_every: u64,
    pub donor_grow_amount: usize,
    pub donor_xor_slice_every: u64,
    pub donor_xor_amount: usize,
    pub donor_replace_slice_every: u64,
    pub rewrite_frame_type_every: u64,
    pub rewrite_frame_type_to: FrameTypeRewrite,
    pub shift_slice_address_every: u64,
    pub shift_slice_address_by: i16,
    pub drop_mpeg_slice_address_every: u8,
    pub drop_mpeg_slice_address_phase: u8,
    pub drop_mpeg_slice_address_mode: MpegSliceDropMode,
    pub xor_slice_every: u64,
    pub xor_amount: usize,
    pub echo_slice_every: u64,
    pub echo_count: u64,
    pub replace_slice_every: u64,
    pub repeat_slice_every: u64,
    pub repeat_count: u64,
    pub drop_headers_after_first: bool,
    pub quiet: bool,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            codec: Codec::H264,
            drop_idr_after: 1,
            recover_every: 0,
            drop_slice_every: 0,
            damage_slice_every: 0,
            damage_amount: 4,
            truncate_slice_every: 0,
            truncate_amount: 16,
            scramble_slice_every: 0,
            scramble_amount: 16,
            rotate_slice_every: 0,
            rotate_amount: 8,
            splice_slice_every: 0,
            splice_amount: 32,
            grow_slice_every: 0,
            grow_amount: 8,
            donor_bank_size: 16,
            donor_splice_slice_every: 0,
            donor_splice_amount: 32,
            donor_grow_slice_every: 0,
            donor_grow_amount: 8,
            donor_xor_slice_every: 0,
            donor_xor_amount: 16,
            donor_replace_slice_every: 0,
            rewrite_frame_type_every: 0,
            rewrite_frame_type_to: FrameTypeRewrite::P,
            shift_slice_address_every: 0,
            shift_slice_address_by: 1,
            drop_mpeg_slice_address_every: 0,
            drop_mpeg_slice_address_phase: 0,
            drop_mpeg_slice_address_mode: MpegSliceDropMode::All,
            xor_slice_every: 0,
            xor_amount: 16,
            echo_slice_every: 0,
            echo_count: 1,
            replace_slice_every: 0,
            repeat_slice_every: 0,
            repeat_count: 1,
            drop_headers_after_first: false,
            quiet: false,
        }
    }
}

#[derive(Debug, Default)]
pub struct Stats {
    pub nals_in: u64,
    pub nals_out: u64,
    pub idr_seen: u64,
    pub idr_passed: u64,
    pub idr_dropped: u64,
    pub slices_seen: u64,
    pub slices_dropped: u64,
    pub slices_damaged: u64,
    pub slices_truncated: u64,
    pub slices_scrambled: u64,
    pub slices_rotated: u64,
    pub slices_spliced: u64,
    pub slices_grown: u64,
    pub donor_units_seen: u64,
    pub donor_units_stored: u64,
    pub slices_donor_spliced: u64,
    pub slices_donor_grown: u64,
    pub slices_donor_xored: u64,
    pub slices_donor_replaced: u64,
    pub frame_types_rewritten: u64,
    pub slice_addresses_shifted: u64,
    pub slice_addresses_dropped: u64,
    pub slices_xored: u64,
    pub slices_echoed: u64,
    pub slices_replaced: u64,
    pub slices_repeated: u64,
    pub headers_seen: u64,
    pub headers_dropped: u64,
    pub bytes_in: u64,
    pub bytes_out: u64,
}

#[derive(Clone)]
struct LastPredictedUnit {
    data: Vec<u8>,
    payload_start: usize,
}

pub struct MoshFilter {
    config: Config,
    stats: Stats,
    seen_vps: bool,
    seen_sps: bool,
    seen_pps: bool,
    seen_mpeg4_headers: [bool; 256],
    seen_mpeg2_headers: [bool; 256],
    mpeg2_current_picture: Option<Mpeg2PictureType>,
    mpeg2_skip_picture: bool,
    donor_mpeg2_current_picture: Option<Mpeg2PictureType>,
    frame_type_index: u64,
    mpeg_slice_index: u64,
    last_predicted_unit: Option<LastPredictedUnit>,
    donor_units: VecDeque<LastPredictedUnit>,
}

impl MoshFilter {
    pub fn new(config: Config) -> Self {
        Self {
            config,
            stats: Stats::default(),
            seen_vps: false,
            seen_sps: false,
            seen_pps: false,
            seen_mpeg4_headers: [false; 256],
            seen_mpeg2_headers: [false; 256],
            mpeg2_current_picture: None,
            mpeg2_skip_picture: false,
            donor_mpeg2_current_picture: None,
            frame_type_index: 0,
            mpeg_slice_index: 0,
            last_predicted_unit: None,
            donor_units: VecDeque::new(),
        }
    }

    pub fn config(&self) -> &Config {
        &self.config
    }

    pub fn config_mut(&mut self) -> &mut Config {
        &mut self.config
    }

    pub fn stats(&self) -> &Stats {
        &self.stats
    }

    pub fn process_unit(&mut self, unit: &[u8], out: &mut dyn Write) -> io::Result<()> {
        self.stats.nals_in += 1;
        self.stats.bytes_in += unit.len() as u64;

        match self.config.codec {
            Codec::H264 => self.process_h264_unit(unit, out),
            Codec::Hevc => self.process_hevc_unit(unit, out),
            Codec::Mpeg4 => self.process_mpeg4_unit(unit, out),
            Codec::Mpeg1 => self.process_mpeg2_unit(unit, out),
            Codec::Mpeg2 => self.process_mpeg2_unit(unit, out),
        }
    }

    pub fn process_donor_unit(&mut self, unit: &[u8]) -> bool {
        self.stats.donor_units_seen += 1;
        let Some(payload_start) = self.donor_payload_start(unit) else {
            return false;
        };

        self.store_donor_unit(unit, payload_start);
        true
    }

    fn donor_payload_start(&mut self, unit: &[u8]) -> Option<usize> {
        match self.config.codec {
            Codec::H264 => {
                if nal_unit_type(unit) == Some(1) {
                    let prefix_len = prefix_len_at(unit, 0)?;
                    Some(prefix_len + 1)
                } else {
                    None
                }
            }
            Codec::Hevc => match hevc_nal_unit_type(unit) {
                Some(0..=15) => {
                    let prefix_len = prefix_len_at(unit, 0)?;
                    Some(prefix_len + 2)
                }
                _ => None,
            },
            Codec::Mpeg4 => {
                if mpeg4_start_code(unit) == Some(0xb6)
                    && matches!(
                        mpeg4_vop_type(unit),
                        Some(Mpeg4VopType::P | Mpeg4VopType::B | Mpeg4VopType::S)
                    )
                {
                    let prefix_len = prefix_len_at(unit, 0)?;
                    Some((prefix_len + 8).min(unit.len()))
                } else {
                    None
                }
            }
            Codec::Mpeg1 | Codec::Mpeg2 => match mpeg_start_code(unit) {
                Some(0x00) => {
                    self.donor_mpeg2_current_picture = mpeg2_picture_type(unit);
                    None
                }
                Some(0x01..=0xaf)
                    if matches!(
                        self.donor_mpeg2_current_picture,
                        Some(Mpeg2PictureType::P | Mpeg2PictureType::B | Mpeg2PictureType::D)
                    ) =>
                {
                    let prefix_len = prefix_len_at(unit, 0)?;
                    Some((prefix_len + 2).min(unit.len()))
                }
                _ => None,
            },
        }
    }

    fn store_donor_unit(&mut self, unit: &[u8], payload_start: usize) {
        if self.config.donor_bank_size == 0 || payload_start >= unit.len() {
            return;
        }

        while self.donor_units.len() >= self.config.donor_bank_size {
            self.donor_units.pop_front();
        }

        self.donor_units.push_back(LastPredictedUnit {
            data: unit.to_vec(),
            payload_start,
        });
        self.stats.donor_units_stored += 1;
    }

    fn donor_unit(&self, seed: u64) -> Option<(Vec<u8>, usize)> {
        if self.donor_units.is_empty() {
            return None;
        }

        let index = seed
            .wrapping_mul(1_103_515_245)
            .wrapping_add(self.donor_units.len() as u64) as usize
            % self.donor_units.len();
        let unit = self.donor_units.get(index)?;
        Some((unit.data.clone(), unit.payload_start))
    }

    fn maybe_rewrite_mpeg4_vop_type(&mut self, unit: &[u8]) -> Option<Vec<u8>> {
        mpeg4_vop_type(unit)?;
        self.frame_type_index += 1;
        if !is_every(self.config.rewrite_frame_type_every, self.frame_type_index) {
            return None;
        }

        let mut edited = unit.to_vec();
        if rewrite_mpeg4_vop_type(&mut edited, self.config.rewrite_frame_type_to) {
            self.stats.frame_types_rewritten += 1;
            Some(edited)
        } else {
            None
        }
    }

    fn maybe_rewrite_mpeg2_picture_type(&mut self, unit: &[u8]) -> Option<Vec<u8>> {
        mpeg2_picture_type(unit)?;
        self.frame_type_index += 1;
        if !is_every(self.config.rewrite_frame_type_every, self.frame_type_index) {
            return None;
        }

        let mut edited = unit.to_vec();
        if rewrite_mpeg2_picture_type(&mut edited, self.config.rewrite_frame_type_to) {
            self.stats.frame_types_rewritten += 1;
            Some(edited)
        } else {
            None
        }
    }

    fn process_h264_unit(&mut self, unit: &[u8], out: &mut dyn Write) -> io::Result<()> {
        let nal_type = nal_unit_type(unit);
        let mut pass = true;

        match nal_type {
            Some(1) => {
                let prefix_len = prefix_len_at(unit, 0).expect("H.264 unit starts with a prefix");
                return self.process_predicted_unit(unit, prefix_len + 1, out);
            }
            Some(5) => {
                return self.process_keyframe_unit(unit, out);
            }
            Some(7) => {
                self.stats.headers_seen += 1;
                if self.config.drop_headers_after_first && self.seen_sps {
                    pass = false;
                    self.stats.headers_dropped += 1;
                }
                self.seen_sps = true;
            }
            Some(8) => {
                self.stats.headers_seen += 1;
                if self.config.drop_headers_after_first && self.seen_pps {
                    pass = false;
                    self.stats.headers_dropped += 1;
                }
                self.seen_pps = true;
            }
            _ => {}
        }

        if pass {
            self.write_unit(unit, 0, out)?;
        }

        Ok(())
    }

    fn process_hevc_unit(&mut self, unit: &[u8], out: &mut dyn Write) -> io::Result<()> {
        let nal_type = hevc_nal_unit_type(unit);
        let mut pass = true;

        match nal_type {
            Some(0..=15) => {
                let prefix_len = prefix_len_at(unit, 0).expect("HEVC unit starts with a prefix");
                return self.process_predicted_unit(unit, prefix_len + 2, out);
            }
            Some(16..=23) => {
                return self.process_keyframe_unit(unit, out);
            }
            Some(32) => {
                self.stats.headers_seen += 1;
                if self.config.drop_headers_after_first && self.seen_vps {
                    pass = false;
                    self.stats.headers_dropped += 1;
                }
                self.seen_vps = true;
            }
            Some(33) => {
                self.stats.headers_seen += 1;
                if self.config.drop_headers_after_first && self.seen_sps {
                    pass = false;
                    self.stats.headers_dropped += 1;
                }
                self.seen_sps = true;
            }
            Some(34) => {
                self.stats.headers_seen += 1;
                if self.config.drop_headers_after_first && self.seen_pps {
                    pass = false;
                    self.stats.headers_dropped += 1;
                }
                self.seen_pps = true;
            }
            _ => {}
        }

        if pass {
            self.write_unit(unit, 0, out)?;
        }

        Ok(())
    }

    fn process_mpeg4_unit(&mut self, unit: &[u8], out: &mut dyn Write) -> io::Result<()> {
        let Some(start_code) = mpeg4_start_code(unit) else {
            return self.write_unit(unit, 0, out);
        };

        if start_code == 0xb6 {
            let vop_type = mpeg4_vop_type(unit);
            let rewritten_unit = self.maybe_rewrite_mpeg4_vop_type(unit);
            let unit = rewritten_unit.as_deref().unwrap_or(unit);

            match vop_type {
                Some(Mpeg4VopType::I) => self.process_keyframe_unit(unit, out),
                Some(Mpeg4VopType::P | Mpeg4VopType::B | Mpeg4VopType::S) => {
                    let prefix_len =
                        prefix_len_at(unit, 0).expect("MPEG-4 unit starts with a prefix");
                    self.process_predicted_unit(unit, (prefix_len + 8).min(unit.len()), out)
                }
                None => self.write_unit(unit, 0, out),
            }
        } else {
            if is_mpeg4_header_code(start_code) {
                self.stats.headers_seen += 1;
                let index = start_code as usize;
                if self.config.drop_headers_after_first && self.seen_mpeg4_headers[index] {
                    self.stats.headers_dropped += 1;
                    return Ok(());
                }
                self.seen_mpeg4_headers[index] = true;
            }

            self.write_unit(unit, 0, out)
        }
    }

    fn process_mpeg2_unit(&mut self, unit: &[u8], out: &mut dyn Write) -> io::Result<()> {
        let Some(start_code) = mpeg_start_code(unit) else {
            return self.write_unit(unit, 0, out);
        };

        match start_code {
            0x00 => {
                let Some(picture_type) = mpeg2_picture_type(unit) else {
                    self.mpeg2_current_picture = None;
                    self.mpeg2_skip_picture = false;
                    return self.write_unit(unit, 0, out);
                };

                self.mpeg2_current_picture = Some(picture_type);
                let rewritten_unit = self.maybe_rewrite_mpeg2_picture_type(unit);
                let unit = rewritten_unit.as_deref().unwrap_or(unit);

                if picture_type == Mpeg2PictureType::I {
                    self.stats.idr_seen += 1;
                    if self.should_pass_keyframe() {
                        self.stats.idr_passed += 1;
                        self.mpeg2_skip_picture = false;
                        self.write_unit(unit, 0, out)
                    } else {
                        self.stats.idr_dropped += 1;
                        self.mpeg2_skip_picture = true;
                        Ok(())
                    }
                } else {
                    self.mpeg2_skip_picture = false;
                    self.write_unit(unit, 0, out)
                }
            }
            0x01..=0xaf => {
                if self.mpeg2_skip_picture {
                    return Ok(());
                }

                self.mpeg_slice_index += 1;
                if self.should_drop_mpeg_slice_address(start_code) {
                    self.stats.slice_addresses_dropped += 1;
                    return Ok(());
                }

                let mut shifted_unit = None;
                if is_every(self.config.shift_slice_address_every, self.mpeg_slice_index) {
                    let mut shifted = unit.to_vec();
                    if shift_mpeg_slice_address(&mut shifted, self.config.shift_slice_address_by) {
                        self.stats.slice_addresses_shifted += 1;
                        shifted_unit = Some(shifted);
                    }
                }
                let unit = shifted_unit.as_deref().unwrap_or(unit);

                match self.mpeg2_current_picture {
                    Some(Mpeg2PictureType::P | Mpeg2PictureType::B | Mpeg2PictureType::D) => {
                        let prefix_len =
                            prefix_len_at(unit, 0).expect("MPEG-2 slice starts with a prefix");
                        self.process_predicted_unit(unit, (prefix_len + 2).min(unit.len()), out)
                    }
                    _ => self.write_unit(unit, 0, out),
                }
            }
            0xb5 => {
                if self.mpeg2_skip_picture {
                    Ok(())
                } else {
                    self.write_unit(unit, 0, out)
                }
            }
            _ => {
                if is_mpeg2_header_code(start_code) {
                    self.stats.headers_seen += 1;
                    let index = start_code as usize;
                    if self.config.drop_headers_after_first && self.seen_mpeg2_headers[index] {
                        self.stats.headers_dropped += 1;
                        return Ok(());
                    }
                    self.seen_mpeg2_headers[index] = true;
                }

                self.write_unit(unit, 0, out)
            }
        }
    }

    fn process_keyframe_unit(&mut self, unit: &[u8], out: &mut dyn Write) -> io::Result<()> {
        self.stats.idr_seen += 1;
        if self.should_pass_keyframe() {
            self.stats.idr_passed += 1;
            self.write_unit(unit, 0, out)
        } else {
            self.stats.idr_dropped += 1;
            Ok(())
        }
    }

    fn process_predicted_unit(
        &mut self,
        unit: &[u8],
        payload_start: usize,
        out: &mut dyn Write,
    ) -> io::Result<()> {
        self.stats.slices_seen += 1;
        if is_every(self.config.drop_slice_every, self.stats.slices_seen) {
            self.stats.slices_dropped += 1;
            return Ok(());
        }

        let previous = self
            .last_predicted_unit
            .as_ref()
            .map(|last| (last.data.clone(), last.payload_start));
        let donor = self.donor_unit(self.stats.slices_seen);
        let should_damage = is_every(self.config.damage_slice_every, self.stats.slices_seen);
        let should_truncate = is_every(self.config.truncate_slice_every, self.stats.slices_seen);
        let should_scramble = is_every(self.config.scramble_slice_every, self.stats.slices_seen);
        let should_rotate = is_every(self.config.rotate_slice_every, self.stats.slices_seen);
        let should_splice =
            is_every(self.config.splice_slice_every, self.stats.slices_seen) && previous.is_some();
        let should_grow =
            is_every(self.config.grow_slice_every, self.stats.slices_seen) && previous.is_some();
        let should_donor_splice =
            is_every(self.config.donor_splice_slice_every, self.stats.slices_seen)
                && donor.is_some();
        let should_donor_grow =
            is_every(self.config.donor_grow_slice_every, self.stats.slices_seen) && donor.is_some();
        let should_donor_xor =
            is_every(self.config.donor_xor_slice_every, self.stats.slices_seen) && donor.is_some();
        let should_xor = is_every(self.config.xor_slice_every, self.stats.slices_seen);
        let should_echo = is_every(self.config.echo_slice_every, self.stats.slices_seen);
        let should_replace =
            is_every(self.config.replace_slice_every, self.stats.slices_seen) && previous.is_some();
        let should_donor_replace = is_every(
            self.config.donor_replace_slice_every,
            self.stats.slices_seen,
        ) && donor.is_some()
            && !should_replace;
        let extra_copies = if is_every(self.config.repeat_slice_every, self.stats.slices_seen) {
            self.stats.slices_repeated += self.config.repeat_count;
            self.config.repeat_count
        } else {
            0
        };

        if should_echo {
            if let Some((previous_data, _)) = &previous {
                for _ in 0..self.config.echo_count {
                    self.write_unit(previous_data, 0, out)?;
                }
                self.stats.slices_echoed += self.config.echo_count;
            }
        }

        let mut edited;
        let mut edited_payload_start = payload_start;

        if should_replace {
            let (previous_data, previous_payload_start) =
                previous.as_ref().expect("previous unit exists for replace");
            edited = previous_data.clone();
            edited_payload_start = *previous_payload_start;
            self.stats.slices_replaced += 1;
        } else if should_donor_replace {
            let (donor_data, donor_payload_start) =
                donor.as_ref().expect("donor unit exists for replace");
            edited = donor_data.clone();
            edited_payload_start = *donor_payload_start;
            self.stats.slices_donor_replaced += 1;
        } else if should_damage
            || should_truncate
            || should_scramble
            || should_rotate
            || should_splice
            || should_grow
            || should_donor_splice
            || should_donor_grow
            || should_donor_xor
            || should_xor
        {
            edited = unit.to_vec();
        } else {
            self.last_predicted_unit = Some(LastPredictedUnit {
                data: unit.to_vec(),
                payload_start,
            });
            return self.write_unit(unit, extra_copies, out);
        }

        if should_damage
            || should_truncate
            || should_scramble
            || should_rotate
            || should_splice
            || should_grow
            || should_donor_splice
            || should_donor_grow
            || should_donor_xor
            || should_xor
        {
            let seed = self.stats.slices_seen;

            if should_donor_xor {
                if let Some((donor_data, donor_payload_start)) = &donor {
                    if xor_payload(
                        &mut edited,
                        edited_payload_start,
                        donor_data,
                        *donor_payload_start,
                        self.config.donor_xor_amount,
                    ) {
                        self.stats.slices_donor_xored += 1;
                    }
                }
            }
            if should_xor {
                if let Some((previous_data, previous_payload_start)) = &previous {
                    if xor_payload(
                        &mut edited,
                        edited_payload_start,
                        previous_data,
                        *previous_payload_start,
                        self.config.xor_amount,
                    ) {
                        self.stats.slices_xored += 1;
                    }
                }
            }
            if should_donor_splice {
                if let Some((donor_data, donor_payload_start)) = &donor {
                    if splice_payload(
                        &mut edited,
                        edited_payload_start,
                        donor_data,
                        *donor_payload_start,
                        seed.wrapping_mul(13),
                        self.config.donor_splice_amount,
                    ) {
                        self.stats.slices_donor_spliced += 1;
                    }
                }
            }
            if should_splice {
                if let Some((previous_data, previous_payload_start)) = &previous {
                    if splice_payload(
                        &mut edited,
                        edited_payload_start,
                        previous_data,
                        *previous_payload_start,
                        seed,
                        self.config.splice_amount,
                    ) {
                        self.stats.slices_spliced += 1;
                    }
                }
            }
            if should_damage {
                self.stats.slices_damaged += 1;
                damage_payload(
                    &mut edited,
                    edited_payload_start,
                    seed,
                    self.config.damage_amount,
                );
            }
            if should_scramble {
                self.stats.slices_scrambled += 1;
                scramble_payload(
                    &mut edited,
                    edited_payload_start,
                    seed,
                    self.config.scramble_amount,
                );
            }
            if should_rotate
                && rotate_payload(
                    &mut edited,
                    edited_payload_start,
                    seed,
                    self.config.rotate_amount,
                )
            {
                self.stats.slices_rotated += 1;
            }
            if should_grow {
                if let Some((previous_data, previous_payload_start)) = &previous {
                    if grow_payload(
                        &mut edited,
                        edited_payload_start,
                        previous_data,
                        *previous_payload_start,
                        seed,
                        self.config.grow_amount,
                    ) {
                        self.stats.slices_grown += 1;
                    }
                }
            }
            if should_donor_grow {
                if let Some((donor_data, donor_payload_start)) = &donor {
                    if grow_payload(
                        &mut edited,
                        edited_payload_start,
                        donor_data,
                        *donor_payload_start,
                        seed.wrapping_mul(17),
                        self.config.donor_grow_amount,
                    ) {
                        self.stats.slices_donor_grown += 1;
                    }
                }
            }
            if should_truncate
                && truncate_payload(
                    &mut edited,
                    edited_payload_start,
                    self.config.truncate_amount,
                )
            {
                self.stats.slices_truncated += 1;
            }
        }

        self.last_predicted_unit = Some(LastPredictedUnit {
            data: unit.to_vec(),
            payload_start,
        });
        self.write_unit(&edited, extra_copies, out)
    }

    fn write_unit(
        &mut self,
        unit: &[u8],
        extra_copies: u64,
        out: &mut dyn Write,
    ) -> io::Result<()> {
        for _ in 0..=extra_copies {
            out.write_all(unit)?;
        }

        let copies = 1 + extra_copies;
        self.stats.nals_out += copies;
        self.stats.bytes_out += unit.len() as u64 * copies;
        Ok(())
    }

    fn should_pass_keyframe(&self) -> bool {
        if self.stats.idr_seen <= self.config.drop_idr_after {
            return true;
        }

        self.config.recover_every != 0
            && (self.stats.idr_seen - self.config.drop_idr_after) % self.config.recover_every == 0
    }

    fn should_drop_mpeg_slice_address(&self, address: u8) -> bool {
        let every = self.config.drop_mpeg_slice_address_every;
        if every == 0 || !(0x01..=0xaf).contains(&address) {
            return false;
        }

        let picture_matches = match self.config.drop_mpeg_slice_address_mode {
            MpegSliceDropMode::All => true,
            MpegSliceDropMode::Key => self.mpeg2_current_picture == Some(Mpeg2PictureType::I),
            MpegSliceDropMode::Predicted => matches!(
                self.mpeg2_current_picture,
                Some(Mpeg2PictureType::P | Mpeg2PictureType::B | Mpeg2PictureType::D)
            ),
        };
        if !picture_matches {
            return false;
        }

        let phase = self.config.drop_mpeg_slice_address_phase % every;
        ((address - 1) % every) == phase
    }

    fn report(&self, err: &mut dyn Write, final_report: bool) -> io::Result<()> {
        if self.config.quiet {
            return Ok(());
        }

        let prefix = if final_report {
            "datamosh: final"
        } else {
            "datamosh"
        };

        writeln!(
            err,
            "{prefix}: codec {}  units {}/{}  key seen/pass/drop {}/{}/{}  predicted seen/drop/damage/truncate/scramble/rotate/splice/grow/address_shift/address_drop/xor/echo/replace/repeat {}/{}/{}/{}/{}/{}/{}/{}/{}/{}/{}/{}/{}/{}  donor seen/stored/splice/grow/xor/replace {}/{}/{}/{}/{}/{}  frame_type_rewrite {}  headers drop {}  bytes {}/{}",
            self.config.codec.name(),
            self.stats.nals_out,
            self.stats.nals_in,
            self.stats.idr_seen,
            self.stats.idr_passed,
            self.stats.idr_dropped,
            self.stats.slices_seen,
            self.stats.slices_dropped,
            self.stats.slices_damaged,
            self.stats.slices_truncated,
            self.stats.slices_scrambled,
            self.stats.slices_rotated,
            self.stats.slices_spliced,
            self.stats.slices_grown,
            self.stats.slice_addresses_shifted,
            self.stats.slice_addresses_dropped,
            self.stats.slices_xored,
            self.stats.slices_echoed,
            self.stats.slices_replaced,
            self.stats.slices_repeated,
            self.stats.donor_units_seen,
            self.stats.donor_units_stored,
            self.stats.slices_donor_spliced,
            self.stats.slices_donor_grown,
            self.stats.slices_donor_xored,
            self.stats.slices_donor_replaced,
            self.stats.frame_types_rewritten,
            self.stats.headers_dropped,
            self.stats.bytes_out,
            self.stats.bytes_in
        )
    }
}

pub struct DatamoshStream {
    filter: MoshFilter,
    buffer: Vec<u8>,
    donor_buffer: Vec<u8>,
}

impl DatamoshStream {
    pub fn new(config: Config) -> Self {
        Self {
            filter: MoshFilter::new(config),
            buffer: Vec::with_capacity(READ_CHUNK_SIZE * 2),
            donor_buffer: Vec::with_capacity(READ_CHUNK_SIZE * 2),
        }
    }

    pub fn filter(&self) -> &MoshFilter {
        &self.filter
    }

    pub fn filter_mut(&mut self) -> &mut MoshFilter {
        &mut self.filter
    }

    pub fn config(&self) -> &Config {
        self.filter.config()
    }

    pub fn config_mut(&mut self) -> &mut Config {
        self.filter.config_mut()
    }

    pub fn stats(&self) -> &Stats {
        self.filter.stats()
    }

    pub fn process_chunk(&mut self, chunk: &[u8], output: &mut dyn Write) -> io::Result<()> {
        self.buffer.extend_from_slice(chunk);
        drain_complete_nals(&mut self.buffer, &mut self.filter, output)
    }

    pub fn process_donor_chunk(&mut self, chunk: &[u8]) -> io::Result<()> {
        self.donor_buffer.extend_from_slice(chunk);
        drain_complete_donor_nals(&mut self.donor_buffer, &mut self.filter)
    }

    pub fn finish(&mut self, output: &mut dyn Write) -> io::Result<()> {
        drain_remaining_nal(&mut self.buffer, &mut self.filter, output)
    }

    pub fn finish_donor(&mut self) -> io::Result<()> {
        drain_remaining_donor_nal(&mut self.donor_buffer, &mut self.filter)
    }
}

pub fn run_stream(
    config: Config,
    mut input: impl Read,
    mut output: impl Write,
    mut err: impl Write,
) -> io::Result<()> {
    let mut stream = DatamoshStream::new(config);
    run_stream_inner(&mut stream, &mut input, &mut output, &mut err)
}

pub fn load_donor_stream(stream: &mut DatamoshStream, mut input: impl Read) -> io::Result<()> {
    let mut chunk = [0_u8; READ_CHUNK_SIZE];

    loop {
        let read = input.read(&mut chunk)?;
        if read == 0 {
            break;
        }

        stream.process_donor_chunk(&chunk[..read])?;
    }

    stream.finish_donor()
}

pub fn run_stream_inner(
    stream: &mut DatamoshStream,
    input: &mut dyn Read,
    output: &mut dyn Write,
    err: &mut dyn Write,
) -> io::Result<()> {
    let mut chunk = [0_u8; READ_CHUNK_SIZE];
    let mut last_report = Instant::now();

    loop {
        let read = input.read(&mut chunk)?;
        if read == 0 {
            break;
        }

        stream.process_chunk(&chunk[..read], output)?;

        if last_report.elapsed() >= REPORT_INTERVAL {
            stream.filter.report(err, false)?;
            last_report = Instant::now();
        }
    }

    stream.finish(output)?;
    output.flush()?;
    stream.filter.report(err, true)?;
    Ok(())
}

fn drain_complete_nals(
    buffer: &mut Vec<u8>,
    filter: &mut MoshFilter,
    output: &mut dyn Write,
) -> io::Result<()> {
    loop {
        let Some((first_start, _)) = find_start_code(buffer, 0) else {
            trim_prefixless_buffer(buffer);
            return Ok(());
        };

        if first_start > 0 {
            buffer.drain(..first_start);
        }

        let prefix_len = prefix_len_at(buffer, 0).expect("buffer starts with a start code");
        let Some((next_start, _)) = find_start_code(buffer, prefix_len) else {
            return Ok(());
        };

        let nal: Vec<u8> = buffer.drain(..next_start).collect();
        if nal.len() > prefix_len {
            filter.process_unit(&nal, output)?;
        }
    }
}

fn drain_complete_donor_nals(buffer: &mut Vec<u8>, filter: &mut MoshFilter) -> io::Result<()> {
    loop {
        let Some((first_start, _)) = find_start_code(buffer, 0) else {
            trim_prefixless_buffer(buffer);
            return Ok(());
        };

        if first_start > 0 {
            buffer.drain(..first_start);
        }

        let prefix_len = prefix_len_at(buffer, 0).expect("buffer starts with a start code");
        let Some((next_start, _)) = find_start_code(buffer, prefix_len) else {
            return Ok(());
        };

        let nal: Vec<u8> = buffer.drain(..next_start).collect();
        if nal.len() > prefix_len {
            filter.process_donor_unit(&nal);
        }
    }
}

fn drain_remaining_nal(
    buffer: &mut Vec<u8>,
    filter: &mut MoshFilter,
    output: &mut dyn Write,
) -> io::Result<()> {
    let Some((first_start, _)) = find_start_code(buffer, 0) else {
        buffer.clear();
        return Ok(());
    };

    if first_start > 0 {
        buffer.drain(..first_start);
    }

    if !buffer.is_empty() {
        let nal = std::mem::take(buffer);
        filter.process_unit(&nal, output)?;
    }

    Ok(())
}

fn drain_remaining_donor_nal(buffer: &mut Vec<u8>, filter: &mut MoshFilter) -> io::Result<()> {
    let Some((first_start, _)) = find_start_code(buffer, 0) else {
        buffer.clear();
        return Ok(());
    };

    if first_start > 0 {
        buffer.drain(..first_start);
    }

    if !buffer.is_empty() {
        let nal = std::mem::take(buffer);
        filter.process_donor_unit(&nal);
    }

    Ok(())
}

fn trim_prefixless_buffer(buffer: &mut Vec<u8>) {
    if buffer.len() <= MAX_PREFIXLESS_BUFFER {
        return;
    }

    let keep = buffer.len().min(3);
    let drain_to = buffer.len() - keep;
    buffer.drain(..drain_to);
}

fn is_every(interval: u64, count: u64) -> bool {
    interval != 0 && count % interval == 0
}

fn damage_payload(unit: &mut [u8], payload_start: usize, seed: u64, amount: usize) {
    if amount == 0 || payload_start >= unit.len() {
        return;
    }

    let payload_len = unit.len() - payload_start;
    let mut state = seed
        .wrapping_mul(1_103_515_245)
        .wrapping_add(payload_len as u64);

    for _ in 0..amount.min(payload_len) {
        state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        let offset = (state as usize) % payload_len;
        let index = payload_start + offset;
        let mask = 1_u8 << (state as u8 & 0x07);
        unit[index] ^= mask;
        if unit[index] == 0 {
            unit[index] = mask;
        }
    }
}

fn scramble_payload(unit: &mut [u8], payload_start: usize, seed: u64, amount: usize) {
    if amount < 2 || payload_start + 1 >= unit.len() {
        return;
    }

    let payload_len = unit.len() - payload_start;
    let active_len = amount.min(payload_len);
    if active_len < 2 {
        return;
    }

    let rotate_by = (seed as usize % (active_len - 1)) + 1;
    unit[payload_start..payload_start + active_len].rotate_left(rotate_by);

    let mut state = seed
        .wrapping_mul(2_654_435_761)
        .wrapping_add(active_len as u64);
    for i in 0..active_len {
        state = state.wrapping_mul(22_695_477).wrapping_add(1);
        let a = payload_start + i;
        let b = payload_start + ((state as usize) % active_len);
        unit.swap(a, b);
    }
}

fn rotate_payload(unit: &mut [u8], payload_start: usize, seed: u64, amount: usize) -> bool {
    if amount < 2 || payload_start + 1 >= unit.len() {
        return false;
    }

    let payload_len = unit.len() - payload_start;
    let active_len = amount.min(payload_len);
    if active_len < 2 {
        return false;
    }

    let start_range = payload_len - active_len + 1;
    let start = payload_start + (seed as usize % start_range);
    let rotate_by = ((seed as usize / 3) % (active_len - 1)) + 1;
    unit[start..start + active_len].rotate_right(rotate_by);
    true
}

fn splice_payload(
    unit: &mut [u8],
    payload_start: usize,
    previous: &[u8],
    previous_payload_start: usize,
    seed: u64,
    amount: usize,
) -> bool {
    if amount == 0 || payload_start >= unit.len() || previous_payload_start >= previous.len() {
        return false;
    }

    let payload_len = unit.len() - payload_start;
    let previous_len = previous.len() - previous_payload_start;
    let active_len = amount.min(payload_len).min(previous_len);
    if active_len == 0 {
        return false;
    }

    let dst_range = payload_len - active_len + 1;
    let src_range = previous_len - active_len + 1;
    let dst = payload_start + (seed.wrapping_mul(3) as usize % dst_range);
    let src = previous_payload_start + (seed.wrapping_mul(5).wrapping_add(1) as usize % src_range);
    unit[dst..dst + active_len].copy_from_slice(&previous[src..src + active_len]);
    true
}

fn grow_payload(
    unit: &mut Vec<u8>,
    payload_start: usize,
    previous: &[u8],
    previous_payload_start: usize,
    seed: u64,
    amount: usize,
) -> bool {
    if amount == 0 || payload_start > unit.len() || previous_payload_start >= previous.len() {
        return false;
    }

    let previous_len = previous.len() - previous_payload_start;
    let active_len = amount.min(previous_len);
    if active_len == 0 {
        return false;
    }

    let src_range = previous_len - active_len + 1;
    let src = previous_payload_start + (seed.wrapping_mul(7).wrapping_add(3) as usize % src_range);
    let payload_len = unit.len().saturating_sub(payload_start);
    let insert_at = payload_start + (seed.wrapping_mul(11) as usize % (payload_len + 1));
    let insert = previous[src..src + active_len].to_vec();
    unit.splice(insert_at..insert_at, insert);
    true
}

fn truncate_payload(unit: &mut Vec<u8>, payload_start: usize, amount: usize) -> bool {
    if amount == 0 || payload_start >= unit.len() {
        return false;
    }

    let payload_len = unit.len() - payload_start;
    if payload_len <= 1 {
        return false;
    }

    let remove = amount.min(payload_len - 1);
    let new_len = unit.len() - remove;
    unit.truncate(new_len);
    true
}

fn xor_payload(
    unit: &mut [u8],
    payload_start: usize,
    previous: &[u8],
    previous_payload_start: usize,
    amount: usize,
) -> bool {
    if amount == 0 || payload_start >= unit.len() || previous_payload_start >= previous.len() {
        return false;
    }

    let payload_len = unit.len() - payload_start;
    let previous_len = previous.len() - previous_payload_start;
    let active_len = amount.min(payload_len).min(previous_len);
    if active_len == 0 {
        return false;
    }

    for i in 0..active_len {
        let previous_byte = previous[previous_payload_start + i];
        let index = payload_start + i;
        unit[index] ^= previous_byte.rotate_left((i & 7) as u32);
        if unit[index] == 0 {
            unit[index] = previous_byte | 1;
        }
    }

    true
}

fn nal_unit_type(nal: &[u8]) -> Option<u8> {
    let prefix_len = prefix_len_at(nal, 0)?;
    nal.get(prefix_len).map(|byte| byte & 0x1f)
}

fn hevc_nal_unit_type(nal: &[u8]) -> Option<u8> {
    let prefix_len = prefix_len_at(nal, 0)?;
    nal.get(prefix_len).map(|byte| (byte >> 1) & 0x3f)
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
enum Mpeg4VopType {
    I,
    P,
    B,
    S,
}

fn mpeg4_start_code(unit: &[u8]) -> Option<u8> {
    let prefix_len = prefix_len_at(unit, 0)?;
    unit.get(prefix_len).copied()
}

fn mpeg4_vop_type(unit: &[u8]) -> Option<Mpeg4VopType> {
    if mpeg4_start_code(unit)? != 0xb6 {
        return None;
    }

    let prefix_len = prefix_len_at(unit, 0)?;
    let bits = unit.get(prefix_len + 1)? >> 6;
    match bits {
        0 => Some(Mpeg4VopType::I),
        1 => Some(Mpeg4VopType::P),
        2 => Some(Mpeg4VopType::B),
        3 => Some(Mpeg4VopType::S),
        _ => None,
    }
}

fn rewrite_mpeg4_vop_type(unit: &mut [u8], target: FrameTypeRewrite) -> bool {
    if mpeg4_start_code(unit) != Some(0xb6) {
        return false;
    }

    let Some(prefix_len) = prefix_len_at(unit, 0) else {
        return false;
    };
    let Some(byte) = unit.get_mut(prefix_len + 1) else {
        return false;
    };

    let bits = match target {
        FrameTypeRewrite::I => 0,
        FrameTypeRewrite::P => 1,
        FrameTypeRewrite::B => 2,
        FrameTypeRewrite::S | FrameTypeRewrite::D => 3,
    };
    *byte = (*byte & 0x3f) | (bits << 6);
    true
}

fn is_mpeg4_header_code(code: u8) -> bool {
    matches!(code, 0xb0 | 0xb2 | 0xb3 | 0xb5) || (0x20..=0x2f).contains(&code)
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
enum Mpeg2PictureType {
    I,
    P,
    B,
    D,
}

fn mpeg_start_code(unit: &[u8]) -> Option<u8> {
    let prefix_len = prefix_len_at(unit, 0)?;
    unit.get(prefix_len).copied()
}

fn shift_mpeg_slice_address(unit: &mut [u8], offset: i16) -> bool {
    let Some(prefix_len) = prefix_len_at(unit, 0) else {
        return false;
    };
    let Some(code) = unit.get_mut(prefix_len) else {
        return false;
    };
    if !(0x01..=0xaf).contains(code) {
        return false;
    }

    let span = 0xaf_i16;
    let shifted = ((*code as i16 - 1) + offset).rem_euclid(span) + 1;
    *code = shifted as u8;
    true
}

fn rewrite_mpeg2_picture_type(unit: &mut [u8], target: FrameTypeRewrite) -> bool {
    if mpeg_start_code(unit) != Some(0x00) {
        return false;
    }

    let Some(prefix_len) = prefix_len_at(unit, 0) else {
        return false;
    };
    let Some(byte) = unit.get_mut(prefix_len + 2) else {
        return false;
    };

    let bits = match target {
        FrameTypeRewrite::I => 1,
        FrameTypeRewrite::P => 2,
        FrameTypeRewrite::B => 3,
        FrameTypeRewrite::D | FrameTypeRewrite::S => 4,
    };
    *byte = (*byte & !0x38) | (bits << 3);
    true
}

fn mpeg2_picture_type(unit: &[u8]) -> Option<Mpeg2PictureType> {
    if mpeg_start_code(unit)? != 0x00 {
        return None;
    }

    let prefix_len = prefix_len_at(unit, 0)?;
    let code = (unit.get(prefix_len + 2)? >> 3) & 0x07;
    match code {
        1 => Some(Mpeg2PictureType::I),
        2 => Some(Mpeg2PictureType::P),
        3 => Some(Mpeg2PictureType::B),
        4 => Some(Mpeg2PictureType::D),
        _ => None,
    }
}

fn is_mpeg2_header_code(code: u8) -> bool {
    matches!(code, 0xb2 | 0xb3 | 0xb7 | 0xb8)
}

fn prefix_len_at(bytes: &[u8], at: usize) -> Option<usize> {
    if at + 4 <= bytes.len()
        && bytes[at] == 0
        && bytes[at + 1] == 0
        && bytes[at + 2] == 0
        && bytes[at + 3] == 1
    {
        return Some(4);
    }

    if at + 3 <= bytes.len() && bytes[at] == 0 && bytes[at + 1] == 0 && bytes[at + 2] == 1 {
        return Some(3);
    }

    None
}

fn find_start_code(bytes: &[u8], from: usize) -> Option<(usize, usize)> {
    let mut i = from;
    while i + 3 <= bytes.len() {
        if let Some(prefix_len) = prefix_len_at(bytes, i) {
            return Some((i, prefix_len));
        }
        i += 1;
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_three_and_four_byte_start_codes() {
        assert_eq!(find_start_code(&[9, 0, 0, 1, 5], 0), Some((1, 3)));
        assert_eq!(find_start_code(&[0, 0, 0, 1, 7], 0), Some((0, 4)));
    }

    #[test]
    fn drops_idr_after_initial_allowance() {
        let mut filter = MoshFilter::new(Config {
            quiet: true,
            ..Config::default()
        });
        let mut out = Vec::new();

        filter.process_unit(&[0, 0, 1, 0x65, 1], &mut out).unwrap();
        filter.process_unit(&[0, 0, 1, 0x41, 2], &mut out).unwrap();
        filter.process_unit(&[0, 0, 1, 0x65, 3], &mut out).unwrap();

        assert_eq!(out, vec![0, 0, 1, 0x65, 1, 0, 0, 1, 0x41, 2]);
        assert_eq!(filter.stats.idr_seen, 2);
        assert_eq!(filter.stats.idr_dropped, 1);
    }

    #[test]
    fn can_periodically_recover_on_later_idr() {
        let mut filter = MoshFilter::new(Config {
            drop_idr_after: 1,
            recover_every: 2,
            quiet: true,
            ..Config::default()
        });
        let mut out = Vec::new();

        filter.process_unit(&[0, 0, 1, 0x65, 1], &mut out).unwrap();
        filter.process_unit(&[0, 0, 1, 0x65, 2], &mut out).unwrap();
        filter.process_unit(&[0, 0, 1, 0x65, 3], &mut out).unwrap();

        assert_eq!(out, vec![0, 0, 1, 0x65, 1, 0, 0, 1, 0x65, 3]);
        assert_eq!(filter.stats.idr_passed, 2);
        assert_eq!(filter.stats.idr_dropped, 1);
    }

    #[test]
    fn detects_hevc_nal_unit_types() {
        assert_eq!(hevc_nal_unit_type(&[0, 0, 1, 0x02, 0x01]), Some(1));
        assert_eq!(hevc_nal_unit_type(&[0, 0, 1, 0x26, 0x01]), Some(19));
        assert_eq!(hevc_nal_unit_type(&[0, 0, 1, 0x40, 0x01]), Some(32));
    }

    #[test]
    fn hevc_drops_later_irap_units_and_keeps_predicted_units() {
        let mut filter = MoshFilter::new(Config {
            codec: Codec::Hevc,
            quiet: true,
            ..Config::default()
        });
        let mut out = Vec::new();

        filter
            .process_unit(&[0, 0, 1, 0x26, 0x01, 1], &mut out)
            .unwrap();
        filter
            .process_unit(&[0, 0, 1, 0x02, 0x01, 2], &mut out)
            .unwrap();
        filter
            .process_unit(&[0, 0, 1, 0x26, 0x01, 3], &mut out)
            .unwrap();

        assert_eq!(out, vec![0, 0, 1, 0x26, 0x01, 1, 0, 0, 1, 0x02, 0x01, 2]);
        assert_eq!(filter.stats.idr_seen, 2);
        assert_eq!(filter.stats.idr_dropped, 1);
        assert_eq!(filter.stats.slices_seen, 1);
    }

    #[test]
    fn hevc_drops_repeated_vps_sps_pps_when_requested() {
        let mut filter = MoshFilter::new(Config {
            codec: Codec::Hevc,
            drop_headers_after_first: true,
            quiet: true,
            ..Config::default()
        });
        let mut out = Vec::new();

        filter
            .process_unit(&[0, 0, 1, 0x40, 0x01, 1], &mut out)
            .unwrap();
        filter
            .process_unit(&[0, 0, 1, 0x42, 0x01, 2], &mut out)
            .unwrap();
        filter
            .process_unit(&[0, 0, 1, 0x44, 0x01, 3], &mut out)
            .unwrap();
        filter
            .process_unit(&[0, 0, 1, 0x40, 0x01, 4], &mut out)
            .unwrap();
        filter
            .process_unit(&[0, 0, 1, 0x42, 0x01, 5], &mut out)
            .unwrap();
        filter
            .process_unit(&[0, 0, 1, 0x44, 0x01, 6], &mut out)
            .unwrap();

        assert_eq!(
            out,
            vec![
                0, 0, 1, 0x40, 0x01, 1, 0, 0, 1, 0x42, 0x01, 2, 0, 0, 1, 0x44, 0x01, 3
            ]
        );
        assert_eq!(filter.stats.headers_dropped, 3);
    }

    #[test]
    fn drops_every_nth_non_idr_slice() {
        let mut filter = MoshFilter::new(Config {
            drop_slice_every: 2,
            quiet: true,
            ..Config::default()
        });
        let mut out = Vec::new();

        filter.process_unit(&[0, 0, 1, 0x41, 1], &mut out).unwrap();
        filter.process_unit(&[0, 0, 1, 0x41, 2], &mut out).unwrap();
        filter.process_unit(&[0, 0, 1, 0x41, 3], &mut out).unwrap();

        assert_eq!(out, vec![0, 0, 1, 0x41, 1, 0, 0, 1, 0x41, 3]);
        assert_eq!(filter.stats.slices_seen, 3);
        assert_eq!(filter.stats.slices_dropped, 1);
    }

    #[test]
    fn damages_non_idr_slice_payload_without_touching_header() {
        let mut filter = MoshFilter::new(Config {
            damage_slice_every: 1,
            damage_amount: 2,
            quiet: true,
            ..Config::default()
        });
        let mut out = Vec::new();
        let nal = [0, 0, 1, 0x41, 0xaa, 0xbb, 0xcc, 0xdd];

        filter.process_unit(&nal, &mut out).unwrap();

        assert_eq!(&out[..4], &[0, 0, 1, 0x41]);
        assert_ne!(out, nal);
        assert_eq!(out.len(), nal.len());
        assert_eq!(filter.stats.slices_damaged, 1);
    }

    #[test]
    fn repeats_every_nth_non_idr_slice() {
        let mut filter = MoshFilter::new(Config {
            repeat_slice_every: 2,
            repeat_count: 2,
            quiet: true,
            ..Config::default()
        });
        let mut out = Vec::new();

        filter.process_unit(&[0, 0, 1, 0x41, 1], &mut out).unwrap();
        filter.process_unit(&[0, 0, 1, 0x41, 2], &mut out).unwrap();

        assert_eq!(
            out,
            vec![
                0, 0, 1, 0x41, 1, 0, 0, 1, 0x41, 2, 0, 0, 1, 0x41, 2, 0, 0, 1, 0x41, 2
            ]
        );
        assert_eq!(filter.stats.nals_out, 4);
        assert_eq!(filter.stats.slices_repeated, 2);
    }

    #[test]
    fn truncates_predicted_slice_payload() {
        let mut filter = MoshFilter::new(Config {
            truncate_slice_every: 1,
            truncate_amount: 2,
            quiet: true,
            ..Config::default()
        });
        let mut out = Vec::new();

        filter
            .process_unit(&[0, 0, 1, 0x41, 1, 2, 3, 4], &mut out)
            .unwrap();

        assert_eq!(out, vec![0, 0, 1, 0x41, 1, 2]);
        assert_eq!(filter.stats.slices_truncated, 1);
    }

    #[test]
    fn scrambles_predicted_slice_payload_without_changing_size() {
        let mut filter = MoshFilter::new(Config {
            scramble_slice_every: 1,
            scramble_amount: 4,
            quiet: true,
            ..Config::default()
        });
        let mut out = Vec::new();
        let input = [0, 0, 1, 0x41, 1, 2, 3, 4, 5];

        filter.process_unit(&input, &mut out).unwrap();

        assert_eq!(&out[..4], &[0, 0, 1, 0x41]);
        assert_eq!(out.len(), input.len());
        assert_ne!(out, input);
        assert_eq!(filter.stats.slices_scrambled, 1);
    }

    #[test]
    fn rotates_predicted_slice_payload_region() {
        let mut filter = MoshFilter::new(Config {
            rotate_slice_every: 1,
            rotate_amount: 4,
            quiet: true,
            ..Config::default()
        });
        let mut out = Vec::new();

        filter
            .process_unit(&[0, 0, 1, 0x41, 1, 2, 3, 4, 5], &mut out)
            .unwrap();

        assert_eq!(out, vec![0, 0, 1, 0x41, 1, 5, 2, 3, 4]);
        assert_eq!(filter.stats.slices_rotated, 1);
    }

    #[test]
    fn splices_previous_payload_into_current_payload() {
        let mut filter = MoshFilter::new(Config {
            splice_slice_every: 2,
            splice_amount: 2,
            quiet: true,
            ..Config::default()
        });
        let mut out = Vec::new();

        filter
            .process_unit(&[0, 0, 1, 0x41, 10, 11, 12, 13], &mut out)
            .unwrap();
        filter
            .process_unit(&[0, 0, 1, 0x41, 1, 2, 3, 4], &mut out)
            .unwrap();

        assert_eq!(
            out,
            vec![0, 0, 1, 0x41, 10, 11, 12, 13, 0, 0, 1, 0x41, 12, 13, 3, 4]
        );
        assert_eq!(filter.stats.slices_spliced, 1);
    }

    #[test]
    fn grows_current_payload_with_previous_payload_bytes() {
        let mut filter = MoshFilter::new(Config {
            grow_slice_every: 2,
            grow_amount: 2,
            quiet: true,
            ..Config::default()
        });
        let mut out = Vec::new();

        filter
            .process_unit(&[0, 0, 1, 0x41, 10, 11, 12, 13], &mut out)
            .unwrap();
        filter
            .process_unit(&[0, 0, 1, 0x41, 1, 2, 3, 4], &mut out)
            .unwrap();

        assert_eq!(
            out,
            vec![
                0, 0, 1, 0x41, 10, 11, 12, 13, 0, 0, 1, 0x41, 1, 2, 12, 13, 3, 4
            ]
        );
        assert_eq!(filter.stats.slices_grown, 1);
    }

    #[test]
    fn splices_donor_payload_into_current_payload() {
        let mut filter = MoshFilter::new(Config {
            donor_splice_slice_every: 1,
            donor_splice_amount: 2,
            quiet: true,
            ..Config::default()
        });
        let mut out = Vec::new();

        assert!(filter.process_donor_unit(&[0, 0, 1, 0x41, 10, 11, 12, 13]));
        filter
            .process_unit(&[0, 0, 1, 0x41, 1, 2, 3, 4], &mut out)
            .unwrap();

        assert_eq!(out, vec![0, 0, 1, 0x41, 10, 11, 3, 4]);
        assert_eq!(filter.stats.donor_units_seen, 1);
        assert_eq!(filter.stats.donor_units_stored, 1);
        assert_eq!(filter.stats.slices_donor_spliced, 1);
    }

    #[test]
    fn replaces_current_unit_with_donor_unit() {
        let mut filter = MoshFilter::new(Config {
            donor_replace_slice_every: 1,
            quiet: true,
            ..Config::default()
        });
        let mut out = Vec::new();

        assert!(filter.process_donor_unit(&[0, 0, 1, 0x41, 10, 11]));
        filter
            .process_unit(&[0, 0, 1, 0x41, 1, 2], &mut out)
            .unwrap();

        assert_eq!(out, vec![0, 0, 1, 0x41, 10, 11]);
        assert_eq!(filter.stats.slices_donor_replaced, 1);
    }

    #[test]
    fn stream_processor_accepts_chunked_donor_input() {
        let mut stream = DatamoshStream::new(Config {
            donor_grow_slice_every: 1,
            donor_grow_amount: 2,
            quiet: true,
            ..Config::default()
        });
        let mut out = Vec::new();

        stream.process_donor_chunk(&[0, 0, 1, 0x41, 10]).unwrap();
        stream.process_donor_chunk(&[11, 12, 13]).unwrap();
        stream.finish_donor().unwrap();
        stream
            .process_chunk(&[0, 0, 1, 0x41, 1, 2, 3, 4], &mut out)
            .unwrap();
        stream.finish(&mut out).unwrap();

        assert_eq!(out, vec![0, 0, 1, 0x41, 1, 2, 12, 13, 3, 4]);
        assert_eq!(stream.stats().slices_donor_grown, 1);
    }

    #[test]
    fn xors_predicted_payload_with_previous_unit() {
        let mut filter = MoshFilter::new(Config {
            xor_slice_every: 2,
            xor_amount: 2,
            quiet: true,
            ..Config::default()
        });
        let mut out = Vec::new();

        filter
            .process_unit(&[0, 0, 1, 0x41, 0x01, 0x02], &mut out)
            .unwrap();
        filter
            .process_unit(&[0, 0, 1, 0x41, 0x04, 0x08], &mut out)
            .unwrap();

        assert_eq!(
            out,
            vec![0, 0, 1, 0x41, 0x01, 0x02, 0, 0, 1, 0x41, 0x05, 0x0c]
        );
        assert_eq!(filter.stats.slices_xored, 1);
    }

    #[test]
    fn echoes_previous_predicted_unit_before_current_unit() {
        let mut filter = MoshFilter::new(Config {
            echo_slice_every: 2,
            echo_count: 1,
            quiet: true,
            ..Config::default()
        });
        let mut out = Vec::new();

        filter.process_unit(&[0, 0, 1, 0x41, 1], &mut out).unwrap();
        filter.process_unit(&[0, 0, 1, 0x41, 2], &mut out).unwrap();

        assert_eq!(
            out,
            vec![0, 0, 1, 0x41, 1, 0, 0, 1, 0x41, 1, 0, 0, 1, 0x41, 2]
        );
        assert_eq!(filter.stats.slices_echoed, 1);
    }

    #[test]
    fn replaces_predicted_unit_with_previous_unit() {
        let mut filter = MoshFilter::new(Config {
            replace_slice_every: 2,
            quiet: true,
            ..Config::default()
        });
        let mut out = Vec::new();

        filter.process_unit(&[0, 0, 1, 0x41, 1], &mut out).unwrap();
        filter.process_unit(&[0, 0, 1, 0x41, 2], &mut out).unwrap();

        assert_eq!(out, vec![0, 0, 1, 0x41, 1, 0, 0, 1, 0x41, 1]);
        assert_eq!(filter.stats.slices_replaced, 1);
    }

    #[test]
    fn detects_mpeg4_vop_types() {
        assert_eq!(
            mpeg4_vop_type(&[0, 0, 1, 0xb6, 0b0000_0000]),
            Some(Mpeg4VopType::I)
        );
        assert_eq!(
            mpeg4_vop_type(&[0, 0, 1, 0xb6, 0b0100_0000]),
            Some(Mpeg4VopType::P)
        );
        assert_eq!(
            mpeg4_vop_type(&[0, 0, 1, 0xb6, 0b1000_0000]),
            Some(Mpeg4VopType::B)
        );
    }

    #[test]
    fn mpeg4_drops_later_i_vops_and_keeps_p_vops() {
        let mut filter = MoshFilter::new(Config {
            codec: Codec::Mpeg4,
            quiet: true,
            ..Config::default()
        });
        let mut out = Vec::new();

        filter
            .process_unit(&[0, 0, 1, 0xb6, 0b0000_0000, 1], &mut out)
            .unwrap();
        filter
            .process_unit(&[0, 0, 1, 0xb6, 0b0100_0000, 2], &mut out)
            .unwrap();
        filter
            .process_unit(&[0, 0, 1, 0xb6, 0b0000_0000, 3], &mut out)
            .unwrap();

        assert_eq!(
            out,
            vec![0, 0, 1, 0xb6, 0b0000_0000, 1, 0, 0, 1, 0xb6, 0b0100_0000, 2]
        );
        assert_eq!(filter.stats.idr_seen, 2);
        assert_eq!(filter.stats.idr_dropped, 1);
        assert_eq!(filter.stats.slices_seen, 1);
    }

    #[test]
    fn mpeg4_drops_repeated_headers_when_requested() {
        let mut filter = MoshFilter::new(Config {
            codec: Codec::Mpeg4,
            drop_headers_after_first: true,
            quiet: true,
            ..Config::default()
        });
        let mut out = Vec::new();

        filter.process_unit(&[0, 0, 1, 0x20, 1], &mut out).unwrap();
        filter.process_unit(&[0, 0, 1, 0x20, 2], &mut out).unwrap();

        assert_eq!(out, vec![0, 0, 1, 0x20, 1]);
        assert_eq!(filter.stats.headers_dropped, 1);
    }

    #[test]
    fn parses_mpeg4_asp_aliases() {
        assert_eq!(Codec::parse("xvid"), Ok(Codec::Mpeg4));
        assert_eq!(Codec::parse("divx"), Ok(Codec::Mpeg4));
        assert_eq!(Codec::parse("mpeg4-asp"), Ok(Codec::Mpeg4));
    }

    #[test]
    fn detects_mpeg2_picture_types() {
        assert_eq!(
            mpeg2_picture_type(&[0, 0, 1, 0x00, 0x00, 0x08]),
            Some(Mpeg2PictureType::I)
        );
        assert_eq!(
            mpeg2_picture_type(&[0, 0, 1, 0x00, 0x00, 0x10]),
            Some(Mpeg2PictureType::P)
        );
        assert_eq!(
            mpeg2_picture_type(&[0, 0, 1, 0x00, 0x00, 0x18]),
            Some(Mpeg2PictureType::B)
        );
    }

    #[test]
    fn shifts_mpeg_slice_address_with_wrapping() {
        let mut unit = vec![0, 0, 1, 0x02, 0xaa];
        assert!(shift_mpeg_slice_address(&mut unit, 3));
        assert_eq!(unit, vec![0, 0, 1, 0x05, 0xaa]);

        let mut unit = vec![0, 0, 1, 0x01, 0xaa];
        assert!(shift_mpeg_slice_address(&mut unit, -1));
        assert_eq!(unit, vec![0, 0, 1, 0xaf, 0xaa]);
    }

    #[test]
    fn parses_mpeg_slice_drop_modes() {
        assert_eq!(MpegSliceDropMode::parse("all"), Ok(MpegSliceDropMode::All));
        assert_eq!(MpegSliceDropMode::parse("i"), Ok(MpegSliceDropMode::Key));
        assert_eq!(
            MpegSliceDropMode::parse("predicted"),
            Ok(MpegSliceDropMode::Predicted)
        );
    }

    #[test]
    fn parses_frame_type_rewrite_modes() {
        assert_eq!(FrameTypeRewrite::parse("i"), Ok(FrameTypeRewrite::I));
        assert_eq!(
            FrameTypeRewrite::parse("predicted"),
            Ok(FrameTypeRewrite::P)
        );
        assert_eq!(FrameTypeRewrite::parse("b"), Ok(FrameTypeRewrite::B));
        assert_eq!(FrameTypeRewrite::parse("sprite"), Ok(FrameTypeRewrite::S));
        assert_eq!(FrameTypeRewrite::parse("dc"), Ok(FrameTypeRewrite::D));
    }

    #[test]
    fn rewrites_mpeg4_vop_type_header_bits() {
        let mut filter = MoshFilter::new(Config {
            codec: Codec::Mpeg4,
            drop_idr_after: 99,
            rewrite_frame_type_every: 1,
            rewrite_frame_type_to: FrameTypeRewrite::B,
            quiet: true,
            ..Config::default()
        });
        let mut out = Vec::new();

        filter
            .process_unit(&[0, 0, 1, 0xb6, 0b0000_0000, 1], &mut out)
            .unwrap();

        assert_eq!(out, vec![0, 0, 1, 0xb6, 0b1000_0000, 1]);
        assert_eq!(filter.stats.frame_types_rewritten, 1);
        assert_eq!(filter.stats.idr_seen, 1);
    }

    #[test]
    fn rewrites_mpeg2_picture_type_header_bits() {
        let mut filter = MoshFilter::new(Config {
            codec: Codec::Mpeg2,
            rewrite_frame_type_every: 1,
            rewrite_frame_type_to: FrameTypeRewrite::B,
            quiet: true,
            ..Config::default()
        });
        let mut out = Vec::new();

        filter
            .process_unit(&[0, 0, 1, 0x00, 0x00, 0x10], &mut out)
            .unwrap();

        assert_eq!(out, vec![0, 0, 1, 0x00, 0x00, 0x18]);
        assert_eq!(filter.stats.frame_types_rewritten, 1);
    }

    #[test]
    fn mpeg2_drops_later_i_picture_until_next_picture() {
        let mut filter = MoshFilter::new(Config {
            codec: Codec::Mpeg2,
            quiet: true,
            ..Config::default()
        });
        let mut out = Vec::new();

        filter.process_unit(&[0, 0, 1, 0xb3, 1], &mut out).unwrap();
        filter
            .process_unit(&[0, 0, 1, 0x00, 0x00, 0x08], &mut out)
            .unwrap();
        filter
            .process_unit(&[0, 0, 1, 0x01, 0xaa], &mut out)
            .unwrap();
        filter
            .process_unit(&[0, 0, 1, 0x00, 0x00, 0x10], &mut out)
            .unwrap();
        filter
            .process_unit(&[0, 0, 1, 0x01, 0xbb], &mut out)
            .unwrap();
        filter
            .process_unit(&[0, 0, 1, 0x00, 0x00, 0x08], &mut out)
            .unwrap();
        filter
            .process_unit(&[0, 0, 1, 0x01, 0xcc], &mut out)
            .unwrap();
        filter
            .process_unit(&[0, 0, 1, 0x00, 0x00, 0x10], &mut out)
            .unwrap();
        filter
            .process_unit(&[0, 0, 1, 0x01, 0xdd], &mut out)
            .unwrap();

        assert_eq!(
            out,
            vec![
                0, 0, 1, 0xb3, 1, 0, 0, 1, 0x00, 0x00, 0x08, 0, 0, 1, 0x01, 0xaa, 0, 0, 1, 0x00,
                0x00, 0x10, 0, 0, 1, 0x01, 0xbb, 0, 0, 1, 0x00, 0x00, 0x10, 0, 0, 1, 0x01, 0xdd
            ]
        );
        assert_eq!(filter.stats.idr_seen, 2);
        assert_eq!(filter.stats.idr_dropped, 1);
        assert_eq!(filter.stats.slices_seen, 2);
    }

    #[test]
    fn mpeg2_shifts_slice_addresses_for_i_and_p_pictures() {
        let mut filter = MoshFilter::new(Config {
            codec: Codec::Mpeg2,
            drop_idr_after: 99,
            shift_slice_address_every: 1,
            shift_slice_address_by: 2,
            quiet: true,
            ..Config::default()
        });
        let mut out = Vec::new();

        filter
            .process_unit(&[0, 0, 1, 0x00, 0x00, 0x08], &mut out)
            .unwrap();
        filter
            .process_unit(&[0, 0, 1, 0x01, 0xaa], &mut out)
            .unwrap();
        filter
            .process_unit(&[0, 0, 1, 0x00, 0x00, 0x10], &mut out)
            .unwrap();
        filter
            .process_unit(&[0, 0, 1, 0x04, 0xbb], &mut out)
            .unwrap();

        assert_eq!(
            out,
            vec![
                0, 0, 1, 0x00, 0x00, 0x08, 0, 0, 1, 0x03, 0xaa, 0, 0, 1, 0x00, 0x00, 0x10, 0, 0, 1,
                0x06, 0xbb
            ]
        );
        assert_eq!(filter.stats.slice_addresses_shifted, 2);
        assert_eq!(filter.stats.slices_seen, 1);
    }

    #[test]
    fn mpeg2_partially_drops_key_picture_slices_by_address_phase() {
        let mut filter = MoshFilter::new(Config {
            codec: Codec::Mpeg2,
            drop_idr_after: 99,
            drop_mpeg_slice_address_every: 2,
            drop_mpeg_slice_address_phase: 0,
            drop_mpeg_slice_address_mode: MpegSliceDropMode::Key,
            quiet: true,
            ..Config::default()
        });
        let mut out = Vec::new();

        filter
            .process_unit(&[0, 0, 1, 0x00, 0x00, 0x08], &mut out)
            .unwrap();
        filter
            .process_unit(&[0, 0, 1, 0x01, 0xaa], &mut out)
            .unwrap();
        filter
            .process_unit(&[0, 0, 1, 0x02, 0xab], &mut out)
            .unwrap();
        filter
            .process_unit(&[0, 0, 1, 0x00, 0x00, 0x10], &mut out)
            .unwrap();
        filter
            .process_unit(&[0, 0, 1, 0x01, 0xbb], &mut out)
            .unwrap();

        assert_eq!(
            out,
            vec![
                0, 0, 1, 0x00, 0x00, 0x08, 0, 0, 1, 0x02, 0xab, 0, 0, 1, 0x00, 0x00, 0x10, 0, 0, 1,
                0x01, 0xbb
            ]
        );
        assert_eq!(filter.stats.slice_addresses_dropped, 1);
        assert_eq!(filter.stats.slices_seen, 1);
    }

    #[test]
    fn mpeg1_uses_mpeg_picture_path() {
        let mut filter = MoshFilter::new(Config {
            codec: Codec::Mpeg1,
            quiet: true,
            ..Config::default()
        });
        let mut out = Vec::new();

        filter
            .process_unit(&[0, 0, 1, 0x00, 0x00, 0x08], &mut out)
            .unwrap();
        filter
            .process_unit(&[0, 0, 1, 0x01, 0xaa], &mut out)
            .unwrap();
        filter
            .process_unit(&[0, 0, 1, 0x00, 0x00, 0x10], &mut out)
            .unwrap();
        filter
            .process_unit(&[0, 0, 1, 0x01, 0xbb], &mut out)
            .unwrap();
        filter
            .process_unit(&[0, 0, 1, 0x00, 0x00, 0x08], &mut out)
            .unwrap();
        filter
            .process_unit(&[0, 0, 1, 0x01, 0xcc], &mut out)
            .unwrap();

        assert_eq!(
            out,
            vec![
                0, 0, 1, 0x00, 0x00, 0x08, 0, 0, 1, 0x01, 0xaa, 0, 0, 1, 0x00, 0x00, 0x10, 0, 0, 1,
                0x01, 0xbb
            ]
        );
        assert_eq!(filter.stats.idr_seen, 2);
        assert_eq!(filter.stats.idr_dropped, 1);
        assert_eq!(filter.stats.slices_seen, 1);
    }

    #[test]
    fn stream_processor_handles_start_codes_across_chunks() {
        let mut stream = DatamoshStream::new(Config {
            drop_idr_after: 99,
            quiet: true,
            ..Config::default()
        });
        let mut out = Vec::new();

        stream.process_chunk(&[0, 0], &mut out).unwrap();
        stream.process_chunk(&[1, 0x67, 1, 0], &mut out).unwrap();
        stream.process_chunk(&[0, 1, 0x41, 2], &mut out).unwrap();
        stream.finish(&mut out).unwrap();

        assert_eq!(out, vec![0, 0, 1, 0x67, 1, 0, 0, 1, 0x41, 2]);
        assert_eq!(stream.stats().nals_in, 2);
        assert_eq!(stream.stats().nals_out, 2);
    }

    #[test]
    fn drains_complete_nals_and_keeps_incomplete_tail() {
        let mut buffer = vec![0, 0, 1, 0x67, 1, 2, 0, 0, 1, 0x68, 3, 4, 0, 0, 1, 0x65, 5];
        let mut filter = MoshFilter::new(Config {
            quiet: true,
            ..Config::default()
        });
        let mut out = Vec::new();

        drain_complete_nals(&mut buffer, &mut filter, &mut out).unwrap();

        assert_eq!(out, vec![0, 0, 1, 0x67, 1, 2, 0, 0, 1, 0x68, 3, 4]);
        assert_eq!(buffer, vec![0, 0, 1, 0x65, 5]);
    }
}
