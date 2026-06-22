//! DCT0 — intra-only transform-domain codec (JPEG/DV-style).
//!
//! The luma plane is taken at full resolution and the chroma planes are 4:2:0-subsampled
//! (stored at half resolution, like JPEG/DV). Each plane is split into 8x8 blocks,
//! forward-DCT'd (a fast even/odd transform), quantized with a JPEG-style table scaled by
//! `quality`, and stored as quantized coefficients. Glitches
//! corrupt the transform-domain data — quantization, DC/AC coefficients, the differential
//! DC predictor (the signature JPEG "color smear that bleeds block-by-block"), coefficient
//! order, and whole-block remaps — before the inverse DCT reconstructs the frame. Unlike
//! MSH0 (block motion) and SCN0 (analog signal) this codec is intra-only: each frame is an
//! independent still, so its glitches read as quantization storms, blocking, ringing, and
//! propagating DC color shifts rather than temporal smear.

use std::io;
use std::sync::OnceLock;

use rayon::prelude::*;

use crate::RawMoshControls;
use crate::dct_bitstream::{
    DctBitstreamMutationStats, DctBitstreamParams, decode_dct_bitstream, encode_dct_bitstream,
    mutate_dct_bitstream,
};
use crate::mosh_codec::codec_thread_pool;

const CHANNELS: usize = 3;
const BLOCK: usize = 8;
pub(crate) const BLOCK_AREA: usize = BLOCK * BLOCK;
const PARALLEL_FRAME_PIXELS: usize = 200_000;

// Standard JPEG quantization tables (natural row-major order).
const QUANT_LUMA: [u16; BLOCK_AREA] = [
    16, 11, 10, 16, 24, 40, 51, 61, 12, 12, 14, 19, 26, 58, 60, 55, 14, 13, 16, 24, 40, 57, 69, 56,
    14, 17, 22, 29, 51, 87, 80, 62, 18, 22, 37, 56, 68, 109, 103, 77, 24, 35, 55, 64, 81, 104, 113,
    92, 49, 64, 78, 87, 103, 121, 120, 101, 72, 92, 95, 98, 112, 100, 103, 99,
];
const QUANT_CHROMA: [u16; BLOCK_AREA] = [
    17, 18, 24, 47, 99, 99, 99, 99, 18, 21, 26, 66, 99, 99, 99, 99, 24, 26, 56, 99, 99, 99, 99, 99,
    47, 66, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99,
    99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99,
];

// Natural-order index visited at each zig-zag position.
pub(crate) const ZIGZAG: [usize; BLOCK_AREA] = [
    0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5, 12, 19, 26, 33, 40, 48, 41, 34, 27, 20,
    13, 6, 7, 14, 21, 28, 35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51, 58, 59,
    52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63,
];

#[derive(Debug, Clone, Copy)]
pub struct DctCodecConfig {
    pub width: usize,
    pub height: usize,
    /// JPEG-style quality, 1 (coarse) .. 100 (fine).
    pub quality: u8,
}

impl DctCodecConfig {
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
        if self.quality == 0 || self.quality > 100 {
            return Err(invalid_input("quality must be in 1..=100"));
        }
        self.frame_len()
            .ok_or_else(|| invalid_input("frame dimensions overflow addressable memory"))?;
        Ok(())
    }
}

impl Default for DctCodecConfig {
    fn default() -> Self {
        Self {
            width: 0,
            height: 0,
            quality: 50,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct DctGlitchParams {
    /// Re-quantize coefficients onto a coarser grid (>=1; 1 = off). Blocking.
    pub quant_scale: f32,
    /// Propagating DC predictor corruption: at every Nth block the running per-channel DC
    /// offset is bumped by `dc_drift`, then added to every later block's DC (scan order).
    pub dc_drift: i16,
    pub dc_drift_every: u64,
    /// Non-propagating DC kick on every Nth block.
    pub dc_block_offset: i16,
    pub dc_block_offset_every: u64,
    /// Zero all AC coefficients beyond this zig-zag index (0 = keep all). Low-pass blur.
    pub ac_zero_above: usize,
    /// Flip the sign of coefficients in every Nth block. Ringing / inversion.
    pub coeff_sign_flip_every: u64,
    /// Rotate AC coefficients within the block by this many zig-zag positions on every Nth
    /// block. Frequency scramble.
    pub coeff_shift: i16,
    pub coeff_shift_every: u64,
    /// Source-block remap: every Nth block takes another block's coefficients from
    /// (bx+block_shift_x, by+block_shift_y). Spatial block displacement.
    pub block_shift_x: i16,
    pub block_shift_y: i16,
    pub block_shift_every: u64,
    /// Repeat the previous block's coefficients on every Nth block.
    pub block_repeat_every: u64,
    /// Reverse the AC coefficients in zig-zag order on every Nth block, swapping low and
    /// high frequencies. Smooth blocks become sharp/noisy and vice versa.
    pub zigzag_reverse_every: u64,
    /// Transpose the 8x8 coefficient block on every Nth block, swapping horizontal and
    /// vertical frequencies — texture/edges rotate 90 degrees (directional smear).
    pub block_transpose_every: u64,
    /// Swap the Cb and Cr coefficient blocks on every Nth chroma block (false colour:
    /// reds<->blues). Uses the 4:2:0 planar layout.
    pub chroma_swap_every: u64,
    /// Temporal feedback: blend the previous (glitched) output back into the encode
    /// input. 0 = pure intra (each frame independent, the codec's natural mode); higher
    /// values make glitches persist and propagate across frames. Clamped to 0..=0.98 so
    /// the feed never fully replaces the live input.
    pub persistence: f32,
}

impl Default for DctGlitchParams {
    fn default() -> Self {
        Self {
            quant_scale: 1.0,
            dc_drift: 0,
            dc_drift_every: 0,
            dc_block_offset: 0,
            dc_block_offset_every: 0,
            ac_zero_above: 0,
            coeff_sign_flip_every: 0,
            coeff_shift: 0,
            coeff_shift_every: 0,
            block_shift_x: 0,
            block_shift_y: 0,
            block_shift_every: 0,
            block_repeat_every: 0,
            zigzag_reverse_every: 0,
            block_transpose_every: 0,
            chroma_swap_every: 0,
            persistence: 0.0,
        }
    }
}

impl DctGlitchParams {
    pub fn has_mutations(&self) -> bool {
        self.quant_scale > 1.0
            || (self.dc_drift != 0 && self.dc_drift_every != 0)
            || (self.dc_block_offset != 0 && self.dc_block_offset_every != 0)
            || self.ac_zero_above != 0
            || self.coeff_sign_flip_every != 0
            || (self.coeff_shift != 0 && self.coeff_shift_every != 0)
            || ((self.block_shift_x != 0 || self.block_shift_y != 0) && self.block_shift_every != 0)
            || self.block_repeat_every != 0
            || self.zigzag_reverse_every != 0
            || self.block_transpose_every != 0
            || self.chroma_swap_every != 0
            || self.persistence > 0.0
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct DctCodecStats {
    pub frames_in: u64,
    pub blocks_encoded: u64,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct DctMutationStats {
    pub blocks_requantized: u64,
    pub dc_drifts: u64,
    pub dc_offsets: u64,
    pub blocks_low_passed: u64,
    pub blocks_sign_flipped: u64,
    pub blocks_coeff_shifted: u64,
    pub blocks_remapped: u64,
    pub blocks_repeated: u64,
}

pub struct DctCodec {
    config: DctCodecConfig,
    stats: DctCodecStats,
    // Luma block grid (full resolution).
    blocks_x: usize,
    blocks_y: usize,
    // Chroma plane size and block grid (4:2:0, half resolution).
    cwidth: usize,
    cheight: usize,
    cblocks_x: usize,
    cblocks_y: usize,
    quant_luma: [f32; BLOCK_AREA],
    quant_chroma: [f32; BLOCK_AREA],
    // Planar quantized coefficients (natural order), 4:2:0 layout:
    //   [ luma blocks | Cb blocks (half-res) | Cr blocks (half-res) ], each block 64 coeffs.
    coeff: Vec<i16>,
    scratch: Vec<i16>,
    feedback: Vec<u8>,
}

impl DctCodec {
    pub fn new(config: DctCodecConfig) -> io::Result<Self> {
        config.validate()?;
        let blocks_x = config.width.div_ceil(BLOCK);
        let blocks_y = config.height.div_ceil(BLOCK);
        let cwidth = config.width.div_ceil(2);
        let cheight = config.height.div_ceil(2);
        let cblocks_x = cwidth.div_ceil(BLOCK);
        let cblocks_y = cheight.div_ceil(BLOCK);
        let scale = quality_scale(config.quality);
        let quant_luma = scaled_table(&QUANT_LUMA, scale);
        let quant_chroma = scaled_table(&QUANT_CHROMA, scale);
        let coeff_len = (blocks_x * blocks_y + 2 * cblocks_x * cblocks_y) * BLOCK_AREA;
        Ok(Self {
            config,
            stats: DctCodecStats::default(),
            blocks_x,
            blocks_y,
            cwidth,
            cheight,
            cblocks_x,
            cblocks_y,
            quant_luma,
            quant_chroma,
            coeff: vec![0; coeff_len],
            scratch: Vec::new(),
            feedback: Vec::new(),
        })
    }

    pub fn config(&self) -> &DctCodecConfig {
        &self.config
    }

    pub fn stats(&self) -> &DctCodecStats {
        &self.stats
    }

    pub fn reset_glitch_state(&mut self) {
        // Clear the temporal-feedback history (used only when persistence > 0) and the
        // accumulating stats counters.
        self.feedback = Vec::new();
        self.stats = DctCodecStats::default();
    }

    fn luma_blocks(&self) -> usize {
        self.blocks_x * self.blocks_y
    }

    fn chroma_blocks(&self) -> usize {
        self.cblocks_x * self.cblocks_y
    }

    pub fn process_rgb_frame(
        &mut self,
        input: &[u8],
        params: &DctGlitchParams,
        output: &mut [u8],
    ) -> io::Result<DctMutationStats> {
        let frame_len = self.config.frame_len().expect("validated frame dimensions");
        if input.len() != frame_len || output.len() != frame_len {
            return Err(invalid_input(format!(
                "input and output frames must be {frame_len} bytes of rgb24"
            )));
        }

        // Temporal feedback (pure codec feedback): blend the previous glitched output
        // into the encode input so corruption propagates through the real transform
        // pipeline. persistence == 0 leaves the codec purely intra (output == per-frame
        // JPEG of the live input).
        let persistence = params.persistence.clamp(0.0, 0.98);
        let blended;
        let encode_source: &[u8] = if persistence > 0.0 && self.feedback.len() == frame_len {
            let inv = 1.0 - persistence;
            blended = input
                .iter()
                .zip(&self.feedback)
                .map(|(&live, &fed)| {
                    (live as f32 * inv + fed as f32 * persistence)
                        .round()
                        .clamp(0.0, 255.0) as u8
                })
                .collect::<Vec<u8>>();
            &blended
        } else {
            input
        };

        self.encode(encode_source);
        let mutation_stats = self.glitch(params);
        self.decode(output);

        self.feedback.clear();
        self.feedback.extend_from_slice(output);

        self.stats.frames_in += 1;
        self.stats.blocks_encoded += (self.luma_blocks() + 2 * self.chroma_blocks()) as u64;
        Ok(mutation_stats)
    }

    /// Entropy-coded (`DTE0` bitstream) path: encode → coefficient-domain glitch → serialize
    /// to the entropy bitstream → corrupt the serialized bytes → decode (desync-tolerant) →
    /// reconstruct. This adds the codec's real decoder-desync glitch surface (the cascading
    /// "broken JPEG" slide from a single bad byte) on top of the coefficient-domain glitches.
    /// With no corruption it is byte-identical to `process_rgb_frame` (entropy coding is
    /// lossless), so the codec stays rational; the glitch lives in genuine bitstream damage.
    pub fn process_rgb_frame_bitstream(
        &mut self,
        input: &[u8],
        params: &DctGlitchParams,
        bitstream_params: &DctBitstreamParams,
        output: &mut [u8],
    ) -> io::Result<(DctMutationStats, DctBitstreamMutationStats)> {
        let frame_len = self.config.frame_len().expect("validated frame dimensions");
        if input.len() != frame_len || output.len() != frame_len {
            return Err(invalid_input(format!(
                "input and output frames must be {frame_len} bytes of rgb24"
            )));
        }

        let persistence = params.persistence.clamp(0.0, 0.98);
        let blended;
        let encode_source: &[u8] = if persistence > 0.0 && self.feedback.len() == frame_len {
            let inv = 1.0 - persistence;
            blended = input
                .iter()
                .zip(&self.feedback)
                .map(|(&live, &fed)| {
                    (live as f32 * inv + fed as f32 * persistence)
                        .round()
                        .clamp(0.0, 255.0) as u8
                })
                .collect::<Vec<u8>>();
            &blended
        } else {
            input
        };

        self.encode(encode_source);
        let mutation_stats = self.glitch(params);

        let (width, height) = (self.config.width, self.config.height);
        let (lb, cb) = (self.luma_blocks(), self.chroma_blocks());
        let mut bytes = encode_dct_bitstream(&self.coeff, width, height, lb, cb);
        let seed = hash_u64(self.stats.frames_in.wrapping_add(1));
        let bitstream_stats = mutate_dct_bitstream(&mut bytes, bitstream_params, seed);
        decode_dct_bitstream(&bytes, &mut self.coeff, lb, cb);

        self.decode(output);

        self.feedback.clear();
        self.feedback.extend_from_slice(output);
        self.stats.frames_in += 1;
        self.stats.blocks_encoded += (lb + 2 * cb) as u64;
        Ok((mutation_stats, bitstream_stats))
    }

    fn encode(&mut self, input: &[u8]) {
        let width = self.config.width;
        let height = self.config.height;
        let cw = self.cwidth;
        let ch = self.cheight;

        // Build the luma plane at full resolution and the Cb/Cr planes at half resolution
        // (4:2:0 — each chroma sample averages its 2x2 source block). All samples are
        // level-shifted by -128 before the DCT, as in JPEG.
        let mut y_plane = vec![0.0_f32; width * height];
        let mut cb_plane = vec![0.0_f32; cw * ch];
        let mut cr_plane = vec![0.0_f32; cw * ch];
        let mut weight = vec![0.0_f32; cw * ch];
        for y in 0..height {
            for x in 0..width {
                let p = (y * width + x) * CHANNELS;
                let ycc = rgb_pixel_to_ycbcr(input[p], input[p + 1], input[p + 2]);
                y_plane[y * width + x] = ycc[0] as f32 - 128.0;
                let ci = (y / 2) * cw + (x / 2);
                cb_plane[ci] += ycc[1] as f32 - 128.0;
                cr_plane[ci] += ycc[2] as f32 - 128.0;
                weight[ci] += 1.0;
            }
        }
        for i in 0..cw * ch {
            let w = weight[i].max(1.0);
            cb_plane[i] /= w;
            cr_plane[i] /= w;
        }

        let lb = self.luma_blocks() * BLOCK_AREA;
        let cb = self.chroma_blocks() * BLOCK_AREA;
        let lbx = self.blocks_x;
        let cbx = self.cblocks_x;
        let ql = self.quant_luma;
        let qc = self.quant_chroma;
        let parallel =
            width.saturating_mul(height) >= PARALLEL_FRAME_PIXELS && codec_thread_pool().is_some();
        let (luma_region, rest) = self.coeff.split_at_mut(lb);
        let (cb_region, cr_region) = rest.split_at_mut(cb);
        let mut run = || {
            encode_plane(&y_plane, width, height, lbx, &ql, luma_region, parallel);
            encode_plane(&cb_plane, cw, ch, cbx, &qc, cb_region, parallel);
            encode_plane(&cr_plane, cw, ch, cbx, &qc, cr_region, parallel);
        };
        if parallel {
            codec_thread_pool().unwrap().install(run);
        } else {
            run();
        }
    }

    fn glitch(&mut self, params: &DctGlitchParams) -> DctMutationStats {
        let mut stats = DctMutationStats::default();
        if self.coeff.is_empty() {
            return stats;
        }
        let quant_scale = params.quant_scale.max(1.0);
        let remap_active = (params.block_shift_every != 0
            && (params.block_shift_x != 0 || params.block_shift_y != 0))
            || params.block_repeat_every != 0;

        // Snapshot for whole-block remaps so a remapped block pulls pristine source coeffs.
        let mut snapshot = std::mem::take(&mut self.scratch);
        if remap_active {
            snapshot.clear();
            snapshot.extend_from_slice(&self.coeff);
        }

        let lb = self.luma_blocks() * BLOCK_AREA;
        let cb = self.chroma_blocks() * BLOCK_AREA;
        let (lbx, lby) = (self.blocks_x, self.blocks_y);
        let (cbx, cby) = (self.cblocks_x, self.cblocks_y);
        let empty: &[i16] = &[];
        // Luma (channel 0) on the full-res grid, then Cb (1) and Cr (2) on the half-res grid.
        glitch_plane(
            &mut self.coeff[0..lb],
            if remap_active {
                &snapshot[0..lb]
            } else {
                empty
            },
            lbx,
            lby,
            0,
            params,
            quant_scale,
            remap_active,
            &mut stats,
        );
        glitch_plane(
            &mut self.coeff[lb..lb + cb],
            if remap_active {
                &snapshot[lb..lb + cb]
            } else {
                empty
            },
            cbx,
            cby,
            1,
            params,
            quant_scale,
            remap_active,
            &mut stats,
        );
        glitch_plane(
            &mut self.coeff[lb + cb..lb + 2 * cb],
            if remap_active {
                &snapshot[lb + cb..lb + 2 * cb]
            } else {
                empty
            },
            cbx,
            cby,
            2,
            params,
            quant_scale,
            remap_active,
            &mut stats,
        );

        // False-colour: swap Cb and Cr coefficient blocks on triggered chroma blocks.
        if params.chroma_swap_every != 0 && cb > 0 {
            let cblocks = cb / BLOCK_AREA;
            let (cb_part, cr_part) = self.coeff[lb..lb + 2 * cb].split_at_mut(cb);
            for b in 0..cblocks {
                if is_every(params.chroma_swap_every, b as u64 + 1) {
                    let base = b * BLOCK_AREA;
                    cb_part[base..base + BLOCK_AREA]
                        .swap_with_slice(&mut cr_part[base..base + BLOCK_AREA]);
                }
            }
        }

        self.scratch = snapshot;
        stats
    }

    fn decode(&self, output: &mut [u8]) {
        let width = self.config.width;
        let height = self.config.height;
        let cw = self.cwidth;
        let ch = self.cheight;
        let lb = self.luma_blocks() * BLOCK_AREA;
        let cb = self.chroma_blocks() * BLOCK_AREA;
        let parallel =
            width.saturating_mul(height) >= PARALLEL_FRAME_PIXELS && codec_thread_pool().is_some();

        // Inverse-transform each plane into reconstructed (level-shifted) samples.
        let mut y_plane = vec![0.0_f32; width * height];
        let mut cb_plane = vec![0.0_f32; cw * ch];
        let mut cr_plane = vec![0.0_f32; cw * ch];
        let coeff = self.coeff.as_slice();
        let ql = &self.quant_luma;
        let qc = &self.quant_chroma;
        let lbx = self.blocks_x;
        let cbx = self.cblocks_x;
        let mut decode_all = || {
            decode_plane(
                &coeff[0..lb],
                lbx,
                width,
                height,
                ql,
                &mut y_plane,
                parallel,
            );
            decode_plane(
                &coeff[lb..lb + cb],
                cbx,
                cw,
                ch,
                qc,
                &mut cb_plane,
                parallel,
            );
            decode_plane(
                &coeff[lb + cb..lb + 2 * cb],
                cbx,
                cw,
                ch,
                qc,
                &mut cr_plane,
                parallel,
            );
        };
        if parallel {
            codec_thread_pool().unwrap().install(decode_all);
        } else {
            decode_all();
        }

        // Combine planes into RGB, upsampling chroma by nearest-neighbour (each 2x2 luma
        // quad shares one chroma sample — the characteristic 4:2:0 colour block).
        let combine = |y: usize, row: &mut [u8]| {
            let crow = (y / 2) * cw;
            for x in 0..width {
                let ci = crow + (x / 2);
                let rgb = ycbcr_to_rgb_pixel(
                    y_plane[y * width + x] + 128.0,
                    cb_plane[ci] + 128.0,
                    cr_plane[ci] + 128.0,
                );
                let o = x * CHANNELS;
                row[o] = rgb[0];
                row[o + 1] = rgb[1];
                row[o + 2] = rgb[2];
            }
        };
        if parallel {
            codec_thread_pool().unwrap().install(|| {
                output
                    .par_chunks_mut(width * CHANNELS)
                    .enumerate()
                    .for_each(|(y, row)| combine(y, row));
            });
        } else {
            output
                .chunks_mut(width * CHANNELS)
                .enumerate()
                .for_each(|(y, row)| combine(y, row));
        }
    }
}

fn quality_scale(quality: u8) -> f32 {
    let q = quality.clamp(1, 100) as f32;
    if q < 50.0 {
        5000.0 / q
    } else {
        200.0 - 2.0 * q
    }
}

fn scaled_table(table: &[u16; BLOCK_AREA], scale: f32) -> [f32; BLOCK_AREA] {
    let mut out = [0.0_f32; BLOCK_AREA];
    for (dst, &value) in out.iter_mut().zip(table.iter()) {
        let scaled = ((value as f32 * scale + 50.0) / 100.0).floor();
        *dst = scaled.clamp(1.0, 255.0);
    }
    out
}

// Exact matrix-product DCT, kept as the reference the fast path is checked against.
#[allow(dead_code)]
fn dct_matrix() -> &'static [[f32; BLOCK]; BLOCK] {
    static MATRIX: OnceLock<[[f32; BLOCK]; BLOCK]> = OnceLock::new();
    MATRIX.get_or_init(|| {
        let mut m = [[0.0_f32; BLOCK]; BLOCK];
        for (u, row) in m.iter_mut().enumerate() {
            let cu = if u == 0 {
                (1.0_f64 / BLOCK as f64).sqrt()
            } else {
                (2.0_f64 / BLOCK as f64).sqrt()
            };
            for (x, value) in row.iter_mut().enumerate() {
                let angle =
                    (2.0 * x as f64 + 1.0) * u as f64 * std::f64::consts::PI / (2.0 * BLOCK as f64);
                *value = (cu * angle.cos()) as f32;
            }
        }
        m
    })
}

// Forward 2D DCT-II of an 8x8 block: F = M * B * M^T.
#[allow(dead_code)]
fn forward_dct(block: &[f32; BLOCK_AREA]) -> [f32; BLOCK_AREA] {
    let m = dct_matrix();
    let mut temp = [0.0_f32; BLOCK_AREA];
    // temp = M * B
    for i in 0..BLOCK {
        for j in 0..BLOCK {
            let mut sum = 0.0_f32;
            for k in 0..BLOCK {
                sum += m[i][k] * block[k * BLOCK + j];
            }
            temp[i * BLOCK + j] = sum;
        }
    }
    let mut out = [0.0_f32; BLOCK_AREA];
    // out = temp * M^T
    for i in 0..BLOCK {
        for j in 0..BLOCK {
            let mut sum = 0.0_f32;
            for k in 0..BLOCK {
                sum += temp[i * BLOCK + k] * m[j][k];
            }
            out[i * BLOCK + j] = sum;
        }
    }
    out
}

// Inverse 2D DCT: B = M^T * F * M.
#[allow(dead_code)]
fn inverse_dct(freq: &[f32; BLOCK_AREA]) -> [f32; BLOCK_AREA] {
    let m = dct_matrix();
    let mut temp = [0.0_f32; BLOCK_AREA];
    // temp = M^T * F
    for i in 0..BLOCK {
        for j in 0..BLOCK {
            let mut sum = 0.0_f32;
            for k in 0..BLOCK {
                sum += m[k][i] * freq[k * BLOCK + j];
            }
            temp[i * BLOCK + j] = sum;
        }
    }
    let mut out = [0.0_f32; BLOCK_AREA];
    // out = temp * M
    for i in 0..BLOCK {
        for j in 0..BLOCK {
            let mut sum = 0.0_f32;
            for k in 0..BLOCK {
                sum += temp[i * BLOCK + k] * m[k][j];
            }
            out[i * BLOCK + j] = sum;
        }
    }
    out
}

// --- Fast separable 8-point DCT via even/odd decomposition ---
// The 4x4 even/odd sub-matrices are derived from the same cosines as the exact matrix
// (dct_matrix), so the fast transform is correct by construction (and checked against the
// matmul by the fast_dct_matches_matmul test) while doing ~half the multiplies.
fn even_odd_matrices() -> &'static ([[f32; 4]; 4], [[f32; 4]; 4]) {
    type EvenOddMatrices = ([[f32; 4]; 4], [[f32; 4]; 4]);
    static M: OnceLock<EvenOddMatrices> = OnceLock::new();
    M.get_or_init(|| {
        let mut even = [[0.0_f32; 4]; 4];
        let mut odd = [[0.0_f32; 4]; 4];
        for m in 0..4 {
            let ce = if m == 0 { (1.0_f64 / 8.0).sqrt() } else { 0.5 };
            for n in 0..4 {
                let fe = (2.0 * n as f64 + 1.0) * m as f64 * std::f64::consts::PI / 8.0;
                let fo =
                    (2.0 * n as f64 + 1.0) * (2.0 * m as f64 + 1.0) * std::f64::consts::PI / 16.0;
                even[m][n] = (ce * fe.cos()) as f32;
                odd[m][n] = (0.5 * fo.cos()) as f32;
            }
        }
        (even, odd)
    })
}

type Mat4 = [[f32; 4]; 4];

#[inline(always)]
fn fdct8(v: &[f32; BLOCK], even: &Mat4, odd: &Mat4) -> [f32; BLOCK] {
    let s = [v[0] + v[7], v[1] + v[6], v[2] + v[5], v[3] + v[4]];
    let d = [v[0] - v[7], v[1] - v[6], v[2] - v[5], v[3] - v[4]];
    let mut y = [0.0_f32; BLOCK];
    for m in 0..4 {
        let er = &even[m];
        let or = &odd[m];
        y[2 * m] = er[0] * s[0] + er[1] * s[1] + er[2] * s[2] + er[3] * s[3];
        y[2 * m + 1] = or[0] * d[0] + or[1] * d[1] + or[2] * d[2] + or[3] * d[3];
    }
    y
}

#[inline(always)]
fn idct8(y: &[f32; BLOCK], even: &Mat4, odd: &Mat4) -> [f32; BLOCK] {
    let mut e = [0.0_f32; 4];
    let mut o = [0.0_f32; 4];
    for m in 0..4 {
        let (ym0, ym1) = (y[2 * m], y[2 * m + 1]);
        let er = &even[m];
        let or = &odd[m];
        for n in 0..4 {
            e[n] += er[n] * ym0;
            o[n] += or[n] * ym1;
        }
    }
    let mut x = [0.0_f32; BLOCK];
    for n in 0..4 {
        x[n] = e[n] + o[n];
        x[7 - n] = e[n] - o[n];
    }
    x
}

fn forward_dct_fast(block: &[f32; BLOCK_AREA]) -> [f32; BLOCK_AREA] {
    let (even, odd) = even_odd_matrices();
    let mut tmp = [0.0_f32; BLOCK_AREA];
    let mut line = [0.0_f32; BLOCK];
    for i in 0..BLOCK {
        line.copy_from_slice(&block[i * BLOCK..i * BLOCK + BLOCK]);
        tmp[i * BLOCK..i * BLOCK + BLOCK].copy_from_slice(&fdct8(&line, even, odd));
    }
    let mut out = [0.0_f32; BLOCK_AREA];
    for j in 0..BLOCK {
        for i in 0..BLOCK {
            line[i] = tmp[i * BLOCK + j];
        }
        let c = fdct8(&line, even, odd);
        for i in 0..BLOCK {
            out[i * BLOCK + j] = c[i];
        }
    }
    out
}

fn inverse_dct_fast(freq: &[f32; BLOCK_AREA]) -> [f32; BLOCK_AREA] {
    let (even, odd) = even_odd_matrices();
    let mut tmp = [0.0_f32; BLOCK_AREA];
    let mut line = [0.0_f32; BLOCK];
    for j in 0..BLOCK {
        for i in 0..BLOCK {
            line[i] = freq[i * BLOCK + j];
        }
        let c = idct8(&line, even, odd);
        for i in 0..BLOCK {
            tmp[i * BLOCK + j] = c[i];
        }
    }
    let mut out = [0.0_f32; BLOCK_AREA];
    for i in 0..BLOCK {
        line.copy_from_slice(&tmp[i * BLOCK..i * BLOCK + BLOCK]);
        out[i * BLOCK..i * BLOCK + BLOCK].copy_from_slice(&idct8(&line, even, odd));
    }
    out
}

// Encode one 8x8 block (all channels) from the input frame into a CHANNELS*64 slice.
// Forward-DCT + quantize every 8x8 block of a single (already level-shifted) plane into
// its coefficient region. Each block is independent, so this parallelizes per block.
fn encode_plane(
    plane: &[f32],
    pw: usize,
    ph: usize,
    blocks_x: usize,
    quant: &[f32; BLOCK_AREA],
    region: &mut [i16],
    parallel: bool,
) {
    let encode_one = |bi: usize, out: &mut [i16]| {
        let bx = bi % blocks_x;
        let by = bi / blocks_x;
        let mut block = [0.0_f32; BLOCK_AREA];
        for ry in 0..BLOCK {
            let sy = (by * BLOCK + ry).min(ph - 1);
            for rx in 0..BLOCK {
                let sx = (bx * BLOCK + rx).min(pw - 1);
                block[ry * BLOCK + rx] = plane[sy * pw + sx];
            }
        }
        let freq = forward_dct_fast(&block);
        for k in 0..BLOCK_AREA {
            out[k] = (freq[k] / quant[k]).round().clamp(-32768.0, 32767.0) as i16;
        }
    };
    if parallel {
        region
            .par_chunks_mut(BLOCK_AREA)
            .enumerate()
            .for_each(|(bi, out)| encode_one(bi, out));
    } else {
        region
            .chunks_mut(BLOCK_AREA)
            .enumerate()
            .for_each(|(bi, out)| encode_one(bi, out));
    }
}

// Transform-domain corruption for one plane (one channel, one block grid, one coeff
// region). `channel` selects the per-channel DC-drift sign bit.
#[allow(clippy::too_many_arguments)]
fn glitch_plane(
    coeff: &mut [i16],
    snapshot: &[i16],
    blocks_x: usize,
    blocks_y: usize,
    channel: usize,
    params: &DctGlitchParams,
    quant_scale: f32,
    remap_active: bool,
    stats: &mut DctMutationStats,
) {
    if blocks_x == 0 || blocks_y == 0 {
        return;
    }

    if remap_active {
        for by in 0..blocks_y {
            for bx in 0..blocks_x {
                let ordinal = (by * blocks_x + bx) as u64 + 1;
                let (src_bx, src_by, counted) = if params.block_shift_every != 0
                    && is_every(params.block_shift_every, ordinal)
                    && (params.block_shift_x != 0 || params.block_shift_y != 0)
                {
                    stats.blocks_remapped += 1;
                    (
                        (bx as i64 + params.block_shift_x as i64).rem_euclid(blocks_x as i64)
                            as usize,
                        (by as i64 + params.block_shift_y as i64).rem_euclid(blocks_y as i64)
                            as usize,
                        true,
                    )
                } else if params.block_repeat_every != 0
                    && is_every(params.block_repeat_every, ordinal)
                {
                    stats.blocks_repeated += 1;
                    let prev = (by * blocks_x + bx).saturating_sub(1);
                    (prev % blocks_x, prev / blocks_x % blocks_y, true)
                } else {
                    (bx, by, false)
                };
                if !counted {
                    continue;
                }
                let dst = (by * blocks_x + bx) * BLOCK_AREA;
                let src = (src_by * blocks_x + src_bx) * BLOCK_AREA;
                coeff[dst..dst + BLOCK_AREA].copy_from_slice(&snapshot[src..src + BLOCK_AREA]);
            }
        }
    }

    // Per-block in-place coefficient corruption.
    for by in 0..blocks_y {
        for bx in 0..blocks_x {
            let ordinal = (by * blocks_x + bx) as u64 + 1;
            let requantize = quant_scale > 1.0;
            let low_pass = params.ac_zero_above != 0;
            let sign_flip = params.coeff_sign_flip_every != 0
                && is_every(params.coeff_sign_flip_every, ordinal);
            let coeff_shift = params.coeff_shift != 0
                && params.coeff_shift_every != 0
                && is_every(params.coeff_shift_every, ordinal);
            let zigzag_reverse =
                params.zigzag_reverse_every != 0 && is_every(params.zigzag_reverse_every, ordinal);
            let block_transpose = params.block_transpose_every != 0
                && is_every(params.block_transpose_every, ordinal);
            let dc_offset = params.dc_block_offset != 0
                && params.dc_block_offset_every != 0
                && is_every(params.dc_block_offset_every, ordinal);
            if requantize {
                stats.blocks_requantized += 1;
            }
            if low_pass {
                stats.blocks_low_passed += 1;
            }
            if sign_flip {
                stats.blocks_sign_flipped += 1;
            }
            if coeff_shift {
                stats.blocks_coeff_shifted += 1;
            }
            if dc_offset {
                stats.dc_offsets += 1;
            }
            let base = (by * blocks_x + bx) * BLOCK_AREA;
            let block = &mut coeff[base..base + BLOCK_AREA];
            if coeff_shift {
                rotate_zigzag_ac(block, params.coeff_shift);
            }
            if zigzag_reverse {
                reverse_zigzag_ac(block);
            }
            if block_transpose {
                transpose_block(block);
            }
            for (zz, &idx) in ZIGZAG.iter().enumerate() {
                if zz != 0 && low_pass && zz > params.ac_zero_above {
                    block[idx] = 0;
                    continue;
                }
                if requantize {
                    let v = block[idx] as f32;
                    block[idx] =
                        ((v / quant_scale).round() * quant_scale).clamp(-32768.0, 32767.0) as i16;
                }
                if sign_flip && zz != 0 {
                    block[idx] = block[idx].saturating_neg();
                }
            }
            if dc_offset {
                block[ZIGZAG[0]] = block[ZIGZAG[0]].saturating_add(params.dc_block_offset);
            }
        }
    }

    // Propagating DC predictor corruption (the signature JPEG color smear).
    if params.dc_drift != 0 && params.dc_drift_every != 0 {
        let mut offset = 0_i32;
        let mut trigger = 0_u64;
        for by in 0..blocks_y {
            for bx in 0..blocks_x {
                let ordinal = (by * blocks_x + bx) as u64 + 1;
                if is_every(params.dc_drift_every, ordinal) {
                    stats.dc_drifts += 1;
                    trigger += 1;
                    // Independent signed step per channel so the colour walks across the
                    // whole space, not just the green<->magenta diagonal.
                    let h = hash_u64(trigger);
                    let sign = if (h >> channel) & 1 == 0 { 1 } else { -1 };
                    offset += params.dc_drift as i32 * sign;
                }
                let dc = (by * blocks_x + bx) * BLOCK_AREA + ZIGZAG[0];
                coeff[dc] = (coeff[dc] as i32 + offset).clamp(-32768, 32767) as i16;
            }
        }
    }
}

// Dequantize + inverse-DCT one block. Fast path: a block whose AC coefficients are all
// zero (common after quantization/glitch) inverse-transforms to a flat block (the DC basis
// row is constant sqrt(1/8), so every sample equals DC * 1/8). Exact, so it is a pure speedup.
fn decode_one_block(coeff: &[i16], quant: &[f32; BLOCK_AREA]) -> [f32; BLOCK_AREA] {
    if coeff[1..BLOCK_AREA].iter().all(|&c| c == 0) {
        [coeff[0] as f32 * quant[0] * 0.125; BLOCK_AREA]
    } else {
        let mut freq = [0.0_f32; BLOCK_AREA];
        for k in 0..BLOCK_AREA {
            freq[k] = coeff[k] as f32 * quant[k];
        }
        inverse_dct_fast(&freq)
    }
}

// Inverse-transform a plane's coefficient region into reconstructed (level-shifted)
// samples. Each block-row writes a disjoint band, so the bands decode in parallel.
fn decode_plane(
    region: &[i16],
    blocks_x: usize,
    pw: usize,
    ph: usize,
    quant: &[f32; BLOCK_AREA],
    out_plane: &mut [f32],
    parallel: bool,
) {
    let _ = ph;
    let decode_band = |by: usize, band: &mut [f32]| {
        let band_rows = band.len() / pw;
        for bx in 0..blocks_x {
            let base = (by * blocks_x + bx) * BLOCK_AREA;
            let block = decode_one_block(&region[base..base + BLOCK_AREA], quant);
            for ry in 0..BLOCK {
                if ry >= band_rows {
                    break;
                }
                for rx in 0..BLOCK {
                    let px = bx * BLOCK + rx;
                    if px >= pw {
                        break;
                    }
                    band[ry * pw + px] = block[ry * BLOCK + rx];
                }
            }
        }
    };
    if parallel {
        out_plane
            .par_chunks_mut(BLOCK * pw)
            .enumerate()
            .for_each(|(by, band)| decode_band(by, band));
    } else {
        out_plane
            .chunks_mut(BLOCK * pw)
            .enumerate()
            .for_each(|(by, band)| decode_band(by, band));
    }
}

// Transpose the 8x8 coefficient block, swapping horizontal and vertical frequencies.
fn transpose_block(block: &mut [i16]) {
    for i in 0..BLOCK {
        for j in (i + 1)..BLOCK {
            block.swap(i * BLOCK + j, j * BLOCK + i);
        }
    }
}

// Reverse the 63 AC coefficients in zig-zag order (swap low and high frequencies),
// leaving DC fixed.
fn reverse_zigzag_ac(block: &mut [i16]) {
    for i in 1..=(BLOCK_AREA - 1) / 2 {
        block.swap(ZIGZAG[i], ZIGZAG[BLOCK_AREA - i]);
    }
}

// Rotate the 63 AC coefficients (in zig-zag order) by `amount`, leaving DC fixed.
fn rotate_zigzag_ac(block: &mut [i16], amount: i16) {
    let ac = BLOCK_AREA - 1;
    let shift = (amount as i64).rem_euclid(ac as i64) as usize;
    if shift == 0 {
        return;
    }
    let mut ordered = [0_i16; BLOCK_AREA - 1];
    for (slot, value) in ordered.iter_mut().enumerate() {
        *value = block[ZIGZAG[slot + 1]];
    }
    for slot in 0..ac {
        let src = (slot + shift) % ac;
        block[ZIGZAG[slot + 1]] = ordered[src];
    }
}

#[inline]
fn rgb_pixel_to_ycbcr(r: u8, g: u8, b: u8) -> [u8; CHANNELS] {
    let r = r as i32;
    let g = g as i32;
    let b = b as i32;
    let y = (77 * r + 150 * g + 29 * b + 128) >> 8;
    let cb = ((-43 * r - 85 * g + 128 * b + 128) >> 8) + 128;
    let cr = ((128 * r - 107 * g - 21 * b + 128) >> 8) + 128;
    [
        y.clamp(0, 255) as u8,
        cb.clamp(0, 255) as u8,
        cr.clamp(0, 255) as u8,
    ]
}

#[inline]
fn ycbcr_to_rgb_pixel(y: f32, cb: f32, cr: f32) -> [u8; CHANNELS] {
    let y = y as i32;
    let cb = cb as i32 - 128;
    let cr = cr as i32 - 128;
    [
        (y + ((359 * cr + 128) >> 8)).clamp(0, 255) as u8,
        (y - ((88 * cb + 183 * cr + 128) >> 8)).clamp(0, 255) as u8,
        (y + ((454 * cb + 128) >> 8)).clamp(0, 255) as u8,
    ]
}

fn is_every(period: u64, ordinal: u64) -> bool {
    period != 0 && ordinal % period == 0
}

fn hash_u64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9e37_79b9_7f4a_7c15);
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    x ^ (x >> 31)
}

fn invalid_input(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, message.into())
}

pub fn load_dct_transform_preset(name: &str, params: &mut DctGlitchParams) -> Result<(), String> {
    *params = DctGlitchParams::default();
    match name {
        "clean" => {}
        "subtle" => {
            params.quant_scale = 2.0;
        }
        "blocks" | "quantize" | "coeff" => {
            params.quant_scale = 8.0;
        }
        "smear" | "dc-smear" | "melt" => {
            // A few discrete DC corruptions that propagate forward (scan order), giving
            // distinct colour regions that bleed block-by-block rather than washing the
            // whole frame out — the classic JPEG DC-predictor smear.
            params.dc_drift = 12;
            params.dc_drift_every = 80;
            params.persistence = 0.5;
        }
        "bleed" | "drift" => {
            params.dc_block_offset = 24;
            params.dc_block_offset_every = 5;
        }
        "blur" | "lowpass" => {
            params.ac_zero_above = 3;
            params.quant_scale = 2.0;
        }
        "ring" | "invert" => {
            params.coeff_sign_flip_every = 3;
            params.block_transpose_every = 5;
        }
        "scramble" | "shuffle" => {
            params.coeff_shift = 7;
            params.coeff_shift_every = 2;
            params.zigzag_reverse_every = 11;
        }
        "slip" | "block-slip" | "vector" => {
            params.block_shift_x = 3;
            params.block_shift_y = 1;
            params.block_shift_every = 4;
        }
        "echo" | "repeat" | "codebook" => {
            params.block_repeat_every = 5;
        }
        "flow" | "feedback" | "drift-flow" => {
            params.dc_block_offset = 18;
            params.dc_block_offset_every = 6;
            params.persistence = 0.7;
        }
        // Entropy-bitstream presets: the coefficient side stays clean; the corruption lives
        // in the DTE0 byte stream (see load_dct_bitstream_preset).
        "desync" | "entropy" | "decoder-desync" => {}
        "shred" | "scan-slip" => {}
        "truncate" | "tail" => {}
        "false-color" | "falsecolor" | "chroma-swap" => {
            // Swap Cb/Cr on alternating chroma blocks (reds<->blues) with a little DC drift
            // so the false colour also moves — a 4:2:0-specific palette glitch.
            params.chroma_swap_every = 2;
            params.dc_drift = 8;
            params.dc_drift_every = 50;
        }
        "destroy" | "composite" | "unstable" | "balanced" => {
            params.quant_scale = 5.0;
            params.dc_drift = 7;
            params.dc_drift_every = 96;
            params.dc_block_offset = 12;
            params.dc_block_offset_every = 11;
            params.ac_zero_above = 6;
            params.coeff_sign_flip_every = 17;
            params.coeff_shift = 5;
            params.coeff_shift_every = 7;
            params.block_shift_x = 2;
            params.block_shift_y = 1;
            params.block_shift_every = 19;
            params.block_repeat_every = 23;
            params.zigzag_reverse_every = 29;
            params.block_transpose_every = 31;
            params.chroma_swap_every = 37;
            params.persistence = 0.4;
        }
        _ => {
            return Err(format!(
                "unknown dct-transform preset `{name}`; expected clean, subtle, blocks, dc-smear, bleed, blur, ring, scramble, block-slip, echo, flow, false-color, desync, shred, truncate, or composite"
            ));
        }
    }
    Ok(())
}

pub fn apply_dct_transform_controls(params: &mut DctGlitchParams, controls: RawMoshControls) {
    // `intensity` is the master; the sub-macros bias glitch groups. At the default
    // controls (all 1.0) every factor is 1.0, so the authored preset is left untouched.
    let master = finite_control(controls.intensity);
    let amount = |sub: f32| master * finite_control(sub);
    let quant = amount(controls.bitstream); // "Quant"
    let dc = amount(controls.temporal); // "DC"
    let structure = amount(controls.motion); // "Structure"
    params.quant_scale = (1.0 + (params.quant_scale - 1.0) * quant).max(1.0);
    params.dc_drift = scale_i16(params.dc_drift, dc);
    params.dc_drift_every = scale_event_interval(params.dc_drift_every, dc);
    params.dc_block_offset = scale_i16(params.dc_block_offset, dc);
    params.dc_block_offset_every = scale_event_interval(params.dc_block_offset_every, dc);
    params.ac_zero_above = scale_ac_cutoff(params.ac_zero_above, quant);
    params.coeff_sign_flip_every = scale_event_interval(params.coeff_sign_flip_every, structure);
    params.coeff_shift = scale_i16(params.coeff_shift, structure);
    params.coeff_shift_every = scale_event_interval(params.coeff_shift_every, structure);
    params.block_shift_x = scale_i16(params.block_shift_x, structure);
    params.block_shift_y = scale_i16(params.block_shift_y, structure);
    params.block_shift_every = scale_event_interval(params.block_shift_every, structure);
    params.block_repeat_every = scale_event_interval(params.block_repeat_every, structure);
    params.zigzag_reverse_every = scale_event_interval(params.zigzag_reverse_every, structure);
    params.block_transpose_every = scale_event_interval(params.block_transpose_every, structure);
    params.chroma_swap_every = scale_event_interval(params.chroma_swap_every, structure);
    params.persistence =
        (params.persistence * master * finite_control(controls.residual)).clamp(0.0, 0.98);
}

fn finite_control(value: f32) -> f32 {
    if value.is_finite() {
        value.max(0.0)
    } else {
        0.0
    }
}

pub fn set_dct_transform_parameter(
    params: &mut DctGlitchParams,
    id: &str,
    value: f32,
) -> Result<(), String> {
    let finite = if value.is_finite() { value } else { 0.0 };
    let as_u64 = |v: f32| v.max(0.0).round() as u64;
    let as_i16 = |v: f32| v.round().clamp(i16::MIN as f32, i16::MAX as f32) as i16;
    match id {
        "quant_scale" => params.quant_scale = finite.max(1.0),
        "dc_drift" => params.dc_drift = as_i16(finite),
        "dc_drift_every" => params.dc_drift_every = as_u64(finite),
        "dc_block_offset" => params.dc_block_offset = as_i16(finite),
        "dc_block_offset_every" => params.dc_block_offset_every = as_u64(finite),
        "ac_zero_above" => params.ac_zero_above = (as_u64(finite) as usize).min(BLOCK_AREA - 1),
        "coeff_sign_flip_every" => params.coeff_sign_flip_every = as_u64(finite),
        "coeff_shift" => params.coeff_shift = as_i16(finite),
        "coeff_shift_every" => params.coeff_shift_every = as_u64(finite),
        "block_shift_x" => params.block_shift_x = as_i16(finite),
        "block_shift_y" => params.block_shift_y = as_i16(finite),
        "block_shift_every" => params.block_shift_every = as_u64(finite),
        "block_repeat_every" => params.block_repeat_every = as_u64(finite),
        "zigzag_reverse_every" => params.zigzag_reverse_every = as_u64(finite),
        "block_transpose_every" => params.block_transpose_every = as_u64(finite),
        "chroma_swap_every" => params.chroma_swap_every = as_u64(finite),
        "persistence" => params.persistence = finite.clamp(0.0, 0.98),
        _ => return Err(format!("unknown dct-transform parameter `{id}`")),
    }
    Ok(())
}

fn scale_i16(value: i16, amount: f32) -> i16 {
    (value as f32 * amount)
        .round()
        .clamp(i16::MIN as f32, i16::MAX as f32) as i16
}

fn scale_event_interval(interval: u64, amount: f32) -> u64 {
    if interval == 0 || amount <= 0.0 {
        return 0;
    }
    (interval as f32 / amount).round().max(1.0) as u64
}

fn scale_ac_cutoff(cutoff: usize, amount: f32) -> usize {
    if cutoff == 0 || amount <= 0.0 {
        return 0;
    }
    let clean = (BLOCK_AREA - 1) as f32;
    (clean + (cutoff as f32 - clean) * amount)
        .round()
        .clamp(1.0, clean) as usize
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gradient(width: usize, height: usize) -> Vec<u8> {
        let mut input = vec![0_u8; width * height * CHANNELS];
        for y in 0..height {
            for x in 0..width {
                let i = (y * width + x) * CHANNELS;
                input[i] = ((x * 3) & 0xff) as u8;
                input[i + 1] = ((y * 5 + 20) & 0xff) as u8;
                input[i + 2] = (((x + y) * 2 + 40) & 0xff) as u8;
            }
        }
        input
    }

    fn mean_abs_error(a: &[u8], b: &[u8]) -> f64 {
        let total: u64 = a
            .iter()
            .zip(b)
            .map(|(x, y)| (*x as i32 - *y as i32).unsigned_abs() as u64)
            .sum();
        total as f64 / a.len() as f64
    }

    #[test]
    fn fast_dct_matches_matmul() {
        // The fast even/odd transform must agree with the exact matrix product on
        // arbitrary blocks (both directions). This is the correctness guard that lets the
        // codec use the fast path. Deterministic pseudo-random input, no RNG crate.
        let mut state: u32 = 0x1234_5678;
        let mut next = || {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (state >> 9) as f32 / 8_388_608.0 * 255.0 - 128.0
        };
        let mut max_fwd = 0.0_f32;
        let mut max_inv = 0.0_f32;
        for _ in 0..64 {
            let mut block = [0.0_f32; BLOCK_AREA];
            for v in block.iter_mut() {
                *v = next();
            }
            let f_fast = forward_dct_fast(&block);
            let f_mat = forward_dct(&block);
            let i_fast = inverse_dct_fast(&block);
            let i_mat = inverse_dct(&block);
            for k in 0..BLOCK_AREA {
                max_fwd = max_fwd.max((f_fast[k] - f_mat[k]).abs());
                max_inv = max_inv.max((i_fast[k] - i_mat[k]).abs());
            }
        }
        assert!(
            max_fwd < 1e-2,
            "forward fast vs matmul diff too large: {max_fwd}"
        );
        assert!(
            max_inv < 1e-2,
            "inverse fast vs matmul diff too large: {max_inv}"
        );
    }

    #[test]
    #[ignore = "timing micro-bench; run with --ignored --nocapture"]
    fn bench_dct_ab() {
        use std::time::Instant;
        let mut state: u32 = 0x9e37_79b9;
        let mut next = || {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (state >> 9) as f32 / 8_388_608.0 * 255.0 - 128.0
        };
        let mut blocks = Vec::with_capacity(64);
        for _ in 0..64 {
            let mut b = [0.0_f32; BLOCK_AREA];
            for v in b.iter_mut() {
                *v = next();
            }
            blocks.push(b);
        }
        let n = 400_000usize;
        let mut acc = 0.0_f32;
        let t = Instant::now();
        for i in 0..n {
            acc += forward_dct(&blocks[i & 63])[0];
        }
        let mat_fwd = t.elapsed();
        let t = Instant::now();
        for i in 0..n {
            acc += forward_dct_fast(&blocks[i & 63])[0];
        }
        let fast_fwd = t.elapsed();
        let t = Instant::now();
        for i in 0..n {
            acc += inverse_dct(&blocks[i & 63])[0];
        }
        let mat_inv = t.elapsed();
        let t = Instant::now();
        for i in 0..n {
            acc += inverse_dct_fast(&blocks[i & 63])[0];
        }
        let fast_inv = t.elapsed();
        println!(
            "FWD matmul={mat_fwd:?} fast={fast_fwd:?} speedup={:.2}x",
            mat_fwd.as_secs_f64() / fast_fwd.as_secs_f64()
        );
        println!(
            "INV matmul={mat_inv:?} fast={fast_inv:?} speedup={:.2}x",
            mat_inv.as_secs_f64() / fast_inv.as_secs_f64()
        );
        println!("acc={acc}");
    }

    #[test]
    fn clean_round_trip_is_close_at_high_quality() {
        let config = DctCodecConfig {
            width: 80,
            height: 48,
            quality: 95,
        };
        let mut codec = DctCodec::new(config).unwrap();
        let input = gradient(80, 48);
        let mut output = vec![0_u8; input.len()];
        codec
            .process_rgb_frame(&input, &DctGlitchParams::default(), &mut output)
            .unwrap();
        let mae = mean_abs_error(&input, &output);
        assert!(mae < 12.0, "clean DCT round-trip MAE too high: {mae}");
    }

    #[test]
    fn clean_bitstream_path_matches_normal_path() {
        // The entropy stage is lossless: with no corruption the DTE0 path must reproduce the
        // exact frame the coefficient path produces.
        let config = DctCodecConfig {
            width: 80,
            height: 48,
            quality: 70,
        };
        let input = gradient(80, 48);
        let params = DctGlitchParams::default();
        let bs = DctBitstreamParams::default();
        let mut a = DctCodec::new(config).unwrap();
        let mut out_normal = vec![0u8; input.len()];
        a.process_rgb_frame(&input, &params, &mut out_normal)
            .unwrap();
        let mut b = DctCodec::new(config).unwrap();
        let mut out_bits = vec![0u8; input.len()];
        b.process_rgb_frame_bitstream(&input, &params, &bs, &mut out_bits)
            .unwrap();
        assert_eq!(
            out_normal, out_bits,
            "clean entropy path must equal the normal clean path"
        );
    }

    #[test]
    fn intensity_zero_disables_every_transform_mutation() {
        let presets = [
            "blocks",
            "dc-smear",
            "bleed",
            "blur",
            "ring",
            "scramble",
            "block-slip",
            "echo",
            "flow",
            "false-color",
            "composite",
        ];

        for preset in presets {
            let mut params = DctGlitchParams::default();
            load_dct_transform_preset(preset, &mut params).unwrap();
            apply_dct_transform_controls(
                &mut params,
                RawMoshControls {
                    intensity: 0.0,
                    ..RawMoshControls::default()
                },
            );
            assert!(
                !params.has_mutations(),
                "intensity zero left `{preset}` active: {params:?}"
            );
        }
    }

    #[test]
    fn bitstream_glitch_changes_output_without_panicking() {
        let config = DctCodecConfig {
            width: 96,
            height: 64,
            quality: 60,
        };
        let input = gradient(96, 64);
        let params = DctGlitchParams::default();
        let bs = DctBitstreamParams {
            enabled: true,
            byte_flip_every: 9,
            drop_every: 23,
            ..Default::default()
        };
        let mut clean = DctCodec::new(config).unwrap();
        let mut clean_out = vec![0u8; input.len()];
        clean
            .process_rgb_frame(&input, &params, &mut clean_out)
            .unwrap();
        let mut codec = DctCodec::new(config).unwrap();
        let mut glitched = vec![0u8; input.len()];
        codec
            .process_rgb_frame_bitstream(&input, &params, &bs, &mut glitched)
            .unwrap();
        assert_ne!(
            clean_out, glitched,
            "entropy corruption should change the frame"
        );
    }

    #[test]
    fn non_multiple_of_block_dimensions_round_trip() {
        // 70x37 is not a multiple of 8 in either axis; edges must still reconstruct.
        let config = DctCodecConfig {
            width: 70,
            height: 37,
            quality: 90,
        };
        let mut codec = DctCodec::new(config).unwrap();
        let input = gradient(70, 37);
        let mut output = vec![0_u8; input.len()];
        codec
            .process_rgb_frame(&input, &DctGlitchParams::default(), &mut output)
            .unwrap();
        assert!(mean_abs_error(&input, &output) < 14.0);
    }

    #[test]
    fn glitches_change_output_without_panicking() {
        let config = DctCodecConfig {
            width: 64,
            height: 64,
            quality: 60,
        };
        let input = gradient(64, 64);
        let presets = [
            DctGlitchParams {
                quant_scale: 6.0,
                ..DctGlitchParams::default()
            },
            DctGlitchParams {
                dc_drift: 40,
                dc_drift_every: 7,
                ..DctGlitchParams::default()
            },
            DctGlitchParams {
                ac_zero_above: 3,
                ..DctGlitchParams::default()
            },
            DctGlitchParams {
                coeff_shift: 5,
                coeff_shift_every: 3,
                block_shift_x: 2,
                block_shift_y: 1,
                block_shift_every: 4,
                ..DctGlitchParams::default()
            },
        ];
        for params in presets {
            let mut codec = DctCodec::new(config).unwrap();
            let mut clean = vec![0_u8; input.len()];
            codec
                .process_rgb_frame(&input, &DctGlitchParams::default(), &mut clean)
                .unwrap();
            let mut glitched = vec![0_u8; input.len()];
            codec
                .process_rgb_frame(&input, &params, &mut glitched)
                .unwrap();
            assert!(params.has_mutations());
            assert_ne!(
                clean, glitched,
                "glitch produced identical output: {params:?}"
            );
        }
    }
}
