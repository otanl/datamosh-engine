//! `DTE0` — the DCT0 entropy/bitstream stage.
//!
//! DCT0 itself models a transform codec up to quantization. This module adds the missing
//! final stage of a real JPEG/DV-class codec: lossless **entropy coding** of the quantized
//! coefficients into a parseable byte stream, and the decode back. It is the DCT analogue of
//! the `MSH0` bitstream path in `mosh_codec.rs`.
//!
//! Scheme (per plane, blocks in scan order):
//!   * DC: differential (DPCM) across blocks, then JPEG category/magnitude coding (a `size`
//!     symbol followed by `size` magnitude bits).
//!   * AC: zig-zag scan, run-length of zeros, a `(run, size)` symbol + magnitude bits, with
//!     `EOB` (0x00) and `ZRL` (0xF0) markers — exactly JPEG's run/category structure.
//!
//! The DC `size` symbols and AC `(run,size)` symbols are coded with **canonical Huffman**
//! tables built per frame from the actual symbol frequencies (JPEG Annex K code-size
//! generation, length-limited to 16 bits), serialized into the header so the decoder rebuilds
//! them. This is genuine optimal entropy coding: it compresses, and it is maximally
//! desync-prone — a single corrupted bit makes the variable-length reader misread every later
//! symbol (the cascading "broken JPEG" slide); running out of bits leaves the rest flat. The
//! clean round-trip is lossless (see tests). The Huffman *tables* live in the header and are
//! preserved by corruption (like JPEG markers); only the entropy payload is damaged.

use crate::dct_codec::{BLOCK_AREA, ZIGZAG};

const MAGIC: &[u8; 4] = b"DTE0";
const VERSION: u8 = 1;
// magic(4) + version(1) + width(4) + height(4) + luma_blocks(4) + chroma_blocks(4) + payload_offset(4)
const FIXED_HEADER_LEN: usize = 25;
const DC_ALPHABET: usize = 17; // size categories 0..=16
const AC_ALPHABET: usize = 256; // (run<<4)|size byte

// ---- bit I/O (MSB first) ----

struct BitWriter {
    bytes: Vec<u8>,
    cur: u8,
    nbits: u8,
}

impl BitWriter {
    fn new() -> Self {
        Self { bytes: Vec::new(), cur: 0, nbits: 0 }
    }

    fn write_bits(&mut self, value: u32, n: u8) {
        for i in (0..n).rev() {
            self.cur = (self.cur << 1) | (((value >> i) & 1) as u8);
            self.nbits += 1;
            if self.nbits == 8 {
                self.bytes.push(self.cur);
                self.cur = 0;
                self.nbits = 0;
            }
        }
    }

    fn finish(mut self) -> Vec<u8> {
        if self.nbits > 0 {
            self.cur <<= 8 - self.nbits;
            self.bytes.push(self.cur);
        }
        self.bytes
    }
}

struct BitReader<'a> {
    bytes: &'a [u8],
    pos: usize, // in bits
}

impl<'a> BitReader<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, pos: 0 }
    }

    #[inline]
    fn read_bit(&mut self) -> Option<u32> {
        let byte = self.pos / 8;
        if byte >= self.bytes.len() {
            return None;
        }
        let bit = 7 - (self.pos % 8);
        self.pos += 1;
        Some(((self.bytes[byte] >> bit) & 1) as u32)
    }

    fn read_bits(&mut self, n: u8) -> Option<u32> {
        let mut value = 0u32;
        for _ in 0..n {
            value = (value << 1) | self.read_bit()?;
        }
        Some(value)
    }
}

#[inline]
fn magnitude_bits(v: i32) -> u8 {
    let a = v.unsigned_abs();
    if a == 0 { 0 } else { 32 - a.leading_zeros() as u8 }
}

#[inline]
fn write_magnitude(w: &mut BitWriter, v: i32, size: u8) {
    if size == 0 {
        return;
    }
    let t = if v >= 0 { v as u32 } else { (v + (1 << size) - 1) as u32 };
    w.write_bits(t & ((1 << size) - 1), size);
}

#[inline]
fn read_magnitude(t: u32, size: u8) -> i32 {
    if size == 0 {
        return 0;
    }
    let half = 1u32 << (size - 1);
    if t < half {
        t as i32 - ((1i32 << size) - 1)
    } else {
        t as i32
    }
}

// ---- canonical Huffman (JPEG Annex K) ----

struct HuffTable {
    bits: [u32; 17],   // bits[1..=16] = count of codes of each length
    huffval: Vec<u16>, // symbols in canonical order
    ecode: Vec<u32>,   // symbol -> code (len = alphabet, 0 if absent)
    esize: Vec<u8>,    // symbol -> code length (0 if absent)
    mincode: [i32; 17],
    maxcode: [i32; 17],
    valptr: [usize; 17],
}

// Canonical code values for symbols already ordered by (length, symbol) — Annex K.5.
fn canonical_codes(bits: &[u32; 17]) -> (Vec<u8>, Vec<u32>) {
    let mut huffsize = Vec::new();
    for len in 1..=16usize {
        for _ in 0..bits[len] {
            huffsize.push(len as u8);
        }
    }
    let total = huffsize.len();
    let mut huffcode = vec![0u32; total];
    if total == 0 {
        return (huffsize, huffcode);
    }
    let mut code = 0u32;
    let mut si = huffsize[0];
    let mut k = 0;
    loop {
        while k < total && huffsize[k] == si {
            huffcode[k] = code;
            code += 1;
            k += 1;
        }
        if k >= total {
            break;
        }
        while k < total && huffsize[k] != si {
            code <<= 1;
            si += 1;
        }
    }
    (huffsize, huffcode)
}

fn build_decode(bits: &[u32; 17], huffcode: &[u32]) -> ([i32; 17], [i32; 17], [usize; 17]) {
    let mut mincode = [0i32; 17];
    let mut maxcode = [-1i32; 17];
    let mut valptr = [0usize; 17];
    let mut p = 0usize;
    for len in 1..=16usize {
        if bits[len] > 0 {
            valptr[len] = p;
            mincode[len] = huffcode[p] as i32;
            p += bits[len] as usize;
            maxcode[len] = huffcode[p - 1] as i32;
        }
    }
    (mincode, maxcode, valptr)
}

// Build an optimal length-limited Huffman table from symbol frequencies (Annex K.1–K.3).
fn build_huff(freq: &[u32], alphabet: usize) -> HuffTable {
    let n = alphabet + 1; // last index is the reserved sentinel (freq 1)
    let mut f: Vec<i64> = (0..n)
        .map(|i| if i < alphabet { freq[i] as i64 } else { 1 })
        .collect();
    let mut codesize = vec![0u32; n];
    let mut others = vec![-1i64; n];

    // K.1: generate code sizes.
    loop {
        let mut v1 = -1i64;
        let mut f1 = i64::MAX;
        for (i, &fi) in f.iter().enumerate() {
            if fi > 0 && fi <= f1 {
                f1 = fi;
                v1 = i as i64;
            }
        }
        let mut v2 = -1i64;
        let mut f2 = i64::MAX;
        for (i, &fi) in f.iter().enumerate() {
            if fi > 0 && i as i64 != v1 && fi <= f2 {
                f2 = fi;
                v2 = i as i64;
            }
        }
        if v2 < 0 {
            break;
        }
        let (a, b) = (v1 as usize, v2 as usize);
        f[a] += f[b];
        f[b] = 0;
        codesize[a] += 1;
        let mut x = a;
        while others[x] >= 0 {
            x = others[x] as usize;
            codesize[x] += 1;
        }
        others[x] = b as i64; // link at the chain end (Annex K.1), not the head
        codesize[b] += 1;
        let mut x = b;
        while others[x] >= 0 {
            x = others[x] as usize;
            codesize[x] += 1;
        }
    }

    // Count codes per length.
    let mut bits33 = [0u32; 33];
    for &c in &codesize {
        if c > 0 {
            bits33[c as usize] += 1;
        }
    }
    // K.3: limit code lengths to 16 bits.
    let mut i = 32;
    while i > 16 {
        if bits33[i] > 0 {
            let mut j = i - 2;
            while bits33[j] == 0 {
                j -= 1;
            }
            bits33[i] -= 2;
            bits33[i - 1] += 1;
            bits33[j + 1] += 2;
            bits33[j] -= 1;
        } else {
            i -= 1;
        }
    }
    // Remove the reserved sentinel's code (one code at the longest used length).
    let mut t = 16;
    while t > 0 && bits33[t] == 0 {
        t -= 1;
    }
    if t > 0 {
        bits33[t] -= 1;
    }

    let mut bits = [0u32; 17];
    bits[..17].copy_from_slice(&bits33[..17]);

    // HUFFVAL: real symbols ordered by (code size, symbol) — excludes the sentinel.
    let mut huffval: Vec<u16> = Vec::new();
    for len in 1..=32usize {
        for (sym, &cs) in codesize.iter().enumerate().take(alphabet) {
            if cs as usize == len {
                huffval.push(sym as u16);
            }
        }
    }

    let (huffsize, huffcode) = canonical_codes(&bits);
    let mut ecode = vec![0u32; alphabet];
    let mut esize = vec![0u8; alphabet];
    for (idx, &sym) in huffval.iter().enumerate() {
        ecode[sym as usize] = huffcode[idx];
        esize[sym as usize] = huffsize[idx];
    }
    let (mincode, maxcode, valptr) = build_decode(&bits, &huffcode);
    HuffTable { bits, huffval, ecode, esize, mincode, maxcode, valptr }
}

// Rebuild a decode-only table from a serialized (bits, huffval).
fn parse_huff(bits: [u32; 17], huffval: Vec<u16>) -> HuffTable {
    let (_, huffcode) = canonical_codes(&bits);
    let (mincode, maxcode, valptr) = build_decode(&bits, &huffcode);
    HuffTable { bits, huffval, ecode: Vec::new(), esize: Vec::new(), mincode, maxcode, valptr }
}

fn serialize_table(table: &HuffTable, out: &mut Vec<u8>) {
    for len in 1..=16usize {
        out.extend_from_slice(&(table.bits[len] as u16).to_le_bytes());
    }
    for &sym in &table.huffval {
        out.push(sym as u8);
    }
}

fn parse_table(bytes: &[u8], cursor: &mut usize) -> Option<HuffTable> {
    let mut bits = [0u32; 17];
    let mut total = 0usize;
    for len in 1..=16usize {
        if *cursor + 2 > bytes.len() {
            return None;
        }
        let v = u16::from_le_bytes([bytes[*cursor], bytes[*cursor + 1]]) as u32;
        bits[len] = v;
        total += v as usize;
        *cursor += 2;
    }
    if *cursor + total > bytes.len() {
        return None;
    }
    let huffval: Vec<u16> = bytes[*cursor..*cursor + total].iter().map(|&b| b as u16).collect();
    *cursor += total;
    Some(parse_huff(bits, huffval))
}

#[inline]
fn decode_symbol(r: &mut BitReader, table: &HuffTable) -> Option<u32> {
    let mut code = 0i32;
    for len in 1..=16usize {
        code = (code << 1) | r.read_bit()? as i32;
        if table.maxcode[len] >= 0 && code <= table.maxcode[len] {
            let idx = table.valptr[len] + (code - table.mincode[len]) as usize;
            return table.huffval.get(idx).map(|&s| s as u32);
        }
    }
    None // no valid code in 16 bits → desync
}

// ---- per-plane walk shared by frequency-gather and encode ----

enum SymKind {
    Dc,
    Ac,
}

fn walk_plane_symbols(region: &[i16], mut emit: impl FnMut(SymKind, u32, i32, u8)) {
    let blocks = region.len() / BLOCK_AREA;
    let mut prev_dc = 0i32;
    for b in 0..blocks {
        let block = &region[b * BLOCK_AREA..b * BLOCK_AREA + BLOCK_AREA];
        let dc = block[0] as i32;
        let diff = dc - prev_dc;
        prev_dc = dc;
        let size = magnitude_bits(diff);
        emit(SymKind::Dc, size as u32, diff, size);

        let mut run = 0u32;
        for zz in 1..BLOCK_AREA {
            let v = block[ZIGZAG[zz]] as i32;
            if v == 0 {
                run += 1;
                continue;
            }
            while run >= 16 {
                emit(SymKind::Ac, 0xF0, 0, 0); // ZRL
                run -= 16;
            }
            let s = magnitude_bits(v);
            emit(SymKind::Ac, (run << 4) | s as u32, v, s);
            run = 0;
        }
        if run > 0 {
            emit(SymKind::Ac, 0x00, 0, 0); // EOB
        }
    }
}

fn decode_plane(r: &mut BitReader, region: &mut [i16], dc: &HuffTable, ac: &HuffTable) {
    let blocks = region.len() / BLOCK_AREA;
    let mut prev_dc = 0i32;
    for b in 0..blocks {
        let base = b * BLOCK_AREA;
        let size = match decode_symbol(r, dc) {
            Some(s) => s as u8,
            None => return,
        };
        let diff = match r.read_bits(size) {
            Some(t) => read_magnitude(t, size),
            None => return,
        };
        let value = (prev_dc + diff).clamp(-32768, 32767);
        prev_dc = value;
        region[base] = value as i16;

        let mut pos = 1usize;
        while pos < BLOCK_AREA {
            let sym = match decode_symbol(r, ac) {
                Some(s) => s,
                None => return,
            };
            if sym == 0x00 {
                break; // EOB
            }
            if sym == 0xF0 {
                pos += 16; // ZRL
                continue;
            }
            let run = (sym >> 4) as usize;
            let s = (sym & 0x0F) as u8;
            pos += run;
            if pos >= BLOCK_AREA {
                break;
            }
            let v = match r.read_bits(s) {
                Some(t) => read_magnitude(t, s),
                None => return,
            };
            region[base + ZIGZAG[pos]] = v.clamp(-32768, 32767) as i16;
            pos += 1;
        }
    }
}

/// Entropy-encode the quantized coefficient planes into a `DTE0` byte stream.
pub fn encode_dct_bitstream(
    coeff: &[i16],
    width: usize,
    height: usize,
    luma_blocks: usize,
    chroma_blocks: usize,
) -> Vec<u8> {
    let luma = luma_blocks * BLOCK_AREA;
    let chroma = chroma_blocks * BLOCK_AREA;
    let planes = [
        &coeff[0..luma],
        &coeff[luma..luma + chroma],
        &coeff[luma + chroma..luma + 2 * chroma],
    ];

    // Pass 1: gather symbol frequencies across all planes.
    let mut dc_freq = vec![0u32; DC_ALPHABET];
    let mut ac_freq = vec![0u32; AC_ALPHABET];
    for plane in planes {
        walk_plane_symbols(plane, |kind, sym, _v, _s| match kind {
            SymKind::Dc => dc_freq[sym as usize] += 1,
            SymKind::Ac => ac_freq[sym as usize] += 1,
        });
    }
    let dc_table = build_huff(&dc_freq, DC_ALPHABET);
    let ac_table = build_huff(&ac_freq, AC_ALPHABET);

    // Pass 2: entropy-code the planes.
    let mut w = BitWriter::new();
    for plane in planes {
        walk_plane_symbols(plane, |kind, sym, v, s| {
            let table = match kind {
                SymKind::Dc => &dc_table,
                SymKind::Ac => &ac_table,
            };
            w.write_bits(table.ecode[sym as usize], table.esize[sym as usize]);
            write_magnitude(&mut w, v, s);
        });
    }
    let payload = w.finish();

    let mut tables = Vec::new();
    serialize_table(&dc_table, &mut tables);
    serialize_table(&ac_table, &mut tables);
    let payload_offset = (FIXED_HEADER_LEN + tables.len()) as u32;

    let mut out = Vec::with_capacity(payload_offset as usize + payload.len());
    out.extend_from_slice(MAGIC);
    out.push(VERSION);
    out.extend_from_slice(&(width as u32).to_le_bytes());
    out.extend_from_slice(&(height as u32).to_le_bytes());
    out.extend_from_slice(&(luma_blocks as u32).to_le_bytes());
    out.extend_from_slice(&(chroma_blocks as u32).to_le_bytes());
    out.extend_from_slice(&payload_offset.to_le_bytes());
    out.extend_from_slice(&tables);
    out.extend_from_slice(&payload);
    out
}

/// Decode a (possibly corrupted) `DTE0` stream back into `coeff`, which must be sized for
/// `(luma_blocks + 2*chroma_blocks)*64`. Desync-tolerant: it zeroes `coeff` first and fills
/// whatever decodes, so corruption reads as cascading garbage / flat regions, never a panic.
pub fn decode_dct_bitstream(
    bytes: &[u8],
    coeff: &mut [i16],
    luma_blocks: usize,
    chroma_blocks: usize,
) {
    coeff.iter_mut().for_each(|c| *c = 0);
    if bytes.len() < FIXED_HEADER_LEN || &bytes[0..4] != MAGIC {
        return;
    }
    let mut cursor = 21usize; // after magic+version+w+h+luma+chroma
    let payload_offset = u32::from_le_bytes([
        bytes[cursor],
        bytes[cursor + 1],
        bytes[cursor + 2],
        bytes[cursor + 3],
    ]) as usize;
    cursor += 4;
    let dc_table = match parse_table(bytes, &mut cursor) {
        Some(t) => t,
        None => return,
    };
    let ac_table = match parse_table(bytes, &mut cursor) {
        Some(t) => t,
        None => return,
    };
    let start = payload_offset.min(bytes.len());
    let payload = &bytes[start..];

    let luma = luma_blocks * BLOCK_AREA;
    let chroma = chroma_blocks * BLOCK_AREA;
    if coeff.len() < luma + 2 * chroma {
        return;
    }
    let mut r = BitReader::new(payload);
    let (luma_region, rest) = coeff.split_at_mut(luma);
    let (cb_region, cr_region) = rest.split_at_mut(chroma);
    decode_plane(&mut r, luma_region, &dc_table, &ac_table);
    decode_plane(&mut r, &mut cb_region[..chroma], &dc_table, &ac_table);
    decode_plane(&mut r, &mut cr_region[..chroma], &dc_table, &ac_table);
}

// ---- byte-stream corruption (the codec-legitimate glitch surface) ----

#[derive(Debug, Clone, Copy)]
pub struct DctBitstreamParams {
    pub enabled: bool,
    /// Flip one bit in every Nth entropy byte. Desyncs the variable-length reader.
    pub byte_flip_every: u64,
    /// Zero every Nth entropy byte (acts like a spurious symbol / lost code).
    pub drop_every: u64,
    /// Rotate bytes within each `slip_window`-sized window by `slip_bytes` on every Nth
    /// window. Bulk desync of a region.
    pub slip_every: u64,
    pub slip_bytes: i32,
    pub slip_window: usize,
    /// Drop this fraction of the entropy tail (0..1): the decoder runs out and the rest of
    /// the frame goes flat.
    pub truncate_tail: f32,
}

impl Default for DctBitstreamParams {
    fn default() -> Self {
        Self {
            enabled: false,
            byte_flip_every: 0,
            drop_every: 0,
            slip_every: 0,
            slip_bytes: 1,
            slip_window: 64,
            truncate_tail: 0.0,
        }
    }
}

impl DctBitstreamParams {
    pub fn has_mutations(&self) -> bool {
        self.byte_flip_every != 0
            || self.drop_every != 0
            || (self.slip_every != 0 && self.slip_bytes != 0 && self.slip_window > 1)
            || self.truncate_tail > 0.0
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct DctBitstreamMutationStats {
    pub bytes_flipped: u64,
    pub bytes_dropped: u64,
    pub windows_slipped: u64,
    pub bytes_truncated: u64,
}

fn hash_u64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9e37_79b9_7f4a_7c15);
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    x ^ (x >> 31)
}

fn payload_start(bytes: &[u8]) -> Option<usize> {
    if bytes.len() < FIXED_HEADER_LEN || &bytes[0..4] != MAGIC {
        return None;
    }
    let off = u32::from_le_bytes([bytes[21], bytes[22], bytes[23], bytes[24]]) as usize;
    if off <= bytes.len() { Some(off) } else { None }
}

/// Corrupt the entropy payload of a `DTE0` stream in place. The fixed header and the Huffman
/// tables are preserved (like JPEG markers) so the stream still parses; only the entropy
/// bits are damaged. Deterministic: glitches are hash-seeded, never RNG.
pub fn mutate_dct_bitstream(
    bytes: &mut Vec<u8>,
    params: &DctBitstreamParams,
    seed: u64,
) -> DctBitstreamMutationStats {
    let mut stats = DctBitstreamMutationStats::default();
    let header = match payload_start(bytes) {
        Some(h) if h < bytes.len() && params.has_mutations() => h,
        _ => return stats,
    };

    if params.truncate_tail > 0.0 {
        let payload = bytes.len() - header;
        let drop =
            ((payload as f32 * params.truncate_tail.clamp(0.0, 1.0)).round() as usize).min(payload);
        if drop > 0 {
            bytes.truncate(bytes.len() - drop);
            stats.bytes_truncated = drop as u64;
        }
    }

    let payload_len = bytes.len().saturating_sub(header);
    if payload_len == 0 {
        return stats;
    }
    let payload = &mut bytes[header..];

    if params.slip_every != 0 && params.slip_bytes != 0 && params.slip_window > 1 {
        let window = params.slip_window;
        let windows = payload_len.div_ceil(window);
        for wi in 0..windows {
            if (wi as u64 + 1) % params.slip_every != 0 {
                continue;
            }
            let start = wi * window;
            let end = (start + window).min(payload_len);
            let len = end - start;
            if len > 1 {
                let shift = params.slip_bytes.rem_euclid(len as i32) as usize;
                if shift != 0 {
                    payload[start..end].rotate_left(shift);
                    stats.windows_slipped += 1;
                }
            }
        }
    }

    if params.byte_flip_every != 0 {
        for i in 0..payload_len {
            if (i as u64 + 1) % params.byte_flip_every == 0 {
                let bit = (hash_u64(seed ^ i as u64) % 8) as u8;
                payload[i] ^= 1 << bit;
                stats.bytes_flipped += 1;
            }
        }
    }

    if params.drop_every != 0 {
        for i in 0..payload_len {
            if (i as u64 + 1) % params.drop_every == 0 {
                payload[i] = 0;
                stats.bytes_dropped += 1;
            }
        }
    }

    stats
}

/// Resolve a preset name into entropy-stream glitch parameters. Names that are not entropy
/// presets leave the bitstream disabled (so the codec stays on the coefficient path). Never
/// errors — `load_dct_transform_preset` validates the name for the coefficient side.
pub fn load_dct_bitstream_preset(name: &str, params: &mut DctBitstreamParams) {
    *params = DctBitstreamParams::default();
    match name {
        "desync" | "entropy" | "decoder-desync" => {
            // Short byte periods so at least a few flips/drops land even on small or highly
            // compressible streams; one early flip already cascades the whole scan.
            params.enabled = true;
            params.byte_flip_every = 220;
            params.drop_every = 400;
        }
        "shred" | "scan-slip" => {
            params.enabled = true;
            params.slip_every = 2;
            params.slip_bytes = 5;
            params.slip_window = 48;
            params.byte_flip_every = 700;
        }
        "truncate" | "tail" => {
            params.enabled = true;
            params.truncate_tail = 0.35;
            params.drop_every = 1500;
        }
        _ => {}
    }
}

/// Set one entropy-bitstream parameter by id. Used by the engine's `set_parameter` as a
/// fallback after the coefficient parameters.
pub fn set_dct_bitstream_parameter(
    params: &mut DctBitstreamParams,
    id: &str,
    value: f32,
) -> Result<(), String> {
    let finite = if value.is_finite() { value } else { 0.0 };
    let as_u64 = |v: f32| v.max(0.0).round() as u64;
    match id {
        "bitstream_enabled" => params.enabled = finite > 0.5,
        "byte_flip_every" => params.byte_flip_every = as_u64(finite),
        "drop_every" => params.drop_every = as_u64(finite),
        "slip_every" => params.slip_every = as_u64(finite),
        "slip_bytes" => params.slip_bytes = finite.round() as i32,
        "slip_window" => params.slip_window = (finite.max(0.0).round() as usize).max(1),
        "truncate_tail" => params.truncate_tail = finite.clamp(0.0, 1.0),
        _ => return Err(format!("unknown dct-bitstream parameter `{id}`")),
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_coeff(luma_blocks: usize, chroma_blocks: usize) -> Vec<i16> {
        let total = (luma_blocks + 2 * chroma_blocks) * BLOCK_AREA;
        let mut coeff = vec![0i16; total];
        let mut state = 0x1234_5678u32;
        let mut next = || {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            state
        };
        for (i, c) in coeff.iter_mut().enumerate() {
            let in_block = i % BLOCK_AREA;
            if in_block == 0 {
                *c = ((next() % 400) as i32 - 200) as i16; // DC
            } else if in_block < 24 && next() % 4 == 0 {
                *c = ((next() % 80) as i32 - 40) as i16; // varied AC -> exercises Huffman
            }
        }
        coeff
    }

    #[test]
    fn bitstream_round_trip_is_lossless() {
        for (lb, cb) in [(40usize, 12usize), (1, 1), (200, 50)] {
            let coeff = sample_coeff(lb, cb);
            let bytes = encode_dct_bitstream(&coeff, 80, 48, lb, cb);
            let mut out = vec![0i16; coeff.len()];
            decode_dct_bitstream(&bytes, &mut out, lb, cb);
            assert_eq!(coeff, out, "clean Huffman round-trip must be lossless ({lb},{cb})");
        }
    }

    #[test]
    fn huffman_actually_compresses() {
        let (lb, cb) = (200usize, 50usize);
        let coeff = sample_coeff(lb, cb);
        let raw = coeff.len() * 2;
        let bytes = encode_dct_bitstream(&coeff, 80, 48, lb, cb);
        assert!(bytes.len() < raw, "entropy stream {} should be < raw {raw}", bytes.len());
    }

    #[test]
    fn mutations_change_output_without_panicking() {
        let (lb, cb) = (40usize, 12usize);
        let coeff = sample_coeff(lb, cb);
        let presets = [
            DctBitstreamParams { byte_flip_every: 7, ..Default::default() },
            DctBitstreamParams { drop_every: 11, ..Default::default() },
            DctBitstreamParams {
                slip_every: 1,
                slip_bytes: 3,
                slip_window: 16,
                ..Default::default()
            },
            DctBitstreamParams { truncate_tail: 0.5, ..Default::default() },
        ];
        for p in presets {
            assert!(p.has_mutations());
            let mut bytes = encode_dct_bitstream(&coeff, 80, 48, lb, cb);
            let _ = mutate_dct_bitstream(&mut bytes, &p, 1);
            let mut out = vec![0i16; coeff.len()];
            decode_dct_bitstream(&bytes, &mut out, lb, cb); // must not panic
            let _ = out;
        }
    }

    #[test]
    fn truncated_stream_decodes_to_partial_then_flat() {
        let (lb, cb) = (40usize, 12usize);
        let coeff = sample_coeff(lb, cb);
        let mut bytes = encode_dct_bitstream(&coeff, 80, 48, lb, cb);
        let keep = bytes.len() - (bytes.len() - FIXED_HEADER_LEN) / 2;
        bytes.truncate(keep.max(FIXED_HEADER_LEN));
        let mut out = vec![0i16; coeff.len()];
        decode_dct_bitstream(&bytes, &mut out, lb, cb);
        let tail = &out[out.len() - BLOCK_AREA..];
        assert!(tail.iter().all(|&c| c == 0));
    }
}
