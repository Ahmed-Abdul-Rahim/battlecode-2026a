//! Integer-only MLP inference (P-052): int8 weights and activations, int32
//! accumulators, ReLU, argmax. No floats, no exp, no allocation per call.
//!
//! Measured cost ≈ 3 fuel per multiply-accumulate (fuel bench: a 64×32 layer
//! costs ~5.9k), so a 2–5k-parameter student fits the 15k Brain stage budget.
//!
//! Blob format `BCW1` (written by ml/distill/quant.py, mirrored by [`Mlp::encode`]):
//!
//! ```text
//! 0..4   magic "BCW1"
//! 4..6   feature schema hash (u16 LE) — must match features_gen::SCHEMA_HASH
//! 6      layer count
//! 7      reserved
//! per layer: n_in u16, n_out u16, shift u8, flags u8 (bit0 = relu), film_offset u16 (0xFFFF = none)
//!            weights i8[n_out][n_in], bias i32[n_out] (LE)
//! last 2 bytes: CRC-16/CCITT-FALSE of everything before
//! ```
//!
//! Hidden layer: `y = clamp(film((acc >> shift)), -127, 127)` then ReLU, where
//! `film(v) = ((v * (64 + γ)) >> 6) + β` with (γ, β) from θ_match. The final
//! layer returns raw i32 accumulators (logits).

use bc_core::crc::crc16;

pub const MAGIC: [u8; 4] = *b"BCW1";
pub const MAX_WIDTH: usize = 128;
const NO_FILM: u16 = 0xFFFF;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WeightsError {
    TooShort,
    BadMagic,
    BadChecksum,
    TooWide,
    ShapeMismatch,
    SchemaMismatch { expected: u16, found: u16 },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Layer {
    pub n_in: usize,
    pub n_out: usize,
    pub shift: u8,
    pub relu: bool,
    /// Index of this layer's first unit in the θ_match FiLM table.
    pub film_offset: Option<usize>,
    pub weights: Vec<i8>,
    pub bias: Vec<i32>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Mlp {
    pub schema_hash: u16,
    pub layers: Vec<Layer>,
}

struct Reader<'a> {
    b: &'a [u8],
    at: usize,
}

impl Reader<'_> {
    fn take(&mut self, n: usize) -> Result<&[u8], WeightsError> {
        let end = self.at.checked_add(n).ok_or(WeightsError::TooShort)?;
        let s = self.b.get(self.at..end).ok_or(WeightsError::TooShort)?;
        self.at = end;
        Ok(s)
    }
    fn u8(&mut self) -> Result<u8, WeightsError> {
        Ok(self.take(1)?[0])
    }
    fn u16(&mut self) -> Result<u16, WeightsError> {
        let s = self.take(2)?;
        Ok(u16::from_le_bytes([s[0], s[1]]))
    }
}

impl Mlp {
    pub fn parse(bytes: &[u8]) -> Result<Self, WeightsError> {
        if bytes.len() < 10 {
            return Err(WeightsError::TooShort);
        }
        let (body, crc) = bytes.split_at(bytes.len() - 2);
        if body[..4] != MAGIC {
            return Err(WeightsError::BadMagic);
        }
        if u16::from_le_bytes([crc[0], crc[1]]) != crc16(body) {
            return Err(WeightsError::BadChecksum);
        }
        let mut r = Reader { b: body, at: 4 };
        let schema_hash = r.u16()?;
        let n_layers = r.u8()? as usize;
        r.u8()?;
        let mut layers = Vec::with_capacity(n_layers);
        for _ in 0..n_layers {
            let n_in = r.u16()? as usize;
            let n_out = r.u16()? as usize;
            let shift = r.u8()?;
            let flags = r.u8()?;
            let film = r.u16()?;
            if n_in > 4 * MAX_WIDTH || n_out > MAX_WIDTH {
                return Err(WeightsError::TooWide);
            }
            let weights = r.take(n_in * n_out)?.iter().map(|&b| b as i8).collect();
            let bias = r
                .take(4 * n_out)?
                .as_chunks::<4>()
                .0
                .iter()
                .map(|c| i32::from_le_bytes(*c))
                .collect();
            layers.push(Layer {
                n_in,
                n_out,
                shift,
                relu: flags & 1 == 1,
                film_offset: (film != NO_FILM).then_some(film as usize),
                weights,
                bias,
            });
        }
        if r.at != body.len() {
            return Err(WeightsError::ShapeMismatch);
        }
        for pair in layers.windows(2) {
            if pair[0].n_out != pair[1].n_in {
                return Err(WeightsError::ShapeMismatch);
            }
        }
        Ok(Self { schema_hash, layers })
    }

    /// Parse and require weights trained against this feature schema.
    pub fn parse_for_schema(bytes: &[u8], expected: u16) -> Result<Self, WeightsError> {
        let mlp = Self::parse(bytes)?;
        if mlp.schema_hash != expected {
            return Err(WeightsError::SchemaMismatch { expected, found: mlp.schema_hash });
        }
        Ok(mlp)
    }

    #[must_use]
    pub fn encode(&self) -> Vec<u8> {
        let mut b = Vec::new();
        b.extend_from_slice(&MAGIC);
        b.extend_from_slice(&self.schema_hash.to_le_bytes());
        b.push(self.layers.len() as u8);
        b.push(0);
        for l in &self.layers {
            b.extend_from_slice(&(l.n_in as u16).to_le_bytes());
            b.extend_from_slice(&(l.n_out as u16).to_le_bytes());
            b.push(l.shift);
            b.push(u8::from(l.relu));
            b.extend_from_slice(&l.film_offset.map_or(NO_FILM, |o| o as u16).to_le_bytes());
            b.extend(l.weights.iter().map(|&w| w as u8));
            for x in &l.bias {
                b.extend_from_slice(&x.to_le_bytes());
            }
        }
        let crc = crc16(&b);
        b.extend_from_slice(&crc.to_le_bytes());
        b
    }

    #[must_use]
    pub fn input_len(&self) -> usize {
        self.layers.first().map_or(0, |l| l.n_in)
    }

    #[must_use]
    pub fn output_len(&self) -> usize {
        self.layers.last().map_or(0, |l| l.n_out)
    }

    #[must_use]
    pub fn param_count(&self) -> usize {
        self.layers.iter().map(|l| l.weights.len() + l.bias.len()).sum()
    }

    /// Run the network; writes logits into `out` and returns how many.
    /// Inputs shorter than the first layer are zero-padded; longer are truncated.
    pub fn forward(&self, input: &[i8], film: Option<&[(i8, i8)]>, out: &mut [i32]) -> usize {
        let mut a = [0i8; 4 * MAX_WIDTH];
        let mut b = [0i8; 4 * MAX_WIDTH];
        let n0 = self.input_len().min(a.len());
        let copy = input.len().min(n0);
        a[..copy].copy_from_slice(&input[..copy]);
        let last = self.layers.len().saturating_sub(1);
        for (li, layer) in self.layers.iter().enumerate() {
            let rows = layer.weights.chunks(layer.n_in.max(1));
            for (o, ((row, &bias), slot)) in rows.zip(&layer.bias).zip(b.iter_mut()).enumerate() {
                let acc = bias + dot(row, &a[..layer.n_in]);
                if li == last {
                    if let Some(logit) = out.get_mut(o) {
                        *logit = acc;
                    }
                    continue;
                }
                let mut v = acc >> layer.shift;
                if let (Some(table), Some(base)) = (film, layer.film_offset) {
                    if let Some(&(gamma, beta)) = table.get(base + o) {
                        v = ((v * (64 + i32::from(gamma))) >> 6) + i32::from(beta);
                    }
                }
                if layer.relu {
                    v = v.max(0);
                }
                *slot = v.clamp(-127, 127) as i8;
            }
            if li != last {
                core::mem::swap(&mut a, &mut b);
            }
        }
        self.output_len().min(out.len())
    }

    /// Index of the largest logit (ties → lowest index).
    #[must_use]
    pub fn argmax(&self, input: &[i8], film: Option<&[(i8, i8)]>) -> Option<usize> {
        let mut logits = [0i32; MAX_WIDTH];
        let n = self.forward(input, film, &mut logits);
        (0..n).max_by_key(|&i| (logits[i], core::cmp::Reverse(i)))
    }
}

/// Σ w·x over the common length. In the bot (wasm32) a simd128 kernel: measured in game on a
/// 64×128 layer, 20.8k fuel instead of 51.1k for the scalar loop, same result.
#[inline]
fn dot(w: &[i8], x: &[i8]) -> i32 {
    #[cfg(target_arch = "wasm32")]
    {
        // SAFETY: simd128 is part of the engine's wasm feature set (cambc's C++ SDK ships
        // wasm_simd128.h; checked in game with bots/_probe).
        #[allow(unsafe_code)]
        unsafe {
            simd::dot(w, x)
        }
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        dot_scalar(w, x)
    }
}

#[cfg_attr(target_arch = "wasm32", allow(dead_code))]
fn dot_scalar(w: &[i8], x: &[i8]) -> i32 {
    w.iter().zip(x).map(|(a, b)| i32::from(*a) * i32::from(*b)).sum()
}

#[cfg(target_arch = "wasm32")]
#[allow(unsafe_code)]
mod simd {
    use core::arch::wasm32::{
        i16x8_extend_high_i8x16, i16x8_extend_low_i8x16, i32x4_add, i32x4_dot_i16x8,
        i32x4_extract_lane, i32x4_splat, v128_load,
    };

    #[target_feature(enable = "simd128")]
    pub(super) unsafe fn dot(w: &[i8], x: &[i8]) -> i32 {
        let n = w.len().min(x.len());
        let mut acc = i32x4_splat(0);
        let mut i = 0;
        while i + 16 <= n {
            // SAFETY: i + 16 <= n bounds both slices; v128_load takes unaligned addresses.
            let (a, b) = unsafe {
                (v128_load(w.as_ptr().add(i).cast()), v128_load(x.as_ptr().add(i).cast()))
            };
            let lo = i32x4_dot_i16x8(i16x8_extend_low_i8x16(a), i16x8_extend_low_i8x16(b));
            let hi = i32x4_dot_i16x8(i16x8_extend_high_i8x16(a), i16x8_extend_high_i8x16(b));
            acc = i32x4_add(acc, i32x4_add(lo, hi));
            i += 16;
        }
        let mut s = i32x4_extract_lane::<0>(acc)
            + i32x4_extract_lane::<1>(acc)
            + i32x4_extract_lane::<2>(acc)
            + i32x4_extract_lane::<3>(acc);
        while i < n {
            s += i32::from(w[i]) * i32::from(x[i]);
            i += 1;
        }
        s
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny() -> Mlp {
        // 3 inputs → 2 hidden (relu, shift 1) → 2 logits.
        Mlp {
            schema_hash: 0xBDF9,
            layers: vec![
                Layer {
                    n_in: 3,
                    n_out: 2,
                    shift: 1,
                    relu: true,
                    film_offset: Some(0),
                    weights: vec![1, 2, 3, -1, -1, -1],
                    bias: vec![0, 10],
                },
                Layer {
                    n_in: 2,
                    n_out: 2,
                    shift: 0,
                    relu: false,
                    film_offset: None,
                    weights: vec![1, 0, 0, 1],
                    bias: vec![0, 5],
                },
            ],
        }
    }

    #[test]
    fn forward_matches_hand_computation() {
        let m = tiny();
        let mut out = [0i32; 4];
        // h0 = (1*10 + 2*20 + 3*30) >> 1 = 70; h1 = relu((10 - 60) >> 1) = 0.
        assert_eq!(m.forward(&[10, 20, 30], None, &mut out), 2);
        assert_eq!(&out[..2], &[70, 5]);
        assert_eq!(m.argmax(&[10, 20, 30], None), Some(0));
    }

    #[test]
    fn film_scales_and_shifts_hidden_units() {
        let m = tiny();
        let mut out = [0i32; 2];
        // γ = 64 doubles h0 (70 → 140, clamped to 127). FiLM runs before ReLU:
        // β = 30 lifts h1 from −25 to 5, which survives ReLU.
        m.forward(&[10, 20, 30], Some(&[(64, 0), (0, 30)]), &mut out);
        assert_eq!(out, [127, 10]);
    }

    #[test]
    fn blob_round_trips_and_rejects_corruption() {
        let m = tiny();
        let blob = m.encode();
        assert_eq!(Mlp::parse(&blob).unwrap(), m);
        assert_eq!(m.param_count(), 6 + 2 + 4 + 2);
        let mut bad = blob.clone();
        bad[12] ^= 0x40;
        assert_eq!(Mlp::parse(&bad), Err(WeightsError::BadChecksum));
        assert_eq!(
            Mlp::parse_for_schema(&blob, 0x1111),
            Err(WeightsError::SchemaMismatch { expected: 0x1111, found: 0xBDF9 })
        );
        assert_eq!(Mlp::parse(b"nope"), Err(WeightsError::TooShort));
    }

    #[test]
    fn matches_python_reference_fixture() {
        let fixture: serde_json::Value =
            serde_json::from_str(include_str!("../../../tests/fixtures/int8_reference.json"))
                .unwrap();
        let blob: Vec<u8> = (0..fixture["blob_hex"].as_str().unwrap().len() / 2)
            .map(|i| {
                u8::from_str_radix(&fixture["blob_hex"].as_str().unwrap()[2 * i..2 * i + 2], 16)
                    .unwrap()
            })
            .collect();
        let m = Mlp::parse(&blob).unwrap();
        for case in fixture["cases"].as_array().unwrap() {
            let input: Vec<i8> = case["input"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_i64().unwrap() as i8)
                .collect();
            let film: Vec<(i8, i8)> = case["film"]
                .as_array()
                .unwrap()
                .iter()
                .map(|p| (p[0].as_i64().unwrap() as i8, p[1].as_i64().unwrap() as i8))
                .collect();
            let expected: Vec<i32> = case["logits"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_i64().unwrap() as i32)
                .collect();
            let mut out = vec![0i32; expected.len()];
            m.forward(&input, Some(&film), &mut out);
            assert_eq!(out, expected);
        }
    }
}
