//! The teacher network (nn/ pipeline): a plain f32 MLP with several masked softmax heads.
//!
//! Blob format `BCN1` (little endian), written by `nn/model.py`:
//!
//! ```text
//! "BCN1" | u32 spec hash | u32 layer count
//! per layer: u32 n_in | u32 n_out | u32 relu (0/1) | f32 W[n_out][n_in] | f32 b[n_out]
//! u32 head count | u32 head sizes...
//! ```
//!
//! The last layer's outputs are the heads' logits, back to back. It runs only where fuel
//! allows (local runs with `--fuel`); the shipped bot uses a distilled int8 student.

/// One dense layer.
#[derive(Clone, Debug, PartialEq)]
pub struct Dense {
    pub n_in: usize,
    pub n_out: usize,
    pub relu: bool,
    pub w: Vec<f32>,
    pub b: Vec<f32>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Net {
    pub spec: u32,
    pub layers: Vec<Dense>,
    pub heads: Vec<usize>,
}

struct Reader<'a> {
    b: &'a [u8],
    at: usize,
}

impl Reader<'_> {
    fn u32(&mut self) -> Option<u32> {
        let s = self.b.get(self.at..self.at + 4)?;
        self.at += 4;
        Some(u32::from_le_bytes([s[0], s[1], s[2], s[3]]))
    }

    fn f32s(&mut self, n: usize) -> Option<Vec<f32>> {
        let s = self.b.get(self.at..self.at + 4 * n)?;
        self.at += 4 * n;
        Some(s.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect())
    }
}

impl Net {
    /// Parse a `BCN1` blob; None if it is empty, malformed or its layers don't chain.
    #[must_use]
    pub fn parse(blob: &[u8]) -> Option<Self> {
        if blob.get(..4)? != b"BCN1" {
            return None;
        }
        let mut r = Reader { b: blob, at: 4 };
        let spec = r.u32()?;
        let n = r.u32()? as usize;
        if n == 0 || n > 16 {
            return None;
        }
        let mut layers = Vec::with_capacity(n);
        for _ in 0..n {
            let (n_in, n_out, relu) = (r.u32()? as usize, r.u32()? as usize, r.u32()? != 0);
            let w = r.f32s(n_in.checked_mul(n_out)?)?;
            let b = r.f32s(n_out)?;
            layers.push(Dense { n_in, n_out, relu, w, b });
        }
        if layers.windows(2).any(|p| p[0].n_out != p[1].n_in) {
            return None;
        }
        let k = r.u32()? as usize;
        let heads: Vec<usize> = (0..k).map(|_| r.u32().map(|v| v as usize)).collect::<Option<_>>()?;
        (heads.iter().sum::<usize>() == layers.last()?.n_out && r.at == blob.len())
            .then_some(Self { spec, layers, heads })
    }

    #[must_use]
    pub fn input_len(&self) -> usize {
        self.layers.first().map_or(0, |l| l.n_in)
    }

    /// All head logits for input `x` (shorter inputs are zero-padded, longer ones cut).
    #[must_use]
    pub fn forward(&self, x: &[f32]) -> Vec<f32> {
        let mut cur: Vec<f32> = (0..self.input_len()).map(|i| x.get(i).copied().unwrap_or(0.0)).collect();
        for l in &self.layers {
            let mut out = l.b.clone();
            for (o, row) in out.iter_mut().zip(l.w.chunks_exact(l.n_in)) {
                let mut acc = 0.0f32;
                for (w, v) in row.iter().zip(&cur) {
                    acc += w * v;
                }
                *o += acc;
                if l.relu && *o < 0.0 {
                    *o = 0.0;
                }
            }
            cur = out;
        }
        cur
    }

    /// The range of head `h` in the output.
    #[must_use]
    pub fn head(&self, h: usize) -> Option<core::ops::Range<usize>> {
        let start: usize = self.heads.get(..h)?.iter().sum();
        Some(start..start + *self.heads.get(h)?)
    }
}

/// Masked softmax pick: `u` = None takes the argmax, otherwise samples with uniform `u`.
/// Returns (index, log-probability), or None when the mask allows nothing.
#[must_use]
pub fn pick(logits: &[f32], mask: &[bool], u: Option<f32>) -> Option<(usize, f32)> {
    let allowed = || logits.iter().zip(mask).enumerate().filter(|(_, (_, &m))| m).map(|(i, (&l, _))| (i, l));
    let max = allowed().map(|(_, l)| l).fold(f32::NEG_INFINITY, f32::max);
    if !max.is_finite() {
        return None;
    }
    let lse = max + allowed().map(|(_, l)| (l - max).exp()).sum::<f32>().ln();
    let chosen = match u {
        None => allowed().fold((usize::MAX, f32::NEG_INFINITY), |b, x| if x.1 > b.1 { x } else { b }),
        Some(u) => {
            let mut acc = 0.0;
            let mut last = None;
            let mut hit = None;
            for (i, l) in allowed() {
                acc += (l - lse).exp();
                last = Some((i, l));
                if u < acc {
                    hit = Some((i, l));
                    break;
                }
            }
            hit.or(last)?
        }
    };
    Some((chosen.0, chosen.1 - lse))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn blob(layers: &[(usize, usize, bool, Vec<f32>, Vec<f32>)], heads: &[u32]) -> Vec<u8> {
        let mut b = b"BCN1".to_vec();
        b.extend(7u32.to_le_bytes());
        b.extend((layers.len() as u32).to_le_bytes());
        for (i, o, relu, w, bias) in layers {
            b.extend((*i as u32).to_le_bytes());
            b.extend((*o as u32).to_le_bytes());
            b.extend(u32::from(*relu).to_le_bytes());
            for v in w.iter().chain(bias) {
                b.extend(v.to_le_bytes());
            }
        }
        b.extend((heads.len() as u32).to_le_bytes());
        for h in heads {
            b.extend(h.to_le_bytes());
        }
        b
    }

    #[test]
    fn parses_and_runs_a_two_layer_net() {
        // x -> relu(Wx + b) -> V h + c ; heads [2, 1]
        let b = blob(
            &[
                (2, 2, true, vec![1.0, 0.0, 0.0, -1.0], vec![0.0, 0.5]),
                (2, 3, false, vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0], vec![0.0, 0.0, 1.0]),
            ],
            &[2, 1],
        );
        let net = Net::parse(&b).unwrap();
        assert_eq!(net.spec, 7);
        assert_eq!(net.head(1), Some(2..3));
        // h = relu([2, 0.5 - 3]) = [2, 0]; out = [2, 0, 3]
        assert_eq!(net.forward(&[2.0, 3.0]), vec![2.0, 0.0, 3.0]);
        assert!(Net::parse(&b[..b.len() - 1]).is_none());
        assert!(Net::parse(&[]).is_none());
    }

    #[test]
    fn pick_respects_mask_and_probabilities_sum_to_one() {
        let l = [1.0, 0.0, 5.0];
        let m = [true, true, false];
        assert_eq!(pick(&l, &m, None).map(|p| p.0), Some(0));
        let total: f32 = [0.0, 0.99].iter().map(|&u| pick(&l, &m, Some(u)).unwrap().1.exp()).sum();
        assert!((total - 1.0).abs() < 1e-4, "{total}");
        assert!(pick(&l, &[false; 3], None).is_none());
    }
}
