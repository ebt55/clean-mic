//! Echo return loss enhancement, label-free, as Diarist's post-pass
//! computes it (`audio/aec/offline.rs`, MIT, see `NOTICE.md`), made
//! streaming; plus the echo lag by GCC-PHAT for diagnostics.
//!
//! 100 Hz high-passed energies in 2 s blocks of the aligned 16 kHz mic,
//! reference, canceller output and final output. A block is far-end-only
//! when the reference is active (at or above -45 dBFS) and the mic stands at
//! most 2 dB above the level the median return loss of the active blocks
//! predicts (more is the owner talking). ERLE is the summed mic energy over
//! the summed output energy of those blocks.

use std::sync::Arc;

use realfft::num_complex::Complex32;
use realfft::{ComplexToReal, RealFftPlanner, RealToComplex};

use crate::stats::median;

const SR: f64 = 16_000.0;
/// Block length (2 s).
pub const ERLE_BLOCK: usize = 32_000;
/// A block is active at this reference level (dBFS).
const ACTIVE_DBFS: f64 = -45.0;
/// Mic over the predicted echo level beyond which a block is double talk.
const DOUBLE_TALK_DB: f64 = 2.0;
/// Room-floor frames (100 ms).
const FLOOR_FRAME: usize = 1_600;

/// 4th-order Butterworth high-pass at 100 Hz (two biquads), streaming;
/// Diarist's `offline::HighPass`.
#[derive(Debug, Clone, Default)]
struct HighPass {
    z: [[f64; 2]; 2],
}

impl HighPass {
    fn coefs() -> [[f64; 5]; 2] {
        let mut c = [[0.0; 5]; 2];
        for (k, q) in [0.541_196_1f64, 1.306_563].into_iter().enumerate() {
            let w0 = std::f64::consts::TAU * 100.0 / SR;
            let alpha = w0.sin() / (2.0 * q);
            let cw = w0.cos();
            let a0 = 1.0 + alpha;
            c[k] = [
                (1.0 + cw) / 2.0 / a0,
                -(1.0 + cw) / a0,
                (1.0 + cw) / 2.0 / a0,
                -2.0 * cw / a0,
                (1.0 - alpha) / a0,
            ];
        }
        c
    }

    /// Energy of the high-passed `x` (the filter state carries over).
    fn energy(&mut self, x: &[f32]) -> f64 {
        let c = Self::coefs();
        let mut e = 0.0;
        for &v in x {
            let mut y = f64::from(v);
            for (cc, z) in c.iter().zip(self.z.iter_mut()) {
                let out = cc[0] * y + z[0];
                z[0] = cc[1] * y - cc[3] * out + z[1];
                z[1] = cc[2] * y - cc[4] * out;
                y = out;
            }
            e += y * y;
        }
        e
    }
}

/// One 2 s block's energies.
#[derive(Debug, Clone, Copy, Default, serde::Serialize)]
pub struct BlockEnergy {
    pub mic: f64,
    pub reference: f64,
    pub aec: f64,
    pub out: f64,
    /// Un-filtered reference energy and samples, for the activity test.
    pub ref_raw: f64,
    pub n: usize,
}

impl BlockEnergy {
    pub fn ref_dbfs(&self) -> f64 {
        10.0 * (self.ref_raw / self.n.max(1) as f64).max(1e-20).log10()
    }

    pub fn mic_dbfs(&self) -> f64 {
        10.0 * (self.mic / self.n.max(1) as f64).max(1e-20).log10()
    }

    pub fn active(&self) -> bool {
        self.n > 0 && self.ref_dbfs() >= ACTIVE_DBFS && self.reference > 0.0
    }

    /// Mic over reference, dB (the return loss, negated).
    fn rel(&self) -> f64 {
        10.0 * (self.mic.max(1e-20) / self.reference.max(1e-20)).log10()
    }

    pub fn erle_aec_db(&self) -> f64 {
        10.0 * (self.mic.max(1e-20) / self.aec.max(1e-20)).log10()
    }

    pub fn erle_out_db(&self) -> f64 {
        10.0 * (self.mic.max(1e-20) / self.out.max(1e-20)).log10()
    }
}

/// The run's figures.
#[derive(Debug, Clone, Default, serde::Serialize)]
pub struct ErleResult {
    pub blocks: usize,
    pub active_blocks: usize,
    pub far_blocks: usize,
    /// Canceller output (comparable with the A/B's 13.5 dB label-free).
    pub erle_aec_db: Option<f64>,
    /// After RNNoise (comparable with 15.2 dB; includes the room floor).
    pub erle_out_db: Option<f64>,
    /// Median mic over reference of the active blocks, dB.
    pub mic_over_ref_db: Option<f64>,
    /// Echo (mic in the far-end-only blocks) over the room floor (the 5th
    /// percentile of 100 ms mic energies): the most ERLE can show.
    pub echo_over_floor_db: Option<f64>,
}

/// Streaming accumulator over aligned 16 kHz signals.
#[derive(Default)]
pub struct Erle {
    hp: [HighPass; 4],
    cur: BlockEnergy,
    blocks: Vec<BlockEnergy>,
    floor_cur: (f64, usize),
    floor_frames: Vec<f64>,
}

impl Erle {
    /// Adds aligned samples (equal lengths); returns each block it
    /// completes.
    pub fn push(&mut self, mic: &[f32], reference: &[f32], aec: &[f32], out: &[f32]) -> Option<BlockEnergy> {
        let n = mic.len();
        debug_assert!(reference.len() == n && aec.len() == n && out.len() == n);
        self.cur.mic += self.hp[0].energy(mic);
        self.cur.reference += self.hp[1].energy(reference);
        self.cur.aec += self.hp[2].energy(aec);
        self.cur.out += self.hp[3].energy(out);
        self.cur.ref_raw += reference.iter().map(|&v| f64::from(v * v)).sum::<f64>();
        self.cur.n += n;
        self.floor_cur.0 += mic.iter().map(|&v| f64::from(v * v)).sum::<f64>();
        self.floor_cur.1 += n;
        if self.floor_cur.1 >= FLOOR_FRAME {
            self.floor_frames.push(self.floor_cur.0 / self.floor_cur.1 as f64);
            self.floor_cur = (0.0, 0);
        }
        if self.cur.n >= ERLE_BLOCK {
            let b = std::mem::take(&mut self.cur);
            self.blocks.push(b);
            return Some(b);
        }
        None
    }

    /// Whether `b` counts as far-end-only against the blocks so far.
    pub fn is_far_only(&self, b: &BlockEnergy) -> bool {
        b.active() && self.median_rel().is_some_and(|m| b.rel() <= m + DOUBLE_TALK_DB)
    }

    fn median_rel(&self) -> Option<f64> {
        let rel: Vec<f64> = self.blocks.iter().filter(|b| b.active()).map(|b| b.rel()).collect();
        median(&rel)
    }

    pub fn result(&self) -> ErleResult {
        let mut r = ErleResult {
            blocks: self.blocks.len(),
            ..ErleResult::default()
        };
        let active: Vec<&BlockEnergy> = self.blocks.iter().filter(|b| b.active()).collect();
        r.active_blocks = active.len();
        r.mic_over_ref_db = self.median_rel();
        let Some(med) = r.mic_over_ref_db else {
            return r;
        };
        let far: Vec<&&BlockEnergy> = active
            .iter()
            .filter(|b| b.rel() <= med + DOUBLE_TALK_DB)
            .collect();
        r.far_blocks = far.len();
        let (em, ea, eo, n) = far.iter().fold((0f64, 0f64, 0f64, 0usize), |a, b| {
            (a.0 + b.mic, a.1 + b.aec, a.2 + b.out, a.3 + b.n)
        });
        if em > 0.0 && ea > 0.0 {
            r.erle_aec_db = Some(10.0 * (em / ea).log10());
        }
        if em > 0.0 && eo > 0.0 {
            r.erle_out_db = Some(10.0 * (em / eo).log10());
        }
        let mut f = self.floor_frames.clone();
        f.sort_by(f64::total_cmp);
        if !f.is_empty() && n > 0 {
            let floor = f[(f.len() - 1) / 20];
            if floor > 0.0 {
                r.echo_over_floor_db = Some(10.0 * ((em / n as f64) / floor).log10());
            }
        }
        r
    }

    pub fn blocks(&self) -> &[BlockEnergy] {
        &self.blocks
    }
}

/// The echo lag by GCC-PHAT over 200 Hz to 6 kHz, -20 to +200 ms.
pub struct LagEstimator {
    n: usize,
    fwd: Arc<dyn RealToComplex<f32>>,
    inv: Arc<dyn ComplexToReal<f32>>,
    ta: Vec<f32>,
    sa: Vec<Complex32>,
    sb: Vec<Complex32>,
}

impl Default for LagEstimator {
    fn default() -> Self {
        let n = 2 * ERLE_BLOCK.next_power_of_two();
        let mut planner = RealFftPlanner::<f32>::new();
        let fwd = planner.plan_fft_forward(n);
        let inv = planner.plan_fft_inverse(n);
        Self {
            ta: vec![0.0; n],
            sa: fwd.make_output_vec(),
            sb: fwd.make_output_vec(),
            n,
            fwd,
            inv,
        }
    }
}

impl LagEstimator {
    /// (lag ms, mic after reference; peak height 0..1) of a block.
    pub fn estimate(&mut self, mic: &[f32], reference: &[f32]) -> Option<(f64, f64)> {
        let len = mic.len().min(reference.len()).min(self.n / 2);
        if len < 1_600 {
            return None;
        }
        self.ta.fill(0.0);
        self.ta[..len].copy_from_slice(&reference[..len]);
        self.fwd.process(&mut self.ta, &mut self.sa).ok()?;
        self.ta.fill(0.0);
        self.ta[..len].copy_from_slice(&mic[..len]);
        self.fwd.process(&mut self.ta, &mut self.sb).ok()?;
        let lo = (200.0 * self.n as f64 / SR) as usize;
        let hi = (6_000.0 * self.n as f64 / SR) as usize;
        for (k, (b, a)) in self.sb.iter_mut().zip(&self.sa).enumerate() {
            let c = *b * a.conj();
            let m = c.norm();
            *b = if (lo..=hi).contains(&k) && m > 1e-20 {
                c / m
            } else {
                Complex32::default()
            };
        }
        self.inv.process(&mut self.sb, &mut self.ta).ok()?;
        let bins = (hi - lo + 1) as f64;
        let (mut best, mut at) = (f32::MIN, 0i64);
        for lag in -320i64..=3_200 {
            let idx = if lag >= 0 { lag as usize } else { self.n - (-lag) as usize };
            let v = self.ta[idx];
            if v > best {
                best = v;
                at = lag;
            }
        }
        // A perfect delay puts every in-band bin in phase: the peak is the
        // band's bin count times two (the real transform's two halves).
        Some((at as f64 * 1_000.0 / SR, f64::from(best) / (2.0 * bins)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn noise(n: usize, seed: &mut u32, amp: f32) -> Vec<f32> {
        (0..n)
            .map(|_| {
                *seed ^= *seed << 13;
                *seed ^= *seed >> 17;
                *seed ^= *seed << 5;
                (*seed as f32 / u32::MAX as f32 - 0.5) * 2.0 * amp
            })
            .collect()
    }

    #[test]
    fn far_end_blocks_and_erle() {
        let mut e = Erle::default();
        let mut seed = 1;
        // 10 blocks of far end only: echo 10 dB under the reference, the
        // output 20 dB under the mic; then 3 blocks with the owner talking.
        for k in 0..13 {
            let reference = noise(ERLE_BLOCK, &mut seed, 0.1);
            let echo: Vec<f32> = reference.iter().map(|v| v * 0.316).collect();
            let talk = if k >= 10 { noise(ERLE_BLOCK, &mut seed, 0.2) } else { vec![0.0; ERLE_BLOCK] };
            let mic: Vec<f32> = echo.iter().zip(&talk).map(|(a, b)| a + b).collect();
            let aec: Vec<f32> = echo.iter().zip(&talk).map(|(a, b)| a * 0.1 + b).collect();
            for i in (0..ERLE_BLOCK).step_by(160) {
                e.push(&mic[i..i + 160], &reference[i..i + 160], &aec[i..i + 160], &aec[i..i + 160]);
            }
        }
        let r = e.result();
        assert_eq!(r.active_blocks, 13);
        assert_eq!(r.far_blocks, 10);
        let erle = r.erle_aec_db.unwrap();
        assert!((erle - 20.0).abs() < 0.2, "{erle}");
        assert!((r.mic_over_ref_db.unwrap() + 10.0).abs() < 0.5);
    }

    #[test]
    fn lag_of_a_delayed_copy() {
        let mut seed = 3;
        let reference = noise(ERLE_BLOCK, &mut seed, 0.1);
        let mut mic = vec![0.0f32; ERLE_BLOCK];
        for i in 48..ERLE_BLOCK {
            mic[i] = 0.3 * reference[i - 48];
        }
        let (lag, peak) = LagEstimator::default().estimate(&mic, &reference).unwrap();
        assert!((lag - 3.0).abs() < 1e-9, "{lag}");
        assert!(peak > 0.5, "{peak}");
    }
}
