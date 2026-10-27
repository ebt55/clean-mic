//! Rate conversion between the 48 kHz device streams and the canceller's
//! 16 kHz: [`Down3`] (48 to 16 kHz, both streams) and [`Interp`] (16 to
//! 48 kHz at an adjustable ratio, the output side).
//!
//! The filters are the offline A/B's (`docs/design.md` section 2): a
//! Kaiser-windowed sinc low-pass with 191 taps at 48 kHz for decimation,
//! and Diarist's 32-tap [`SincInterpolator`] for interpolation. Here they
//! run streaming and causally; the decimator computes only every third
//! output, which the A/B's version did not.

use crate::aec::SincInterpolator;

/// Input samples (48 kHz) by which [`Down3`]'s output lags its input: output
/// `n` is centred on input `3 n - DOWN3_DELAY`.
pub const DOWN3_DELAY: usize = 95;
/// Input samples (16 kHz) by which [`Interp`]'s output lags its input.
pub const INTERP_DELAY: usize = 16;

const DOWN3_HALF: usize = 95;
/// Kaiser window shape (the A/B's filters).
const KAISER_BETA: f64 = 8.0;

/// 48 to 16 kHz: low-pass at 0.95 of the 8 kHz output Nyquist (7.6 kHz),
/// evaluated at every third input sample. Takes any number of input
/// samples per call; output `n` needs input `3 n`.
pub struct Down3 {
    taps: Vec<f32>,
    buf: Vec<f32>,
    /// Position in `buf` of the first tap of the next output.
    next: usize,
}

impl Default for Down3 {
    fn default() -> Self {
        Self::new()
    }
}

impl Down3 {
    pub fn new() -> Self {
        let n = 2 * DOWN3_HALF + 1;
        let cutoff = 0.95 / 3.0;
        let i0b = bessel_i0(KAISER_BETA);
        let mut taps: Vec<f32> = (0..n)
            .map(|t| {
                let x = t as f64 - DOWN3_HALF as f64;
                let sinc = if x == 0.0 {
                    cutoff
                } else {
                    (std::f64::consts::PI * cutoff * x).sin() / (std::f64::consts::PI * x)
                };
                let r = x / (DOWN3_HALF + 1) as f64;
                (sinc * bessel_i0(KAISER_BETA * (1.0 - r * r).sqrt()) / i0b) as f32
            })
            .collect();
        let sum: f32 = taps.iter().sum();
        for t in &mut taps {
            *t /= sum;
        }
        Self {
            // History of zeros: output 0 is centred on input 0 - DOWN3_DELAY.
            buf: vec![0.0; n - 1],
            next: 0,
            taps,
        }
    }

    /// Appends `input` and pushes every output it completes onto `out`.
    pub fn process(&mut self, input: &[f32], out: &mut Vec<f32>) {
        self.buf.extend_from_slice(input);
        let n = self.taps.len();
        while self.next + n <= self.buf.len() {
            let w = &self.buf[self.next..self.next + n];
            out.push(dot(w, &self.taps));
            self.next += 3;
        }
        if self.next > 8_192 {
            self.buf.drain(..self.next);
            self.next = 0;
        }
    }
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    // Four partial sums: lets the compiler vectorise without -ffast-math.
    let mut acc = [0f32; 4];
    let (ca, ra) = a.as_chunks::<4>();
    let (cb, rb) = b.as_chunks::<4>();
    for (x, y) in ca.iter().zip(cb) {
        for k in 0..4 {
            acc[k] += x[k] * y[k];
        }
    }
    let mut s = acc[0] + acc[1] + acc[2] + acc[3];
    for (x, y) in ra.iter().zip(rb) {
        s += x * y;
    }
    s
}

/// 16 to 48 kHz by band-limited interpolation at fractional positions, one
/// output every `step` input samples (`1/3` nominal; stage C steers it for
/// the output's clock). The input is pre-padded with [`INTERP_DELAY`]
/// zeros, so at `step = 1/3` every 160 input samples give exactly 480
/// outputs.
pub struct Interp {
    sinc: SincInterpolator,
    buf: Vec<f32>,
    /// The next output sits at `base + k * step` in `buf`: one product per
    /// position rather than a running sum, so whole positions stay whole.
    base: f64,
    k: u64,
    step: f64,
}

/// Slack for rounding when deciding whether a position is whole.
const POS_EPS: f64 = 1e-6;

impl Default for Interp {
    fn default() -> Self {
        Self::new()
    }
}

impl Interp {
    pub fn new() -> Self {
        Self {
            sinc: SincInterpolator::default(),
            buf: vec![0.0; INTERP_DELAY],
            base: 0.0,
            k: 0,
            step: 1.0 / 3.0,
        }
    }

    fn pos(&self) -> f64 {
        self.base + self.k as f64 * self.step
    }

    /// Input samples per output sample.
    pub fn set_step(&mut self, step: f64) {
        if step != self.step {
            self.base = self.pos();
            self.k = 0;
            self.step = step;
        }
    }

    pub fn step(&self) -> f64 {
        self.step
    }

    /// Appends `input` and pushes every output it completes onto `out`.
    pub fn process(&mut self, input: &[f32], out: &mut Vec<f32>) {
        self.buf.extend_from_slice(input);
        loop {
            let pos = self.pos();
            // The interpolator reads 16 samples past floor(pos).
            if ((pos + POS_EPS).floor() as usize) + INTERP_DELAY >= self.buf.len() {
                break;
            }
            out.push(self.sinc.at(&self.buf, pos));
            self.k += 1;
        }
        let keep = (self.pos().floor() as usize).saturating_sub(INTERP_DELAY + 1);
        if keep > 4_096 {
            self.buf.drain(..keep);
            self.base -= keep as f64;
            // Keep the product small over days of running (one rounding).
            if self.k > 1 << 30 {
                self.base = self.pos();
                self.k = 0;
            }
        }
    }
}

fn bessel_i0(x: f64) -> f64 {
    let mut sum = 1.0;
    let mut term = 1.0;
    let q = x * x / 4.0;
    for k in 1..60 {
        term *= q / (k as f64 * k as f64);
        sum += term;
        if term < sum * 1e-16 {
            break;
        }
    }
    sum
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tone(rate: f64, hz: f64, n: usize) -> Vec<f32> {
        (0..n)
            .map(|i| (std::f64::consts::TAU * hz * i as f64 / rate).sin() as f32)
            .collect()
    }

    fn rms(x: &[f32]) -> f64 {
        (x.iter().map(|&v| f64::from(v) * f64::from(v)).sum::<f64>() / x.len() as f64).sqrt()
    }

    #[test]
    fn down3_passes_speech_band_and_stops_above_8k() {
        for (hz, lo, hi) in [(1_000.0, 0.99, 1.01), (6_000.0, 0.98, 1.02), (12_000.0, 0.0, 0.003)] {
            let x = tone(48_000.0, hz, 48_000);
            let mut d = Down3::new();
            let mut y = Vec::new();
            // Odd chunk sizes: streaming must not depend on them.
            for c in x.chunks(337) {
                d.process(c, &mut y);
            }
            assert_eq!(y.len(), 16_000);
            let g = rms(&y[1_000..15_000]) / rms(&x[3_000..45_000]);
            assert!((lo..=hi).contains(&g), "{hz} Hz gain {g}");
        }
    }

    #[test]
    fn down3_delay_is_95_input_samples() {
        let mut x = vec![0.0f32; 4_800];
        x[3_000] = 1.0;
        let mut y = Vec::new();
        Down3::new().process(&x, &mut y);
        let peak = y
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.abs().total_cmp(&b.1.abs()))
            .unwrap()
            .0;
        // 3,000 + 95 = 3 * 1,031 + 2: the peak lands on output 1,032 or 1,031.
        assert!((1_031..=1_032).contains(&peak), "{peak}");
    }

    #[test]
    fn interp_gives_480_per_160_and_a_clean_tone() {
        let x = tone(16_000.0, 1_000.0, 16_000);
        let mut it = Interp::new();
        let mut y = Vec::new();
        for c in x.chunks(160) {
            let before = y.len();
            it.process(c, &mut y);
            assert_eq!(y.len() - before, 480);
        }
        // Output k is input k/3 - 16; compare with the ideal 48 kHz tone.
        let ideal: Vec<f32> = (0..y.len())
            .map(|k| {
                let t = (k as f64 / 3.0 - INTERP_DELAY as f64) / 16_000.0;
                (std::f64::consts::TAU * 1_000.0 * t).sin() as f32
            })
            .collect();
        let err: Vec<f32> = y.iter().zip(&ideal).map(|(a, b)| a - b).collect();
        let snr = 20.0 * (rms(&ideal[3_000..45_000]) / rms(&err[3_000..45_000])).log10();
        assert!(snr > 50.0, "SNR {snr:.1} dB");
    }

    #[test]
    fn down_then_up_keeps_the_band() {
        let x = tone(48_000.0, 2_500.0, 48_000);
        let mut d = Down3::new();
        let mut mid = Vec::new();
        d.process(&x, &mut mid);
        let mut it = Interp::new();
        let mut y = Vec::new();
        it.process(&mid, &mut y);
        let g = rms(&y[6_000..42_000]) / rms(&x[6_000..42_000]);
        assert!((0.98..=1.02).contains(&g), "{g}");
    }
}
