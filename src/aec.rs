//! The echo canceller, copied from Diarist's `crates/core/src/audio/aec.rs`
//! (commit 15443dd; MIT, Copyright (c) 2026 Ebin Babu Thomas, see
//! `NOTICE.md`). Unchanged apart from this paragraph and the removed
//! `offline` module; the text below is Diarist's and its references
//! (research 14, plan 15, meeting 12, `tests/aec_spike.rs`) point into the
//! Diarist repository. Clean Mic runs it in Diarist's post-pass
//! configuration with no lookahead (`docs/design.md` section 3.1), on a
//! reference placed causally by `crate::clock`.
//!
//! Diarist: app-side acoustic echo canceller
//! (`docs/research/14-echo-cancellation.md` section 2, where the spike
//! measured it on meeting 12). The post-pass runs it over the stored WAVs
//! before transcribing the mic (`offline`, plan 15 chunk B2a); the live
//! pipeline does not yet (chunk B2b).
//!
//! With the meeting on speakers the microphone hears the far end. The
//! loopback is what the speakers played, so [`Aec`] learns the echo path
//! (speaker, room, microphone) as a linear filter from the loopback (the
//! "reference") to the mic and subtracts its estimate of the echo from the
//! mic.
//!
//! # Algorithm and source
//!
//! A multidelay block frequency-domain adaptive filter (MDF; J.-S. Soo and
//! K. K. Pang, "Multidelay block frequency domain adaptive filter", IEEE
//! Trans. ASSP 38(2), 1990). The filter is cut into `M` partitions of one
//! block each; every partition is applied by FFT (overlap-save, FFT size two
//! blocks), and the gradient is projected back onto a causal one-block
//! response for partition 0 and one rotating partition per block (the
//! "alternately updated" variant). The step size is the per-bin optimal
//! learning rate of J.-M. Valin, "On adjusting the learning rate in frequency
//! domain echo cancellation with double-talk", IEEE Trans. ASLP 15(3), 2007:
//! the estimated residual echo over the error power, with the residual
//! estimated from a running regression of the error's spectral power on the
//! echo estimate's (the "leak"). That rate falls by itself when the owner
//! talks over the far end, because the error grows and the echo estimate
//! does not. On top sits the two-path model: a background filter adapts all
//! the time and is copied into the foreground filter, whose output is used,
//! only when it has been measurably better; a background that becomes
//! measurably worse (it diverged in double talk) is reset from the
//! foreground. A proportionate weighting gives the active partitions more of
//! the step. Pre-emphasis (0.9) whitens speech for faster convergence, a DC
//! notch removes the mic's offset, and a watchdog resets everything when the
//! output carries far more energy than the input.
//!
//! The structure, the constants and the foreground/background tests follow
//! Speex's `libspeexdsp/mdf.c` (Jean-Marc Valin, 2003-2008, BSD 3-clause
//! licence). This is an independent Rust implementation of that algorithm,
//! not a translation of the file.
//!
//! # Signals
//!
//! 16 kHz mono, `f32` at full scale 1.0 (the pipeline's float convention);
//! internally PCM16 units, which the regularisation constants assume. Blocks
//! are [`AecConfig::block`] samples, a divisor of 1,600, so a 100 ms frame is
//! whole blocks; [`Aec::process`] takes any whole number of blocks.
//!
//! # Alignment (the caller's job)
//!
//! The mic and the reference must already sit on one timeline: sample `i`
//! of both slices is the same instant, with the echo arriving at a lag at or
//! above 0 (after the reference) and within the filter's span. The
//! recommended timeline is the mic's own sample clock: the mic is never
//! resampled (what the engine and the archive hear stays bit-for-bit the
//! capture minus the echo estimate, and the output can replace `mic.wav`
//! one-to-one under the mic's clock map), and the reference is mapped onto
//! it through meeting time. For mic sample `m`: QPC = mic map at `m`, then
//! loopback sample = the loopback map's inverse at that QPC, fractional, read
//! with [`SincInterpolator`]. Rounding to whole samples instead would slip
//! the reference by one sample every 1.2 s at the measured −51.6 ppm, and
//! each slip is a step in the echo path the filter has to relearn.
//!
//! The clock maps are not enough on their own. On meeting 12 the echo's lag
//! under them is a sawtooth, sliding 8.8 ppm and jumping back 5 ms every
//! 9.5 minutes, and it wobbles by about 60 µs from second to second; both
//! come from the render path after the loopback tap, which no anchor sees.
//! The evaluation (`tests/aec_spike.rs`) re-times the reference from the
//! signals (the lag per 2 s and the phase slope per 1 s); research 14
//! section 2 has the numbers. Unaligned, the filter does not keep up: the
//! unit test with a −50 ppm reference keeps about 11 dB against 32 dB
//! corrected.
//!
//! [`AecConfig::bulk_delay`] moves the span: negative values delay the mic
//! (a lookahead on the reference) so that echo up to that much before the
//! reference stays inside the span (clock-map jitter of a few ms); positive
//! values delay the reference for outputs whose echo arrives later than the
//! filter reaches (Bluetooth or HDMI, 100 to 300 ms). The default, −5 ms
//! with a 200 ms filter, covers lags of −5 to +195 ms.
//!
//! The reference should be the loopback before the app's AGC: the AGC
//! raises a quiet far end by up to 40 dB and moves at 8 to 40 dB/s, which the
//! filter sees as a changing echo path (on meeting 12 it made the output
//! louder than the input in 34 two-second windows and added energy to the
//! owner's double-talk words; with the AGC undone, none).
//!
//! # Latency
//!
//! Output sample `i` belongs to input sample `i - latency()`. A negative
//! bulk delay adds that many samples (the mic waits for the reference),
//! residual suppression adds one block (overlap-add). Block processing adds
//! nothing beyond what a caller that already works in 100 ms frames waits
//! for, but live, the mic frame can be cancelled only once the loopback has
//! delivered the same span plus the lookahead.

use std::collections::VecDeque;
use std::sync::Arc;

use realfft::num_complex::Complex32;
use realfft::{ComplexToReal, RealFftPlanner, RealToComplex};

/// The sample rate the canceller's constants are set for.
pub const AEC_RATE: u32 = 16_000;
/// Samples in the pipeline's 100 ms frame; a block must divide it.
pub const FRAME_100MS: usize = 1_600;
/// Full-scale `f32` to the internal PCM16 units.
const SCALE: f32 = 32_768.0;
const PREEMPH: f32 = 0.9;
/// DC notch pole radius at 16 kHz (about a 45 Hz corner).
const NOTCH_RADIUS: f32 = 0.982;
/// Lowest leak estimate: no hope of more than about 33 dB of cancellation.
const MIN_LEAK: f64 = 0.005;
/// Two-path model: smoothing and decision constants.
const VAR1_SMOOTH: f64 = 0.36;
const VAR2_SMOOTH: f64 = 0.7225;
const VAR1_UPDATE: f64 = 0.5;
const VAR2_UPDATE: f64 = 0.25;
const VAR_BACKTRACK: f64 = 4.0;

/// Canceller parameters.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AecConfig {
    /// Samples per block: a divisor of 1,600 (80, 160, 200, 320, 400 ...).
    pub block: usize,
    /// Echo path the filter models, in samples, rounded up to whole blocks
    /// (2,048 to 4,096 for 128 to 256 ms).
    pub filter_len: usize,
    /// Signed bulk delay in samples. Negative: the mic is delayed (a
    /// lookahead; costs that much latency); positive: the reference is.
    pub bulk_delay: i32,
    /// Spectral residual echo suppression after the linear filter (mild),
    /// or none.
    pub residual: Option<ResidualConfig>,
}

impl Default for AecConfig {
    /// 10 ms blocks, a 200 ms filter, 5 ms lookahead, no residual stage.
    fn default() -> Self {
        Self {
            block: 160,
            filter_len: 3_200,
            bulk_delay: -80,
            residual: None,
        }
    }
}

impl AecConfig {
    /// Filter length in milliseconds after rounding to whole blocks.
    pub fn filter_ms(&self) -> f32 {
        self.partitions() as f32 * self.block as f32 * 1_000.0 / AEC_RATE as f32
    }

    fn partitions(&self) -> usize {
        self.filter_len.div_ceil(self.block.max(1)).max(1)
    }

    /// Checks the block and filter sizes.
    pub fn validate(&self) -> Result<(), String> {
        if self.block < 16 || !FRAME_100MS.is_multiple_of(self.block) {
            return Err(format!(
                "block {} is not a divisor of {FRAME_100MS} of at least 16",
                self.block
            ));
        }
        if self.filter_len < self.block || self.filter_len > 16_000 {
            return Err(format!(
                "filter length {} is outside one block to 1 s",
                self.filter_len
            ));
        }
        if self.bulk_delay.unsigned_abs() > 16_000 {
            return Err(format!("bulk delay {} is over 1 s", self.bulk_delay));
        }
        Ok(())
    }
}

/// Residual echo suppression: a Wiener gain per bin against the echo the
/// linear filter is estimated to have left (the leak times the echo
/// estimate's spectrum, held over the reverberation), floored so that it
/// never takes more than [`ResidualConfig::max_attenuation_db`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ResidualConfig {
    /// Deepest attenuation of a bin, dB.
    pub max_attenuation_db: f32,
    /// Over-estimation factor of the residual echo (1 = as estimated).
    pub over: f32,
}

impl Default for ResidualConfig {
    fn default() -> Self {
        Self {
            max_attenuation_db: 12.0,
            over: 1.0,
        }
    }
}

/// Counters for diagnostics and the evaluation.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct AecStats {
    /// Blocks processed.
    pub blocks: u64,
    /// Watchdog resets: the output carried far more energy than the input
    /// for 50 blocks in a row, or a value went non-finite (divergence).
    pub resets: u32,
    /// Background filter copied into the foreground.
    pub foreground_updates: u64,
    /// Background filter reset from the foreground (it got worse).
    pub background_resets: u64,
    /// Blocks whose adaptation was skipped for a saturated mic.
    pub saturated_blocks: u64,
    /// The current leak estimate (residual echo share, 0.005 to 1).
    pub leak: f32,
    /// The initial adaptation phase is over.
    pub adapted: bool,
}

/// The echo canceller: [`AecConfig`] plus the adaptive state.
pub struct Aec {
    cfg: AecConfig,
    mdf: Mdf,
    res: Option<Residual>,
    mic_delay: Delay,
    ref_delay: Delay,
    mic_blk: Vec<f32>,
    ref_blk: Vec<f32>,
    out_blk: Vec<f32>,
    echo_blk: Vec<f32>,
    res_blk: Vec<f32>,
}

impl Aec {
    /// A canceller with an empty (zero) echo path.
    pub fn new(cfg: AecConfig) -> Result<Self, String> {
        cfg.validate()?;
        let f = cfg.block;
        let lookahead = (-cfg.bulk_delay).max(0) as usize;
        let ref_delay = cfg.bulk_delay.max(0) as usize;
        Ok(Self {
            mdf: Mdf::new(f, cfg.partitions()),
            res: cfg.residual.map(|r| Residual::new(f, r)),
            mic_delay: Delay::new(lookahead),
            ref_delay: Delay::new(ref_delay),
            mic_blk: vec![0.0; f],
            ref_blk: vec![0.0; f],
            out_blk: vec![0.0; f],
            echo_blk: vec![0.0; f],
            res_blk: vec![0.0; f],
            cfg,
        })
    }

    pub fn config(&self) -> &AecConfig {
        &self.cfg
    }

    /// Output sample `i` belongs to input sample `i - latency()`.
    pub fn latency(&self) -> usize {
        self.mic_delay.len()
            + if self.res.is_some() {
                self.cfg.block
            } else {
                0
            }
    }

    pub fn stats(&self) -> AecStats {
        self.mdf.stats
    }

    /// Freezes (false) or resumes (true) adaptation; frozen, the foreground
    /// filter keeps cancelling with the echo path it has. For evaluations.
    pub fn set_adaptation(&mut self, on: bool) {
        self.mdf.adapt = on;
    }

    /// Back to an empty echo path (keeps the delay lines' contents).
    pub fn reset(&mut self) {
        self.mdf.reset();
        if let Some(r) = &mut self.res {
            r.reset();
        }
    }

    /// Cancels the echo of `reference` in `mic` into `out`. All three have
    /// the same length, a whole number of blocks; full scale 1.0.
    ///
    /// # Panics
    ///
    /// On lengths that differ or are not whole blocks.
    pub fn process(&mut self, mic: &[f32], reference: &[f32], out: &mut [f32]) {
        let f = self.cfg.block;
        assert!(
            mic.len() == reference.len() && mic.len() == out.len(),
            "mic, reference and out must have one length"
        );
        assert!(
            mic.len().is_multiple_of(f),
            "length must be whole blocks of {f}"
        );
        for ((m, r), o) in mic
            .chunks_exact(f)
            .zip(reference.chunks_exact(f))
            .zip(out.chunks_exact_mut(f))
        {
            self.mic_delay.run(m, &mut self.mic_blk);
            self.ref_delay.run(r, &mut self.ref_blk);
            for v in self.mic_blk.iter_mut().chain(self.ref_blk.iter_mut()) {
                *v *= SCALE;
            }
            self.mdf.block(
                &self.mic_blk,
                &self.ref_blk,
                &mut self.out_blk,
                &mut self.echo_blk,
            );
            let src = match &mut self.res {
                Some(res) => {
                    res.process(
                        &self.out_blk,
                        &self.echo_blk,
                        self.mdf.stats.leak,
                        &mut self.res_blk,
                    );
                    &self.res_blk
                }
                None => &self.out_blk,
            };
            for (o, &v) in o.iter_mut().zip(src) {
                *o = (v / SCALE).clamp(-1.0, 1.0);
            }
        }
    }
}

/// A FIFO delay of a fixed number of samples.
struct Delay {
    buf: VecDeque<f32>,
    len: usize,
}

impl Delay {
    fn new(len: usize) -> Self {
        let mut buf = VecDeque::with_capacity(len + FRAME_100MS);
        buf.extend(std::iter::repeat_n(0.0, len));
        Self { buf, len }
    }

    fn len(&self) -> usize {
        self.len
    }

    fn run(&mut self, input: &[f32], out: &mut [f32]) {
        if self.len == 0 {
            out.copy_from_slice(input);
            return;
        }
        self.buf.extend(input.iter().copied());
        for o in out.iter_mut() {
            *o = self.buf.pop_front().unwrap_or(0.0);
        }
    }
}

/// The DC notch on the mic: a double zero at DC, poles at
/// [`NOTCH_RADIUS`]. The output path applies it to the mic, so a linear
/// decomposition of the output (tests, evaluations) applies it too.
#[derive(Debug, Clone, Copy, Default)]
pub struct Notch {
    mem: [f32; 2],
}

impl Notch {
    pub fn step(&mut self, vin: f32) -> f32 {
        let r = NOTCH_RADIUS;
        let den2 = r * r + 0.7 * (1.0 - r) * (1.0 - r);
        let vout = self.mem[0] + vin;
        self.mem[0] = self.mem[1] + 2.0 * (-vin + r * vout);
        self.mem[1] = vin - den2 * vout;
        r * vout
    }

    /// The filtered copy of a whole signal (in any units: it is linear).
    pub fn apply(x: &[f32]) -> Vec<f32> {
        let mut n = Self::default();
        x.iter().map(|&v| n.step(v)).collect()
    }
}

/// Real FFT of size `n` with the forward transform scaled by `1/n` and the
/// inverse unscaled, so that a round trip is the identity.
struct Fft {
    n: usize,
    fwd: Arc<dyn RealToComplex<f32>>,
    inv: Arc<dyn ComplexToReal<f32>>,
    fwd_scratch: Vec<Complex32>,
    inv_scratch: Vec<Complex32>,
}

impl Fft {
    fn new(n: usize) -> Self {
        let mut planner = RealFftPlanner::<f32>::new();
        let fwd = planner.plan_fft_forward(n);
        let inv = planner.plan_fft_inverse(n);
        Self {
            n,
            fwd_scratch: fwd.make_scratch_vec(),
            inv_scratch: inv.make_scratch_vec(),
            fwd,
            inv,
        }
    }

    /// `time` (destroyed) into `spec` (`n/2 + 1` bins).
    fn forward(&mut self, time: &mut [f32], spec: &mut [Complex32]) {
        self.fwd
            .process_with_scratch(time, spec, &mut self.fwd_scratch)
            .expect("forward FFT sizes");
        let s = 1.0 / self.n as f32;
        for c in spec.iter_mut() {
            *c *= s;
        }
    }

    /// `spec` (destroyed) into `time`.
    fn inverse(&mut self, spec: &mut [Complex32], time: &mut [f32]) {
        spec[0].im = 0.0;
        if let Some(last) = spec.last_mut() {
            last.im = 0.0;
        }
        self.inv
            .process_with_scratch(spec, time, &mut self.inv_scratch)
            .expect("inverse FFT sizes");
    }
}

/// The adaptive filter proper, in PCM16 units, one block at a time.
struct Mdf {
    /// Block, FFT size, partitions, bins.
    f: usize,
    n: usize,
    m: usize,
    k: usize,
    fft: Fft,
    time: Vec<f32>,
    spec: Vec<Complex32>,
    acc: Vec<Complex32>,
    // Input conditioning.
    notch: Notch,
    mem_x: f32,
    mem_d: f32,
    mem_e: f32,
    x_prev: Vec<f32>,
    x_cur: Vec<f32>,
    d: Vec<f32>,
    notched: Vec<f32>,
    y_fg: Vec<f32>,
    y_bg: Vec<f32>,
    e_bg: Vec<f32>,
    echo: Vec<f32>,
    /// Far-end spectra of the last `m` blocks, a ring; `head` is the newest.
    x_hist: Vec<Complex32>,
    head: usize,
    w_fg: Vec<Complex32>,
    w_bg: Vec<Complex32>,
    e_spec: Vec<Complex32>,
    prop: Vec<f32>,
    power: Vec<f32>,
    power_1: Vec<f32>,
    eh: Vec<f32>,
    yh: Vec<f32>,
    rf: Vec<f32>,
    yf: Vec<f32>,
    window: Vec<f32>,
    pey: f64,
    pyy: f64,
    davg1: f64,
    davg2: f64,
    dvar1: f64,
    dvar2: f64,
    sum_adapt: f64,
    saturated: usize,
    screwed_up: u32,
    count: u64,
    adapt: bool,
    stats: AecStats,
}

impl Mdf {
    fn new(f: usize, m: usize) -> Self {
        let n = 2 * f;
        let k = f + 1;
        let window = (0..n)
            .map(|i| 0.5 - 0.5 * (std::f32::consts::TAU * i as f32 / n as f32).cos())
            .collect();
        let mut s = Self {
            f,
            n,
            m,
            k,
            fft: Fft::new(n),
            time: vec![0.0; n],
            spec: vec![Complex32::default(); k],
            acc: vec![Complex32::default(); k],
            notch: Notch::default(),
            mem_x: 0.0,
            mem_d: 0.0,
            mem_e: 0.0,
            x_prev: vec![0.0; f],
            x_cur: vec![0.0; f],
            d: vec![0.0; f],
            notched: vec![0.0; f],
            y_fg: vec![0.0; f],
            y_bg: vec![0.0; f],
            e_bg: vec![0.0; f],
            echo: vec![0.0; f],
            x_hist: vec![Complex32::default(); m * k],
            head: 0,
            w_fg: vec![Complex32::default(); m * k],
            w_bg: vec![Complex32::default(); m * k],
            e_spec: vec![Complex32::default(); k],
            prop: vec![0.0; m],
            power: vec![0.0; k],
            power_1: vec![0.0; k],
            eh: vec![0.0; k],
            yh: vec![0.0; k],
            rf: vec![0.0; k],
            yf: vec![0.0; k],
            window,
            pey: 1.0,
            pyy: 1.0,
            davg1: 0.0,
            davg2: 0.0,
            dvar1: 0.0,
            dvar2: 0.0,
            sum_adapt: 0.0,
            saturated: 0,
            screwed_up: 0,
            count: 0,
            adapt: true,
            stats: AecStats::default(),
        };
        s.reset();
        s
    }

    /// Clears the adaptive state (not the counters).
    fn reset(&mut self) {
        let zero = Complex32::default();
        for v in [&mut self.x_hist, &mut self.w_fg, &mut self.w_bg] {
            v.fill(zero);
        }
        for v in [
            &mut self.power,
            &mut self.power_1,
            &mut self.eh,
            &mut self.yh,
            &mut self.x_prev,
        ] {
            v.fill(0.0);
        }
        self.notch = Notch::default();
        self.mem_x = 0.0;
        self.mem_d = 0.0;
        self.mem_e = 0.0;
        // Initial proportionate weights: decaying over the partitions.
        let decay = (-2.4 / self.m as f32).exp();
        let mut p = 0.7f32;
        for w in self.prop.iter_mut() {
            *w = p;
            p *= decay;
        }
        let sum: f32 = self.prop.iter().sum();
        for w in self.prop.iter_mut() {
            *w = 0.8 * *w / sum;
        }
        self.pey = 1.0;
        self.pyy = 1.0;
        self.davg1 = 0.0;
        self.davg2 = 0.0;
        self.dvar1 = 0.0;
        self.dvar2 = 0.0;
        self.sum_adapt = 0.0;
        self.saturated = 0;
        self.screwed_up = 0;
        self.stats.adapted = false;
        self.stats.leak = 1.0;
    }

    /// The time-domain response (the last block of the overlap-save
    /// output) of the foreground (`fg`) or background filter to the far-end
    /// history, into `y_fg` or `y_bg`.
    fn filter(&mut self, fg: bool) {
        let (k, m) = (self.k, self.m);
        self.acc.fill(Complex32::default());
        let w = if fg { &self.w_fg } else { &self.w_bg };
        for j in 0..m {
            let xj = &self.x_hist[((self.head + j) % m) * k..][..k];
            let wj = &w[j * k..][..k];
            for ((a, &x), &wv) in self.acc.iter_mut().zip(xj).zip(wj) {
                *a += x * wv;
            }
        }
        self.spec.copy_from_slice(&self.acc);
        self.fft.inverse(&mut self.spec, &mut self.time);
        let dst = if fg { &mut self.y_fg } else { &mut self.y_bg };
        dst.copy_from_slice(&self.time[self.f..]);
    }

    /// One block: `mic` and `refb` in PCM16 units; writes the cancelled mic
    /// to `out` and what was subtracted from the (notched) mic to `echo`.
    fn block(&mut self, mic: &[f32], refb: &[f32], out: &mut [f32], echo: &mut [f32]) {
        let (f, n, m, k) = (self.f, self.n, self.m, self.k);
        self.stats.blocks += 1;

        // 1. Conditioning: pre-emphasis on both, DC notch on the mic.
        let mut sxx = 0f64;
        for (xc, &x) in self.x_cur.iter_mut().zip(refb) {
            let xp = x - PREEMPH * self.mem_x;
            self.mem_x = x;
            *xc = xp;
            sxx += f64::from(xp * xp);
        }
        let mut sdd = 0f64;
        for ((&v, nd), d) in mic.iter().zip(&mut self.notched).zip(&mut self.d) {
            if v.abs() >= 32_000.0 {
                self.saturated = m + 1;
            }
            let nv = self.notch.step(v);
            *nd = nv;
            let dp = nv - PREEMPH * self.mem_d;
            self.mem_d = nv;
            *d = dp;
            sdd += f64::from(dp * dp);
        }

        // 2. The far-end spectrum of [previous block, this block].
        self.head = (self.head + m - 1) % m;
        self.time[..f].copy_from_slice(&self.x_prev);
        self.time[f..].copy_from_slice(&self.x_cur);
        self.x_prev.copy_from_slice(&self.x_cur);
        self.fft.forward(&mut self.time, &mut self.spec);
        self.x_hist[self.head * k..][..k].copy_from_slice(&self.spec);

        // 3. Foreground and background responses and errors.
        self.filter(true);
        self.filter(false);
        let (mut sff, mut see, mut dbf) = (0f64, 0f64, 10f64);
        for i in 0..f {
            let ef = f64::from(self.d[i] - self.y_fg[i]);
            let eb = self.d[i] - self.y_bg[i];
            let dy = f64::from(self.y_fg[i] - self.y_bg[i]);
            self.e_bg[i] = eb;
            sff += ef * ef;
            see += f64::from(eb * eb);
            dbf += dy * dy;
        }

        // 4. Two-path model: promote a better background, or reset a worse one.
        let mut transition = false;
        if self.adapt {
            let diff = sff - see;
            self.davg1 = 0.6 * self.davg1 + 0.4 * diff;
            self.davg2 = 0.85 * self.davg2 + 0.15 * diff;
            self.dvar1 = VAR1_SMOOTH * self.dvar1 + 0.16 * sff * dbf;
            self.dvar2 = VAR2_SMOOTH * self.dvar2 + 0.0225 * sff * dbf;
            let promote = diff * diff.abs() > sff * dbf
                || self.davg1 * self.davg1.abs() > VAR1_UPDATE * self.dvar1
                || self.davg2 * self.davg2.abs() > VAR2_UPDATE * self.dvar2;
            if promote {
                self.davg1 = 0.0;
                self.davg2 = 0.0;
                self.dvar1 = 0.0;
                self.dvar2 = 0.0;
                self.w_fg.copy_from_slice(&self.w_bg);
                self.stats.foreground_updates += 1;
                transition = true;
            } else {
                let backtrack = -diff * diff.abs() > VAR_BACKTRACK * sff * dbf
                    || -self.davg1 * self.davg1.abs() > VAR_BACKTRACK * self.dvar1
                    || -self.davg2 * self.davg2.abs() > VAR_BACKTRACK * self.dvar2;
                if backtrack {
                    self.w_bg.copy_from_slice(&self.w_fg);
                    self.y_bg.copy_from_slice(&self.y_fg);
                    for i in 0..f {
                        self.e_bg[i] = self.d[i] - self.y_fg[i];
                    }
                    see = sff;
                    self.davg1 = 0.0;
                    self.davg2 = 0.0;
                    self.dvar1 = 0.0;
                    self.dvar2 = 0.0;
                    self.stats.background_resets += 1;
                }
            }
        }
        // The echo estimate used: the foreground's, cross-faded into the
        // new foreground over the block after a promotion.
        for i in 0..f {
            self.echo[i] = if transition {
                self.window[i + f] * self.y_fg[i] + self.window[i] * self.y_bg[i]
            } else {
                self.y_fg[i]
            };
        }

        // 5. Output with de-emphasis.
        for i in 0..f {
            let e = self.d[i] - self.echo[i];
            let o = e + PREEMPH * self.mem_e;
            self.mem_e = o;
            out[i] = o;
            echo[i] = self.notched[i] - o;
        }

        // 6. Watchdog.
        let sane = sff.is_finite()
            && see.is_finite()
            && sxx.is_finite()
            && sff < n as f64 * 1e9
            && sxx < n as f64 * 1e9;
        if !sane {
            self.screwed_up += 50;
        } else if sff / 4.0 > sdd + n as f64 * 10_000.0 {
            self.screwed_up += 1;
        } else {
            self.screwed_up = 0;
        }
        if self.screwed_up >= 50 {
            self.stats.resets += 1;
            self.reset();
            out.copy_from_slice(&self.notched);
            echo.fill(0.0);
            return;
        }
        if !self.adapt {
            return;
        }

        // 7. Learning rate (Valin 2007) from the background's error.
        let see = see.max(n as f64 * 100.0);
        let (mut sey, mut syy) = (0f64, 0f64);
        for i in 0..f {
            sey += f64::from(self.e_bg[i] * self.y_bg[i]);
            syy += f64::from(self.y_bg[i] * self.y_bg[i]);
        }
        self.time[..f].fill(0.0);
        self.time[f..].copy_from_slice(&self.e_bg);
        self.fft.forward(&mut self.time, &mut self.e_spec);
        self.time[..f].fill(0.0);
        self.time[f..].copy_from_slice(&self.y_bg);
        self.fft.forward(&mut self.time, &mut self.spec);
        let ss = 0.35 / m as f32;
        let spec_average = f as f32 / AEC_RATE as f32;
        let x0 = &self.x_hist[self.head * k..][..k];
        let (mut pey, mut pyy) = (0f64, 0f64);
        for (i, x) in x0.iter().enumerate() {
            let rf = self.e_spec[i].norm_sqr();
            let yf = self.spec[i].norm_sqr();
            self.rf[i] = rf;
            self.yf[i] = yf;
            self.power[i] = (1.0 - ss) * self.power[i] + 1.0 + ss * x.norm_sqr();
            let eh = f64::from(rf - self.eh[i]);
            let yh = f64::from(yf - self.yh[i]);
            pey += eh * yh;
            pyy += yh * yh;
            self.eh[i] = (1.0 - spec_average) * self.eh[i] + spec_average * rf;
            self.yh[i] = (1.0 - spec_average) * self.yh[i] + spec_average * yf;
        }
        let pyy = pyy.sqrt();
        let pey = if pyy > 0.0 { pey / pyy } else { 0.0 };
        let beta0 = 2.0 * f as f64 / f64::from(AEC_RATE);
        let beta_max = 0.5 * f as f64 / f64::from(AEC_RATE);
        let alpha = (beta0 * syy).min(beta_max * see) / see;
        self.pey = (1.0 - alpha) * self.pey + alpha * pey;
        self.pyy = ((1.0 - alpha) * self.pyy + alpha * pyy).max(1.0);
        self.pey = self.pey.clamp(MIN_LEAK * self.pyy, self.pyy);
        let leak = self.pey / self.pyy;
        self.stats.leak = leak as f32;
        // Residual-to-error ratio, floored by the echo still visible in the
        // error, capped at 0.5.
        let mut rer = (1e-4 * sxx + 3.0 * leak * syy) / see;
        rer = rer.max(sey * sey / (1.0 + see * syy)).min(0.5);
        if !self.stats.adapted && self.sum_adapt > m as f64 && leak > 0.03 && syy > 0.0 {
            self.stats.adapted = true;
        }
        if self.stats.adapted {
            let (leak, rer) = (leak as f32, rer as f32);
            for i in 0..k {
                let e = self.rf[i] + 1.0;
                let r = (leak * self.yf[i]).min(0.5 * e);
                let r = 0.7 * r + 0.3 * rer * e;
                self.power_1[i] = r / (e * (self.power[i] + 10.0));
            }
            self.adjust_prop();
        } else {
            // Fixed rate until the filter has learnt something.
            let mut rate = 0.0;
            if sxx > n as f64 * 1_000.0 {
                rate = (0.25 * sxx).min(0.25 * see) / see;
            }
            for i in 0..k {
                self.power_1[i] = rate as f32 / (self.power[i] + 10.0);
            }
            self.sum_adapt += rate;
        }

        // 8. Gradient step on the background, then the causality constraint.
        if self.saturated == 0 {
            for j in 0..m {
                let p = self.prop[j];
                let xj = &self.x_hist[((self.head + j) % m) * k..][..k];
                let wj = &mut self.w_bg[j * k..][..k];
                for i in 0..k {
                    wj[i] += (xj[i].conj() * self.e_spec[i]) * (p * self.power_1[i]);
                }
            }
        } else {
            self.saturated -= 1;
            self.stats.saturated_blocks += 1;
        }
        let rotating = if m > 1 {
            1 + (self.count % (m as u64 - 1)) as usize
        } else {
            0
        };
        self.constrain(0);
        if rotating != 0 {
            self.constrain(rotating);
        }
        self.count += 1;
    }

    /// Projects background partition `j` onto a causal one-block response.
    fn constrain(&mut self, j: usize) {
        let (f, k) = (self.f, self.k);
        self.spec.copy_from_slice(&self.w_bg[j * k..][..k]);
        self.fft.inverse(&mut self.spec, &mut self.time);
        self.time[f..].fill(0.0);
        self.fft.forward(&mut self.time, &mut self.spec);
        self.w_bg[j * k..][..k].copy_from_slice(&self.spec);
    }

    /// Proportionate step: each partition's share follows its energy.
    fn adjust_prop(&mut self) {
        let k = self.k;
        let mut max_sum = 1.0f32;
        for (j, p) in self.prop.iter_mut().enumerate() {
            let e: f32 = self.w_bg[j * k..][..k].iter().map(|c| c.norm_sqr()).sum();
            *p = (1.0 + e).sqrt();
            max_sum = max_sum.max(*p);
        }
        let mut sum = 1.0f32;
        for p in self.prop.iter_mut() {
            *p += 0.1 * max_sum;
            sum += *p;
        }
        for p in self.prop.iter_mut() {
            *p = 0.99 * *p / sum;
        }
    }
}

/// The residual echo suppressor: a short-time spectral gain on the
/// canceller's output, 50 percent overlap-add with square-root Hann windows
/// of two blocks (one block of latency).
struct Residual {
    f: usize,
    fft: Fft,
    win: Vec<f32>,
    g_min: f32,
    over: f32,
    e_prev: Vec<f32>,
    y_prev: Vec<f32>,
    ola: Vec<f32>,
    time: Vec<f32>,
    e_spec: Vec<Complex32>,
    y_spec: Vec<Complex32>,
    echo_psd: Vec<f32>,
    prev_gain: Vec<f32>,
    prev_pe: Vec<f32>,
}

impl Residual {
    fn new(f: usize, cfg: ResidualConfig) -> Self {
        let n = 2 * f;
        let k = f + 1;
        let win = (0..n)
            .map(|i| (0.5 - 0.5 * (std::f32::consts::TAU * i as f32 / n as f32).cos()).sqrt())
            .collect();
        Self {
            f,
            fft: Fft::new(n),
            win,
            g_min: 10f32.powf(-cfg.max_attenuation_db.abs() / 20.0),
            over: cfg.over.max(0.0),
            e_prev: vec![0.0; f],
            y_prev: vec![0.0; f],
            ola: vec![0.0; f],
            time: vec![0.0; n],
            e_spec: vec![Complex32::default(); k],
            y_spec: vec![Complex32::default(); k],
            echo_psd: vec![0.0; k],
            prev_gain: vec![1.0; k],
            prev_pe: vec![0.0; k],
        }
    }

    fn reset(&mut self) {
        self.echo_psd.fill(0.0);
        self.prev_gain.fill(1.0);
        self.prev_pe.fill(0.0);
    }

    /// `e`: the canceller's output block; `y`: what it subtracted; writes
    /// the suppressed output of the previous block to `out`.
    fn process(&mut self, e: &[f32], y: &[f32], leak: f32, out: &mut [f32]) {
        let f = self.f;
        for (i, t) in self.time.iter_mut().enumerate() {
            let v = if i < f { self.e_prev[i] } else { e[i - f] };
            *t = v * self.win[i];
        }
        self.fft.forward(&mut self.time, &mut self.e_spec);
        for (i, t) in self.time.iter_mut().enumerate() {
            let v = if i < f { self.y_prev[i] } else { y[i - f] };
            *t = v * self.win[i];
        }
        self.fft.forward(&mut self.time, &mut self.y_spec);
        self.e_prev.copy_from_slice(e);
        self.y_prev.copy_from_slice(y);
        let leak2 = (2.0 * leak).min(1.0);
        const DD: f32 = 0.9;
        for i in 0..self.e_spec.len() {
            let r = leak2 * self.y_spec[i].norm_sqr();
            // Held over the reverberation: decays 0.6 per block.
            self.echo_psd[i] = (0.6 * self.echo_psd[i]).max(r);
            let noise = self.over * self.echo_psd[i] + 1e-3;
            let pe = self.e_spec[i].norm_sqr();
            let post = pe / noise;
            let prio = DD * self.prev_gain[i] * self.prev_gain[i] * self.prev_pe[i] / noise
                + (1.0 - DD) * (post - 1.0).max(0.0);
            let g = (prio / (1.0 + prio)).max(self.g_min);
            self.prev_gain[i] = g;
            self.prev_pe[i] = pe;
            self.e_spec[i] *= g;
        }
        self.fft.inverse(&mut self.e_spec, &mut self.time);
        for (i, t) in self.time.iter_mut().enumerate() {
            *t *= self.win[i];
        }
        let (head, tail) = self.time.split_at(f);
        for ((o, ola), (&h, &t)) in out.iter_mut().zip(&mut self.ola).zip(head.iter().zip(tail)) {
            *o = *ola + h;
            *ola = t;
        }
    }
}

/// Band-limited (windowed-sinc, Kaiser) interpolation of a signal at
/// fractional positions: how the caller places the loopback on the mic's
/// clock. A table of `phases` sub-sample phases, linear between them.
#[derive(Debug, Clone)]
pub struct SincInterpolator {
    half: usize,
    phases: usize,
    table: Vec<f32>,
}

impl Default for SincInterpolator {
    /// 32 taps, 512 phases, passband to 0.95 of Nyquist (7.6 kHz).
    fn default() -> Self {
        Self::new(16, 512, 0.95)
    }
}

impl SincInterpolator {
    /// `half` taps on each side, `phases` table phases, `cutoff` as a
    /// fraction of Nyquist.
    pub fn new(half: usize, phases: usize, cutoff: f64) -> Self {
        let half = half.max(1);
        let phases = phases.max(1);
        let taps = 2 * half;
        let beta = 8.0;
        let i0b = bessel_i0(beta);
        let mut table = Vec::with_capacity((phases + 1) * taps);
        for p in 0..=phases {
            let frac = p as f64 / phases as f64;
            let start = table.len();
            for t in 0..taps {
                // Offset of tap `t` from the wanted position.
                let x = t as f64 - (half as f64 - 1.0) - frac;
                let sinc = if x.abs() < 1e-12 {
                    cutoff
                } else {
                    (std::f64::consts::PI * cutoff * x).sin() / (std::f64::consts::PI * x)
                };
                let r = x / half as f64;
                let w = if r.abs() >= 1.0 {
                    0.0
                } else {
                    bessel_i0(beta * (1.0 - r * r).sqrt()) / i0b
                };
                table.push((sinc * w) as f32);
            }
            let sum: f32 = table[start..].iter().sum();
            for c in &mut table[start..] {
                *c /= sum;
            }
        }
        Self {
            half,
            phases,
            table,
        }
    }

    /// The signal's value at fractional sample `pos` (zero outside `src`).
    pub fn at(&self, src: &[f32], pos: f64) -> f32 {
        let i0 = pos.floor();
        let frac = pos - i0;
        let fp = frac * self.phases as f64;
        let p = (fp.floor() as usize).min(self.phases - 1);
        let t = (fp - p as f64) as f32;
        let taps = 2 * self.half;
        let c0 = &self.table[p * taps..][..taps];
        let c1 = &self.table[(p + 1) * taps..][..taps];
        let first = i0 as i64 - self.half as i64 + 1;
        let mut acc = 0.0f32;
        if first >= 0 && first as usize + taps <= src.len() {
            let s = &src[first as usize..][..taps];
            for ((&v, &a), &b) in s.iter().zip(c0).zip(c1) {
                acc += v * (a + t * (b - a));
            }
        } else {
            for tap in 0..taps {
                let idx = first + tap as i64;
                if idx >= 0 && (idx as usize) < src.len() {
                    acc += src[idx as usize] * (c0[tap] + t * (c1[tap] - c0[tap]));
                }
            }
        }
        acc
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

    const SR: usize = AEC_RATE as usize;

    /// xorshift32 in [-1, 1).
    struct Rng(u32);
    impl Rng {
        fn next(&mut self) -> f32 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 17;
            self.0 ^= self.0 << 5;
            (self.0 as f32 / u32::MAX as f32) * 2.0 - 1.0
        }
        fn uniform(&mut self, lo: f32, hi: f32) -> f32 {
            lo + (hi - lo) * (self.next() * 0.5 + 0.5)
        }
    }

    /// Speech-like noise bursts: noise through two resonators whose centre
    /// frequencies change per burst, 120 to 400 ms on with 15 ms ramps,
    /// 60 to 300 ms off, ±4 dB per burst; RMS about `rms_dbfs` while on.
    /// `active` restricts the bursts to the given (start, end) sample spans.
    fn speechlike(len: usize, seed: u32, rms_dbfs: f32, active: &[(usize, usize)]) -> Vec<f32> {
        let mut rng = Rng(seed);
        let mut out = vec![0.0f32; len];
        let mut i = 0usize;
        while i < len {
            let on = (rng.uniform(0.12, 0.40) * SR as f32) as usize;
            let off = (rng.uniform(0.06, 0.30) * SR as f32) as usize;
            let gain = 10f32.powf(rng.uniform(-4.0, 4.0) / 20.0);
            let formants = [rng.uniform(300.0, 1_000.0), rng.uniform(1_000.0, 3_200.0)];
            let mut st = [[0.0f32; 2]; 2];
            let ramp = (0.015 * SR as f32) as usize;
            let mut burst = Vec::with_capacity(on);
            for n in 0..on {
                let v = rng.next();
                let mut y = 0.1 * v;
                for (fi, &fc) in formants.iter().enumerate() {
                    let r = 0.9f32;
                    let w = std::f32::consts::TAU * fc / SR as f32;
                    let s = &mut st[fi];
                    let o = v + 2.0 * r * w.cos() * s[0] - r * r * s[1];
                    s[1] = s[0];
                    s[0] = o;
                    y += o * (1.0 - r);
                }
                let env = if n < ramp {
                    n as f32 / ramp as f32
                } else if on - n < ramp {
                    (on - n) as f32 / ramp as f32
                } else {
                    1.0
                };
                burst.push(y * env);
            }
            let rms = (burst.iter().map(|v| v * v).sum::<f32>() / on as f32)
                .sqrt()
                .max(1e-9);
            let target = 10f32.powf(rms_dbfs / 20.0) * gain;
            for (n, v) in burst.iter().enumerate() {
                let idx = i + n;
                if idx < len && active.iter().any(|&(a, b)| idx >= a && idx < b) {
                    out[idx] = v / rms * target;
                }
            }
            i += on + off;
        }
        out
    }

    /// A short room: 3 ms direct path, a few early reflections, an
    /// exponential tail (RT60 about 200 ms) to 120 ms; energy gain −10 dB.
    fn room(seed: u32) -> Vec<f32> {
        let mut rng = Rng(seed);
        let len = (0.12 * SR as f32) as usize;
        let mut h = vec![0.0f32; len];
        h[48] = 1.0;
        for _ in 0..6 {
            let d = (rng.uniform(0.004, 0.025) * SR as f32) as usize;
            h[d] += rng.uniform(-0.5, 0.5);
        }
        let decay = (-6.9 / (0.2 * SR as f32)).exp();
        let mut a = 0.25f32;
        for v in h.iter_mut().skip(60) {
            *v += a * rng.next();
            a *= decay;
        }
        let e: f32 = h.iter().map(|v| v * v).sum();
        let g = (0.1 / e).sqrt();
        h.iter().map(|v| v * g).collect()
    }

    /// Linear convolution by FFT (overlap-add), the length of `x`.
    fn convolve(x: &[f32], h: &[f32]) -> Vec<f32> {
        let b = 4_096;
        let n = (b + h.len()).next_power_of_two();
        let mut fft = Fft::new(n);
        let mut t = vec![0.0f32; n];
        t[..h.len()].copy_from_slice(h);
        let mut hs = vec![Complex32::default(); n / 2 + 1];
        fft.forward(&mut t, &mut hs);
        let mut xs = hs.clone();
        let mut y = vec![0.0f32; x.len() + n];
        for (c, chunk) in x.chunks(b).enumerate() {
            t.fill(0.0);
            t[..chunk.len()].copy_from_slice(chunk);
            fft.forward(&mut t, &mut xs);
            for (a, &hv) in xs.iter_mut().zip(&hs) {
                *a *= hv * n as f32;
            }
            fft.inverse(&mut xs, &mut t);
            for (o, &v) in y[c * b..].iter_mut().zip(&t) {
                *o += v;
            }
        }
        y.truncate(x.len());
        y
    }

    fn noise(len: usize, seed: u32, dbfs: f32) -> Vec<f32> {
        let mut rng = Rng(seed);
        let a = 10f32.powf(dbfs / 20.0) * 3f32.sqrt();
        (0..len).map(|_| rng.next() * a).collect()
    }

    /// Runs `aec` over whole signals; output aligned with the input.
    fn run(aec: &mut Aec, mic: &[f32], reference: &[f32]) -> Vec<f32> {
        let f = aec.config().block;
        let lat = aec.latency();
        let total = (mic.len() + lat).div_ceil(f) * f;
        let mut m = mic.to_vec();
        m.resize(total, 0.0);
        let mut r = reference.to_vec();
        r.resize(total, 0.0);
        let mut out = vec![0.0; total];
        for ((mc, rc), oc) in m
            .chunks(FRAME_100MS.min(total))
            .zip(r.chunks(FRAME_100MS.min(total)))
            .zip(out.chunks_mut(FRAME_100MS.min(total)))
        {
            aec.process(mc, rc, oc);
        }
        out.drain(..lat);
        out.truncate(mic.len());
        out
    }

    fn energy(x: &[f32], spans: &[(usize, usize)]) -> f64 {
        spans
            .iter()
            .map(|&(a, b)| {
                x[a..b]
                    .iter()
                    .map(|&v| f64::from(v) * f64::from(v))
                    .sum::<f64>()
            })
            .sum()
    }

    fn db(num: f64, den: f64) -> f64 {
        10.0 * (num.max(1e-30) / den.max(1e-30)).log10()
    }

    fn secs(a: f32, b: f32) -> (usize, usize) {
        ((a * SR as f32) as usize, (b * SR as f32) as usize)
    }

    struct Scene {
        reference: Vec<f32>,
        echo: Vec<f32>,
        near: Vec<f32>,
        floor: Vec<f32>,
        mic: Vec<f32>,
    }

    /// Reference at −26 dBFS while talking, echo 10 dB below it through
    /// the room, near end `near_over_echo_db` above the echo, floor at
    /// −70 dBFS.
    fn scene(
        seconds: f32,
        far: &[(usize, usize)],
        near: &[(usize, usize)],
        near_over_echo_db: f32,
    ) -> Scene {
        let len = (seconds * SR as f32) as usize;
        let reference = speechlike(len, 0x1234_5678, -26.0, far);
        let echo = convolve(&reference, &room(99));
        let near = speechlike(len, 0x0BAD_F00D, -36.0 + near_over_echo_db, near);
        let floor = noise(len, 4242, -70.0);
        let mic = (0..len).map(|i| echo[i] + near[i] + floor[i]).collect();
        Scene {
            reference,
            echo,
            near,
            floor,
            mic,
        }
    }

    #[test]
    fn config_checks() {
        assert!(AecConfig::default().validate().is_ok());
        let bad = AecConfig {
            block: 256,
            ..AecConfig::default()
        };
        assert!(Aec::new(bad).is_err());
        let c = AecConfig {
            block: 320,
            filter_len: 2_048,
            ..AecConfig::default()
        };
        assert_eq!(c.partitions(), 7);
        assert!((c.filter_ms() - 140.0).abs() < 1e-3);
        let aec = Aec::new(AecConfig {
            residual: Some(ResidualConfig::default()),
            ..AecConfig::default()
        })
        .unwrap();
        assert_eq!(aec.latency(), 80 + 160);
    }

    #[test]
    fn interpolator_is_exact_on_samples_and_band_limited_between() {
        let it = SincInterpolator::default();
        let x: Vec<f32> = (0..2_000)
            .map(|i| (i as f32 * 0.05).sin() + 0.3 * (i as f32 * 0.9).cos())
            .collect();
        for i in [100usize, 777, 1_500] {
            assert!((it.at(&x, i as f64) - x[i]).abs() < 1e-3);
        }
        // Half a sample of a 1 kHz tone.
        let w = std::f64::consts::TAU * 1_000.0 / SR as f64;
        let tone: Vec<f32> = (0..4_000).map(|i| (w * i as f64).sin() as f32).collect();
        let mut err = 0f64;
        for i in 1_000..3_000 {
            let pos = i as f64 + 0.5;
            err = err.max((f64::from(it.at(&tone, pos)) - (w * pos).sin()).abs());
        }
        assert!(err < 2e-3, "{err}");
    }

    /// Far end only: ERLE at least 20 dB after 3 s.
    #[test]
    fn converges_on_far_end_only() {
        let all = [secs(0.0, 12.0)];
        let s = scene(12.0, &all, &[], 0.0);
        let mut aec = Aec::new(AecConfig::default()).unwrap();
        let out = run(&mut aec, &s.mic, &s.reference);
        let after = [secs(3.0, 12.0)];
        let erle = db(energy(&s.mic, &after), energy(&out, &after));
        let first = [secs(0.0, 1.0)];
        let early = db(energy(&s.mic, &first), energy(&out, &first));
        // The noise floor caps what can be measured.
        let ceiling = db(energy(&s.mic, &after), energy(&s.floor, &after));
        println!(
            "far end only: ERLE {erle:.1} dB after 3 s ({early:.1} dB in the first second; \
             floor ceiling {ceiling:.1} dB); {:?}",
            aec.stats()
        );
        assert!(erle >= 20.0, "ERLE {erle:.1} dB");
        assert_eq!(aec.stats().resets, 0);
    }

    /// Near end only (after convergence): the owner's level within 1 dB,
    /// with and without residual suppression.
    #[test]
    fn keeps_the_near_end() {
        let far = [secs(0.0, 6.0), secs(14.0, 18.0)];
        let near = [secs(6.5, 13.5)];
        let s = scene(18.0, &far, &near, 5.0);
        for residual in [None, Some(ResidualConfig::default())] {
            let mut aec = Aec::new(AecConfig {
                residual,
                ..AecConfig::default()
            })
            .unwrap();
            let out = run(&mut aec, &s.mic, &s.reference);
            // Skip the room's tail after the far end stops.
            let span = [secs(6.5, 13.5)];
            let change = db(energy(&out, &span), energy(&s.mic, &span));
            let lsd = log_spectral_distance(&s.mic, &out, span[0]);
            println!(
                "near end only, residual {}: level change {change:+.2} dB, log-spectral distance {lsd:.2} dB",
                residual.is_some()
            );
            assert!(change.abs() <= 1.0, "{change:.2} dB");
        }
    }

    /// RMS over frames and bins of the dB difference of two signals'
    /// spectra (32 ms frames, bins 100 Hz to 7 kHz, frames over −60 dBFS).
    fn log_spectral_distance(a: &[f32], b: &[f32], span: (usize, usize)) -> f64 {
        let n = 512;
        let mut fft = Fft::new(n);
        let win: Vec<f32> = (0..n)
            .map(|i| 0.5 - 0.5 * (std::f32::consts::TAU * i as f32 / n as f32).cos())
            .collect();
        let mut ta = vec![0.0f32; n];
        let mut sa = vec![Complex32::default(); n / 2 + 1];
        let mut sb = sa.clone();
        let (lo, hi) = (100 * n / SR, 7_000 * n / SR);
        let (mut sum, mut frames) = (0f64, 0usize);
        let mut i = span.0;
        while i + n <= span.1 {
            let e: f32 = a[i..i + n].iter().map(|v| v * v).sum::<f32>() / n as f32;
            if 10.0 * e.max(1e-20).log10() > -60.0 {
                for (j, t) in ta.iter_mut().enumerate() {
                    *t = a[i + j] * win[j];
                }
                fft.forward(&mut ta, &mut sa);
                for (j, t) in ta.iter_mut().enumerate() {
                    *t = b[i + j] * win[j];
                }
                fft.forward(&mut ta, &mut sb);
                let mut acc = 0f64;
                for k in lo..hi {
                    let d = 10.0 * (f64::from(sb[k].norm_sqr()) + 1e-20).log10()
                        - 10.0 * (f64::from(sa[k].norm_sqr()) + 1e-20).log10();
                    acc += d * d;
                }
                sum += (acc / (hi - lo) as f64).sqrt();
                frames += 1;
            }
            i += n / 2;
        }
        if frames == 0 {
            0.0
        } else {
            sum / frames as f64
        }
    }

    /// 60 s alternating far end only, double talk, near end only and double
    /// talk again, 3 s each, with the owner only 5 dB above the echo: no
    /// reset, the echo stays cancelled after every double-talk stretch, and
    /// double talk never adds energy.
    #[test]
    fn double_talk_does_not_diverge() {
        let mut far = Vec::new();
        let mut near = Vec::new();
        let mut far_only = Vec::new();
        let mut dt = Vec::new();
        let mut t = 0.0f32;
        while t < 60.0 {
            // far only, double talk, near only, double talk
            far.push(secs(t, t + 6.0));
            far_only.push(secs(t + 0.5, t + 3.0));
            near.push(secs(t + 3.0, t + 12.0));
            dt.push(secs(t + 3.0, t + 6.0));
            dt.push(secs(t + 9.0, t + 12.0));
            far.push(secs(t + 9.0, t + 12.0));
            t += 12.0;
        }
        let s = scene(60.0, &far, &near, 5.0);
        let mut aec = Aec::new(AecConfig::default()).unwrap();
        let out = run(&mut aec, &s.mic, &s.reference);
        let len = s.mic.len();
        let clip = |v: &[(usize, usize)]| -> Vec<(usize, usize)> {
            v.iter()
                .map(|&(a, b)| (a, b.min(len)))
                .filter(|&(a, b)| a < b)
                .collect()
        };
        let (far_only, dt) = (clip(&far_only), clip(&dt));
        // Residual echo, exactly: out = notch(echo + near + floor) - y.
        let other = Notch::apply(&(0..len).map(|i| s.near[i] + s.floor[i]).collect::<Vec<_>>());
        let resid: Vec<f32> = (0..len).map(|i| out[i] - other[i]).collect();
        let mut line = String::new();
        for (n, span) in far_only.iter().enumerate().skip(1) {
            let erle = db(energy(&s.mic, &[*span]), energy(&out, &[*span]));
            line.push_str(&format!(" {erle:.1}"));
            assert!(erle >= 15.0, "far-end stretch {n}: ERLE {erle:.1} dB");
        }
        let erle_dt = db(energy(&s.echo, &dt), energy(&resid, &dt));
        let added = db(energy(&out, &dt), energy(&s.mic, &dt));
        let near_kept = db(energy(&out, &dt), energy(&s.near, &dt));
        println!(
            "double talk: far-end-only ERLE per stretch{line} dB; in double talk: echo removed \
             {erle_dt:.1} dB, output vs mic {added:+.2} dB, output vs near end {near_kept:+.2} dB; {:?}",
            aec.stats()
        );
        assert_eq!(aec.stats().resets, 0);
        assert!(
            erle_dt >= 10.0,
            "echo removed in double talk {erle_dt:.1} dB"
        );
        assert!(added <= 0.0, "double talk added {added:.2} dB");
    }

    /// The reference's clock 50 ppm slow against the mic's (the echo slides
    /// 0.8 samples a second, 3 ms over the minute): without the caller's
    /// correction against with it ([`SincInterpolator`]).
    #[test]
    fn reference_drift_needs_the_callers_correction() {
        let secs_total = 60.0f32;
        let len = (secs_total * SR as f32) as usize;
        let all = [(0usize, len + 4_000)];
        // The loopback (reference clock) and what the speaker played.
        let lb = speechlike(len + 4_000, 0x5151_5151, -26.0, &all);
        let acoustic = convolve(&lb, &room(7));
        let floor = noise(len, 77, -70.0);
        // The mic samples the room on its own clock: mic sample m is
        // reference time m * (1 - 50e-6).
        let ratio = 1.0 - 50e-6;
        let fine = SincInterpolator::new(32, 2_048, 0.95);
        let mic: Vec<f32> = (0..len)
            .map(|m| fine.at(&acoustic, m as f64 * ratio) + floor[m])
            .collect();
        let it = SincInterpolator::default();
        let corrected: Vec<f32> = (0..len).map(|m| it.at(&lb, m as f64 * ratio)).collect();
        let late = [secs(40.0, 60.0)];
        let mut results = Vec::new();
        for (name, reference) in [("uncorrected", &lb[..len]), ("corrected", &corrected[..])] {
            let mut aec = Aec::new(AecConfig::default()).unwrap();
            let out = run(&mut aec, &mic, reference);
            let mut per10 = String::new();
            for b in 0..6 {
                let span = [secs(b as f32 * 10.0, b as f32 * 10.0 + 10.0)];
                per10.push_str(&format!(
                    " {:.1}",
                    db(energy(&mic, &span), energy(&out, &span))
                ));
            }
            let erle = db(energy(&mic, &late), energy(&out, &late));
            println!("-50 ppm drift, {name}: ERLE per 10 s{per10} dB; 40-60 s {erle:.1} dB");
            results.push(erle);
        }
        assert!(results[1] >= 20.0, "corrected {:.1} dB", results[1]);
        assert!(results[1] > results[0], "{results:?}");
    }

    /// Microseconds per 100 ms frame (run with `--release` for the number
    /// that counts), with and without the residual stage, 128 and 256 ms.
    #[test]
    fn benchmark_per_100ms_frame() {
        let far = [secs(0.0, 20.0)];
        let near = [secs(5.0, 20.0)];
        let s = scene(20.0, &far, &near, 5.0);
        for (filter_len, residual) in [
            (2_048, None),
            (3_200, None),
            (4_096, None),
            (3_200, Some(ResidualConfig::default())),
        ] {
            let mut aec = Aec::new(AecConfig {
                filter_len,
                residual,
                ..AecConfig::default()
            })
            .unwrap();
            let mut out = vec![0.0f32; FRAME_100MS];
            let started = std::time::Instant::now();
            let frames = s.mic.len() / FRAME_100MS;
            for i in 0..frames {
                let r = i * FRAME_100MS..(i + 1) * FRAME_100MS;
                aec.process(&s.mic[r.clone()], &s.reference[r], &mut out);
            }
            let us = started.elapsed().as_secs_f64() * 1e6 / frames as f64;
            println!(
                "benchmark: {:.0} ms filter, residual {}: {us:.1} us per 100 ms frame ({:.2} percent of a core)",
                aec.config().filter_ms(),
                residual.is_some(),
                us / 1e3
            );
        }
    }
}
