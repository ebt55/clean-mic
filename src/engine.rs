//! The processing engine (`docs/design.md` section 4): capture packets in,
//! one cleaned 10 ms block at a time out.
//!
//! [`Engine::packet`] places each packet on its stream's [`Timeline`],
//! decimates it to 16 kHz, and then processes every 10 ms mic block whose
//! reference is available: the loopback has delivered the span the
//! [`Placement`] maps the block to, or the wait for it has run out (nothing
//! is playing), in which case the loopback timeline is extended with zeros.
//! "Now" is the arrival time of the latest packet, so a replay of the same
//! packets makes the same decisions and the same output, bit for bit.

use std::time::Instant;

use crate::aec::{AecStats, SincInterpolator};
use crate::chain::{Chain, ChainConfig, BLOCK, BLOCK48};
use crate::clock::{Placement, Timeline, TimelineStats, HNS_PER_SEC};
use crate::resample::{Down3, DOWN3_DELAY};
use crate::stats::Summary;

/// Which capture stream a packet came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum Stream {
    Mic,
    Loopback,
}

/// One WASAPI packet, downmixed to 48 kHz mono.
#[derive(Debug, Clone, Default)]
pub struct Packet {
    pub stream: Option<Stream>,
    /// QPC (100 ns) of the first frame, as WASAPI reports it.
    pub qpc: i64,
    /// QPC when the capture thread read it.
    pub arrival: i64,
    pub silent: bool,
    pub discontinuity: bool,
    pub timestamp_error: bool,
    pub samples: Vec<f32>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EngineConfig {
    pub chain: ChainConfig,
    /// How long after its time a reference span may still arrive before it
    /// counts as silence (100 ns units).
    pub wait_hns: f64,
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            chain: ChainConfig::default(),
            wait_hns: 0.020 * HNS_PER_SEC,
        }
    }
}

/// One processed block, handed to the [`Sink`].
pub struct Block<'a> {
    /// Index of the block's first sample on the 16 kHz mic timeline.
    pub t0: u64,
    pub mic16: &'a [f32],
    /// The reference placed on the mic's clock.
    pub ref16: &'a [f32],
    /// The canceller's output; lags `mic16` by [`Chain::aec_delay`].
    pub aec16: &'a [f32],
    /// Final output frames completed by this block; lag the mic by
    /// [`Chain::delay48`] (48 kHz samples on the mic timeline times three).
    pub out48: &'a [f32],
    /// Processing time of this block, µs (its share of decimation included).
    pub dsp_us: f32,
    /// QPC when the block was done minus the capture QPC of its oldest
    /// output sample, ms (`None` before the mic's clock fit exists).
    pub latency_ms: Option<f32>,
    /// Mic against loopback, ppm (the placement's rate).
    pub drift_ppm: f64,
}

/// Where the engine's results go (the run folder, the console, the cable).
pub trait Sink {
    /// Every packet, before it is placed (raw recording for replays).
    fn packet(&mut self, _p: &Packet) {}
    fn block(&mut self, b: &Block<'_>);
}

/// The engine's account of a run.
#[derive(Debug, Clone, Default, serde::Serialize)]
pub struct EngineReport {
    pub blocks: u64,
    pub mic: TimelineStats,
    pub loopback: TimelineStats,
    /// Each stream's drift against QPC and the mic against the loopback
    /// (the placement's rate), ppm.
    pub mic_drift_ppm: f64,
    pub loopback_drift_ppm: f64,
    pub mic_vs_loopback_ppm: f64,
    pub placement_jumps: u64,
    pub placement_error_samples: f64,
    /// Blocks whose reference came (partly) from zeros extended by
    /// timestamp, and how many separate stretches of them.
    pub silence_blocks: u64,
    pub silence_stretches: u64,
    pub dsp_us: Summary,
    pub latency_ms: Summary,
    pub aec: AecStatsReport,
}

#[derive(Debug, Clone, Default, serde::Serialize)]
pub struct AecStatsReport {
    pub resets: u32,
    pub foreground_updates: u64,
    pub background_resets: u64,
    pub saturated_blocks: u64,
    pub leak: f32,
    pub adapted: bool,
}

impl From<AecStats> for AecStatsReport {
    fn from(s: AecStats) -> Self {
        Self {
            resets: s.resets,
            foreground_updates: s.foreground_updates,
            background_resets: s.background_resets,
            saturated_blocks: s.saturated_blocks,
            leak: s.leak,
            adapted: s.adapted,
        }
    }
}

/// Decimated samples with the timeline index of the first one.
#[derive(Default)]
struct Buf16 {
    base: u64,
    data: Vec<f32>,
}

impl Buf16 {
    fn end(&self) -> u64 {
        self.base + self.data.len() as u64
    }

    /// Drops samples before `index` once enough have piled up.
    fn trim_before(&mut self, index: u64) {
        let drop = index.saturating_sub(self.base) as usize;
        if drop > 16_000 {
            let drop = drop.min(self.data.len());
            self.data.drain(..drop);
            self.base += drop as u64;
        }
    }
}

pub struct Engine {
    cfg: EngineConfig,
    live: bool,
    mic: Timeline,
    lb: Timeline,
    mic_down: Down3,
    lb_down: Down3,
    mic16: Buf16,
    lb16: Buf16,
    placement: Placement,
    sinc: SincInterpolator,
    chain: Chain,
    /// Next mic block to process.
    t_next: u64,
    now: i64,
    in_silence: bool,
    pos: Vec<f64>,
    ref16: Vec<f32>,
    aec16: Vec<f32>,
    out48: Vec<f32>,
    pending_ns: u64,
    dsp_us: Vec<f32>,
    latency_ms: Vec<f32>,
    report: EngineReport,
}

impl Engine {
    /// `live`: latency is measured against the QPC clock; in a replay,
    /// against the packets' arrival times.
    pub fn new(cfg: EngineConfig, live: bool) -> Result<Self, String> {
        Ok(Self {
            chain: Chain::new(cfg.chain)?,
            cfg,
            live,
            mic: Timeline::default(),
            lb: Timeline::default(),
            mic_down: Down3::new(),
            lb_down: Down3::new(),
            mic16: Buf16::default(),
            lb16: Buf16::default(),
            placement: Placement::default(),
            sinc: SincInterpolator::default(),
            t_next: 0,
            now: 0,
            in_silence: false,
            pos: vec![0.0; BLOCK],
            ref16: vec![0.0; BLOCK],
            aec16: vec![0.0; BLOCK],
            out48: Vec::with_capacity(2 * BLOCK48),
            pending_ns: 0,
            dsp_us: Vec::new(),
            latency_ms: Vec::new(),
            report: EngineReport::default(),
        })
    }

    pub fn chain(&self) -> &Chain {
        &self.chain
    }

    pub fn chain_mut(&mut self) -> &mut Chain {
        &mut self.chain
    }

    /// Drift of the mic against the loopback (the placement's rate), ppm.
    pub fn drift_ppm(&self) -> f64 {
        (self.placement.rate() - 1.0) * 1e6
    }

    pub fn packet(&mut self, p: &Packet, sink: &mut dyn Sink) {
        sink.packet(p);
        self.now = self.now.max(p.arrival);
        let started = Instant::now();
        let n = p.samples.len();
        let (tl, down, buf) = match p.stream {
            Some(Stream::Mic) => (&mut self.mic, &mut self.mic_down, &mut self.mic16),
            Some(Stream::Loopback) => (&mut self.lb, &mut self.lb_down, &mut self.lb16),
            None => return,
        };
        let placed = tl.place(p.qpc, n, p.discontinuity, p.timestamp_error);
        let zeros = [0.0f32; 480];
        let mut left = placed.zeros;
        while left > 0 {
            let k = left.min(zeros.len());
            down.process(&zeros[..k], &mut buf.data);
            left -= k;
        }
        if placed.skip < n {
            if p.silent {
                let mut left = n - placed.skip;
                while left > 0 {
                    let k = left.min(zeros.len());
                    down.process(&zeros[..k], &mut buf.data);
                    left -= k;
                }
            } else {
                down.process(&p.samples[placed.skip..], &mut buf.data);
            }
        }
        self.pending_ns += started.elapsed().as_nanos() as u64;
        self.pump(sink);
    }

    /// Processes every mic block whose reference is ready or overdue.
    fn pump(&mut self, sink: &mut dyn Sink) {
        while self.mic16.end() >= self.t_next + BLOCK as u64 {
            let mut silence = false;
            if self.lb.started() {
                let (mic_fit, lb_fit) = (&self.mic.fit, &self.lb.fit);
                let start = Placement::target(mic_fit, lb_fit, self.t_next as f64);
                let end = Placement::target(mic_fit, lb_fit, (self.t_next + BLOCK as u64) as f64);
                let (Some(start), Some(end)) = (start, end) else { break };
                // The placement may run a little ahead of the fits' target
                // (it slews; beyond 1 ms it jumps to the target instead).
                let end = match self.placement.next() {
                    Some(p) if (p - start).abs() <= 16.0 => end.max(p + (end - start) + 0.01),
                    _ => end,
                };
                // The interpolator reads 16 samples past floor(pos); one more
                // for the slew.
                let needed = (end.floor() + 18.0).max(0.0) as u64;
                if self.lb16.end() <= needed {
                    // Decimator output n needs loopback sample 3 n.
                    let need_pos = 3 * needed;
                    let q_need = self.lb.fit.qpc_at(need_pos as f64).unwrap_or(f64::MAX);
                    if (self.now as f64) - q_need < self.cfg.wait_hns {
                        break;
                    }
                    let started = Instant::now();
                    let have = self.lb.position();
                    let add = self.lb.extend_to(need_pos + 3);
                    debug_assert!(have + add as u64 == self.lb.position());
                    let zeros = [0.0f32; 480];
                    let mut left = add;
                    while left > 0 {
                        let k = left.min(zeros.len());
                        self.lb_down.process(&zeros[..k], &mut self.lb16.data);
                        left -= k;
                    }
                    self.pending_ns += started.elapsed().as_nanos() as u64;
                    silence = true;
                }
            }
            self.process_block(silence, sink);
        }
    }

    fn process_block(&mut self, silence: bool, sink: &mut dyn Sink) {
        let started = Instant::now();
        let t0 = self.t_next;
        let m0 = (t0 - self.mic16.base) as usize;
        let placed = self.lb.started()
            && self
                .placement
                .block(&self.mic.fit, &self.lb.fit, t0, &mut self.pos)
                .is_some();
        if placed {
            let base = self.lb16.base as f64;
            for (r, &p) in self.ref16.iter_mut().zip(&self.pos) {
                *r = self.sinc.at(&self.lb16.data, p - base);
            }
        } else {
            self.ref16.fill(0.0);
        }
        self.out48.clear();
        let mic = &self.mic16.data[m0..m0 + BLOCK];
        self.chain
            .process(mic, &self.ref16, &mut self.aec16, &mut self.out48);
        let ns = started.elapsed().as_nanos() as u64 + std::mem::take(&mut self.pending_ns);
        let dsp_us = ns as f32 / 1e3;
        // The oldest output sample of this block is mic sample
        // t0 - delay48 / 3; its capture time against now.
        let lag16 = self.chain.delay48() as f64 / 3.0;
        let done = if self.live { crate::qpc_now() } else { self.now };
        let latency_ms = self
            .mic
            .fit
            .qpc_at(3.0 * (t0 as f64 - lag16) - DOWN3_DELAY as f64)
            .filter(|_| t0 as f64 > lag16)
            .map(|q| ((done as f64 - q) / 1e4) as f32);
        sink.block(&Block {
            t0,
            mic16: mic,
            ref16: &self.ref16,
            aec16: &self.aec16,
            out48: &self.out48,
            dsp_us,
            latency_ms,
            drift_ppm: (self.placement.rate() - 1.0) * 1e6,
        });
        self.report.blocks += 1;
        if silence {
            self.report.silence_blocks += 1;
            if !self.in_silence {
                self.report.silence_stretches += 1;
            }
        }
        self.in_silence = silence;
        self.dsp_us.push(dsp_us);
        if let Some(l) = latency_ms {
            self.latency_ms.push(l);
        }
        self.t_next += BLOCK as u64;
        self.mic16.trim_before(self.t_next);
        if placed {
            let oldest = (self.pos[0].floor() as i64 - 64).max(0) as u64;
            self.lb16.trim_before(oldest);
        }
    }

    /// The account so far.
    pub fn report(&self) -> EngineReport {
        let mut r = self.report.clone();
        r.mic = self.mic.stats;
        r.loopback = self.lb.stats;
        r.mic_drift_ppm = self.mic.fit.drift_ppm();
        r.loopback_drift_ppm = self.lb.fit.drift_ppm();
        r.mic_vs_loopback_ppm = self.drift_ppm();
        r.placement_jumps = self.placement.jumps;
        r.placement_error_samples = self.placement.last_error;
        r.dsp_us = Summary::of(&self.dsp_us);
        r.latency_ms = Summary::of(&self.latency_ms);
        r.aec = self.chain.aec_stats().into();
        r
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::clock::HNS_PER_SAMPLE;

    /// Collects the outputs.
    #[derive(Default)]
    struct Collect {
        mic: Vec<f32>,
        reference: Vec<f32>,
        aec: Vec<f32>,
        out: Vec<f32>,
    }

    impl Sink for Collect {
        fn block(&mut self, b: &Block<'_>) {
            self.mic.extend_from_slice(b.mic16);
            self.reference.extend_from_slice(b.ref16);
            self.aec.extend_from_slice(b.aec16);
            self.out.extend_from_slice(b.out48);
        }
    }

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

    /// The reference placed on the mic's clock matches what the mic heard:
    /// a loopback at 0 ppm, a mic at -51.6 ppm hearing it 3 ms late, packets
    /// arriving 10 ms after their first frame. After the fits settle, the
    /// placed reference and the mic correlate at a lag of 3 ms.
    #[test]
    fn placed_reference_lines_up_with_the_echo() {
        let secs = 20usize;
        let mut seed = 5;
        // Band-limited noise: white noise through a short moving average.
        let raw = noise(48_000 * (secs + 1), &mut seed, 0.3);
        let src: Vec<f32> = raw.windows(4).map(|w| w.iter().sum::<f32>() / 4.0).collect();
        let mut eng = Engine::new(
            EngineConfig {
                chain: ChainConfig {
                    rnnoise: false,
                    ..ChainConfig::default()
                },
                ..EngineConfig::default()
            },
            false,
        )
        .unwrap();
        let mut sink = Collect::default();
        let origin = 1e10;
        let mic_hns = HNS_PER_SAMPLE * (1.0 - 51.6e-6);
        let it = SincInterpolator::new(32, 2_048, 0.95);
        let echo_delay = 0.003 * 48_000.0;
        let (mut mi, mut li) = (0usize, 0usize);
        while mi < 48_000 * secs {
            // Interleave by arrival time.
            let mq = origin + mi as f64 * mic_hns;
            let lq = origin + li as f64 * HNS_PER_SAMPLE;
            if lq <= mq {
                let samples = src[li..li + 480].to_vec();
                let qpc = lq.round() as i64;
                eng.packet(
                    &Packet {
                        stream: Some(Stream::Loopback),
                        qpc,
                        arrival: qpc + 100_000,
                        samples,
                        ..Packet::default()
                    },
                    &mut sink,
                );
                li += 480;
            } else {
                // The mic samples the source (loopback clock) at its own rate.
                let samples: Vec<f32> = (mi..mi + 480)
                    .map(|m| {
                        let t = m as f64 * mic_hns / HNS_PER_SAMPLE - echo_delay;
                        0.3 * it.at(&src, t)
                    })
                    .collect();
                let qpc = mq.round() as i64;
                eng.packet(
                    &Packet {
                        stream: Some(Stream::Mic),
                        qpc,
                        arrival: qpc + 100_000,
                        samples,
                        ..Packet::default()
                    },
                    &mut sink,
                );
                mi += 480;
            }
        }
        let r = eng.report();
        assert!((r.mic_vs_loopback_ppm + 51.6).abs() < 2.0, "{r:?}");
        assert_eq!(r.silence_blocks, 0, "{r:?}");
        // Correlation peak of mic against reference over the last 10 s.
        let (a, b) = (sink.mic.len() - 160_000, sink.mic.len() - 1_000);
        let mut best = (0i64, f64::MIN);
        for lag in -10i64..=100 {
            let c: f64 = (a..b)
                .map(|i| f64::from(sink.mic[i]) * f64::from(sink.reference[(i as i64 - lag) as usize]))
                .sum();
            if c > best.1 {
                best = (lag, c);
            }
        }
        assert_eq!(best.0, 48, "lag {} samples", best.0);
        // And the canceller removes most of it.
        let e = |x: &[f32]| x[a..b].iter().map(|&v| f64::from(v * v)).sum::<f64>();
        let shifted: Vec<f32> = sink.aec[160..].to_vec();
        let erle = 10.0 * (e(&sink.mic) / e(&shifted[..sink.mic.len() - 160])).log10();
        assert!(erle > 25.0, "ERLE {erle:.1} dB");
    }

    /// With no loopback packets at all the mic still flows (zeros as the
    /// reference, no waiting); when the loopback stops mid-run the engine
    /// waits at most `wait_hns` and extends it with zeros.
    #[test]
    fn silence_never_stalls_the_mic() {
        let mut eng = Engine::new(EngineConfig::default(), false).unwrap();
        let mut sink = Collect::default();
        let mut seed = 9;
        let origin = 5e9;
        for k in 0..300usize {
            let q = (origin + (k * 480) as f64 * HNS_PER_SAMPLE) as i64;
            if k < 100 {
                eng.packet(
                    &Packet {
                        stream: Some(Stream::Loopback),
                        qpc: q,
                        arrival: q + 100_000,
                        samples: noise(480, &mut seed, 0.1),
                        ..Packet::default()
                    },
                    &mut sink,
                );
            }
            eng.packet(
                &Packet {
                    stream: Some(Stream::Mic),
                    qpc: q,
                    arrival: q + 100_000,
                    samples: noise(480, &mut seed, 0.01),
                    ..Packet::default()
                },
                &mut sink,
            );
        }
        let r = eng.report();
        // 300 packets of 10 ms; the last few blocks wait for the wait time.
        assert!(r.blocks >= 296, "{}", r.blocks);
        assert_eq!(r.silence_stretches, 1, "{r:?}");
        assert!(r.silence_blocks >= 190, "{r:?}");
    }
}
