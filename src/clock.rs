//! Timestamps and clock drift (`docs/design.md` section 3.2).
//!
//! Every WASAPI packet carries the QPC time (100 ns units) of its first
//! frame. [`Timeline`] places a stream's packets on a gapless 48 kHz sample
//! index the way Diarist's capture pipeline does: continuity is judged
//! packet to packet, a gap over 5 ms is zero-filled by its duration, an
//! overlap is cut, so slow drift never triggers a fill.
//!
//! [`Fit`] is the causal counterpart of Diarist's `ClockMap`
//! (`audio/clock.rs`, MIT, see `NOTICE.md`): instead of interpolating
//! between stored once-a-second anchors, which looks up to a second ahead,
//! it fits QPC against sample index by least squares over a trailing window
//! and extrapolates. The slope is pooled over segments, and each segment
//! (a stretch with no fill, cut or zero extension in it) gets its own
//! intercept, so a resumed loopback's rounding never bends the slope.
//!
//! [`Placement`] turns the two fits into the reference's fractional
//! position for each mic sample, slewing slowly towards the fits' target so
//! that timestamp jitter never becomes a step in the echo path.

use std::collections::VecDeque;

/// QPC units (100 ns) per second.
pub const HNS_PER_SEC: f64 = 1e7;
/// Device rate of both streams (autoconverted to it).
pub const RATE: f64 = 48_000.0;
/// Nominal QPC units per 48 kHz sample.
pub const HNS_PER_SAMPLE: f64 = HNS_PER_SEC / RATE;
/// Timestamp jitter tolerated before a gap is filled or an overlap cut.
pub const GAP_TOLERANCE_HNS: f64 = 50_000.0;

/// Trailing window of the fit.
const FIT_WINDOW_HNS: f64 = 60.0 * HNS_PER_SEC;
/// At most one fit point per this much QPC time within a segment.
const FIT_SPACING_HNS: f64 = 500_000.0;
/// Below this much pooled span the nominal slope is used.
const MIN_SLOPE_SPAN: f64 = 2.0 * RATE;

/// Least-squares map from a stream's sample index to QPC time over a
/// trailing window: a pooled slope, an intercept per segment.
#[derive(Debug, Clone, Default)]
pub struct Fit {
    /// (sample index, QPC, segment), in arrival order.
    points: VecDeque<(f64, f64, u32)>,
    segment: u32,
    /// QPC units per sample, and the QPC at sample 0 on the current
    /// segment's line. `None` before the first point.
    line: Option<(f64, f64)>,
}

impl Fit {
    /// Starts a new segment: the next point gets its own intercept.
    pub fn break_segment(&mut self) {
        if self.points.back().is_some_and(|p| p.2 == self.segment) {
            self.segment += 1;
        }
    }

    /// Adds a timestamp (thinned to one per [`FIT_SPACING_HNS`] within a
    /// segment) and refits.
    pub fn push(&mut self, index: f64, qpc: f64) {
        if let Some(&(_, q, s)) = self.points.back() {
            if s == self.segment && qpc - q < FIT_SPACING_HNS {
                return;
            }
        }
        self.points.push_back((index, qpc, self.segment));
        while self
            .points
            .front()
            .is_some_and(|p| qpc - p.1 > FIT_WINDOW_HNS)
        {
            self.points.pop_front();
        }
        self.refit();
    }

    fn refit(&mut self) {
        struct Seg {
            id: u32,
            n: f64,
            sx: f64,
            sy: f64,
            x0: f64,
            x1: f64,
        }
        let mut segs: Vec<Seg> = Vec::new();
        for &(x, y, s) in &self.points {
            match segs.last_mut() {
                Some(g) if g.id == s => {
                    g.n += 1.0;
                    g.sx += x;
                    g.sy += y;
                    g.x1 = x;
                }
                _ => segs.push(Seg {
                    id: s,
                    n: 1.0,
                    sx: x,
                    sy: y,
                    x0: x,
                    x1: x,
                }),
            }
        }
        let Some(last) = segs.last() else {
            self.line = None;
            return;
        };
        // The pooled within-segment regression.
        let span: f64 = segs.iter().map(|g| g.x1 - g.x0).sum();
        let (mut sxy, mut sxx) = (0.0, 0.0);
        let mut k = 0;
        for &(x, y, s) in &self.points {
            if segs[k].id != s {
                k += 1;
            }
            let (mx, my) = (segs[k].sx / segs[k].n, segs[k].sy / segs[k].n);
            sxy += (x - mx) * (y - my);
            sxx += (x - mx) * (x - mx);
        }
        let slope = if span >= MIN_SLOPE_SPAN && sxx > 0.0 {
            sxy / sxx
        } else {
            HNS_PER_SAMPLE
        };
        let (mx, my) = (last.sx / last.n, last.sy / last.n);
        self.line = Some((slope, my - slope * mx));
    }

    pub fn is_empty(&self) -> bool {
        self.line.is_none()
    }

    /// QPC units per sample (nominal before the first fit).
    pub fn slope(&self) -> f64 {
        self.line.map_or(HNS_PER_SAMPLE, |l| l.0)
    }

    /// Clock drift against QPC, ppm (positive: samples come more slowly
    /// than nominal), as Diarist's `ClockMap::drift_ppm`.
    pub fn drift_ppm(&self) -> f64 {
        (self.slope() / HNS_PER_SAMPLE - 1.0) * 1e6
    }

    /// QPC time of (fractional) sample `s` on the current segment's line.
    pub fn qpc_at(&self, s: f64) -> Option<f64> {
        self.line.map(|(b, a)| a + b * s)
    }

    /// The inverse of [`Self::qpc_at`].
    pub fn sample_at(&self, qpc: f64) -> Option<f64> {
        self.line.map(|(b, a)| (qpc - a) / b)
    }
}

/// What [`Timeline::place`] did with a packet.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Placed {
    /// Zeros placed before the packet (a gap).
    pub zeros: usize,
    /// Samples cut from the packet's head (an overlap).
    pub skip: usize,
}

/// Counters of a stream's timeline.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, serde::Serialize)]
pub struct TimelineStats {
    pub packets: u64,
    pub samples: u64,
    pub gaps: u64,
    pub gap_samples: u64,
    pub overlaps: u64,
    pub overlap_samples: u64,
    pub zero_extensions: u64,
    pub zero_extension_samples: u64,
    pub timestamp_errors: u64,
    pub discontinuities: u64,
}

/// One stream's gapless 48 kHz sample index, placed by QPC.
#[derive(Debug, Clone, Default)]
pub struct Timeline {
    /// Samples on the timeline so far.
    position: u64,
    /// Expected QPC of the next sample: the last packet's timestamp plus its
    /// length at the fitted rate.
    next_qpc: Option<f64>,
    /// Zeros were extended by timestamp after the last real samples.
    extended: bool,
    pub fit: Fit,
    pub stats: TimelineStats,
}

impl Timeline {
    pub fn position(&self) -> u64 {
        self.position
    }

    pub fn started(&self) -> bool {
        self.next_qpc.is_some()
    }

    /// QPC just after the last sample placed.
    pub fn end_qpc(&self) -> Option<f64> {
        self.next_qpc
    }

    /// Places a packet of `n` samples whose first frame is at `qpc`
    /// (`timestamp_error` or a zero timestamp: not usable, the packet is
    /// taken as continuous).
    pub fn place(&mut self, qpc: i64, n: usize, discontinuity: bool, timestamp_error: bool) -> Placed {
        self.stats.packets += 1;
        if discontinuity {
            self.stats.discontinuities += 1;
        }
        let slope = self.fit.slope();
        let bad = timestamp_error || qpc == 0;
        if bad {
            self.stats.timestamp_errors += 1;
        }
        let qpc = match (bad, self.next_qpc) {
            (false, _) => qpc as f64,
            (true, Some(next)) => next,
            // Nothing to place it against yet.
            (true, None) => return Placed { zeros: 0, skip: n },
        };
        let mut placed = Placed::default();
        if let Some(next) = self.next_qpc {
            let gap = qpc - next;
            if gap > GAP_TOLERANCE_HNS {
                placed.zeros = (gap / slope).round() as usize;
                self.stats.gaps += 1;
                self.stats.gap_samples += placed.zeros as u64;
                self.fit.break_segment();
            } else if gap < -GAP_TOLERANCE_HNS {
                placed.skip = ((-gap / slope).round() as usize).min(n);
                self.stats.overlaps += 1;
                self.stats.overlap_samples += placed.skip as u64;
                self.fit.break_segment();
            } else if self.extended || discontinuity {
                self.fit.break_segment();
            }
        }
        self.position += placed.zeros as u64;
        let kept = n - placed.skip;
        if kept > 0 {
            if !bad {
                self.fit
                    .push(self.position as f64, qpc + placed.skip as f64 * slope);
            }
            self.extended = false;
            // Judged packet to packet: drift never accumulates into a fill.
            self.next_qpc = Some(qpc + n as f64 * slope);
        } else if self.next_qpc.is_none() {
            self.next_qpc = Some(qpc + n as f64 * slope);
        }
        // A packet cut whole leaves the expected time where it was.
        self.position += kept as u64;
        self.stats.samples = self.position;
        placed
    }

    /// Extends the timeline with zeros up to sample `target` (the stream has
    /// delivered nothing for that span in time: the loopback while nothing
    /// plays). Returns the zeros added. A real packet that later turns out
    /// to overlap them has its head cut.
    pub fn extend_to(&mut self, target: u64) -> usize {
        if target <= self.position {
            return 0;
        }
        let n = (target - self.position) as usize;
        let slope = self.fit.slope();
        if let Some(next) = self.next_qpc.as_mut() {
            *next += n as f64 * slope;
        }
        self.position = target;
        self.extended = true;
        self.stats.zero_extensions += 1;
        self.stats.zero_extension_samples += n as u64;
        self.stats.samples = self.position;
        n
    }
}

/// Slew limits of [`Placement`].
const SLEW_TAU_SAMPLES: f64 = 2.0 * 16_000.0;
const MAX_SLEW: f64 = 100e-6;
/// Errors beyond this (16 kHz samples, 1 ms) are jumped, not slewed.
const JUMP_SAMPLES: f64 = 16.0;

/// The reference's fractional position (16 kHz loopback samples) for each
/// 16 kHz mic sample, from the two fits. Mic sample `t` sits at mic
/// timeline sample `3 t - 95` (the decimator's delay), the same for the
/// loopback, so `p(t) = (lb.sample_at(mic.qpc_at(3 t - 95)) + 95) / 3`.
#[derive(Debug, Clone, Default)]
pub struct Placement {
    /// Position for the next block's first sample.
    p: Option<f64>,
    /// Loopback samples per mic sample used for the last block.
    rate: f64,
    pub jumps: u64,
    /// Last error to the fits' target (16 kHz samples).
    pub last_error: f64,
}

impl Placement {
    /// The fits' target for mic sample `t`.
    pub fn target(mic: &Fit, lb: &Fit, t: f64) -> Option<f64> {
        let d = crate::resample::DOWN3_DELAY as f64;
        let q = mic.qpc_at(3.0 * t - d)?;
        Some((lb.sample_at(q)? + d) / 3.0)
    }

    /// Reference positions for mic samples `t0 .. t0 + out.len()`; `None`
    /// while either fit is empty.
    pub fn block(&mut self, mic: &Fit, lb: &Fit, t0: u64, out: &mut [f64]) -> Option<()> {
        let n = out.len() as f64;
        let target = Self::target(mic, lb, t0 as f64)?;
        let target_end = Self::target(mic, lb, t0 as f64 + n)?;
        let ratio = (target_end - target) / n;
        let p = match self.p {
            Some(p) if (target - p).abs() <= JUMP_SAMPLES => p,
            prev => {
                if prev.is_some() {
                    self.jumps += 1;
                }
                target
            }
        };
        let e = target - p;
        self.last_error = e;
        let slew = (e / SLEW_TAU_SAMPLES).clamp(-MAX_SLEW, MAX_SLEW);
        self.rate = ratio + slew;
        for (j, o) in out.iter_mut().enumerate() {
            *o = p + self.rate * j as f64;
        }
        self.p = Some(p + self.rate * n);
        Some(())
    }

    /// Where the next block starts, if placed before.
    pub fn next(&self) -> Option<f64> {
        self.p
    }

    /// Loopback samples per mic sample in the last block.
    pub fn rate(&self) -> f64 {
        self.rate
    }

    pub fn reset(&mut self) {
        self.p = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `packets` packets of 480 samples of a stream at `ppm` against QPC
    /// whose sample 0 was at `origin`; `count` is the stream's sample
    /// counter; timestamps carry ±`jitter` hns.
    fn feed(
        tl: &mut Timeline,
        origin: f64,
        ppm: f64,
        count: &mut u64,
        packets: usize,
        jitter: f64,
        seed: &mut u32,
    ) {
        let hns = HNS_PER_SAMPLE * (1.0 + ppm * 1e-6);
        for _ in 0..packets {
            *seed ^= *seed << 13;
            *seed ^= *seed >> 17;
            *seed ^= *seed << 5;
            let j = (*seed as f64 / u32::MAX as f64 - 0.5) * 2.0 * jitter;
            let q = origin + *count as f64 * hns + j;
            tl.place(q.round() as i64, 480, false, false);
            *count += 480;
        }
    }

    #[test]
    fn fit_recovers_drift_through_jitter() {
        let mut tl = Timeline::default();
        let (mut seed, mut count) = (99, 0);
        feed(&mut tl, 1e9, -51.6, &mut count, 6_000, 3_000.0, &mut seed);
        let ppm = tl.fit.drift_ppm();
        assert!((ppm + 51.6).abs() < 1.0, "{ppm}");
        assert_eq!(tl.stats.gaps + tl.stats.overlaps, 0);
    }

    #[test]
    fn gap_is_filled_and_the_new_segment_keeps_the_slope() {
        let mut tl = Timeline::default();
        let (mut seed, mut count) = (7, 0);
        let hns = HNS_PER_SAMPLE * (1.0 - 30e-6);
        feed(&mut tl, 1e9, -30.0, &mut count, 1_000, 500.0, &mut seed);
        // 2 s of nothing, then the stream resumes on the same clock.
        count += 96_000;
        feed(&mut tl, 1e9, -30.0, &mut count, 500, 500.0, &mut seed);
        assert_eq!(tl.stats.gaps, 1);
        // Two timestamps' jitter (±500 hns each) in the fill.
        assert!((tl.stats.gap_samples as i64 - 96_000).abs() <= 6, "{}", tl.stats.gap_samples);
        assert!((tl.fit.drift_ppm() + 30.0).abs() < 1.0, "{}", tl.fit.drift_ppm());
        // The current line predicts the resumed stream's times.
        let q = tl.fit.qpc_at(tl.position() as f64).unwrap();
        let truth = 1e9 + count as f64 * hns;
        assert!((q - truth).abs() < 300.0, "{}", q - truth);
    }

    #[test]
    fn zero_extension_then_late_packet_is_cut() {
        let mut tl = Timeline::default();
        let (mut seed, mut count) = (3, 0);
        feed(&mut tl, 1e9, 0.0, &mut count, 100, 0.0, &mut seed);
        let end = tl.end_qpc().unwrap();
        let pos = tl.position();
        tl.extend_to(pos + 4_800);
        // The real stream was only late: its packets start where the zeros
        // did. The first is cut whole, the second partly.
        let p = tl.place(end.round() as i64, 480, false, false);
        assert_eq!(p.skip, 480);
        let p = tl.place((end + 480.0 * HNS_PER_SAMPLE).round() as i64, 9_600, false, false);
        assert_eq!(p.zeros, 0);
        assert_eq!(p.skip, 4_800 - 480);
        assert_eq!(tl.position(), pos + 480 + 9_600);
        assert_eq!(tl.stats.zero_extensions, 1);
    }

    #[test]
    fn placement_follows_minus_51_6_ppm_without_jumps() {
        let mut mic = Timeline::default();
        let mut lb = Timeline::default();
        let (mut s1, mut s2, mut c1, mut c2) = (11, 13, 0, 0);
        let mut pl = Placement::default();
        let mut pos = vec![0.0; 160];
        let mut prev_end: Option<f64> = None;
        let mut max_step_err = 0f64;
        let mut t0 = 0u64;
        for _ in 0..600 {
            // 1 s of packets of each stream, then the mic blocks they cover.
            feed(&mut mic, 1e9, -51.6, &mut c1, 100, 2_000.0, &mut s1);
            feed(&mut lb, 1e9 + 30_000.0, 0.0, &mut c2, 100, 2_000.0, &mut s2);
            for _ in 0..100 {
                if pl.block(&mic.fit, &lb.fit, t0, &mut pos).is_some() {
                    if let Some(e) = prev_end {
                        max_step_err = max_step_err.max((pos[0] - e).abs());
                    }
                    prev_end = pl.next();
                }
                t0 += 160;
            }
        }
        // Continuous from block to block, the rate at the drift.
        assert!(max_step_err < 1e-6, "{max_step_err}");
        let ppm = (pl.rate() - 1.0) * 1e6;
        assert!((ppm + 51.6).abs() < 2.0, "{ppm}");
        assert_eq!(pl.jumps, 0);
        assert!(pl.last_error.abs() < 0.3, "{}", pl.last_error);
    }
}
