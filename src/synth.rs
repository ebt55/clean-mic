//! A synthetic session for tests and `clean-mic bench`: no device needed.
//!
//! The far end is speech-like noise bursts at about -26 dBFS. The mic hears
//! it through a short echo path (3 ms direct, 15 ms reflection, -10 dB) on
//! its own clock (-51.6 ppm against the loopback, as measured on the owner's
//! PC), plus the owner's bursts for 4 s in every 20 s and a -70 dBFS floor.
//! From 40 to 50 s nothing plays and the loopback delivers no packets.
//! Packets of 480 samples arrive 10 ms after their first frame, interleaved
//! by arrival time.

use crate::aec::SincInterpolator;
use crate::clock::HNS_PER_SAMPLE;
use crate::engine::{Packet, Stream};

/// Mic clock against the loopback's.
pub const DRIFT_PPM: f64 = -51.6;
/// Echo delay (48 kHz samples): 3 ms.
pub const ECHO_DELAY: f64 = 144.0;
/// Where the loopback is silent (no packets), seconds.
pub const SILENT: (f64, f64) = (40.0, 50.0);

struct Rng(u32);

impl Rng {
    fn next(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 17;
        self.0 ^= self.0 << 5;
        self.0 as f32 / u32::MAX as f32 * 2.0 - 1.0
    }
}

/// Noise bursts through a resonance, on 60 % of each 0.5 s, RMS about
/// `dbfs` while on; `active(t)` gates them.
fn bursts(n: usize, seed: u32, dbfs: f32, active: impl Fn(f64) -> bool) -> Vec<f32> {
    let mut rng = Rng(seed);
    let amp = 10f32.powf(dbfs / 20.0) * 3.5;
    let (mut y1, mut y2) = (0f32, 0f32);
    (0..n)
        .map(|i| {
            let t = i as f64 / 48_000.0;
            let f = 600.0 + 400.0 * (t * 1.7).sin();
            let w = std::f32::consts::TAU * f as f32 / 48_000.0;
            let r = 0.98f32;
            let v = rng.next() + 2.0 * r * w.cos() * y1 - r * r * y2;
            y2 = y1;
            y1 = v;
            let on = (t * 2.0).fract() < 0.6 && active(t);
            if on {
                v * amp * (1.0 - r)
            } else {
                0.0
            }
        })
        .collect()
}

pub fn far_end_playing(t: f64) -> bool {
    !(SILENT.0..SILENT.1).contains(&t)
}

pub fn owner_talking(t: f64) -> bool {
    (12.0..16.0).contains(&(t % 20.0))
}

/// The packets of a `secs`-second session, in arrival order.
pub fn session(secs: f64) -> Vec<Packet> {
    let n = (secs * 48_000.0) as usize;
    let far = bursts(n + 48_000, 0x1357_9BDF, -26.0, far_end_playing);
    let near = bursts(n, 0x2468_ACE0, -30.0, owner_talking);
    let mut rng = Rng(0x0F0F_0F0F);
    let it = SincInterpolator::new(16, 512, 0.9);
    let mic_hns = HNS_PER_SAMPLE * (1.0 + DRIFT_PPM * 1e-6);
    let origin = 1e10;
    let mut out = Vec::new();
    let (mut mi, mut li) = (0usize, 0usize);
    while mi + 480 <= n {
        let mq = origin + mi as f64 * mic_hns;
        let lq = origin + li as f64 * HNS_PER_SAMPLE;
        if lq <= mq && li + 480 <= far.len() {
            if far_end_playing(li as f64 / 48_000.0) {
                let qpc = lq.round() as i64;
                out.push(Packet {
                    stream: Some(Stream::Loopback),
                    qpc,
                    arrival: qpc + 100_000,
                    samples: far[li..li + 480].to_vec(),
                    ..Packet::default()
                });
            }
            li += 480;
        } else {
            let samples: Vec<f32> = (mi..mi + 480)
                .map(|m| {
                    // Mic sample m is loopback time m * (1 + drift).
                    let t = m as f64 * mic_hns / HNS_PER_SAMPLE;
                    let echo = 0.3 * it.at(&far, t - ECHO_DELAY) + 0.1 * it.at(&far, t - 720.0);
                    echo + near[m] + 3e-4 * 1.7 * rng.next()
                })
                .collect();
            let qpc = mq.round() as i64;
            out.push(Packet {
                stream: Some(Stream::Mic),
                qpc,
                arrival: qpc + 100_000,
                samples,
                ..Packet::default()
            });
            mi += 480;
        }
    }
    out.sort_by_key(|p| p.arrival);
    out
}
