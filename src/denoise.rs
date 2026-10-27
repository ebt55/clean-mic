//! RNNoise (nnnoiseless 0.5.2) after the canceller, at 48 kHz in 480-sample
//! (10 ms) frames. It removes most of the canceller's leftover echo words
//! (the A/B: 2,493 to 658 in 111 minutes) for about a point of WER on the
//! owner's words. Its output lags its input by one frame.

use nnnoiseless::DenoiseState;

/// Samples per RNNoise frame (10 ms at 48 kHz).
pub const FRAME: usize = DenoiseState::FRAME_SIZE;
/// Output lag (48 kHz samples), measured in the A/B with a probe.
pub const DELAY: usize = FRAME;

/// RNNoise, or a matching one-frame delay when off, so that switching it
/// never moves the output in time.
pub struct Denoise {
    state: Box<DenoiseState<'static>>,
    on: bool,
    inb: Vec<f32>,
    outb: Vec<f32>,
    delay: Vec<f32>,
    /// Last frame's voice probability.
    vad: f32,
}

impl Denoise {
    pub fn new(on: bool) -> Self {
        Self {
            state: DenoiseState::new(),
            on,
            inb: vec![0.0; FRAME],
            outb: vec![0.0; FRAME],
            delay: vec![0.0; FRAME],
            vad: 0.0,
        }
    }

    pub fn is_on(&self) -> bool {
        self.on
    }

    pub fn vad(&self) -> f32 {
        self.vad
    }

    /// One frame in place (full scale 1.0).
    ///
    /// # Panics
    ///
    /// When `frame` is not [`FRAME`] samples long.
    pub fn process(&mut self, frame: &mut [f32]) {
        assert_eq!(frame.len(), FRAME, "RNNoise frames are {FRAME} samples");
        if self.on {
            // RNNoise works in PCM16 units.
            for (d, &s) in self.inb.iter_mut().zip(frame.iter()) {
                *d = s * 32_768.0;
            }
            self.vad = self.state.process_frame(&mut self.outb, &self.inb);
            for (d, &s) in frame.iter_mut().zip(&self.outb) {
                *d = (s / 32_768.0).clamp(-1.0, 1.0);
            }
        } else {
            for (d, s) in frame.iter_mut().zip(self.delay.iter_mut()) {
                std::mem::swap(d, s);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn off_is_a_one_frame_delay() {
        let mut d = Denoise::new(false);
        let a: Vec<f32> = (0..FRAME).map(|i| i as f32 / 1e4).collect();
        let mut f = a.clone();
        d.process(&mut f);
        assert!(f.iter().all(|&v| v == 0.0));
        let mut g = vec![0.5; FRAME];
        d.process(&mut g);
        assert_eq!(g, a);
    }

    /// Fan-like noise (white noise through a one-pole low-pass, about
    /// -45 dBFS) drops by far more than 20 dB. Full-band white noise would
    /// not: RNNoise takes it for voice (VAD about 0.9, 1 dB less).
    #[test]
    fn on_lowers_steady_noise() {
        let mut d = Denoise::new(true);
        let mut seed = 0x1234_5678u32;
        let mut y = 0f32;
        let mut e_in = 0f64;
        let mut e_out = 0f64;
        for k in 0..300 {
            let mut f: Vec<f32> = (0..FRAME)
                .map(|_| {
                    seed ^= seed << 13;
                    seed ^= seed >> 17;
                    seed ^= seed << 5;
                    y = 0.95 * y + 0.05 * (seed as f32 / u32::MAX as f32 - 0.5) * 0.1;
                    y
                })
                .collect();
            let ein: f64 = f.iter().map(|&v| f64::from(v * v)).sum();
            d.process(&mut f);
            if k >= 100 {
                e_in += ein;
                e_out += f.iter().map(|&v| f64::from(v * v)).sum::<f64>();
            }
        }
        let db = 10.0 * (e_in / e_out.max(1e-20)).log10();
        assert!(db > 20.0, "noise reduced by only {db:.1} dB");
    }
}
