//! The signal chain after alignment: the canceller at 16 kHz, back to
//! 48 kHz, then RNNoise (`docs/design.md` section 3). No I/O and no clocks:
//! the engine feeds it one 10 ms block of mic and placed reference at a
//! time, live or in a replay, and unit tests feed it synthetic signals.

use crate::aec::{Aec, AecConfig, AecStats, ResidualConfig};
use crate::denoise::{self, Denoise};
use crate::resample::{Interp, INTERP_DELAY};

/// Canceller block: 10 ms at 16 kHz.
pub const BLOCK: usize = 160;
/// The same 10 ms at 48 kHz.
pub const BLOCK48: usize = 3 * BLOCK;

/// The A/B's recommended live canceller: Diarist's post-pass configuration
/// (10 ms blocks, 200 ms filter, residual suppression at its defaults) with
/// no lookahead (bulk delay 0).
pub fn live_aec_config() -> AecConfig {
    AecConfig {
        block: BLOCK,
        filter_len: 3_200,
        bulk_delay: 0,
        residual: Some(ResidualConfig::default()),
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ChainConfig {
    pub aec: AecConfig,
    /// Off: the mic passes the canceller's stage unchanged (delay-matched),
    /// for baseline runs.
    pub cancel: bool,
    pub rnnoise: bool,
}

impl Default for ChainConfig {
    fn default() -> Self {
        Self {
            aec: live_aec_config(),
            cancel: true,
            rnnoise: true,
        }
    }
}

pub struct Chain {
    cfg: ChainConfig,
    aec: Aec,
    /// Delay line standing in for the canceller when it is off.
    bypass: Vec<f32>,
    interp: Interp,
    denoise: Denoise,
    up: Vec<f32>,
}

impl Chain {
    pub fn new(cfg: ChainConfig) -> Result<Self, String> {
        let aec = Aec::new(cfg.aec)?;
        if cfg.aec.block != BLOCK {
            return Err(format!("the chain runs {BLOCK}-sample blocks"));
        }
        Ok(Self {
            bypass: vec![0.0; aec.latency()],
            aec,
            interp: Interp::new(),
            denoise: Denoise::new(cfg.rnnoise),
            up: Vec::with_capacity(4 * BLOCK48),
            cfg,
        })
    }

    pub fn config(&self) -> &ChainConfig {
        &self.cfg
    }

    pub fn aec_stats(&self) -> AecStats {
        self.aec.stats()
    }

    /// Lag of the canceller's output (16 kHz samples).
    pub fn aec_delay(&self) -> usize {
        self.aec.latency()
    }

    /// Lag of `out48` behind the mic, in 48 kHz samples on the 16 kHz
    /// timeline times three: canceller, interpolator and RNNoise.
    pub fn delay48(&self) -> usize {
        3 * (self.aec.latency() + INTERP_DELAY) + denoise::DELAY
    }

    /// Input samples (16 kHz) per output sample (48 kHz); `1/3` nominal.
    pub fn set_output_step(&mut self, step: f64) {
        self.interp.set_step(step);
    }

    /// One block: `mic` and `reference` ([`BLOCK`] samples each, the
    /// reference already on the mic's clock) into the canceller's output
    /// (`aec_out`, [`BLOCK`] samples) and whatever whole RNNoise frames that
    /// completes, appended to `out48` (one frame per block at the nominal
    /// step).
    pub fn process(
        &mut self,
        mic: &[f32],
        reference: &[f32],
        aec_out: &mut [f32],
        out48: &mut Vec<f32>,
    ) {
        if self.cfg.cancel {
            self.aec.process(mic, reference, aec_out);
        } else {
            self.bypass.extend_from_slice(mic);
            aec_out.copy_from_slice(&self.bypass[..BLOCK]);
            self.bypass.drain(..BLOCK);
        }
        self.interp.process(aec_out, &mut self.up);
        let whole = self.up.len() / denoise::FRAME * denoise::FRAME;
        for frame in self.up[..whole].chunks_exact_mut(denoise::FRAME) {
            self.denoise.process(frame);
            out48.extend_from_slice(frame);
        }
        self.up.drain(..whole);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn delays_add_up_to_the_designed_30_ms() {
        let c = Chain::new(ChainConfig::default()).unwrap();
        // 10 ms residual overlap + 1 ms interpolator + 10 ms RNNoise; the
        // block itself and the 2 ms decimator come on top (engine).
        assert_eq!(c.aec_delay(), 160);
        assert_eq!(c.delay48(), 3 * 176 + 480);
    }

    /// An impulse on the mic with a silent reference comes out at
    /// `delay48()`, with RNNoise off (it would gate a lone click).
    #[test]
    fn impulse_lands_at_the_reported_delay() {
        let mut c = Chain::new(ChainConfig {
            rnnoise: false,
            ..ChainConfig::default()
        })
        .unwrap();
        let mut mic = vec![0.0f32; 16_000];
        mic[4_000] = 0.5;
        let zeros = vec![0.0f32; BLOCK];
        let mut a = vec![0.0f32; BLOCK];
        let mut out = Vec::new();
        for m in mic.chunks(BLOCK) {
            c.process(m, &zeros, &mut a, &mut out);
        }
        assert_eq!(out.len(), 48_000);
        let peak = out
            .iter()
            .enumerate()
            .max_by(|x, y| x.1.abs().total_cmp(&y.1.abs()))
            .unwrap()
            .0;
        assert_eq!(peak, 3 * 4_000 + c.delay48());
    }
}
