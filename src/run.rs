//! A run folder: what `clean-mic live` records and `clean-mic replay`
//! reads back.
//!
//! Inputs, exactly as captured (for replays):
//! - `mic-raw.wav`, `loopback-raw.wav`: every packet's samples in arrival
//!   order, 48 kHz mono f32, no fills;
//! - `packets.csv`: one line per packet in the order the engine got it.
//!
//! Outputs, aligned with each other:
//! - `mic16.wav`, `ref16.wav`, `aec16.wav`, `out16.wav`: the decimated mic,
//!   the reference placed on its clock, the canceller's output and the
//!   final output brought back to 16 kHz, sample-aligned (16 kHz PCM16);
//! - `out48.wav`: the final output at 48 kHz, aligned with `mic-raw.wav`
//!   while the mic has no gaps (PCM16).
//!
//! `report.json` holds the engine's, the ERLE's and the run's figures.

use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};

use crate::chain::BLOCK;
use crate::engine::{Block, Packet, Sink, Stream};
use crate::erle::{BlockEnergy, Erle, ErleResult, LagEstimator, ERLE_BLOCK};
use crate::resample::{Down3, DOWN3_DELAY};
use crate::stats::{median, Summary};

type Wav = hound::WavWriter<BufWriter<File>>;

fn wav(path: &Path, rate: u32, float: bool) -> Result<Wav, String> {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: rate,
        bits_per_sample: if float { 32 } else { 16 },
        sample_format: if float {
            hound::SampleFormat::Float
        } else {
            hound::SampleFormat::Int
        },
    };
    hound::WavWriter::create(path, spec).map_err(|e| format!("create {}: {e}", path.display()))
}

fn put16(w: &mut Option<Wav>, x: &[f32]) {
    if let Some(w) = w {
        for &v in x {
            let _ = w.write_sample((v.clamp(-1.0, 1.0) * 32_767.0).round() as i16);
        }
    }
}

fn put32(w: &mut Option<Wav>, x: &[f32]) {
    if let Some(w) = w {
        for &v in x {
            let _ = w.write_sample(v);
        }
    }
}

/// A queue that drops its first `skip` samples (delay compensation).
#[derive(Default)]
struct Aligned {
    skip: usize,
    data: Vec<f32>,
}

impl Aligned {
    fn new(skip: usize) -> Self {
        Self {
            skip,
            data: Vec::new(),
        }
    }

    fn push(&mut self, x: &[f32]) {
        let s = self.skip.min(x.len());
        self.skip -= s;
        self.data.extend_from_slice(&x[s..]);
    }
}

/// The ERLE block line printed and kept per 2 s.
#[derive(Debug, Clone, serde::Serialize)]
pub struct BlockLine {
    pub t_s: f64,
    pub energy: BlockEnergy,
    pub far_only: bool,
    pub lag_ms: Option<f64>,
    pub lag_peak: Option<f64>,
    pub dsp_us_mean: f64,
    pub dsp_us_max: f64,
    pub latency_ms: Option<f64>,
    pub drift_ppm: f64,
}

/// The writer's share of the report.
#[derive(Debug, Clone, Default, serde::Serialize)]
pub struct WriterReport {
    pub seconds: f64,
    pub erle: ErleResult,
    /// Echo lag of the far-end-only blocks with a clear peak (over 0.1).
    pub lag_ms: Summary,
    pub blocks: Vec<BlockLine>,
}

/// Writes a run folder from engine callbacks; prints a line per 2 s.
pub struct RunWriter {
    dir: PathBuf,
    console: bool,
    mic_raw: Option<Wav>,
    lb_raw: Option<Wav>,
    csv: Option<BufWriter<File>>,
    seq: u64,
    mic16: Option<Wav>,
    ref16: Option<Wav>,
    aec16: Option<Wav>,
    out16: Option<Wav>,
    out48: Option<Wav>,
    q_mic: Aligned,
    q_ref: Aligned,
    q_aec: Aligned,
    q_out: Aligned,
    q_out48: Aligned,
    down_out: Down3,
    tmp: Vec<f32>,
    erle: Erle,
    lag: LagEstimator,
    blk_mic: Vec<f32>,
    blk_ref: Vec<f32>,
    aligned_samples: u64,
    dsp_win: Vec<f32>,
    lat_win: Vec<f32>,
    drift: f64,
    lines: Vec<BlockLine>,
}

impl RunWriter {
    /// `record_packets`: write the raw inputs (live runs; a replay has
    /// them already). `aec_delay` and `delay48` are the chain's.
    pub fn create(dir: &Path, record_packets: bool, console: bool, aec_delay: usize, delay48: usize) -> Result<Self, String> {
        std::fs::create_dir_all(dir).map_err(|e| format!("create {}: {e}", dir.display()))?;
        let (mic_raw, lb_raw, csv) = if record_packets {
            let mut csv = BufWriter::new(
                File::create(dir.join("packets.csv")).map_err(|e| format!("packets.csv: {e}"))?,
            );
            writeln!(csv, "seq,stream,frames,qpc,arrival,silent,discontinuity,timestamp_error")
                .map_err(|e| e.to_string())?;
            (
                Some(wav(&dir.join("mic-raw.wav"), 48_000, true)?),
                Some(wav(&dir.join("loopback-raw.wav"), 48_000, true)?),
                Some(csv),
            )
        } else {
            (None, None, None)
        };
        // out48[k] is mic16 time (k - delay48) / 3, which is mic48 sample
        // k - delay48 - DOWN3_DELAY; out16[n] is centred on out48 sample
        // 3 n - DOWN3_DELAY, mic16 time n - (delay48 + DOWN3_DELAY) / 3.
        let skip16 = (delay48 + DOWN3_DELAY).div_ceil(3);
        Ok(Self {
            console,
            mic_raw,
            lb_raw,
            csv,
            seq: 0,
            mic16: Some(wav(&dir.join("mic16.wav"), 16_000, false)?),
            ref16: Some(wav(&dir.join("ref16.wav"), 16_000, false)?),
            aec16: Some(wav(&dir.join("aec16.wav"), 16_000, false)?),
            out16: Some(wav(&dir.join("out16.wav"), 16_000, false)?),
            out48: Some(wav(&dir.join("out48.wav"), 48_000, false)?),
            q_mic: Aligned::new(0),
            q_ref: Aligned::new(0),
            q_aec: Aligned::new(aec_delay),
            q_out: Aligned::new(skip16),
            q_out48: Aligned::new(delay48 + DOWN3_DELAY),
            down_out: Down3::new(),
            tmp: Vec::with_capacity(1_024),
            erle: Erle::default(),
            lag: LagEstimator::default(),
            blk_mic: Vec::with_capacity(ERLE_BLOCK),
            blk_ref: Vec::with_capacity(ERLE_BLOCK),
            aligned_samples: 0,
            dsp_win: Vec::new(),
            lat_win: Vec::new(),
            drift: 0.0,
            lines: Vec::new(),
            dir: dir.to_path_buf(),
        })
    }

    pub fn dir(&self) -> &Path {
        &self.dir
    }

    fn drain_aligned(&mut self) {
        loop {
            let n = self
                .q_mic
                .data
                .len()
                .min(self.q_ref.data.len())
                .min(self.q_aec.data.len())
                .min(self.q_out.data.len());
            if n < BLOCK {
                break;
            }
            let (m, r, a, o) = (
                &self.q_mic.data[..BLOCK],
                &self.q_ref.data[..BLOCK],
                &self.q_aec.data[..BLOCK],
                &self.q_out.data[..BLOCK],
            );
            put16(&mut self.mic16, m);
            put16(&mut self.ref16, r);
            put16(&mut self.aec16, a);
            put16(&mut self.out16, o);
            self.blk_mic.extend_from_slice(m);
            self.blk_ref.extend_from_slice(r);
            let done = self.erle.push(m, r, a, o);
            self.aligned_samples += BLOCK as u64;
            for q in [&mut self.q_mic, &mut self.q_ref, &mut self.q_aec, &mut self.q_out] {
                q.data.drain(..BLOCK);
            }
            if let Some(b) = done {
                self.block_done(b);
            }
        }
    }

    fn block_done(&mut self, b: BlockEnergy) {
        let far_only = self.erle.is_far_only(&b);
        let lag = if b.active() {
            self.lag.estimate(&self.blk_mic, &self.blk_ref)
        } else {
            None
        };
        self.blk_mic.clear();
        self.blk_ref.clear();
        let dsp = Summary::of(&self.dsp_win);
        let lat = Summary::of(&self.lat_win);
        self.dsp_win.clear();
        self.lat_win.clear();
        let line = BlockLine {
            t_s: self.aligned_samples as f64 / 16_000.0,
            energy: b,
            far_only,
            lag_ms: lag.map(|l| l.0),
            lag_peak: lag.map(|l| l.1),
            dsp_us_mean: dsp.mean,
            dsp_us_max: dsp.max,
            latency_ms: (lat.n > 0).then_some(lat.mean),
            drift_ppm: self.drift,
        };
        if self.console {
            let erle = self.erle.result();
            let state = if far_only {
                format!(
                    "far end  ERLE {:5.1} dB (+RNNoise {:5.1})",
                    b.erle_aec_db(),
                    b.erle_out_db()
                )
            } else if b.active() {
                "double talk / far end loud      ".to_owned()
            } else {
                "no far end                      ".to_owned()
            };
            let run = erle
                .erle_aec_db
                .map_or("   -".to_owned(), |v| format!("{v:4.1}"));
            let lag = match line.lag_ms.zip(line.lag_peak) {
                Some((l, p)) if p > 0.1 => format!("{l:6.1} ms"),
                _ => "     - ".to_owned(),
            };
            println!(
                "{:6.0} s  ref {:6.1} dBFS  mic {:6.1} dBFS  {state}  run {run} dB  lag {lag}  drift {:6.1} ppm  dsp {:4.0}/{:4.0} us  latency {}",
                line.t_s,
                b.ref_dbfs(),
                b.mic_dbfs(),
                self.drift,
                dsp.mean,
                dsp.max,
                line.latency_ms.map_or("-".to_owned(), |l| format!("{l:.0} ms")),
            );
        }
        self.lines.push(line);
    }

    /// Closes the files and returns the writer's figures.
    pub fn finish(mut self) -> Result<WriterReport, String> {
        // Whatever is left of a 2 s block still counts for nothing; the
        // aligned files simply end at the shortest stream.
        self.drain_aligned();
        if let Some(mut c) = self.csv.take() {
            c.flush().map_err(|e| e.to_string())?;
        }
        for w in [
            self.mic_raw.take(),
            self.lb_raw.take(),
            self.mic16.take(),
            self.ref16.take(),
            self.aec16.take(),
            self.out16.take(),
            self.out48.take(),
        ]
        .into_iter()
        .flatten()
        {
            w.finalize().map_err(|e| e.to_string())?;
        }
        let lags: Vec<f32> = self
            .lines
            .iter()
            .filter(|l| l.far_only && l.lag_peak.is_some_and(|p| p > 0.1))
            .filter_map(|l| l.lag_ms.map(|v| v as f32))
            .collect();
        Ok(WriterReport {
            seconds: self.aligned_samples as f64 / 16_000.0,
            erle: self.erle.result(),
            lag_ms: Summary::of(&lags),
            blocks: self.lines,
        })
    }
}

impl Sink for RunWriter {
    fn packet(&mut self, p: &Packet) {
        let Some(stream) = p.stream else { return };
        let w = match stream {
            Stream::Mic => &mut self.mic_raw,
            Stream::Loopback => &mut self.lb_raw,
        };
        put32(w, &p.samples);
        if let Some(c) = &mut self.csv {
            let _ = writeln!(
                c,
                "{},{},{},{},{},{},{},{}",
                self.seq,
                match stream {
                    Stream::Mic => "mic",
                    Stream::Loopback => "loopback",
                },
                p.samples.len(),
                p.qpc,
                p.arrival,
                u8::from(p.silent),
                u8::from(p.discontinuity),
                u8::from(p.timestamp_error),
            );
        }
        self.seq += 1;
    }

    fn block(&mut self, b: &Block<'_>) {
        self.q_out48.push(b.out48);
        if !self.q_out48.data.is_empty() {
            put16(&mut self.out48, &self.q_out48.data);
            self.q_out48.data.clear();
        }
        self.tmp.clear();
        self.down_out.process(b.out48, &mut self.tmp);
        self.q_out.push(&self.tmp);
        self.q_mic.push(b.mic16);
        self.q_ref.push(b.ref16);
        self.q_aec.push(b.aec16);
        self.dsp_win.push(b.dsp_us);
        if let Some(l) = b.latency_ms {
            self.lat_win.push(l);
        }
        self.drift = b.drift_ppm;
        self.drain_aligned();
    }
}

/// Reads a run folder's packets back, in the order the engine got them.
pub fn read_packets(dir: &Path) -> Result<Vec<Packet>, String> {
    let read_wav = |name: &str| -> Result<Vec<f32>, String> {
        let p = dir.join(name);
        let mut r = hound::WavReader::open(&p).map_err(|e| format!("{}: {e}", p.display()))?;
        r.samples::<f32>()
            .collect::<Result<Vec<f32>, _>>()
            .map_err(|e| format!("{}: {e}", p.display()))
    };
    let mic = read_wav("mic-raw.wav")?;
    let lb = read_wav("loopback-raw.wav")?;
    let csv = File::open(dir.join("packets.csv")).map_err(|e| format!("packets.csv: {e}"))?;
    let (mut mi, mut li) = (0usize, 0usize);
    let mut out = Vec::new();
    for (k, line) in BufReader::new(csv).lines().enumerate() {
        let line = line.map_err(|e| e.to_string())?;
        if k == 0 {
            continue;
        }
        let f: Vec<&str> = line.split(',').collect();
        if f.len() != 8 {
            return Err(format!("packets.csv line {}: {} fields", k + 1, f.len()));
        }
        let num = |i: usize| -> Result<i64, String> {
            f[i].parse::<i64>()
                .map_err(|e| format!("packets.csv line {}: {e}", k + 1))
        };
        let frames = num(2)? as usize;
        let (stream, src, at) = match f[1] {
            "mic" => (Stream::Mic, &mic, &mut mi),
            "loopback" => (Stream::Loopback, &lb, &mut li),
            s => return Err(format!("packets.csv line {}: stream {s}", k + 1)),
        };
        let samples = src
            .get(*at..*at + frames)
            .ok_or_else(|| format!("packets.csv line {}: past the end of the WAV", k + 1))?
            .to_vec();
        *at += frames;
        out.push(Packet {
            stream: Some(stream),
            qpc: num(3)?,
            arrival: num(4)?,
            silent: num(5)? != 0,
            discontinuity: num(6)? != 0,
            timestamp_error: num(7)? != 0,
            samples,
        });
    }
    Ok(out)
}

/// Median of a series of optional values (report helper).
pub fn median_of(v: impl Iterator<Item = f64>) -> Option<f64> {
    let v: Vec<f64> = v.collect();
    median(&v)
}
