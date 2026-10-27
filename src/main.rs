//! `clean-mic`: the stage B command line (`docs/design.md` section 9).
//!
//! ```text
//! clean-mic devices
//! clean-mic live   [--mic QuadCast] [--speakers NAME] [--seconds 180] [--out DIR]
//!                  [--no-aec] [--no-rnnoise] [--wait-ms 20] [--bulk-delay-ms 0]
//! clean-mic replay RUN_DIR [--out DIR] [--no-aec] [--no-rnnoise] [--wait-ms 20]
//!                  [--bulk-delay-ms 0] [--quiet]
//! clean-mic bench  [--seconds 60]
//! clean-mic wer    --passage docs/passages.md --span-from A.words.json B.words.json ...
//! ```

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::time::{Duration, Instant};

use clean_mic::chain::{live_aec_config, ChainConfig, BLOCK};
use clean_mic::engine::{Block, Engine, EngineConfig, EngineReport, Packet, Sink, Stream};
use clean_mic::resample::DOWN3_DELAY;
use clean_mic::run::{read_packets, RunWriter, WriterReport};
use clean_mic::wer;

/// The stage B gate: live far-end-only ERLE within 2 dB of the offline
/// A/B's 13.9 dB.
const GATE_ERLE_DB: f64 = 11.9;

static STOP: AtomicBool = AtomicBool::new(false);

type Res<T> = Result<T, String>;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let result = match args.first().map(String::as_str) {
        Some("devices") => devices(),
        Some("live") => live(&args[1..]),
        Some("replay") => replay(&args[1..]),
        Some("bench") => bench(&args[1..]),
        Some("wer") => wer_cmd(&args[1..]),
        _ => {
            eprintln!("{}", include_str!("usage.txt"));
            std::process::exit(2);
        }
    };
    if let Err(e) = result {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}

/// `--name value` options and positional arguments.
struct Opts {
    pairs: Vec<(String, Option<String>)>,
    positional: Vec<String>,
}

impl Opts {
    fn parse(args: &[String], flags: &[&str]) -> Self {
        let mut pairs = Vec::new();
        let mut positional = Vec::new();
        let mut i = 0;
        while i < args.len() {
            let a = &args[i];
            if let Some(name) = a.strip_prefix("--") {
                if flags.contains(&name) {
                    pairs.push((name.to_owned(), None));
                } else {
                    pairs.push((name.to_owned(), args.get(i + 1).cloned()));
                    i += 1;
                }
            } else {
                positional.push(a.clone());
            }
            i += 1;
        }
        Self { pairs, positional }
    }

    fn get(&self, name: &str) -> Option<&str> {
        self.pairs
            .iter()
            .rev()
            .find(|p| p.0 == name)
            .and_then(|p| p.1.as_deref())
    }

    fn flag(&self, name: &str) -> bool {
        self.pairs.iter().any(|p| p.0 == name)
    }

    fn num(&self, name: &str, default: f64) -> Res<f64> {
        self.get(name)
            .map_or(Ok(default), |v| v.parse().map_err(|e| format!("--{name} {v}: {e}")))
    }

    fn check(&self, known: &[&str]) -> Res<()> {
        for (n, _) in &self.pairs {
            if !known.contains(&n.as_str()) {
                return Err(format!("unknown option --{n}"));
            }
        }
        Ok(())
    }
}

fn engine_config(o: &Opts) -> Res<EngineConfig> {
    // Positive: the reference is delayed (outputs with long latency);
    // negative: the mic is (a lookahead, for echo ahead of the reference).
    let bulk_ms = o.num("bulk-delay-ms", 0.0)?;
    Ok(EngineConfig {
        chain: ChainConfig {
            aec: clean_mic::aec::AecConfig {
                bulk_delay: (bulk_ms * 16.0).round() as i32,
                ..live_aec_config()
            },
            cancel: !o.flag("no-aec"),
            rnnoise: !o.flag("no-rnnoise"),
        },
        wait_hns: o.num("wait-ms", 20.0)? * 1e4,
    })
}

/// Algorithmic latency: the 10 ms block, the decimator and the chain.
fn algorithmic_ms(eng: &Engine) -> f64 {
    10.0 + DOWN3_DELAY as f64 / 48.0 + eng.chain().delay48() as f64 / 48.0
}

fn unix_now() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_secs())
}

#[cfg(windows)]
fn devices() -> Res<()> {
    let list = clean_mic::capture::list_devices()?;
    for capture in [false, true] {
        println!("{}", if capture { "\nCapture" } else { "Playback" });
        for d in list.iter().filter(|d| d.capture == capture) {
            let mut tags = Vec::new();
            if d.default_console {
                tags.push("default");
            }
            if d.default_comms {
                tags.push("default communications");
            }
            println!(
                "  {}{}  [{}]",
                d.name,
                if tags.is_empty() {
                    String::new()
                } else {
                    format!(" ({})", tags.join(", "))
                },
                d.mix_format
            );
        }
    }
    Ok(())
}

#[cfg(not(windows))]
fn devices() -> Res<()> {
    Err("WASAPI devices exist on Windows only".into())
}

/// Sends the engine's callbacks to the writer thread.
struct ChannelSink(mpsc::Sender<Msg>);

enum Msg {
    Packet(Packet),
    Block {
        t0: u64,
        mic16: Vec<f32>,
        ref16: Vec<f32>,
        aec16: Vec<f32>,
        out48: Vec<f32>,
        dsp_us: f32,
        latency_ms: Option<f32>,
        drift_ppm: f64,
    },
}

impl Sink for ChannelSink {
    fn packet(&mut self, p: &Packet) {
        let _ = self.0.send(Msg::Packet(p.clone()));
    }

    fn block(&mut self, b: &Block<'_>) {
        let _ = self.0.send(Msg::Block {
            t0: b.t0,
            mic16: b.mic16.to_vec(),
            ref16: b.ref16.to_vec(),
            aec16: b.aec16.to_vec(),
            out48: b.out48.to_vec(),
            dsp_us: b.dsp_us,
            latency_ms: b.latency_ms,
            drift_ppm: b.drift_ppm,
        });
    }
}

fn writer_thread(mut w: RunWriter, rx: mpsc::Receiver<Msg>) -> std::thread::JoinHandle<Res<WriterReport>> {
    std::thread::spawn(move || {
        for m in rx {
            match m {
                Msg::Packet(p) => w.packet(&p),
                Msg::Block {
                    t0,
                    mic16,
                    ref16,
                    aec16,
                    out48,
                    dsp_us,
                    latency_ms,
                    drift_ppm,
                } => w.block(&Block {
                    t0,
                    mic16: &mic16,
                    ref16: &ref16,
                    aec16: &aec16,
                    out48: &out48,
                    dsp_us,
                    latency_ms,
                    drift_ppm,
                }),
            }
        }
        w.finish()
    })
}

#[derive(serde::Serialize)]
struct Report<'a> {
    tool: &'static str,
    version: &'static str,
    kind: &'a str,
    started_unix: u64,
    options: Vec<String>,
    devices: serde_json::Value,
    algorithmic_latency_ms: f64,
    wall_s: f64,
    process_cpu_s: Option<f64>,
    process_cpu_percent_of_core: Option<f64>,
    gate_erle_db: f64,
    engine: EngineReport,
    writer: &'a WriterReport,
}

fn summary(r: &Report) {
    let e = &r.writer.erle;
    let eng = &r.engine;
    println!();
    println!("Run: {:.0} s, {} blocks", r.writer.seconds, eng.blocks);
    match e.erle_aec_db {
        Some(v) => println!(
            "Far-end-only ERLE (canceller): {v:.1} dB over {} of {} active 2 s blocks; stage B gate {:.1} dB: {}",
            e.far_blocks,
            e.active_blocks,
            r.gate_erle_db,
            if v >= r.gate_erle_db { "PASS" } else { "below" }
        ),
        None => println!("Far-end-only ERLE: no far-end-only blocks (was anything playing?)"),
    }
    if let Some(v) = e.erle_out_db {
        println!("  after RNNoise: {v:.1} dB (includes the room floor RNNoise removes)");
    }
    if let Some(v) = e.echo_over_floor_db {
        println!("  echo over the room floor: {v:.1} dB (the most ERLE can show)");
    }
    if let Some(v) = e.mic_over_ref_db {
        println!("  mic over reference (median, active blocks): {v:.1} dB");
    }
    if r.writer.lag_ms.n > 0 {
        println!(
            "Echo lag: median {:.1} ms, max {:.1} ms over {} blocks",
            r.writer.lag_ms.p50, r.writer.lag_ms.max, r.writer.lag_ms.n
        );
    }
    println!(
        "Drift: mic {:.1} ppm, loopback {:.1} ppm against QPC; mic against loopback {:.1} ppm (research: -51.6)",
        eng.mic_drift_ppm, eng.loopback_drift_ppm, eng.mic_vs_loopback_ppm
    );
    println!(
        "DSP per 10 ms block: mean {:.0} us, p99 {:.0} us, max {:.0} us ({:.2} % of a core)",
        eng.dsp_us.mean,
        eng.dsp_us.p99,
        eng.dsp_us.max,
        eng.dsp_us.mean / 100.0
    );
    if let Some(p) = r.process_cpu_percent_of_core {
        println!("Whole process: {p:.2} % of one core (capture, processing, writing files)");
    }
    println!(
        "Latency: algorithmic {:.1} ms; capture to output mean {:.1} ms, p99 {:.1} ms, max {:.1} ms",
        r.algorithmic_latency_ms, eng.latency_ms.mean, eng.latency_ms.p99, eng.latency_ms.max
    );
    println!(
        "Reference: {} silent stretches ({} blocks), {} placement jumps; mic gaps {}, loopback gaps {}, overlaps {}/{}",
        eng.silence_stretches,
        eng.silence_blocks,
        eng.placement_jumps,
        eng.mic.gaps,
        eng.loopback.gaps,
        eng.mic.overlaps,
        eng.loopback.overlaps
    );
    println!(
        "Canceller: {} resets, leak {:.3}, adapted {}",
        eng.aec.resets, eng.aec.leak, eng.aec.adapted
    );
}

fn write_report(dir: &Path, r: &Report) -> Res<()> {
    let s = serde_json::to_string_pretty(r).map_err(|e| e.to_string())?;
    std::fs::write(dir.join("report.json"), s).map_err(|e| format!("report.json: {e}"))
}

fn default_runs_dir() -> PathBuf {
    std::env::var_os("LOCALAPPDATA")
        .map(PathBuf::from)
        .unwrap_or_else(std::env::temp_dir)
        .join("clean-mic")
        .join("runs")
}

#[cfg(windows)]
fn install_ctrl_c() {
    use windows::core::BOOL;
    use windows::Win32::System::Console::SetConsoleCtrlHandler;
    unsafe extern "system" fn handler(_: u32) -> BOOL {
        STOP.store(true, Ordering::SeqCst);
        BOOL(1)
    }
    // SAFETY: a static handler that only sets an atomic.
    let _ = unsafe { SetConsoleCtrlHandler(Some(handler), true) };
}

#[cfg(windows)]
fn live(args: &[String]) -> Res<()> {
    use clean_mic::capture::{self, DeviceSel};

    let o = Opts::parse(args, &["no-aec", "no-rnnoise"]);
    o.check(&["mic", "speakers", "seconds", "out", "no-aec", "no-rnnoise", "wait-ms", "bulk-delay-ms"])?;
    let cfg = engine_config(&o)?;
    let seconds = o.num("seconds", 180.0)?;
    let started_unix = unix_now();
    let dir = o.get("out").map_or_else(
        || default_runs_dir().join(format!("run-{started_unix}")),
        PathBuf::from,
    );
    let mut eng = Engine::new(cfg, true)?;
    let writer = RunWriter::create(&dir, true, true, eng.chain().aec_delay(), eng.chain().delay48())?;
    let (wtx, wrx) = mpsc::channel();
    let wthread = writer_thread(writer, wrx);
    let mut sink = ChannelSink(wtx);

    install_ctrl_c();
    let (ptx, prx) = mpsc::channel::<Packet>();
    let speakers = match o.get("speakers") {
        Some(n) => DeviceSel::RenderNamed(n.to_owned()),
        None => DeviceSel::DefaultRender,
    };
    let lb = capture::start(Stream::Loopback, speakers, ptx.clone())?;
    let mic = capture::start(
        Stream::Mic,
        DeviceSel::CaptureNamed(o.get("mic").unwrap_or("QuadCast").to_owned()),
        ptx,
    )?;
    println!("Recording into {}", dir.display());
    println!(
        "Mic: {} [{}], period {:.1} ms (minimum {:.1})",
        mic.opened.device_name, mic.opened.mix_format, mic.opened.period_ms.0, mic.opened.period_ms.1
    );
    println!(
        "Reference: loopback of {} [{}]",
        lb.opened.device_name, lb.opened.mix_format
    );
    println!("Running for {seconds:.0} s; Ctrl+C stops early. A line every 2 s:");
    clean_mic::pro_audio_thread();
    let t0 = Instant::now();
    let cpu0 = clean_mic::process_cpu_seconds();
    let mut failure = None;
    while !STOP.load(Ordering::SeqCst) && t0.elapsed().as_secs_f64() < seconds {
        match prx.recv_timeout(Duration::from_millis(100)) {
            Ok(p) => eng.packet(&p, &mut sink),
            Err(mpsc::RecvTimeoutError::Timeout) => {}
            Err(mpsc::RecvTimeoutError::Disconnected) => break,
        }
        if !mic.running() || !lb.running() {
            failure = mic.error().or_else(|| lb.error()).or(Some("a capture stopped".into()));
            break;
        }
    }
    let devices = serde_json::json!({ "mic": mic.opened, "loopback": lb.opened });
    mic.stop();
    lb.stop();
    // Packets already queued still belong to the run.
    while let Ok(p) = prx.try_recv() {
        eng.packet(&p, &mut sink);
    }
    let wall_s = t0.elapsed().as_secs_f64();
    let cpu = clean_mic::process_cpu_seconds().zip(cpu0).map(|(a, b)| a - b);
    let engine = eng.report();
    drop(sink);
    let writer = wthread.join().map_err(|_| "writer thread panicked")??;
    let report = Report {
        tool: "clean-mic",
        version: env!("CARGO_PKG_VERSION"),
        kind: "live",
        started_unix,
        options: args.to_vec(),
        devices,
        algorithmic_latency_ms: algorithmic_ms(&eng),
        wall_s,
        process_cpu_s: cpu,
        process_cpu_percent_of_core: cpu.map(|c| 100.0 * c / wall_s.max(1e-9)),
        gate_erle_db: GATE_ERLE_DB,
        engine,
        writer: &writer,
    };
    write_report(&dir, &report)?;
    summary(&report);
    println!("Files and report.json in {}", dir.display());
    match failure {
        Some(e) => Err(format!("stopped early: {e}")),
        None => Ok(()),
    }
}

#[cfg(not(windows))]
fn live(_: &[String]) -> Res<()> {
    Err("live capture needs Windows (WASAPI)".into())
}

fn replay(args: &[String]) -> Res<()> {
    let o = Opts::parse(args, &["no-aec", "no-rnnoise", "quiet"]);
    o.check(&["out", "no-aec", "no-rnnoise", "wait-ms", "bulk-delay-ms", "quiet"])?;
    let src = PathBuf::from(o.positional.first().ok_or("replay needs a run folder")?);
    let cfg = engine_config(&o)?;
    let started_unix = unix_now();
    let dir = o.get("out").map_or_else(
        || src.join(format!("replay-{started_unix}")),
        PathBuf::from,
    );
    let packets = read_packets(&src)?;
    let mut eng = Engine::new(cfg, false)?;
    let mut w = RunWriter::create(&dir, false, !o.flag("quiet"), eng.chain().aec_delay(), eng.chain().delay48())?;
    let t0 = Instant::now();
    let cpu0 = clean_mic::process_cpu_seconds();
    for p in &packets {
        eng.packet(p, &mut w);
    }
    let wall_s = t0.elapsed().as_secs_f64();
    let cpu = clean_mic::process_cpu_seconds().zip(cpu0).map(|(a, b)| a - b);
    let engine = eng.report();
    let writer = w.finish()?;
    let devices = std::fs::read(src.join("report.json"))
        .ok()
        .and_then(|b| serde_json::from_slice::<serde_json::Value>(&b).ok())
        .map(|v| v["devices"].clone())
        .unwrap_or_default();
    let report = Report {
        tool: "clean-mic",
        version: env!("CARGO_PKG_VERSION"),
        kind: "replay",
        started_unix,
        options: args.to_vec(),
        devices,
        algorithmic_latency_ms: algorithmic_ms(&eng),
        wall_s,
        process_cpu_s: cpu,
        process_cpu_percent_of_core: None,
        gate_erle_db: GATE_ERLE_DB,
        engine,
        writer: &writer,
    };
    write_report(&dir, &report)?;
    summary(&report);
    // A replay of an unchanged chain reproduces the live output exactly.
    if let (Ok(a), Ok(b)) = (read16(&src.join("out48.wav")), read16(&dir.join("out48.wav"))) {
        let n = a.len().min(b.len());
        let diff = a[..n].iter().zip(&b[..n]).map(|(x, y)| (x - y).abs()).max().unwrap_or(0);
        println!(
            "Against the run's own out48.wav: {} samples compared, largest difference {diff} (PCM16 steps)",
            n
        );
    }
    println!("Replay written to {}", dir.display());
    Ok(())
}

fn read16(p: &Path) -> Res<Vec<i32>> {
    let mut r = hound::WavReader::open(p).map_err(|e| e.to_string())?;
    r.samples::<i16>()
        .map(|s| s.map(i32::from).map_err(|e| e.to_string()))
        .collect()
}

/// The chain's CPU cost on the synthetic session (`clean_mic::synth`):
/// speech-like far end through a short echo path with -51.6 ppm drift, the
/// owner's bursts, a stretch with no loopback; packets as devices deliver
/// them. Times the engine only, not the signal generation.
fn bench(args: &[String]) -> Res<()> {
    let o = Opts::parse(args, &[]);
    o.check(&["seconds"])?;
    let secs = o.num("seconds", 60.0)?;
    let packets = clean_mic::synth::session(secs);
    let mut eng = Engine::new(EngineConfig::default(), false)?;
    struct Nop(Vec<f32>);
    impl Sink for Nop {
        fn block(&mut self, b: &Block<'_>) {
            self.0.push(b.dsp_us);
        }
    }
    let mut sink = Nop(Vec::new());
    let started = Instant::now();
    for p in &packets {
        eng.packet(p, &mut sink);
    }
    let wall = started.elapsed().as_secs_f64();
    let s = clean_mic::stats::Summary::of(&sink.0);
    println!(
        "{} blocks of 10 ms: DSP mean {:.1} us, p50 {:.1}, p99 {:.1}, max {:.1} ({:.2} % of a core); engine wall time {:.2} s",
        s.n,
        s.mean,
        s.p50,
        s.p99,
        s.max,
        s.mean / 100.0,
        wall
    );
    println!("Algorithmic latency: {:.1} ms", algorithmic_ms(&eng));
    println!("Block size {BLOCK} samples at 16 kHz; drift {:.1} ppm", eng.drift_ppm());
    Ok(())
}

fn wer_cmd(args: &[String]) -> Res<()> {
    let o = Opts::parse(args, &[]);
    o.check(&["passage", "span-from"])?;
    let text = std::fs::read_to_string(o.get("passage").unwrap_or("docs/passages.md"))
        .map_err(|e| format!("passage: {e}"))?;
    let reference = wer::passage(&text);
    let span = match o.get("span-from") {
        Some(p) => {
            let words = wer::load_words(Path::new(p))?;
            let s = wer::find_span(&reference, &words).ok_or("the passage was not found in --span-from")?;
            println!("Reading span from {p}: {:.1} to {:.1} s", s.0, s.1);
            Some(s)
        }
        None => None,
    };
    let files: Vec<&str> = o.positional.iter().map(String::as_str).collect();
    println!("{:<40} {:>5} {:>5} {:>5} {:>5} {:>7}", "transcript", "words", "subs", "dels", "ins", "WER");
    for f in files {
        let words = wer::load_words(Path::new(f))?;
        let hyp = match span {
            Some((a, b)) => wer::within(&words, a - 0.5, b + 0.5),
            None => words.into_iter().map(|w| w.2).collect(),
        };
        let e = wer::errors(&reference, &hyp);
        println!(
            "{:<40} {:>5} {:>5} {:>5} {:>5} {:>6.1} %",
            Path::new(f).file_name().map_or(f.into(), |n| n.to_string_lossy()),
            e.reference,
            e.substitutions,
            e.deletions,
            e.insertions,
            100.0 * e.wer()
        );
    }
    Ok(())
}
