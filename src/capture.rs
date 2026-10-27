//! WASAPI capture of the mic and of the default playback's loopback, after
//! Diarist's `audio/capture` (MIT, see `NOTICE.md`): shared mode, polled,
//! every packet with its QPC timestamp and its arrival time. Both streams
//! are opened at 48 kHz mono f32 with the engine's format conversion, which
//! for the QuadCast and the speakers is a downmix only (both run at 48 kHz).
//!
//! Opening a stream here records audio: only `clean-mic live` calls
//! [`start`], and only when the owner runs it.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Sender};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::Duration;

use wasapi::{initialize_mta, DeviceEnumerator, Direction, Role, SampleType, StreamMode, WaveFormat};

use crate::engine::{Packet, Stream};

/// Shared-mode buffer asked for (100 ms); the loop polls every 2 ms.
const BUFFER_HNS: i64 = 1_000_000;
const POLL: Duration = Duration::from_millis(2);

#[derive(Debug, Clone, serde::Serialize)]
pub struct DeviceInfo {
    pub id: String,
    pub name: String,
    pub capture: bool,
    pub default_console: bool,
    pub default_comms: bool,
    /// The endpoint's shared-mode mix format, e.g. "48000 Hz, 2 ch, 16 bit
    /// int".
    pub mix_format: String,
}

/// Which endpoint a capture opens.
#[derive(Debug, Clone)]
pub enum DeviceSel {
    /// A capture endpoint whose name contains this (case-insensitive).
    CaptureNamed(String),
    /// The default console playback endpoint, as loopback.
    DefaultRender,
    /// A playback endpoint whose name contains this, as loopback.
    RenderNamed(String),
}

/// What was opened.
#[derive(Debug, Clone, Default, serde::Serialize)]
pub struct Opened {
    pub device_id: String,
    pub device_name: String,
    pub mix_format: String,
    /// Default and minimum device period, ms.
    pub period_ms: (f64, f64),
    pub buffer_frames: u32,
}

fn describe(f: &WaveFormat) -> String {
    let kind = match f.get_subformat() {
        Ok(SampleType::Float) => "float",
        Ok(SampleType::Int) => "int",
        Err(_) => "?",
    };
    format!(
        "{} Hz, {} ch, {} bit {kind}",
        f.get_samplespersec(),
        f.get_nchannels(),
        f.get_validbitspersample()
    )
}

/// Runs `f` on a fresh MTA thread (COM may be STA on the caller's).
fn on_mta<T: Send + 'static>(f: impl FnOnce() -> Result<T, String> + Send + 'static) -> Result<T, String> {
    std::thread::spawn(move || {
        let _ = initialize_mta();
        f()
    })
    .join()
    .map_err(|_| "audio thread panicked".to_owned())?
}

/// Lists playback and capture endpoints with their defaults and mix
/// formats. Opens no stream.
pub fn list_devices() -> Result<Vec<DeviceInfo>, String> {
    on_mta(|| {
        let en = DeviceEnumerator::new().map_err(|e| format!("device enumerator: {e}"))?;
        let mut out = Vec::new();
        for (dir, capture) in [(Direction::Render, false), (Direction::Capture, true)] {
            let default_of = |role: Role| {
                en.get_default_device_for_role(&dir, &role)
                    .and_then(|d| d.get_id())
                    .unwrap_or_default()
            };
            let console = default_of(Role::Console);
            let comms = default_of(Role::Communications);
            let coll = en
                .get_device_collection(&dir)
                .map_err(|e| format!("device list: {e}"))?;
            for i in 0..coll.get_nbr_devices().unwrap_or(0) {
                let Ok(d) = coll.get_device_at_index(i) else {
                    continue;
                };
                let Ok(id) = d.get_id() else { continue };
                let mix_format = d
                    .get_device_format()
                    .map(|f| describe(&f))
                    .unwrap_or_else(|e| format!("({e})"));
                out.push(DeviceInfo {
                    name: d.get_friendlyname().unwrap_or_else(|_| id.clone()),
                    default_console: id == console,
                    default_comms: id == comms,
                    capture,
                    mix_format,
                    id,
                });
            }
        }
        Ok(out)
    })
}

/// A running capture thread.
pub struct Capture {
    stop: Arc<AtomicBool>,
    error: Arc<Mutex<Option<String>>>,
    thread: Option<JoinHandle<()>>,
    pub opened: Opened,
}

impl Capture {
    /// The error that ended the thread, if any.
    pub fn error(&self) -> Option<String> {
        self.error.lock().ok().and_then(|e| e.clone())
    }

    pub fn running(&self) -> bool {
        self.thread.as_ref().is_some_and(|t| !t.is_finished())
    }

    pub fn stop(mut self) {
        self.join();
    }

    fn join(&mut self) {
        self.stop.store(true, Ordering::SeqCst);
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
    }
}

impl Drop for Capture {
    fn drop(&mut self) {
        self.join();
    }
}

/// Opens `sel` and starts sending its packets, tagged `stream`, to `tx`.
pub fn start(stream: Stream, sel: DeviceSel, tx: Sender<Packet>) -> Result<Capture, String> {
    let stop = Arc::new(AtomicBool::new(false));
    let error = Arc::new(Mutex::new(None));
    let (ready_tx, ready_rx) = mpsc::channel::<Result<Opened, String>>();
    let (st, er) = (Arc::clone(&stop), Arc::clone(&error));
    let thread = std::thread::Builder::new()
        .name(format!("capture-{stream:?}"))
        .spawn(move || {
            let _ = initialize_mta();
            crate::pro_audio_thread();
            if let Err(e) = run(stream, &sel, &tx, &st, &ready_tx) {
                if let Ok(mut g) = er.lock() {
                    *g = Some(e.clone());
                }
                let _ = ready_tx.send(Err(e));
            }
        })
        .map_err(|e| format!("spawn: {e}"))?;
    let mut cap = Capture {
        stop,
        error,
        thread: Some(thread),
        opened: Opened::default(),
    };
    match ready_rx.recv_timeout(Duration::from_secs(10)) {
        Ok(Ok(opened)) => {
            cap.opened = opened;
            Ok(cap)
        }
        Ok(Err(e)) => Err(e),
        Err(_) => Err(format!("{stream:?} capture did not start within 10 s")),
    }
}

fn run(
    stream: Stream,
    sel: &DeviceSel,
    tx: &Sender<Packet>,
    stop: &AtomicBool,
    ready: &mpsc::Sender<Result<Opened, String>>,
) -> Result<(), String> {
    let en = DeviceEnumerator::new().map_err(|e| format!("device enumerator: {e}"))?;
    let find = |dir: Direction, part: &str| -> Result<wasapi::Device, String> {
        let coll = en
            .get_device_collection(&dir)
            .map_err(|e| format!("device list: {e}"))?;
        let want = part.to_lowercase();
        for i in 0..coll.get_nbr_devices().unwrap_or(0) {
            if let Ok(d) = coll.get_device_at_index(i) {
                if d.get_friendlyname()
                    .is_ok_and(|n| n.to_lowercase().contains(&want))
                {
                    return Ok(d);
                }
            }
        }
        Err(format!("no active {dir:?} endpoint named like {part:?}"))
    };
    let device = match sel {
        DeviceSel::CaptureNamed(n) => find(Direction::Capture, n)?,
        DeviceSel::RenderNamed(n) => find(Direction::Render, n)?,
        DeviceSel::DefaultRender => en
            .get_default_device_for_role(&Direction::Render, &Role::Console)
            .map_err(|e| format!("default playback endpoint: {e}"))?,
    };
    let device_id = device.get_id().map_err(|e| e.to_string())?;
    let device_name = device.get_friendlyname().unwrap_or_else(|_| device_id.clone());
    let mix_format = device
        .get_device_format()
        .map(|f| describe(&f))
        .unwrap_or_default();
    let mut client = device
        .get_iaudioclient()
        .map_err(|e| format!("{device_name}: audio client: {e}"))?;
    let period_ms = client
        .get_device_period()
        .map(|(d, m)| (d as f64 / 1e4, m as f64 / 1e4))
        .unwrap_or_default();
    let format = WaveFormat::new(32, 32, &SampleType::Float, 48_000, 1, None);
    // A render endpoint initialised for capture is the loopback.
    client
        .initialize_client(
            &format,
            &Direction::Capture,
            &StreamMode::PollingShared {
                autoconvert: true,
                buffer_duration_hns: BUFFER_HNS,
            },
        )
        .map_err(|e| format!("{device_name}: initialise: {e}"))?;
    let buffer_frames = client.get_buffer_size().map_err(|e| e.to_string())?;
    let capture = client
        .get_audiocaptureclient()
        .map_err(|e| format!("{device_name}: capture client: {e}"))?;
    client
        .start_stream()
        .map_err(|e| format!("{device_name}: start: {e}"))?;
    let _ = ready.send(Ok(Opened {
        device_id,
        device_name: device_name.clone(),
        mix_format,
        period_ms,
        buffer_frames,
    }));
    let mut buf = vec![0u8; buffer_frames.max(4_800) as usize * 4];
    let result = loop {
        if stop.load(Ordering::SeqCst) {
            break Ok(());
        }
        let mut got = false;
        loop {
            let n = match capture.get_next_packet_size() {
                Ok(Some(n)) => n as usize,
                Ok(None) => 0,
                Err(e) => return Err(format!("{device_name}: {e}")),
            };
            if n == 0 {
                break;
            }
            if buf.len() < n * 4 {
                buf.resize(n * 4, 0);
            }
            let (frames, info) = capture
                .read_from_device(&mut buf)
                .map_err(|e| format!("{device_name}: read: {e}"))?;
            if frames == 0 {
                break;
            }
            got = true;
            let arrival = crate::qpc_now();
            let samples: Vec<f32> = buf[..frames as usize * 4]
                .as_chunks::<4>()
                .0
                .iter()
                .map(|b| f32::from_le_bytes(*b))
                .collect();
            let p = Packet {
                stream: Some(stream),
                qpc: info.timestamp as i64,
                arrival,
                silent: info.flags.silent,
                discontinuity: info.flags.data_discontinuity,
                timestamp_error: info.flags.timestamp_error,
                samples,
            };
            if tx.send(p).is_err() {
                break;
            }
        }
        if !got {
            std::thread::sleep(POLL);
        }
    };
    let _ = client.stop_stream();
    result
}
