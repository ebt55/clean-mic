//! A whole run folder on the synthetic session: recorded as `clean-mic
//! live` records it, measured, then replayed from its own files, which must
//! reproduce the output exactly.

use std::path::{Path, PathBuf};

use clean_mic::engine::{Engine, EngineConfig};
use clean_mic::run::{read_packets, RunWriter, WriterReport};
use clean_mic::synth;

fn run(dir: &Path, packets: &[clean_mic::engine::Packet], record: bool) -> (WriterReport, clean_mic::engine::EngineReport) {
    let mut eng = Engine::new(EngineConfig::default(), false).unwrap();
    let mut w = RunWriter::create(dir, record, false, eng.chain().aec_delay(), eng.chain().delay48()).unwrap();
    for p in packets {
        eng.packet(p, &mut w);
    }
    (w.finish().unwrap(), eng.report())
}

fn read(p: &Path) -> Vec<i16> {
    hound::WavReader::open(p)
        .unwrap()
        .samples::<i16>()
        .map(Result::unwrap)
        .collect()
}

#[test]
fn record_measure_and_replay() {
    let dir: PathBuf = std::env::temp_dir().join(format!("clean-mic-test-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let packets = synth::session(60.0);
    let (w, e) = run(&dir, &packets, true);
    let erle = w.erle.erle_aec_db.expect("far-end-only blocks");
    println!(
        "synthetic: ERLE {erle:.1} dB (+RNNoise {:.1}) over {} of {} active blocks; echo over floor {:.1} dB; lag {:.2} ms; \
         drift {:.1} ppm; silence {} stretches / {} blocks; jumps {}; latency {:.1} ms",
        w.erle.erle_out_db.unwrap_or(f64::NAN),
        w.erle.far_blocks,
        w.erle.active_blocks,
        w.erle.echo_over_floor_db.unwrap_or(f64::NAN),
        w.lag_ms.p50,
        e.mic_vs_loopback_ppm,
        e.silence_stretches,
        e.silence_blocks,
        e.placement_jumps,
        e.latency_ms.mean,
    );
    assert!(erle > 20.0, "ERLE {erle:.1} dB");
    assert!((w.lag_ms.p50 - 3.0).abs() < 0.2, "lag {}", w.lag_ms.p50);
    assert!((e.mic_vs_loopback_ppm - synth::DRIFT_PPM).abs() < 2.0);
    // The 10 s without loopback packets: one stretch of zeros by timestamp.
    assert_eq!(e.silence_stretches, 1);
    assert!((950..=1_050).contains(&e.silence_blocks), "{}", e.silence_blocks);
    assert_eq!(e.placement_jumps, 0);
    assert_eq!(e.aec.resets, 0);
    // Owner's 4 s in every 20 s are not far-end-only.
    assert!(w.erle.far_blocks < w.erle.active_blocks);

    // Aligned outputs have one length.
    let n = read(&dir.join("mic16.wav")).len();
    for f in ["ref16.wav", "aec16.wav", "out16.wav"] {
        assert_eq!(read(&dir.join(f)).len(), n, "{f}");
    }

    let replay_dir = dir.join("replay");
    let back = read_packets(&dir).unwrap();
    assert_eq!(back.len(), packets.len());
    let (w2, _) = run(&replay_dir, &back, false);
    assert_eq!(read(&dir.join("out48.wav")), read(&replay_dir.join("out48.wav")));
    assert_eq!(read(&dir.join("aec16.wav")), read(&replay_dir.join("aec16.wav")));
    assert_eq!(w2.erle.erle_aec_db, w.erle.erle_aec_db);
    let _ = std::fs::remove_dir_all(&dir);
}
