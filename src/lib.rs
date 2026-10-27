//! Clean Mic: echo cancellation for the owner's HyperX QuadCast against
//! whatever the PC plays (`docs/design.md`).
//!
//! The signal path: [`capture`] (WASAPI, Windows) delivers timestamped
//! packets; [`engine`] places them on [`clock`] timelines, decimates them
//! ([`resample`]), places the reference on the mic's clock and runs the
//! [`chain`]: the canceller ([`aec`], Diarist's) and RNNoise ([`denoise`]).
//! [`run`] records a run folder and [`erle`] and [`wer`] measure it.

pub mod aec;
#[cfg(windows)]
pub mod capture;
pub mod chain;
pub mod clock;
pub mod denoise;
pub mod engine;
pub mod erle;
pub mod resample;
pub mod run;
pub mod stats;
pub mod synth;
pub mod wer;

/// The QPC clock in 100 ns units, the timebase of WASAPI's timestamps.
#[cfg(windows)]
pub fn qpc_now() -> i64 {
    use windows::Win32::System::Performance::{QueryPerformanceCounter, QueryPerformanceFrequency};
    let (mut count, mut freq) = (0i64, 0i64);
    // SAFETY: both calls only write the given integers.
    unsafe {
        let _ = QueryPerformanceCounter(&mut count);
        let _ = QueryPerformanceFrequency(&mut freq);
    }
    if freq <= 0 {
        return 0;
    }
    (i128::from(count) * 10_000_000 / i128::from(freq)) as i64
}

/// A monotonic clock in 100 ns units (tests on other platforms).
#[cfg(not(windows))]
pub fn qpc_now() -> i64 {
    use std::sync::OnceLock;
    static START: OnceLock<std::time::Instant> = OnceLock::new();
    (START.get_or_init(std::time::Instant::now).elapsed().as_nanos() / 100) as i64
}

/// Registers the calling thread with MMCSS as "Pro Audio" (higher, steadier
/// scheduling for the capture and processing threads). Best effort.
pub fn pro_audio_thread() {
    #[cfg(windows)]
    {
        use windows::core::w;
        use windows::Win32::System::Threading::AvSetMmThreadCharacteristicsW;
        let mut index = 0u32;
        // SAFETY: a constant task name and a valid out pointer; the handle
        // stays registered for the thread's life.
        let _ = unsafe { AvSetMmThreadCharacteristicsW(w!("Pro Audio"), &mut index) };
    }
}

/// CPU time (user plus kernel) this process has used, seconds.
pub fn process_cpu_seconds() -> Option<f64> {
    #[cfg(windows)]
    {
        use windows::Win32::Foundation::FILETIME;
        use windows::Win32::System::Threading::{GetCurrentProcess, GetProcessTimes};
        let (mut c, mut e, mut k, mut u) = (
            FILETIME::default(),
            FILETIME::default(),
            FILETIME::default(),
            FILETIME::default(),
        );
        // SAFETY: the pseudo handle needs no closing; four valid out pointers.
        unsafe { GetProcessTimes(GetCurrentProcess(), &mut c, &mut e, &mut k, &mut u) }.ok()?;
        let t = |f: FILETIME| (u64::from(f.dwHighDateTime) << 32 | u64::from(f.dwLowDateTime)) as f64 / 1e7;
        Some(t(k) + t(u))
    }
    #[cfg(not(windows))]
    {
        None
    }
}
