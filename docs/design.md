# Clean Mic: design

A small Windows tool that removes the far end's voice coming out of the desktop
speakers from the owner's HyperX QuadCast, for every app that records the mic.
It does acoustic echo cancellation (AEC) against what the PC actually plays,
which phones and meeting apps do internally, but system-wide. One mic, one
reference, one output; not a general audio router.

27 Oct 2026. Sources: the owner's private research notes on QuadCast speaker
pickup (25 Oct 2026) and their offline A/B of echo cancellers (26 Oct 2026),
both run on this PC. Numbers from the owner's recordings are cited here only
as totals.

## 1. Why

| Problem | What removes it |
|---|---|
| Fans, keyboard, hum | Neural noise suppression (RNNoise, NVIDIA Broadcast, Krisp) |
| The far end's voices from the speakers | Echo cancellation against the playback. Noise suppressors keep these voices, because they are speech. |

Meeting apps already cancel echo for their own calls, because they play the
far end themselves. The gap is everything else: recordings, voice typing,
streaming, apps without AEC. No Windows product found in the research cancels
speaker playback system-wide using the playback as the reference.

Mic placement (cardioid, close, speakers behind the mic) comes first and is
worth an estimated 10 to 20 dB. Clean Mic adds about 14 dB on top.

## 2. The chosen chain (from the offline A/B)

The A/B compared Diarist's canceller (`aec.rs`) with sonora 0.2.0 (a Rust port
of WebRTC AEC3) on a private 111-minute meeting recorded on this PC, a spliced
double-talk set built from the owner's own utterances over that meeting's far
end, and a public-speech (FLEURS) fixture with known transcripts.

| Measure | Raw mic | `aec.rs`, clock anchors, no lookahead | + RNNoise (the chain) | sonora AEC3 |
|---|---|---|---|---|
| Far-end-only ERLE, labelled / label-free | 0 | **13.9 / 13.5 dB** | 15.8 / 15.2 dB (1) | 17.3 / 16.0 dB |
| Echo words in the owner's transcript (111 min) | 13,162 | 2,493 | **658** | 146 |
| Owner's words kept (of 1,462) | 1,462 | 1,415 | **1,400** | 1,384 |
| Spliced double talk: owner's WER | 17.1 % | 8.9 % | **8.7 %** | 29.8 % |
| FLEURS, utterances over the far end: WER (no echo: 9.2 %) | 39.4 % | 9.2 % | **10.3 %** | 16.6 % |

(1) Includes the room floor RNNoise removes.

sonora removes more echo while only the far end talks, but it damages the
owner's words in double talk. Words outrank echo residue, so the chain is
`aec.rs` plus RNNoise. The A/B also showed that the original gate ("ERLE and
the owner's level") missed this: **every stage gate below includes a
double-talk word-error check.**

Without clock-drift correction the same canceller kept only 3.6 dB on the
meeting. Drift correction is not optional.

## 3. Pipeline

```
QuadCast (WASAPI shared capture,             Default playback endpoint
 48 kHz, autoconvert to mono f32)             (WASAPI loopback, 48 kHz mono f32)
        |  packets + QPC timestamps                 |  packets + QPC timestamps
        v                                           v
  mic timeline (gap fill by QPC)              loopback timeline (gap fill by QPC;
        |                                      zeros while nothing plays)
        v                                           v
  48 -> 16 kHz (192-tap low-pass)             48 -> 16 kHz (same filter)
        |                                           |
        |                       clock-drift correction onto the mic clock:
        |                       QPC fits -> fractional loopback position per
        |                       mic sample -> 32-tap sinc interpolation
        v                                           |
  aec.rs, post-pass configuration  <--- reference --+
  (10 ms blocks, 200 ms filter, residual suppression, no lookahead)
        v
  16 -> 48 kHz (32-tap sinc; in stage C also the output rate control)
        v
  RNNoise (nnnoiseless 0.5.2, 480-sample frames), strength control (stage D)
        v
  stage B: WAV files + live ERLE          stage C on: render into
                                          "CABLE Input" (VB-Audio Virtual Cable)
                                          -> apps pick "CABLE Output" as the mic
```

CABLE Output is already the default communications microphone on this PC, so
apps left on that default pick up the cleaned mic with no change. Changing any
default device or VB-Cable setting is the owner's job, never the tool's.

### 3.1 Canceller

`aec.rs` copied from Diarist (MIT, same owner) with its attribution kept: a
multidelay block frequency-domain adaptive filter with Valin's double-talk
robust step size, a foreground/background two-path model and a Wiener residual
echo suppressor (structure after Speex `mdf.c`; independent implementation).

Configuration (the A/B's "anchors, no lookahead"):
- 16 kHz, 160-sample (10 ms) blocks;
- 3,200-sample (200 ms) filter, covering echo lags 0 to 200 ms;
- `bulk_delay = 0` (no lookahead; the A/B measured 13.9 dB without against
  14.0 dB with 5 ms, so the 5 ms are not worth their latency);
- residual suppression at the defaults (12 dB deepest, over-estimate 1.0).

`aec.rs` is 16 kHz only, so the output is band-limited to about 7.6 kHz:
wideband speech, as most calls carry. Carrying the 8 to 24 kHz band through
(with the low band's suppression gain applied to it) is a possible later
option, gated on not raising echo words.

### 3.2 Timestamps and clock drift

Mic and speakers run on different clocks: -51.6 ppm measured here, about
3.1 ms per minute. On top, the render chain after the loopback tap slides
8.8 ppm and jumps back about 5 ms every 9.45 minutes, which no timestamp sees.

Timeline per stream (as Diarist's capture pipeline does):
- every WASAPI packet carries the QPC time of its first frame;
- continuity is judged packet to packet; a gap over 5 ms is zero-filled by its
  QPC duration, an overlap is cut, so slow drift never triggers fills;
- the loopback delivers nothing while nothing plays. The engine then extends
  the loopback timeline with zeros by timestamp once the mic needs a span the
  loopback has not delivered within a short wait (section 4), and cuts the head
  off any late packet that would overlap them.

Drift correction (causal, live): each stream keeps a least-squares fit of QPC
against sample index over a trailing window (about 60 s, restarted at a device
change). For mic sample `m` the reference position is
`p(m) = loopback_fit⁻¹(mic_fit(m))`, read from the 16 kHz loopback with
Diarist's `SincInterpolator`. The placement advances by the fitted rate ratio
per sample and slews towards the fit's target slowly (a few ppm), so packet
timestamp jitter never becomes a step in the echo path. A target more than a
few ms away (a device reopen) is a jump instead, and the canceller re-adapts.

The A/B's "anchors" reference used Diarist's stored one-per-second anchors,
interpolated between them, which looks up to 1 s ahead. Live, the fit is
extrapolated; with a steady crystal drift the two should agree, and stage B
measures it.

The 5 ms render-path jumps stay a known limit: after each, `aec.rs` takes
about 5 s to recover (the A/B measured 5.1, 6.8 and 10.3 dB in the first 1,
2 and 5 s). A causal lag tracker or a relearn on a detected jump would close
most of that gap; it is a candidate improvement after stage B, measured as
ERLE in the 5 s after each jump.

A diagnostic lag estimate (GCC-PHAT of mic against reference over 2 s
windows) is reported live. If the echo lag ever sits below 0 or above about
190 ms (Bluetooth or HDMI output), the reference needs a bulk delay; stage E
sets it automatically per output device.

### 3.3 Noise suppression

RNNoise via nnnoiseless 0.5.2 at 48 kHz after the canceller. It removes most
of the canceller's leftover echo words for about a point of WER on the owner's
words. Stage D adds a bypass switch and a strength control (a delay-matched
dry/wet mix, so strength sets the deepest attenuation).

### 3.4 Output (stage C)

Render into "CABLE Input" in shared mode. The cable consumes on its own clock,
not the QuadCast's, so a slow controller on the render buffer's fill level
steers the 16 to 48 kHz interpolator's ratio by up to about ±200 ppm (the
interpolator already takes fractional positions, so this costs no extra
stage). Without it the cable under- or overruns every few minutes and clicks.

## 4. Real-time engine

Threads:
1. **Mic capture**: event-driven shared mode on the QuadCast (by name, never
   the default), packets into a lock-free ring.
2. **Loopback capture**: the default console render endpoint, loopback flag,
   polled every 5 ms, packets into a second ring.
3. **Processing**: places packets on the two timelines, decimates, and for
   each 10 ms mic block, as soon as the loopback covers `p(block end)` plus
   the interpolator's half length, places the reference and runs canceller,
   interpolator and RNNoise. If the loopback has not covered that span 20 ms
   after the block's capture time, it is treated as silence (zeros by
   timestamp).
4. **Output**: stage B writes WAVs from a fourth thread so file I/O never
   stalls processing; stage C adds the render thread.

The processing chain (`chain.rs`) is plain code with no WASAPI in it, so the
same chain runs in unit tests and in `clean-mic replay`, which re-runs a
recorded session (raw mic, raw loopback and their packet timestamps) through
the identical code. Tuning then needs no new recordings.

## 5. Latency budget

| Stage | Delay |
|---|---|
| Mic capture period | 10 ms (QuadCast default period; 3 ms minimum) |
| Waiting for the matching loopback | 0 to 10 ms (20 ms ceiling, then zeros) |
| 48 -> 16 kHz filter | 2 ms (half of 192 taps at 48 kHz) |
| Canceller block | 10 ms |
| Residual suppressor overlap-add | 10 ms |
| 16 -> 48 kHz interpolator | 1 ms |
| RNNoise lookahead | 10 ms |
| Render buffer (stage C) | 10 ms period plus the controller's target fill |
| **Through the app** | **about 45 to 65 ms** |
| VB-Cable plus the reading app's buffer | unknown; stage C measures it |

Algorithmic latency is the A/B's 30 ms plus 3 ms of resampling filters. For
calls, up to about 150 ms one way counts as good (ITU-T G.114), so the app's
share is acceptable; VB-Cable's own buffer may be the biggest item, and
lowering it is a control-panel change the owner would make.

A 5 ms canceller block would cut 10 ms but is unmeasured; not now.

## 6. CPU

The A/B measured the chain at 110.5 µs per 10 ms of audio on this Ryzen 9
5900X (1.1 % of one core), with naive resamplers taking 57 µs of it. Polyphase
decimation (only every third output computed) should halve that. Budget: under
3 % of one core in total, including capture and output threads. No GPU.

Measured on 27 Oct 2026 (`clean-mic bench`, the synthetic session of
`src/synth.rs`, release build, three 120 s runs): 83 to 86 µs per 10 ms block
for the whole engine (both decimators, drift placement, canceller,
interpolator, RNNoise), 0.85 % of one core; p99 104 to 141 µs, worst block
0.8 ms, with another agent's GPU job running at the time.

## 7. Failure handling and safety

- **Default playback changes**: reopen the loopback on the new device, restart
  its drift fit and reset the canceller (stage E). Headphones or headset form
  factor: bypass the canceller, since there is no echo to remove.
- **QuadCast unplugged, sleep and resume**: output silence, retry every
  second, reopen when the mic returns (stage E).
- **Feedback loop**: if anything plays CABLE Output to the speakers, the
  owner's voice enters the reference and gets cancelled. The engine watches
  the correlation between its output and the reference and warns (stage E).
- **Off is obvious**: when the app is off, apps on CABLE Output hear silence.
  The tray icon must make "off" unmistakable.
- **Recording**: only the stage B CLI and the replay tool write audio, and only
  into a run folder outside the repo. The tray app never records. WAVs are in
  `.gitignore`.

## 8. Measurements

**ERLE (echo return loss enhancement), label-free**, as Diarist's post-pass
computes it: 100 Hz high-passed energies in 2 s blocks; a block counts when the
reference is active (at or above -45 dBFS) and the mic stands at most 2 dB
above the level the median return loss of the active blocks predicts (more is
the owner talking). ERLE is the summed mic energy over the summed output energy
in those blocks, with the output delay-compensated. It is measured on the
canceller's output (comparable with the A/B's 13.5 / 13.9 dB) and after RNNoise
(comparable with 15.2 / 15.8 dB). The run report also gives the ceiling the mic's
own floor sets (echo over floor), since a well-placed mic leaves less echo to
remove.

**Double-talk word error rate.** The owner reads a fixed public-domain passage
(`docs/passages.md`) twice:
1. **control**: no playback;
2. **double talk**: a talk or podcast playing on the speakers at meeting
   volume throughout.

The run folder's raw mic and cleaned output are transcribed with the speech
engine already installed for Diarist, through the agents' shared GPU queue,
and scored against the passage text (word-level edit distance; insertions,
which is where echo words land, count). The A/B's references: owner's WER
over the far end 8.7 % with the chain against 17.1 % raw (spliced set), and
10.3 % against 9.2 % with no echo at all (FLEURS).

**CPU**: processing time per 10 ms block (mean, p99, max) and process CPU time
over the run.

**Latency**: algorithmic (from the chain, checked with a probe) plus measured
buffering: the QPC time a block's output is ready minus the QPC capture time of
its first mic sample. Stage C adds mic-to-CABLE-Output on a digital loop
(the tool reads its own output back from CABLE Output and matches it to its
input by cross-correlation; no sound is played).

## 9. Stages and gates

Stage A (the offline A/B) passed on 26 Oct 2026. The owner hears the gate
report before the next stage starts.

| Stage | Work | Gate |
|---|---|---|
| **B. Live CLI** | `clean-mic live`: both captures, timelines, drift correction, canceller and RNNoise; writes the raw mic, raw loopback, placed reference, canceller output and final output as WAVs, plus packet timestamps and a JSON report; live ERLE, lag, drift and CPU on the console. `clean-mic replay` re-runs a run folder. | 1. Far-end-only ERLE (label-free, canceller output) **at least 11.9 dB**, within 2 dB of the offline 13.9 dB (label-free offline: 13.5 dB), with the floor ceiling stated. 2. **Double talk**: the cleaned output's WER no worse than the raw mic's in double talk, and within 3 points of the control read. 3. CPU and latency measured and reported. 4. Measured drift consistent with -51.6 ppm on the same devices. |
| **C. Cable output** | Render into CABLE Input with buffer-level rate control; latency on a digital loop | 1. 30 minutes with no under- or overruns after the first 5 s. 2. Mic-to-CABLE-Output latency known. 3. **Double talk** through the cable within 1 point of stage B. |
| **D. Noise suppression controls** | RNNoise bypass and strength | 1. Owner's listening test: voice intact at each strength. 2. **Double talk**: RNNoise on costs at most 1.5 points of WER against off. |
| **E. Tray app** | Tray icon (on, off, bypass, mode), level meters, start with Windows, device-change handling, automatic bulk delay per output, feedback warning | Checklist: default playback change recovers within 2 s; headphones bypass; QuadCast unplug and replug recovers; feedback warning fires in a simulated loop; "off" obvious. **Double talk** within 1 point of stage D. |
| **F. Soak** | Two hours of normal use, sleep and resume, device switches, unplugging | Mean CPU under 3 % of a core; no crash; at most one glitch per hour; every recovery within 2 s; **double talk** at the end within 1 point of stage E. |

## 10. Known limits

- A second or two (up to about 5 s for full removal) of echo after the mic
  moves, the volume changes, or the render path jumps.
- More residue with loud or distorted speakers, or DSP modes in the external
  speaker box ("dialogue", "night"): the echo is no longer linear.
- A faint far-end trace under the owner's voice in double talk; RNNoise keeps
  part of it, because it is speech.
- Output band-limited to about 7.6 kHz (section 3.1).
- Not smartphone-perfect: a phone has one clock, fixed geometry and factory
  tuning.

## 11. Code and licences

- `src/aec.rs`: Diarist's `crates/core/src/audio/aec.rs` (MIT, Copyright (c)
  2026 the Diarist author, who owns this project too), unchanged apart from
  module paths. Algorithm after Speex `libspeexdsp/mdf.c` (Jean-Marc Valin,
  BSD-3-Clause), independently implemented.
- `src/erle.rs`: the label-free ERLE of Diarist's `aec/offline.rs`, made
  streaming.
- `src/clock.rs`: the QPC-to-sample mapping of Diarist's `audio/clock.rs`,
  made causal (trailing least-squares fits instead of stored anchors).
- `src/capture.rs`: after Diarist's WASAPI capture (`audio/capture`).
- Dependencies: `wasapi` (MIT), `nnnoiseless` (BSD-3-Clause, RNNoise model
  BSD-3-Clause), `realfft`/`rustfft` (MIT/Apache-2.0), `hound` (Apache-2.0),
  `rtrb` (MIT/Apache-2.0), `windows` (MIT/Apache-2.0).

`NOTICE.md` carries the full notices.
