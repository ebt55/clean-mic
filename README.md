# Clean Mic

An echo-cancelling microphone for Windows. When you take calls on desktop
speakers instead of headphones, your microphone hears the other people coming
out of your speakers. Meeting apps hide this inside their own calls, but
recordings, voice typing, streaming and apps without their own echo canceller
get the far end's voices mixed into yours.

Clean Mic removes your speakers' playback from your microphone for every
app, using the playback itself as the reference.

**Status: in development.** The live command-line stage works and is tested
on synthetic signals; live measurements on the target setup are next. There
is no installer or tray app yet. Built for one setup first (a USB microphone,
external speakers, Windows 11), see "Limits".

## How it works

The same method phones and meeting apps use internally, applied system-wide:

1. Capture the microphone (WASAPI) and, at the same time, the loopback of the
   default playback device: exactly what the speakers are being sent.
2. Put both on one timeline. The microphone and the speakers run on different
   clocks (about 50 ppm apart on the development PC, 3 ms per minute), so
   every packet's QPC timestamp feeds a running fit, and the playback is
   resampled onto the microphone's clock. Without this correction the
   canceller loses most of its effect.
3. An adaptive filter learns the echo path (speaker, room, microphone) and
   subtracts its estimate of the echo, followed by a residual echo
   suppressor. The canceller is Diarist's (multidelay frequency-domain
   adaptive filter with a double-talk robust step size, after Speex's MDF),
   run at 16 kHz with 10 ms blocks and a 200 ms filter.
4. RNNoise removes room noise and most of the echo the canceller leaves.
5. Next stage: the result goes into a virtual audio cable, which apps select
   as their microphone.

Noise suppressors alone (NVIDIA Broadcast, RNNoise, Krisp's noise mode) keep
the far end's voices, because they are speech. Echo cancellation can remove
them because it knows what was played.

### Why this canceller

An offline comparison on a private 111-minute meeting recorded on the
development PC, plus public-speech test sets, chose this chain over a port of
WebRTC's AEC3. AEC3 removed more echo while only the far end talked, but
damaged the owner's words when both talked at once. The chosen chain:

| | Raw mic | Clean Mic's chain |
|---|---|---|
| Echo words in the owner's transcript (111 min) | 13,162 | 658 |
| Owner's words kept (of 1,462) | 1,462 | 1,400 |
| Owner's word error rate, talking over the far end (spliced test set) | 17.1 % | 8.7 % |
| Echo removed while only the far end talks | 0 | about 14 dB (16 dB with RNNoise) |

Processing costs under 1 % of one CPU core and adds 33 ms of algorithmic
delay. No GPU.

## Limits

- **Not smartphone-perfect.** A phone has one clock, fixed geometry and
  factory tuning. Expect a faint, garbled trace of the far end at times.
- **A second or more of echo after a change**: moving the microphone, changing
  the volume, or a jump in the playback path (one happens about every
  10 minutes on the development PC). Full removal returns within about 5 s.
- **Loud or distorted speakers**, or speaker "dialogue" or "night" modes, make
  the echo non-linear and leave more residue.
- **Talking over the far end** leaves a little far-end residue under your
  voice.
- **Band-limited output**: the canceller runs at 16 kHz, so the cleaned mic
  carries up to about 7.6 kHz, like a typical call.
- **Placement still matters most**: a cardioid microphone close to your mouth
  with the speakers behind it is worth more than any software.
- One microphone, one playback device, Windows only.

## Building and trying it

Rust 1.88 or later, Windows:

```bash
cargo build --release
cargo test --release
target/release/clean-mic devices
```

`clean-mic live` records the microphone and the playback, cancels the echo
and writes a run folder with WAVs and a report (echo removal, echo lag,
clock drift, CPU, latency). Run `clean-mic` without arguments for the
options. Recording only happens while that command runs, into
`%LOCALAPPDATA%\clean-mic\runs`; nothing leaves the PC.

## Documents

- [docs/design.md](docs/design.md): the design, the measurements and the
  stage gates.
- [docs/stage-b.md](docs/stage-b.md): the live test procedure.
- [docs/passages.md](docs/passages.md): the reading passage for the
  double-talk check.
- [NOTICE.md](NOTICE.md): code taken from Diarist and the dependencies'
  licences.

## Licence

MIT, see [LICENSE](LICENSE).
