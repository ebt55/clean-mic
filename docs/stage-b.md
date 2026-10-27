# Stage B: the live test

What the owner runs, and what the agent does with it afterwards. The gate is
in `docs/design.md` section 9.

`clean-mic.exe` does not use the network, so simplewall should not ask
about it. It records only while a `live` command runs, into
`%LOCALAPPDATA%\clean-mic\runs\run-<time>`, outside the repo. Nothing is
uploaded anywhere.

## Before you start

- No headphones: the speakers play as they do in a meeting, at your usual
  meeting volume.
- Leave the QuadCast where and how you normally use it. Note whether the
  placement changes (cardioid, close, speakers behind the mic) are in place;
  the report should say which.
- Pick a talk or podcast with one or two people speaking (any language), and
  pause it at the start.
- Have `docs/passages.md` open to read from.

The program is `target\release\clean-mic.exe` in the repository folder;
run the commands below from there.

## Take 1: far end, then you talking over it (about 4 minutes)

1. In a terminal:

   ```bash
   target/release/clean-mic.exe live --seconds 240
   ```

2. When it prints "Running for 240 s", start the talk playing.
3. **Stay silent for one minute.** This is the far-end-only part that the
   echo-removal figure comes from. A line appears every 2 s; "far end ERLE"
   is the echo removed in that block.
4. At about 60 s, **read the passage aloud** while the talk keeps playing.
   It takes about two minutes.
5. When you finish, pause the talk and stay quiet until the run ends (or
   press Ctrl+C after 10 seconds).

## Take 2: the control (about 2.5 minutes)

6. Nothing playing:

   ```bash
   target/release/clean-mic.exe live --seconds 150
   ```

7. Read the passage once more, then wait for the end (or Ctrl+C).

Then tell the agent "done". It finds both folders under
`%LOCALAPPDATA%\clean-mic\runs`.

## Optional: listen

In each run folder, open `mic-raw.wav` (what the QuadCast heard) and
`out48.wav` (the cleaned mic) in Audacity: File > Import > Audio, both
files, then press Solo on one track at a time. They are aligned. The cleaned
output is band-limited to about 7.6 kHz by design, so it sounds a little
duller than the raw mic.

## What the agent does next

1. Replays each run from its own files and checks that the output is
   reproduced exactly.
2. Reads `report.json`: far-end-only ERLE (gate: at least 11.9 dB on the
   canceller's output), echo over the room floor, echo lag, drift (the
   research measured -51.6 ppm), CPU, latency, gaps.
3. Transcribes `mic16.wav` and `out16.wav` of both takes with the speech
   engine installed for Diarist, through the agents' shared GPU queue, and
   scores them:

   ```bash
   clean-mic wer --span-from take1/mic16.words.json take1/mic16.words.json take1/out16.words.json
   ```

   Gate: the cleaned take 1 no worse than the raw take 1, and within 3 points
   of the control (take 2).
4. Reports the gate to the owner before any stage C work starts.

If something is off (for example an echo lag outside 0 to 190 ms), the
recordings can be re-run offline with other settings (`clean-mic replay
RUN_DIR --bulk-delay-ms ...`) without recording again.
