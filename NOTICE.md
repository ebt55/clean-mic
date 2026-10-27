# Notices

## Code copied from Diarist

`src/aec.rs` is copied from Diarist (`crates/core/src/audio/aec.rs`), and
`src/erle.rs`, `src/clock.rs` and `src/capture.rs` are adapted from Diarist's
`audio/aec/offline.rs`, `audio/clock.rs` and `audio/capture`. Diarist is
licensed under the MIT License:

```
MIT License

Copyright (c) 2026 Ebin Babu Thomas

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Algorithm credit

The echo canceller's structure, constants and foreground/background tests
follow Speex's `libspeexdsp/mdf.c` (Jean-Marc Valin, 2003-2008, BSD
3-Clause). `src/aec.rs` is an independent Rust implementation of that
algorithm, not a translation of the file.

## Dependencies

- `nnnoiseless` (RNNoise in Rust, with RNNoise's model): BSD 3-Clause.
- `wasapi`: MIT.
- `realfft`, `rustfft`, `rtrb`, `windows`: MIT or Apache-2.0.
- `hound`: Apache-2.0.
