//! Word error rate for the double-talk check (`docs/design.md` section 8):
//! the owner reads `docs/passages.md`, the speech engine transcribes the
//! raw and cleaned recordings into `<stem>.words.json`, and this scores
//! them against the passage. Insertions count, which is where echo words
//! land.
//!
//! The reading span is found once, in one transcript (normally the raw
//! mic's), by aligning the passage with free hypothesis words before and
//! after it; every transcript of the same run is then scored inside that
//! time span, since the run's files are sample-aligned.

use std::path::Path;

/// One transcribed word: start and end in seconds, text.
pub type Word = (f64, f64, String);

/// Lower case, letters, digits and apostrophes only.
pub fn normalize(text: &str) -> Vec<String> {
    text.split_whitespace()
        .map(|w| {
            w.chars()
                .filter(|c| c.is_alphanumeric() || *c == '\'')
                .flat_map(char::to_lowercase)
                .collect::<String>()
        })
        .map(|w| w.trim_matches('\'').to_owned())
        .filter(|w| !w.is_empty())
        .collect()
}

/// The passage between `<!-- passage:start -->` and `<!-- passage:end -->`
/// (the whole file when the markers are missing).
pub fn passage(text: &str) -> Vec<String> {
    let body = match (text.find("<!-- passage:start -->"), text.find("<!-- passage:end -->")) {
        (Some(a), Some(b)) if b > a => &text[a + "<!-- passage:start -->".len()..b],
        _ => text,
    };
    normalize(body)
}

/// `{"words": [{"start", "end", "word"}]}` as the speech engine writes it,
/// in time order.
pub fn load_words(path: &Path) -> Result<Vec<Word>, String> {
    let v: serde_json::Value = serde_json::from_slice(
        &std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?,
    )
    .map_err(|e| format!("{}: {e}", path.display()))?;
    let mut out: Vec<Word> = v["words"]
        .as_array()
        .ok_or_else(|| format!("{}: no words array", path.display()))?
        .iter()
        .flat_map(|w| {
            let (s, e) = (w["start"].as_f64().unwrap_or(0.0), w["end"].as_f64().unwrap_or(0.0));
            // One engine word may carry punctuation or two tokens.
            normalize(w["word"].as_str().unwrap_or(""))
                .into_iter()
                .map(move |t| (s, e, t))
        })
        .collect();
    out.sort_by(|a, b| a.0.total_cmp(&b.0));
    Ok(out)
}

/// Edit counts of a hypothesis against a reference.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, serde::Serialize)]
pub struct Errors {
    pub reference: usize,
    pub substitutions: usize,
    pub deletions: usize,
    pub insertions: usize,
}

impl Errors {
    pub fn wer(&self) -> f64 {
        (self.substitutions + self.deletions + self.insertions) as f64 / self.reference.max(1) as f64
    }
}

/// Levenshtein alignment; ties prefer a match or substitution.
pub fn errors(reference: &[String], hyp: &[String]) -> Errors {
    let (n, m) = (reference.len(), hyp.len());
    // d[i][j]: cost, with (subs, dels, ins) of one optimal path.
    let mut d = vec![vec![(0usize, 0usize, 0usize, 0usize); m + 1]; n + 1];
    for (j, c) in d[0].iter_mut().enumerate() {
        *c = (j, 0, 0, j);
    }
    for i in 1..=n {
        d[i][0] = (i, 0, i, 0);
        for j in 1..=m {
            let sub = u8::from(reference[i - 1] != hyp[j - 1]) as usize;
            let a = d[i - 1][j - 1];
            let diag = (a.0 + sub, a.1 + sub, a.2, a.3);
            let b = d[i - 1][j];
            let del = (b.0 + 1, b.1, b.2 + 1, b.3);
            let c = d[i][j - 1];
            let ins = (c.0 + 1, c.1, c.2, c.3 + 1);
            d[i][j] = if diag.0 <= del.0 && diag.0 <= ins.0 {
                diag
            } else if del.0 <= ins.0 {
                del
            } else {
                ins
            };
        }
    }
    let r = d[n][m];
    Errors {
        reference: n,
        substitutions: r.1,
        deletions: r.2,
        insertions: r.3,
    }
}

/// The time span of the passage in a transcript: semi-global alignment
/// (hypothesis words before and after the passage are free), then the
/// first and last hypothesis words aligned to it.
pub fn find_span(reference: &[String], hyp: &[Word]) -> Option<(f64, f64)> {
    let (n, m) = (reference.len(), hyp.len());
    if n == 0 || m == 0 {
        return None;
    }
    // cost[i][j] over reference[..i] and hyp[..j]; skipping leading hyp
    // words is free. `first[i][j]`: index of the first hyp word consumed.
    let inf = usize::MAX / 2;
    let mut cost = vec![vec![0usize; m + 1]; n + 1];
    let mut first = vec![vec![usize::MAX; m + 1]; n + 1];
    for i in 1..=n {
        cost[i][0] = i;
        for j in 1..=m {
            let sub = u8::from(reference[i - 1] != hyp[j - 1].2) as usize;
            let or_here = |f: usize| if f == usize::MAX { j - 1 } else { f };
            let opts = [
                (cost[i - 1][j - 1] + sub, or_here(first[i - 1][j - 1])),
                (cost[i - 1][j] + 1, first[i - 1][j]),
                (cost[i][j - 1] + 1, or_here(first[i][j - 1])),
            ];
            let best = opts.iter().min_by_key(|o| o.0).copied().unwrap_or((inf, usize::MAX));
            cost[i][j] = best.0;
            first[i][j] = best.1;
        }
    }
    // Trailing hyp words are free too: the best end column, the latest on
    // a tie (a last word substituted rather than deleted).
    let (end, _) = (1..=m)
        .map(|j| (j, cost[n][j]))
        .min_by_key(|x| (x.1, std::cmp::Reverse(x.0)))?;
    let start = first[n][end];
    if start == usize::MAX || start >= end {
        return None;
    }
    Some((hyp[start].0, hyp[end - 1].1))
}

/// The words of `hyp` that overlap `[a, b]` seconds.
pub fn within(hyp: &[Word], a: f64, b: f64) -> Vec<String> {
    hyp.iter()
        .filter(|w| w.1 >= a && w.0 <= b)
        .map(|w| w.2.clone())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn words(s: &str) -> Vec<String> {
        normalize(s)
    }

    #[test]
    fn counts_each_kind() {
        let r = words("the cat sat on the mat");
        assert_eq!(errors(&r, &r).wer(), 0.0);
        let e = errors(&r, &words("the cat sat on a mat today"));
        assert_eq!((e.substitutions, e.deletions, e.insertions), (1, 0, 1));
        let e = errors(&r, &words("cat sat the mat"));
        assert_eq!(e.substitutions + e.deletions + e.insertions, 2);
        assert_eq!(e.reference, 6);
    }

    #[test]
    fn normalizes_punctuation_and_case() {
        assert_eq!(words("Hello, World! It's 5 o'clock."), ["hello", "world", "it's", "5", "o'clock"]);
        let text = "intro\n<!-- passage:start -->\nOne two.\n<!-- passage:end -->\nouter";
        assert_eq!(passage(text), ["one", "two"]);
    }

    #[test]
    fn span_ignores_echo_words_around_the_reading() {
        let r = words("on monday morning the office was quiet");
        let mut hyp: Vec<Word> = Vec::new();
        let mut t = 0.0;
        for w in "and the weather today is fine on monday morning the office was quite later more news".split(' ') {
            hyp.push((t, t + 0.3, w.to_owned()));
            t += 0.4;
        }
        let (a, b) = find_span(&r, &hyp).unwrap();
        // "on" is word 6 (2.4 s); "quite" is word 12 (ends at 5.1 s).
        assert!((a - 2.4).abs() < 1e-9 && (b - 5.1).abs() < 1e-9, "{a} {b}");
        let e = errors(&r, &within(&hyp, a, b));
        assert_eq!((e.substitutions, e.deletions, e.insertions), (1, 0, 0));
    }
}
