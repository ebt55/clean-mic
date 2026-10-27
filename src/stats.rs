//! Distribution summaries for the run report.

/// Mean, median, p99 and maximum of a series.
#[derive(Debug, Clone, Copy, Default, PartialEq, serde::Serialize)]
pub struct Summary {
    pub n: usize,
    pub mean: f64,
    pub p50: f64,
    pub p99: f64,
    pub max: f64,
}

impl Summary {
    pub fn of(v: &[f32]) -> Self {
        let mut s: Vec<f64> = v
            .iter()
            .map(|&x| f64::from(x))
            .filter(|x| x.is_finite())
            .collect();
        if s.is_empty() {
            return Self::default();
        }
        s.sort_by(f64::total_cmp);
        let q = |p: f64| s[((s.len() - 1) as f64 * p).round() as usize];
        Self {
            n: s.len(),
            mean: s.iter().sum::<f64>() / s.len() as f64,
            p50: q(0.5),
            p99: q(0.99),
            max: s[s.len() - 1],
        }
    }
}

/// Median of the finite values (`None` when there are none).
pub fn median(v: &[f64]) -> Option<f64> {
    let mut s: Vec<f64> = v.iter().copied().filter(|x| x.is_finite()).collect();
    if s.is_empty() {
        return None;
    }
    s.sort_by(f64::total_cmp);
    let n = s.len();
    Some(if n % 2 == 1 {
        s[n / 2]
    } else {
        0.5 * (s[n / 2 - 1] + s[n / 2])
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn summary_and_median() {
        let v: Vec<f32> = (1..=100).map(|x| x as f32).collect();
        let s = Summary::of(&v);
        assert_eq!(s.n, 100);
        assert!((s.mean - 50.5).abs() < 1e-9);
        assert_eq!(s.max, 100.0);
        assert_eq!(s.p99, 99.0);
        assert_eq!(median(&[3.0, 1.0, 2.0]), Some(2.0));
        assert_eq!(median(&[4.0, 1.0, 2.0, 3.0]), Some(2.5));
        assert_eq!(median(&[]), None);
    }
}
