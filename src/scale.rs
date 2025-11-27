use crate::{ShortVec, Transform};

/// Multiply each coordinate value by a constant factor.
#[derive(Debug, Clone, PartialEq)]
pub struct Scale(ShortVec<f64>);

impl Scale {
    pub fn try_new(scale: ShortVec<f64>) -> Result<Self, String> {
        for s in scale.iter() {
            if s.is_subnormal() {
                return Err("Scale is subnormal".into());
            }
            if s.is_nan() {
                return Err("Scale is NaN".into());
            }
            if s.is_infinite() {
                return Err("Scale is infinite".into());
            }
            if s.is_sign_negative() {
                return Err("Scale is negative".into());
            }
            if *s == 0.0 {
                return Err("Scale is zero".into());
            }
        }
        Ok(Self(scale))
    }
}

impl Transform for Scale {
    fn transform(&self, pt: &[f64]) -> ShortVec<f64> {
        self.0.iter().zip(pt.iter()).map(|(s, p)| s * p).collect()
    }

    // fn invert(&self) -> Option<Self> {
    //     Some(Self(self.0.iter().map(|s| 1.0 / s).collect()))
    // }

    fn input_ndim(&self) -> Option<usize> {
        Some(self.0.len())
    }

    fn output_ndim(&self) -> Option<usize> {
        Some(self.0.len())
    }
}
