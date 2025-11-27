use crate::{ShortVec, Transform};

/// Translate each coordinate by adding a constant value.
#[derive(Debug, Clone)]
pub struct Translate(ShortVec<f64>);

impl Translate {
    pub fn try_new(translate: ShortVec<f64>) -> Result<Self, String> {
        for t in translate.iter() {
            if t.is_nan() {
                return Err("Translation is NaN".into());
            }
            if t.is_infinite() {
                return Err("Translation is infinite".into());
            }
        }
        Ok(Self(translate))
    }
}

impl Transform for Translate {
    fn transform(&self, pt: &[f64]) -> ShortVec<f64> {
        self.0.iter().zip(pt.iter()).map(|(t, p)| t + p).collect()
    }

    fn invert(&self) -> Option<Box<dyn Transform>> {
        Some(Box::new(Translate(
            self.0.iter().map(|t| -t).collect(),
        )))
    }

    fn input_ndim(&self) -> Option<usize> {
        Some(self.0.len())
    }

    fn output_ndim(&self) -> Option<usize> {
        Some(self.0.len())
    }
}
