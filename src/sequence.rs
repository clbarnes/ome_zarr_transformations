use crate::{ShortVec, Transform};

/// Apply a sequence of transforms in order.
#[derive(Debug)]
pub struct Sequence(Vec<Box<dyn Transform>>);

impl Sequence {
    pub fn new(transforms: Vec<Box<dyn Transform>>) -> Self {
        Self(transforms)
    }
}

impl Transform for Sequence {
    fn transform(&self, pt: &[f64]) -> ShortVec<f64> {
        let mut pt_v = ShortVec::from_slice(pt);
        for t in self.0.iter() {
            pt_v = t.transform(&pt_v);
        }
        pt_v
    }

    fn input_ndim(&self) -> Option<usize> {
        self.0.iter().filter_map(|t| t.input_ndim()).next()
    }

    fn output_ndim(&self) -> Option<usize> {
        self.0.iter().rev().filter_map(|t| t.output_ndim()).next()
    }
}
