use crate::ShortVec;

use super::Transform;

/// A no-op transform which returns the input point as the output point.
#[derive(Debug, Default, Clone, Copy)]
pub struct Identity;

impl Transform for Identity {
    fn transform(&self, pt: &[f64]) -> ShortVec<f64> {
        ShortVec::from_slice(pt)
    }

    // fn invert(&self) -> Option<Self> {
    //     Some(Self)
    // }

    fn input_ndim(&self) -> Option<usize> {
        None
    }

    fn output_ndim(&self) -> Option<usize> {
        None
    }
}
