mod identity;
pub use identity::Identity;
mod affine;
pub use affine::Affine;
mod by_dimension;
pub use by_dimension::{ByDimension, ByDimensionBuilder};
mod map_axis;
pub use map_axis::MapAxis;
mod rotation;
pub use rotation::Rotation;
mod scale;
pub use scale::Scale;
mod sequence;
pub use sequence::Sequence;
mod translate;
use smallvec::SmallVec;
pub use translate::Translate;

mod matrix;
pub use matrix::{Matrix, MatrixBuilder};

pub const COORD_SIZE: usize = 6;

pub trait Transform: std::fmt::Debug {
    /// Transform a point from the input space to the output space.
    fn transform(&self, pt: &[f64]) -> ShortVec<f64>;

    /// Return the inverse transformation, if it exists.
    fn invert(&self) -> Option<Box<dyn Transform>> {
        None
    }

    // /// Transform many points from the input space to the output space.
    // ///
    // /// Trait provides a default implementation, which can be overridden if more efficient options exist.
    // fn transform_many<T: AsRef<[f64]>>(&self, pts: &[T]) -> Vec<ShortVec<f64>> {
    //     // N.B. trying to make this generic over IntoIterator<Item=&[f64]> breaks dyn-compatibility.
    //     pts.into_iter()
    //         .map(|pt| self.transform(pt.as_ref()))
    //         .collect()
    // }

    /// None if not constrained.
    fn input_ndim(&self) -> Option<usize>;

    /// None if not constrained.
    fn output_ndim(&self) -> Option<usize>;
}

/// A short vector type alias for convenience,
/// which may be replaced by arrayvec/smallvec/tinyvec in future
/// as an optimisation.
type ShortVec<T> = SmallVec<[T; COORD_SIZE]>;

#[cfg(test)]
mod tests {}
