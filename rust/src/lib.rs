use smallvec::{SmallVec};
#[cfg(test)]
mod tests;

mod transforms;
pub use transforms::{Affine, Bijection, ByDimension, ByDimensionBuilder, Identity, MapAxis, Rotation, Scale, Sequence, SequenceBuilder, Translate};

mod alloc;
pub use alloc::{AllocatingTransformer, CustomAllocatingTransformer};

mod traits;
pub use traits::{Transformation};
mod matrix;
pub use matrix::{Matrix, MatrixBuilder};
use smallvec::smallvec;

pub const COORD_SIZE: usize = 6;

/// A short vector type alias for convenience,
/// which may be replaced by arrayvec/smallvec/tinyvec in future
/// as an optimisation.
type ShortVec<T> = SmallVec<[T; COORD_SIZE]>;

#[allow(unused)]
/// Convenience function for copying a 2D slice from input to output.
pub(crate) fn copy_into<T: Copy, InInner: AsRef<[T]>, OutInner: AsMut<[T]>>(input: &[InInner], output: &mut [OutInner]) {
    input
        .iter()
        .zip(output.iter_mut())
        .for_each(|(inp, outp)| outp.as_mut().copy_from_slice(inp.as_ref()));
}

/// Convenience function for turning a slice of sliceables into a vec of slices.
/// Allocates a new vec.
pub(crate) fn as_refs<T, Inner: AsRef<[T]>>(input: &[Inner]) -> Vec<&[T]> {
    input.iter().map(|v| v.as_ref()).collect()
}

/// Convenience function for turning a mut slice of sliceables into a vec of mut slices.
/// Allocates a new vec.
pub(crate) fn as_muts<T, Inner: AsMut<[T]>>(input: &mut [Inner]) -> Vec<&mut [T]> {
    input.iter_mut().map(|v| v.as_mut()).collect()
}

pub(crate) fn vec_of_vec<T: Copy>(outer_len: usize, inner_len: usize, val: T) -> Vec<Vec<T>> {
    vec![vec![val; inner_len]; outer_len]
}

#[allow(unused)]
pub(crate) fn vec_of_shortvec<T: Copy>(outer_len: usize, inner_len: usize, val: T) -> Vec<ShortVec<T>> {
    vec![smallvec![val; inner_len]; outer_len]
}

#[allow(unused)]
pub(crate) fn smallvec_of_vec<T: Copy>(outer_len: usize, inner_len: usize, val: T) -> ShortVec<Vec<T>> {
    smallvec![vec![val; inner_len]; outer_len]
}
