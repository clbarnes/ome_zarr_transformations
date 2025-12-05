use std::sync::Arc;

/// Core spatial transformation interface.
///
/// Implementations may not perform any bounds checks on the input,
/// as these transformations generally happen in performance-critical hot loops.
/// Therefore, they may panic if coordinates or output buffers of incorrect length are given.
pub trait Transformation: std::fmt::Debug + Send + Sync {
    /// Transform a single point from the input space to the output space.
    /// Writes to a pre-allocated output buffer.
    fn transform_into(&self, pt: &[f64], buf: &mut [f64]);

    /// Transform multiple points from the input space into the output space.
    /// Writes to pre-allocated output buffers.
    ///
    /// The trait default implementation simply calls [Transformation::transform_into] in turn;
    /// specific transforms may override it.
    fn bulk_transform_into(&self, pts: &[&[f64]], bufs: &mut [&mut [f64]]) {
        for (pt, buf) in pts.iter().zip(bufs.iter_mut()) {
            self.transform_into(pt, buf);
        }
    }

    /// Transform multiple points given in columnar format.
    /// Writes to pre-allocated output buffers.
    ///
    /// The trait implementation is inefficient,
    /// simply wrapping [Transformation::transform_into],
    /// and should be overridden by implementors where optimisations are available.
    fn column_transform_into(&self, columns: &[&[f64]], bufs: &mut [&mut [f64]]) {
        let in_dim = self.input_ndim();
        // todo: check whether smallvec is faster here
        let mut in_pt = vec![f64::NAN; in_dim];
        let mut out_pt = vec![f64::NAN; self.output_ndim()];
        for pt_idx in 0..columns[0].len() {
            for (idx, col) in columns.iter().enumerate() {
                // todo: check whether pushing is faster here
                in_pt[idx] = col[pt_idx];
            }
            self.transform_into(&in_pt, &mut out_pt);
            for (out_col, p) in bufs.iter_mut().zip(out_pt.iter()) {
                out_col[pt_idx] = *p;
            }
        }
    }

    /// Return the inverse transformation, if it exists.
    ///
    /// By default, transformations are considered non-invertible;
    /// specific transformations may override this.
    fn invert(&self) -> Option<Arc<dyn Transformation>>;

    /// Whether this transformation represents the identity,
    /// i.e. input and output are the same number of dimensions
    /// and the coordinate values (and positions) are not changed.
    /// This allows some downstream optimisations.
    ///
    /// `true` means it definitely is an identity.
    /// For certain transformations, checking for identity may be very expensive;
    /// these should return `false` and users should be aware that a `false` value is not definitive.
    fn is_identity(&self) -> bool;

    fn input_ndim(&self) -> usize;

    fn output_ndim(&self) -> usize;
}

/// Trait for a type which, given a coordinate as an input,
/// will return an array of values by writing into a pre-allocated buffer.
///
/// This has a very similar API to a [Transformation],
/// but is intended to represent indexing an (N+1)D array with an ND index,
/// e.g. for the Coordinates and Displacement transforms.
pub trait ArrayProvider: std::fmt::Debug + Send + Sync {
    /// Transform a single point from the input space to the output space.
    /// Writes to a pre-allocated output buffer.
    fn get_into(&self, pt: &[f64], buf: &mut [f64]);

    /// Transform multiple points from the input space into the output space.
    /// Writes to pre-allocated output buffers.
    ///
    /// The trait default implementation simply calls [ArrayProvider::transform_into] in turn;
    /// specific transforms may override it.
    fn bulk_get_into(&self, pts: &[&[f64]], bufs: &mut [&mut [f64]]) {
        for (pt, buf) in pts.iter().zip(bufs.iter_mut()) {
            self.get_into(pt, buf);
        }
    }

    /// Transform multiple points given in columnar format.
    /// Writes to pre-allocated output buffers.
    ///
    /// The trait implementation is inefficient,
    /// simply wrapping [Transformation::transform_into],
    /// and should be overridden by implementors where optimisations are available.
    fn column_get_into(&self, columns: &[&[f64]], bufs: &mut [&mut [f64]]) {
        let in_dim = self.index_len();
        // todo: check whether smallvec is faster here
        let mut in_pt = vec![f64::NAN; in_dim];
        let mut out_pt = vec![f64::NAN; self.output_len()];
        for pt_idx in 0..columns[0].len() {
            for (idx, col) in columns.iter().enumerate() {
                // todo: check whether pushing is faster here
                in_pt[idx] = col[pt_idx];
            }
            self.get_into(&in_pt, &mut out_pt);
            for (out_col, p) in bufs.iter_mut().zip(out_pt.iter()) {
                out_col[pt_idx] = *p;
            }
        }
    }

    fn index_len(&self) -> usize;

    fn output_len(&self) -> usize;
}

/// Trait representing an N-D array to look up (or interpolate) a value into.
///
/// This is useful for turning a coordinate transformation into an image transformation.
pub trait ValueProvider<T>: std::fmt::Debug + Send + Sync {
    /// Get a value at the given coordinate.
    fn get(&self, coord: &[f64]) -> T;

    /// Default implementation just calls [ValueProvider::get] successively.
    fn bulk_get_into(&self, coords: &[&[f64]], buf: &mut [T]) {
        for (c, b) in coords.iter().zip(buf.iter_mut()) {
            *b = self.get(c);
        }
    }

    /// Default implementation just calls [ValueProvider::get] successively,
    /// and is probably inefficient.
    fn column_get_into(&self, columns: &[&[f64]], buf: &mut [T]) {
        let mut coord = vec![f64::NAN; columns.len()];
        for idx in 0..columns[0].len() {
            for (dim_idx, col) in columns.iter().enumerate() {
                coord[dim_idx] = col[idx];
            }
            buf[0] = self.get(&coord);
        }
    }
}
