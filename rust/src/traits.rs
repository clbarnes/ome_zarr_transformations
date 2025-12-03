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
