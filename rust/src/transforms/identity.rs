use std::sync::Arc;

use crate::Transformation;

/// A no-op transform which returns the input point as the output point.
///
/// Defined for one dimensionality.
#[derive(Debug, Default, Clone, Copy)]
pub struct Identity(usize);

impl Identity {
    pub fn new(ndim: usize) -> Self {
        Self(ndim)
    }
}

impl Transformation for Identity {
    fn invert(&self) -> Option<Arc<dyn Transformation>> {
        Some(Arc::new(*self))
    }

    fn input_ndim(&self) -> usize {
        self.0
    }

    fn output_ndim(&self) -> usize {
        self.0
    }

    fn transform_into(&self, pt: &[f64], buf: &mut [f64]) {
        buf.copy_from_slice(pt);
    }

    fn column_transform_into(&self, columns: &[&[f64]], bufs: &mut [&mut [f64]]) {
        for (c, b) in columns.iter().zip(bufs.iter_mut()) {
            b.copy_from_slice(c);
        }
    }

    fn is_identity(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::Identity;
    use crate::tests::{
        check_inverse_transform_bulk, check_inverse_transform_col, check_inverse_transform_coord,
        check_transform_bulk, check_transform_col,
    };

    fn make_transform() -> Identity {
        Identity::new(3)
    }

    #[test]
    fn test_bulk() {
        check_transform_bulk(make_transform());
    }

    #[test]
    fn test_columns() {
        check_transform_col(make_transform());
    }

    #[test]
    fn test_inverse() {
        check_inverse_transform_coord(make_transform());
    }

    #[test]
    fn test_inverse_bulk() {
        check_inverse_transform_bulk(make_transform());
    }

    #[test]
    fn test_inverse_columns() {
        check_inverse_transform_col(make_transform());
    }
}
