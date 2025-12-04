use std::sync::Arc;

use crate::{Transformation, traits::ArrayProvider};

#[derive(Debug)]
pub struct Coordinate {
    provider: Arc<dyn ArrayProvider>,
}

impl Coordinate {
    pub fn new_any(provider: Arc<dyn ArrayProvider>) -> Self {
        Self { provider }
    }
    pub fn new<P: ArrayProvider + 'static>(provider: P) -> Self {
        Self::new_any(Arc::new(provider))
    }
}

impl Transformation for Coordinate {
    fn transform_into(&self, pt: &[f64], buf: &mut [f64]) {
        self.provider.get_into(pt, buf);
    }

    fn bulk_transform_into(&self, pts: &[&[f64]], bufs: &mut [&mut [f64]]) {
        self.provider.bulk_get_into(pts, bufs);
    }

    fn column_transform_into(&self, columns: &[&[f64]], bufs: &mut [&mut [f64]]) {
        self.provider.column_get_into(columns, bufs);
    }

    fn invert(&self) -> Option<std::sync::Arc<dyn Transformation>> {
        None
    }

    fn is_identity(&self) -> bool {
        false
    }

    fn input_ndim(&self) -> usize {
        self.provider.index_len()
    }

    fn output_ndim(&self) -> usize {
        self.provider.output_len()
    }
}
