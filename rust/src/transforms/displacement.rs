use std::sync::Arc;

use crate::{Transformation, traits::ArrayProvider};

#[derive(Debug)]
pub struct Displacement {
    provider: Arc<dyn ArrayProvider>,
}

impl Displacement {
    pub fn new_any(provider: Arc<dyn ArrayProvider>) -> Self {
        Self { provider }
    }
    pub fn new<P: ArrayProvider + 'static>(provider: P) -> Self {
        Self::new_any(Arc::new(provider))
    }
}

impl Transformation for Displacement {
    fn transform_into(&self, pt: &[f64], buf: &mut [f64]) {
        self.provider.get_into(pt, buf);
        for (i, o) in pt.iter().zip(buf.iter_mut()) {
            *o += i;
        }
    }

    fn bulk_transform_into(&self, pts: &[&[f64]], bufs: &mut [&mut [f64]]) {
        self.provider.bulk_get_into(pts, bufs);
        for (pt, buf) in pts.iter().zip(bufs.iter_mut()) {
            for (i, o) in pt.iter().zip(buf.iter_mut()) {
                *o += i;
            }
        }
    }

    fn column_transform_into(&self, columns: &[&[f64]], bufs: &mut [&mut [f64]]) {
        self.provider.column_get_into(columns, bufs);
        for (col, out_col) in columns.iter().zip(bufs.iter_mut()) {
            for (i, o) in col.iter().zip(out_col.iter_mut()) {
                *o += i;
            }
        }
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
