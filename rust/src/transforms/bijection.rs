use std::sync::Arc;

use crate::Transformation;

#[derive(Debug, Clone)]
pub struct Bijection {
    forward: Arc<dyn Transformation>,
    reverse: Arc<dyn Transformation>,
}

impl Bijection {
    pub fn try_new_arc(
        forward: Arc<dyn Transformation>,
        reverse: Arc<dyn Transformation>,
    ) -> Result<Self, String> {
        if forward.input_ndim() != reverse.output_ndim()
            || reverse.input_ndim() != forward.output_ndim()
        {
            return Err(
                "Forward and reverse transforms do not match input/ output dimensionality".into(),
            );
        }
        Ok(Self { forward, reverse })
    }

    pub fn try_new<T1: Transformation + 'static, T2: Transformation + 'static>(
        forward: T1,
        reverse: T2,
    ) -> Result<Self, String> {
        Self::try_new_arc(Arc::new(forward), Arc::new(reverse))
    }
}

impl Transformation for Bijection {
    fn invert(&self) -> Option<Arc<dyn Transformation>> {
        Some(Arc::new(Self {
            forward: self.reverse.clone(),
            reverse: self.forward.clone(),
        }))
    }

    fn input_ndim(&self) -> usize {
        self.forward.input_ndim()
    }

    fn output_ndim(&self) -> usize {
        self.forward.output_ndim()
    }

    fn transform_into(&self, pt: &[f64], buf: &mut [f64]) {
        self.forward.transform_into(pt, buf);
    }

    fn bulk_transform_into(&self, pts: &[&[f64]], bufs: &mut [&mut [f64]]) {
        self.forward.bulk_transform_into(pts, bufs);
    }

    fn column_transform_into(&self, columns: &[&[f64]], bufs: &mut [&mut [f64]]) {
        self.forward.column_transform_into(columns, bufs);
    }

    fn is_identity(&self) -> bool {
        self.forward.is_identity() && self.reverse.is_identity()
    }
}

#[cfg(test)]
mod tests {
    use super::Bijection;
    use crate::tests::{
        check_inverse_transform_bulk, check_inverse_transform_col, check_inverse_transform_coord,
        check_transform_bulk, check_transform_col,
    };
    use crate::transforms::Translate;

    fn make_transform() -> Bijection {
        Bijection::try_new(
            Translate::try_new(&[1.0, 2.0, 3.0]).unwrap(),
            Translate::try_new(&[-1.0, -2.0, -3.0]).unwrap(),
        )
        .unwrap()
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
