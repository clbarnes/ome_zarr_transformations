use std::sync::Arc;

use smallvec::ToSmallVec;

use crate::{ShortVec, Transformation};

/// Multiply each coordinate value by a constant factor.
#[derive(Debug, Clone, PartialEq)]
pub struct Scale(ShortVec<f64>);

impl Scale {
    pub fn try_new(scale: &[f64]) -> Result<Self, String> {
        for s in scale.iter() {
            if s.is_subnormal() {
                return Err("Scale is subnormal".into());
            }
            if s.is_nan() {
                return Err("Scale is NaN".into());
            }
            if s.is_infinite() {
                return Err("Scale is infinite".into());
            }
            if s.is_sign_negative() {
                return Err("Scale is negative".into());
            }
            if *s == 0.0 {
                return Err("Scale is zero".into());
            }
        }
        Ok(Self(scale.to_smallvec()))
    }
}

impl Transformation for Scale {
    fn transform_into(&self, pt: &[f64], buf: &mut [f64]) {
        for ((o, p), s) in buf.iter_mut().zip(pt.iter()).zip(self.0.iter()) {
            *o = s * p;
        }
    }

    fn column_transform_into(&self, columns: &[&[f64]], bufs: &mut [&mut [f64]]) {
        for ((col_in, buf_in), s) in columns.iter().zip(bufs.iter_mut()).zip(self.0.iter()) {
            for (c, b) in col_in.iter().zip(buf_in.iter_mut()) {
                *b = c * s;
            }
        }
    }

    fn invert(&self) -> Option<Arc<dyn Transformation>> {
        Some(Arc::new(Scale(self.0.iter().map(|s| 1.0 / s).collect())))
    }

    fn input_ndim(&self) -> usize {
        self.0.len()
    }

    fn output_ndim(&self) -> usize {
        self.0.len()
    }

    fn is_identity(&self) -> bool {
        self.0.iter().all(|s| *s == 1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::Scale;
    use crate::tests::{
        check_inverse_transform_bulk, check_inverse_transform_col, check_inverse_transform_coord,
        check_transform_bulk, check_transform_col,
    };

    fn make_transform() -> Scale {
        Scale::try_new(&[1.0, 0.5, 2.0]).unwrap()
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
