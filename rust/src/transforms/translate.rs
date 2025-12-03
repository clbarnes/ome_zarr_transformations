use std::sync::Arc;

use smallvec::ToSmallVec;

use crate::{ShortVec, Transformation};

/// Translate each coordinate by adding a constant value.
#[derive(Debug, Clone)]
pub struct Translate(ShortVec<f64>);

impl Translate {
    pub fn try_new(translate: &[f64]) -> Result<Self, String> {
        for t in translate.iter() {
            if t.is_nan() {
                return Err("Translation is NaN".into());
            }
            if t.is_infinite() {
                return Err("Translation is infinite".into());
            }
        }
        Ok(Self(translate.to_smallvec()))
    }
}

impl Transformation for Translate {
    fn transform_into(&self, pt: &[f64], buf: &mut [f64]) {
        for ((o, p), t) in buf.iter_mut().zip(pt.iter()).zip(self.0.iter()) {
            *o = t + p;
        }
    }

    fn column_transform_into(&self, columns: &[&[f64]], bufs: &mut [&mut [f64]]) {
        for ((col_in, buf_in), t) in columns.iter().zip(bufs.iter_mut()).zip(self.0.iter()) {
            for (c, b) in col_in.iter().zip(buf_in.iter_mut()) {
                *b = c + t;
            }
        }
    }

    fn invert(&self) -> Option<Arc<dyn Transformation>> {
        Some(Arc::new(Translate(self.0.iter().map(|t| -t).collect())))
    }

    fn input_ndim(&self) -> usize {
        self.0.len()
    }

    fn output_ndim(&self) -> usize {
        self.0.len()
    }

    fn is_identity(&self) -> bool {
        self.0.iter().all(|t| *t == 0.0)
    }
}

#[cfg(test)]
mod tests {
    use super::Translate;
    use crate::tests::{
        check_inverse_transform_bulk, check_inverse_transform_col, check_inverse_transform_coord,
        check_transform_bulk, check_transform_col,
    };

    fn make_transform() -> Translate {
        Translate::try_new(&[1.0, 0.5, 2.0]).unwrap()
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
