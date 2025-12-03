use std::sync::Arc;

use crate::{ShortVec, Transformation, as_muts, as_refs, vec_of_vec};
use smallvec::smallvec;

/// Apply a sequence of transforms in order.
#[derive(Debug)]
pub struct Sequence {
    transforms: Vec<Arc<dyn Transformation>>,
    max_inner_ndim: usize,
}

impl Sequence {
    fn try_new(transforms: Vec<Arc<dyn Transformation>>) -> Result<Self, String> {
        if transforms.len() < 2 {
            return Err("Sequence must have >= 2 transformations".into());
        }
        let max_inner_ndim = transforms
            .iter()
            .skip(1)
            .map(|t| t.input_ndim())
            .max()
            .unwrap();
        Ok(Self {
            transforms,
            max_inner_ndim,
        })
    }

    pub fn builder() -> SequenceBuilder {
        SequenceBuilder(vec![])
    }

    fn transform_into_inner(
        &self,
        pt: &[f64],
        out_buf: &mut [f64],
        mut buf0: ShortVec<f64>,
        mut buf1: ShortVec<f64>,
    ) -> (ShortVec<f64>, ShortVec<f64>) {
        for (idx, t) in self.transforms.iter().enumerate() {
            let input_ndim = t.input_ndim();
            let output_ndim = t.output_ndim();

            if idx == 0 {
                t.transform_into(pt, &mut buf1[..output_ndim]);
            } else if idx == self.transforms.len() - 1 {
                t.transform_into(&buf0[..input_ndim], out_buf);
            } else {
                t.transform_into(&buf0[..input_ndim], &mut buf1[..output_ndim]);
            }
            (buf0, buf1) = (buf1, buf0);
        }
        (buf0, buf1)
    }
}

impl Transformation for Sequence {
    fn transform_into(&self, pt: &[f64], buf: &mut [f64]) {
        self.transform_into_inner(
            pt,
            buf,
            smallvec![f64::NAN; self.max_inner_ndim],
            smallvec![f64::NAN; self.max_inner_ndim],
        );
    }

    fn bulk_transform_into(&self, pts: &[&[f64]], bufs: &mut [&mut [f64]]) {
        let mut buf0: ShortVec<f64> = smallvec![f64::NAN; self.max_inner_ndim];
        let mut buf1: ShortVec<f64> = smallvec![f64::NAN; self.max_inner_ndim];

        for (pt, buf) in pts.iter().zip(bufs.iter_mut()) {
            (buf0, buf1) = self.transform_into_inner(pt, buf, buf0, buf1);
        }
    }

    fn column_transform_into(&self, columns: &[&[f64]], bufs: &mut [&mut [f64]]) {
        let n_pts = columns[0].len();
        let mut buf0_vec = vec_of_vec(self.max_inner_ndim, n_pts, f64::NAN);
        let mut buf1_vec = vec_of_vec(self.max_inner_ndim, n_pts, f64::NAN);

        let mut buf0_input = true;
        let last_idx = self.transforms.len() - 1;

        for (idx, t) in self.transforms.iter().enumerate() {
            let in_ndim = t.input_ndim();
            let out_ndim = t.output_ndim();

            // guaranteed to have length >= 2
            if idx == 0 {
                t.column_transform_into(columns, &mut as_muts(&mut buf1_vec[..out_ndim]));
            } else if idx == last_idx {
                if buf0_input {
                    t.column_transform_into(&as_refs(&buf0_vec[..in_ndim]), bufs);
                } else {
                    t.column_transform_into(&as_refs(&buf1_vec[..in_ndim]), bufs);
                }
            } else {
                if buf0_input {
                    t.column_transform_into(
                        &as_refs(&buf0_vec[..in_ndim]),
                        &mut as_muts(&mut buf1_vec[..out_ndim]),
                    );
                } else {
                    t.column_transform_into(
                        &as_refs(&buf1_vec[..in_ndim]),
                        &mut as_muts(&mut buf0_vec[..out_ndim]),
                    );
                }
            }
            buf0_input = !buf0_input;
        }
    }

    fn invert(&self) -> Option<Arc<dyn Transformation>> {
        let mut inv_transforms = Vec::with_capacity(self.transforms.len());
        for t in self.transforms.iter().rev() {
            inv_transforms.push(t.invert()?);
        }
        Some(Arc::new(Sequence::try_new(inv_transforms).unwrap()))
    }

    fn input_ndim(&self) -> usize {
        self.transforms.first().unwrap().input_ndim()
    }

    fn output_ndim(&self) -> usize {
        self.transforms.last().unwrap().output_ndim()
    }
}

#[derive(Debug, Default)]
pub struct SequenceBuilder(Vec<Arc<dyn Transformation>>);

impl SequenceBuilder {
    pub fn with_capacity(capacity: usize) -> Self {
        Self(Vec::with_capacity(capacity))
    }

    pub(crate) fn add_arced(&mut self, t: Arc<dyn Transformation>) -> Result<(), String> {
        if let Some(last_ndim) = self.0.last().map(|prev| prev.output_ndim()) {
            if t.input_ndim() != last_ndim {
                return Err("New transformation input dimensionality does not match previous output dimensionality".into());
            }
        }
        self.0.push(t);
        Ok(())
    }

    pub fn add_transform<T: Transformation + 'static>(&mut self, t: T) -> Result<(), String> {
        self.add_arced(Arc::new(t))
    }

    pub fn build(self) -> Result<Sequence, String> {
        Sequence::try_new(self.0)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::Sequence;
    use crate::tests::{
        check_inverse_transform_bulk, check_inverse_transform_col, check_inverse_transform_coord,
        check_transform_bulk, check_transform_col,
    };
    use crate::{Scale, Translate};

    fn make_transform() -> Sequence {
        Sequence::try_new(vec![
            Arc::new(Scale::try_new(&[1.0, 0.5, 2.0]).unwrap()),
            Arc::new(Translate::try_new(&[10.0, -6.0, 0.5]).unwrap()),
        ])
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
