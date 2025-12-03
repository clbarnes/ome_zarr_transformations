use std::sync::Arc;

use crate::{Identity, ShortVec, Transformation, as_muts, as_refs, vec_of_vec};
use smallvec::smallvec;

/// Apply a sequence of transforms in order.
#[derive(Debug)]
pub struct Sequence {
    transforms: Vec<Arc<dyn Transformation>>,
    /// How wide should the buffers
    max_inner_ndim: usize,
}

impl Sequence {
    fn try_new(transforms: Vec<Arc<dyn Transformation>>) -> Result<Self, String> {
        if transforms.len() < 2 {
            return Err("Sequence must have >= 2 non-identity transformations".into());
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
        // vec might be better here as we index a lot
        let mut buf0: ShortVec<f64> = smallvec![f64::NAN; self.max_inner_ndim];
        let mut buf1: ShortVec<f64> = smallvec![f64::NAN; self.max_inner_ndim];

        // delegating to inner bulk_transform_into method would mean a lot of extra
        // allocations as we'd need to trim and extend the inner bufs.
        for (pt, buf) in pts.iter().zip(bufs.iter_mut()) {
            (buf0, buf1) = self.transform_into_inner(pt, buf, buf0, buf1);
        }
    }

    fn column_transform_into(&self, columns: &[&[f64]], bufs: &mut [&mut [f64]]) {
        let n_pts = columns[0].len();
        // todo: can we re-use the output bufs as an intermediate buffer?
        //   - only in cases where it's at least as wide as _every_ intermediate dimensionality
        //     (technically every other intermediate dimensionality, as we could have a wide and a narrow one)
        //   - this may cause problems/ require extra allocations when treating the mutable slices as immutable
        //     when it's acting as the source coordinates
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
            } else if buf0_input {
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

    fn is_identity(&self) -> bool {
        self.transforms.iter().all(|t| t.is_identity())
    }
}

#[derive(Debug, Default)]
pub struct SequenceBuilder(Vec<Arc<dyn Transformation>>);

impl SequenceBuilder {
    pub fn with_capacity(capacity: usize) -> Self {
        Self(Vec::with_capacity(capacity))
    }

    pub(crate) fn add_arced(&mut self, t: Arc<dyn Transformation>) -> Result<&mut Self, String> {
        if let Some(last_ndim) = self.0.last().map(|prev| prev.output_ndim())
            && t.input_ndim() != last_ndim
        {
            return Err("New transformation input dimensionality does not match previous output dimensionality".into());
        }
        self.0.push(t);
        Ok(self)
    }

    pub fn add_transform<T: Transformation + 'static>(
        &mut self,
        t: T,
    ) -> Result<&mut Self, String> {
        self.add_arced(Arc::new(t))
    }

    /// Try to build a sequence.
    /// Fails if the sequence has fewer than 2 transformations.
    /// Does not skip identity transformations.
    pub fn build(self) -> Result<Sequence, String> {
        Sequence::try_new(self.0)
    }

    /// Build any type of transformation which can represent this sequence.
    /// Fails if the sequence has no transformations.
    ///
    /// If all transformations are identity, returns a single identity transformation.
    /// If there is only one non-identity transformation, returns that.
    /// Otherwise, returns the sequence of non-identity transformations.
    pub fn build_any(mut self) -> Result<Arc<dyn Transformation>, String> {
        let Some(ndim) = self.0.last().map(|t| t.input_ndim()) else {
            return Err("No transforms given".into());
        };
        self.0.retain(|t| !t.is_identity());
        let t = match self.0.len() {
            0 => Arc::new(Identity::new(ndim)),
            1 => self.0.pop().unwrap(),
            _ => Arc::new(Sequence::try_new(self.0)?),
        };
        Ok(t)
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
