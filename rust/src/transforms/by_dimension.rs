use std::{collections::BTreeSet, f64, sync::Arc};

use smallvec::smallvec;

use crate::{Identity, ShortVec, Transformation};

impl ByDimension {
    /// Create a new builder for a ByDimension transform.
    pub fn builder(in_ndim: usize, out_ndim: usize) -> ByDimensionBuilder {
        ByDimensionBuilder::new(in_ndim, out_ndim)
    }
}

#[derive(Debug)]
struct SubTransform {
    transform: Arc<dyn Transformation>,
    in_dims: Vec<usize>,
    out_dims: Vec<usize>,
}

#[derive(Debug)]
pub struct ByDimension(Vec<SubTransform>);

impl Transformation for ByDimension {
    fn transform_into(&self, pt: &[f64], buf: &mut [f64]) {
        let mut ordered_pt: ShortVec<f64> = smallvec![f64::NAN; pt.len()];
        let mut ordered_buf: ShortVec<f64> = smallvec![f64::NAN; buf.len()];
        for bt in self.0.iter() {
            for (i, o) in bt.in_dims.iter().zip(ordered_pt.iter_mut()) {
                *o = pt[*i];
            }
            bt.transform.transform_into(
                &ordered_pt[..bt.in_dims.len()],
                &mut ordered_buf[..bt.out_dims.len()],
            );
            for (out_dim, val) in bt.out_dims.iter().zip(ordered_buf.iter()) {
                buf[*out_dim] = *val;
            }
        }
    }

    fn bulk_transform_into(&self, pts: &[&[f64]], bufs: &mut [&mut [f64]]) {
        // todo: vecs might be faster here as we index a lot
        let mut ordered_pt: ShortVec<f64> = smallvec![f64::NAN; pts.len()];
        let mut ordered_buf: ShortVec<f64> = smallvec![f64::NAN; bufs.len()];

        for (pt, buf) in pts.iter().zip(bufs.iter_mut()) {
            for bt in self.0.iter() {
                for (i, o) in bt.in_dims.iter().zip(ordered_pt.iter_mut()) {
                    *o = pt[*i];
                }
                bt.transform.transform_into(
                    &ordered_pt[..bt.in_dims.len()],
                    &mut ordered_buf[..bt.out_dims.len()],
                );
                for (out_dim, val) in bt.out_dims.iter().zip(ordered_buf.iter()) {
                    buf[*out_dim] = *val;
                }
            }
        }
    }

    fn column_transform_into(&self, columns: &[&[f64]], bufs: &mut [&mut [f64]]) {
        let mut input_cols = Vec::with_capacity(columns.len());
        let mut order: Vec<usize> = (0..bufs.len()).collect();
        let mut swaps = Vec::with_capacity(bufs.len());
        let mut start = 0;

        for bt in self.0.iter() {
            // create an inner cols vec which contains references into the original
            input_cols.clear();
            for &idx in bt.in_dims.iter() {
                input_cols.push(columns[idx])
            }

            // Swap the desired columns to the front of bufs.
            // We can't create a new vec because the compiler can't guarantee
            // we wouldn't have 2 mutable references to the same column.
            for (local_tgt_idx, desired_idx) in bt.out_dims.iter().enumerate() {
                let tgt_idx = local_tgt_idx + start;
                // skipping items we've already swapped to the front
                let src_idx = order[tgt_idx..]
                    .iter()
                    .enumerate()
                    // find the current location of the desired output column
                    .find_map(|(sub_idx, loc)| (desired_idx == loc).then_some(sub_idx + tgt_idx))
                    .unwrap();

                // already in the right place
                if src_idx == local_tgt_idx {
                    continue;
                }
                // swap the columns
                bufs.swap(tgt_idx, src_idx);
                // update the column map
                order.swap(tgt_idx, src_idx);
                // log that we have made this swap, so we can undo it later
                swaps.push((tgt_idx, src_idx));
            }

            let end = start + bt.out_dims.len();
            bt.transform
                .column_transform_into(&input_cols, &mut bufs[start..end]);
            start = end;
        }

        // Put the columns back again.
        // This is not required in all workflows but it's cheap.
        for (a, b) in swaps.iter().rev() {
            bufs.swap(*a, *b);
        }
    }

    fn invert(&self) -> Option<Arc<dyn Transformation>> {
        let mut out = Vec::with_capacity(self.0.len());
        for bt in self.0.iter() {
            out.push(SubTransform {
                transform: bt.transform.invert()?,
                in_dims: bt.out_dims.clone(),
                out_dims: bt.in_dims.clone(),
            });
        }
        Some(Arc::new(Self(out)))
    }

    fn input_ndim(&self) -> usize {
        self.0.iter().map(|bt| bt.in_dims.len()).sum()
    }

    fn output_ndim(&self) -> usize {
        self.0.iter().map(|bt| bt.out_dims.len()).sum()
    }
}

pub struct ByDimensionBuilder {
    in_dims: BTreeSet<usize>,
    out_dims: BTreeSet<usize>,
    sub_transforms: Vec<SubTransform>,
}

impl ByDimensionBuilder {
    fn new(in_ndim: usize, out_ndim: usize) -> Self {
        Self {
            in_dims: (0..in_ndim).collect(),
            out_dims: (0..out_ndim).collect(),
            sub_transforms: Vec::with_capacity(in_ndim.max(out_ndim)),
        }
    }

    fn add_arced(
        &mut self,
        transform: Arc<dyn Transformation>,
        in_dims: &[usize],
        out_dims: &[usize],
    ) -> Result<&mut Self, String> {
        for &out_dim in out_dims.iter() {
            if !self.out_dims.remove(&out_dim) {
                return Err(format!("Output index {} already used", out_dim));
            }
        }

        for &in_dim in in_dims.into_iter() {
            if !self.in_dims.remove(&in_dim) {
                return Err(format!("Input index {} already used", in_dim));
            }
        }

        self.sub_transforms.push(SubTransform {
            transform,
            in_dims: in_dims.to_vec(),
            out_dims: out_dims.to_vec(),
        });

        Ok(self)
    }

    pub fn add_transform<T: Transformation + 'static>(
        &mut self,
        transform: T,
        in_dims: &[usize],
        out_dims: &[usize],
    ) -> Result<&mut Self, String> {
        self.add_arced(Arc::new(transform), in_dims, out_dims)
    }

    fn fill_missing_dims(&mut self) -> Result<(), String> {
        if self.in_dims.len() != self.out_dims.len() {
            return Err(format!(
                "{} in-dims and {} out-dims left unassigned",
                self.in_dims.len(),
                self.out_dims.len()
            ));
        }
        if self.in_dims.len() > 0 {
            self.sub_transforms.push(SubTransform {
                transform: Arc::new(Identity::new(self.in_dims.len())),
                in_dims: self.in_dims.iter().copied().collect(),
                out_dims: self.out_dims.iter().copied().collect(),
            });
        }
        Ok(())
    }

    pub fn build(mut self) -> Result<ByDimension, String> {
        self.fill_missing_dims()?;

        Ok(ByDimension(self.sub_transforms))
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_ulps_eq;

    use super::ByDimension;
    use crate::tests::{
        check_inverse_transform_bulk, check_inverse_transform_col, check_inverse_transform_coord,
        check_transform_bulk, check_transform_col, init_logger,
    };
    use crate::{Scale, Transformation, Translate, as_muts, as_refs, vec_of_vec};

    fn make_transform() -> ByDimension {
        let mut builder = ByDimension::builder(3, 3);
        builder
            .add_transform(Translate::try_new(&[-1.0, 2.0]).unwrap(), &[0, 2], &[1, 0])
            .unwrap()
            .add_transform(Scale::try_new(&[100.0]).unwrap(), &[1], &[2])
            .unwrap();
        builder.build().unwrap()
    }

    #[test]
    fn test_columns_manual() {
        init_logger();
        let cols = vec![vec![1.0, 4.0], vec![2.0, 5.0], vec![3.0, 6.0]];
        let col_refs = as_refs(&cols);

        let mut out = vec_of_vec(3, 2, f64::NAN);
        let mut out_mut = as_muts(&mut out);

        let t = make_transform();
        t.column_transform_into(&col_refs, &mut out_mut);

        let expected = vec![vec![5.0, 8.0], vec![0.0, 3.0], vec![200.0, 500.0]];
        log::debug!("Got cols {out:?}");
        log::debug!("Expected cols {expected:?}");
        for (actual_col, expected_col) in out.iter().zip(expected.iter()) {
            assert_ulps_eq!(actual_col.as_slice(), expected_col.as_slice());
        }
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
