use std::{
    collections::{BTreeMap, BTreeSet},
    f64,
};

use crate::{Identity, ShortVec, Transform};

#[derive(Debug, Clone)]
struct InIdxs {
    /// The index of the transform in the `out` vector which this input coordinate is delegated to.
    transform_idx: usize,
    /// The index in that transform's input coordinate which this input coordinate occupies.
    in_dim_idx: usize,
}

#[derive(Debug)]
struct OutTransform {
    /// The transformation to apply to the input points.
    transform: Box<dyn Transform>,
    /// The indices of the output coordinate this transform fills.
    out_idxs: ShortVec<usize>,
}

/// Delegate subsets of the input coordinates to different transformations.
#[derive(Debug)]
pub struct ByDimension {
    /// For input coordinate `i`, `idxs[i]` details which transform to delegate to (as an index into `out`)
    /// and which index in that transform's input coordinate it occupies.
    idxs: ShortVec<InIdxs>,
    /// For each output transform, the transform itself and the output coordinate indices it fills.
    out: ShortVec<OutTransform>,
}

impl Transform for ByDimension {
    fn transform(&self, pt: &[f64]) -> ShortVec<f64> {
        let mut vecs: ShortVec<_> = self
            .out
            .iter()
            .map(|ot| vec![f64::NAN; ot.out_idxs.len()])
            .collect();

        for (p, i) in pt.iter().zip(self.idxs.iter()) {
            vecs[i.transform_idx][i.in_dim_idx] = *p;
        }

        let mut out = smallvec::smallvec![f64::NAN; self.output_ndim().unwrap()];

        for (inner_pt, out_t) in vecs.iter().zip(self.out.iter()) {
            for (val, idx) in out_t
                .transform
                .transform(inner_pt)
                .into_iter()
                .zip(out_t.out_idxs.iter())
            {
                out[*idx] = val;
            }
        }
        out
    }

    // fn invert(&self) -> Option<Box<dyn Transform>> {
    //     todo!()
    // }

    fn input_ndim(&self) -> Option<usize> {
        Some(self.idxs.len())
    }

    fn output_ndim(&self) -> Option<usize> {
        Some(self.out.iter().map(|t| t.out_idxs.len()).sum())
    }
}

pub struct ByDimensionBuilder {
    in_dims: BTreeSet<usize>,
    out_dims: BTreeSet<usize>,
    in_idxs: BTreeMap<usize, InIdxs>,
    out: ShortVec<OutTransform>,
}

impl ByDimensionBuilder {
    pub fn new(in_ndim: usize, out_ndim: usize) -> Self {
        Self {
            in_dims: (0..in_ndim).collect(),
            out_dims: (0..out_ndim).collect(),
            in_idxs: Default::default(),
            out: Default::default(),
        }
    }

    fn add_boxed(
        &mut self,
        transform: Box<dyn Transform>,
        in_dims: ShortVec<usize>,
        out_dims: ShortVec<usize>,
    ) -> Result<(), String> {
        for &i in out_dims.iter() {
            if !self.out_dims.remove(&i) {
                return Err(format!("Output index {} already used", i));
            }
        }

        for (in_dim_idx, in_dim) in in_dims.into_iter().enumerate() {
            if !self.in_dims.remove(&in_dim) {
                return Err(format!("Input index {} already used", in_dim));
            }
            self.in_idxs.insert(
                in_dim,
                InIdxs {
                    transform_idx: self.out.len(),
                    in_dim_idx,
                },
            );
        }

        self.out.push(OutTransform {
            transform,
            out_idxs: out_dims,
        });

        Ok(())
    }

    pub fn add_transform<T: Transform + 'static>(
        &mut self,
        transform: T,
        in_dims: ShortVec<usize>,
        out_dims: ShortVec<usize>,
    ) -> Result<(), String> {
        self.add_boxed(Box::new(transform), in_dims, out_dims)
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
            self.add_transform(
                Identity,
                self.in_dims.iter().copied().collect(),
                self.out_dims.iter().copied().collect(),
            )?;
        }
        Ok(())
    }

    pub fn build(mut self) -> Result<ByDimension, String> {
        self.fill_missing_dims()?;

        Ok(ByDimension {
            idxs: self.in_idxs.into_values().collect(),
            out: self.out,
        })
    }
}
