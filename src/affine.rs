use crate::{ShortVec, Transform, matrix::Matrix};

#[derive(Debug, Clone)]
pub struct Affine {
    /// For a transform from N to M dimensions,
    /// this has M rows and N columns
    unaugmented: Matrix,
    translation: ShortVec<f64>,
}

impl Affine {
    pub fn new(unaugmented: Matrix, translation: ShortVec<f64>) -> Result<Self, String> {
        // TODO: check for homogeneity
        if unaugmented.nrows() != translation.len() {
            return Err(
                "Affine: dimension mismatch between unaugmented matrix and translation vector"
                    .to_string(),
            );
        }
        Ok(Self {
            unaugmented,
            translation,
        })
    }

    /// Create an Affine transform from an augmented matrix,
    /// i.e. which includes the translation as the last column
    /// and a bottom row of [0, 0, ..., 1].
    pub fn from_augmented(augmented: &Matrix) -> Result<Self, String> {
        let nrows = augmented.nrows() - 1;
        let ncols = augmented.ncols() - 1;

        let mut unaugmented_data = Vec::with_capacity(nrows * ncols);
        let mut translation = ShortVec::with_capacity(nrows);

        for r in 0..nrows {
            for c in 0..ncols {
                unaugmented_data.push(augmented[(r, c)]);
            }
            translation.push(augmented[(r, ncols)]);
        }

        let unaugmented = Matrix::new(unaugmented_data, ncols)?;

        Ok(Self {
            unaugmented,
            translation,
        })
    }

    /// Create an Affine transform from a matrix which includes the translation as the last column,
    /// but does not have the augmented matrix's bottom row of [0, 0, ..., 1].
    pub fn from_translated(augmented: &Matrix) -> Result<Self, String> {
        let nrows = augmented.nrows();
        let ncols = augmented.ncols() - 1;

        let mut unaugmented_data = Vec::with_capacity(nrows * ncols);
        let mut translation = ShortVec::with_capacity(nrows);

        for r in 0..nrows {
            for c in 0..ncols {
                unaugmented_data.push(augmented[(r, c)]);
            }
            translation.push(augmented[(r, ncols)]);
        }

        let unaugmented = Matrix::new(unaugmented_data, ncols)?;

        Ok(Self {
            unaugmented,
            translation,
        })
    }
}

impl Transform for Affine {
    fn transform(&self, pt: &[f64]) -> ShortVec<f64> {
        let mut out = self.unaugmented.matmul(pt);
        for (o, t) in out.iter_mut().zip(self.translation.iter()) {
            *o += t;
        }
        out
    }

    fn input_ndim(&self) -> Option<usize> {
        Some(self.unaugmented.ncols())
    }

    fn output_ndim(&self) -> Option<usize> {
        Some(self.unaugmented.nrows())
    }
}
