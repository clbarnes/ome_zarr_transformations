use smallvec::ToSmallVec;

use crate::{ShortVec, Transformation, matrix::Matrix};

#[derive(Debug, Clone)]
pub struct Affine {
    /// For a transform from N to M dimensions,
    /// this has M rows and N columns
    unaugmented: Matrix,
    translation: ShortVec<f64>,
}

impl Affine {
    pub fn try_new(unaugmented: Matrix, translation: &[f64]) -> Result<Self, String> {
        // TODO: check for homogeneity
        if unaugmented.nrows() != translation.len() {
            return Err(
                "Affine: dimension mismatch between unaugmented matrix and translation vector"
                    .to_string(),
            );
        }
        Ok(Self {
            unaugmented,
            translation: translation.to_smallvec(),
        })
    }

    /// Create an Affine transform from an augmented matrix,
    /// i.e. which includes the translation as the last column
    /// and a bottom row of [0, 0, ..., 1].
    pub fn try_from_augmented(augmented: &Matrix) -> Result<Self, String> {
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

        let unaugmented = Matrix::try_new(unaugmented_data, ncols)?;

        Ok(Self {
            unaugmented,
            translation,
        })
    }

    /// Create an Affine transform from a matrix which includes the translation as the last column,
    /// but does not have the augmented matrix's bottom row of [0, 0, ..., 1].
    pub fn try_from_translated(augmented: &Matrix) -> Result<Self, String> {
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

        let unaugmented = Matrix::try_new(unaugmented_data, ncols)?;

        Ok(Self {
            unaugmented,
            translation,
        })
    }
}

impl Transformation for Affine {
    fn transform_into(&self, pt: &[f64], buf: &mut [f64]) {
        self.unaugmented.matmul_into(pt, buf);
        for (o, t) in buf.iter_mut().zip(self.translation.iter()) {
            *o += t;
        }
    }

    // TODO
    // fn invert(&self) -> Option<Arc<dyn Transform>> {
    //     None
    // }

    fn input_ndim(&self) -> usize {
        self.unaugmented.ncols()
    }

    fn output_ndim(&self) -> usize {
        self.unaugmented.nrows()
    }

    fn invert(&self) -> Option<std::sync::Arc<dyn Transformation>> {
        // todo
        None
    }

    fn is_identity(&self) -> bool {
        if self.translation.iter().any(|t| *t != 0.0) {
            return false;
        }
        self.unaugmented.is_identity()
    }

    fn column_transform_into(&self, columns: &[&[f64]], bufs: &mut [&mut [f64]]) {
        self.unaugmented.matmul_transposed_into(columns, bufs);
        for (col, t) in bufs.iter_mut().zip(self.translation.iter()) {
            for c in col.iter_mut() {
                *c += t;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::Affine;
    use crate::{
        Matrix,
        tests::{
            check_inverse_transform_bulk, check_inverse_transform_col,
            check_inverse_transform_coord, check_transform_bulk, check_transform_col,
        },
    };

    fn make_transform() -> Affine {
        // todo: better test case
        #[rustfmt::skip]
        let arr = vec![
            1.0, 0.0, 0.0, 20.0,
            0.0, 1.0, 0.0, -3.0,
            0.0, 0.0, 1.0, 2.5,
        ];
        Affine::try_from_translated(&Matrix::try_new(arr, 4).unwrap()).unwrap()
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
