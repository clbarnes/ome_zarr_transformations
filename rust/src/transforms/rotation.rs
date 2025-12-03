use std::sync::Arc;

use crate::{Transformation, matrix::Matrix};

const EPSILON: f64 = 1e-10;

#[derive(Debug, Clone)]
pub struct Rotation {
    /// For a transform from N to M dimensions,
    /// this has M rows and N columns
    matrix: Matrix,
}

impl Rotation {
    pub fn try_new(matrix: Matrix) -> Result<Self, String> {
        if matrix.nrows() != matrix.ncols() {
            return Err("Rotation: rotation matrix must be square".to_string());
        }
        if !matrix.has_orthonormal_rows() {
            // rows is fine here because the matrix is square,
            // in which case an orthonormal matrix's tranpose is also orthonormal
            return Err("Rotation: rotation matrix must be orthonormal".to_string());
        }
        if (matrix.determinant().unwrap() - 1.0).abs() > EPSILON {
            return Err("Rotation: rotation matrix must have determinant = 1".to_string());
        }
        Ok(Self { matrix })
    }
}

impl Transformation for Rotation {
    fn transform_into(&self, pt: &[f64], buf: &mut [f64]) {
        self.matrix.matmul_into(pt, buf);
    }

    fn column_transform_into(&self, columns: &[&[f64]], bufs: &mut [&mut [f64]]) {
        self.matrix.matmul_transposed_into(columns, bufs);
    }

    fn invert(&self) -> Option<Arc<dyn Transformation>> {
        Some(Arc::new(Self {
            matrix: self.matrix.transpose(),
        }))
    }

    fn input_ndim(&self) -> usize {
        self.matrix.ncols()
    }

    fn output_ndim(&self) -> usize {
        self.matrix.nrows()
    }

    fn is_identity(&self) -> bool {
        self.matrix.is_identity()
    }
}

#[cfg(test)]
mod tests {
    use super::Rotation;
    use crate::{
        Matrix,
        tests::{
            check_inverse_transform_bulk, check_inverse_transform_col,
            check_inverse_transform_coord, check_transform_bulk, check_transform_col,
        },
    };

    fn make_transform() -> Rotation {
        // todo: better test case
        #[rustfmt::skip]
        let arr = vec![
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0,
        ];
        Rotation::try_new(Matrix::try_new(arr, 3).unwrap()).unwrap()
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
