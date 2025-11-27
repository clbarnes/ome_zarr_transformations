use crate::{ShortVec, Transform, matrix::Matrix};

const EPSILON: f64 = 1e-10;

#[derive(Debug, Clone)]
pub struct Rotation {
    /// For a transform from N to M dimensions,
    /// this has M rows and N columns
    matrix: Matrix,
}

impl Rotation {
    pub fn new(matrix: Matrix) -> Result<Self, String> {
        if matrix.nrows() != matrix.ncols() {
            return Err("Rotation: rotation matrix must be square".to_string());
        }
        if !matrix.orthonormal_rows() {
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

impl Transform for Rotation {
    fn transform(&self, pt: &[f64]) -> ShortVec<f64> {
        self.matrix.matmul(pt)
    }

    fn input_ndim(&self) -> Option<usize> {
        Some(self.matrix.ncols())
    }

    fn output_ndim(&self) -> Option<usize> {
        Some(self.matrix.nrows())
    }
}
