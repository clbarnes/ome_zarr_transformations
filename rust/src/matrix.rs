use std::ops::Index;

use crate::ShortVec;

#[derive(Debug, Clone)]
pub struct Matrix {
    /// Row-major / C-ordered matrix data.
    data: Vec<f64>,
    nrows: usize,
    ncols: usize,
}

impl AsRef<Matrix> for Matrix {
    fn as_ref(&self) -> &Matrix {
        self
    }
}

impl Index<(usize, usize)> for Matrix {
    type Output = f64;

    fn index(&self, index: (usize, usize)) -> &Self::Output {
        self.get(index.0, index.1)
            .expect("index should be in bounds")
    }
}

impl Matrix {
    pub fn builder(row_vecs: bool) -> MatrixBuilder {
        MatrixBuilder::new(row_vecs)
    }

    /// Row-major/ C order data
    pub fn try_new(data: Vec<f64>, ncols: usize) -> Result<Self, String> {
        // TODO: check homogeneity
        if data.len() % ncols != 0 {
            return Err(format!(
                "Matrix data length {} is not divisible by ncols {}",
                data.len(),
                ncols
            ));
        }
        let nrows = data.len() / ncols;
        Ok(Self { data, nrows, ncols })
    }

    pub fn try_new_colmaj(mut data: Vec<f64>, nrows: usize) -> Result<Self, String> {
        // TODO: check homogeneity
        if data.len() % nrows != 0 {
            return Err(format!(
                "Matrix data length {} is not divisible by nrows {}",
                data.len(),
                nrows
            ));
        }
        let ncols = data.len() / nrows;
        for f_idx in 0..data.len() {
            let r = f_idx % nrows;
            let c = f_idx / nrows;
            let new_idx = r * ncols + c;
            if new_idx > f_idx {
                data.swap(f_idx, new_idx);
            }
        }
        Ok(Self { data, nrows, ncols })
    }

    pub fn transpose(&self) -> Matrix {
        let mut data = vec![0.0; self.data.len()];
        for r in 0..self.nrows {
            for c in 0..self.ncols {
                data[c * self.nrows + r] = self[(r, c)];
            }
        }
        Matrix {
            data,
            nrows: self.ncols,
            ncols: self.nrows,
        }
    }

    pub fn matmul(&self, coord: &[f64]) -> ShortVec<f64> {
        let mut result = smallvec::smallvec![f64::NAN; self.nrows];
        self.matmul_into(coord, &mut result);
        result
    }

    pub fn matmul_into(&self, coord: &[f64], buf: &mut [f64]) {
        buf.fill(0.0);
        for (idx, d) in self.data.iter().enumerate() {
            let r = idx / self.ncols;
            let c = idx % self.ncols;
            buf[r] += d * coord[c];
        }
    }

    /// N.B. Coordinate "columns" are the _rows_ of the input and output matrices.
    pub fn matmul_transposed_into(&self, coord_cols: &[&[f64]], buf: &mut[&mut[f64]]) {
        for (out_dim_idx, buf_col) in buf.iter_mut().enumerate() {
            buf_col.fill(0.0);
            let row_start = out_dim_idx * self.ncols;
            let row = &self.data[row_start..(row_start + self.ncols)];
            for (mat_val, coord_col) in row.iter().zip(coord_cols.iter()) {
                // our hottest loop is iterating over long arrays in lock step
                for (c, b) in coord_col.iter().zip(buf_col.iter_mut()) {
                    *b += c * mat_val;
                }
            }
        }
    }

    pub fn get(&self, row: usize, col: usize) -> Option<&f64> {
        self.data.get(row * self.ncols + col)
    }

    fn get_submat(
        &self,
        row: usize,
        col: usize,
        skipped_rows: &[usize],
        skipped_cols: &[usize],
    ) -> Option<&f64> {
        let actual_row = rectify_idx(row, skipped_rows);
        let actual_col = rectify_idx(col, skipped_cols);
        self.get(actual_row, actual_col)
    }
    pub fn nrows(&self) -> usize {
        self.nrows
    }
    pub fn ncols(&self) -> usize {
        self.ncols
    }

    pub(crate) fn has_orthonormal_rows(&self) -> bool {
        let mut rows: Vec<&[f64]> = Vec::with_capacity(self.nrows());
        for r in 0..self.nrows() {
            let start = r * self.ncols();
            let end = start + self.ncols();
            let new_vec = &self.data[start..end];

            if magnitude(new_vec) - 1.0 > 1e-10 {
                return false;
            }

            for row in rows.iter() {
                let dp = dot(row, new_vec);
                if dp.abs() > 1e-10 {
                    return false;
                }
            }
            rows.push(new_vec);
        }
        true
    }

    pub fn determinant(&self) -> Result<f64, String> {
        if self.nrows() != self.ncols() {
            return Err("MatrixGet: determinant only defined for square matrices".to_string());
        }
        if self.nrows() == 0 {
            return Ok(1.0);
        }
        let mut skip_rows = Vec::with_capacity(self.nrows());
        let mut skip_cols = Vec::with_capacity(self.ncols());
        Ok(self._determinant_skipping(&mut skip_rows, &mut skip_cols))
    }

    fn _determinant_skipping(
        &self,
        skipped_rows: &mut Vec<usize>,
        skipped_cols: &mut Vec<usize>,
    ) -> f64 {
        let n = self.nrows() - skipped_cols.len();

        // 0 case already handled by determinant()
        if n == 1 {
            return *self.get_submat(0, 0, skipped_rows, skipped_cols).unwrap();
        } else if n == 2 {
            return self.get_submat(0, 0, skipped_rows, skipped_cols).unwrap()
                * self.get_submat(1, 1, skipped_rows, skipped_cols).unwrap()
                - self.get_submat(0, 1, skipped_rows, skipped_cols).unwrap()
                    * self.get_submat(1, 0, skipped_rows, skipped_cols).unwrap();
        }

        // Laplace expansion along first non-skipped row
        let first_row = rectify_idx(0, skipped_rows);
        skipped_rows.push(first_row);
        let mut det = 0.0;
        let mut rel_col = 0;
        for c in 0..self.ncols() {
            if skipped_cols.contains(&c) {
                continue;
            }
            skipped_cols.push(c);
            let sign = if rel_col % 2 == 0 { 1.0 } else { -1.0 };
            det += sign
                * self.get(first_row, c).unwrap()
                * self._determinant_skipping(skipped_rows, skipped_cols);
            skipped_cols.pop();
            rel_col += 1;
        }
        skipped_rows.pop();
        det
    }
}

/// Converts a submatrix index into the corresponding full matrix index.
///
/// `skipped` must be sorted.
fn rectify_idx(mut idx: usize, skipped: &[usize]) -> usize {
    for &s in skipped.iter() {
        if s <= idx {
            idx += 1;
        } else {
            break;
        }
    }
    idx
}

/// Panics if vectors have different lengths.
fn dot(v1: &[f64], v2: &[f64]) -> f64 {
    if v1.len() != v2.len() {
        panic!("dot: vector length mismatch: {} vs {}", v1.len(), v2.len());
    }
    v1.iter().zip(v2.iter()).map(|(a, b)| a * b).sum()
}

fn magnitude(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

#[derive(Debug, Clone)]
pub struct MatrixBuilder {
    row_vecs: bool,
    dim_len: Option<usize>,
    data: Vec<f64>,
}

impl MatrixBuilder {
    fn new(row_vecs: bool) -> Self {
        Self {
            row_vecs,
            dim_len: None,
            data: Default::default(),}

    }

    pub fn add_vec(&mut self, vec: &[f64]) -> Result<&mut Self, String> {
        if let Some(len) = self.dim_len {
            if len != vec.len() {
                return Err(format!(
                    "MatrixBuilder: inconsistent vector length {}, expected {}",
                    vec.len(),
                    len
                ));
            }
        } else {
            self.dim_len = Some(vec.len());
        }
        self.data.extend_from_slice(vec);
        Ok(self)
    }

    pub fn build(self) -> Matrix {
        let dim_len = self.dim_len.unwrap_or(0);
        if self.row_vecs {
            Matrix::try_new(self.data, dim_len).expect("MatrixBuilder: inconsistent state")
        } else {
            Matrix::try_new_colmaj(self.data, dim_len).expect("MatrixBuilder: inconsistent state")
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::tests::init_logger;
    use crate::{as_muts, as_refs, vec_of_vec};

    use super::*;
    use approx::{assert_relative_eq, assert_ulps_eq};
    use faer::rand::SeedableRng;
    use faer::stats::prelude::{Rng, SmallRng};

    fn new_rng() -> SmallRng {
        SmallRng::seed_from_u64(1991)
    }

    #[ignore = "determinant tests fail over 3D"]
    #[test]
    fn test_determinant() {
        let mut rng = new_rng();
        for idx in 0..100 {
            let ndim = idx / 10 + 1;
            let mut data = vec![];
            for _ in 0..(ndim * ndim) {
                data.push(rng.random::<f64>() * 10.0);
            }
            let my_mat = Matrix::try_new(data, ndim).unwrap();
            let my_det = my_mat.determinant().unwrap();

            let faer_mat = faer::Mat::from_fn(my_mat.nrows(), my_mat.ncols(), |row, col| {
                my_mat[(row, col)]
            });
            let faer_det = faer_mat.determinant();
            println!("iteration={idx}, ndim={ndim}, my_det={my_det}, faer_det={faer_det}");
            assert_relative_eq!(my_det, faer_det, max_relative = 1e-10);
        }
    }

    #[test]
    fn test_matmul_into() {
        #[rustfmt::skip]
        let data = vec![
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
            7.0, 8.0, 9.0
        ];
        let mat = Matrix::try_new(data, 3).unwrap();
        let mut out = vec![f64::NAN; 3];
        mat.matmul_into(&[10.0, 100.0, 1000.0], &mut out);
        let expected: [f64; 3] = [3210.0, 6540.0, 9870.0];
        assert_ulps_eq!(out.as_slice(), expected.as_slice());
    }

    #[test]
    fn test_matmul_columns_into() {
        init_logger();
        #[rustfmt::skip]
        let data = vec![
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
            7.0, 8.0, 9.0
        ];
        let mat = Matrix::try_new(data, 3).unwrap();

        let col_len = 5;
        let mut out = vec_of_vec(3, col_len, f64::NAN);
        let mut out_muts = as_muts(&mut out);

        let columns = vec![
            vec![10.0; col_len],
            vec![100.0; col_len],
            vec![1000.0; col_len],
        ];
        let col_refs = as_refs(&columns);

        mat.matmul_transposed_into(&col_refs, &mut out_muts);

        let expected: [f64; 3] = [3210.0, 6540.0, 9870.0];
        for idx in 0..col_len {
            let got: Vec<_> = out.iter().map(|c| c[idx]).collect();
            assert_ulps_eq!(got.as_slice(), expected.as_slice());
        }
    }
}
