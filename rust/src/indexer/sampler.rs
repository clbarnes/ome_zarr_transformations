use std::marker::PhantomData;

use crate::{
    Transformation,
    indexer::{Ravelled, value::RealIndex},
    transforms::Affine,
};

pub struct Sampler<T, I: RealIndex<T>> {
    idx_buffer: Ravelled<f64>,
    coord_buffer: Ravelled<f64>,
    columns: bool,
    indexer: I,
    grid_shape: Vec<usize>,
    _t: PhantomData<T>,
}

impl<T, I: RealIndex<T>> Sampler<T, I> {
    fn n_coords(&self) -> usize {
        self.grid_shape.iter().product()
    }

    pub fn try_new(indexer: I, grid_shape: &[usize], columns: bool) -> Result<Self, String> {
        if grid_shape.len() != indexer.ndim() {
            return Err("Incompatible grid dimensionality".into());
        }

        let idx_buffer = if columns {
            column_base_coords(grid_shape)
        } else {
            row_base_coords(grid_shape)
        };
        let coord_buffer = idx_buffer.clone();

        Ok(Self {
            coord_buffer,
            idx_buffer,
            columns,
            indexer,
            grid_shape: grid_shape.to_vec(),
            _t: Default::default(),
        })
    }

    /// Affine columns should be orthogonal, but this is not checked.
    pub fn set_orientation(&mut self, affine: Affine) {
        let input_refs: Vec<_> = self.idx_buffer.chunks().collect();
        let mut output_muts: Vec<_> = self.coord_buffer.chunks_mut().collect();
        if self.columns {
            affine.column_transform_into(&input_refs, &mut output_muts);
        } else {
            affine.bulk_transform_into(&input_refs, &mut output_muts);
        }
    }

    pub fn get_into(&self, buf: &mut [T]) {
        let coords: Vec<_> = self.coord_buffer.chunks().collect();
        if self.columns {
            self.indexer.column_get_into(&coords, buf);
        } else {
            self.indexer.bulk_get_into(&coords, buf);
        }
    }

    pub fn grid_shape(&self) -> &[usize] {
        &self.grid_shape
    }
}

impl<T: Default, I: RealIndex<T>> Sampler<T, I> {
    pub fn get(&self) -> Vec<T> {
        let mut buf: Vec<_> = std::iter::repeat_with(Default::default)
            .take(self.n_coords())
            .collect();
        let coords: Vec<_> = self.coord_buffer.chunks().collect();
        if self.columns {
            self.indexer.column_get_into(&coords, &mut buf);
        } else {
            self.indexer.bulk_get_into(&coords, &mut buf);
        }
        buf
    }
}

fn column_base_coords(extents: &[usize]) -> Ravelled<f64> {
    use std::cmp::Ordering::*;
    let n_coords: usize = extents.iter().product();
    let n_dim = extents.len();

    let mut data = Vec::with_capacity(n_coords * n_dim);

    for (dim_idx, ext) in extents.iter().enumerate() {
        let mut repeat_whole = 1;
        let mut repeat_each = 1;
        for (dim_idx2, ext2) in extents.iter().enumerate() {
            match dim_idx2.cmp(&dim_idx) {
                Less => {
                    repeat_whole *= ext2;
                }
                Equal => (),
                Greater => {
                    repeat_each *= ext2;
                }
            }
        }
        for _ in 0..repeat_whole {
            for elem in 0..*ext {
                let val = elem as f64;
                for _ in 0..repeat_each {
                    data.push(val);
                }
            }
        }
    }
    Ravelled::new_data(n_coords, data).unwrap()
}

fn row_base_coords(extents: &[usize]) -> Ravelled<f64> {
    let n_coords: usize = extents.iter().product();
    let n_dim = extents.len();
    let f_extents: Vec<_> = extents.iter().map(|e| *e as f64).collect();
    let mut data = Vec::with_capacity(n_coords * n_dim);
    let mut coord = vec![0.0; n_dim];

    for _ in 0..n_coords {
        data.extend_from_slice(&coord);
        for (c, max) in coord.iter_mut().zip(f_extents.iter()).rev() {
            let new_val = *c + 1.0;
            if new_val >= *max {
                *c = 0.0;
            } else {
                *c = new_val;
                break;
            }
        }
    }
    Ravelled::new_data(n_dim, data).unwrap()
}

#[cfg(test)]
mod tests {
    use crate::tests::init_logger;

    use super::*;

    #[test]
    fn test_row_base_coords() {
        init_logger();
        let extents = vec![3, 2];
        let ravel = row_base_coords(&extents);
        let expected = vec![
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [2.0, 0.0],
            [2.0, 1.0],
        ];
        for (actual, expected) in ravel.chunks().zip(expected.iter()) {
            assert_eq!(actual, expected);
        }
    }

    #[test]
    fn test_column_base_coords() {
        init_logger();
        let extents = vec![3, 2];
        let ravel = column_base_coords(&extents);
        let expected = vec![
            [0.0, 0.0, 1.0, 1.0, 2.0, 2.0],
            [0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
        ];
        for (actual, expected) in ravel.chunks().zip(expected.iter()) {
            assert_eq!(actual, expected);
        }
    }
}
