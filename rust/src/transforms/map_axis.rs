use std::{collections::BTreeSet, sync::Arc};

use smallvec::ToSmallVec;

use crate::{ShortVec, Transformation};

/// Permute axes of the input point.
///
/// For an input point `p` and map axis vector `m`,
/// index `i` in the output point is given by `p[m[i]]`.
#[derive(Debug, Clone)]
pub struct MapAxis(ShortVec<usize>);

impl MapAxis {
    /// For an input point `p` and map axis vector `m`,
    /// index `i` in the output point is given by `p[m[i]]`.
    pub fn try_new(map: &[usize]) -> Result<Self, String> {
        let visited: BTreeSet<_> = map.iter().collect();
        if visited.len() != map.len() {
            return Err(
                "MapAxis: multiple input dimensions map to the same output dimension".into(),
            );
        }
        if visited.last().is_some_and(|mx| **mx != map.len() - 1) {
            return Err("MapAxis: not all output dimensions are addressed".into());
        }

        Ok(Self(map.to_smallvec()))
    }
}

impl Transformation for MapAxis {
    fn transform_into(&self, pt: &[f64], buf: &mut [f64]) {
        for (o, m) in buf.iter_mut().zip(self.0.iter()) {
            *o = pt[*m]
        }
    }

    fn invert(&self) -> Option<Arc<dyn Transformation>> {
        let mut inv_map = smallvec::smallvec![0; self.0.len()];
        for (out_idx, in_idx) in self.0.iter().enumerate() {
            inv_map[*in_idx] = out_idx;
        }
        Some(Arc::new(MapAxis(inv_map)))
    }

    fn column_transform_into(&self, columns: &[&[f64]], bufs: &mut [&mut [f64]]) {
        for (idx, buf_col) in self.0.iter().zip(bufs.iter_mut()) {
            buf_col.copy_from_slice(columns[*idx]);
        }
    }

    fn input_ndim(&self) -> usize {
        self.0.len()
    }

    fn output_ndim(&self) -> usize {
        self.0.len()
    }

    fn is_identity(&self) -> bool {
        self.0.iter().enumerate().all(|(a, b)| a == *b)
    }
}

#[cfg(test)]
mod tests {
    use super::MapAxis;
    use crate::tests::{
        check_inverse_transform_bulk, check_inverse_transform_col, check_inverse_transform_coord,
        check_transform_bulk, check_transform_col,
    };

    fn make_transform() -> MapAxis {
        MapAxis::try_new(&[2, 0, 1]).unwrap()
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
