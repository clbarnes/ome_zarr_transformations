use std::collections::BTreeSet;

use crate::{ShortVec, Transform};

/// Permute axes of the input point.
///
/// For an input point `p` and map axis vector `m`,
/// index `i` in the output point is given by `p[m[i]]`.
#[derive(Debug, Clone)]
pub struct MapAxis(ShortVec<usize>);

impl MapAxis {
    /// For an input point `p` and map axis vector `m`,
    /// index `i` in the output point is given by `p[m[i]]`.
    pub fn try_new(map: ShortVec<usize>) -> Result<Self, String> {
        let visited: BTreeSet<_> = map.iter().collect();
        if visited.len() != map.len() {
            return Err(format!(
                "MapAxis: multiple input dimensions map to the same output dimension"
            ));
        }
        if visited.last().is_some_and(|mx| **mx != map.len() - 1) {
            return Err(format!("MapAxis: not all output dimensions are addressed"));
        }

        Ok(Self(map))
    }
}

impl Transform for MapAxis {
    fn transform(&self, pt: &[f64]) -> ShortVec<f64> {
        self.0.iter().map(|idx| pt[*idx]).collect()
    }

    fn input_ndim(&self) -> Option<usize> {
        Some(self.0.len())
    }

    fn output_ndim(&self) -> Option<usize> {
        Some(self.0.len())
    }
}
