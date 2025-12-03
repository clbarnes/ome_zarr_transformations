use ordered_float::OrderedFloat;
use std::{
    collections::HashMap,
    sync::{Arc, RwLock},
};

use petgraph::algo::astar;
use petgraph::prelude::*;

use crate::{Identity, SequenceBuilder, Transformation};

const DEFAULT_COST: f64 = 1.0;

#[derive(Debug, Clone)]
pub struct Edge {
    transform: Arc<dyn Transformation>,
    cost: OrderedFloat<f64>,
}

impl Edge {
    pub fn new_cost<T: Into<Arc<dyn Transformation>>>(transform: T, cost: f64) -> Self {
        Self {
            transform: transform.into(),
            cost: OrderedFloat(cost),
        }
    }

    pub fn new<T: Into<Arc<dyn Transformation>>>(transform: T) -> Self {
        Self::new_cost(transform, DEFAULT_COST)
    }
}

type InnerPathCache = RwLock<HashMap<(NodeIndex, NodeIndex), Option<Arc<dyn Transformation>>>>;

/// Contains a locked map from (src, tgt) tuple to the possible transformation going from src to tgt,
/// if it has been queried before.
/// An entry will be unoccupied if this path has not been queried before,
/// None if it was queried before and found not to exist,
/// and Some if a tranformation was found.
#[derive(Debug, Default)]
struct PathCache(InnerPathCache);

impl PathCache {
    /// Look-before-you-leap poison-clearing.
    fn clear_poison(&self) {
        if self.0.is_poisoned() {
            self.0.clear_poison();
        }
    }

    fn clear_mut(&mut self) {
        self.clear_poison();
        self.0.get_mut().unwrap().clear();
    }

    fn insert(
        &self,
        src: NodeIndex,
        tgt: NodeIndex,
        maybe_t: Option<Arc<dyn Transformation>>,
    ) -> Option<Option<Arc<dyn Transformation>>> {
        self.clear_poison();
        self.0.write().unwrap().insert((src, tgt), maybe_t)
    }

    fn get(&self, src: &NodeIndex, tgt: &NodeIndex) -> Option<Option<Arc<dyn Transformation>>> {
        self.clear_poison();
        let outer = self.0.read().unwrap();
        outer.get(&(*src, *tgt)).map(|t| t.as_ref().cloned())
    }
}

/// This type optimises for performance rather than a faithful representation of the given transformations.
/// In practice, this means it filters out superfluous identity transformations.
#[derive(Debug, Default)]
pub struct TransformGraph<C: std::hash::Hash + Eq + Clone> {
    graph: StableDiGraph<C, Edge>,
    coord_systems: HashMap<C, NodeInfo>,
    path_cache: PathCache,
}

#[derive(Debug, Copy, Clone)]
struct NodeInfo {
    idx: NodeIndex,
    ndim: usize,
}

impl<C: std::hash::Hash + Eq + Clone> TransformGraph<C> {
    fn ensure_coord_system(&mut self, node: C, ndim: usize) -> Result<NodeIndex, String> {
        if let Some(n) = self.coord_systems.get(&node) {
            if n.ndim != ndim {
                return Err(format!(
                    "Existing coordinate system is {}D; new is {}D",
                    n.ndim, ndim
                ));
            }
            Ok(n.idx)
        } else {
            let idx = self.graph.add_node(node.clone());
            self.coord_systems.insert(node, NodeInfo { idx, ndim });
            Ok(idx)
        }
    }

    /// Returns whether the inverse edge was added, if it was requested.
    /// Fails if the new edge's dimensionality is inconsistent with existing edges.
    ///
    /// Weight is set to 0 if the transformation is an identity.
    ///
    /// Adding edges clears all cached paths.
    pub fn add_edge(
        &mut self,
        src: impl Into<C>,
        tgt: impl Into<C>,
        transform: Arc<dyn Transformation>,
        weight: f64,
        with_inverse: bool,
    ) -> Result<bool, String> {
        self.path_cache.clear_mut();

        let src_s = src.into();
        let tgt_s = tgt.into();

        // Do not add self-edges.
        if src_s == tgt_s {
            self.ensure_coord_system(src_s, transform.input_ndim())?;
            return Ok(true);
        }

        let u = self.ensure_coord_system(src_s, transform.input_ndim())?;
        let v = self.ensure_coord_system(tgt_s, transform.output_ndim())?;

        // Simplify identity transforms.
        let (t, w): (Arc<dyn Transformation>, f64) = if transform.is_identity() {
            (Arc::new(Identity::new(transform.input_ndim())), 0.0)
        } else {
            (transform, weight)
        };

        let mut added_inverse = false;
        // Add the inverse if requested, if it exists.
        if with_inverse && let Some(inverse) = t.invert() {
            self.graph.add_edge(v, u, Edge::new_cost(inverse, w));
            added_inverse = true;
        }

        self.graph.add_edge(u, v, Edge::new_cost(t, w));
        Ok(added_inverse)
    }

    fn best_edge(&self, src: NodeIndex, tgt: NodeIndex) -> Option<&Edge> {
        self.graph
            .edges_connecting(src, tgt)
            .min_by_key(|e| e.weight().cost)
            .map(|e| e.weight())
    }

    /// Get a transformation between two coordinate systems, if it exists.
    ///
    /// If the systems are equivalent, an [Identity] will be returned.
    /// Note that transforming with an [Identity] is a handful of memcpys which could be avoided.
    ///
    /// If the two values have a single non-identity edge between them, the transformation of that edge will be returned.
    /// Longer paths will return a [crate::Sequence].
    pub fn find_path(&self, from: C, to: C) -> Option<Arc<dyn Transformation>> {
        let start = self.coord_systems.get(&from)?;

        // if the source and target are the same, use an identity
        if from == to {
            return Some(Arc::new(Identity::new(start.ndim)));
        }

        let u = start.idx;
        let v = self.coord_systems.get(&to)?.idx;

        // if the path (or lack thereof) is cached, use that
        if let Some(maybe) = self.path_cache.get(&u, &v) {
            return maybe;
        }

        // If a direct edge exists, use it.
        // This is probably a negligible optimisation, simply saving astar's setup and first iteration.
        if let Some(e) = self.best_edge(u, v) {
            // If the edge is an identity, use that for performance.
            let t = if e.transform.is_identity() {
                Arc::new(Identity::new(start.ndim))
            } else {
                e.transform.clone()
            };
            self.path_cache.insert(u, v, Some(t.clone()));
            return Some(t);
        }

        let zero = OrderedFloat(0.0);

        // Find the shortest path between nodes.
        let Some((_cost, path)) = astar(&self.graph, u, |n| n == v, |e| e.weight().cost, |_| zero)
        else {
            // no path exists
            self.path_cache.insert(u, v, None);
            return None;
        };

        let t = match path.len() {
            0 => unreachable!("paths must include the start and end point"),
            1 => unreachable!("already checked for src=tgt"),
            2 => unreachable!("already checked for single-edge path"),
            n => {
                let mut builder = SequenceBuilder::with_capacity(n - 1);
                for ab in path.windows(2) {
                    builder
                        .add_arced(self.best_edge(ab[0], ab[1])?.transform.clone())
                        .expect("already checked dimensionality");
                }
                builder
                    .build_any()
                    .expect("already checked sequence length")
            }
        };

        self.path_cache.insert(u, v, Some(t.clone()));
        Some(t)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use crate::{TransformGraph, Transformation, Translate};

    /// ```text
    /// a <==> b <==> c
    ///         \---> d
    /// ```
    fn make_graph() -> TransformGraph<&'static str> {
        let mut tg = TransformGraph::default();
        tg.add_edge(
            "a",
            "b",
            Arc::new(Translate::try_new(&[1.0, 2.0]).unwrap()),
            1.0,
            true,
        )
        .unwrap();
        tg.add_edge(
            "b",
            "c",
            Arc::new(Translate::try_new(&[10.0, 20.0]).unwrap()),
            1.0,
            true,
        )
        .unwrap();
        tg.add_edge(
            "b",
            "d",
            Arc::new(Translate::try_new(&[100.0, 200.0]).unwrap()),
            1.0,
            false,
        )
        .unwrap();
        tg
    }

    fn check_transform(t: Arc<dyn Transformation>, input: &[f64], expected: &[f64]) {
        let mut out_buf = vec![f64::NAN; expected.len()];
        t.transform_into(input, &mut out_buf);
        assert_eq!(&out_buf, expected);
    }

    #[test]
    fn test_forward() {
        let tg = make_graph();
        let t = tg.find_path("a", "c").unwrap();
        check_transform(t, &[0.0, 0.0], &[11.0, 22.0]);

        let t2 = tg.find_path("a", "d").unwrap();
        check_transform(t2, &[0.0, 0.0], &[101.0, 202.0]);
    }

    #[test]
    fn test_reverse() {
        let tg = make_graph();
        let t = tg.find_path("c", "a").unwrap();
        check_transform(t, &[0.0, 0.0], &[-11.0, -22.0]);
    }

    #[test]
    fn test_no_reverse() {
        let tg = make_graph();
        assert!(tg.find_path("d", "a").is_none())
    }
}
