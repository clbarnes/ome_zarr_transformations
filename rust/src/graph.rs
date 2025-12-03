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

type PathCache = HashMap<(NodeIndex, NodeIndex), Option<Arc<dyn Transformation>>>;

#[derive(Debug, Default)]
pub struct TransformGraph<C: std::hash::Hash + Eq + Clone> {
    graph: StableDiGraph<C, Edge>,
    coord_systems: HashMap<C, NodeInfo>,
    path_cache: RwLock<PathCache>,
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

    /// Returns whether the inverse edge was added.
    /// Fails if the new edge's dimensionality is inconsistent with existing edges.
    pub fn add_edge(
        &mut self,
        src: impl Into<C>,
        tgt: impl Into<C>,
        transform: impl Into<Arc<dyn Transformation>>,
        weight: f64,
        with_inverse: bool,
    ) -> Result<bool, String> {
        self.clear_cache();
        let t = transform.into();

        let u = self.ensure_coord_system(src.into(), t.input_ndim())?;
        let v = self.ensure_coord_system(tgt.into(), t.output_ndim())?;

        let mut added_inverse = false;
        if with_inverse {
            if let Some(inverse) = t.invert() {
                self.graph.add_edge(v, u, Edge::new_cost(inverse, weight));
                added_inverse = true;
            }
        }

        self.graph.add_edge(u, v, Edge::new_cost(t, weight));
        Ok(added_inverse)
    }

    fn best_edge(&self, src: NodeIndex, tgt: NodeIndex) -> Option<&Edge> {
        self.graph
            .edges_connecting(src, tgt)
            .min_by_key(|e| e.weight().cost)
            .map(|e| e.weight())
    }

    fn cache_get(
        &self,
        src: &NodeIndex,
        tgt: &NodeIndex,
    ) -> Option<Option<Arc<dyn Transformation>>> {
        let outer = self.path_cache.read().expect("should not be poisonned");
        outer.get(&(*src, *tgt)).map(|t| t.as_ref().cloned())
    }

    fn cache_insert(
        &self,
        src: NodeIndex,
        tgt: NodeIndex,
        t: Option<Arc<dyn Transformation>>,
    ) -> Option<Option<Arc<dyn Transformation>>> {
        self.path_cache
            .write()
            .expect("should not be poisonned")
            .insert((src, tgt), t)
    }

    fn clear_cache(&mut self) {
        self.path_cache.get_mut().unwrap().clear();
    }

    pub fn find_path(
        &self,
        from: impl AsRef<C>,
        to: impl AsRef<C>,
    ) -> Option<Arc<dyn Transformation>> {
        let from_ref = from.as_ref();
        let to_ref = to.as_ref();

        let start = self.coord_systems.get(from_ref)?;

        if from_ref == to_ref {
            return Some(Arc::new(Identity::new(start.ndim)));
        }

        let u = start.idx;
        let v = self.coord_systems.get(to_ref)?.idx;

        if let Some(maybe) = self.cache_get(&u, &v) {
            return maybe;
        }

        let zero = OrderedFloat(0.0);
        let Some((_cost, path)) = astar(&self.graph, u, |n| n == v, |e| e.weight().cost, |_| zero)
        else {
            self.cache_insert(u, v, None);
            return None;
        };

        let t = match path.len() {
            0 | 1 => unreachable!(),
            2 => self
                .best_edge(path[0], path[1])
                .map(|e| e.transform.clone())
                .expect("already checked for path existence"),
            n => {
                let mut builder = SequenceBuilder::with_capacity(n - 1);
                for ab in path.windows(2) {
                    builder
                        .add_arced(self.best_edge(ab[0], ab[1])?.transform.clone())
                        .expect("already checked dimensionality");
                }
                Arc::new(builder.build().expect("already checked sequence length"))
            }
        };

        self.cache_insert(u, v, Some(t.clone()));
        Some(t)
    }
}
