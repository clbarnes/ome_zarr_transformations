//! Wrappers for coordinate transformations.
//!
//! Coordinate transformations try to minimise allocations
//! so that they can be called in tight loops in a variety of situations.
//! But writing the same boilerplate to pre-allocate output buffers is annoying,
//! so these types handle that.
use std::{iter, sync::Arc};

use crate::Transformation;

pub trait AllocatingTransformer<
    T: Transformation,
    Coord: AsRef<[f64]> + AsMut<[f64]>,
    Col: AsRef<[f64]> + AsMut<[f64]>,
>
{
    /// Get the transformation used by this transformer.
    fn transformation(&self) -> &T;

    /// Create a buffer to hold one coordinate of length `ndim`.
    fn coord_buffer(&self, ndim: usize) -> Coord;

    fn col_buffer(&self, len: usize) -> Col;

    /// Transform a given point into a newly-allocated buffer.
    fn transform(&self, pt: &[f64]) -> Coord {
        let t = self.transformation();
        let mut out = self.coord_buffer(t.output_ndim());
        t.transform_into(pt, out.as_mut());
        out
    }

    fn bulk_transform(&self, pts: &[&[f64]]) -> Vec<Coord> {
        let t = self.transformation();
        let mut out = Vec::from_iter(
            iter::repeat_with(|| self.coord_buffer(t.output_ndim())).take(pts.len()),
        );
        let mut out_refs: Vec<&mut [_]> = out.iter_mut().map(|c| c.as_mut()).collect();
        t.bulk_transform_into(pts, &mut out_refs);
        out
    }

    fn column_transform(&self, columns: &[&[f64]]) -> Vec<Col> {
        let Some(n_pts) = columns.first().map(|p| p.len()) else {
            return vec![];
        };
        let t = self.transformation();
        let mut out =
            Vec::from_iter(iter::repeat_with(|| self.col_buffer(n_pts)).take(t.output_ndim()));
        let mut out_refs: Vec<&mut [_]> = out.iter_mut().map(|v| v.as_mut()).collect();
        t.column_transform_into(columns, &mut out_refs);
        out
    }
}

pub struct CustomAllocatingTransformer<
    T: Transformation,
    Coord: AsRef<[f64]> + AsMut<[f64]>,
    Col: AsRef<[f64]> + AsMut<[f64]>,
    FCoord: Fn(usize) -> Coord,
    FCol: Fn(usize) -> Col,
> {
    transform: Arc<T>,
    f_coord: FCoord,
    f_col: FCol,
}

impl<
    T: Transformation,
    Coord: AsRef<[f64]> + AsMut<[f64]>,
    Col: AsRef<[f64]> + AsMut<[f64]>,
    FCoord: Fn(usize) -> Coord,
    FCol: Fn(usize) -> Col,
> CustomAllocatingTransformer<T, Coord, Col, FCoord, FCol>
{
    pub fn new(transform: Arc<T>, f_coord: FCoord, f_col: FCol) -> Self {
        Self {
            transform,
            f_coord,
            f_col,
        }
    }
}

impl<
    T: Transformation,
    Coord: AsRef<[f64]> + AsMut<[f64]>,
    Col: AsRef<[f64]> + AsMut<[f64]>,
    FCoord: Fn(usize) -> Coord,
    FCol: Fn(usize) -> Col,
> AllocatingTransformer<T, Coord, Col>
    for CustomAllocatingTransformer<T, Coord, Col, FCoord, FCol>
{
    fn transformation(&self) -> &T {
        self.transform.as_ref()
    }

    fn coord_buffer(&self, ndim: usize) -> Coord {
        (self.f_coord)(ndim)
    }

    fn col_buffer(&self, len: usize) -> Col {
        (self.f_col)(len)
    }
}
