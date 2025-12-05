use std::{marker::PhantomData, sync::Arc};

use crate::{ShortVec, Transformation, indexer::Ravelled};
use smallvec::smallvec;

pub struct ChunkOffset {
    pub chunk_id: ShortVec<usize>,
    pub offset_idx: ShortVec<usize>,
}

pub trait ChunkedIndex<T, B: BoundedIndex<T>> {
    fn get_chunk_offset(&self, coord: &[usize]) -> Option<ChunkOffset>;
    fn get_chunk(&self, chunk_id: &[usize]) -> Option<&B>;
    fn extents(&self) -> &[usize];
}

pub trait BoundedIndex<T> {
    fn get(&self, coord: &[usize]) -> Option<T>;
    fn get_unchecked(&self, coord: &[usize]) -> T;

    fn bulk_get_into(&self, coord: &[&[usize]], buf: &mut [Option<T>]) {
        for (c, b) in coord.iter().zip(buf.iter_mut()) {
            *b = self.get(c);
        }
    }

    fn bulk_get_into_unchecked(&self, coord: &[&[usize]], buf: &mut [T]) {
        for (c, b) in coord.iter().zip(buf.iter_mut()) {
            *b = self.get_unchecked(c);
        }
    }

    fn column_get_into(&self, columns: &[&[usize]], buf: &mut [Option<T>]) {
        let mut coord = vec![0; columns.len()];
        for idx in 0..columns[0].len() {
            for (coord_val, col) in coord.iter_mut().zip(columns.iter()) {
                *coord_val = col[idx];
            }
            buf[idx] = self.get(&coord);
        }
    }

    fn column_get_into_unchecked(&self, columns: &[&[usize]], buf: &mut [T]) {
        let mut coord = vec![0; columns.len()];
        for idx in 0..columns[0].len() {
            for (coord_val, col) in coord.iter_mut().zip(columns.iter()) {
                *coord_val = col[idx];
            }
            buf[idx] = self.get_unchecked(&coord);
        }
    }
    fn ndim(&self) -> usize {
        self.extents().len()
    }
    fn extents(&self) -> &[usize];
}

pub trait UnboundedIndex<T> {
    fn get(&self, coord: &[isize]) -> T;
    fn bulk_get_into(&self, coord: &[&[isize]], buf: &mut [T]) {
        for (c, b) in coord.iter().zip(buf.iter_mut()) {
            *b = self.get(c);
        }
    }
    fn column_get_into(&self, columns: &[&[isize]], buf: &mut [T]) {
        let mut coord = vec![isize::MAX; columns.len()];
        for (idx, b) in buf.iter_mut().enumerate() {
            for (c, col) in coord.iter_mut().zip(columns.iter()) {
                *c = col[idx];
            }
            *b = self.get(&coord);
        }
    }
    fn ndim(&self) -> usize;
}

pub trait RealIndex<T> {
    fn get(&self, coord: &[f64]) -> T;

    fn bulk_get_into(&self, coords: &[&[f64]], buf: &mut [T]) {
        for (c, b) in coords.iter().zip(buf.iter_mut()) {
            *b = self.get(c);
        }
    }

    fn column_get_into(&self, columns: &[&[f64]], buf: &mut [T]) {
        let mut coord = vec![f64::NAN; columns.len()];
        for idx in 0..columns[0].len() {
            for (dim_idx, col) in columns.iter().enumerate() {
                coord[dim_idx] = col[idx];
            }
            buf[0] = self.get(&coord);
        }
    }

    fn ndim(&self) -> usize;
}

pub struct Const<T: Copy, A: BoundedIndex<T>> {
    constant: T,
    bounded: A,
    extents: Vec<isize>,
}

impl<T: Copy, A: BoundedIndex<T>> Const<T, A> {
    pub fn new(bounded: A, constant: T) -> Self {
        let extents = bounded.extents().iter().map(|u| *u as isize).collect();
        Self {
            bounded,
            constant,
            extents,
        }
    }
}

fn unbound_to_bound_elem(c: &isize, max: &isize) -> Option<usize> {
    if c.is_negative() || c >= max {
        return None;
    }
    Some(*c as usize)
}

fn unbound_to_bound_coord(coord: &[isize], extents: &[isize], buf: &mut [usize]) -> bool {
    unbound_to_bound_iter(coord, extents, buf)
}

fn unbound_to_bound_iter<'a>(
    coord: impl IntoIterator<Item = &'a isize>,
    extents: impl IntoIterator<Item = &'a isize>,
    buf: &mut [usize],
) -> bool {
    for ((c, max), b) in coord
        .into_iter()
        .zip(extents.into_iter())
        .zip(buf.iter_mut())
    {
        let Some(c2) = unbound_to_bound_elem(c, max) else {
            return false;
        };
        *b = c2;
    }
    true
}

impl<T: Copy + Default, A: BoundedIndex<T>> UnboundedIndex<T> for Const<T, A> {
    fn get(&self, coord: &[isize]) -> T {
        let mut new_coord: ShortVec<usize> = smallvec![usize::MAX; coord.len()];
        if unbound_to_bound_coord(coord, &self.extents, &mut new_coord) {
            self.bounded.get_unchecked(&new_coord)
        } else {
            self.constant
        }
    }

    fn bulk_get_into(&self, coords: &[&[isize]], mut buf: &mut [T]) {
        let mut new_coords = Vec::with_capacity(coords.len());
        let mut indices = Vec::with_capacity(coords.len());
        for (idx, (coord, b)) in coords.iter().zip(buf.iter_mut()).enumerate() {
            let mut new_coord: ShortVec<usize> = smallvec![usize::MAX; coord.len()];
            if unbound_to_bound_coord(coord, &self.extents, &mut new_coord) {
                new_coords.push(new_coord);
                indices.push(idx);
            } else {
                *b = self.constant;
            }
        }
        if new_coords.is_empty() {
            return;
        }

        let new_coord_refs: Vec<_> = new_coords.iter().map(|c| c.as_ref()).collect();
        if new_coord_refs.len() == coords.len() {
            self.bounded
                .bulk_get_into_unchecked(&new_coord_refs, &mut buf)
        } else {
            let mut out_buf = vec![Default::default(); new_coords.len()];
            self.bounded
                .bulk_get_into_unchecked(&new_coord_refs, &mut out_buf);
            for (idx, val) in indices.into_iter().zip(out_buf.into_iter()) {
                buf[idx] = val;
            }
        }
    }

    fn column_get_into(&self, columns: &[&[isize]], buf: &mut [T]) {
        let mut new_cols: Vec<Vec<usize>> = vec![Vec::with_capacity(columns[0].len()); self.ndim()];
        let mut skip = vec![false; columns[0].len()];
        let mut coord = vec![usize::MAX; columns.len()];

        for ((idx, b), s) in (0..columns[0].len())
            .into_iter()
            .zip(buf.iter_mut())
            .zip(skip.iter_mut())
        {
            if unbound_to_bound_iter(columns.iter().map(|c| &c[idx]), &self.extents, &mut coord) {
                new_cols
                    .iter_mut()
                    .zip(coord.iter())
                    .for_each(|(col, c)| col.push(*c));
            } else {
                *b = self.constant;
                *s = true;
            }
        }
        let unskipped = new_cols[0].len();
        if unskipped == 0 {
            return;
        }

        let col_refs: Vec<&[usize]> = new_cols.iter().map(|c| c.as_ref()).collect();
        let mut inner_buf = vec![Default::default(); unskipped];
        self.bounded
            .column_get_into_unchecked(&col_refs, &mut inner_buf);
        for (b, res) in skip
            .into_iter()
            .zip(buf.iter_mut())
            .filter_map(|(s, b)| (!s).then_some(b))
            .zip(inner_buf.into_iter())
        {
            *b = res;
        }
    }

    fn ndim(&self) -> usize {
        self.bounded.ndim()
    }
}

pub struct NearestNeighbour<T, U: UnboundedIndex<T>> {
    unbounded: U,
    _t: PhantomData<T>,
}

impl<T, U: UnboundedIndex<T>> NearestNeighbour<T, U> {
    pub fn new(unbounded: U) -> Self {
        Self {
            unbounded,
            _t: Default::default(),
        }
    }
}

impl<T, U: UnboundedIndex<T>> From<U> for NearestNeighbour<T, U> {
    fn from(value: U) -> Self {
        Self::new(value)
    }
}

impl<T, U: UnboundedIndex<T>> RealIndex<T> for NearestNeighbour<T, U> {
    fn get(&self, coord: &[f64]) -> T {
        let ints: ShortVec<isize> = coord.iter().map(|f| f.round_ties_even() as isize).collect();
        self.unbounded.get(&ints)
    }

    fn bulk_get_into(&self, coords: &[&[f64]], buf: &mut [T]) {
        let new_coords: Vec<ShortVec<isize>> = coords
            .iter()
            .map(|coord| coord.iter().map(|f| f.round_ties_even() as isize).collect())
            .collect();
        let new_coord_refs: Vec<&[isize]> = new_coords.iter().map(|n| n.as_ref()).collect();
        self.unbounded.bulk_get_into(&new_coord_refs, buf);
    }

    fn column_get_into(&self, columns: &[&[f64]], buf: &mut [T]) {
        let new_columns: Vec<Vec<_>> = columns
            .iter()
            .map(|col| col.iter().map(|f| f.round_ties_even() as isize).collect())
            .collect();
        let refs: Vec<&[isize]> = new_columns.iter().map(|n| n.as_ref()).collect();
        self.unbounded.column_get_into(&refs, buf);
    }

    fn ndim(&self) -> usize {
        self.unbounded.ndim()
    }
}

pub struct Transformed<T, R: RealIndex<T>> {
    indexer: R,
    transform: Arc<dyn Transformation>,
    _t: PhantomData<T>,
}

impl<T, R: RealIndex<T>> Transformed<T, R> {
    pub fn try_new(indexer: R, transform: Arc<dyn Transformation>) -> Result<Self, String> {
        if transform.output_ndim() != indexer.ndim() {
            return Err("Dimensionality mismatch".into());
        }
        Ok(Self {
            indexer,
            transform,
            _t: Default::default(),
        })
    }
}

impl<T, R: RealIndex<T>> RealIndex<T> for Transformed<T, R> {
    fn get(&self, coord: &[f64]) -> T {
        let mut new_coord: ShortVec<f64> = smallvec![f64::NAN; self.transform.output_ndim()];
        self.transform.transform_into(coord, &mut new_coord);
        self.indexer.get(&new_coord)
    }

    fn bulk_get_into(&self, coords: &[&[f64]], buf: &mut [T]) {
        let mut ravelled = Ravelled::new_full(self.transform.output_ndim(), coords.len(), f64::NAN);
        self.transform
            .bulk_transform_into(coords, &mut ravelled.chunks_mut().collect::<Vec<_>>());
        self.indexer
            .bulk_get_into(&ravelled.chunks().collect::<Vec<_>>(), buf);
    }

    fn column_get_into(&self, columns: &[&[f64]], buf: &mut [T]) {
        let mut ravelled = Ravelled::new_full(columns[0].len(), columns.len(), f64::NAN);
        self.transform
            .column_transform_into(columns, &mut ravelled.chunks_mut().collect::<Vec<_>>());
        self.indexer
            .column_get_into(&ravelled.chunks().collect::<Vec<_>>(), buf);
    }

    fn ndim(&self) -> usize {
        self.transform.input_ndim()
    }
}
