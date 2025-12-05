use std::{collections::BTreeMap, marker::PhantomData};

use crate::{
    ShortVec,
    indexer::value::{BoundedIndex, ChunkOffset, ChunkedIndex},
};

pub struct ChunkedIndexer<T, B: BoundedIndex<T>, C: ChunkedIndex<T, B>> {
    chunked: C,
    _t: PhantomData<T>,
    _b: PhantomData<B>,
}

impl<T, B: BoundedIndex<T>, C: ChunkedIndex<T, B>> ChunkedIndexer<T, B, C> {
    pub fn new(chunked: C) -> Self {
        Self {
            chunked,
            _t: Default::default(),
            _b: Default::default(),
        }
    }
}

type CoordToOpt<'a, T> = BTreeMap<ShortVec<usize>, ShortVec<&'a mut Option<T>>>;

impl<T: Clone, B: BoundedIndex<T>, C: ChunkedIndex<T, B>> BoundedIndex<T>
    for ChunkedIndexer<T, B, C>
{
    fn get(&self, coord: &[usize]) -> Option<T> {
        let co = self.chunked.get_chunk_offset(coord)?;
        let c = self.chunked.get_chunk(&co.chunk_id)?;
        c.get(&co.offset_idx)
    }

    fn bulk_get_into(&self, coords: &[&[usize]], buf: &mut [Option<T>]) {
        // map of chunk ID to offset coordinate to mutable references into buf
        let mut chunks: BTreeMap<ShortVec<usize>, CoordToOpt<T>> = Default::default();
        for (coord, b) in coords.iter().zip(buf.iter_mut()) {
            let Some(co) = self.chunked.get_chunk_offset(coord) else {
                *b = None;
                continue;
            };
            chunks
                .entry(co.chunk_id)
                .or_default()
                .entry(co.offset_idx)
                .or_default()
                .push(b);
        }
        let mut inner_b = vec![];
        for (chunk_id, coord_to_b) in chunks {
            // would be nice to promote this allocation out of the loop,
            // but it contains refs and we can't prove to the compiler
            // that they are cleared within the loop
            let Some(chunk) = self.chunked.get_chunk(&chunk_id) else {
                for b in coord_to_b.into_values().flat_map(|bs| bs.into_iter()) {
                    *b = None;
                }
                continue;
            };
            {
                let mut coord_refs = vec![];
                for c in coord_to_b.keys() {
                    coord_refs.push(c.as_slice());
                    inner_b.push(None);
                }
                chunk.bulk_get_into(&coord_refs, &mut inner_b);
            }
            for (val_from, val_into) in inner_b.drain(..).zip(coord_to_b.into_values()) {
                for vi in val_into {
                    *vi = val_from.clone();
                }
            }
            inner_b.clear();
        }
    }

    fn bulk_get_into_unchecked(&self, coords: &[&[usize]], buf: &mut [T]) {
        let fill = buf.first().cloned().unwrap();
        // map of chunk ID to offset coordinate to mutable references into buf
        let mut chunks: BTreeMap<ShortVec<usize>, BTreeMap<ShortVec<usize>, ShortVec<&mut T>>> =
            Default::default();
        for (coord, b) in coords.iter().zip(buf.iter_mut()) {
            let co = self.chunked.get_chunk_offset(coord).unwrap();
            chunks
                .entry(co.chunk_id)
                .or_default()
                .entry(co.offset_idx)
                .or_default()
                .push(b);
        }
        let mut inner_b = vec![];
        for (chunk_id, coord_to_b) in chunks {
            // would be nice to promote this allocation out of the loop,
            // but it contains refs and we can't prove to the compiler
            // that they are cleared within the loop
            let chunk = self.chunked.get_chunk(&chunk_id).unwrap();
            {
                let mut coord_refs = vec![];
                for c in coord_to_b.keys() {
                    coord_refs.push(c.as_slice());
                    inner_b.push(fill.clone());
                }
                chunk.bulk_get_into_unchecked(&coord_refs, &mut inner_b);
            }
            for (val_from, val_into) in inner_b.drain(..).zip(coord_to_b.into_values()) {
                for vi in val_into {
                    *vi = val_from.clone();
                }
            }
            inner_b.clear();
        }
    }

    fn column_get_into(&self, columns: &[&[usize]], buf: &mut [Option<T>]) {
        let mut chunks: BTreeMap<ShortVec<usize>, CoordToOpt<T>> = Default::default();
        let n_dim = columns.len();
        let n_coords = columns[0].len();
        let mut coord = vec![];
        for (coord_idx, b) in (0..n_coords).zip(buf.iter_mut()) {
            coord.clear();
            for col in columns.iter() {
                coord.push(col[coord_idx]);
            }
            let Some(co) = self.chunked.get_chunk_offset(&coord) else {
                *b = None;
                continue;
            };
            chunks
                .entry(co.chunk_id)
                .or_default()
                .entry(co.offset_idx)
                .or_default()
                .push(b);
        }

        let mut inner_coords: Vec<Vec<usize>> = vec![vec![]; n_dim];
        let mut inner_b = vec![];
        for (chunk_id, coord_to_b) in chunks {
            // would be nice to promote this allocation out of the loop,
            // but it contains refs and we can't prove to the compiler
            // that they are cleared within the loop
            let Some(chunk) = self.chunked.get_chunk(&chunk_id) else {
                for b in coord_to_b.into_values().flat_map(|bs| bs.into_iter()) {
                    *b = None;
                }
                continue;
            };
            for coord in coord_to_b.keys() {
                for (col, c) in inner_coords.iter_mut().zip(coord.iter()) {
                    col.push(*c);
                }
                inner_b.push(None);
            }
            {
                let coord_refs: Vec<_> = inner_coords.iter().map(|coord| coord.as_ref()).collect();
                chunk.column_get_into(&coord_refs, &mut inner_b);
            }
            for (val_from, val_into) in inner_b.drain(..).zip(coord_to_b.into_values()) {
                for vi in val_into {
                    *vi = val_from.clone();
                }
            }
            inner_b.clear();
        }
    }

    fn column_get_into_unchecked(&self, columns: &[&[usize]], buf: &mut [T]) {
        let fill = buf.first().cloned().unwrap();
        let mut chunks: BTreeMap<ShortVec<usize>, BTreeMap<ShortVec<usize>, ShortVec<&mut T>>> =
            Default::default();
        let n_dim = columns.len();
        let n_coords = columns[0].len();
        let mut coord = vec![];
        for (coord_idx, b) in (0..n_coords).zip(buf.iter_mut()) {
            coord.clear();
            for col in columns.iter() {
                coord.push(col[coord_idx]);
            }
            let co = self.chunked.get_chunk_offset(&coord).unwrap();
            chunks
                .entry(co.chunk_id)
                .or_default()
                .entry(co.offset_idx)
                .or_default()
                .push(b);
        }

        let mut inner_coords: Vec<Vec<usize>> = vec![vec![]; n_dim];
        let mut inner_b = vec![];
        for (chunk_id, coord_to_b) in chunks {
            // would be nice to promote this allocation out of the loop,
            // but it contains refs and we can't prove to the compiler
            // that they are cleared within the loop
            let chunk = self.chunked.get_chunk(&chunk_id).unwrap();
            for coord in coord_to_b.keys() {
                for (col, c) in inner_coords.iter_mut().zip(coord.iter()) {
                    col.push(*c);
                }
                inner_b.push(fill.clone());
            }
            {
                let coord_refs: Vec<_> = inner_coords.iter().map(|coord| coord.as_ref()).collect();
                chunk.column_get_into_unchecked(&coord_refs, &mut inner_b);
            }
            for (val_from, val_into) in inner_b.drain(..).zip(coord_to_b.into_values()) {
                for vi in val_into {
                    *vi = val_from.clone();
                }
            }
            inner_b.clear();
        }
    }

    fn get_unchecked(&self, coord: &[usize]) -> T {
        let co = self.chunked.get_chunk_offset(coord).unwrap();
        let c = self.chunked.get_chunk(&co.chunk_id).unwrap();
        c.get_unchecked(&co.offset_idx)
    }

    fn extents(&self) -> &[usize] {
        self.chunked.extents()
    }
}

#[allow(unused)]
#[derive(Debug, Clone)]
pub struct RegularChunker {
    chunk_shape: Vec<usize>,
    n_chunks: Vec<usize>,
}

#[allow(unused)]
impl RegularChunker {
    pub fn new(chunk_shape: &[usize], n_chunks: &[usize]) -> Result<Self, String> {
        if chunk_shape.len() != n_chunks.len() {
            return Err("Inconsistent dimension".into());
        }
        Ok(Self {
            chunk_shape: chunk_shape.to_vec(),
            n_chunks: n_chunks.to_vec(),
        })
    }

    pub fn get_chunk_idx(&self, coord: &[usize]) -> Option<ChunkOffset> {
        let mut chunk_id: ShortVec<usize> = smallvec::smallvec![usize::MAX; coord.len()];
        let mut offset_idx: ShortVec<usize> = smallvec::smallvec![usize::MAX; coord.len()];
        for ((((c, cs), nc), ci), oi) in coord
            .iter()
            .zip(self.chunk_shape.iter())
            .zip(self.n_chunks.iter())
            .zip(chunk_id.iter_mut())
            .zip(offset_idx.iter_mut())
        {
            *ci = c / cs;
            if *ci >= *nc {
                return None;
            }
            *oi = c % cs;
        }
        Some(ChunkOffset {
            chunk_id,
            offset_idx,
        })
    }
}
