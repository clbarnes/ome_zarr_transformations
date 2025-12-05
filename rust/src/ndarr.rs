use crate::indexer::value::BoundedIndex;

pub trait Layout {
    fn shape(&self) -> &[usize];
    fn ndim(&self) -> usize {
        self.shape().len()
    }
    fn numel(&self) -> usize {
        self.shape().iter().product()
    }
    fn linear_idx(&self, index: &[usize]) -> Option<usize>;
    fn contiguous_dimension(&self) -> Option<usize>;
    // fn array_idx(&self, linear: usize, buf: &mut [usize]);
}

#[derive(Debug, Clone)]
pub struct ColumnMajor {
    shape: Vec<usize>,
}

impl ColumnMajor {
    pub fn new(shape: &[usize]) -> Self {
        Self {
            shape: shape.to_vec(),
        }
    }
}

impl Layout for ColumnMajor {
    fn shape(&self) -> &[usize] {
        &self.shape
    }

    fn linear_idx(&self, index: &[usize]) -> Option<usize> {
        let mut total = 0;
        let mut prev_s = 1;
        for (s, i) in self.shape.iter().zip(index.iter()) {
            if i >= s {
                return None;
            }
            total += i * prev_s;
            prev_s = *s;
        }
        Some(total)
    }

    fn contiguous_dimension(&self) -> Option<usize> {
        Some(0)
    }

    // fn array_idx(&self, linear: usize, buf: &mut [usize]) {
    //     todo!()
    // }
}

#[derive(Debug, Clone)]
pub struct RowMajor {
    shape: Vec<usize>,
}

impl RowMajor {
    pub fn new(shape: &[usize]) -> Self {
        Self {
            shape: shape.to_vec(),
        }
    }
}

impl Layout for RowMajor {
    fn shape(&self) -> &[usize] {
        &self.shape
    }

    fn contiguous_dimension(&self) -> Option<usize> {
        Some(self.shape.len() - 1)
    }

    fn linear_idx(&self, index: &[usize]) -> Option<usize> {
        let mut total = 0;
        let mut prev_s = 1;
        for (s, i) in self.shape.iter().rev().zip(index.iter().rev()) {
            if i >= s {
                return None;
            }
            total += i * prev_s;
            prev_s = *s;
        }
        Some(total)
    }

    // fn array_idx(&self, linear: usize, buf: &mut [usize]) {
    //     todo!()
    // }
}

#[derive(Debug, Clone)]
pub struct VecNdArray<T, L: Layout> {
    data: Vec<T>,
    layout: L,
}

impl<T, L: Layout> VecNdArray<T, L> {
    pub fn new(data: Vec<T>, layout: L) -> Result<Self, String> {
        if data.len() != layout.numel() {
            return Err("data does not match layout".into());
        }
        Ok(Self::new_unchecked(data, layout))
    }

    pub fn new_unchecked(data: Vec<T>, layout: L) -> Self {
        Self { data, layout }
    }

    pub fn chunks(&self) -> Option<impl Iterator<Item = &[T]>> {
        let d = self.layout.contiguous_dimension()?;
        let s = self.layout.shape()[d];
        Some(self.data.chunks_exact(s))
    }

    pub fn chunks_mut(&mut self) -> Option<impl Iterator<Item = &mut [T]>> {
        let d = self.layout.contiguous_dimension()?;
        let s = self.layout.shape()[d];
        Some(self.data.chunks_exact_mut(s))
    }

    pub fn into_data(self) -> Vec<T> {
        self.data
    }
}

impl<T: Copy, L: Layout> BoundedIndex<T> for VecNdArray<T, L> {
    fn get(&self, coord: &[usize]) -> Option<T> {
        self.layout.linear_idx(coord).map(|idx| self.data[idx])
    }

    fn get_unchecked(&self, coord: &[usize]) -> T {
        self.get(coord).unwrap()
    }

    fn extents(&self) -> &[usize] {
        self.layout.shape()
    }
}

impl<T, L: Layout> AsRef<[T]> for VecNdArray<T, L> {
    fn as_ref(&self) -> &[T] {
        self.data.as_ref()
    }
}

impl<T, L: Layout> AsMut<[T]> for VecNdArray<T, L> {
    fn as_mut(&mut self) -> &mut [T] {
        self.data.as_mut_slice()
    }
}
