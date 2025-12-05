use super::value::BoundedIndex;
use ndarray::{ArrayD, ArrayRefD, ArrayViewD};

pub struct ArrayRefWrapper<T: Copy>(ArrayRefD<T>);

impl<T: Copy> BoundedIndex<T> for ArrayRefWrapper<T> {
    fn get(&self, coord: &[usize]) -> Option<T> {
        self.0.get(coord).copied()
    }

    fn get_unchecked(&self, coord: &[usize]) -> T {
        self.0[coord]
    }

    fn extents(&self) -> &[usize] {
        self.0.shape()
    }
}

pub struct ArrayWrapper<T: Copy>(ArrayD<T>);

impl<T: Copy> BoundedIndex<T> for ArrayWrapper<T> {
    fn get(&self, coord: &[usize]) -> Option<T> {
        self.0.get(coord).copied()
    }

    fn get_unchecked(&self, coord: &[usize]) -> T {
        self.0[coord]
    }

    fn extents(&self) -> &[usize] {
        self.0.shape()
    }
}

pub struct ArrayViewWrapper<'a, T: Copy>(ArrayViewD<'a, T>);

impl<'a, T: Copy> BoundedIndex<T> for ArrayViewWrapper<'a, T> {
    fn get(&self, coord: &[usize]) -> Option<T> {
        self.0.get(coord).copied()
    }

    fn get_unchecked(&self, coord: &[usize]) -> T {
        self.0[coord]
    }

    fn extents(&self) -> &[usize] {
        self.0.shape()
    }
}
