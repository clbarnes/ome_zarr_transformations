#[cfg(feature = "image")]
mod idx_image;
mod sampler;
#[cfg(feature = "image")]
pub use idx_image::ImageWrapper;
pub use sampler::Sampler;
#[cfg(feature = "ndarray")]
mod idx_ndarray;
#[cfg(feature = "ndarray")]
pub use idx_ndarray::{ArrayRefWrapper, ArrayViewWrapper, ArrayWrapper};
mod idx_chunked;
pub use idx_chunked::ChunkedIndexer;
pub mod value;

#[derive(Debug, Clone)]
pub(crate) struct Ravelled<T> {
    data: Vec<T>,
    chunk_size: usize,
}

impl<T: Clone> Ravelled<T> {
    pub fn new_full(chunk_size: usize, n_chunks: usize, fill: T) -> Self {
        Self {
            data: vec![fill; chunk_size * n_chunks],
            chunk_size,
        }
    }
}

impl<T> Ravelled<T> {
    pub fn new_data(chunk_size: usize, data: Vec<T>) -> Result<Self, String> {
        if !data.len().is_multiple_of(chunk_size) {
            return Err("data vec is not a multiple of chunk size".into());
        }
        Ok(Ravelled { data, chunk_size })
    }

    pub fn chunks(&self) -> impl Iterator<Item = &[T]> {
        self.data.chunks_exact(self.chunk_size)
    }

    pub fn chunks_mut(&mut self) -> impl Iterator<Item = &mut [T]> {
        self.data.chunks_exact_mut(self.chunk_size)
    }
}
