use pyo3::prelude::*;

mod traits;
pub use traits::PyTransform;

/// A Python module implemented in Rust.
#[pymodule]
mod ome_zarr_transformations_rs {
    use pyo3::prelude::*;

    /// Formats the sum of two numbers as string.
    #[pyfunction]
    fn sum_as_string(a: usize, b: usize) -> PyResult<String> {
        Ok((a + b).to_string())
    }
}
