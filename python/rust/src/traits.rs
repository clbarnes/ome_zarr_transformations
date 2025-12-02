use std::sync::Arc;

use numpy::ndarray::{ArrayD, ShapeBuilder};
use numpy::{IntoPyArray, PyArrayDyn, PyReadonlyArrayDyn, PyUntypedArrayMethods};
use ome_zarr_transformations::Transformation;
use pyo3::{Bound, Python};

pub struct PyTransform {
    transform: Arc<dyn Transformation>,
}

impl PyTransform {
    /// Transform for an input where the elements of a single coordinate are contiguous,
    /// e.g. `xyzxyzxyzxyz` in memory.
    ///
    /// The order of the array, and the shape of any other dimensions, does not matter.
    pub fn transform_numpy_coord_contiguous<'py>(
        &self,
        python: Python<'py>,
        input_arr: PyReadonlyArrayDyn<'py, f64>,
    ) -> Bound<'py, PyArrayDyn<f64>> {
        let in_shape = input_arr.shape();
        let mut out_shape = in_shape.to_vec();

        let in_ndim = self.transform.input_ndim();
        let out_ndim = self.transform.output_ndim();

        let arr = input_arr.as_array();

        let is_c = if arr.is_standard_layout() {
            let len = out_shape.len();
            out_shape[len - 1] = out_ndim;
            true
        } else if arr.t().is_standard_layout() {
            out_shape[0] = out_ndim;
            false
        } else {
            unimplemented!("non-contiguous")
        };

        let slice = arr.as_slice_memory_order().expect("should be contiguous");

        let coords: Vec<_> = slice.chunks(in_ndim).collect();
        let mut out = vec![f64::NAN; coords.len() * out_ndim];
        let mut buf: Vec<_> = out.chunks_mut(out_ndim).collect();

        self.transform.bulk_transform_into(&coords, &mut buf);

        let new_arr =
            ArrayD::from_shape_vec(out_shape.set_f(!is_c), out).expect("should be correct shape");
        new_arr.into_pyarray(python)
    }

    /// Transform for an input where the elements of a single dimension are contiguous,
    /// e.g. `xxxxyyyyzzzz` in memory.
    ///
    /// The order of the array, and the shape of any other dimensions, does not matter.
    pub fn transform_numpy_dim_contiguous<'py>(
        &self,
        python: Python<'py>,
        input_arr: PyReadonlyArrayDyn<'py, f64>,
    ) -> Bound<'py, PyArrayDyn<f64>> {
        let in_shape = input_arr.shape();
        let mut out_shape = in_shape.to_vec();

        let in_ndim = self.transform.input_ndim();
        let out_ndim = self.transform.output_ndim();

        let arr = input_arr.as_array();

        let is_c = if arr.is_standard_layout() {
            out_shape[0] = out_ndim;
            true
        } else if arr.t().is_standard_layout() {
            let len = out_shape.len();
            out_shape[len - 1] = out_ndim;
            false
        } else {
            unimplemented!("non-contiguous")
        };

        let slice = arr.as_slice_memory_order().expect("should be contiguous");
        let n_pts = slice.len() / in_ndim;
        let columns: Vec<_> = slice.chunks(n_pts).collect();

        let mut out = vec![f64::NAN; n_pts * out_ndim];
        let mut buf: Vec<_> = out.chunks_mut(n_pts).collect();

        self.transform.column_transform_into(&columns, &mut buf);
        let new_arr =
            ArrayD::from_shape_vec(out_shape.set_f(!is_c), out).expect("should be correct shape");
        new_arr.into_pyarray(python)
    }

    // todo: arrow
    // arrayref.to_data().buffers() -> for each -> buffer.typed_data
}
