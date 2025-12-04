use std::{
    iter,
    sync::{LazyLock},
};

use crate::{Transformation};
use faer::rand::{Rng, SeedableRng, rngs::SmallRng};

pub const SMALL_NUMBER: f64 = 1e-10;
pub static COORDS_3D_1000: LazyLock<Vec<Vec<f64>>> = LazyLock::new(|| make_coords(1000, 3));
pub static COORDS_3D_1000_COLS: LazyLock<Vec<Vec<f64>>> =
    LazyLock::new(|| transpose(COORDS_3D_1000.as_ref()));

pub fn init_logger() {
    #[allow(unused_must_use)]
    env_logger::try_init();
}

fn make_coords(n_pts: usize, ndim: usize) -> Vec<Vec<f64>> {
    let mut rng = SmallRng::seed_from_u64(1991);

    iter::repeat_with(|| {
        iter::repeat_with(|| rng.random::<f64>() * 100.0)
            .take(ndim)
            .collect()
    })
    .take(n_pts)
    .collect()
}

fn transpose(coords: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let npoints = coords.len();
    let ndim = coords[0].len();
    let mut columns = vec![vec![f64::NAN; npoints]; ndim];
    for (i, pt) in coords.iter().enumerate() {
        for (j, &v) in pt.iter().enumerate() {
            columns[j][i] = v;
        }
    }
    columns
}

fn transform<T: Transformation>(t: &T, coord: &[f64]) -> Vec<f64> {
    let mut out = vec![f64::NAN; t.output_ndim()];
    t.transform_into(coord, &mut out);
    out
}

fn bulk_transform<T: Transformation, C: AsRef<[f64]>>(t: &T, coords: &[C]) -> Vec<Vec<f64>> {
    let refs: Vec<_> = coords.iter().map(|c| c.as_ref()).collect();
    let mut out = vec![vec![f64::NAN; t.output_ndim()]; coords.len()];
    let mut out_refs: Vec<_> = out.iter_mut().map(|b| b.as_mut()).collect();
    t.bulk_transform_into(&refs, &mut out_refs);
    out
}

fn column_transform<T: Transformation, C: AsRef<[f64]>>(t: &T, columns: &[C]) -> Vec<Vec<f64>> {
    let refs: Vec<_> = columns.iter().map(|c| c.as_ref()).collect();
    let mut out = vec![vec![f64::NAN; refs[0].len()]; t.output_ndim()];
    let mut out_refs: Vec<_> = out.iter_mut().map(|b| b.as_mut()).collect();
    t.column_transform_into(&refs, &mut out_refs);
    out
}

/// Assert that transforming coordinates in bulk matches transforming them one by one.
pub fn check_transform_bulk<T: Transformation>(t: T) {
    init_logger();
    let coords: &[Vec<f64>] = COORDS_3D_1000.as_ref();

    let results_many = bulk_transform(&t, coords);
    for (orig, many_transformed) in coords.iter().zip(results_many.iter()) {
        let result_single = transform(&t, orig);
        approx::assert_ulps_eq!(
            result_single.as_slice(),
            many_transformed.as_slice(),
            epsilon = SMALL_NUMBER
        );
    }
}

/// Assert that transforming the coordinates by column matches transforming them one-by-one.
pub fn check_transform_col<T: Transformation>(t: T) {
    init_logger();
    let coords: &[Vec<f64>] = COORDS_3D_1000.as_ref();
    let columns: &[Vec<f64>] = COORDS_3D_1000_COLS.as_ref();

    let transformed_columns = column_transform(&t, columns);

    for (coord_idx, pt) in coords.iter().enumerate() {
        let transformed_pt = transform(&t, pt);
        let col_transformed_pt: Vec<_> = (0..columns.len())
            .map(|dim_idx| transformed_columns[dim_idx][coord_idx])
            .collect();
        approx::assert_ulps_eq!(
            transformed_pt.as_slice(),
            col_transformed_pt.as_slice(),
            epsilon = SMALL_NUMBER
        );
    }
}

/// Assert that inverting a transformation recovers the original coordinate (more or less).
pub fn check_inverse_transform_coord<T: Transformation>(t: T) {
    init_logger();
    let Some(inv_t) = t.invert() else {
        return;
    };

    let coords: &[Vec<f64>] = COORDS_3D_1000.as_ref();
    let mut transformed = vec![f64::NAN; t.output_ndim()];
    let mut inverted = vec![f64::NAN; inv_t.output_ndim()];
    for pt in coords.iter() {
        t.transform_into(pt, &mut transformed);
        inv_t.transform_into(&transformed, &mut inverted);
        approx::assert_ulps_eq!(pt.as_slice(), inverted.as_slice(), epsilon = SMALL_NUMBER);
    }
}

/// Assert that inverting a bulk transformation recovers the original coordinates (more or less).
pub fn check_inverse_transform_bulk<T: Transformation>(t: T) {
    init_logger();
    let Some(inv_t) = t.invert() else {
        return;
    };

    let coords: &[Vec<f64>] = COORDS_3D_1000.as_ref();
    let coords_refs: Vec<&[f64]> = coords.iter().map(|c| c.as_ref()).collect();

    let mut transformed = vec![vec![f64::NAN; t.output_ndim()]; coords.len()];
    {
        let mut transformed_mut: Vec<&mut [_]> =
            transformed.iter_mut().map(|c| c.as_mut()).collect();
        t.bulk_transform_into(&coords_refs, &mut transformed_mut);
    }

    let transformed_refs: Vec<&[_]> = transformed.iter().map(|c| c.as_ref()).collect();
    let mut inverted = vec![vec![f64::NAN; inv_t.output_ndim()]; coords.len()];
    {
        let mut inverted_mut: Vec<&mut [_]> = inverted.iter_mut().map(|c| c.as_mut()).collect();
        inv_t.bulk_transform_into(&transformed_refs, &mut inverted_mut);
    }

    for (orig, invert) in coords.iter().zip(inverted.iter()) {
        approx::assert_ulps_eq!(orig.as_slice(), invert.as_slice(), epsilon = SMALL_NUMBER);
    }
}

/// Assert that inverting a columnar transformation recovers the original columns (more or less).
pub fn check_inverse_transform_col<T: Transformation>(t: T) {
    init_logger();

    let Some(inv_t) = t.invert() else {
        return;
    };

    let columns: &[Vec<f64>] = COORDS_3D_1000_COLS.as_ref();
    let n_pts = columns[0].len();
    let col_refs: Vec<_> = columns.iter().map(|col| col.as_ref()).collect();

    let mut transformed_columns = vec![vec![f64::NAN; n_pts]; columns.len()];
    {
        let mut transformed_columns_mut: Vec<&mut [_]> =
            transformed_columns.iter_mut().map(|c| c.as_mut()).collect();
        t.column_transform_into(&col_refs, &mut transformed_columns_mut);
    }
    let transformed_columns_ref: Vec<&[_]> =
        transformed_columns.iter().map(|c| c.as_ref()).collect();

    let mut inverted_columns = vec![vec![f64::NAN; n_pts]; inv_t.output_ndim()];
    {
        let mut inverted_columns_mut: Vec<&mut [_]> =
            inverted_columns.iter_mut().map(|c| c.as_mut()).collect();
        inv_t.column_transform_into(&transformed_columns_ref, &mut inverted_columns_mut);
    }

    for idx in 0..n_pts {
        let orig: Vec<_> = columns.iter().map(|c| c[idx]).collect();
        let inverted: Vec<_> = inverted_columns.iter().map(|c| c[idx]).collect();

        approx::assert_ulps_eq!(orig.as_slice(), inverted.as_slice(), epsilon = SMALL_NUMBER);
    }
}
