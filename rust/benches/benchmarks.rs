use criterion::{Criterion, criterion_group, criterion_main};
use faer::rand::{Rng, SeedableRng, rngs::SmallRng};
use ome_zarr_transformations::transforms::{
    Affine, Bijection, ByDimension, Identity, MapAxis, Rotation, Scale, Sequence, Translate,
};
use ome_zarr_transformations::{Matrix, Transformation};
use std::{hint::black_box, sync::Arc};

/// An implementation of the identity transform which uses the trait default implementations
/// for as many methods as possible, to determine the overhead of those implementations.
#[derive(Default, Debug, Copy, Clone)]
struct DefaultIdentity(usize);

impl Transformation for DefaultIdentity {
    fn transform_into(&self, pt: &[f64], buf: &mut [f64]) {
        buf.copy_from_slice(pt);
    }

    fn input_ndim(&self) -> usize {
        self.0
    }

    fn output_ndim(&self) -> usize {
        self.0
    }

    fn invert(&self) -> Option<Arc<dyn Transformation>> {
        Some(Arc::new(*self))
    }

    fn is_identity(&self) -> bool {
        true
    }
}

fn coords(n_rows: usize, n_cols: usize) -> Vec<Vec<f64>> {
    let mut rng = SmallRng::seed_from_u64(1991);
    let mut pts = Vec::with_capacity(n_rows);
    for _ in 0..n_rows {
        let mut pt = Vec::with_capacity(n_cols);
        for _ in 0..n_cols {
            pt.push(rng.random::<f64>() * 100.0);
        }
        pts.push(pt);
    }
    pts
}

fn identity_matrix(ndim: usize) -> Matrix {
    let mut builder = Matrix::builder(true);
    for r in 0..ndim {
        let mut row = vec![0.0; ndim];
        row[r] = 1.0;
        builder.add_vec(&row).unwrap();
    }
    builder.build()
}

fn coord_array(transpose: bool) -> Vec<Vec<f64>> {
    if transpose {
        coords(3, 1000)
    } else {
        coords(1000, 3)
    }
}

fn transpose(coords: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let mut out = vec![Vec::with_capacity(coords.len()); coords[0].len()];
    for coord in coords.iter() {
        for (&val, tgt) in coord.iter().zip(out.iter_mut()) {
            tgt.push(val);
        }
    }
    out
}

struct Bencher<'c> {
    name: String,
    criterion: &'c mut Criterion,
}

impl<'c> Bencher<'c> {
    fn new<S: Into<String>>(name: S, criterion: &'c mut Criterion) -> Self {
        Self {
            name: name.into(),
            criterion,
        }
    }

    fn coords<T: Transformation>(&mut self, t: &T) {
        let coords = coord_array(false);
        let mut out = vec![f64::NAN; t.output_ndim()];
        self.criterion
            .bench_function(&format!("{}[coord]", self.name), |b| {
                b.iter(|| {
                    for pt in coords.iter() {
                        black_box(t.transform_into(pt, &mut out));
                    }
                })
            });

        let mut bulk = vec![vec![f64::NAN; t.output_ndim()]; coords.len()];
        let coord_refs: Vec<&[_]> = coords.iter().map(|c| c.as_ref()).collect();
        let mut bulk_refs: Vec<&mut [_]> = bulk.iter_mut().map(|c| c.as_mut()).collect();
        self.criterion
            .bench_function(&format!("{}[bulk]", self.name), |b| {
                b.iter(|| {
                    black_box(t.bulk_transform_into(&coord_refs, &mut bulk_refs));
                })
            });

        let n_coords = coords.len();
        let cols = transpose(&coords);
        let col_refs: Vec<&[_]> = cols.iter().map(|c| c.as_ref()).collect();
        let mut out = vec![vec![f64::NAN; n_coords]; t.output_ndim()];
        let mut out_refs: Vec<&mut [_]> = out.iter_mut().map(|v| v.as_mut()).collect();
        self.criterion
            .bench_function(&format!("{}[column]", self.name), |b| {
                b.iter(|| {
                    black_box(t.column_transform_into(&col_refs, &mut out_refs));
                })
            });
    }
}

fn default_identity(c: &mut Criterion) {
    let mut bencher = Bencher::new(stringify!(DefaultIdentity), c);
    let t = DefaultIdentity(3);
    bencher.coords(&t);
}

fn identity(c: &mut Criterion) {
    let mut bencher = Bencher::new(stringify!(Identity), c);
    let t = Identity::new(3);
    bencher.coords(&t);
}

fn scale(c: &mut Criterion) {
    let mut bencher = Bencher::new(stringify!(Scale), c);
    let t = Scale::try_new(&[2.0, 3.0, 4.0]).unwrap();
    bencher.coords(&t);
}

fn translate(c: &mut Criterion) {
    let mut bencher = Bencher::new(stringify!(Translate), c);
    let t = Translate::try_new(&[2.0, 3.0, 4.0]).unwrap();
    bencher.coords(&t);
}

fn map_axis(c: &mut Criterion) {
    let mut bencher = Bencher::new(stringify!(MapAxis), c);
    let t = MapAxis::try_new(&[2, 1, 0]).unwrap();
    bencher.coords(&t);
}

fn affine(c: &mut Criterion) {
    let mut bencher = Bencher::new(stringify!(Affine), c);
    let matrix = identity_matrix(4);
    let t = Affine::try_from_augmented(&matrix).unwrap();
    bencher.coords(&t);
}

fn rotation(c: &mut Criterion) {
    let mut bencher = Bencher::new(stringify!(Rotation), c);
    let matrix = identity_matrix(3);
    let t = Rotation::try_new(matrix).unwrap();
    bencher.coords(&t);
}

fn sequence(c: &mut Criterion) {
    let mut bencher = Bencher::new(stringify!(Sequence), c);
    let mut builder = Sequence::builder();
    for _ in 0..3 {
        builder.add_transform(Identity::new(3)).unwrap();
    }
    let t = builder.build().unwrap();
    bencher.coords(&t);
}

fn bijection(c: &mut Criterion) {
    let mut bencher = Bencher::new(stringify!(Bijection), c);
    let t = Bijection::try_new(Identity::new(3), Identity::new(3)).unwrap();
    bencher.coords(&t);
}

fn by_dimension(c: &mut Criterion) {
    let mut bencher = Bencher::new(stringify!(ByDimension), c);
    let mut builder = ByDimension::builder(3, 3);
    for idx in 0..3 {
        builder
            .add_transform(Identity::new(1), &[idx], &[2 - idx])
            .unwrap();
    }
    let t = builder.build().unwrap();
    bencher.coords(&t);
}

criterion_group!(
    atoms,
    default_identity,
    identity,
    scale,
    translate,
    map_axis,
    affine,
    rotation,
    sequence,
    bijection,
    by_dimension
);
criterion_main!(atoms);
