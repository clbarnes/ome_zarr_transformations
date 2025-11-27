use criterion::{Criterion, criterion_group, criterion_main};
use faer::rand::{Rng, SeedableRng, rngs::SmallRng};
use ome_zarr_transformations::{
    Affine, ByDimensionBuilder, Identity, MapAxis, Matrix, MatrixBuilder, Rotation, Scale,
    Sequence, Transform, Translate,
};
use std::hint::black_box;

fn coords(npoints: usize, ndim: usize) -> Vec<Vec<f64>> {
    let mut rng = SmallRng::seed_from_u64(1991);
    let mut pts = Vec::with_capacity(npoints);
    for _ in 0..npoints {
        let mut pt = Vec::with_capacity(ndim);
        for _ in 0..ndim {
            pt.push(rng.random::<f64>() * 100.0);
        }
        pts.push(pt);
    }
    pts
}

fn identity_matrix(ndim: usize) -> Matrix {
    let mut builder = MatrixBuilder::new(true);
    for r in 0..ndim {
        let mut row = vec![0.0; ndim];
        row[r] = 1.0;
        builder.add_vec(&row).unwrap();
    }
    builder.build()
}

fn coord_array() -> Vec<Vec<f64>> {
    coords(1000, 3)
}

fn bench_transform<T: Transform>(c: &mut Criterion, name: &str, t: &T) {
    let coords = coord_array();
    c.bench_function(name, |b| {
        b.iter(|| {
            for pt in coords.iter() {
                black_box(t.transform(pt));
            }
        })
    });
}

fn identity(c: &mut Criterion) {
    bench_transform(c, "identity", &Identity);
}

fn scale(c: &mut Criterion) {
    bench_transform(c, "scale", &Scale::try_new(vec![2.0, 3.0, 4.0]).unwrap());
}

fn translate(c: &mut Criterion) {
    bench_transform(
        c,
        "translate",
        &Translate::try_new(vec![2.0, 3.0, 4.0]).unwrap(),
    );
}

fn map_axis(c: &mut Criterion) {
    bench_transform(c, "map_axis", &MapAxis::try_new(vec![2, 1, 0]).unwrap());
}

fn affine(c: &mut Criterion) {
    let matrix = identity_matrix(4);
    let affine = Affine::from_augmented(&matrix).unwrap();
    bench_transform(c, "affine", &affine);
}

fn rotation(c: &mut Criterion) {
    let matrix = identity_matrix(3);
    let rotation = Rotation::new(matrix).unwrap();
    bench_transform(c, "rotation", &rotation);
}

fn sequence(c: &mut Criterion) {
    let sequence = Sequence::new(vec![
        Box::new(Identity),
        Box::new(Identity),
        Box::new(Identity),
    ]);
    bench_transform(c, "sequence", &sequence);
}

fn by_dimension(c: &mut Criterion) {
    let mut builder = ByDimensionBuilder::new(3, 3);
    for idx in 0..3 {
        builder
            .add_transform(Identity, vec![idx], vec![2 - idx])
            .unwrap();
    }
    let by_dim = builder.build().unwrap();
    bench_transform(c, "by_dimension", &by_dim);
}

criterion_group!(
    atoms,
    identity,
    scale,
    translate,
    map_axis,
    affine,
    rotation,
    sequence,
    by_dimension
);
criterion_main!(atoms);
