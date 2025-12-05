use image::codecs::jpeg::JpegEncoder;
use image::{DynamicImage, ImageBuffer, Rgba};
use ome_zarr_transformations::indexer::ImageWrapper;
use ome_zarr_transformations::indexer::value::{Const, NearestNeighbour, RealIndex, Transformed};
use ome_zarr_transformations::transforms::{Affine, ByDimension, Identity};
use ome_zarr_transformations::{Matrix, Transformation, image};
use std::path::PathBuf;
use std::sync::Arc;

fn image_dir() -> PathBuf {
    let mut pb = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .canonicalize()
        .unwrap();
    pb.pop();
    pb.push("data");
    pb
}

/// Image is of a white cat licking its face, covered in food.
/// It is 1500px wide, 2000px high, in RGBA.
fn img_tutu() -> DynamicImage {
    let mut p = image_dir();
    p.push("tutu.jpeg");
    eprintln!("Reading image {p:?}");
    image::ImageReader::open(&p).unwrap().decode().unwrap()
}

fn transformations() -> Arc<dyn Transformation> {
    // yx transforms
    #[rustfmt::skip]
    let mat_data = vec![
        2.0, 0.5, -512.0,
        0.0, 4.0, -512.0,
        0.0, 0.0, 1.0,
    ];
    let mat = Matrix::try_new(mat_data, 3).unwrap();
    let aff = Affine::try_from_augmented(&mat).unwrap();

    // do not apply transforms to channel axis
    let mut bd = ByDimension::builder(3, 3);
    bd.add_transform(aff, &[1, 2], &[1, 2])
        .unwrap()
        .add_transform(Identity::new(1), &[0], &[0])
        .unwrap();
    bd.build_any().unwrap()
}

fn main() {
    let args: Vec<_> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("No output path given");
        return;
    }
    let outpath = PathBuf::from(&args[1]);
    let img = img_tutu();

    let wrapped = ImageWrapper::new(img);
    let extended = Const::new(wrapped, 0);
    let interpolated = NearestNeighbour::new(extended);
    let transform = transformations();
    let indexable = Transformed::try_new(interpolated, transform).unwrap();

    let buf = ImageBuffer::from_fn(1024, 1024, |x, y| {
        let mut arr = [0; 4];
        let xf = x as f64;
        let yf = y as f64;
        for (cf, a) in [0.0, 1.0, 2.0, 3.0].into_iter().zip(arr.iter_mut()) {
            *a = indexable.get(&[cf, yf, xf]);
        }
        Rgba(arr)
    });

    let mut f = std::fs::File::create(&outpath).unwrap();
    let mut encoder = JpegEncoder::new(&mut f);
    encoder.encode_image(&buf).unwrap();
}
