use image::{GenericImageView, Pixel};

use crate::indexer::value::BoundedIndex;

pub struct ImageWrapper<T, Im, Px>
where
    T: Copy,
    Px: Pixel<Subpixel = T>,
    Im: GenericImageView<Pixel = Px>,
{
    image: Im,
    extents: [usize; 3],
}

impl<T, Im, Px> ImageWrapper<T, Im, Px>
where
    T: Copy,
    Px: Pixel<Subpixel = T>,
    Im: GenericImageView<Pixel = Px>,
{
    /// Create a CYX array from the image.
    pub fn new(image: Im) -> Self {
        let (w, h) = image.dimensions();
        let c = Px::CHANNEL_COUNT;
        let extents = [c as usize, h as usize, w as usize];
        Self { image, extents }
    }
}

impl<T, Im, Px> BoundedIndex<T> for ImageWrapper<T, Im, Px>
where
    T: Copy,
    Px: Pixel<Subpixel = T>,
    Im: GenericImageView<Pixel = Px>,
{
    fn get(&self, coord: &[usize]) -> Option<T> {
        if coord.iter().zip(self.extents.iter()).any(|(c, e)| c > e) {
            None
        } else {
            Some(self.get_unchecked(coord))
        }
    }

    fn get_unchecked(&self, coord: &[usize]) -> T {
        let px = self.image.get_pixel(coord[2] as u32, coord[1] as u32);
        px.channels()[coord[0]]
    }

    fn extents(&self) -> &[usize] {
        &self.extents
    }
}
