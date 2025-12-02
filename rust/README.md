# ome_zarr_transformations

A low-level rust library for the coordinate transformations described in [OME-Zarr RFC 5](https://ngff.openmicroscopy.org/rfc/5/index.html).

## Usage notes

This library...

- tries to minimise allocations by expecting users to pass in their own output buffers
  - see the [AllocatingTransformer] trait for a convenient wrapper
- tries to maximise performance by not checking the length of every coordinate and output buffer in every iteration
  - users are expected to check that the shape of their input coordinates and output buffers matches the transform's dimensionality
- supports coordinates given as rows (`[[z1, y1, x1], [z2, y2, x2]]`) or columns (`[[z1, z2], [y1, y2], [x1, x2]]`)
  - optimal performance depends on the transformation, but generally columns is faster
- provides default implementations for bulk row-wise and column-wise transformation, given an implementation for transforming a single coordinate
  - a custom implementations can be a lot faster, depending on the transformation

## Known limitations

Serialising and deserialising transform configuration from OME-Zarr metadata is out of scope for this library; see the [ome_zarr_metadata](https://ngff.openmicroscopy.org/rfc/5/index.html) crate.

### Not yet implemented

- [ ] `affine` inversion
- [ ] `affine` homogeneity check
- [ ] `coordinates`
- [ ] `displacement`
