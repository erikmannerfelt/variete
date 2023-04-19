# Comparison to other packages

There are many other packages that do similar things to `variete`, but none that fill its exact niche.
To showcase it, one way is to compare code between other known packages.
A common scenario of processes could be:

1. Open two rasters from disk
2. Reproject the second raster to fit the bounds of the first raster
3. Subtract the two rasters
4. Save the difference to a file on disk.

In `variete`, the associated code would look like this:

```python
import variete

raster1 = variete.load("raster1.tif")
raster2 = variete.load("raster2.tif").warp(raster1)

diff = raster1 - raster2

diff.write("diff.tif")
```


## geoutils
The API is heavily inspired from [geoutils](https://github.com/GlacioHack/geoutils), largely because the projects share core developers.
`geoutils` can be seen as the eager equivalent of `variete` in its current state (Apr. 2023) as all operations are performed in-memory.
The packages complement each other; `geoutils` features higher flexibility with the types of supported in-memory operations, while `variete` excels at memory efficiency on large files.

Our raster subtraction scenario from above is highly similar to `variete`, but rasters are loaded into memory:

```python
import geoutils as gu

raster1 = gu.Raster("raster1.tif")
raster2 = gu.Raster("raster2.tif").reproject(raster1)

diff = raster1 - raster2

diff.save("diff.tif")
```

## xarray / rioxarray / dask
[xarray](https://github.com/pydata/xarray) is excellent at lazy evaluation due to its [dask](https://github.com/dask/dask) backend, and [rioxarray](https://github.com/corteva/rioxarray) allows for simple lazy loading of georeferenced rasters.
This family of packages is far ahead in terms of flexibility and multi-node scheduling.
Its API can however be daunting, and its functionality rather shines complex use-cases.
`variete` fills the gap between the eager simplicity of `geoutils` and the lazy complexity of `xarray`.

## rasterio
Most modern geospatial packages use [rasterio](https://github.com/rasterio/rasterio) for geospatial raster operations.
Its goal is to make GDAL functionality easier to use, and it adds many useful tools and error messages along the way.
Indeed, `rasterio` even has a [VRT module](https://rasterio.readthedocs.io/en/latest/api/rasterio.vrt.html) which can handle the (very limited) construction of simple VRTs.
Code complexity is however often a recurring problem with `rasterio`, as even just simple operations (like reprojecting, reading or writing) can easily be tens of lines of code.
Most of the functionality is also eager, meaning OOM issues and large intermediate files are almost granted.
`variete` uses the safety net that `rasterio` provides for disk-based operations (reading/writing/sampling), while retaining the simplicity that is inspired by `geoutils`.
Most of the properties of a {class}`variete.VRaster` are identical to a `rio.DatasetReader` class (like `transform`, `crs`, `shape`), so switching between the two packages should be trivial, and many use-cases require using both.

The raster subtraction scenario is much more verbose, but essentially does the same thing as `geoutils` (in memory):

```python
import rasterio as rio
import rasterio.warp
import numpy as np
profile = {}

with rio.open("raster1.tif") as raster:
  raster1 = raster.read(1, masked=True).filled(np.nan)

  profile.update(raster.profile | {"transform": raster.transform, "crs": raster.crs})

with rio.open("raster2.tif") as raster:
  raster2 = np.empty_like(raster1)

  rasterio.warp.reproject(
    raster.read(1, masked=True).filled(np.nan),
    destination=raster2,
    src_transform=raster.transform,
    dst_transform=profile["transform"],
    src_crs=raster.crs,
    dst_crs=profile["crs"],
    src_res=raster.res,
    dst_res=profile["res"],
    resampling=rasterio.warp.Resampling.bilinear,
  )

diff = raster1 - raster2

with rio.open("diff.tif", "w", **profile) as raster:
  raster.write(1, diff)
```
