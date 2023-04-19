# variete — Making lazily evaluated raster operations simple
The [GDAL](https://gdal.org) library leverages functionality that is used in most (if not all) modern geospatial software.
Most operations in GDAL are eager, i.e. they are run sequentially and produce an output directly.
The GDAL Virtual Format (VRT) allows for lazy evaluation of operations, meaning only what is requested is actually calculated, instead of the whole raster at once.
Formulating advanced VRTs, however, are difficult and therefore limits the usage of the format.
Enter the niche of `variete`, VRTs made simple; with `variete`, VRTs are used as the backend and allow lazy evalation of rasters in a "Pythonic" interface.

### Why lazy evaluation is a good idea
Lazy evaluation means delaying calculations up until the moment they are needed, and not before.
This can lead to ambiguity, but when done right, provides tools for getting a better overview of the process, lower memory management, and generally higher flexibility in the end.
Imagine working on a 20-step process (like warping or adding rasters).
Eager evaluation; the opposite of lazy evaluation, would mean loading the initial raster, performing an operation, running the next operation, etc.
With simplistic eager evaluation, this may require 20 files either on disk or in memory.
If the rasters are large, they may need to be saved occasionally, or upon every operation to not run out of memory (OOM).
Even worse, if the raster is too large to fit in memory, it may have to be chunked and operations need to be run on all chunks independently.
There are many good approaches for chunked eager evaluation, but it often still leads to high code complexity and many large intermediate files.
With proper lazy evaluation, no intermediate files (other than ones that the user may request) will be created, and complexity can be kept at a minimum as chunking and parallelization is handled by the backend instead.


**See other sources that highlight the power of VRTs**

- [The VRT driver specification in GDAL](https://gdal.org/drivers/raster/vrt.html)
- [Guide to Virtual Raster's](https://arcg.is/0HqOeu)


### About the name
*Variété* in French means *variety*, *type* or *genre*.
More importantly, `variete` sounds like a very poor pronounciation of V-R-T, which led to the origin of the name of the package.
It is thus a "variety of GDALs approach to lazy evaluation"!
As for pronounciation of the package name, it is like spoken Latin; every language and individual speaker probably has their own version.


## Features
`variete` provides a simple interface to generate on-the-fly instructions for:

- Warping between coordinate systems.
- Cropping / resampling rasters.
- Generating mosaics.
- Performing arithmetic on or between rasters.
- Loading the result as a numpy array, or saving it as a GDAL-friendly stack of VRTs.

Since VRT is the main driver of `variete`, lazy outputs can be read by **any** GDAL-powered software.

### What `variete` **cannot** do
Some caveats inherent to VRTs, or inherent to lazy file-based evaluation in general, cannot be circumvented.
For example:

  - Arithmetic on virtual rasters only works with constants and other virtual rasters. It cannot be done with in-memory datasets such as numpy arrays. In this case, either the array needs to be saved on disk, or the virtual raster needs to be explicitly loaded into memory.
  - Nodata handling in the VRT framework is rudimentary; when subtracting two virtual rasters with nodata values, nodata is ignored. Therefore, `variete.load()` defaults to assigning all nodata values to `np.nan`, which partly circumvents the problem.

## Examples

### Elevation change rate
Generating elevation change rates between two DEMs requires three lines of code in `variete` (excluding imports and plotting):
```python
import variete
import matplotlib.pyplot as plt

dem_2000 = variete.load("dem_2000.tif")
dem_2020 = variete.load("dem_2020.tif")

dhdt = (dem_2020 - dem_2000) / (2020 - 2000)

plt.plot(dhdt.read(1), cmap="RdBu", vmin=-2, vmax=2)
plt.show()

dhdt.write("dhdt_2000-2020.tif")
```

Resampling between the DEMs (in case their spatial extents differ) is done on the fly using bilinear interpolation to the DEM on the left hand side (`dem_2020`).
The exact parameters can be customized if needed.

Note that the pixel values of the `dhdt` variable are not calculated until `dhdt.read(1)` is called; before, it simply represents the "recipe" on how to generate the pixels.

### Sampling a warped raster
Imagine the scenario where a few points need to be sampled from a raster, but the raster is in a different coordinate system.
This can technically be handled by first transforming the points to the raster CRS and then sampling, but with `variete`, it is much easier!
`variete` can warp between coordinate systems with just one (lazy) call, and then the warped raster can be sampled directly.

```python
import variete
import pandas as pd

# This raster may be in a different coordinate system, like a geographic WGS84.
# the .warp call reprojects the raster to a more suitable coordinate system in this case.
# Note that no processing has yet been performed
raster = variete.load("raster.tif").warp(crs="WGS 84 / UTM Zone 33N")
points = pd.read_csv("points.csv")

# When sample is called, only the blocks of the raster that is needed for reprojection is loaded, then reprojected, then sampled from.
points["raster_value"] = raster.sample(points["easting"], points["northing"])
```

## Installation
`variete` only depends on [rasterio](https://github.com/rasterio/rasterio) and its sub-dependencies.
If `rasterio` is installed and it works [(see their installation page)](https://rasterio.readthedocs.io/en/stable/installation.html), `variete` can be installed with one of the following:

```bash
pip install variete
```

or, for the latest main branch commit:

```bash
pip install git+https://github.com/erikmannerfelt/variete.git
```

## Comparison to other packages
### geoutils
The API is heavily inspired from [geoutils](https://github.com/GlacioHack/geoutils), largely because the projects share core developers.
`geoutils` can be seen as the eager equivalent of `variete` in its current state (Apr. 2023) as all operations are performed in-memory.
The packages complement each other; `geoutils` features higher flexibility with the types of supported in-memory operations, while `variete` excels at memory efficiency on large files.

### xarray / rioxarray / dask
[xarray](https://github.com/pydata/xarray) is excellent at lazy evaluation due to its [dask](https://github.com/dask/dask) backend, and [rioxarray](https://github.com/corteva/rioxarray) allows for simple lazy loading of georeferenced rasters.
This family of packages is far ahead in terms of flexibility and multi-node scheduling.
Its API can however be daunting, and its functionality rather shines complex use-cases.
`variete` fills the gap between the eager simplicity of `geoutils` and the lazy complexity of `xarray`.

### rasterio
Most modern geospatial packages use [rasterio](https://github.com/rasterio/rasterio) for geospatial raster operations.
Its goal is to make GDAL functionality easier to use, and it adds many useful tools and error messages along the way.
Indeed, `rasterio` even has a [VRT module](https://rasterio.readthedocs.io/en/latest/api/rasterio.vrt.html) which can handle the (very limited) construction of simple VRTs.
Code complexity is however often a recurring problem with `rasterio`, as even just one operation (like reprojecting, reading or writing) can easily be 5–10 lines of code.
Most of the functionality is also eager, meaning OOM issues and large intermediate files are almost granted.
`variete` uses the safety net that `rasterio` provides for disk-based operations (reading/writing/sampling), while retaining the simplicity that is inspired by `geoutils`.
Most of the properties of a `variete.VRaster` are identical to a `rio.DatasetReader` class (like `transform`, `crs`, `shape`), so switching between the two packages should be trivial, and many use-cases require using both.

## Contributing
We are happy for contributions to added functionality or bugfixes for `variete` that follow the main principles of the package!
Special emphasis is put on:

- Increased GDAL VRT interoperability
- Ease of use
- Bugfixes, testing and documentation improvements

For new features that fall outside of the "`variete` as a VRT wrapper" scope, further discussion is needed.
New functionality may better fit the scope of the related packages [geoutils](https://github.com/GlacioHack/geoutils) or [xdem](https://github.com/GlacioHack/xdem).

## Roadmap
1. Aim for full VRT compatibility. Currently, only a subset of the VRT functionality is supported.
2. Implement graph optimization. Many subsequent steps may possible to simplify to fewer steps, leading to fewer VRTs being active at a time. This will probably increase performance, and will make saving VRT stacks less of a file-explosion.
3. Work toward maximum test and documentation coverage to make sure the core functionality is watertight.
4. Improve inter-operability with `geoutils` and `xdem`. `variete.VRaster`s should be easy to convert to in-memory `geoutils.Raster`s, and vice versa.
