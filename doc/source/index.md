---
title: variete
---

# variete — Making lazily evaluated raster operations simple
The [GDAL](https://gdal.org) library leverages functionality that is used in most (if not all) modern geospatial software.
Most operations in GDAL are eager, i.e. they are run sequentially and produce an output directly.
The GDAL Virtual Format (VRT) allows for lazy evaluation of operations, meaning only what is requested is actually calculated, instead of the whole raster at once.
Formulating advanced VRTs, however, are difficult and therefore limits the usage of the format.
Enter the niche of `variete`, VRTs made simple; with `variete`, VRTs are used as the backend and allow lazy evaluation of rasters in a "Pythonic" interface.

## Why lazy evaluation is a good idea
Lazy evaluation means delaying calculations up until the moment they are needed, and not before.
This can lead to ambiguity, but when done right, provides tools for getting a better overview of the process, lower memory management, and generally higher flexibility in the end.
Imagine working on a 20-step process (like warping or adding rasters).
Eager evaluation; the opposite of lazy evaluation, would mean loading the initial raster, performing an operation, running the next operation, etc.
With simplistic eager evaluation, this may require 20 files either on disk or in memory.
If the rasters are large, they may need to be saved occasionally, or upon every operation to not run out of memory (OOM).
Even worse, if the raster is too large to fit in memory, it may have to be chunked and operations need to be run on all chunks independently.
There are many good approaches for chunked eager evaluation, but it often still leads to high code complexity and many large intermediate files.
With proper lazy evaluation, no intermediate files (other than ones that the user may request) will be created, and complexity can be kept at a minimum as chunking and parallelization is handled by the backend instead.

## About the name
*Variété* in French means *variety*, *type* or *genre*.
More importantly, `variete` sounds like a very poor pronunciation of V-R-T, which led to the origin of the name of the package.
It is thus a "variety of GDALs approach to lazy evaluation"!
As for pronunciation of the package name, it is like spoken Latin; every language and individual speaker probably has their own version.

## Features
`variete` provides a simple interface to generate on-the-fly instructions for:

- Warping between spatial extents, resolutions, and coordinate systems.
- Cropping / resampling rasters.
- Generating mosaics.
- Performing arithmetic on or between rasters.
- Loading the result as a numpy array, or saving it as a GDAL-friendly stack of VRTs.

Since VRT is the main driver of `variete`, lazy outputs can be read by **any** GDAL-powered software.

### What `variete` **cannot** do
Some caveats inherent to VRTs, or inherent to lazy file-based evaluation in general, cannot be circumvented.
For example:

  - Arithmetic on virtual rasters only works with constants and other virtual rasters. It cannot be done with in-memory datasets such as numpy arrays. In this case, either the array needs to be saved on disk, or the virtual raster needs to be explicitly loaded into memory.
  - Nodata handling in the VRT framework is rudimentary; when subtracting two virtual rasters with nodata values, nodata is ignored. Therefore, {func}`variete.load()` defaults to assigning all nodata values to `np.nan`, which partly circumvents the problem.

{class}`variete.VRaster`

```{toctree}
:caption: Contents
:maxdepth: 3

self

installation.rst
package_comparison.rst
api.rst
```


# Indices and tables

- {ref}`genindex`
- {ref}`modindex`
- {ref}`search`
