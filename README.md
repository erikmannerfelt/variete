# variete — Making lazily evaluated raster operations simple 
The [GDAL](https://gdal.org) library leverages functionality that is used in most (if not all) modern geospatial software.
Most operations in GDAL are eager, i.e. they are run sequentially and produce an output directly.
The GDAL Virtual Format (VRT) allows for lazy evaluation of operations, i.e. only what is requested is actually calculated, instead of the whole raster at once.
Formulating advanced VRTs, however, are difficult and therefore limits the usage of the format.
Enter the niche of `variete`, VRTs made simple; with `variete`, VRTs are used as the backend and allow lazy evalation of rasters in a "Pythonic" interface.

### Why lazy evaluation is a good idea
**TODO**

## Features
`variete` provides a simple interface to generate on-the-fly instructions for:

- Warping between coordinate systems.
- Cropping / resampling rasters.
- Generating mosaics.
- Performing arithmetic on or between rasters.
- Loading the result as a numpy array, or saving it as a GDAL-friendly stack of VRTs.

Since VRT is the main driver of `variete`, lazy outputs can be read by **any** GDAL-powered software.

## Examples

Generating elevation change rates between two DEMs requires three lines of code in `variete` (excluding imports and plotting):
```python
import variete
import matplotlib.pyplot as plt

dem_2000 = variete.load("dem_2000.tif")
dem_2020 = variete.load("dem_2020.tif")

dhdt = (dem_2020 - dem_2000) / (2020 - 2000)

plt.plot(dhdt.read(1), cmap="RdBu", vmin=-2, vmax=2)
plt.show()
``` 

Resampling between the DEMs (in case their spatial extents differ) is done on the fly using bilinear interpolation to the DEM on the left hand side (`dem_2020`).
The exact parameters can be customized if needed.

Note that the pixel values of the `dhdt` variable are not calculated until `dhdt.read(1)` is called; before, it simply represents the "recipe" on how to generate the pixels.


## Installation

**TODO**

### Requirements

- `gdal`
- `numpy`
- `rasterio`


## Comparison to other packages

**TODO**


## Contributing

**TODO**


## Roadmap

**TODO**

