from __future__ import annotations
from pathlib import Path
import copy
import tempfile
import rasterio as rio
from rasterio.coords import BoundingBox
from rasterio.transform import Affine
from rasterio.crs import CRS
from rasterio.windows import Window
from rasterio.warp import Resampling
import numpy as np
from typing import overload, Literal

from variete.vrt.vrt import VRTDataset, AnyVRTDataset, load_vrt, vrt_warp, build_vrt
from variete.vrt.raster_bands import VRTDerivedRasterBand
from variete.vrt.pixel_functions import ScalePixelFunction
from variete.vrt import pixel_functions
from variete.vrt.sources import SimpleSource
from variete import misc


class VRasterStep:
    dataset: AnyVRTDataset
    name: str

    def __init__(self, dataset: AnyVRTDataset, name: str):
        for attr in ["dataset", "name"]:
            setattr(self, attr, locals()[attr])


class VRaster:
    steps: list[VRasterStep]

    def __init__(self, steps: list[VRasterStep] | None = None):
        self.steps = steps or []

    @classmethod
    def load_file(cls, filepath: str | Path):
        step = VRasterStep(VRTDataset.from_file(filepath), name="load_file")

        return cls(steps=[step])

    def save_vrt(self, filepath: str | Path) -> list[Path]:
        if self.last.is_nested():
            return self.last.save_vrt_nested(filepath)
        else:
            self.last.save_vrt(filepath)
            return [filepath]

    def _check_compatibility(self, other: "VRaster") -> str | None:
        if self.crs != other.crs:
            return f"CRS is different: {self.crs} != {other.crs}"

        if self.n_bands != other.n_bands:
            return f"Number of bands must be the same: {self.n_bands} != {other.n_bands}"

    @overload
    def read(
        self,
        band: int | list[int] | None,
        out: np.ndarray | np.ma.masked_array | None,
        window: Window | None,
        masked: Literal[True],
        **kwargs,
    ) -> np.ma.masked_array:
        ...

    @overload
    def read(
        self,
        band: int | list[int] | None,
        out: np.ndarray | np.ma.masked_array | None,
        window: Window | None,
        masked: Literal[False],
        **kwargs,
    ) -> np.ndarray:
        ...

    @overload
    def read(
        self,
        band: int | list[int] | None,
        out: np.ndarray | np.ma.masked_array | None,
        window: Window | None,
        masked: Literal[False],
        **kwargs,
    ) -> np.ndarray:
        ...

    def read(
        self,
        band: int | list[int] | None = None,
        out: np.ndarray | np.ma.masked_array | None = None,
        window: Window | None = None,
        masked: bool = False,
        **kwargs,
    ) -> np.ndarray | np.ma.masked_array:

        with tempfile.TemporaryDirectory() as temp_dir:
            filepath = self.last.to_tempfiles(temp_dir)[1]

            with rio.open(filepath) as raster:
                return raster.read(band, out=out, masked=masked, window=window, **kwargs)

    def add(self, other: int | float) -> "VRaster":
        new_vraster = self.copy()
        new = new_vraster.last.copy()

        if isinstance(other, VRaster):
            if (message := self._check_compatibility(other)) is not None:
                raise AssertionError(message)

            for i, band in enumerate(new.raster_bands):
                if misc.nested_getattr(band, ["pixel_function", "name"]) == "scale":
                    band.sources.append(
                        SimpleSource(
                            source_filename=other.last,
                            source_band=i + 1,
                        )
                    )
                else:
                    new_band = VRTDerivedRasterBand.from_raster_band(
                        band=band, pixel_function=pixel_functions.SumPixelFunction()
                    )
                    new_band.sources = [
                        SimpleSource(
                            source_filename=new_vraster.last,
                            source_band=i + 1,
                        ),
                        SimpleSource(
                            source_filename=other.last,
                            source_band=i + 1,
                        ),
                    ]

                    new.raster_bands[i] = new_band
            name = "add_vraster"

        else:
            for i, band in enumerate(new.raster_bands):
                if misc.nested_getattr(band, ["pixel_function", "name"]) == "scale":
                    if band.offset is not None:
                        band.offset += other
                    else:
                        band.offset = other
                else:
                    new_band = VRTDerivedRasterBand.from_raster_band(band=band, pixel_function=ScalePixelFunction())
                    new_band.sources = [
                        SimpleSource(
                            source_filename=new_vraster.last,
                        )
                    ]
                    new_band.offset = other

                    new.raster_bands[i] = new_band
            name = "add_constant"

        new_vraster.steps.append(VRasterStep(new, name))
        return new_vraster

    def __add__(self, other: int | float | "VRaster") -> "VRaster":
        return self.add(other)

    def __radd__(self, other: int | float | "VRaster") -> "VRaster":
        return self.__add__(other)

    def multiply(self, other: int | float | "VRaster") -> "VRaster":
        new_vraster = self.copy()

        new = new_vraster.last.copy()
        if isinstance(other, VRaster):
            if (message := self._check_compatibility(other)) is not None:
                raise AssertionError(message)

            for i, band in enumerate(new.raster_bands):
                if misc.nested_getattr(band, ["pixel_function", "name"]) == "mul":
                    band.sources.append(
                        SimpleSource(
                            source_filename=other.last,
                            source_band=i + 1,
                        )
                    )
                else:
                    new_band = VRTDerivedRasterBand.from_raster_band(
                        band=band, pixel_function=pixel_functions.MulPixelFunction()
                    )
                    new_band.sources = [
                        SimpleSource(
                            source_filename=new_vraster.last,
                            source_band=i + 1,
                        ),
                        SimpleSource(
                            source_filename=other.last,
                            source_band=i + 1,
                        ),
                    ]

                    new.raster_bands[i] = new_band

            # raise NotImplementedError("Not yet implemented for VRaster")
            name = "multiply_vraster"
        else:
            for i, band in enumerate(new.raster_bands):
                if misc.nested_getattr(band, ["pixel_function", "name"]) == "scale":
                    if band.scale is not None:
                        band.scale *= other
                    else:
                        band.scale = other
                    if band.offset is not None:
                        band.offset *= other
                else:
                    new_band = VRTDerivedRasterBand.from_raster_band(band=band, pixel_function=ScalePixelFunction())
                    new_band.sources = [
                        SimpleSource(
                            source_filename=new_vraster.last,
                        )
                    ]
                    new_band.scale = other

                    new.raster_bands[i] = new_band
            name = "multiply_constant"

        new_vraster.steps.append(VRasterStep(new, name))
        return new_vraster

    def __mul__(self, other: int | float | "VRaster") -> "VRaster":
        return self.multiply(other)

    def __rmul__(self, other: int | float | "VRaster") -> "VRaster":
        return self.__mul__(other)

    def __neg__(self) -> "VRaster":
        return self.multiply(-1)

    def warp(
        self,
        crs: CRS | int | str | None = None,
        res: tuple[float, float] | float | None = None,
        shape: tuple[int, int] | None = None,
        bounds: BoundingBox | list[float] | None = None,
        transform: Affine | None = None,
        resampling: Resampling | str = "bilinear",
        multithread: bool = False,
    ):
        new_vraster = self.copy()

        with tempfile.TemporaryDirectory() as temp_dir:
            _, vrt_filepath = new_vraster.last.to_tempfiles(temp_dir)

            warped_path = vrt_filepath.with_stem("warped")
            vrt_warp(
                output_filepath=warped_path,
                input_filepath=vrt_filepath,
                dst_crs=crs,
                dst_res=res,
                dst_shape=shape,
                dst_bounds=bounds,
                dst_transform=transform,
                resampling=resampling,
                multithread=multithread,
            )

            warped = load_vrt(warped_path)

            new_path = vrt_filepath.with_stem("new")
            build_vrt(new_path, warped_path)

            new = load_vrt(new_path)

        warped.source_dataset = new_vraster.last

        for band in new.raster_bands:
            band.sources = [SimpleSource(source_filename=warped, source_band=band.band)]

        new_vraster.steps.append(VRasterStep(warped, "warp"))
        new_vraster.steps.append(VRasterStep(new, "warp_wrapped"))

        return new_vraster

    def replace_nodata(self, value: int | float):
        new_vraster = self.copy()
        new = new_vraster.last.copy()

        for i, band in enumerate(new.raster_bands):
            new_band = VRTDerivedRasterBand.from_raster_band(
                band=band, pixel_function=pixel_functions.ReplaceNodataPixelFunction(value=value)
            )
            new_band.sources = [SimpleSource(source_filename=new_vraster.last, source_band=i + 1)]
            new.raster_bands[i] = new_band

        new_vraster.steps.append(VRasterStep(new, "replace_nodata"))
        return new_vraster

    def inverse(self) -> "VRaster":
        new_vraster = self.copy()
        new = new_vraster.last.copy()
        for i, band in enumerate(new.raster_bands):
            new_band = VRTDerivedRasterBand.from_raster_band(
                band=band, pixel_function=pixel_functions.InvPixelFunction()
            )
            new_band.sources = [SimpleSource(source_filename=new_vraster.last, source_band=i + 1)]
            new.raster_bands[i] = new_band

        new_vraster.steps.append(VRasterStep(new, "inverse"))
        return new_vraster

    def divide(self, other: int | float | "VRaster") -> "VRaster":
        if isinstance(other, VRaster):
            new_vraster = self.copy()
            new = new_vraster.last.copy()
            if (message := self._check_compatibility(other)) is not None:
                raise AssertionError(message)

            for i, band in enumerate(new.raster_bands):
                new_band = VRTDerivedRasterBand.from_raster_band(
                    band=band, pixel_function=pixel_functions.DivPixelFunction()
                )
                new_band.sources = [
                    SimpleSource(
                        source_filename=new_vraster.last,
                        source_band=i + 1,
                    ),
                    SimpleSource(
                        source_filename=other.last.copy(),
                        source_band=i + 1,
                    ),
                ]

                new.raster_bands[i] = new_band

            new_vraster.steps.append(VRasterStep(new, "divide_vraster"))
            return new_vraster
        else:
            new = self.multiply(1 / other)
            new.steps[-1].name = "divide_constant"
        return new

    def __div__(self, other: int | float | "VRaster") -> "VRaster":
        return self.divide(other)

    def __rdiv__(self, other: int | float | "VRaster") -> "VRaster":
        return self.inverse().__rmul__(other)

    def __rtruediv__(self, other: int | float | "VRaster") -> "VRaster":
        return self.__rdiv__(other)

    def __truediv__(self, other: int | float | "VRaster") -> "VRaster":
        return self.__div__(other)

    def subtract(self, other: int | float | "VRaster") -> "VRaster":
        if isinstance(other, VRaster):
            negative = other.multiply(-1)
            new = self.add(negative)
            new.steps[-1].name = "subtract_vraster"
        else:
            new = self.add(-other)
            new.steps[-1].name = "subtract_constant"
        return new

    def __sub__(self, other: int | float | "VRaster") -> "VRaster":
        return self.subtract(other)

    def __rsub__(self, other: int | float | "VRaster") -> "VRaster":
        return self.__neg__().__add__(other)

    @property
    def n_bands(self) -> int:
        return self.last.n_bands

    @property
    def crs(self) -> CRS:
        return self.last.crs

    @property
    def transform(self) -> Affine:
        return self.last.transform

    @property
    def bounds(self) -> BoundingBox:
        return self.last.bounds

    @property
    def res(self):
        return self.last.res

    def copy(self):
        return copy.deepcopy(self)

    @property
    def shape(self):
        return self.last.shape

    @property
    def last(self):
        return self.steps[-1].dataset

    def sample(self, x_coord: float, y_coord: float, band: int | list[int] = 1, masked: bool = False):

        if self.last.is_nested():
            with tempfile.TemporaryDirectory(prefix="variete") as temp_dir:
                return load_vrt(self.last.to_tempfiles(temp_dir=temp_dir)[1]).sample(
                    x_coord=x_coord, y_coord=y_coord, band=band, masked=masked
                )

        return self.steps[-1].dataset.sample(x_coord=x_coord, y_coord=y_coord, band=band, masked=masked)

    def sample_rowcol(self, row: int, col: int, band: int | list[int] = 1, masked: bool = False):
        x_coord, y_coord = rio.transform.xy(self.transform, row, col)
        return self.sample(x_coord, y_coord, band=band, masked=masked)


def load(filepath: str | Path, nodata_to_nan: bool = True) -> VRaster:

    vraster = VRaster.load_file(filepath)

    if nodata_to_nan:
        replace = False
        for band in vraster.last.raster_bands:
            if band.nodata is not None:
                replace = True

        if replace:
            vraster = vraster.replace_nodata(np.nan)

    return vraster
