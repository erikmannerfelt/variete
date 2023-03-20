from pathlib import Path
import copy
import tempfile
import rasterio as rio
from rasterio.coords import BoundingBox
from rasterio.transform import Affine
from rasterio.crs import CRS

from variete.vrt.vrt import VRTDataset, AnyVRTDataset, load_vrt
from variete.vrt.raster_bands import VRTDerivedRasterBand
from variete.vrt.pixel_functions import ScalePixelFunction
from variete.vrt import pixel_functions
from variete.vrt.sources import SimpleSource

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
        step = VRasterStep(
            VRTDataset.from_file(filepath),
            name="load_file"
        )

        return cls(steps=[step])
             
    def add(self, other: int | float) -> "VRaster":
        new_vraster = self.copy()
        new = new_vraster.last.copy()

        #if hasattr(other, "_check_compatibility"):
        if isinstance(other, VRaster):
            if (message := self._check_compatibility(other)) is not None:
                raise AssertionError(message)

            for i, band in enumerate(new.raster_bands):
                if isinstance(getattr(band, "pixel_function", None), pixel_functions.SumPixelFunction):
                #if isinstance(band, VRTDerivedRasterBand):
                    band.sources.append(SimpleSource(
                        source_filename=other.last,
                        source_band=i + 1,
                    ))
                else:
                    new_band = VRTDerivedRasterBand.from_raster_band(
                        band=band,
                        pixel_function=pixel_functions.SumPixelFunction()
                    )
                    new_band.sources = [
                        SimpleSource(
                            source_filename=new_vraster.last,
                            source_band=i + 1,
                        ),
                        SimpleSource(
                            source_filename=other.last,
                            source_band=i + 1,
                        )
                    ]

                    new.raster_bands[i] = new_band
            name = "add_vraster"
               
        else:
            for i, band in enumerate(new.raster_bands):
                if isinstance(band, VRTDerivedRasterBand):
                    if band.offset is not None:
                        band.offset += other
                    else:
                        band.offset = other
                else:
                    new_band = VRTDerivedRasterBand.from_raster_band(
                        band=band,
                        pixel_function=ScalePixelFunction()
                    )
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

    def multiply(self, other: int | float) -> "VRaster":
        new_vraster = self.copy()

        new = new_vraster.last.copy()
        if isinstance(other, VRaster):
            if (message := self._check_compatibility(other)) is not None:
                raise AssertionError(message)

            raise NotImplementedError("Not yet implemented for VRaster")
            name = "multiply_vraster"
        else:
            for i, band in enumerate(new.raster_bands):
                if isinstance(band, VRTDerivedRasterBand):
                    if band.scale is not None:
                        band.scale *= other
                    else:
                        band.scale = other
                else:
                    new_band = VRTDerivedRasterBand.from_raster_band(
                        band=band,
                        pixel_function=ScalePixelFunction()
                    )
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


    def _check_compatibility(self, other: "VRaster") -> str | None:
        if self.crs != other.crs:
            return f"CRS is different: {self.crs} != {other.crs}"

        if self.n_bands != other.n_bands:
            return f"Number of bands must be the same: {self.n_bands} != {other.n_bands}"


    def divide(self, other: int | float) -> "VRaster":
        if isinstance(other, VRaster):
            raise NotImplementedError("Not yet implemented for VRaster")
        else:
            new = self.multiply(1 / other)
            new.steps[-1].name = "divide_constant"
        return new

    def subtract(self, other: int | float) -> "VRaster":
        if isinstance(other, VRaster):
            negative = other.multiply(-1)
            new = self.add(negative)
            new.steps[-1].name = "subtract_vraster"
        else:
            new = self.add(-other)
            new.steps[-1].name = "subtract_constant"
        return new
        
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
                return load_vrt(self.last.to_tempfiles(temp_dir=temp_dir)[1]).sample(x_coord=x_coord, y_coord=y_coord, band=band, masked=masked)
                
        return self.steps[-1].dataset.sample(x_coord=x_coord, y_coord=y_coord, band=band, masked=masked)

    def sample_rowcol(self, row: int, col: int, band: int | list[int] = 1, masked: bool = False):
        x_coord, y_coord = rio.transform.xy(self.transform, row, col)
        return self.sample(x_coord, y_coord, band=band, masked=masked)
        #self.steps[-1].dataset.sample_rowcol(row=row, col=col)

        
        

