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
        prev = new_vraster.steps[-1].dataset

        new = prev.copy()
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
                        source_filename=prev,
                    )
                ]
                new_band.offset = other

                new.raster_bands[i] = new_band

        new_vraster.steps.append(VRasterStep(new, "add"))
        return new_vraster


    @property
    def crs(self) -> CRS:
        return self.last.crs
    
    @property
    def transform(self) -> Affine:
        return self.last.transform

    def bounds(self) -> BoundingBox:
        return BoundingBox(*rio.transform.array_bounds(*self.shape, self.transform))

    def res(self):
        return self.last.res()

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

        
        


