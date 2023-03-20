from pyproj import CRS
from pathlib import Path
import rasterio as rio
from affine import Affine
import xml.etree.ElementTree as ET
from typing import Literal, Callable, Iterable
import tempfile
from osgeo import gdal
from rasterio.coords import BoundingBox
from rasterio.warp import Resampling
import warnings
from variete.vrt.raster_bands import AnyRasterBand, WarpedVRTRasterBand, raster_band_from_etree
from variete import misc
import copy
from tempfile import TemporaryDirectory
import hashlib
import numpy as np


def build_vrt(
    output_filepath: Path | str,
    filepaths: Path | str | list[Path | str],
    calculate_resolution: Literal["highest"] | Literal["lowest"] | Literal["average"] | Literal["user"] = "average",
    res: tuple[float, float] = None,
    separate: bool = False,
    output_bounds: BoundingBox | None = None,
    resample_algorithm: Resampling = Resampling.bilinear,
    target_aligned_pixels: bool = False,
    band_list: list[int] | None = None,
    add_alpha: bool = False,
    output_crs: CRS | int | str | None = None,
    allow_projection_difference: bool = False,
    src_nodata: int | float | None = None,
    vrt_nodata: int | float | None = None,
    strict: bool = True,
):

    if target_aligned_pixels and res is None:
        raise ValueError(f"{target_aligned_pixels=} requires that 'res' is specified")
    if any(isinstance(filepaths, t) for t in [str, Path]):
        filepaths = [filepaths]
    if res is not None:
        x_res = res[0]
        y_res = res[1]
    else:
        x_res = y_res = None

    if output_crs is not None:
        if isinstance(output_crs, int):
            output_crs = CRS.from_epsg(output_crs).to_wkt()
        elif isinstance(output_crs, CRS):
            output_crs = output_crs.to_wkt()
        else:
            output_crs = str(output_crs)

    gdal.BuildVRT(
        str(output_filepath),
        list(map(str, filepaths)),
        resolution=calculate_resolution,
        xRes=x_res,
        yRes=y_res,
        separate=separate,
        outputBounds=list(output_bounds) if output_bounds is not None else None,
        resampleAlg=resample_algorithm,
        targetAlignedPixels=target_aligned_pixels,
        bandList=band_list,
        addAlpha=add_alpha,
        outputSRS=output_crs,
        allowProjectionDifference=allow_projection_difference,
        srcNodata=src_nodata,
        VRTNodata=vrt_nodata,
        strict=strict,
    )


def build_warped_vrt(
    vrt_filepath: Path | str,
    filepath: Path | str,
    dst_crs: CRS | int | str,
    resample_algorithm: Resampling = Resampling.bilinear,
    max_error: float = 0.125,
    src_crs: CRS | int | str | None = None,
) -> None:

    crss = {"dst_wkt": dst_crs, "src_wkt": src_crs}
    for key, crs in crss.items():
        if crs is None:
            if key == "dst_wkt":
                raise TypeError("dst_crs has to be provided")
            continue
        if isinstance(crs, int):
            crss[key] = CRS.from_epsg(crs).to_wkt()
        elif isinstance(crs, CRS):
            crss[key] = crs.to_wkt()
        else:
            crss[key] = crs

    dataset = gdal.Open(str(filepath))
    vrt_dataset = gdal.AutoCreateWarpedVRT(dataset, crss["src_wkt"], crss["dst_wkt"], resample_algorithm, max_error)
    vrt_dataset.GetDriver().CreateCopy(str(vrt_filepath), vrt_dataset)

    del dataset
    del vrt_dataset


class VRTDataset:
    shape: tuple[int, int]
    crs: CRS
    crs_mapping: str
    transform: Affine
    raster_bands: list[AnyRasterBand]
    subclass: str | None
    # block_size: tuple[int, int] | None

    def __init__(
        self,
        shape: tuple[int, int],
        crs: CRS,
        transform: Affine,
        raster_bands: list[AnyRasterBand],
        crs_mapping: str = "2,1",
    ):

        for attr in ["shape", "crs", "crs_mapping", "transform", "raster_bands"]:
            setattr(self, attr, locals()[attr])

        self.subclass = self.warp_options = None

    def __repr__(self):
        return "\n".join(
            [f"VRTDataset: shape={self.shape}, crs=EPSG:{self.crs.to_epsg()}, bounds: {self.bounds}"]
            + ["\t" + "\n\t".join(band.__repr__().splitlines()) for band in self.raster_bands]
        )

    @property
    def n_bands(self) -> int:
        return len(self.raster_bands)

    @property
    def bounds(self) -> rio.coords.BoundingBox:
        return rio.coords.BoundingBox(*rio.transform.array_bounds(*self.shape, self.transform))

    @property
    def res(self) -> tuple[float, float]:
        """
        Return the X/Y resolution of the dataset.
        """
        return self.transform.a, -self.transform.e

    def to_etree(self):
        vrt = ET.Element("VRTDataset", {"rasterXSize": str(self.shape[1]), "rasterYSize": str(self.shape[0])})

        crs = ET.SubElement(vrt, "SRS", {"dataAxisToSRSAxisMapping": self.crs_mapping})
        crs.text = misc.crs_to_string(self.crs)

        transform = ET.SubElement(vrt, "GeoTransform")
        transform.text = misc.transform_to_gdal(self.transform)

        for band in self.raster_bands:
            vrt.append(band.to_etree())

        return vrt

    def to_xml(self):
        vrt = self.to_etree()
        ET.indent(vrt)
        return ET.tostring(vrt).decode()

    @classmethod
    def from_etree(cls, root: ET.Element):
        x_size, y_size = [int(root.get(f"raster{k}Size")) for k in ["X", "Y"]]

        srs_elem = root.find("SRS")
        crs = CRS.from_string(srs_elem.text)
        crs_mapping = srs_elem.get("dataAxisToSRSAxisMapping")

        geotransform_elem = root.find("GeoTransform")

        transform = misc.parse_gdal_transform(geotransform_elem.text)

        raster_bands = []
        for band in root.findall("VRTRasterBand"):

            raster_bands.append(raster_band_from_etree(band))

        return cls(
            shape=(y_size, x_size), crs=crs, transform=transform, raster_bands=raster_bands, crs_mapping=crs_mapping
        )

    def copy(self):
        return copy.deepcopy(self)

    @classmethod
    def from_xml(cls, xml: str):
        vrt = ET.fromstring(xml)
        return cls.from_etree(vrt)

    @classmethod
    def load_vrt(cls, filepath: Path):
        with open(filepath) as infile:
            return cls.from_xml(infile.read())

    def save_vrt(self, filepath: Path) -> None:
        with open(filepath, "w") as outfile:
            outfile.write(self.to_xml())

    def save_vrt_nested(self, filepath: Path | str) -> list[Path]:
        return list(set(self._save_vrt_nested(filepath=Path(filepath).absolute(), nested_level=[])))

    def _save_vrt_nested(self, filepath: Path, nested_level: list[int]) -> list[Path]:
        if len(nested_level) == 0:
            save_filepath = filepath
        else:
            save_filepath = filepath.with_stem(filepath.stem + "-nested-" + "-".join(map(str, nested_level)))

        nested_level += [0]
        filepaths = [save_filepath]
        j = 1
        vrt = self.copy()
        for raster_band in vrt.raster_bands:
            for source in raster_band.sources:
                if hasattr(source.source_filename, "_save_vrt_nested"):
                    #new_filepath = filepath.with_stem(filepath.stem + "-" + str(j).zfill(2))
                    new_nest = nested_level.copy()
                    new_nest[-1] = j
                    new_filepaths = source.source_filename._save_vrt_nested(filepath, new_nest)
                    source.source_filename = new_filepaths[0]
                    source.relative_filename = False
                    filepaths += new_filepaths
                    j += 1

        vrt.save_vrt(save_filepath)
        #print(f"Saved {save_filepath}: {nested_level}")
        
        return filepaths
        

    @classmethod
    def from_file(cls, filepaths: Path | str | list[Path | str], **kwargs):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_vrt = Path(temp_dir).joinpath("temp.vrt")

            build_vrt(output_filepath=temp_vrt, filepaths=filepaths, **kwargs)
            return cls.load_vrt(temp_vrt)

    def sha1(self) -> str:
        return hashlib.sha1(str(self.__dict__).encode()).hexdigest()

    def is_nested(self) -> bool:
        for raster_band in self.raster_bands:
            for source in raster_band.sources:
                if hasattr(source.source_filename, "to_tempfiles"):
                    return True
        return False

    def to_tempfiles(self, temp_dir: TemporaryDirectory | str | Path | None = None) -> tuple[TemporaryDirectory, Path]:

        if temp_dir is None:
            temp_dir = TemporaryDirectory(prefix="variete")

        
        temp_dir_path = Path(getattr(temp_dir, "name", temp_dir))
        filepath = temp_dir_path.joinpath("vrtdataset.vrt")

        self.save_vrt_nested(filepath)
        return temp_dir, filepath

        vrt = self.copy()
        for raster_band in vrt.raster_bands:
            for source in raster_band.sources:
                if hasattr(source.source_filename, "to_tempfiles"):
                    _, new_filepath = source.source_filename.to_tempfiles(temp_dir=temp_dir)
                    source.source_filename = new_filepath
                    source.relative_filename = False

        temp_dir_path = Path(getattr(temp_dir, "name", temp_dir))

        filepath = temp_dir_path.joinpath(vrt.sha1()).with_suffix(".vrt")

        vrt.save_vrt(filepath)

        return temp_dir, filepath

    def to_memfile(self) -> rio.MemoryFile:
        if self.is_nested():
            raise ValueError("Nested VRTs require temporary saving to work (see to_memfile_nested")
        return rio.MemoryFile(self.to_xml().encode(), ext=".vrt")

    def to_memfile_nested(
        self, temp_dir: TemporaryDirectory | str | Path | None
    ) -> tuple[TemporaryDirectory, rio.MemoryFile]:
        if not self.is_nested():
            return (temp_dir, self.to_memfile())

        if temp_dir is None:
            temp_dir = TemporaryDirectory(prefix="variete")

        _, filepath = self.to_tempfiles(temp_dir=temp_dir)

        with open(filepath, "rb") as infile:
            return (temp_dir, rio.MemoryFile(infile.read()))

    @property
    def open_rio(self) -> Callable[None, rio.DatasetReader]:
        if self.is_nested():
            raise ValueError("Nested VRTs require temporary saving to work (see open_rio_nested")
        return self.to_memfile().open

    def open_rio_nested(
        self, temp_dir: TemporaryDirectory | str | Path | None = None
    ) -> tuple[TemporaryDirectory, Callable[None, rio.DatasetReader]]:
        if not self.is_nested():
            return (temp_dir, self.open_rio)

        if temp_dir is None:
            temp_dir = TemporaryDirectory(prefix="variete")

        return (temp_dir, self.to_memfile_nested(temp_dir=temp_dir)[1].open)

    def sample(self, x_coord: float, y_coord: float, band: int | list[int] = 1, masked: bool = False):
        if not isinstance(x_coord, Iterable):
            x_coord = [x_coord]
            y_coord = [y_coord]
        with self.open_rio() as raster:
            values = np.fromiter(
                raster.sample(zip(x_coord, y_coord), indexes=band, masked=masked),
                dtype=self.raster_bands[band - 1].dtype,
                count=len(x_coord),
            ).ravel()
            if values.size > 1:
                return values
            return values[0]


class WarpedVRTDataset(VRTDataset):
    shape: tuple[int, int]
    crs: CRS
    crs_mapping: str
    transform: Affine
    block_size: tuple[int, int]
    raster_bands: list[WarpedVRTRasterBand]
    warp_memory_limit: float
    resample_algorithm: Resampling
    dst_dtype: str
    options: dict[str, str]
    source_dataset: str | Path
    relative_filename: bool | None
    band_mapping: list[tuple[int, int]]
    max_error: float
    approximate: bool
    src_transform: Affine
    src_inv_transform: Affine
    dst_transform: Affine
    dst_inv_transform: Affine

    def __init__(
        self,
        shape: tuple[int, int],
        crs: CRS,
        transform: Affine,
        raster_bands: list[WarpedVRTRasterBand],
        resample_algorithm: Resampling,
        block_size: tuple[int, int],
        dst_dtype: str,
        options: dict[str, str],
        source_dataset: str | Path,
        band_mapping: list[tuple[int, int]],
        src_transform: Affine,
        src_inv_transform: Affine,
        dst_transform: Affine,
        dst_inv_transform: Affine,
        crs_mapping: str = "2,1",
        relative_filename: bool | None = None,
        max_error: float = 0.125,
        approximate: bool = True,
        warp_memory_limit: float = 6.71089e07,
    ):
        if crs_mapping is None:
            crs_mapping = "2,1"

        if relative_filename is None:
            if isinstance(source_dataset, Path):
                self.relative_filename = not source_dataset.is_absolute()
            else:
                self.relative_filename = True
        else:
            self.relative_filename = relative_filename

        attrs = (
            ["shape", "crs", "transform", "raster_bands", "resample_algorithm", "block_size", "dst_dtype"]
            + ["options", "source_dataset", "band_mapping", "src_transform", "src_inv_transform", "dst_transform"]
            + ["dst_inv_transform", "crs_mapping", "warp_memory_limit", "max_error", "approximate"]
        )
        for attr in attrs:
            setattr(self, attr, locals()[attr])

    @classmethod
    def from_etree(cls, root: ET.Element):

        initial = VRTDataset.from_etree(root)

        block_size = tuple([int(root.find(f"Block{dim}Size").text) for dim in ["X", "Y"]])

        warp_options = root.find("GDALWarpOptions")

        resample_algorithm = misc.resampling_gdal_to_rio(warp_options.find("ResampleAlg").text)
        dst_dtype = misc.dtype_gdal_to_numpy(warp_options.find("WorkingDataType").text)
        warp_memory_limit = float(warp_options.find("WarpMemoryLimit").text)

        source_dataset_elem = warp_options.find("SourceDataset")
        source_dataset = source_dataset_elem.text

        if not source_dataset.startswith("/vsi"):
            source_dataset = Path(source_dataset)

        relative_filename = bool(int(source_dataset_elem.get("relativeToVRT")))

        options = {}
        for option_elem in warp_options.findall("Option"):
            options[option_elem.get("name")] = option_elem.text

        transformer = warp_options.find("Transformer").find("ApproxTransformer")

        max_error = float(transformer.find("MaxError").text)

        proj_transformer = transformer.find("BaseTransformer").find("GenImgProjTransformer")

        transforms = {}
        for key, gdal_key in [
            ("src_transform", "SrcGeoTransform"),
            ("src_inv_transform", "SrcInvGeoTransform"),
            ("dst_transform", "DstGeoTransform"),
            ("dst_inv_transform", "DstInvGeoTransform"),
        ]:
            transforms[key] = misc.parse_gdal_transform(proj_transformer.find(gdal_key).text)

        band_mapping = []
        for band_map in warp_options.find("BandList").findall("BandMapping"):
            band_mapping.append((int(band_map.get("src")), int(band_map.get("dst"))))

        return cls(
            shape=initial.shape,
            crs=initial.crs,
            transform=initial.transform,
            raster_bands=initial.raster_bands,
            crs_mapping=initial.crs_mapping,
            block_size=block_size,
            resample_algorithm=resample_algorithm,
            approximate=True,
            warp_memory_limit=warp_memory_limit,
            dst_dtype=dst_dtype,
            relative_filename=relative_filename,
            source_dataset=source_dataset,
            max_error=max_error,
            options=options,
            band_mapping=band_mapping,
            **transforms,
        )

    def to_etree(self):
        vrt = ET.Element(
            "VRTDataset",
            {"rasterXSize": str(self.shape[1]), "rasterYSize": str(self.shape[0]), "subClass": "VRTWarpedDataset"},
        )

        crs = ET.SubElement(vrt, "SRS", {"dataAxisToSRSAxisMapping": self.crs_mapping})
        crs.text = misc.crs_to_string(self.crs)

        transform = ET.SubElement(vrt, "GeoTransform")
        transform.text = misc.transform_to_gdal(self.transform)

        for band in self.raster_bands:
            vrt.append(band.to_etree())

        for i, dim in enumerate(["X", "Y"]):
            size = ET.SubElement(vrt, f"Block{dim}Size")
            size.text = str(self.block_size[i])

        warp = ET.SubElement(vrt, "GDALWarpOptions")

        warp.append(misc.new_element("WarpMemoryLimit", str(self.warp_memory_limit)))
        warp.append(misc.new_element("ResampleAlg", misc.resampling_rio_to_gdal(self.resample_algorithm)))

        warp.append(
            misc.new_element(
                "WorkingDataType",
                misc.dtype_numpy_to_gdal(self.dst_dtype),
            )
        )

        for key in self.options:
            warp.append(misc.new_element("Option", self.options[key], {"name": key}))

        warp.append(
            misc.new_element(
                "SourceDataset", str(self.source_dataset), {"relativeToVRT": str(int(self.relative_filename))}
            )
        )

        transformer = ET.SubElement(ET.SubElement(warp, "Transformer"), "ApproxTransformer")

        transformer.append(misc.new_element("MaxError", str(self.max_error)))

        base_tr = ET.SubElement(ET.SubElement(transformer, "BaseTransformer"), "GenImgProjTransformer")

        for key, gdal_key in [
            ("src_transform", "SrcGeoTransform"),
            ("src_inv_transform", "SrcInvGeoTransform"),
            ("dst_transform", "DstGeoTransform"),
            ("dst_inv_transform", "DstInvGeoTransform"),
        ]:
            base_tr.append(misc.new_element(gdal_key, misc.transform_to_gdal(getattr(self, key)).replace(" ", "")))

        band_list = ET.SubElement(warp, "BandList")

        for src, dst in self.band_mapping:
            band_list.append(misc.new_element("BandMapping", None, {"src": src, "dst": dst}))

        return vrt

    @classmethod
    def from_file(cls, filepath: Path | str, dst_crs: CRS | int | str, **kwargs):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_vrt = Path(temp_dir).joinpath("temp.vrt")

            build_warped_vrt(vrt_filepath=temp_vrt, filepath=filepath, dst_crs=dst_crs, **kwargs)

            vrt = cls.load_vrt(temp_vrt)

        # Nodata values are not transferred with GDALs WarpedVRT builder, so this has to be done manually
        with rio.open(filepath) as raster:
            for band in vrt.raster_bands:
                band.nodata = raster.nodata

        return vrt


AnyVRTDataset = VRTDataset | WarpedVRTDataset


def dataset_from_etree(elem: ET.Element) -> AnyVRTDataset:

    if elem.tag != "VRTDataset":
        raise ValueError(f"Invalid root tag for VRT: {elem.tag}")

    subclass = elem.get("subClass")

    if subclass == "VRTWarpedDataset":
        return WarpedVRTDataset.from_etree(elem)

    if subclass is not None:
        warnings.warn(f"Unexpected subClass tag: {subclass}. Ignoring it")

    return VRTDataset.from_etree(elem)


def load_vrt(filepath: str | Path) -> AnyVRTDataset:
    with open(filepath) as infile:
        root = ET.fromstring(infile.read())

    return dataset_from_etree(root)


def main():

    filepath = Path("Marma_DEM_2021.tif")
    vrt_path = Path("stack.vrt")

    # pixel_function = SumPixelFunction(5)


if __name__ == "__main__":
    main()
