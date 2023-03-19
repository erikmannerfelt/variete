from pyproj import CRS
from pathlib import Path
import rasterio as rio
from affine import Affine
import xml.etree.ElementTree as ET
from typing import Literal, Callable
import tempfile
from osgeo import gdal
from rasterio.coords import BoundingBox
from rasterio.warp import Resampling
import warnings
from variete.vrt.raster_bands import AnyRasterBand, WarpedVRTRasterBand, raster_band_from_etree
from variete import misc


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

        self.shape = shape
        self.crs = crs
        self.crs_mapping = crs_mapping
        self.transform = transform
        self.raster_bands = raster_bands
        self.subclass = None
        # self.block_size = None
        self.warp_options = None

    def __repr__(self):
        return "\n".join(
            [f"VRTDataset: shape={self.shape}, crs=EPSG:{self.crs.to_epsg()}, bounds: {self.bounds()}"]
            + ["\t" + "\n\t".join(band.__repr__().splitlines()) for band in self.raster_bands]
        )

    def bounds(self) -> rio.coords.BoundingBox:
        return rio.coords.BoundingBox(*rio.transform.array_bounds(*self.shape, self.transform))

    def res(self) -> tuple[float, float]:
        """
        Return the X/Y resolution of the dataset.
        """
        return self.transform.a, -self.transform.e

    def to_etree(self):
        vrt = ET.Element("VRTDataset", {"rasterXSize": str(self.shape[1]), "rasterYSize": str(self.shape[0])})

        crs = ET.SubElement(vrt, "SRS", {"dataAxisToSRSAxisMapping": self.crs_mapping})
        crs.text = f"EPSG:{self.crs.to_epsg()}"

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

    @classmethod
    def from_file(cls, filepaths: Path | str | list[Path | str], **kwargs):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_vrt = Path(temp_dir).joinpath("temp.vrt")

            build_vrt(output_filepath=temp_vrt, filepaths=filepaths, **kwargs)
            return cls.load_vrt(temp_vrt)

    def to_memfile(self) -> rio.MemoryFile:
        return rio.MemoryFile(self.to_xml().encode(), ext=".vrt")

    @property
    def open_rio(self) -> Callable[None, rio.DatasetReader]:
        return self.to_memfile().open


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

        self.shape = shape
        self.crs = crs
        self.transform = transform
        self.raster_bands = raster_bands
        self.resample_algorithm = resample_algorithm
        self.block_size = block_size
        self.dst_dtype = dst_dtype
        self.options = options
        self.source_dataset = source_dataset
        self.band_mapping = band_mapping
        self.src_transform = src_transform
        self.src_inv_transform = src_inv_transform
        self.dst_transform = dst_transform
        self.dst_inv_transform = dst_inv_transform

        if relative_filename is None:
            if isinstance(source_dataset, Path):
                self.relative_filename = not source_dataset.is_absolute()
            else:
                self.relative_filename = True
        else:
            self.relative_filename = relative_filename

        self.max_error = max_error
        self.approximate = approximate
        self.warp_memory_limit = warp_memory_limit
        self.crs_mapping = crs_mapping

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
        crs.text = f"EPSG:{self.crs.to_epsg()}"

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
            misc.new_element("SourceDataset", str(self.source_dataset), {"relativeToVRT": str(int(self.relative_filename))})
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




def dataset_from_etree(elem: ET.Element) -> VRTDataset | WarpedVRTDataset:

    if elem.tag != "VRTDataset":
        raise ValueError(f"Invalid root tag for VRT: {elem.tag}")

    subclass = elem.get("subClass")

    if subclass == "VRTWarpedDataset":
        return WarpedVRTDataset.from_etree(elem)

    if subclass is not None:
        warnings.warn(f"Unexpected subClass tag: {subclass}. Ignoring it")

    return VRTDataset.from_etree(elem)


def load_vrt(filepath: str | Path) -> VRTDataset | WarpedVRTDataset:
    with open(filepath) as infile:
        root = ET.fromstring(infile.read())

    return dataset_from_etree(root)


def main():

    filepath = Path("Marma_DEM_2021.tif")
    vrt_path = Path("stack.vrt")

    #pixel_function = SumPixelFunction(5)


if __name__ == "__main__":
    main()
