from pyproj import CRS
from pathlib import Path
import rasterio as rio
from affine import Affine
import xml.etree.ElementTree as ET
from typing import Literal
import tempfile
from osgeo import gdal
from rasterio.coords import BoundingBox
from rasterio.warp import Resampling
import warnings
import pyproj


def get_resampling_gdal_to_numpy():
    resamplings = {"NearestNeighbour": Resampling.nearest, "CubicSpline": Resampling.cubic_spline}

    for value in Resampling.__dict__:
        if value.startswith("_") or value.endswith("_"):
            continue

        resamplings[value.capitalize()] = getattr(Resampling, value)

    return resamplings


def resampling_gdal_to_rio(string: str) -> Resampling:
    return get_resampling_gdal_to_numpy()[string]


def resampling_rio_to_gdal(resampling: Resampling) -> str:
    inverted = {v: k for k, v in get_resampling_gdal_to_numpy().items()}
    return inverted[resampling]


def get_dtype_gdal_to_numpy() -> dict[str, str]:

    dtypes = {"Byte": "uint8"}
    for dtype in ["float32", "float64", "int16", "int32"]:
        dtypes[dtype.capitalize()] = dtype

    for gdal_dtype in ["UInt16", "UInt32"]:
        dtypes[gdal_dtype] = gdal_dtype.lower()

    return dtypes


def dtype_gdal_to_numpy(dtype: str) -> str:
    return get_dtype_gdal_to_numpy()[dtype]


def dtype_numpy_to_gdal(dtype: str) -> str:
    inverted = {v: k for k, v in get_dtype_gdal_to_numpy().items()}
    return inverted[dtype]


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


class VRTStep:
    step: str


class SourceProperties:
    shape: tuple[int, int]
    dtype: str
    block_size: tuple[int, int]

    def __init__(self, shape: tuple[int, int], dtype: str, block_size: tuple[int, int]):
        self.shape = shape
        self.dtype = dtype
        self.block_size = block_size

    def __repr__(self):
        return f"SourceProperties: shape: {self.shape}, dtype: {self.dtype}, block_size: {self.block_size}"

    def to_etree(self) -> ET.Element:

        return ET.Element(
            "SourceProperties",
            {
                "RasterXSize": str(self.shape[1]),
                "RasterYSize": str(self.shape[0]),
                "DataType": self.dtype.capitalize(),
                "BlockYSize": str(self.block_size[0]),
                "BlockXSize": str(self.block_size[1]),
            },
        )

    @classmethod
    def from_etree(cls, elem: ET.Element):

        shape = (int(elem.get("RasterYSize")), int(elem.get("RasterXSize")))
        dtype = elem.get("DataType").lower()
        block_size = (int(elem.get("BlockYSize")), int(elem.get("BlockXSize")))

        return cls(shape=shape, dtype=dtype, block_size=block_size)


class Window:
    x_off: float
    y_off: float
    x_size: float
    y_size: float

    def __init__(self, x_off: float, y_off: float, x_size: float, y_size: float):
        self.x_off = x_off
        self.y_off = y_off
        self.x_size = x_size
        self.y_size = y_size

    def __repr__(self):
        return f"Window: x_off: {self.x_off}, y_off: {self.y_off}, x_size: {self.x_size}, y_size: {self.y_size}"

    def to_etree(self, name: str = "SrcRect"):
        return ET.Element(
            name,
            {
                key: str(int(value) if isinstance(value, int) or float(value).is_integer() else value)
                for key, value in [
                    ("xOff", self.x_off),
                    ("yOff", self.y_off),
                    ("xSize", self.x_size),
                    ("ySize", self.y_size),
                ]
            },
        )

    @classmethod
    def from_etree(cls, elem: ET.Element):
        return cls(
            x_off=float(elem.get("xOff")),
            y_off=float(elem.get("yOff")),
            x_size=float(elem.get("xSize")),
            y_size=float(elem.get("ySize")),
        )


class ComplexSource:
    source_filename: Path | str
    source_band: int
    source_properties: SourceProperties | None
    relative_filename: bool
    nodata: int | float | None
    src_window: Window
    dst_window: Window
    source_kind: str

    def __init__(
        self,
        source_filename: Path | str,
        source_band: int,
        source_properties: SourceProperties,
        nodata: int | float | None,
        src_window: Window,
        dst_window: Window,
        relative_filename: bool | None = None,
        source_kind: str = "ComplexSource",
    ):

        if relative_filename is None:
            if isinstance(source_filename, Path):
                self.relative_filename = not source_filename.is_absolute()
            else:
                self.relative_filename = True
        else:
            self.relative_filename = relative_filename
        self.source_filename = source_filename
        self.source_band = source_band
        self.source_properties = source_properties
        self.nodata = nodata
        self.src_window = src_window
        self.dst_window = dst_window
        self.source_kind = source_kind

    def __repr__(self):
        return "\n".join(
            [
                self.source_kind,
                f"\tSource filename: {self.source_filename}",
                f"\tSource band: {self.source_band}",
                f"\tSource properties: {self.source_properties.__repr__()}",
                f"\tNodata: {self.nodata}",
                f"\tSource window: {self.src_window.__repr__()}",
                f"\tDest. window: {self.dst_window.__repr__()}",
            ]
        )

    def to_etree(self):
        source_xml = ET.Element(self.source_kind)

        filename_xml = ET.SubElement(
            source_xml, "SourceFilename", attrib={"relativeToVRT": str(int(self.relative_filename))}
        )
        filename_xml.text = str(self.source_filename)

        band_xml = ET.SubElement(source_xml, "SourceBand")
        band_xml.text = str(self.source_band)

        if self.source_properties is not None:
            source_xml.append(self.source_properties.to_etree())
        source_xml.append(self.src_window.to_etree("SrcRect"))
        source_xml.append(self.dst_window.to_etree("DstRect"))

        if self.nodata is not None:
            nodata_xml = ET.SubElement(source_xml, "NODATA")
            nodata_xml.text = str(int(self.nodata) if self.nodata.is_integer() else self.nodata)

        return source_xml

    @classmethod
    def from_etree(cls, elem: ET.Element):
        source_kind = elem.tag
        filename_elem = elem.find("SourceFilename")

        relative_filename = bool(int(filename_elem.get("relativeToVRT")))
        source_filename = filename_elem.text

        if not source_filename.startswith("/vsi"):
            source_filename = Path(source_filename)

        source_band = int(elem.find("SourceBand").text)

        if (prop_elem := elem.find("SourceProperties")) is not None:
            source_properties = SourceProperties.from_etree(prop_elem)
        else:
            source_properties = None

        src_window = Window.from_etree(elem.find("SrcRect"))
        dst_window = Window.from_etree(elem.find("DstRect"))

        if (nodata_elem := elem.find("NODATA")) is not None:
            nodata = float(nodata_elem.text)
        else:
            nodata = None

        return cls(
            source_filename=source_filename,
            source_band=source_band,
            source_properties=source_properties,
            nodata=nodata,
            src_window=src_window,
            dst_window=dst_window,
            relative_filename=relative_filename,
            source_kind=source_kind,
        )


class SimpleSource(ComplexSource):
    source_filename: Path | str
    source_band: int
    source_properties: SourceProperties | None
    nodata: int | float | None
    src_window: Window
    dst_window: Window
    relative_filename: bool | None
    source_kind: str

    def __init__(
        self,
        source_filename: Path | str,
        source_band: int,
        src_window: Window,
        dst_window: Window,
        relative_filename: bool | None = None,
    ):
        if relative_filename is None:
            if isinstance(source_filename, Path):
                self.relative_filename = not source_filename.is_absolute()
            else:
                self.relative_filename = True
        else:
            self.relative_filename = relative_filename

        self.source_filename = source_filename
        self.source_band = source_band
        self.src_window = src_window
        self.dst_window = dst_window

        self.nodata = None
        self.source_kind = "SimpleSource"
        self.source_properties = None


def source_from_etree(elem: ET.Element) -> ComplexSource | SimpleSource:

    if elem.tag == "ComplexSource":
        return ComplexSource.from_etree(elem)
    elif elem.tag == "SimpleSource":
        return SimpleSource.from_etree(elem)

    warnings.warn(f"Unknown source tag: '{elem.tag}'. Trying to treat as ComplexSource")
    return ComplexSource.from_etree(elem)


class VRTRasterBand:
    dtype: str
    band: int
    nodata: int | float | None
    color_interp: str
    sources: list[ComplexSource | SimpleSource]

    def __init__(
        self,
        dtype: str,
        band: int,
        nodata: int | float | None,
        color_interp: str,
        sources: list[ComplexSource | SimpleSource],
    ):
        self.dtype = dtype
        self.band = band
        self.nodata = nodata
        self.color_interp = color_interp
        self.sources = sources

    def __repr__(self):
        return "\n".join(
            [
                f"VRTRasterBand: dtype: {self.dtype}, band: {self.band}, nodata: {self.nodata}, color_interp: {self.color_interp}"
            ]
            + ["\t" + "\n\t".join(source.__repr__().splitlines()) for source in self.sources]
        )

    def to_etree(self):
        band_xml = ET.Element("VRTRasterBand", {"dataType": self.dtype.capitalize(), "band": str(self.band)})

        nodata_xml = ET.SubElement(band_xml, "NoDataValue")
        nodata_xml.text = str(int(self.nodata) if self.nodata.is_integer() else self.nodata)

        color_interp_xml = ET.SubElement(band_xml, "ColorInterp")
        color_interp_xml.text = self.color_interp.capitalize()

        for source in self.sources:
            band_xml.append(source.to_etree())

        return band_xml

    @classmethod
    def from_etree(cls, elem: ET.Element):
        dtype = elem.get("dataType").lower()
        band = int(elem.get("band"))

        nodata = float(elem.find("NoDataValue").text)

        color_interp = getattr(elem.find("ColorInterp"), "text", "undefined")

        sources = []
        for source in elem.findall("*"):
            if "Source" not in source.tag:
                continue
            sources.append(source_from_etree(source))

        return cls(dtype=dtype, band=band, nodata=nodata, color_interp=color_interp, sources=sources)


class WarpedVRTRasterBand(VRTRasterBand):
    dtype: str
    band: int
    color_interp: str

    def __init__(self, dtype: str, band: int, color_interp: str):
        self.dtype = dtype
        self.band = band
        self.color_interp = color_interp

    def __repr__(self):
        return f"WarpedVRTRasterBand: dtype: {self.dtype}, band: {self.band}, nodata: {self.nodata}, color_interp: {self.color_interp}"

    @classmethod
    def from_etree(cls, elem: ET.Element):

        sub_class = elem.get("subClass")
        assert sub_class == "VRTWarpedRasterBand", f"Wrong subclass. Expected VRTWarpedRasterBand, got {sub_class}"

        dtype = dtype_gdal_to_numpy(elem.get("dataType"))
        band = int(elem.get("band"))

        color_interp = getattr(elem.find("ColorInterp"), "text", "undefined")

        return cls(dtype=dtype, band=band, color_interp=color_interp)

    def to_etree(self):

        band = ET.Element("VRTRasterBand", {"dataType": dtype_numpy_to_gdal(self.dtype), "band": str(self.band), "subClass": "VRTWarpedRasterBand"})

        color_interp_elem = ET.SubElement(band, "ColorInterp") 
        color_interp_elem.text = self.color_interp

        return band
        


def raster_band_from_etree(elem: ET.Element):

    if elem.tag != "VRTRasterBand":
        raise ValueError("Invalid raster band tag: {elem.tag}")

    subclass = elem.get("subClass")

    if subclass is None:
        return VRTRasterBand.from_etree(elem)

    if subclass == "VRTWarpedRasterBand":
        return WarpedVRTRasterBand.from_etree(elem)

    warnings.warn(f"Unknown VRTRasterBand class: '{subclass}'. Trying to treat as a classless VRTRasterBand")
    return VRTRasterBand.from_etree(elem)


class VRTDataset:
    shape: tuple[int, int]
    crs: CRS
    crs_mapping: str
    transform: Affine
    raster_bands: list[VRTRasterBand]
    subclass: str | None
    # block_size: tuple[int, int] | None

    def __init__(
        self,
        shape: tuple[int, int],
        crs: CRS,
        transform: Affine,
        raster_bands: list[VRTRasterBand],
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

        if self.subclass is not None:
            vrt.set("subClass", self.subclass)

        crs = ET.SubElement(vrt, "SRS", {"dataAxisToSRSAxisMapping": self.crs_mapping})
        crs.text = self.crs.to_wkt()

        transform = ET.SubElement(vrt, "GeoTransform")
        transform.text = transform_to_gdal(self.transform)

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
        crs = CRS.from_wkt(srs_elem.text)
        crs_mapping = srs_elem.get("dataAxisToSRSAxisMapping")

        geotransform_elem = root.find("GeoTransform")

        transform = Affine.from_gdal(*[float(v) for v in geotransform_elem.text.split(", ")])

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
    def from_files(cls, filepaths: Path | str | list[Path | str], **kwargs):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_vrt = Path(temp_dir).joinpath("temp.vrt")

            build_vrt(output_filepath=temp_vrt, filepaths=filepaths, **kwargs)
            return cls.load_vrt(temp_vrt)


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

        resample_algorithm = resampling_gdal_to_rio(warp_options.find("ResampleAlg").text)
        dst_dtype = warp_options.find("WorkingDataType").text
        warp_memory_limit = float(warp_options.find("WarpMemoryLimit").text)

        source_dataset_elem = warp_options.find("SourceDataset")
        source_dataset = source_dataset_elem.text

        if not source_dataset.startswith("/vsi"):
            source_dataset = Path(source_dataset)

        relative_filename = bool(int(source_dataset_elem.get("relativeToVRT")))

        options = {}
        for option_elem in warp_options.findall("Option"):
            options[option_elem.tag] = option_elem.text

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
            transforms[key] = parse_gdal_transform(proj_transformer.find(gdal_key).text)

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
        vrt = ET.Element("VRTDataset", {"rasterXSize": str(self.shape[1]), "rasterYSize": str(self.shape[0]), "subClass": "VRTWarpedDataset"})

        crs = ET.SubElement(vrt, "SRS", {"dataAxisToSRSAxisMapping": self.crs_mapping})
        crs.text = self.crs.to_wkt()

        transform = ET.SubElement(vrt, "GeoTransform")
        transform.text = transform_to_gdal(self.transform)

        for band in self.raster_bands:
            vrt.append(band.to_etree())

        return vrt


def transform_to_gdal(transform: Affine) -> str:
    return ", ".join(map(str, transform.to_gdal()))


def parse_gdal_transform(string: str) -> Affine:
    return Affine.from_gdal(*map(float, string.split(",")))


def main():

    # raster = Raster.from_filepath("Marma_DEM_2021.tif")
    # raster = VRTDataset.from_xml("stack.vrt")
    # _raster = VRTDataset.from_dataset(Path("Marma_DEM_2008.tif").absolute())

    # raster2 = VRTDataset.from_multiple_datasets(["Marma_DEM_2008.tif", "Marma_DEM_2021.tif"], separate=True)

    # raster.save_vrt("hello.vrt")

    # print(raster2)

    raster = WarpedVRTDataset.load_vrt("example_data/warped_vrt.vrt")

    #raster.save_vrt("warped.vrt")
    print(raster.to_xml())

    # vrt = ET.Element("VRTDataset", {"rasterXSize": str(raster.shape[1]), "rasterYsize": str(raster.shape[0]) })

    # print(raster.to_vrt_xml())
    # print(ET.tostring(vrt))
    # print(raster.to_vrt())


if __name__ == "__main__":
    main()
