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


class Tree:
    key: str
    properties: dict[str, str]
    content: str | list["Tree"] | None

    def __init__(self, key: str, properties: dict[str, str] | None = None, content: str | list["Tree"] | None = None):
        self.key = key
        self.properties = properties or {}
        self.content = content

    def to_xml(self, indent: int = 0):
        attrs = " ".join(f'{key}="{value}"' for key, value in self.properties.items())
        if len(attrs) > 0:
            attrs = " " + attrs

        indent_space = "  " * indent

        if self.content is None or self.content == "":
            return f"{indent_space}<{self.key}{attrs} />"
        elif isinstance(self.content, Tree):
            content = self.content.to_xml(indent=indent + 1)
            return f"{indent_space}<{self.key}{attrs}>\n{indent_space}{content}\n{indent_space}</{self.key}>"
        elif isinstance(self.content, list) and isinstance(self.content[0], Tree):
            content = f"{indent_space}\n".join([c.to_xml(indent=indent + 1) for c in self.content])
            return f"{indent_space}<{self.key}{attrs}>{indent_space}\n{content}\n{indent_space}</{self.key}>"
        else:
            content = str(self.content)
            return f"{indent_space}<{self.key}{attrs}>{content}</{self.key}>"


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


class Raster:
    crs: CRS
    filepath: Path
    transform: Affine
    shape: tuple[int, int]
    dtypes: list[str]
    bands: list[int]
    block_shapes: list[tuple[int, int]]
    steps: list[VRTStep]
    nodata: int | float | None = None

    def __init__(
        self,
        filepath: Path,
        crs: CRS,
        transform: Affine,
        shape: tuple[int, int],
        dtypes: list[str],
        bands: list[int],
        block_shapes: list[tuple[int, int]],
        nodata: int | float | None = None,
    ):
        self.crs = crs
        self.filepath = Path(filepath)
        self.transform = transform
        self.shape = shape
        self.dtypes = dtypes
        self.bands = bands
        self.block_shapes = block_shapes
        self.steps = []
        self.nodata = nodata

    @staticmethod
    def from_filepath(filepath: Path) -> "Raster":
        with rio.open(filepath) as raster:
            return Raster(
                filepath=filepath,
                crs=raster.crs,
                transform=raster.transform,
                shape=raster.shape,
                dtypes=raster.dtypes,
                bands=list(range(1, raster.count + 1)),
                nodata=raster.nodata,
                block_shapes=raster.block_shapes,
            )

    def to_vrt_xml(self):
        vrt = ET.Element("VRTDataset", {"rasterXSize": str(self.shape[1]), "rasterYsize": str(self.shape[0])})

        crs = ET.SubElement(vrt, "SRS", {"dataAxisToSRSAxisMapping": "2,1"})
        crs.text = self.crs.to_wkt()

        transform = ET.SubElement(vrt, "GeoTransform")
        transform.text = transform_to_gdal(self.transform)

        ET.indent(vrt)

        return ET.tostring(vrt).decode()

    def to_vrt(self):

        vrt = Tree("VRTDataset", {"rasterXSize": str(self.shape[1]), "rasterYSize": str(self.shape[0])}, None)
        vrt.content = [
            Tree("SRS", {"dataAxisToSRSAxisMapping": "2,1"}, self.crs.to_wkt()),
            Tree("GeoTransform", None, transform_to_gdal(self.transform)),
        ]

        for band in self.bands:
            vrt.content += [
                Tree(
                    "VRTRasterBand",
                    {"dataType": str(self.dtypes[band - 1]), "band": str(band)},
                    [
                        Tree("NoDataValue", None, self.nodata),
                        Tree("ColorInterp", None, "Gray"),
                        Tree(
                            "ComplexSource",
                            None,
                            [
                                Tree(
                                    "SourceFilename",
                                    {"relativeToVRT": "0" if self.filepath.is_absolute() else "1"},
                                    self.filepath,
                                ),
                                Tree("SourceBand", None, band),
                                Tree(
                                    "SourceProperties",
                                    {
                                        "RasterXSize": self.shape[1],
                                        "RasterYSize": self.shape[0],
                                        "DataType": self.dtypes[band - 1],
                                        "BlockXSize": self.block_shapes[band - 1][0],
                                        "BlockYSize": self.block_shapes[band - 1][1],
                                    },
                                ),
                                Tree(
                                    "SrcRect",
                                    {
                                        "xOff": "0",
                                        "yOff": "0",
                                        "xSize": str(self.shape[1]),
                                        "ySize": str(self.shape[0]),
                                    },
                                ),
                                Tree("DstRect", {"xOff": "0", "yOff": "0", "xSize": "433", "ySize": "476"}),
                                Tree("NODATA", None, self.nodata),
                            ],
                        ),
                    ],
                )
            ]

        return vrt.to_xml()


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
    source_properties: SourceProperties
    relative_filename: bool
    nodata: int | float | None
    src_window: Window
    dst_window: Window

    def __init__(
        self,
        source_filename: Path | str,
        source_band: int,
        source_properties: SourceProperties,
        nodata: int | float | None,
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
        self.source_properties = source_properties
        self.nodata = nodata
        self.src_window = src_window
        self.dst_window = dst_window

    def __repr__(self):
        return "\n".join(
            [
                "ComplexSource",
                f"\tSource filename: {self.source_filename}",
                f"\tSource band: {self.source_band}",
                f"\tSource properties: {self.source_properties.__repr__()}",
                f"\tNodata: {self.nodata}",
                f"\tSource window: {self.src_window.__repr__()}",
                f"\tDest. window: {self.dst_window.__repr__()}",
            ]
        )

    def to_etree(self):
        source_xml = ET.Element("ComplexSource")

        filename_xml = ET.SubElement(
            source_xml, "SourceFilename", attrib={"relativeToVRT": str(int(self.relative_filename))}
        )
        filename_xml.text = str(self.source_filename)

        band_xml = ET.SubElement(source_xml, "SourceBand")
        band_xml.text = str(self.source_band)

        source_xml.append(self.source_properties.to_etree())
        source_xml.append(self.src_window.to_etree("SrcRect"))
        source_xml.append(self.dst_window.to_etree("DstRect"))

        if self.nodata is not None:
            nodata_xml = ET.SubElement(source_xml, "NODATA")
            nodata_xml.text = str(int(self.nodata) if self.nodata.is_integer() else self.nodata)

        return source_xml

    @classmethod
    def from_etree(cls, elem: ET.Element):
        filename_elem = elem.find("SourceFilename")

        relative_filename = bool(int(filename_elem.get("relativeToVRT")))
        source_filename = filename_elem.text

        if not source_filename.startswith("/vsi"):
            source_filename = Path(source_filename)

        source_band = int(elem.find("SourceBand").text)
        source_properties = SourceProperties.from_etree(elem.find("SourceProperties"))

        src_window = Window.from_etree(elem.find("SrcRect"))
        dst_window = Window.from_etree(elem.find("DstRect"))

        nodata = float(elem.find("NODATA").text)

        return cls(
            source_filename=source_filename,
            source_band=source_band,
            source_properties=source_properties,
            nodata=nodata,
            src_window=src_window,
            dst_window=dst_window,
            relative_filename=relative_filename,
        )


class VRTRasterBand:
    dtype: str
    band: int
    nodata: int | float | None
    color_interp: str
    sources: list[ComplexSource]

    def __init__(
        self, dtype: str, band: int, nodata: int | float | None, color_interp: str, sources: list[ComplexSource]
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
        for source in elem.findall("ComplexSource"):
            sources.append(ComplexSource.from_etree(source))

        return cls(dtype=dtype, band=band, nodata=nodata, color_interp=color_interp, sources=sources)


class VRTDataset:
    shape: tuple[int, int]
    crs: CRS
    crs_mapping: str
    transform: Affine
    raster_bands: list[VRTRasterBand]

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

    def to_xml(self):

        vrt = ET.Element("VRTDataset", {"rasterXSize": str(self.shape[1]), "rasterYSize": str(self.shape[0])})

        if hasattr(self, "sub_class"):
            vrt.set("subClass", self.sub_class)

        crs = ET.SubElement(vrt, "SRS", {"dataAxisToSRSAxisMapping": self.crs_mapping})
        crs.text = self.crs.to_wkt()

        transform = ET.SubElement(vrt, "GeoTransform")
        transform.text = transform_to_gdal(self.transform)

        for band in self.raster_bands:
            vrt.append(band.to_etree())

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

            raster_bands.append(VRTRasterBand.from_etree(band))

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
    def from_files(
        cls,
        filepaths: Path | str | list[Path | str],
        **kwargs,
    ):

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_vrt = Path(temp_dir).joinpath("temp.vrt")

            build_vrt(
                output_filepath=temp_vrt,
                filepaths=filepaths,
                **kwargs
            )
            return cls.load_vrt(temp_vrt)

        raster_bands = []
        with rio.open(filepath) as raster:
            shape = raster.shape
            crs = raster.crs
            transform = raster.transform

            for i, band_n in enumerate(raster.indexes):
                window = Window(0, 0, shape[1], shape[0])
                band = VRTRasterBand(
                    dtype=raster.dtypes[i],
                    band=band_n,
                    nodata=raster.nodata,
                    color_interp=raster.colorinterp[i].name,
                    sources=[
                        ComplexSource(
                            source_filename=filepath,
                            source_band=band_n,
                            source_properties=SourceProperties(
                                shape=shape,
                                dtype=raster.dtypes[i],
                                block_size=raster.block_shapes[i],
                            ),
                            nodata=raster.nodata,
                            src_window=window,
                            dst_window=window,
                        )
                    ],
                )
                raster_bands.append(band)

        return cls(shape=shape, crs=crs, transform=transform, raster_bands=raster_bands)

    @classmethod
    def from_multiple_datasets(
        cls, filepaths: list[str | Path], resolution: Literal["first"] = "first", separate: bool = False
    ):

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_vrt = Path(temp_dir).joinpath("temp.vrt")

            gdal.BuildVRT(str(temp_vrt), list(map(str, filepaths)), separate=separate)

            return cls.load_vrt(temp_vrt)

        vrts = [cls.from_dataset(fp) for fp in filepaths]

        res = vrts[0].res()

        for i, vrt in enumerate(vrts):
            if i == 0:
                continue
            if vrt.crs != vrts[0].crs:
                raise ValueError(f"Dataset CRSs differ: {filepaths[0]}, {filepaths[i]}")

            x_diff_m = vrts[0].transform.c - vrt.transform.c
            y_diff_m = vrts[0].transform.f - vrt.transform.f

            pixel_shift_x = x_diff_m / res[0]
            pixel_shift_y = y_diff_m / res[1]

            for band in vrt.raster_bands:

                for source in band.sources:
                    source.src_window.x_size = source.src_window.x_size - pixel_shift_x
                    source.src_window.y_size = source.src_window.y_size

                    source.src_window.x_off = pixel_shift_x
                    source.src_window.y_off = pixel_shift_y

                if separate:
                    band.band = vrts[0].raster_bands[-1].band + 1
                    vrts[0].raster_bands.append(band)

                else:
                    for main_band in vrts[0].raster_bands:
                        if band.band == main_band.band:
                            main_band.sources += band.sources
                            break
                    else:
                        band.band += 1
                        vrts[0].raster_bands.append(band)

        return vrts[0]


def transform_to_gdal(transform: Affine) -> str:
    return ", ".join(map(str, transform.to_gdal()))


def main():

    # raster = Raster.from_filepath("Marma_DEM_2021.tif")
    # raster = VRTDataset.from_xml("stack.vrt")
    raster = VRTDataset.from_dataset(Path("Marma_DEM_2008.tif").absolute())

    raster2 = VRTDataset.from_multiple_datasets(["Marma_DEM_2008.tif", "Marma_DEM_2021.tif"], separate=True)

    # raster.save_vrt("hello.vrt")

    print(raster2)

    # vrt = ET.Element("VRTDataset", {"rasterXSize": str(raster.shape[1]), "rasterYsize": str(raster.shape[0]) })

    # print(raster.to_vrt_xml())
    # print(ET.tostring(vrt))
    # print(raster.to_vrt())


if __name__ == "__main__":
    main()
