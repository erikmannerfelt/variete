from __future__ import annotations

import copy
import warnings
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import TYPE_CHECKING

from variete import misc

if TYPE_CHECKING:
    from variete.vrt.vrt import VRTDataset


class SourceProperties:
    shape: tuple[int, int]
    dtype: str
    block_size: tuple[int, int]

    def __init__(self, shape: tuple[int, int], dtype: str, block_size: tuple[int, int]):
        for attr in ["shape", "dtype", "block_size"]:
            setattr(self, attr, locals()[attr])

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
    def from_etree(cls, elem: ET.Element) -> SourceProperties:
        shape = (int(elem.get("RasterYSize", 0)), int(elem.get("RasterXSize", 0)))
        dtype = elem.get("DataType", "Byte").lower()
        block_size = (int(elem.get("BlockYSize", "1")), int(elem.get("BlockXSize", "1")))

        return cls(shape=shape, dtype=dtype, block_size=block_size)

    def __repr__(self) -> str:
        return f"SourceProperties: shape: {self.shape}, dtype: {self.dtype}, block_size: {self.block_size}"


class Window:
    x_off: float
    y_off: float
    x_size: float
    y_size: float

    def __init__(self, x_off: float, y_off: float, x_size: float, y_size: float):
        for attr in ["x_off", "y_off", "x_size", "y_size"]:
            setattr(self, attr, locals()[attr])

    def to_etree(self, name: str = "SrcRect") -> ET.Element:
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
    def from_etree(cls, elem: ET.Element) -> Window:
        return cls(
            x_off=float(elem.get("xOff", 0)),
            y_off=float(elem.get("yOff", 0)),
            x_size=float(elem.get("xSize", 0)),
            y_size=float(elem.get("ySize", 0)),
        )

    def __repr__(self) -> str:
        return f"Window: x_off: {self.x_off}, y_off: {self.y_off}, x_size: {self.x_size}, y_size: {self.y_size}"


class ComplexSource:
    source_filename: Path | str | VRTDataset
    source_band: int
    source_properties: SourceProperties | None
    relative_filename: bool
    nodata: int | float | None
    src_window: Window | None
    dst_window: Window | None
    source_kind: str

    def __init__(
        self,
        source_filename: Path | str | VRTDataset,
        source_band: int,
        source_properties: SourceProperties | None,
        nodata: int | float | None,
        src_window: Window | None,
        dst_window: Window | None,
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

    def copy(self) -> ComplexSource:
        return copy.deepcopy(self)

    def to_etree(self) -> ET.Element:
        source_xml = ET.Element(self.source_kind)

        if hasattr(self.source_filename, "to_etree"):
            raise NotImplementedError()

        filename_xml = ET.SubElement(
            source_xml, "SourceFilename", attrib={"relativeToVRT": str(int(self.relative_filename))}
        )
        filename_xml.text = str(self.source_filename)

        if self.source_band is not None:
            band_xml = ET.SubElement(source_xml, "SourceBand")
            band_xml.text = str(self.source_band)

        if self.source_properties is not None:
            source_xml.append(self.source_properties.to_etree())

        if self.src_window is not None:
            source_xml.append(self.src_window.to_etree("SrcRect"))
        if self.dst_window is not None:
            source_xml.append(self.dst_window.to_etree("DstRect"))

        if self.nodata is not None:
            source_xml.append(misc.new_element("NODATA", misc.number_to_gdal(self.nodata)))

        return source_xml

    @classmethod
    def from_etree(cls, elem: ET.Element) -> ComplexSource:
        source_kind = elem.tag
        filename_elem = elem.find("SourceFilename")
        if filename_elem is None:
            raise AssertionError("Expected SourceFilename key")

        relative_filename = bool(int(filename_elem.get("relativeToVRT", 0)))
        source_filename_str = filename_elem.text
        if source_filename_str is None:
            raise AssertionError("Empty SourceFilename")

        if not source_filename_str.startswith("/vsi"):
            source_filename: Path | str = Path(source_filename_str)
        else:
            source_filename = source_filename_str

        source_band = 1
        if (sub_elem := elem.find("SourceBand")) is not None:
            if (text := sub_elem.text) is not None:
                if text.isnumeric():
                    source_band = int(text)
        # source_band = int(getattr(elem.find("SourceBand"), "text", 1))

        if (prop_elem := elem.find("SourceProperties")) is not None:
            source_properties = SourceProperties.from_etree(prop_elem)
        else:
            source_properties = None

        src_window = dst_window = None
        if (sub_elem := elem.find("SrcRect")) is not None:
            src_window = Window.from_etree(sub_elem)

        if (sub_elem := elem.find("DstRect")) is not None:
            dst_window = Window.from_etree(sub_elem)

        if (nodata_elem := elem.find("NODATA")) is not None:
            if nodata_elem.text is None:
                nodata = nodata_elem.text
            else:
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

    def __repr__(self) -> str:
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


class SimpleSource(ComplexSource):
    source_filename: Path | str | VRTDataset
    source_band: int
    source_properties: None
    nodata: None
    src_window: None
    dst_window: None
    relative_filename: bool

    def __init__(
        self,
        source_filename: Path | str | VRTDataset,
        source_band: int | None = None,
        src_window: None = None,
        dst_window: None = None,
        relative_filename: None = None,
        source_kind: None = None,
        source_properties: None = None,
        nodata: None = None,
    ):
        if relative_filename is None:
            if isinstance(source_filename, Path):
                self.relative_filename = not source_filename.is_absolute()
            else:
                self.relative_filename = True
        else:
            self.relative_filename = relative_filename

        for attr in ["source_filename", "source_band", "src_window", "dst_window"]:
            setattr(self, attr, locals()[attr])

        self.nodata = self.source_properties = None
        self.source_kind = "SimpleSource"


Source = ComplexSource | SimpleSource


def source_from_etree(elem: ET.Element) -> Source:
    if elem.tag == "ComplexSource":
        return ComplexSource.from_etree(elem)
    elif elem.tag == "SimpleSource":
        return SimpleSource.from_etree(elem)

    warnings.warn(f"Unknown source tag: '{elem.tag}'. Trying to treat as ComplexSource", stacklevel=2)
    return ComplexSource.from_etree(elem)
