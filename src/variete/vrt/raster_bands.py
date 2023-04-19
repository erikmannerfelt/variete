from __future__ import annotations

import warnings
import xml.etree.ElementTree as ET

from variete import misc
from variete.vrt.pixel_functions import AnyPixelFunction, pixel_function_from_etree
from variete.vrt.sources import Source, source_from_etree


class VRTRasterBand:
    dtype: str
    band: int
    nodata: int | float | None
    color_interp: str
    sources: list[Source]
    offset: int | float | None
    scale: int | float | None

    def __init__(
        self,
        dtype: str,
        band: int,
        nodata: int | float | None,
        color_interp: str,
        sources: list[Source],
        offset: int | float | None = None,
        scale: int | float | None = None,
    ) -> None:
        for attr in ["dtype", "band", "nodata", "color_interp", "sources", "scale", "offset"]:
            setattr(self, attr, locals()[attr])

    def to_etree(self) -> ET.Element:
        band_xml = ET.Element(
            "VRTRasterBand", {"dataType": misc.dtype_numpy_to_gdal(self.dtype), "band": str(self.band)}
        )

        for key, gdal_key in [("nodata", "NoDataValue"), ("offset", "Offset"), ("scale", "Scale")]:
            if (value := getattr(self, key)) is not None:
                band_xml.append(misc.new_element(gdal_key, misc.number_to_gdal(value)))

        color_interp_xml = ET.SubElement(band_xml, "ColorInterp")
        color_interp_xml.text = self.color_interp.capitalize()

        for source in self.sources:
            band_xml.append(source.to_etree())

        return band_xml

    @classmethod
    def from_etree(cls, elem: ET.Element) -> VRTRasterBand:
        dtype = misc.dtype_gdal_to_numpy(elem.get("dataType", "float32"))
        band = int(elem.get("band", 1))

        scalars: dict[str, float | None] = {}
        for key, gdal_key in [("nodata", "NoDataValue"), ("offset", "Offset"), ("scale", "Scale")]:
            if (sub_elem := elem.find(gdal_key)) is not None:
                if sub_elem.text is None:
                    raise AssertionError(f"{gdal_key} exists but is empty")
                scalars[key] = float(sub_elem.text)
            else:
                scalars[key] = None

        color_interp = getattr(elem.find("ColorInterp"), "text", "undefined")

        sources = []
        for source in elem.findall("*"):
            if "Source" not in source.tag:
                continue
            sources.append(source_from_etree(source))

        return cls(dtype=dtype, band=band, color_interp=color_interp, sources=sources, **scalars)

    def __repr__(self) -> str:
        return "\n".join(
            [
                f"VRTRasterBand: dtype: {self.dtype}, band: {self.band},"
                + f" nodata: {self.nodata}, color_interp: {self.color_interp}"
            ]
            + ["\t" + "\n\t".join(source.__repr__().splitlines()) for source in self.sources]
        )


class WarpedVRTRasterBand(VRTRasterBand):
    dtype: str
    band: int
    color_interp: str
    nodata: float | int | None

    def __init__(self, dtype: str, band: int, color_interp: str, nodata: float | int | None = None) -> None:
        for attr in ["dtype", "band", "nodata", "color_interp"]:
            setattr(self, attr, locals()[attr])

    @classmethod
    def from_etree(cls, elem: ET.Element) -> WarpedVRTRasterBand:
        sub_class = elem.get("subClass")
        assert sub_class == "VRTWarpedRasterBand", f"Wrong subclass. Expected VRTWarpedRasterBand, got {sub_class}"

        dtype = misc.dtype_gdal_to_numpy(elem.get("dataType", "float32"))
        band = int(elem.get("band", "1"))

        color_interp = getattr(elem.find("ColorInterp"), "text", "undefined")

        if (sub_elem := elem.find("NoDataValue")) is not None:
            if sub_elem.text is None:
                raise AssertionError("NoDataValue key exists but is empty")
            nodata = float(sub_elem.text)
        else:
            nodata = None

        return cls(dtype=dtype, band=band, color_interp=color_interp, nodata=nodata)

    def to_etree(self) -> ET.Element:
        band = ET.Element(
            "VRTRasterBand",
            {
                "dataType": misc.dtype_numpy_to_gdal(self.dtype),
                "band": str(self.band),
                "subClass": "VRTWarpedRasterBand",
            },
        )

        color_interp_elem = ET.SubElement(band, "ColorInterp")
        color_interp_elem.text = self.color_interp

        if self.nodata is not None:
            nodata_elem = ET.SubElement(band, "NoDataValue")
            nodata_elem.text = misc.number_to_gdal(self.nodata)

        return band

    def __repr__(self) -> str:
        return (
            f"WarpedVRTRasterBand: dtype: {self.dtype}, band: {self.band},"
            + f" nodata: {self.nodata}, color_interp: {self.color_interp}"
        )


class VRTDerivedRasterBand(VRTRasterBand):
    pixel_function: AnyPixelFunction

    def __init__(
        self,
        dtype: str,
        band: int,
        nodata: int | float | None,
        color_interp: str,
        sources: list[Source],
        pixel_function: AnyPixelFunction,
        offset: int | float | None = None,
        scale: int | float | None = None,
    ) -> None:
        for attr in ["dtype", "band", "nodata", "color_interp", "sources", "scale", "offset", "pixel_function"]:
            setattr(self, attr, locals()[attr])

    @classmethod
    def from_etree(cls, elem: ET.Element) -> VRTDerivedRasterBand:
        sub_class = elem.get("subClass")
        assert sub_class == "VRTDerivedRasterBand", f"Wrong subclass. Expected VRTDerivedRasterBand, got {sub_class}"

        base = VRTRasterBand.from_etree(elem)

        pixel_function = pixel_function_from_etree(elem)

        return cls(
            dtype=base.dtype,
            band=base.band,
            nodata=base.nodata,
            color_interp=base.color_interp,
            sources=base.sources,
            pixel_function=pixel_function,
            offset=base.offset,
            scale=base.scale,
        )

    @classmethod
    def from_raster_band(cls, band: VRTRasterBand, pixel_function: AnyPixelFunction) -> VRTDerivedRasterBand:
        return cls(
            dtype=band.dtype,
            band=band.band,
            nodata=band.nodata,
            color_interp=band.color_interp,
            sources=band.sources,
            pixel_function=pixel_function,
            offset=band.offset,
            scale=band.scale,
        )

    def to_etree(self) -> ET.Element:
        base = VRTRasterBand.to_etree(self)
        base.set("subClass", "VRTDerivedRasterBand")

        for sub_elem in self.pixel_function.to_etree_keys():
            base.append(sub_elem)

        return base

    def __repr__(self) -> str:
        return "\n".join(
            [
                f"VRTDerivedRasterBand: dtype: {self.dtype}, band: {self.band},"
                + f" nodata: {self.nodata}, color_interp: {self.color_interp}, {self.pixel_function.__repr__()}",
            ]
            + ["\t" + "\n\t".join(source.__repr__().splitlines()) for source in self.sources]
        )


AnyRasterBand = VRTRasterBand | VRTDerivedRasterBand


def raster_band_from_etree(elem: ET.Element) -> AnyRasterBand:
    if elem.tag != "VRTRasterBand":
        raise ValueError(f"Invalid raster band tag: {elem.tag}")

    subclass = elem.get("subClass")

    if subclass == "VRTWarpedRasterBand":
        return WarpedVRTRasterBand.from_etree(elem)

    if subclass == "VRTDerivedRasterBand":
        return VRTDerivedRasterBand.from_etree(elem)

    if subclass is not None:
        warnings.warn(
            f"Unknown VRTRasterBand class: '{subclass}'. Trying to treat as a classless VRTRasterBand", stacklevel=2
        )
    return VRTRasterBand.from_etree(elem)
