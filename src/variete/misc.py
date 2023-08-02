from __future__ import annotations

from typing import Any, Literal, overload

import lxml.etree as ET
from rasterio import CRS, Affine
from rasterio.warp import Resampling


def get_resampling_gdal_to_numpy() -> dict[str, Resampling]:
    resamplings = {"NearestNeighbour": Resampling.nearest, "CubicSpline": Resampling.cubic_spline}

    for value in Resampling.__dict__:
        if value.startswith("_") or value.endswith("_"):
            continue
        resampling = getattr(Resampling, value)
        if resampling in resamplings.values():
            continue

        resamplings[value.capitalize()] = resampling

    return resamplings


def number_to_gdal(number: float | int | str) -> str:
    if isinstance(number, str):
        return number

    if isinstance(number, int):
        return str(number)
    return str(int(number)) if number.is_integer() else str(number)


def resampling_gdal_to_rio(string: str) -> Resampling:
    return get_resampling_gdal_to_numpy()[string]


def resampling_rio_to_gdal(resampling: Resampling) -> str:
    inverted = {v: k for k, v in get_resampling_gdal_to_numpy().items()}
    return inverted[resampling]


def get_dtype_gdal_to_numpy() -> dict[str, str]:
    dtypes = {"Byte": "uint8"}
    for dtype in ["float32", "float64", "int16", "int32", "int64"]:
        dtypes[dtype.capitalize()] = dtype

    for gdal_dtype in ["UInt16", "UInt32"]:
        dtypes[gdal_dtype] = gdal_dtype.lower()

    return dtypes


def dtype_gdal_to_numpy(dtype: str) -> str:
    return get_dtype_gdal_to_numpy()[dtype]


def dtype_numpy_to_gdal(dtype: str) -> str:
    inverted = {v: k for k, v in get_dtype_gdal_to_numpy().items()}
    return inverted[dtype]


def new_element(tag: str, text: str | None = None, attributes: dict[str, str] | None = None) -> ET.Element:
    if attributes is None:
        attributes = {}

    elem = ET.Element(tag, {str(k): number_to_gdal(v) for k, v in attributes.items()})
    if text is not None:
        elem.text = str(text)
    return elem


@overload
def find_element(
    parent: ET.Element,
    key: str | list[str],
    text: Literal["both"],
    default: str | None = None,
) -> tuple[ET.Element, str]:
    ...


@overload
def find_element(
    parent: ET.Element, key: str | list[str], text: Literal[False] = False, default: str | None = None
) -> ET.Element:
    ...


@overload
def find_element(parent: ET.Element, key: str | list[str], text: Literal[True], default: str | None = None) -> str:
    ...


def find_element(
    parent: ET.Element, key: str | list[str], text: bool | Literal["both"] = False, default: str | None = None
) -> ET.Element | str | tuple[ET.Element, str]:
    if isinstance(key, str):
        keys = [key]
    else:
        keys = key

    latest = parent
    for latest_key in keys:
        if (element := latest.find(latest_key)) is not None:
            latest = element
        else:
            raise KeyError(f"Key {'chain' if len(keys) > 0 else ''} {'/'.join(keys)} not found")

    if text:
        content = latest.text
        if content is None:
            if default is None:
                raise AssertionError(f"Content was expected but {key} is empty")
            content = default

        if text == "both":
            return latest, content
        return content

    return latest


def transform_to_gdal(transform: Affine) -> str:
    return ", ".join(map(number_to_gdal, transform.to_gdal()))


def parse_gdal_transform(string: str) -> Affine:
    return Affine.from_gdal(*map(float, string.split(",")))


def crs_to_string(crs: CRS) -> str:
    if (epsg_code := crs.to_epsg()) is not None:
        return f"EPSG:{epsg_code}"

    return crs.to_wkt()


def nested_getattr(obj: object, names: list[str], default: Any | None = None) -> Any:
    for name in names:
        obj = getattr(obj, name, default)

    return obj
