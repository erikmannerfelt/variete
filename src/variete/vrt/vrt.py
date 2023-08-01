from __future__ import annotations

import copy
import hashlib
import tempfile
import warnings
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Callable, Iterable, Literal, Sequence

import lxml.etree as ET
import numpy as np
import numpy.typing as npt
import rasterio as rio
from affine import Affine
from osgeo import gdal
from rasterio import CRS
from rasterio.coords import BoundingBox
from rasterio.warp import Resampling

from variete import misc
from variete.vrt.raster_bands import AnyRasterBand, raster_band_from_etree


def build_vrt(
    output_filepath: Path | str,
    filepaths: Path | str | list[Path | str],
    calculate_resolution: Literal["highest"] | Literal["lowest"] | Literal["average"] | Literal["user"] = "average",
    res: tuple[float, float] | None = None,
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
) -> None:
    if target_aligned_pixels and res is None:
        raise ValueError(f"{target_aligned_pixels=} requires that 'res' is specified")
    if isinstance(filepaths, (str, Path)):
        filepaths = [filepaths]
    if res is not None:
        x_res: float | None = res[0]
        y_res: float | None = res[1]
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


def vrt_warp(
    output_filepath: Path | str,
    input_filepath: Path | str,
    # src_crs: CRS | int | str | None = None,
    dst_crs: CRS | int | str | None = None,
    dst_res: tuple[float, float] | float | None = None,
    # src_res: tuple[float, float] | None = None,
    dst_shape: tuple[int, int] | None = None,
    # src_bounds: BoundingBox | list[float] | None = None,
    dst_bounds: BoundingBox | list[float] | None = None,
    # src_transform: Affine | None = None,
    dst_transform: Affine | None = None,
    src_nodata: int | float | None = None,
    dst_nodata: int | float | None = None,
    resampling: Resampling | str = "bilinear",
    multithread: bool = False,
) -> None:
    if isinstance(resampling, str):
        resampling = getattr(Resampling, resampling)

    kwargs = {
        "resampleAlg": misc.resampling_rio_to_gdal(resampling),
        "multithread": multithread,
        "format": "VRT",
        "dstNodata": dst_nodata,
        "srcNodata": src_nodata,
    }

    # This is strange. Warped pixels that are outside the range of the original raster get assigned to 0
    # Unclear if this can be overridden somehow! It should be dst_nodata or np.nan
    if kwargs["dstNodata"] is None:
        kwargs["dstNodata"] = 0

    for key, crs in [("dstSRS", dst_crs)]:
        if crs is None:
            if key == "dst_wkt":
                raise TypeError("dst_crs has to be provided")
            continue
        if isinstance(crs, int):
            kwargs[key] = CRS.from_epsg(crs).to_wkt()
        elif isinstance(crs, CRS):
            kwargs[key] = crs.to_wkt()
        else:
            kwargs[key] = crs

    if dst_transform is not None and dst_shape is None:
        raise ValueError("dst_transform requires dst_shape, which was not supplied.")
    if dst_transform is not None and dst_res is not None:
        raise ValueError("dst_transform and dst_res cannot be used at the same time.")
    if dst_transform is not None and dst_bounds is not None:
        raise ValueError("dst_transform and dst_bounds cannot be used at the same time.")

    if dst_shape is not None and dst_res is not None:
        raise ValueError("dst_shape and dst_res cannot be used at the same time.")

    if dst_transform is not None:
        # kwargs["dstTransform"] = dst_transform.to_gdal()
        kwargs["outputBounds"] = list(rio.transform.array_bounds(*dst_shape, dst_transform))

    if dst_shape is not None:
        kwargs["width"] = dst_shape[1]
        kwargs["height"] = dst_shape[0]

    if dst_res is not None:
        if isinstance(dst_res, Sequence):
            kwargs["xRes"] = dst_res[0]  # type: ignore
            kwargs["yRes"] = dst_res[1]  # type: ignore
        else:
            kwargs["xRes"] = dst_res
            kwargs["yRes"] = dst_res

    gdal.Warp(str(output_filepath), str(input_filepath), **kwargs)


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
    ) -> None:
        for attr in ["shape", "crs", "crs_mapping", "transform", "raster_bands"]:
            setattr(self, attr, locals()[attr])

        self.subclass = self.warp_options = None

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

    def to_etree(self) -> ET.Element:
        vrt = ET.Element("VRTDataset", {"rasterXSize": str(self.shape[1]), "rasterYSize": str(self.shape[0])})

        crs = ET.SubElement(vrt, "SRS", {"dataAxisToSRSAxisMapping": self.crs_mapping})
        crs.text = misc.crs_to_string(self.crs)

        transform = ET.SubElement(vrt, "GeoTransform")
        transform.text = misc.transform_to_gdal(self.transform)

        for band in self.raster_bands:
            vrt.append(band.to_etree())

        return vrt

    def to_xml(self) -> str:
        vrt = self.to_etree()
        ET.indent(vrt)
        return ET.tostring(vrt).decode()

    @classmethod
    def from_etree(cls, root: ET.Element) -> VRTDataset:
        x_size, y_size = (int(root.get(f"raster{k}Size", 0)) for k in ["X", "Y"])

        srs_elem, srs_text = misc.find_element(root, "SRS", "both")
        crs = CRS.from_string(srs_text)
        crs_mapping = srs_elem.get("dataAxisToSRSAxisMapping", "2,1")

        transform = misc.parse_gdal_transform(misc.find_element(root, "GeoTransform", True))

        raster_bands = []
        for band in root.findall("VRTRasterBand"):
            raster_bands.append(raster_band_from_etree(band))

        return cls(
            shape=(y_size, x_size), crs=crs, transform=transform, raster_bands=raster_bands, crs_mapping=crs_mapping
        )

    def copy(self) -> VRTDataset:
        return copy.deepcopy(self)

    @classmethod
    def from_xml(cls, xml: str) -> VRTDataset:
        vrt = ET.fromstring(xml)
        return cls.from_etree(vrt)

    @classmethod
    def load_vrt(cls, filepath: Path) -> VRTDataset:
        with open(filepath) as infile:
            return cls.from_xml(infile.read())

    def save_vrt(self, filepath: str | Path) -> None:
        with open(filepath, "w") as outfile:
            outfile.write(self.to_xml())

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
                    # new_filepath = filepath.with_stem(filepath.stem + "-" + str(j).zfill(2))
                    new_nest = nested_level.copy()
                    new_nest[-1] = j
                    new_filepaths = source.source_filename._save_vrt_nested(filepath, new_nest)
                    source.source_filename = new_filepaths[0]
                    source.relative_filename = False
                    filepaths += new_filepaths
                    j += 1

        vrt.save_vrt(save_filepath)
        # print(f"Saved {save_filepath}: {nested_level}")

        return filepaths

    def save_vrt_nested(self, filepath: Path | str) -> list[Path]:
        return list(set(self._save_vrt_nested(filepath=Path(filepath).absolute(), nested_level=[])))

    @classmethod
    def from_file(cls, filepaths: Path | str | list[Path | str], **kwargs: dict[str, Any]) -> VRTDataset:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_vrt = Path(temp_dir).joinpath("temp.vrt")

            build_vrt(output_filepath=temp_vrt, filepaths=filepaths, **kwargs)  # type: ignore
            return cls.load_vrt(temp_vrt)

    def sha1(self) -> str:
        return hashlib.sha1(str(self.__dict__).encode()).hexdigest()

    def is_nested(self) -> bool:
        for raster_band in self.raster_bands:
            for source in raster_band.sources:
                if hasattr(source.source_filename, "to_tempfiles"):
                    return True
        return False

    def to_tempfiles(
        self, temp_dir: TemporaryDirectory[str] | str | Path | None = None
    ) -> tuple[TemporaryDirectory[str] | str | Path, Path]:
        if temp_dir is None:
            temp_dir = TemporaryDirectory(prefix="variete")
        if isinstance(temp_dir, TemporaryDirectory):
            temp_dir_path = Path(temp_dir.name)
        else:
            temp_dir_path = Path(temp_dir)
        filepath = temp_dir_path.joinpath("vrtdataset.vrt")

        self.save_vrt_nested(filepath)
        return temp_dir, filepath

    def to_memfile(self) -> rio.MemoryFile:
        if self.is_nested():
            raise ValueError("Nested VRTs require temporary saving to work (see to_memfile_nested")
        return rio.MemoryFile(self.to_xml().encode(), ext=".vrt")

    def to_memfile_nested(
        self, temp_dir: TemporaryDirectory[str] | str | Path | None
    ) -> tuple[TemporaryDirectory[str] | Path | str | None, rio.MemoryFile]:
        if not self.is_nested():
            return (temp_dir, self.to_memfile())

        if temp_dir is None:
            temp_dir = TemporaryDirectory(prefix="variete")

        _, filepath = self.to_tempfiles(temp_dir=temp_dir)

        with open(filepath, "rb") as infile:
            return (temp_dir, rio.MemoryFile(infile.read()))

    @property
    def open_rio(self) -> Callable[..., rio.DatasetReader]:
        if self.is_nested():
            raise ValueError("Nested VRTs require temporary saving to work (see open_rio_nested")
        return self.to_memfile().open

    def open_rio_nested(
        self, temp_dir: TemporaryDirectory[str] | str | Path | None = None
    ) -> tuple[TemporaryDirectory[str] | str | Path | None, Callable[..., rio.DatasetReader]]:
        if not self.is_nested():
            return (temp_dir, self.open_rio)

        if temp_dir is None:
            temp_dir = TemporaryDirectory(prefix="variete")

        return (temp_dir, self.to_memfile_nested(temp_dir=temp_dir)[1].open)

    def sample(
        self,
        x_coord: float | Iterable[float],
        y_coord: float | Iterable[float],
        band: int | list[int] = 1,
        masked: bool = False,
    ) -> int | float | npt.NDArray[Any]:
        x_coords: Iterable[float] = []
        y_coords: Iterable[float] = []
        if isinstance(x_coord, float):
            x_coords = [x_coord]
        else:
            x_coords = x_coord
        if isinstance(y_coord, float):
            y_coords = [y_coord]
        else:
            y_coords = y_coord
        with self.open_rio() as raster:
            values = np.fromiter(
                raster.sample(zip(x_coords, y_coords), indexes=band, masked=masked),
                dtype=self.raster_bands[0].dtype,
                count=-1 if not hasattr(x_coords, "__len__") else len(x_coords),  # type: ignore
            ).ravel()
            if values.size > 1:
                return values
            return values[0]

    def __repr__(self) -> str:
        return "\n".join(
            [f"VRTDataset: shape={self.shape}, crs=EPSG:{self.crs.to_epsg()}, bounds: {self.bounds}"]
            + ["\t" + "\n\t".join(band.__repr__().splitlines()) for band in self.raster_bands]
        )


class WarpedVRTDataset(VRTDataset):
    """A VRTDataset that specifies a GDAL warp operation."""

    shape: tuple[int, int]
    crs: CRS
    crs_mapping: str
    transform: Affine
    block_size: tuple[int, int]
    raster_bands: list[AnyRasterBand]
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
        raster_bands: list[AnyRasterBand],
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
    def from_etree(cls, root: ET.Element) -> WarpedVRTDataset:
        initial = VRTDataset.from_etree(root)

        block_size = int(misc.find_element(root, "BlockXSize", True, "1")), int(
            misc.find_element(root, "BlockYSize", True, "1")
        )
        # block_size = tuple([int(getattr(root.find(f"Block{dim}Size"), "text", 0)) for dim in ["X", "Y"]])

        warp_options = misc.find_element(root, "GDALWarpOptions", False, None)

        resample_algorithm = misc.resampling_gdal_to_rio(
            misc.find_element(warp_options, "ResampleAlg", True, "bilinear")
        )
        dst_dtype = misc.dtype_gdal_to_numpy(misc.find_element(warp_options, "WorkingDataType", True, "float32"))

        warp_memory_limit = float(misc.find_element(warp_options, "WarpMemoryLimit", True, "0"))

        source_dataset_elem, source_dataset_text = misc.find_element(warp_options, "SourceDataset", text="both")
        source_dataset: str | Path = source_dataset_text

        if not source_dataset_text.startswith("/vsi"):
            source_dataset = Path(source_dataset)

        relative_filename = bool(int(source_dataset_elem.get("relativeToVRT", 0)))

        options = {}
        for option_elem in warp_options.findall("Option"):
            if (name := option_elem.get("name")) is not None:
                if option_elem.text is not None:
                    options[name] = option_elem.text

        transformer = misc.find_element(warp_options, ["Transformer", "ApproxTransformer"])

        max_error = float(getattr(transformer.find("MaxError"), "text", 0.125))

        proj_transformer = misc.find_element(transformer, ["BaseTransformer", "GenImgProjTransformer"])

        transforms = {}
        for key, gdal_key in [
            ("src_transform", "SrcGeoTransform"),
            ("src_inv_transform", "SrcInvGeoTransform"),
            ("dst_transform", "DstGeoTransform"),
            ("dst_inv_transform", "DstInvGeoTransform"),
        ]:
            transforms[key] = misc.parse_gdal_transform(misc.find_element(proj_transformer, gdal_key, text=True))

        band_mapping = []
        for band_map in misc.find_element(warp_options, "BandList").findall("BandMapping"):
            src = band_map.get("src")
            if src is None:
                raise ValueError("Invalid src in BandMapping")

            dst = band_map.get("dst")
            if dst is None:
                raise ValueError("Invalid dst in BandMapping")

            band_mapping.append((int(src), int(dst)))

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

    def to_etree(self) -> ET.Element:
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
                "SourceDataset", str(self.source_dataset), {"relativeToVRT": str(int(self.relative_filename or 0))}
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
            band_list.append(misc.new_element("BandMapping", None, {"src": str(src), "dst": str(dst)}))

        return vrt

    def is_nested(self) -> bool:
        return hasattr(self.source_dataset, "to_tempfiles")

    def _save_vrt_nested(self, filepath: Path, nested_level: list[int]) -> list[Path]:
        if len(nested_level) == 0:
            save_filepath = filepath
        else:
            save_filepath = filepath.with_stem(filepath.stem + "-nested-" + "-".join(map(str, nested_level)))

        nested_level += [0]
        filepaths = [save_filepath]
        vrt = self.copy()
        if vrt.is_nested():
            new_nest = nested_level[:-1] + [1]
            new_filepaths = vrt.source_dataset._save_vrt_nested(filepath, new_nest)
            vrt.source_dataset = new_filepaths[0]
            vrt.relative_filename = False
            filepaths += new_filepaths

        vrt.save_vrt(save_filepath)
        # print(f"Saved {save_filepath}: {nested_level}")

        return filepaths

    @classmethod  # type: ignore
    def from_file(
        cls, filepath: Path | str, dst_crs: CRS | int | str, **kwargs: dict[str, Any]
    ) -> VRTDataset:  # type: ignore
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_vrt = Path(temp_dir).joinpath("temp.vrt")

            build_warped_vrt(vrt_filepath=temp_vrt, filepath=filepath, dst_crs=dst_crs, **kwargs)  # type: ignore

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
        warnings.warn(f"Unexpected subClass tag: {subclass}. Ignoring it", stacklevel=2)

    return VRTDataset.from_etree(elem)


def load_vrt(filepath: str | Path) -> AnyVRTDataset:
    with open(filepath) as infile:
        root = ET.fromstring(infile.read())

    return dataset_from_etree(root)
