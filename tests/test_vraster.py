import tempfile
from pathlib import Path

import numpy as np
import pytest
import rasterio as rio
import rasterio.warp
from test_vrt import make_test_raster

import variete
from variete.vraster import VRaster


def _print_nested(vraster: VRaster) -> None:
    """Print the content of all vrts in a rendered nested VRaster."""
    with tempfile.TemporaryDirectory() as temp_dir:
        vraster.last.to_tempfiles(temp_dir)

        for filepath in Path(temp_dir).iterdir():
            print(f"{filepath}:")
            with open(filepath) as infile:
                print(infile.read())

            print("\n")


def test_load_vraster() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        test_raster_path = Path(temp_dir).joinpath("test.tif")
        raster_params = make_test_raster(test_raster_path)

        vrst = VRaster.load_file(test_raster_path)

        assert vrst.crs == raster_params["crs"]
        assert vrst.transform == raster_params["transform"]
        assert vrst.shape == raster_params["data"].shape

        vrst2 = variete.load(test_raster_path, nodata_to_nan=True)
        assert vrst2.steps[-1].name == "replace_nodata"

        vrst3 = variete.load(test_raster_path, nodata_to_nan=False)
        assert vrst3.steps[-1].name == vrst2.steps[-2].name


def test_vraster_read() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        test_raster_path = Path(temp_dir).joinpath("test.tif")
        raster_params = make_test_raster(test_raster_path)

        vrst = VRaster.load_file(test_raster_path)

        assert np.array_equal(vrst.read(1), raster_params["data"])
        assert np.array_equal(vrst.read(), raster_params["data"].reshape((1,) + raster_params["data"].shape))

        assert not hasattr(vrst.read(), "mask")

        assert hasattr(vrst.read(masked=True), "mask")


def test_constant_add_subtract() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        test_raster_path = Path(temp_dir).joinpath("test.tif")
        make_test_raster(test_raster_path)

        vrst = VRaster.load_file(test_raster_path)

        offset = 5
        vrst_added = vrst.add(offset)
        vrst_subtracted = vrst_added.subtract(5)

        assert len(vrst.steps) == 1
        assert len(vrst_added.steps) == 2
        assert len(vrst_subtracted.steps) == 3
        assert vrst_added.steps[-1].name == "add_constant"
        assert vrst_subtracted.steps[-1].name == "subtract_constant"
        assert vrst_subtracted.steps[-2].name == "add_constant"

        assert vrst_added.last.raster_bands[0].sources[0].source_filename == vrst_added.steps[-2].dataset

        # The band has been added and subtracted by 5, so the final offset should be 0
        assert vrst_subtracted.last.raster_bands[0].offset == 0.0

        assert vrst.sample_rowcol(0, 0) + offset == vrst_added.sample_rowcol(0, 0)
        assert vrst.sample_rowcol(0, 0) == vrst_subtracted.sample_rowcol(0, 0)


def test_vraster_add_subtract() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        test_raster_path_a = Path(temp_dir).joinpath("test_a.tif")
        make_test_raster(test_raster_path_a, mean_val=2)
        test_raster_path_b = Path(temp_dir).joinpath("test_b.tif")
        make_test_raster(test_raster_path_b, mean_val=5)

        vrst_a = VRaster.load_file(test_raster_path_a)
        vrst_b = VRaster.load_file(test_raster_path_b)

        vrst_added = vrst_a.add(vrst_b)

        assert vrst_a.sample_rowcol(0, 0) + vrst_b.sample_rowcol(0, 0) == vrst_added.sample_rowcol(0, 0)

        vrst_subtracted = vrst_added.subtract(vrst_b)

        assert vrst_subtracted.sample_rowcol(0, 0) == vrst_a.sample_rowcol(0, 0)


def test_constant_multiply_divide() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        test_raster_path = Path(temp_dir).joinpath("test.tif")
        make_test_raster(test_raster_path, mean_val=4)

        vrst = VRaster.load_file(test_raster_path)

        factor = 2
        vrst_multiplied = vrst.multiply(factor)
        vrst_divided = vrst_multiplied.divide(factor)

        assert vrst_multiplied.steps[-1].name == "multiply_constant"
        assert vrst_divided.steps[-1].name == "divide_constant"
        assert vrst_divided.steps[-2].name == "multiply_constant"

        assert vrst_multiplied.last.raster_bands[0].scale == factor

        # The band has been multiplied and divided by 2, so the scale should now be 1.
        assert vrst_divided.last.raster_bands[0].scale == 1.0

        assert (
            vrst_multiplied.steps[-1].dataset.raster_bands[0].sources[0].source_filename
            == vrst_multiplied.steps[-2].dataset
        )

        assert vrst.sample_rowcol(0, 0) * factor == vrst_multiplied.sample_rowcol(0, 0)
        assert vrst.sample_rowcol(0, 0) == vrst_divided.sample_rowcol(0, 0)


def test_vraster_multiply() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        test_raster_path_a = Path(temp_dir).joinpath("test_a.tif")
        make_test_raster(test_raster_path_a, mean_val=2)
        test_raster_path_b = Path(temp_dir).joinpath("test_b.tif")
        make_test_raster(test_raster_path_b, mean_val=5)

        vrst_a = VRaster.load_file(test_raster_path_a)
        vrst_b = VRaster.load_file(test_raster_path_b)

        vrst_multiplied = vrst_a.multiply(vrst_b)

        assert vrst_a.sample_rowcol(0, 0) * vrst_b.sample_rowcol(0, 0) == vrst_multiplied.sample_rowcol(0, 0)


def test_vraster_inverse() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        test_raster_path = Path(temp_dir).joinpath("test.tif")
        make_test_raster(test_raster_path, mean_val=2)

        vrst = VRaster.load_file(test_raster_path)

        vrst_inverse = vrst.inverse()

        assert 1 / vrst.sample_rowcol(0, 0) == vrst_inverse.sample_rowcol(0, 0)


def test_vraster_divide() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        test_raster_path_a = Path(temp_dir).joinpath("test_a.tif")
        make_test_raster(test_raster_path_a, mean_val=2)
        test_raster_path_b = Path(temp_dir).joinpath("test_b.tif")
        make_test_raster(test_raster_path_b, mean_val=5)

        vrst_a = VRaster.load_file(test_raster_path_a)
        vrst_b = VRaster.load_file(test_raster_path_b)

        vrst_divided = vrst_a.divide(vrst_b)

        assert vrst_a.sample_rowcol(0, 0) / vrst_b.sample_rowcol(0, 0) == vrst_divided.sample_rowcol(0, 0)


def test_sample() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        test_raster_path = Path(temp_dir).joinpath("test.tif")
        raster_params = make_test_raster(test_raster_path)

        vrst = VRaster.load_file(test_raster_path)

        bounds = vrst.bounds

        left, upper = bounds.left + vrst.res[0] / 2, bounds.top - vrst.res[1] / 2
        right, bottom = bounds.right - vrst.res[0] / 2, bounds.bottom + vrst.res[1] / 2

        upper_left_val = vrst.sample(left, upper)
        lower_right_val = vrst.sample(right, bottom)
        lr_rowcol = vrst.shape[0] - 1, vrst.shape[1] - 1

        assert upper_left_val == raster_params["data"][0, 0]
        assert lower_right_val == raster_params["data"][-1, -1]

        assert upper_left_val == vrst.sample_rowcol(0, 0)
        assert lower_right_val == vrst.sample_rowcol(*lr_rowcol)

        assert np.array_equal(
            vrst.sample_rowcol([0, lr_rowcol[0]], [0, lr_rowcol[1]]), [upper_left_val, lower_right_val]
        )


def test_save_vrt() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        test_raster_path_a = Path(temp_dir).joinpath("test_a.tif")
        make_test_raster(test_raster_path_a, mean_val=2)
        test_raster_path_b = Path(temp_dir).joinpath("test_b.tif")
        make_test_raster(test_raster_path_b, mean_val=5)

        vrst_a = VRaster.load_file(test_raster_path_a)
        vrst_b = VRaster.load_file(test_raster_path_b)

        vrst_added = vrst_a.add(vrst_b)
        vrst_multiplied = vrst_added.multiply(vrst_added)

        # _print_nested(vrst_multiplied)

        save_path = test_raster_path_a.with_stem("test_a_save").with_suffix(".vrt")
        vrst_a.save_vrt(save_path)

        loaded = VRaster.load_file(save_path)

        assert loaded.crs == vrst_a.crs
        assert loaded.transform == vrst_a.transform

        save_path = save_path.with_stem("test_added_save")

        vrst_multiplied.save_vrt(save_path)
        loaded = VRaster.load_file(save_path)

        assert loaded.crs == vrst_a.crs
        assert loaded.transform == vrst_a.transform

        assert loaded.sample_rowcol(0, 0) == vrst_multiplied.sample_rowcol(0, 0)


def test_replace_nodata() -> None:
    nodata_data = np.ones((50, 100), dtype="float32")
    nodata_data[:5, :5] = -9999

    other_data = np.ones_like(nodata_data) + 1

    with tempfile.TemporaryDirectory() as temp_dir:
        test_raster_path_a = Path(temp_dir).joinpath("test_a.tif")
        make_test_raster(test_raster_path_a, assign_values=nodata_data)
        test_raster_path_b = Path(temp_dir).joinpath("test_b.tif")
        make_test_raster(test_raster_path_b, assign_values=other_data)

        vrst_a = VRaster.load_file(test_raster_path_a)
        vrst_a_filled = vrst_a.replace_nodata(np.nan)

        vrst_b = VRaster.load_file(test_raster_path_b)
        vrst_b_filled = vrst_b.replace_nodata(np.nan)

        diff = vrst_a_filled.subtract(vrst_b_filled)
        prod = vrst_a_filled.multiply(vrst_b_filled)
        div = vrst_a_filled.divide(vrst_b_filled)
        inv = vrst_a_filled.inverse()

        for vrst, expected_ul, expected_mid in [
            (vrst_a, -9999.0, 1.0),
            (vrst_a_filled, "nan", 1.0),
            (vrst_b, 2.0, 2.0),
            (vrst_b_filled, 2.0, 2.0),
            (diff, "nan", -1.0),
            (prod, "nan", 2.0),
            (div, "nan", 0.5),
            (inv, "nan", 1.0),
        ]:
            value_ul = vrst.sample_rowcol(0, 0)
            if expected_ul == "nan":
                assert np.isnan(value_ul)
            else:
                assert value_ul == expected_ul

            assert vrst.sample_rowcol(25, 25) == expected_mid


def test_overloading() -> None:
    two_arr = np.ones((50, 100), dtype="float32") + 1
    four_arr = two_arr.copy() + 2

    with tempfile.TemporaryDirectory() as temp_dir:
        test_raster_path_a = Path(temp_dir).joinpath("test_a.tif")
        make_test_raster(test_raster_path_a, assign_values=two_arr)

        test_raster_path_b = Path(temp_dir).joinpath("test_b.tif")
        make_test_raster(test_raster_path_b, assign_values=four_arr)

        two = VRaster.load_file(test_raster_path_a)
        four = VRaster.load_file(test_raster_path_b)

        one = two / 2
        five = four + 1

        tests = [
            (two + four, 6.0),
            (four - one, 3.0),
            (four / two, 2.0),
            (four * two, 8.0),
            (four + 1, 5.0),
            (five - 1, 4.0),
            (four / 4, 1.0),
            (five * 5, 25.0),
            (1 + four, 5.0),
            (6 - four, 2.0),
            (8 * one, 8.0),
            (4 / four, 1.0),
        ]

        for i, (vrst, expected) in enumerate(tests):
            assert vrst.sample_rowcol(0, 0) == expected, f"test {i} failed"


def test_raster_warp() -> None:
    two_arr = np.ones((50, 100), dtype="float32") + 1
    four_arr = two_arr.copy() + 2

    with tempfile.TemporaryDirectory() as temp_dir:
        test_raster_path_a = Path(temp_dir).joinpath("test_a.tif")
        raster_a_params = make_test_raster(test_raster_path_a, assign_values=two_arr)

        test_raster_path_a_warp = test_raster_path_a.with_stem("test_a_warped.tif")

        raster_a_warp_params = {"data": np.zeros_like(raster_a_params["data"]), "crs": rio.CRS.from_epsg(3006)}
        _, raster_a_warp_params["transform"] = rasterio.warp.reproject(
            raster_a_params["data"],
            raster_a_warp_params["data"],
            dst_crs=raster_a_warp_params["crs"],
            src_transform=raster_a_params["transform"],
            src_crs=raster_a_params["crs"],
            resampling=rasterio.warp.Resampling.bilinear,
        )

        with rio.open(
            test_raster_path_a_warp,
            "w",
            driver="GTiff",
            width=raster_a_warp_params["data"].shape[1],
            height=raster_a_warp_params["data"].shape[0],
            count=1,
            crs=raster_a_warp_params["crs"],
            transform=raster_a_warp_params["transform"],
            dtype=raster_a_warp_params["data"].dtype,
        ) as raster:
            raster.write(raster_a_warp_params["data"], 1)

        test_raster_path_b = Path(temp_dir).joinpath("test_b.tif")
        make_test_raster(test_raster_path_b, assign_values=four_arr)

        vrst_a = VRaster.load_file(test_raster_path_a)
        vrst_a_warp = VRaster.load_file(test_raster_path_a_warp)
        vrst_b = VRaster.load_file(test_raster_path_b)

        assert vrst_a_warp.crs == raster_a_warp_params["crs"]
        assert vrst_a.transform == vrst_b.transform

        with pytest.raises(AssertionError, match="CRS is different.*"):
            vrst_a - vrst_a_warp

        vrst_a_warp_inverse = vrst_a_warp.warp(crs=vrst_a.crs, shape=vrst_a.shape, transform=vrst_a.transform)

        vrst_a_warp_inverse_b = vrst_a_warp.warp(reference=vrst_a)

        for warped in [vrst_a_warp_inverse, vrst_a_warp_inverse_b]:
            assert vrst_a._check_compatibility(warped) is None
            assert np.nanmedian(np.abs(warped.read(1) - vrst_a.read(1))) < 0.1


def test_write() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        test_raster_path_a = Path(temp_dir).joinpath("test_a.tif")
        make_test_raster(test_raster_path_a)
        vrst_a = VRaster.load_file(test_raster_path_a)

        out_filepath = test_raster_path_a.with_stem("written")

        # Validate that writing doesn't raise an error
        vrst_a.write(out_filepath)

        # The parent directory will not exist
        with pytest.raises(AssertionError, match=".*parent directory does not exist.*"):
            vrst_a.write("a/ab/c/d/e/f/g/h/i/j/k/l/")

        # The progress=True flag and a custom callback cannot be provided at the same time.
        with pytest.raises(ValueError, match="'progress' needs to be False if.*"):
            vrst_a.write(out_filepath, progress=True, callback=lambda *_: ...)


@pytest.mark.parametrize("compress", ["deflate", "lzw", None])  # type: ignore
@pytest.mark.parametrize("dtype", ["uint8", "uint16", "int16", "int32", "float32", "float64"])  # type: ignore
@pytest.mark.parametrize("tiled", [True, False])  # type: ignore
def test_write_scenarios(compress: str | None, dtype: str, tiled: bool) -> None:
    two_arr = np.ones((50, 100), dtype=dtype) + 1
    four_arr = two_arr.copy() + 2

    with tempfile.TemporaryDirectory() as temp_dir:
        test_raster_path_a = Path(temp_dir).joinpath("test_a.tif")
        raster_a_params = make_test_raster(test_raster_path_a, assign_values=two_arr, nodata=None, dtype=dtype)
        vrst_a = VRaster.load_file(test_raster_path_a)

        out_filepath = test_raster_path_a.with_stem("written")

        vrst_a.write(out_filepath, compress=compress, tiled=tiled)
        with rio.open(out_filepath) as raster:
            assert raster.dtypes[0] == four_arr.dtype
            assert raster.transform == raster_a_params["transform"]
            assert raster.crs == raster_a_params["crs"]

            assert raster.profile["tiled"] == tiled

            if compress is None:
                assert "compress" not in raster.profile
            else:
                assert raster.profile["compress"] == compress

            assert np.equal(raster_a_params["data"], raster.read(1)).all()
