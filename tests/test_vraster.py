from variete.vraster import VRaster
import tempfile
from pathlib import Path 
import rasterio as rio
import rasterio.warp
import numpy as np
import warnings
import pytest
import os

from test_vrt import make_test_raster


def test_load_vraster():
    with tempfile.TemporaryDirectory() as temp_dir:
        test_raster_path = Path(temp_dir).joinpath("test.tif")
        raster_params = make_test_raster(test_raster_path)

        vrst = VRaster.load_file(test_raster_path)

        assert vrst.crs == raster_params["crs"]
        assert vrst.transform == raster_params["transform"]
        assert vrst.shape == raster_params["data"].shape

def test_constant_add_subtract():
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
        assert vrst_subtracted.last.raster_bands[0].offset == 0.

        assert vrst.sample_rowcol(0, 0) + offset == vrst_added.sample_rowcol(0, 0)
        assert vrst.sample_rowcol(0, 0) == vrst_subtracted.sample_rowcol(0, 0)

def test_vraster_add_subtract():
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


    

def test_constant_multiply_divide():

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

        print(vrst_divided.steps[-2].dataset.raster_bands[0].scale)

        # The band has been multiplied and divided by 2, so the scale should now be 1.
        assert vrst_divided.last.raster_bands[0].scale == 1.
    
        assert vrst_multiplied.steps[-1].dataset.raster_bands[0].sources[0].source_filename == vrst_multiplied.steps[-2].dataset

        assert vrst.sample_rowcol(0, 0) * factor == vrst_multiplied.sample_rowcol(0, 0)
        assert vrst.sample_rowcol(0, 0) == vrst_divided.sample_rowcol(0, 0)


def test_sample():
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

        assert np.array_equal(vrst.sample_rowcol([0, lr_rowcol[0]], [0, lr_rowcol[1]]) , [upper_left_val, lower_right_val])

