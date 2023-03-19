from variete.vraster import VRaster
import tempfile
from pathlib import Path 
import rasterio as rio
import rasterio.warp
import numpy as np
import warnings
import pytest

from test_vrt import make_test_raster


def test_load_vraster():
    with tempfile.TemporaryDirectory() as temp_dir:
        test_raster_path = Path(temp_dir).joinpath("test.tif")
        raster_params = make_test_raster(test_raster_path)

        vrst = VRaster.load_file(test_raster_path)

        assert vrst.crs == raster_params["crs"]
        assert vrst.transform == raster_params["transform"]
        assert vrst.shape == raster_params["data"].shape

def test_add():
    with tempfile.TemporaryDirectory() as temp_dir:
        test_raster_path = Path(temp_dir).joinpath("test.tif")
        make_test_raster(test_raster_path)

        vrst = VRaster.load_file(test_raster_path)

        offset = 5
        vrst_added = vrst.add(offset)

        assert len(vrst.steps) == 1
        assert len(vrst_added.steps) == 2
        assert vrst_added.steps[-1].name == "add"

        assert vrst_added.steps[-1].dataset.raster_bands[0].sources[0].source_filename == vrst_added.steps[-2].dataset

        assert vrst.sample_rowcol(0, 0) + offset == vrst_added.sample_rowcol(0, 0)

def test_multiply():

    with tempfile.TemporaryDirectory() as temp_dir:
        test_raster_path = Path(temp_dir).joinpath("test.tif")
        make_test_raster(test_raster_path, mean_val=4)

        vrst = VRaster.load_file(test_raster_path)

        factor = 2
        vrst_multiplied = vrst.multiply(factor)

        assert vrst_multiplied.steps[-1].name == "multiply"
    
        assert vrst_multiplied.steps[-1].dataset.raster_bands[0].sources[0].source_filename == vrst_multiplied.steps[-2].dataset

        assert vrst.sample_rowcol(0, 0) * factor == vrst_multiplied.sample_rowcol(0, 0)


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

