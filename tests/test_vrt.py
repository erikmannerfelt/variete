import variete.vrt
from variete.vrt import raster_bands, pixel_functions
import tempfile
from pathlib import Path 
import rasterio as rio
import rasterio.warp
import numpy as np
import warnings
import os
import pytest


def make_test_raster(filepath: Path, nodata: float = -9999., mean_val: int | float | None = None, assign_values: np.ndarray | None = None):
    crs = rio.crs.CRS.from_epsg(32633)
    transform = rio.transform.from_origin(5e5, 8.7e6, 10, 10) 

    if assign_values is not None:
        data = assign_values
    else:
        data = np.multiply(*np.meshgrid(
            np.sin(np.linspace(0, np.pi * 2, 100)) * 5,
            np.sin(np.linspace(0, np.pi / 2, 50)) * 10,
        )).astype("float32")

        if mean_val is not None:
            data += mean_val - data.mean()

    with rio.open(filepath, "w", "GTiff", width=data.shape[1], dtype="float32", height=data.shape[0], count=1, crs=crs, transform=transform, nodata=nodata) as raster:
        raster.write(data, 1)

    return {"crs": crs, "transform": transform, "data": data}

def test_create_vrt():
    with tempfile.TemporaryDirectory() as temp_dir:
        test_raster = Path(temp_dir).joinpath("test.tif")
        raster = make_test_raster(test_raster)

        vrt = variete.vrt.VRTDataset.from_file(test_raster)

        assert vrt.transform == raster["transform"]
        assert vrt.crs == raster["crs"]
        assert vrt.shape == raster["data"].shape

        vrt_path = Path(temp_dir).joinpath("test.vrt")

        vrt.save_vrt(vrt_path)

        vrt_loaded = variete.vrt.VRTDataset.load_vrt(vrt_path)

        assert vrt.crs == vrt_loaded.crs

        orig_lines = vrt.to_xml().splitlines()
        loaded_lines = vrt_loaded.to_xml().splitlines()
        for i in range(len(orig_lines)):
            if "<SRS" in orig_lines[i]:
                continue
            assert orig_lines[i] == loaded_lines[i]


def test_multiple_vrt():

    with tempfile.TemporaryDirectory() as temp_dir_str:
        temp_dir = Path(temp_dir_str)

        test_raster_paths = [temp_dir.joinpath(f"test{i}.tif") for i in [0, 1]]
        rasters = [make_test_raster(fp) for fp in test_raster_paths]

        warp_transform = list(rasters[0]["transform"].to_gdal())
        warp_x_shift = 2.5
        warp_transform[0] += warp_x_shift
        warp_transform = rio.Affine.from_gdal(*warp_transform)


        data_warp = np.empty((rasters[0]["data"].shape[0] - 3, rasters[0]["data"].shape[1]), dtype="float32")
        rasterio.warp.reproject(
            rasters[0]["data"].copy(),
            src_transform=rasters[0]["transform"],
            src_crs=rasters[0]["crs"],
            dst_crs=rasters[0]["crs"],
            dst_transform=warp_transform,
            destination=data_warp,
        )

        warp_filepath = test_raster_paths[0].with_stem("test0_warp")
        with rio.open(warp_filepath, "w", "GTiff", width=data_warp.shape[1], height=data_warp.shape[0], count=1, dtype="float32", nodata=-9998, transform=warp_transform, crs=rasters[0]["crs"]) as raster:
            raster.write(data_warp, 1)
       
        test_raster_paths.append(warp_filepath)


        # gdal_vrt = temp_dir.joinpath("gdal.vrt")
        # gdal.BuildVRT(str(gdal_vrt), list(map(str, test_raster_paths))) 

        # with open(gdal_vrt) as infile:
        #     print(infile.read())

        #raise NotImplementedError()
        
            
        vrt_mosaic = variete.vrt.VRTDataset.from_file(test_raster_paths, separate=False)

        assert vrt_mosaic.crs == rasters[0]["crs"]
        assert vrt_mosaic.transform == rasters[0]["transform"]
        assert len(vrt_mosaic.raster_bands) == 1
        assert len(vrt_mosaic.raster_bands[0].sources) == len(test_raster_paths)

        assert vrt_mosaic.raster_bands[0].sources[-1].dst_window.x_off == warp_x_shift / warp_transform.a

        for i, source in enumerate(vrt_mosaic.raster_bands[0].sources):
            assert source.source_filename == test_raster_paths[i]

        vrt_separate = variete.vrt.VRTDataset.from_file(test_raster_paths, separate=True)

        assert vrt_separate.crs == rasters[0]["crs"]
        assert vrt_separate.transform == rasters[0]["transform"]
        assert len(vrt_separate.raster_bands) == len(test_raster_paths)
        assert len(vrt_separate.raster_bands[0].sources) == 1


def test_warped_vrt():

    with tempfile.TemporaryDirectory() as temp_dir:
        test_raster_path = Path(temp_dir).joinpath("test.tif")
        make_test_raster(test_raster_path)

        warped = variete.vrt.WarpedVRTDataset.from_file(test_raster_path, dst_crs=32634)
        vrt_path = test_raster_path.with_suffix(".vrt")
        warped.save_vrt(vrt_path)

        with rio.open(vrt_path) as raster:
            assert raster.crs.to_epsg() == 32634
            assert raster.nodata == -9999
        

def create_vrt():
    with tempfile.TemporaryDirectory() as temp_dir:
        test_raster_path = Path(temp_dir).joinpath("test.tif")
        raster = make_test_raster(test_raster_path)

        vrt = variete.vrt.VRTDataset.from_file(test_raster_path)

    return raster, vrt


def test_with_open():
    raster_params, vrt = create_vrt()

    with vrt.open_rio() as raster:
        assert raster.crs == raster_params["crs"]
        assert raster.transform == raster_params["transform"]

    memfile = vrt.to_memfile()

    assert isinstance(memfile, rio.MemoryFile)


def test_load_vrt():
    with tempfile.TemporaryDirectory() as temp_dir:
        test_raster_path = Path(temp_dir).joinpath("test.tif")
        make_test_raster(test_raster_path)

        vrt_path = test_raster_path.with_suffix(".vrt")

        vrt = variete.vrt.VRTDataset.from_file(test_raster_path)
        vrt.save_vrt(vrt_path)

        vrt_warp_path = vrt_path.with_stem("warped")
        vrt_warp = variete.vrt.WarpedVRTDataset.from_file(vrt_path, dst_crs=vrt.crs.to_epsg() + 1)
        vrt_warp.save_vrt(vrt_warp_path)

        loaded_vrt = variete.vrt.load_vrt(vrt_path)
        loaded_warp_vrt = variete.vrt.load_vrt(vrt_warp_path)

        assert type(loaded_vrt) == variete.vrt.VRTDataset
        assert type(loaded_warp_vrt) == variete.vrt.WarpedVRTDataset
        assert type(loaded_warp_vrt) != variete.vrt.VRTDataset

        for [vrt0, vrt1] in [(vrt, loaded_vrt), (vrt_warp, loaded_warp_vrt)]:
            assert vrt0.crs == vrt1.crs
            assert vrt0.transform == vrt1.transform


def test_set_offset_scale():
    warnings.simplefilter("error")
    with tempfile.TemporaryDirectory() as temp_dir:
        test_raster_path = Path(temp_dir).joinpath("test.tif")
        raster_params = make_test_raster(test_raster_path, nodata=None, mean_val=5)

        #vrt_path = test_raster_path.with_suffix(".vrt")
        vrt = variete.vrt.VRTDataset.from_file(test_raster_path)

        offset = 2
        scale = 3
        vrt.raster_bands[0].offset = offset
        vrt.raster_bands[0].scale = scale

        vrt.raster_bands[0] = raster_bands.VRTDerivedRasterBand.from_raster_band(vrt.raster_bands[0], pixel_functions.ScalePixelFunction()) 

        print(vrt.to_xml())

        with vrt.open_rio() as raster:
            loaded_raster = raster.read(1)

        assert np.sum((scale * raster_params["data"]) + offset) == np.sum(loaded_raster)


def test_pixel_function():
    warnings.simplefilter("error")

    with tempfile.TemporaryDirectory() as temp_dir:
        test_raster_path = Path(temp_dir).joinpath("test.tif")
        raster_params = make_test_raster(test_raster_path, nodata=None, mean_val=5)

        #vrt_path = test_raster_path.with_suffix(".vrt")
        vrt = variete.vrt.VRTDataset.from_file(test_raster_path)

        constant = 5
        add_function = pixel_functions.SumPixelFunction(constant)

        vrt.raster_bands[0] = raster_bands.VRTDerivedRasterBand.from_raster_band(vrt.raster_bands[0], add_function)

        vrt.raster_bands[0].sources.append(vrt.raster_bands[0].sources[0].copy())

        with vrt.open_rio() as raster:
            added_raster = raster.read(1)

        assert np.round(np.nanmean((added_raster - constant) / raster_params["data"]), 4) == 2.0
            

def test_copy():

    _, vrt_a = create_vrt()

    vrt_a_copy = vrt_a.copy()

    vrt_a.crs = "1"
    vrt_a.raster_bands[0].sources[0].source_filename = "f"

    assert vrt_a_copy.crs != "1"
    assert vrt_a_copy.raster_bands[0].sources[0].source_filename != "f"

    

def test_open_nested():
    warnings.simplefilter("error")
    with tempfile.TemporaryDirectory() as temp_dir:
        test_raster_path = Path(temp_dir).joinpath("test.tif")
        make_test_raster(test_raster_path, nodata=None, mean_val=5)

        #vrt_path = test_raster_path.with_suffix(".vrt")
        vrt_a = variete.vrt.VRTDataset.from_file(test_raster_path)

        vrt_b = vrt_a.copy()
        #vrt2 = vrt.copy()
        vrt_a.raster_bands[0].sources[0].source_filename = vrt_b

        with pytest.raises(ValueError):
            vrt_a.open_rio()
            vrt_a.to_memfile()

        temp_dir2, reader = vrt_a.open_rio_nested()
        with reader() as raster:
            assert raster.crs == vrt_a.crs

        temp_dir2.cleanup()
            
            
def test_to_tempfiles():
    warnings.simplefilter("error")

    with tempfile.TemporaryDirectory() as temp_dir:
        test_raster_path = Path(temp_dir).joinpath("test.tif")
        make_test_raster(test_raster_path, nodata=None, mean_val=5)

        #vrt_path = test_raster_path.with_suffix(".vrt")
        vrt_a = variete.vrt.VRTDataset.from_file(test_raster_path)

        vrt_b = vrt_a.copy()
        #vrt2 = vrt.copy()
        vrt_a.raster_bands[0].sources[0].source_filename = vrt_b

        temp_dir2, temp_a_filepath = vrt_a.to_tempfiles()

        loaded_vrt_a = variete.vrt.load_vrt(temp_a_filepath)

        assert vrt_a.crs == loaded_vrt_a.crs

        temp_filepaths = os.listdir(temp_dir2.name)
        assert len(temp_filepaths) == 2

        temp_b_filepath = Path(temp_dir2.name).joinpath([p for p in temp_filepaths if p != temp_a_filepath.name][0])

        assert loaded_vrt_a.raster_bands[0].sources[0].source_filename == temp_b_filepath

        temp_dir2.cleanup()


def test_sample():
    warnings.simplefilter("error")
        
        
    with tempfile.TemporaryDirectory() as temp_dir:
        test_raster_path = Path(temp_dir).joinpath("test.tif")
        raster_params = make_test_raster(test_raster_path, nodata=None, mean_val=5)

        #vrt_path = test_raster_path.with_suffix(".vrt")
        vrt = variete.vrt.VRTDataset.from_file(test_raster_path)

        
        left, upper = vrt.transform.c + vrt.res[0] / 2, vrt.transform.f - vrt.res[1] / 2

        sampled = vrt.sample(left, upper)

        assert sampled == raster_params["data"][0, 0]


    
  

def test_main():

    with tempfile.TemporaryDirectory() as temp_dir:
        test_raster = Path(temp_dir).joinpath("test.tif")
        make_test_raster(test_raster)
    
