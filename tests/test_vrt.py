import variete.vrt as vrtraster
import tempfile
from pathlib import Path 
import rasterio as rio
import rasterio.warp
import numpy as np
from osgeo import gdal


def make_test_raster(filepath: Path):
    crs = rio.crs.CRS.from_epsg(32633)
    transform = rio.transform.from_origin(5e5, 8.7e6, 10, 10) 

    data = np.multiply(*np.meshgrid(
        np.sin(np.linspace(0, np.pi * 2, 100)) * 5,
        np.sin(np.linspace(0, np.pi / 2, 50)) * 10,
    ))

    with rio.open(filepath, "w", "GTiff", width=data.shape[1], dtype="float32", height=data.shape[0], count=1, crs=crs, transform=transform, nodata=-9999) as raster:
        raster.write(data, 1)

    return {"crs": crs, "transform": transform, "data": data}


def test_create_vrt():
    with tempfile.TemporaryDirectory() as temp_dir:
        test_raster = Path(temp_dir).joinpath("test.tif")
        raster = make_test_raster(test_raster)

        vrt = vrtraster.VRTDataset.from_file(test_raster)

        assert vrt.transform == raster["transform"]
        assert vrt.crs == raster["crs"]
        assert vrt.shape == raster["data"].shape

        vrt_path = Path(temp_dir).joinpath("test.vrt")

        vrt.save_vrt(vrt_path)

        vrt_loaded = vrtraster.VRTDataset.load_vrt(vrt_path)

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
        
            
        vrt_mosaic = vrtraster.VRTDataset.from_file(test_raster_paths, separate=False)

        assert vrt_mosaic.crs == rasters[0]["crs"]
        assert vrt_mosaic.transform == rasters[0]["transform"]
        assert len(vrt_mosaic.raster_bands) == 1
        assert len(vrt_mosaic.raster_bands[0].sources) == len(test_raster_paths)

        assert vrt_mosaic.raster_bands[0].sources[-1].dst_window.x_off == warp_x_shift / warp_transform.a

        for i, source in enumerate(vrt_mosaic.raster_bands[0].sources):
            assert source.source_filename == test_raster_paths[i]

        vrt_separate = vrtraster.VRTDataset.from_file(test_raster_paths, separate=True)

        assert vrt_separate.crs == rasters[0]["crs"]
        assert vrt_separate.transform == rasters[0]["transform"]
        assert len(vrt_separate.raster_bands) == len(test_raster_paths)
        assert len(vrt_separate.raster_bands[0].sources) == 1

  

def test_main():

    with tempfile.TemporaryDirectory() as temp_dir:
        test_raster = Path(temp_dir).joinpath("test.tif")
        make_test_raster(test_raster)
        print(test_raster)
    
    print(vrtraster)

    ...