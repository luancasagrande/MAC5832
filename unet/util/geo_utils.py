import numpy as np
import rasterio
from rasterio.plot import reshape_as_image
from affine import Affine
from rasterio.features import rasterize
from osgeo import gdal
from osgeo import ogr, osr
from skimage import io
from skimage.transform import resize, rotate
import math
from PIL import Image



def read_image(in_path):
    dataset = rasterio.open(in_path)
    aux = dataset.read()
    aux = reshape_as_image(aux)
    return aux

def get_multi_polygon(in_list):
    multipolygon = ogr.Geometry(ogr.wkbMultiPolygon)
    for item in in_list:
        multipolygon.AddGeometry(item)
    return multipolygon


def rasterize_list_geometries(in_list, in_shapeOut, in_geoTransform, in_dtype = 'uint8'):
    try:
        if(len(in_list)==0):
            return np.zeros(in_shapeOut, dtype=in_dtype)
        gtAffine = Affine(in_geoTransform[1], in_geoTransform[2], in_geoTransform[0],
                             in_geoTransform[4], in_geoTransform[5], in_geoTransform[3])
        return rasterize(in_list, out_shape=in_shapeOut, transform=gtAffine, dtype=in_dtype)
    except Exception as e:
        return np.zeros(in_shapeOut, dtype=in_dtype)


def gdal_rasterize(src_rast, mask_vect, in_outPath, in_attribute = 'ATTRIBUTE=MYFLD'):
    rast_ds = gdal.Open(src_rast)
    gt = rast_ds.GetGeoTransform()
    b = rast_ds.GetRasterBand(1)

    # Get vector metadata
    mask_ds = ogr.Open(mask_vect)
    mask_lyr = mask_ds.GetLayer(0)

    # Get EPSG info
    rast_srs = osr.SpatialReference(wkt=rast_ds.GetProjection())
    rast_srs.AutoIdentifyEPSG()
    rast_epsg = rast_srs.GetAttrValue('AUTHORITY',1)

    mask_srs = mask_lyr.GetSpatialRef()
    mask_srs.AutoIdentifyEPSG()
    mask_epsg = mask_srs.GetAttrValue('AUTHORITY',1)

    # Get raster corner points
    ul = gdal.ApplyGeoTransform(gt,0,0)
    ur = gdal.ApplyGeoTransform(gt,b.XSize,0)
    lr = gdal.ApplyGeoTransform(gt,b.XSize,b.YSize)
    ll = gdal.ApplyGeoTransform(gt,0,b.YSize)



    # Create raster to store mask
    drv = gdal.GetDriverByName('GTiff')

    mask_rast = drv.Create(in_outPath,b.XSize,b.YSize, 1, gdal.GDT_Float32,
                        options=['TILED=YES','COMPRESS=DEFLATE'])
    mask_rast.SetGeoTransform(gt)
    mask_rast.SetProjection(rast_srs.ExportToWkt())
    mask_band = mask_rast.GetRasterBand(1)
    mask_band.Fill(0)

    # Rasterize filtered layer into the mask tif
    gdal.RasterizeLayer(mask_rast, [1], mask_lyr,
                        options=[in_attribute])
    mask_rast = None
    mask_ds = None
    rast_ds = None


def gdal_translate(in_filePath, in_outFile, in_extent):
    ext = [in_extent[0], in_extent[3], in_extent[1], in_extent[2]]
    gdal.Translate(in_outFile, in_filePath, projWin=ext, projWinSRS='EPSG:4326')


def create_multi_band_geotiff(in_img, in_dataSet, in_outPath, in_outType=gdal.GDT_Byte):
    try:
        driver = in_dataSet.GetDriver()
        num_bands = 1
        if len(in_img.shape) == 3:
            num_bands = in_img.shape[2]
        outDS = driver.Create(in_outPath, in_img.shape[1], in_img.shape[0], num_bands, in_outType, ['COMPRESS=LZW'])
        in_geoTransform = in_dataSet.GetGeoTransform()
        outDS.SetGeoTransform(in_geoTransform)
        proj = in_dataSet.GetProjection()
        outDS.SetProjection(proj)
        for bandId in range(num_bands):
            if num_bands > 1:
                outDS.GetRasterBand(bandId + 1).WriteArray(in_img[:, :, bandId])
            else:
                outDS.GetRasterBand(bandId + 1).WriteArray(in_img)
        outDS.FlushCache()
        outDS = None
    except Exception as e:
        raise Exception("GDAL_MANAGER: Problem to compose geotiff: %s" % (str(e)))


def mock_shape_extent():
    x = r'C:\Users\Artur Oliveira\projetosdev\rivereye\code\mainsettings\mapretriever\static\mapretriever\demo\map0\0_risk.shp'
    y = r'C:\Users\Artur Oliveira\projetosdev\rivereye\code\temp\map.tiff'
    return shape_to_raster_based_on_extent([-47.46807756866092, -24.06611045358281, -47.42883822847743, -24.045727067751912], x, y)

def format_raster_to_report(in_raster, in_path, in_dict_color_map = None):
    """Format input raster to match report requirements
    ----------
    in_raster : 2D raster
    in_path : str
        Path where the formatted raster will be stored
    in_out_path : str
        Output directory where the file will be saved. Please note that this method is expecting that the path is valid
    in_dict_color_map : dict
        Dict that maps ids to output color
    """
    if(in_dict_color_map is None):
        in_dict_color_map = {1: [165, 191, 221],
                             2: [0, 255, 0],
                             3: [255, 0, 0]}

    if (in_raster.shape[0] > in_raster.shape[1]):
        in_raster = rotate(in_raster, 90)

    raster_out = np.zeros([in_raster.shape[0], in_raster.shape[1], 3], np.uint8)
    for key in in_dict_color_map:
        mask = in_raster == key
        raster_out[mask] = in_dict_color_map[key]

    raster_out = resize(raster_out, (400,600), anti_aliasing=True, preserve_range=True)

    io.imsave(in_path, raster_out.astype(np.uint8))


def shape_to_raster_based_on_extent(in_extent, in_shape_path, in_out_path, in_attribute = 'ATTRIBUTE=MYFLD',
                                    in_xsize = None, in_ysize = None, in_projection = None):
    """Compose a raster based on input shape and extent
    ----------
    in_extent : list float [minx, miny, maxx, maxy]
        List of float values taht describe the extent
    in_shape_path : str
        Directory from shape that will be rasterized
    in_out_path : str
        Output directory where the file will be saved. Please note that this method is expecting that the path is valid
    in_attribute : str
        Reference attribute to burn raster (Attribute name that can differentiate the polygons)
    in_xsize : float
        x resolution. Please note that if projection is not based on 4326 this value is mandatory
    in_ysize : float
        y resolution. Please note that if projection is not based on 4326 this value is mandatory
    in_ysize : str
        4326 projection
    """

    if(in_xsize is None or in_ysize is None or in_projection is None):
        #Sentinental 2 params for approximation
        in_x_res = 0.00011502124818
        in_y_res = -0.000091061392237
        #4326 projection base
        projection = 'GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563,AUTHORITY["EPSG","7030"]],' \
                     'AUTHORITY["EPSG","6326"]],PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],' \
                     'AXIS["Latitude",NORTH],AXIS["Longitude",EAST],AUTHORITY["EPSG","4326"]]'

    gt = [np.min(in_extent[0:2]), in_x_res, 0.0, np.max(in_extent[2:4]), 0.0, in_y_res]
    height = int(math.ceil(np.abs((in_extent[1] - in_extent[3]) / gt[-1])))
    width = int(math.ceil(np.abs((in_extent[0] - in_extent[2]) / gt[1])))

    gt = tuple(gt)
    # Get vector metadata
    mask_ds = ogr.Open(in_shape_path)
    mask_lyr = mask_ds.GetLayer(0)

    # Get EPSG info
    rast_srs = osr.SpatialReference(wkt=projection)
    rast_srs.AutoIdentifyEPSG()
    rast_epsg = rast_srs.GetAttrValue('AUTHORITY', 1)

    mask_srs = mask_lyr.GetSpatialRef()
    mask_srs.AutoIdentifyEPSG()
    mask_epsg = mask_srs.GetAttrValue('AUTHORITY', 1)

    # Create raster to store mask
    drv = gdal.GetDriverByName('GTiff')

    mask_rast = drv.Create(in_out_path, width, height, 1, gdal.GDT_Float32,
                           options=['TILED=YES', 'COMPRESS=DEFLATE'])
    mask_rast.SetGeoTransform(gt)
    mask_rast.SetProjection(rast_srs.ExportToWkt())
    mask_band = mask_rast.GetRasterBand(1)
    mask_band.Fill(0)

    # Rasterize filtered layer into the mask tif
    gdal.RasterizeLayer(mask_rast, [1], mask_lyr,
                        options=[in_attribute])
    mask_rast = None
    mask_ds = None
    rast_ds = None
    im = io.imread(in_out_path)
    format_raster_to_report(im, in_out_path[:-5] + ".jpg")



