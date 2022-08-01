from osgeo import ogr, osr, gdal
import copy

def read_shape_file_from_list_geo(in_path):
    file = ogr.Open(in_path)
    if file is None:
        return []
    layer = file.GetLayer(0)
    # numFeatures = layer.GetFeatureCount()

    ldefn = layer.GetLayerDefn()
    layerFieldNames = []
    for n in range(ldefn.GetFieldCount()):
        fdefn = ldefn.GetFieldDefn(n)
        layerFieldNames.append(fdefn.name)

    features = []
    for feature in layer:
        try:
            geom = feature.GetGeometryRef()
            #geo = GEOSGeometry(str(geom.ExportToWkt()))

            fieldsGeo = {}
            for field in layerFieldNames:
                fieldsGeo[field] = feature.GetField(field)

            fieldsGeo['envelop'] = geom.GetEnvelope()
            fieldsGeo['geometry'] = copy.deepcopy(geom)
            features.append(fieldsGeo)
        except Exception as e:
            continue

    return features

def add_area_per_polygon(in_shape):
    driver = ogr.GetDriverByName("ESRI Shapefile")
    dataSource = driver.Open(in_shape, 1)
    layer = dataSource.GetLayer()
    new_field = ogr.FieldDefn("AREA", ogr.OFTReal)
    new_field.SetWidth(32)
    new_field.SetPrecision(2)  # added line to set precision
    layer.CreateField(new_field)

    for feature in layer:
        geom = feature.GetGeometryRef()
        area = geom.GetArea()
        feature.SetField("Area", area)
        layer.SetFeature(feature)

    dataSource = None


def polygon_response(raster, poligonized_shp):
    src_ds = gdal.Open(raster)

    srcband = src_ds.GetRasterBand(1)
    #  create output datasource
    dst_layername = poligonized_shp
    drv = ogr.GetDriverByName("ESRI Shapefile")
    dst_ds = drv.CreateDataSource( dst_layername + ".shp" )
    targetprj = osr.SpatialReference(wkt=src_ds.GetProjection())
    dst_layer = dst_ds.CreateLayer(dst_layername, srs = targetprj)
    newField = ogr.FieldDefn('MYFLD', ogr.OFTInteger)
    dst_layer.CreateField(newField)

#    dst_layer.ImportFromEPSG(4326)
#    dst_layer.SetProjection(proj)

    gdal.Polygonize(srcband, None, dst_layer, 0, [], callback=None )
    dst_ds.Destroy()