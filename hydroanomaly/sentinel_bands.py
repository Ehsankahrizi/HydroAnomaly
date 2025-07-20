"""
Sentinel-2 Satellite Data Retrieval using Google Earth Engine (GEE)

This module provides a function to retrieve Sentinel-2 satellite band data
for a specified location and time period, with masking and cloud filtering.
"""
import ee
import geemap 
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
import warnings

# Retrive data from Google Earth Engine ========================================================
def get_sentinel_bands_gee(
    latitude: float,
    longitude: float,
    start_date: str,
    end_date: str,
    bands: list = None,
    buffer_meters: int = 20,
    cloudy_pixel_percentage: int = 20,
    masks_to_apply: list = None
) -> pd.DataFrame:
    """
    Retrieve Sentinel-2 bands from Google Earth Engine, applying custom masking.

    Args:
        latitude (float): Latitude of center point.
        longitude (float): Longitude of center point.
        start_date (str): Start date as "YYYY-MM-DD".
        end_date (str): End date as "YYYY-MM-DD".
        bands (list): List of bands to retrieve (default is common Sentinel-2 bands).
        buffer_meters (int): Buffer size around the point, in meters.
        cloudy_pixel_percentage (int): Maximum allowed cloud percentage for each image.
        masks_to_apply (list): Masking strategies (e.g., ["water", "no_cloud_shadow", ...]).

    Returns:
        pd.DataFrame: DataFrame with band reflectance values per date.

    Example:
        >>> import ee
        >>> ee.Authenticate()
        >>> ee.Initialize()
        >>> df = get_sentinel_bands_gee(29.77, -95.06, "2021-01-01", "2021-12-31")
        >>> print(df.head())
    """
    if bands is None:
        bands = ['B2','B3','B4','B8','SCL']
    if masks_to_apply is None:
        masks_to_apply = ["water", "no_cloud_shadow", "no_clouds", "no_snow_ice", "no_saturated"]

    point = ee.Geometry.Point([longitude, latitude])
    buffered_point = point.buffer(buffer_meters)

    s2 = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
          .filterBounds(buffered_point)
          .filterDate(start_date, end_date)
          .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', cloudy_pixel_percentage))
          .select(bands))

    def dynamic_scl_mask(image):
        scl = image.select('SCL')
        mask = ee.Image.constant(1)
        if "water" in masks_to_apply:
            mask = mask.And(scl.eq(6))
        if "no_cloud_shadow" in masks_to_apply:
            mask = mask.And(scl.neq(3))
        if "no_clouds" in masks_to_apply:
            cloud_mask = scl.neq(8).And(scl.neq(9)).And(scl.neq(10))
            mask = mask.And(cloud_mask)
        if "no_snow_ice" in masks_to_apply:
            mask = mask.And(scl.neq(11))
        if "no_saturated" in masks_to_apply:
            mask = mask.And(scl.neq(1))
        return image.updateMask(mask)

    s2_masked = s2.map(dynamic_scl_mask)

    def extract_features(image):
        datetime = image.date().format('YYYY-MM-dd HH:mm:ss')
        values = image.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=buffered_point,
            scale=20,
            maxPixels=1e8
        )
        return ee.Feature(None, values.set('datetime', datetime))

    features = s2_masked.map(extract_features)
    fc = ee.FeatureCollection(features).filter(ee.Filter.notNull(['B2']))

    data = fc.getInfo()
    rows = [f['properties'] for f in data['features']]
    df = pd.DataFrame(rows)
    if not df.empty:
        df['datetime'] = pd.to_datetime(df['datetime'])
        df = df.sort_values('datetime')
        df = df.set_index('datetime')
    return df


# Showing map with NDWI (Normalized Difference Water Index) ========================================================
def show_sentinel_ndwi_map(
    latitude: float,
    longitude: float,
    start_date: str,
    end_date: str,
    buffer_meters: int = 20,
    cloudy_pixel_percentage: int = 20,
    zoom: int = 15
):
    """
    Display an interactive map showing the NDWI, point, and buffer.

    Args:
        latitude (float): Latitude of the center point.
        longitude (float): Longitude of the center point.
        start_date (str): Start date as "YYYY-MM-DD".
        end_date (str): End date as "YYYY-MM-DD".
        buffer_meters (int): Buffer radius in meters.
        cloudy_pixel_percentage (int): Max allowed cloud percentage.
        zoom (int): Zoom level for map.
    """
    point = ee.Geometry.Point([longitude, latitude])
    buffer = point.buffer(buffer_meters)

    image = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
             .filterBounds(buffer)
             .filterDate(start_date, end_date)
             .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", cloudy_pixel_percentage))
             .median())

    ndwi = image.normalizedDifference(["B3", "B8"]).rename("NDWI")

    ndwi_vis = {
        'min': 0,
        'max': 1,
        'palette': ['white', 'cyan', 'blue']}

    Map = geemap.Map()
    Map.centerObject(buffer, zoom=zoom)
    Map.addLayer(ndwi, ndwi_vis, "NDWI (Water)")
    Map.addLayer(point, {'color': 'yellow'}, 'Point')
    Map.addLayer(buffer, {'color': 'red'}, 'Buffer')
    Map.add_colorbar(ndwi_vis, label="NDWI", layer_name="NDWI (Water)")
    return Map


# Aliases for user convenience
get_sentinel_bands = get_sentinel_bands_gee
get_satellite_data = get_sentinel_bands_gee
get_sentinel = get_sentinel_bands_gee
