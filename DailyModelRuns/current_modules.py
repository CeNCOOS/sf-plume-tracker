
import datetime as dt
import xarray as xr
import numpy as np
import geopandas as gpd
import pandas as pd
import shapely,tqdm,glob,cmocean, os
import requests, csv, time,sys


def get_tides_series(df):
    ''' Using the model outpu dataframe use the start and end times to get the tide series from NOAA Tides and Currents API
    Input:
        df: dataframe with a datetime column 't'
    Output:
        tides: dataframe with tide series from NOAA Tides and Currents API
    '''
    time_df=df['t'].dt.strftime('%Y%m%d')
    try:
        tides = pd.read_csv("https://api.tidesandcurrents.noaa.gov/api/prod/datagetter?begin_date={}&end_date={}&station=9414290&product=water_level&datum=MLLW&time_zone=gmt&units=metric&format=csv".format(time_df.iloc[0],time_df.iloc[-1]))
        tides['dateTime'] = pd.to_datetime(tides['Date Time'])
        tides.index = tides.dateTime
        return tides
    except Exception as e:
        print('Error in retrieving tide series from NOAA Tides and Currents API')
        print(e)
    return None


def get_high_tides():
    try:
        noaa_api_request="https://api.tidesandcurrents.noaa.gov/api/prod/datagetter?date=recent&station=9414290&product=predictions&interval=hilo&datum=mllw&units=metric&time_zone=gmt&application=web_services&format=json"
        r=requests.get(noaa_api_request)
        if r.ok:
            df=pd.DataFrame(r.json()['predictions'])
            df['t']=pd.to_datetime(df["t"])
            # we can get the height of the tide also
            # tides=df.query("type=='H'"),[['t','v']]
            high_tides=df.query("type=='H'")['t']
            other=df.query("type=='H'")[['t','v']]
            return other
            #return high_tides
        else:
            raise Exception("Bad request response")
    except requests.exceptions.ConnectionError:
        print('Trouble connecting to NOAA Tides and Currents API')
    except Exception as e:
        print('Error in retrieving high tides from NOAA Tides and Currents API')
        print(e)
    return None

def check_recent_tides(tides_df):
    """
        Get high tides that were at least 48 hours previous to now.  This will allow for two days of model run
    """
    try:
        elapsed_time=dt.datetime.utcnow()-tides_df['t']
        ix=elapsed_time[elapsed_time > dt.timedelta(days=2)].index # get index where tides are over 48 hours old
        heights=tides_df['v'][ix]
        maxtideindex=np.argmax(heights)
        ts=pd.to_datetime(list(tides_df['t'][ix].values))
        ts=ts[maxtideindex]
        return ts
    except Exception as e:
        print('Error in checking recent tides')
        print(e)
    return None 

def load_roi_shapefiles():
    ''' Load region of interest shapefiles
        Output:
            gdf: geopandas dataframe with region of interest shapefiles
        ''' 
    try:
        sf_penninsula = gpd.read_file("/home/pdaniel/SurfaceCurrentMaps/DailyModelRuns/data/sf_peninsula.json",driver='GeoJSON',features='sf_peninsula')
        gg_mouth = gpd.read_file("/home/pdaniel/SurfaceCurrentMaps/DailyModelRuns/data/sf-bay-seed.json",driver='GeoJSON',features='sf_bay-seed')
        bolinas = gpd.read_file("/home/pdaniel/SurfaceCurrentMaps/DailyModelRuns/data/bolinas.json",driver='GeoJSON',features='bolinas')
        drakes = gpd.read_file("/home/pdaniel/SurfaceCurrentMaps/DailyModelRuns/data/drakes_region.json",driver='GeoJSON',features='drakes_region')
        gdf = pd.concat([sf_penninsula,gg_mouth,bolinas,drakes])
        gdf['name'] = ['sf_peninsula','gg_mouth','bolinas','drakes']
        return gdf
    except Exception as e:
        print('Error in loading region of interest shapefiles')
        print(e)
    return None

 def load_bathy_data():
    """ 
    Load Bathymetry data (.tiff) from outside the SF Bay Area
    
    """
    try:
        ds = xr.open_dataset('/home/pdaniel/SurfaceCurrentMaps/DailyModelRuns/data/sf_bay_topo.nc')
        #ds = xr.open_dataset('/home/pdaniel/SurfaceCurrentMaps/DailyModelRuns/data/sf_bay_topo.tiff',engine='rasterio')
        elv = ds['band_data'].values
        xx = ds.x.values
        yy = ds.y.values
        return xx,yy,elv
    except Exception as e:
        print('Error in loading bathymetry data')
        print(e)
    return None,None,None   

 def load_surface_currents(fname='/home/pdaniel/SuraceCurrentMaps/data/hfr-sfbay-2023_spring.nc',):
    """
    Load HFR surface currents data from the SF Bay Area
    """
    try:
        start_date=dt.datetime.utcnow()-dt.timedelta(days=2)
        #ds = xr.open_dataset('./data/surface_currents/hfr-sfbay-2024-april.nc')
        ds=xr.open_dataset('https://dods.ndbc.noaa.gov/thredds/dodsC/hfradar_uswc_2km')
        ds = ds.sel(time=slice(start_date,start_date+dt.timedelta(hours=48)),lat=slice(37.5,38),lon=slice(-123,-122.2))
        ds = ds[['u','v']]
        return ds
    except Exception as e:
        print('Error in loading surface currents data')
        print(e)
    return None 
    
   

