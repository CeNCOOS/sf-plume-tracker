### Module for running the updated SF Bay Area plume model and saving daily outputs to netCDF files


from opendrift.readers import reader_netCDF_CF_generic
from opendrift.readers import reader_global_landmask
from opendrift.models.oceandrift import OceanDrift
import datetime as dt
import xarray as xr
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature 
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import geopandas as gpd
import pandas as pd
import shapely,tqdm,glob,cmocean, os
import requests, csv, time,sys
#
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

def check_recent_tides(tides_df):
    """
        Get high tides that were at least 48 hours previous to now.  This will allow two days of model to run
    """
    elapsed_time=dt.datetime.utcnow()-tides_df['t']
    ix=elapsed_time[elapsed_time > dt.timedelta(days=2)].index # get index where tides are over 48 hours old
    heights=tides_df['v'][ix]
    maxtideindex=np.argmax(heights)
    ts=pd.to_datetime(list(tides_df['t'][ix].values))
    ts=ts[maxtideindex]

    return ts

def load_roi_shapefiles():
    sf_penninsula = gpd.read_file("/home/pdaniel/SurfaceCurrentMaps/DailyModelRuns/data/sf_peninsula.json",driver='GeoJSON',features='sf_peninsula')
    gg_mouth = gpd.read_file("/home/pdaniel/SurfaceCurrentMaps/DailyModelRuns/data/sf-bay-seed.json",driver='GeoJSON',features='sf_bay-seed')
    bolinas = gpd.read_file("/home/pdaniel/SurfaceCurrentMaps/DailyModelRuns/data/bolinas.json",driver='GeoJSON',features='bolinas')
    drakes = gpd.read_file("/home/pdaniel/SurfaceCurrentMaps/DailyModelRuns/data/drakes_region.json",driver='GeoJSON',features='drakes_region')
    gdf = pd.concat([sf_penninsula,gg_mouth,bolinas,drakes])
    gdf['name'] = ['sf_peninsula','gg_mouth','bolinas','drakes']
    return gdf

def particle_tracking(date):
    o = OceanDrift(loglevel=50)
    reader_landmask = reader_global_landmask.Reader()
    url='https://dods.ndbc.noaa.gov/thredds/dodsC/hfradar_uswc_2km'
    #url = './data/surface_currents/hfr-sfbay-2024-april.nc'
    o.add_reader(reader_netCDF_CF_generic.Reader(url))
    o.add_reader(reader_landmask)
    #o.set_config('general:coastline_approximation_precision', .001)  # approx 100m
    o.set_config('general:coastline_action', 'stranding') 
    o.set_config('general:time_step_minutes',15)
    o.set_config('general:time_step_output_minutes',30)
    o.set_config('drift:scheme','runge-kutta')
    #o.set_config('drift:advection_scheme', 'runge-kutta')
    o.set_config('drift:stokes_drift', False)
    o.set_config('drift:current_uncertainty_uniform', .1)
    o.set_config('seed:ocean_only', False)
    o.seed_from_shapefile("/home/pdaniel/SurfaceCurrentMaps/DailyModelRuns/data/seed_shapefiles/sf-bay-seed-small-polygon.shp",number=100,time=date,layername=None)
    fname = "concave_hrf_" + date.strftime("%Y%m%dT%H%M%S")
    base_folder = "/home/pdaniel/SurfaceCurrentMaps/DailyModelRuns/model_output/netcdf_model_v2/"
    outfile = os.path.join(base_folder, fname + "_continuous.nc")
    o.run(steps=48*4, outfile = outfile) # 48 hours since 48*15min steps
    return o


def particle_in_polygon(model_output,gdf,time_step):
    """ Estimate the reatlive portion of particles in different predefined regions"""
    lons = model_output.history['lon']
    lats = model_output.history['lat']
    
    llns = lons[:,time_step]
    llns = llns[llns.mask == False]
    lts = lats[:,time_step]
    lts = lts[lts.mask == False]
    
    bolinas = 0
    mouth = 0
    peninsula = 0
    drakes = 0
    total = len(llns)
    for ln,lt in zip(llns,lts):
        out = gdf.contains(shapely.geometry.Point(ln,lt))
        if out.sum() > 0:
            if gdf.loc[out,'name'].values[0] == 'bolinas':
                bolinas = bolinas + 1
            elif gdf.loc[out,'name'].values[0] == 'gg_mouth':
                mouth = mouth + 1
            elif gdf.loc[out,'name'].values[0] == 'sf_peninsula':
                peninsula = peninsula + 1
            elif gdf.loc[out,'name'].values[0] == 'drakes':
                drakes = drakes + 1
                
    return [mouth/total, bolinas/total, peninsula/total, drakes/total]


def load_bathy_data():
    """ 
    Load Bathymetry data (.tiff) from outside the SF Bay Area
    
    """
    ds = xr.open_dataset('/home/pdaniel/SurfaceCurrentMaps/DailyModelRuns/data/sf_bay_topo.nc')
    #ds = xr.open_dataset('/home/pdaniel/SurfaceCurrentMaps/DailyModelRuns/data/sf_bay_topo.tiff',engine='rasterio')
    elv = ds['band_data'].values
    xx = ds.x.values
    yy = ds.y.values
    return xx,yy,elv


def load_surface_currents(fname='/home/pdaniel/SuraceCurrentMaps/data/hfr-sfbay-2023_spring.nc',):
    """
    Load HFR surface currents data from the SF Bay Area
    """
    start_date=dt.datetime.utcnow()-dt.timedelta(days=2)
    #ds = xr.open_dataset('./data/surface_currents/hfr-sfbay-2024-april.nc')
    ds=xr.open_dataset('https://dods.ndbc.noaa.gov/thredds/dodsC/hfradar_uswc_2km')
    ds = ds.sel(time=slice(start_date,start_date+dt.timedelta(hours=48)),lat=slice(37.5,38),lon=slice(-123,-122.2))
    ds = ds[['u','v']]
    return ds