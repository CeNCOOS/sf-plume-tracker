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
def get_tides_series(start_time,end_time):
    ''' Using the model output dataframe use the start and end times to get the tide series from NOAA Tides and Currents API
    Input:
        df: dataframe with a datetime column 't'
    Output:
        tides: dataframe with tide series from NOAA Tides and Currents API
    '''
    #time_df=df['t'].dt.strftime('%Y%m%d')
    start_str = start_time.strftime('%Y%m%d')
    end_str = end_time.strftime('%Y%m%d')
    try:
        tides = pd.read_csv("https://api.tidesandcurrents.noaa.gov/api/prod/datagetter?begin_date={}&end_date={}&station=9414290&product=water_level&datum=MLLW&time_zone=gmt&units=metric&format=csv".format(start_str,end_str))
        tides['dateTime'] = pd.to_datetime(tides['Date Time'])
        tides.index = tides.dateTime
        return tides
    except Exception as e:
        print('Error in retrieving tide series from NOAA Tides and Currents API')
        print(e)
    return None


def get_high_tides():
    ''' Get the most recent high tides from NOAA Tides and Currents API
        Output:
            tides: dataframe with high tide series from NOAA Tides and Currents API
    '''
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
    '''
    Get high tides that were at least 48 hours previous to now.  This will allow for two days of model run
        Input:
            tides_df: dataframe with high tide series from NOAA Tides and Currents API
        Output:
            ts: timestamp of the most recent high tide that is at least 48 hours old
    '''
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
    ''' Load predefined regions of interest shapefiles for the SF Bay Area
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

def particle_tracking(date):
    ''' Run the updated SF Bay Area plume model using OpenDrift and save the output to a netCDF file
    Input:
        date: datetime object representing the start time of the model run
    Output:
        o: OceanDrift object containing the model output
    '''
    try:
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
    except Exception as e:
        print('Error in running the particle tracking model')
        print(e)    
    return None


def particle_in_polygon(model_output,gdf,time_step):
    '''
    Estimate the relative portion of particles in different predefined regions
        Input:
        model_output: OceanDrift object containing the model output
        gdf: geopandas dataframe with region of interest shapefiles
        time_step: integer representing the time step to analyze
        Output:
            list with relative portions of particles in each region [golden gate mouth, bolinas, sf peninsula, drakes]
    '''
    try:
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
    except Exception as e:
        print('Error in estimating particle distribution in polygons')
        print(e)
    return [0,0,0,0]
    

def load_bathy_data():
    '''
    Load Bathymetry data (.tiff) from outside the SF Bay Area
        Output: 
            xx: 1D array of x coordinates
            yy: 1D array of y coordinates 
            elv: 2D array of elevation values
    '''
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
    '''
    Load HFR surface currents data from the SF Bay Area
    Input:
        fname: string representing the file path to the netCDF file containing HFR surface currents data
    Output:
        ds: xarray dataset containing the HFR surface currents data for the last 48 hours
    '''
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