## This module creates static and animated visualizations of the particle tracking model output.
## Outputs are saved to model_output/static and model_output/animations folders
## And pushed to skyrocket8 at /var/www/html/data/hfr-particle-tracking-sfbay/

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

def load_roi_shapefiles():
    ''' Load Regions of Interest shapefiles
        Output:
            gdf: geopandas dataframe with regions of interest
    '''
    sf_penninsula = gpd.read_file("/home/pdaniel/SurfaceCurrentMaps/DailyModelRuns/data/sf_peninsula.json",driver='GeoJSON',features='sf_peninsula')
    gg_mouth = gpd.read_file("/home/pdaniel/SurfaceCurrentMaps/DailyModelRuns/data/sf-bay-seed.json",driver='GeoJSON',features='sf_bay-seed')
    bolinas = gpd.read_file("/home/pdaniel/SurfaceCurrentMaps/DailyModelRuns/data/bolinas.json",driver='GeoJSON',features='bolinas')
    drakes = gpd.read_file("/home/pdaniel/SurfaceCurrentMaps/DailyModelRuns/data/drakes_region.json",driver='GeoJSON',features='drakes_region')
    gdf = pd.concat([sf_penninsula,gg_mouth,bolinas,drakes])
    gdf['name'] = ['sf_peninsula','gg_mouth','bolinas','drakes']
    return gdf

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
    '''
    Load Bathymetry data (.tiff) from outside the SF Bay Area
        Output: 
            xx: 1D array of x coordinates
            yy: 1D array of y coordinates 
            elv: 2D array of elevation values
    '''
    ds = xr.open_dataset('/home/pdaniel/SurfaceCurrentMaps/DailyModelRuns/data/sf_bay_topo.nc')
    #ds = xr.open_dataset('/home/pdaniel/SurfaceCurrentMaps/DailyModelRuns/data/sf_bay_topo.tiff',engine='rasterio')
    elv = ds['band_data'].values
    xx = ds.x.values
    yy = ds.y.values
    return xx,yy,elv


def load_surface_currents(fname='/home/pdaniel/SuraceCurrentMaps/data/hfr-sfbay-2023_spring.nc',):
    '''
    Load HFR surface currents data from the SF Bay Area
    Input:
        fname: string representing the file path to the netCDF file containing HFR surface currents data
    Output:
        ds: xarray dataset containing the HFR surface currents data for the last 48 hours
    '''
    start_date=dt.datetime.utcnow()-dt.timedelta(days=2)
    #ds = xr.open_dataset('./data/surface_currents/hfr-sfbay-2024-april.nc')
    ds=xr.open_dataset('https://dods.ndbc.noaa.gov/thredds/dodsC/hfradar_uswc_2km')
    ds = ds.sel(time=slice(start_date,start_date+dt.timedelta(hours=48)),lat=slice(37.5,38),lon=slice(-123,-122.2))
    ds = ds[['u','v']]
    return ds


def make_map(xx,yy,elv):
    '''
    Create a map with bathymetry and coastlines
        Input:
            xx: 1D array of x coordinates
            yy: 1D array of y coordinates 
            elv: 2D array of elevation values
        Output:
            fig: matplotlib figure object
            ax: matplotlib axis object for the map
            ax_narrow: matplotlib axis object for the tide plot
    '''

    # This need to be redefined if we want to have a tidal plot also on the figure
    fig=plt.figure(figsize=(10,16))
    gs=fig.add_gridspec(4,1,hspace=0.1)
    ax=fig.add_subplot(gs[1:3,0],projection=ccrs.PlateCarree())
    #fig=plt.figure(figsize=(8,8))
    #gs=fig.add_gridspec(4,1,hspace=0.1)
    #ax=fig.add_suplot(gs[1:3,0],projection=ccrs.PlateCarree())

    #fig, ax = plt.subplots(1,subplot_kw={'projection': ccrs.PlateCarree()})
    #fig.set_size_inches(8,8)
    cmap = cmocean.cm.haline

    ax.add_feature(cfeature.LAND,zorder=-1)
    ax.add_feature(cfeature.COASTLINE,zorder=-1)


    ax.set_xlim(-123, -122.35)
    ax.set_ylim(37.5, 38.1)

    
    gl = ax.gridlines(draw_labels=True, linestyle='--',zorder=-3)
    gl.top_labels = False
    gl.right_labels = False


    levels = [0,10,20,50,100,200,500,1000]
    ax.contourf(xx,yy,-1*elv[0],zorder=-2,cmap=cmocean.cm.ice_r,levels=levels)
    #cont = ax.contour(xx,yy,-1*elv[0],levels=levels[1:],colors='k',linewidths=1,lw='solid')
    cont = ax.contour(xx,yy,-1*elv[0],levels=levels[1:],colors='k',linewidths=1,linestyles='solid')
    ax.clabel(cont, inline=True, fontsize=10, fmt='%1.0f')

    # Create a custom colorbar
    norm = mcolors.Normalize(vmin=0, vmax=96)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    # Add colorbar to the plot
    #cbar = plt.colorbar(sm, ax=ax, orientation='horizontal' ,anchor=(0.81,2.6),shrink=0.15,aspect=10)
    cbar = plt.colorbar(sm, ax=ax, orientation='vertical' ,anchor=(0.0,0.5),shrink=0.5,aspect=10)
    cbar.set_ticks([0,48,96])
    cbar.set_ticklabels(['0','24','48'])
    #cbar.set_ticklabels(['0','24','48'],fontweight='bold')
    #cbar.set_label('Hours', fontsize=10,labelpad=-40)
    cbar.set_label('Hours', fontsize=10,labelpad=-55)

    #cbar.set_label('Hours', fontsize=10, fontweight='bold', labelpad=-40)
    ax_narrow=fig.add_subplot(gs[3,0])
    pos_ax = ax.get_position()
    pos_axn = ax_narrow.get_position()
    new_pos_axn = [pos_ax.x0, pos_axn.y0-0.09, pos_ax.width, pos_axn.height+0.09]
    ax_narrow.set_position(new_pos_axn)
    return fig, ax, ax_narrow

def generate_static_plot(o,start_date):
    '''
    Generate a static plot of the particle trajectories and tide series
        Input:
            o: OceanDrift object containing the model output
            start_date: datetime object representing the start date of the model run
        Output:
            Saves a static plot to the model_output/static folder and pushes it to skyrocket8
    '''
    lons = o.history['lon']
    lats = o.history['lat']
    cmap = cmocean.cm.haline
    tide_series=get_tides_series(o.get_time_array()[0][0],o.get_time_array()[0][-1])

    xx,yy,elv = load_bathy_data()
    ds = load_surface_currents()


    fig, ax, ax_narrow = make_map(xx,yy,elv)

    for track in range(0, lats.shape[0]):
        llns = lons[track,:]
        llns = llns[llns.mask == False]
        lts = lats[track,:]
        lts = lts[lts.mask == False]
        
        ax.scatter(llns[0],lts[0],edgecolors='k',s=20,facecolors='k',zorder=100,marker='.')
        
        if llns.shape[0] <  48:
            last_known = llns.shape[0] -1
        
        else:
            last_known = 48
            
        ax.plot(llns[:last_known-1],lts[:last_known-1],color='.5',lw=1)
        
        ax.scatter(llns[last_known-1], lts[last_known-1],
                edgecolors='k',
                s=40,
                color=cmap(last_known/96),
                zorder=100,
                marker='o')
        
        if (last_known == 48) & (llns.shape[0] ==  96):
            last_known = 96
        else:
            last_known = llns.shape[0] -1
        
        if last_known > 48:
            ax.plot(llns[:last_known-1],lts[:last_known-1],color='.5',lw=1,ls='--')
            ax.scatter(llns[last_known-1], lts[last_known-1],
                edgecolors='k',
                s=40,
                color=cmap(last_known/96),
                zorder=100,
                marker='o')
            
    ax_narrow.plot(tide_series['dateTime'],tide_series[' Water Level'],color='k')
        # why would this have hours*2?
        #ax_narrow.scatter(tide_series['dateTime'][hours*2],tide_series[' Water Level'][hours*2],color='b')
    #ax_narrow.scatter(tide_series['dateTime'][hours],tide_series[' Water Level'][hours],color='b')
    ax_narrow.set_xlim(tide_series['dateTime'].iloc[0],tide_series['dateTime'].iloc[-1])
    ax_narrow.set_ylim(tide_series[' Water Level'].min(),tide_series[' Water Level'].max())
           
    day_str = (start_date).strftime("%Y-%m-%d")
    hour_str = (start_date).strftime("%H:%M")
    ax.text(1,1.05,day_str,fontweight='bold',fontsize=18,transform=ax.transAxes,ha='right',va='bottom')
    ax.text(1,1,hour_str,fontweight='bold',fontsize=18,transform=ax.transAxes,ha='right',va='bottom')

    plt.savefig(f"/home/pdaniel/SurfaceCurrentMaps/DailyModelRuns/model_output/static/sf_plume_static_{day_str}_res.png",bbox_inches='tight',pad_inches=0.1)
    plt.savefig(f"/home/pdaniel/SurfaceCurrentMaps/DailyModelRuns/model_output/static/sf_plume_static_latest_res.png",bbox_inches='tight',pad_inches=0.1)
    pcmd=f"scp -i /etc/ssh/keys/pdaniel/scp_rsa /home/pdaniel/SurfaceCurrentMaps/DailyModelRuns/model_output/static/sf_plume_static_latest_res.png skyrocket8.mbari.org:/var/www/html/data/hfr-particle-tracking-sfbay/sf_plume_static_latest_res.png"
    os.system(pcmd)


def generate_animation_img_stack(o, start_date, add_current_vectors=False):
    '''
    Generate an animation image stack of the particle trajectories and tide series
        Input:
            o: OceanDrift object containing the model output
            start_date: datetime object representing the start date of the model run
            add_current_vectors: boolean indicating whether to add current vectors to the plot
        Output:
            Saves an image stack to the model_output/animation-temp folder for creating an animation later
    '''
    
    # Remove all files in the temp_img_stack folder
    if add_current_vectors:
        output_dir = '/home/pdaniel/SurfaceCurrentMaps/DailyModelRuns/model_output/animation-temp/vector'
    else:
        output_dir = '/home/pdaniel/SurfaceCurrentMaps/DailyModelRuns/model_output/animation-temp/no_vector'
    delete_list = glob.glob(os.path.join(output_dir,"*.png"))
    os.system(f"rm {' '.join(delete_list)}")
    # using model output get the tide series for the model run
    tide_series=get_tides_series(o.get_time_array()[0][0],o.get_time_array()[0][-1])
    if add_current_vectors:
        hfr_current_vectors = load_surface_currents()    
        
    lons = o.history['lon']
    lats = o.history['lat']
    cmap = cmocean.cm.haline
    xx,yy,elv = load_bathy_data()
    gdf = load_roi_shapefiles()

    #lenlen=len(lons)
    [junk,ntm]=lons.shape
    #print('lon shape is '+str(lons.shape))
    for hours in tqdm.tqdm(range(ntm),file=sys.stdout):
    #for hours in tqdm.tqdm(range(96)):
    # hours = 24
        fig, ax, ax_narrow = make_map(xx,yy,elv)

        # Plot starting points
        llns = lons[:,0]
        llns = llns[llns.mask == False]
        lts = lats[:,0]
        lts = lts[lts.mask == False]
        ax.scatter(llns,lts,zorder=100,color='k',alpha=.5,s=20,marker='.')

        # Plot particle position at current Hour
        llns = lons[:,hours]
        llns = llns[llns.mask == False]
        lts = lats[:,hours]
        lts = lts[lts.mask == False]
        ax.scatter(llns,lts,color=cmap(hours/48),s=20)

        # Plot pervious 6 hours trajectory
        for j in range(lons.shape[0]):
            llns = lons[j,:]
            llns = llns[llns.mask == False]
            lts = lats[j,:]
            lts = lts[lts.mask == False]
            if llns.shape[0] > 1:
                if hours > 6:
                    ax.plot(llns[hours-6:hours],lts[hours-6:hours],color='.25',lw=2,zorder=-1)
                else:
                    ax.plot(llns[:hours],lts[:hours],color='.25',lw=2,zorder=-1)


        # Add Zones
        for loc in gdf['geometry']:
            ax.add_geometries([loc], crs=ccrs.PlateCarree(), facecolor='none', edgecolor='black',linewidth=2,zorder=-2)


        # Date and Hour Text
        day_str = (start_date + dt.timedelta(minutes=30)*hours).strftime("%Y-%m-%d")
        hour_str = (start_date + dt.timedelta(minutes=30)*hours).strftime("%H:%M")
        ax.text(1,1.05,day_str,fontweight='bold',fontsize=18,transform=ax.transAxes,ha='right',va='bottom')
        ax.text(1,1,hour_str,fontweight='bold',fontsize=18,transform=ax.transAxes,ha='right',va='bottom')


        # Inset Percentage Particle Plot
        ins = ax.inset_axes([0.45,0.75,0.2,0.2])
        values = particle_in_polygon(o,gdf,hours)
        values = [round(values[i]*100) for i in range(4)]
        ins.bar([1,2,3,4],values,align='center')
        ins.set_xticks([1,2,3,4])
        ins.set_xlim(0.5,4.5)
        ins.set_ylim(0,100)
        ins.set_ylabel('% Particles', fontdict={'fontweight':'bold'}, labelpad=-7 )
        ins.set_yticks([0,50,100])
        ins.set_yticklabels(['0','50','100'],fontweight='bold')
        ins.xaxis.set_ticklabels(['M','BLS','SF ','Dra'],fontweight='bold')
        ins.patch.set_facecolor('None')
        sns.despine(ax=ins)

        if add_current_vectors:
            vectors = hfr_current_vectors.sel(time=start_date+dt.timedelta(minutes=30*hours),method='nearest')[['u','v']]
            ax.quiver(hfr_current_vectors.lon, 
                      hfr_current_vectors.lat, 
                      vectors.u, 
                      vectors.v, 
                      scale=10)
        #
        # Add tide plot
        ax_narrow.plot(tide_series['dateTime'],tide_series[' Water Level'],color='k')
        # why would this have hours*2?
        theindextide=tide_series.index.get_loc(start_date+dt.timedelta(minutes=30*hours),method='nearest')
        ax_narrow.scatter(tide_series['dateTime'][theindextide],tide_series[' Water Level'][theindextide],color='b')
        #ax_narrow.scatter(tide_series['dateTime'][hours*2],tide_series[' Water Level'][hours*2],color='b')
        #ax_narrow.scatter(tide_series['dateTime'][hours],tide_series[' Water Level'][hours],color='b')
        ax_narrow.set_xlim(tide_series['dateTime'].iloc[0],tide_series['dateTime'].iloc[-1])
        ax_narrow.set_ylim(tide_series[' Water Level'].min(),tide_series[' Water Level'].max())

        # Save Figure
        if add_current_vectors:
            plt.savefig(f"/home/pdaniel/SurfaceCurrentMaps/DailyModelRuns/model_output/animation-temp/vector/{start_date.year}_sf_{str(hours).zfill(2)}_vector_res.png",bbox_inches='tight',pad_inches=0.1)
        
        else:
            plt.savefig(f"/home/pdaniel/SurfaceCurrentMaps/DailyModelRuns/model_output/animation-temp/no_vector/{start_date.year}_sf_{str(hours).zfill(2)}_res.png",bbox_inches='tight',pad_inches=0.1)
        
        plt.close()
    #Make anitimation from Image Stack
    if add_current_vectors:
        cmd = f"gm convert -delay 10 -loop 0 $(ls -1v /home/pdaniel/SurfaceCurrentMaps/DailyModelRuns/model_output/animation-temp/vector/*.png) /home/pdaniel/SurfaceCurrentMaps/DailyModelRuns/model_output/animations/{start_date.strftime('%Y%m%dT%H%M%S')}_48_hour_vector.gif"
        cmd1= f"gm convert -delay 10 -loop 0 $(ls -1v /home/pdaniel/SurfaceCurrentMaps/DailyModelRuns/model_output/animation-temp/vector/*.png) /home/pdaniel/SurfaceCurrentMaps/DailyModelRuns/model_output/animations/plumetrack_48_hour_vector.gif"
        pcmd=f"scp -i /etc/ssh/keys/pdaniel/scp_rsa /home/pdaniel/SurfaceCurrentMaps/DailyModelRuns/model_output/animations/plumetrack_48_hour_vector.gif skyrocket8.mbari.org:/var/www/html/data/hfr-particle-tracking-sfbay/plumetrack_48_hour_vector.gif"
    
    else:
        cmd = f"gm convert -delay 10 -loop 0 $(ls -1v /home/pdaniel/SurfaceCurrentMaps/DailyModelRuns/model_output/animation-temp/no_vector/*.png) /home/pdaniel/SurfaceCurrentMaps/DailyModelRuns/model_output/animations/{start_date.strftime('%Y%m%dT%H%M%S')}_48_hour.gif"
        cmd1= f"gm convert -delay 10 -loop 0 $(ls -1v /home/pdaniel/SurfaceCurrentMaps/DailyModelRuns/model_output/animation-temp/no_vector/*.png) /home/pdaniel/SurfaceCurrentMaps/DailyModelRuns/model_output/animations/plumetrack_48_hour.gif"
        pcmd=f"scp -i /etc/ssh/keys/pdaniel/scp_rsa /home/pdaniel/SurfaceCurrentMaps/DailyModelRuns/model_output/animations/plumetrack_48_hour.gif skyrocket8.mbari.org:/var/www/html/data/hfr-particle-tracking-sfbay/plumetrack_48_hour.gif"
    os.system(cmd)
    os.system(cmd1)
    os.system(pcmd)