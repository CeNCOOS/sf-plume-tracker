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
import requests, csv, time
#
def get_high_tides():
    try:
        noaa_api_request="https://api.tidesandcurrents.noaa.gov/api/prod/datagetter?date=recent&station=9414290&product=predictions&interval=hilo&datum=mllw&units=metric&time_zone=gmt&application=web_services&format=json"
        r=requests.get(noaa_api_request)
        if r.ok:
            df=pd.DataFrame(r.json()['predictions'])
            df['t']=pd.to_datetime(df["t"])
            high_tides=df.query("type=='H'")['t']
            return high_tides
        else:
            raise Exception("Bad request response")
    except requests.exceptions.ConnectionError:
        print('Trouble connecting to NOAA Tides and Currents API')

def check_recent_tides(tides_df):
    """
        Get high tides that were at least 48 hours previous to now.  This will allow two days of model to run
    """
    elapsed_time=dt.datetime.utcnow()-tides_df
    ix=elapsed_time[elapsed_time > dt.timedelta(days=2)].index # get index where tides are over 48 hours old
    ts=pd.to_datetime(list(tides_df[ix].values))
    ts=ts+dt.timedelta(hours=1.5)
    #date_str=ts.strftime('%Y-%m-%d %H:%M:%S')
    return ts
    #return date_str.tolist()

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
    o.seed_from_shapefile("/home/pdaniel/SurfaceCurrentMaps/DailyModelRuns/data/seed_shapefiles/sf-bay-seed-small-polygon.shp",number=50,time=date,layername=None)
    o.run(steps=48*4) # 24 hours
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


def make_map(xx,yy,elv):
    fig, ax = plt.subplots(1,subplot_kw={'projection': ccrs.PlateCarree()})
    fig.set_size_inches(8,8)
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
    cbar = plt.colorbar(sm, ax=ax, orientation='horizontal' ,anchor=(0.81,2.6),shrink=0.15,aspect=10)
    cbar.set_ticks([0,48,96])
    cbar.set_ticklabels(['0','24','48'])
    #cbar.set_ticklabels(['0','24','48'],fontweight='bold')
    cbar.set_label('Hours', fontsize=10,labelpad=-40)
    #cbar.set_label('Hours', fontsize=10, fontweight='bold', labelpad=-40)

    return fig, ax

def generate_static_plot(o,start_date):
    lons = o.history['lon']
    lats = o.history['lat']
    cmap = cmocean.cm.haline
    xx,yy,elv = load_bathy_data()
    ds = load_surface_currents()


    fig, ax = make_map(xx,yy,elv)

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
            
            
    day_str = (start_date).strftime("%Y-%m-%d")
    hour_str = (start_date).strftime("%H:%M")
    ax.text(1,1.05,day_str,fontweight='bold',fontsize=18,transform=ax.transAxes,ha='right',va='bottom')
    ax.text(1,1,hour_str,fontweight='bold',fontsize=18,transform=ax.transAxes,ha='right',va='bottom')

    plt.savefig(f"/home/pdaniel/SurfaceCurrentMaps/DailyModelRuns/model_output/static/sf_plume_static_{day_str}_res.png",bbox_inches='tight',pad_inches=0.1)
    plt.savefig(f"/home/pdaniel/SurfaceCurrentMaps/DailyModelRuns/model_output/static/sf_plume_static_latest_res.png",bbox_inches='tight',pad_inches=0.1)
    pcmd=f"scp -i /etc/ssh/keys/pdaniel/scp_rsa /home/pdaniel/SurfaceCurrentMaps/DailyModelRuns/model_output/static/sf_plume_static_latest_res.png skyrocket8.mbari.org:/var/www/html/data/hfr-particle-tracking-sfbay/sf_plume_static_latest_res.png"
    os.system(pcmd)


def generate_animation_img_stack(o, start_date, add_current_vectors=False):
    
    # Remove all files in the temp_img_stack folder
    if add_current_vectors:
        output_dir = '/home/pdaniel/SurfaceCurrentMaps/DailyModelRuns/model_output/animation-temp/vector'
    else:
        output_dir = '/home/pdaniel/SurfaceCurrentMaps/DailyModelRuns/model_output/animation-temp/no_vector'
    delete_list = glob.glob(os.path.join(output_dir,"*.png"))
    os.system(f"rm {' '.join(delete_list)}")
    
    
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
    for hours in tqdm.tqdm(range(ntm)):
    #for hours in tqdm.tqdm(range(96)):
    # hours = 24
        fig, ax = make_map(xx,yy,elv)

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

#
#
#
def main():
    tides=get_high_tides()
    recent_tides=check_recent_tides(tides)
    start_date=dt.datetime(recent_tides[-1].year,recent_tides[-1].month,recent_tides[-1].day,recent_tides[-1].hour,recent_tides[-1].minute)
    #start_date=recent_tides[-1]
    #print('Tide dates')
    #print(recent_tides[-1])
    #start_date=dt.datetime.utcnow()-dt.timedelta(days=2)
    #print('Start date format')
    #print(start_date)
    #start_date =dt.datetime(2024,4,12,12)
    o = particle_tracking(start_date)
            
    generate_static_plot(o,start_date)
    generate_animation_img_stack(o, start_date, add_current_vectors=True)
    generate_animation_img_stack(o, start_date, add_current_vectors=False)

if __name__=="__main__":
    main()

##generate_animation_img_stack(o,start_date)
##
##sns.set_context("talk")
##lons_2d = lons.reshape(30,30,49)
##lats_2d = lats.reshape(30,30,49)
### single_day = ds.sel(time=start_date,method='nearest')
##fig, ax = plt.subplots(1,subplot_kw={'projection': ccrs.PlateCarree()})
##fig.set_size_inches(8,9)
### single_day['speed'] = np.sqrt(single_day['u_mean']**2 + single_day['v_mean']**2) * 100
##
##ax.add_feature(cfeature.LAND)
##ax.add_feature(cfeature.OCEAN)
##ax.add_feature(cfeature.COASTLINE)
##cmap = plt.cm.Blues_r
##cmap_inner = plt.cm.Reds_r
##
##
##ax.set_xlim(-122.5,-121.74)
##ax.set_ylim(36.4,37)
##gl = ax.gridlines(draw_labels=True, linestyle='--')
##gl.top_labels = False
##gl.right_labels = False
##
##res_time,lon,lat = o.get_residence_time(5000)
##cax = ax.contourf(lon[1:],lat[1:],res_time.T/60,levels=[5,10,15,20,30],zorder=-1)
##
##cbar = plt.colorbar(cax,pad=0.05,shrink=.4,extendrect=True,orientation="horizontal",anchor=(.8,6.5),drawedges=False)
##cbar.set_label("hours",labelpad=-60)
##ax.set_xlim(-122.5,-121.74)
##ax.set_ylim(36.4,37)
##
### ax.text(x=0.05,y=.95,s="2022-05-25\n48 hr Residence Time",verticalalignment='top', transform=ax.transAxes)
### plt.savefig("./figures/relaxation_residences_2022.png", dpi=300)
##ax.text(x=0.05,y=.95,s=f"{start_date}\n48 hr Residence Time",verticalalignment='top', transform=ax.transAxes)
### plt.savefig("./figures/upwelling_residences_2022.png", dpi=300)
###
###
##total_length, distances, speeds = o.get_trajectory_lengths()
##
##total_length.reshape(30,30)
##
##sns.set_context("paper")
##lons_2d = lons.reshape(30,30,49)
##lats_2d = lats.reshape(30,30,49)
### single_day = ds.sel(time=start_date,method='nearest')
##fig, ax = plt.subplots(1,subplot_kw={'projection': ccrs.PlateCarree()})
##fig.set_size_inches(8,8)
### single_day['speed'] = np.sqrt(single_day['u_mean']**2 + single_day['v_mean']**2) * 100
##
##ax.add_feature(cfeature.LAND)
##ax.add_feature(cfeature.OCEAN)
##ax.add_feature(cfeature.COASTLINE)
##cmap = plt.cm.Blues_r
##cmap_inner = plt.cm.Reds_r
##
##
##i = 0
##llns = lons_2d[::2,:,i]
##llns = llns[llns.mask == False]
##lts = lats_2d[::2,:,i]
##lts = lts[lts.mask == False]
##
##cax = ax.scatter(llns,lts,zorder=100,c=total_length.reshape(30,30)[::2,:],alpha=1,s=20)
##cbar = plt.colorbar(cax,pad=0.05,shrink=.5)
##cbar.set_label("Trajectory Distance [m]")
##
##ax.set_xlim(-122.5,-121.74)
##ax.set_ylim(36.4,37)
##ax.gridlines(draw_labels=True, linestyle='--',)
##
### ax.scatter(lons[:,-1], lats[:,-1])
### single_day.plot.quiver(x='lon',y="lat",u="u_mean",v="v_mean",ax=ax)
### single_day['speed'].plot(ax=ax)
### single_day.plot.quiver(x='lon',y="lat",u="u_mean",v="v_mean",ax=ax)
##
### ax.set_title(f"{single_day.time.values}")
##
##avg_speed = np.mean(speeds,axis=0).reshape(30,30)
##
##sns.set_context("paper")
##lons_2d = lons.reshape(30,30,49)
##lats_2d = lats.reshape(30,30,49)
### single_day = ds.sel(time=start_date,method='nearest')
##fig, ax = plt.subplots(1,subplot_kw={'projection': ccrs.PlateCarree()})
##fig.set_size_inches(8,8)
### single_day['speed'] = np.sqrt(single_day['u_mean']**2 + single_day['v_mean']**2) * 100
##
##ax.add_feature(cfeature.LAND)
##ax.add_feature(cfeature.OCEAN)
##ax.add_feature(cfeature.COASTLINE)
##cmap = plt.cm.Blues_r
##cmap_inner = plt.cm.Reds_r
##
##
##i = 0
##llns = lons_2d[::2,:,i]
##llns = llns[llns.mask == False]
##lts = lats_2d[::2,:,i]
##lts = lts[lts.mask == False]
##
##cax = ax.scatter(llns,lts,zorder=100,c=avg_speed[::2,:],alpha=1,s=20)
##cbar = plt.colorbar(cax,pad=0.05,shrink=.5)
##cbar.set_label("Average Particle Speed [m]")
##
##ax.set_xlim(-122.5,-121.74)
##ax.set_ylim(36.4,37)
##ax.gridlines(draw_labels=True, linestyle='--',)
##
##
