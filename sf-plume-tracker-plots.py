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

def load_roi_shapefiles():
    sf_penninsula = gpd.read_file("./data/bay_zones/sf_peninsula.json",driver='GeoJSON',features='sf_peninsula')
    gg_mouth = gpd.read_file("./data/bay_zones/sf-bay-seed.json",driver='GeoJSON',features='sf_bay-seed')
    bolinas = gpd.read_file("./data/bay_zones/bolinas.json",driver='GeoJSON',features='bolinas')
    drakes = gpd.read_file("./data/bay_zones/drakes_region.json",driver='GeoJSON',features='drakes_region')
    gdf = pd.concat([sf_penninsula,gg_mouth,bolinas,drakes])
    gdf['name'] = ['sf_peninsula','gg_mouth','bolinas','drakes']
    return gdf

def particle_tracking(date):
    o = OceanDrift(loglevel=50)
    reader_landmask = reader_global_landmask.Reader()
    url = './data/surface_currents/hfr-sfbay-2024-april.nc'
    o.add_reader(reader_netCDF_CF_generic.Reader(url))
    o.add_reader(reader_landmask)
    o.set_config('general:coastline_approximation_precision', .001)  # approx 100m
    o.set_config('general:coastline_action', 'stranding') 
    o.set_config('general:time_step_minutes',15)
    o.set_config('general:time_step_output_minutes',30)
    o.set_config('drift:advection_scheme', 'runge-kutta')
    o.set_config('drift:stokes_drift', False)
    o.set_config('drift:current_uncertainty_uniform', .1)
    o.set_config('seed:ocean_only', False)
    o.seed_from_shapefile("./data/seed_shapefile/sf-bay-seed-small-polygon.shp",number=50,time=date,layername=None)
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
    ds = xr.open_dataset('./data/bathymetry/sf_bay_topo.tiff',engine='rasterio')
    elv = ds['band_data'].values
    xx = ds.x.values
    yy = ds.y.values
    return xx,yy,elv


def load_surface_currents(start_date, fname='./data/surface_currents/hfr-sfbay-2024-april.nc'):
    """
    Load HFR surface currents data from the SF Bay Area
    """
    ds = xr.open_dataset('./data/surface_currents/hfr-sfbay-2024-april.nc')
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
    cont = ax.contour(xx,yy,-1*elv[0],levels=levels[1:],colors='k',linewidths=1,lw='solid')
    ax.clabel(cont, inline=True, fontsize=10, fmt='%1.0f')

    # Create a custom colorbar
    norm = mcolors.Normalize(vmin=0, vmax=96)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    # Add colorbar to the plot
    cbar = plt.colorbar(sm, ax=ax, orientation='horizontal' ,anchor=(0.81,2.6),shrink=0.15,aspect=10)
    cbar.set_ticks([0,48,96])
    cbar.set_ticklabels(['0','24','48'],fontweight='bold')
    cbar.set_label('Hours', fontsize=10, fontweight='bold', labelpad=-40)

    return fig, ax

def generate_static_plot(o,start_date):
    lons = o.history['lon']
    lats = o.history['lat']
    cmap = cmocean.cm.haline
    xx,yy,elv = load_bathy_data()


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
            
        ax.plot(llns[:last_known-1],lts[:last_known-1],c='.5',lw=1)
        
        ax.scatter(llns[last_known-1], lts[last_known-1],
                edgecolors='k',
                s=40,
                c=cmap(last_known/96),
                zorder=100,
                marker='o')
        
        if (last_known == 48) & (llns.shape[0] ==  96):
            last_known = 96
        else:
            last_known = llns.shape[0] -1
        
        if last_known > 48:
            ax.plot(llns[:last_known-1],lts[:last_known-1],c='.5',lw=1,ls='--')
            ax.scatter(llns[last_known-1], lts[last_known-1],
                edgecolors='k',
                s=40,
                c=cmap(last_known/96),
                zorder=100,
                marker='o')
            
            
    day_str = (start_date).strftime("%Y-%m-%d")
    hour_str = (start_date).strftime("%H:%M")
    ax.text(1,1.05,day_str,fontweight='bold',fontsize=18,transform=ax.transAxes,ha='right',va='bottom')
    ax.text(1,1,hour_str,fontweight='bold',fontsize=18,transform=ax.transAxes,ha='right',va='bottom')

    plt.savefig(f"./figures/static/sf_plume_static_{day_str}_res.png",bbox_inches='tight',pad_inches=0.1)


def generate_animation_img_stack(o, start_date, add_current_vectors=False):
    
    # Remove all files in the temp_img_stack folder
    if add_current_vectors:
        output_dir = './figures/temp_img_stack/vector'
    else:
        output_dir = './figures/temp_img_stack/no_vector'
    delete_list = glob.glob(os.path.join(output_dir,"*.png"))
    os.system(f"rm {' '.join(delete_list)}")
    
    
    if add_current_vectors:
        hfr_current_vectors = load_surface_currents(start_date=start_date)    
        
    lons = o.history['lon']
    lats = o.history['lat']
    cmap = cmocean.cm.haline
    xx,yy,elv = load_bathy_data()
    gdf = load_roi_shapefiles()

    for hours in tqdm.tqdm(range(96)):
    # hours = 24
        fig, ax = make_map(xx,yy,elv)

        # Plot starting points
        llns = lons[:,0]
        llns = llns[llns.mask == False]
        lts = lats[:,0]
        lts = lts[lts.mask == False]
        ax.scatter(llns,lts,zorder=100,c='k',alpha=.5,s=20,marker='.')

        # Plot particle position at current Hour
        llns = lons[:,hours]
        llns = llns[llns.mask == False]
        lts = lats[:,hours]
        lts = lts[lts.mask == False]
        ax.scatter(llns,lts,c=cmap(hours/48),s=20)

        # Plot pervious 6 hours trajectory
        for j in range(lons.shape[0]):
            llns = lons[j,:]
            llns = llns[llns.mask == False]
            lts = lats[j,:]
            lts = lts[lts.mask == False]
            if llns.shape[0] > 1:
                if hours > 6:
                    ax.plot(llns[hours-6:hours],lts[hours-6:hours],c='.25',lw=2,zorder=-1)
                else:
                    ax.plot(llns[:hours],lts[:hours],c='.25',lw=2,zorder=-1)


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
            plt.savefig(f"./figures/temp_img_stack/vector/{start_date.year}_sf_{str(hours).zfill(2)}_vector_res.png",bbox_inches='tight',pad_inches=0.1)
        
        else:
            plt.savefig(f"./figures/temp_img_stack/no_vector/{start_date.year}_sf_{str(hours).zfill(2)}_res.png",bbox_inches='tight',pad_inches=0.1)
        
        plt.close()
    #Make anitimation from Image Stack
    if add_current_vectors:
        cmd = f"convert -delay 10 -loop 0 $(ls -1v ./figures/temp_img_stack/vector/*.png) ./figures/animations/{start_date.strftime('%Y%m%dT%H%M%S')}_48_hour_vector.gif"
    
    else:
        cmd = f"convert -delay 10 -loop 0 $(ls -1v ./figures/temp_img_stack/no_vector/*.png) ./figures/animations/{start_date.strftime('%Y%m%dT%H%M%S')}_48_hour.gif"
    os.system(cmd)


    

def main():
    start_date = dt.datetime(2024,4,12,12)
    o = particle_tracking(start_date)
    generate_static_plot(o,start_date)
    generate_animation_img_stack(o,start_date,add_current_vectors=False)
    generate_animation_img_stack(o,start_date,add_current_vectors=True)


if __name__ == "__main__":
    main()