from math import e
import xarray as xr
import tqdm
import numpy as np
import pandas as pd
import datetime as dt
from datetime import timedelta
import seaborn as sns
import cmocean as cm
from shapely.geometry import Point, Polygon
import os
import cartopy.crs as ccrs
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.colorbar as mcbar
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import geopandas as gpd
from mpl_toolkits.axes_grid1 import make_axes_locatable
import warnings
import scipy.ndimage as ndimage
from tqdm import tqdm
import glob
from PIL import Image
warnings.filterwarnings('ignore')

def make_map():
    fig = plt.figure(figsize=(12,9))
    cart_proj = ccrs.PlateCarree()
    ax = plt.axes(projection=cart_proj)
    ax.coastlines('10m', linewidth=0.8,zorder=200)
    ax.set_xlim(-123.05,-122.3)
    ax.set_ylim(37.5,38.)
    return(fig, ax)


def make_map_colorbar():
    fig = plt.figure(figsize=(12,9))
    gs = fig.add_gridspec(2, 1, height_ratios=[4, 1],hspace=.75)
    cart_proj = ccrs.PlateCarree()
    ax = fig.add_subplot(gs[0],projection=ccrs.PlateCarree())
    ax = plt.axes(projection=cart_proj)
    ax.coastlines('10m', linewidth=0.8,zorder=200)
    ax.set_xlim(-123.05,-122.3)
    ax.set_ylim(37.5,38.)
    ax_cb = fig.add_subplot(gs[1])
    return(fig, ax, ax_cb)


def plot_radars(current_time, radar, ax):
    ''' '''
    current_radar = radar.sel(lat=slice(37.244221, 38.233120),lon=slice(-123.99,  -122.177032)).sel(time=current_time,method='nearest')
    current_radar['speed'] = np.sqrt(current_radar['u']**2 + current_radar['v']**2)
    u = current_radar['u']
    v = current_radar['v']
    x = current_radar['lon'].values
    y = current_radar['lat'].values
    xx,yy = np.meshgrid(x,y)
    # current_radar['speed'].plot(alpha=.5)
    cax = plt.contourf(x,y,current_radar['speed'],vmin=0,vmax=1,alpha=.75)
    plt.colorbar(cax)
    q = ax.quiver(xx,yy,u,v)

# This is a function to make a custome color map for the bathymetry
def custom_div_cmap(numcolors=11, name='custom_div_cmap',
                    mincol='blue', midcol='white', maxcol='red'):
    """ Create a custom diverging colormap with n colors
    Default is blue to white to red with 11 colors.  Colors can be specified
    in any way understandable by matplotlib.colors.ColorConverter.to_rgb()
    """
    cmap = mcolors.LinearSegmentedColormap.from_list(name=name, 
                                             colors =[mincol, midcol, maxcol],
                                             N=numcolors)
    return cmap


def make_bathy_map(bathy):
    fig, ax, ax_tides = make_map_colorbar()
    if bathy is not None:
        y = bathy['x'].values
        x = bathy['y'].values
        xx,yy = np.meshgrid(y,x)
        elv = bathy['z'].values
        elv_smooth = ndimage.gaussian_filter(elv, sigma=2, order=0) # This is a smoothing function for the contours
        elv[elv>.5] = np.nan # This removes replaces all data that is above 0.5 meters in elevation
        # Set the levels for the plotting bathymetery (ie detph contours)
        blevels = np.concatenate((np.arange(-150,-40,20),np.arange(-40,1,5))) 
        N = len(blevels)-1
        bnorm = matplotlib.colors.BoundaryNorm(blevels, ncolors=N, clip=False)
        cmap2 = custom_div_cmap(N, mincol='DarkBlue', midcol='CornflowerBlue' ,maxcol='w')
        cmap2.set_over('0.7') # light gray for anything above 0 meters
        # Contour colors
        pc = ax.contourf(xx,yy,elv_smooth, norm=bnorm, vmin=blevels.min(), vmax=blevels.max(), levels=blevels, cmap=cmap2, extend='both',alpha=.75)
        plt.colorbar(pc, ticks=blevels,fraction=0.026, pad=0.02)
        # Contour lines
        # ax.contour(xx,yy,elv_smooth,levels=[-1], colors='red',zorder=20) # Highlight coastline (ie 1 meter isobath)
        ax.contour(xx,yy,elv_smooth,levels=[-20], colors='black',zorder=20) # Highlight 20 meter isobath 

    # THis is all formating stuff
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True) # This maps the gridlines, which are basically straight at this scale
    # # # Turn off the lable ticks
    gl.xlines = False
    gl.ylines = False

    gl.xlabels_top = False
    gl.ylabels_right = False
    return fig, ax, ax_tides


def plot_radars(current_time, radar, ax):
    ''' '''
    current_radar = radar.sel(lat=slice(37.244221, 38.233120),lon=slice(-123.99,  -122.177032)).sel(time=current_time,method='nearest')
    current_radar['speed'] = np.sqrt(current_radar['u']**2 + current_radar['v']**2)
    u = current_radar['u']
    v = current_radar['v']
    x = current_radar['lon'].values
    y = current_radar['lat'].values
    xx,yy = np.meshgrid(x,y)
    q = ax.quiver(xx,yy,u,v,zorder=100)


def build_gif(frame_directory, fp_out):
    fp_in = os.path.join(frame_directory,"sfbay_01_frame*png")
    # https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html#gif
    img, *imgs = [Image.open(f) for f in sorted(glob.glob(fp_in))]
    img.save(fp=fp_out, format='GIF', append_images=imgs,
            save_all=True, duration=100, loop=0)


def make_barplot_data(df, ds, polygons):
    """[summary]
    """
    shapes = ['inside_bay', 'inside_bar','south_side','north_side','north_outside','south_outside']

    inside_bay = []
    inside_bar = []
    south_side = []
    north_side = []
    north_outside = []
    south_outside = []

    time = []
    total = []
    crs = {'init': 'epsg:4326'}
    for j, t in enumerate(pd.to_datetime(ds['time'].values)):
        step_df = df.query("time == @t").reset_index('time',drop=True).query("status >= 0")
        total.append(len(step_df))
        points = gpd.GeoDataFrame(step_df, geometry=gpd.points_from_xy(step_df.lon, step_df.lat),crs=crs)
        inside_bay.append(points.within(polygons.loc[0, 'geometry']).sum())
        inside_bar.append(points.within(polygons.loc[1, 'geometry']).sum())
        south_side.append(points.within(polygons.loc[2, 'geometry']).sum())
        north_side.append(points.within(polygons.loc[3, 'geometry']).sum())
        north_outside.append(points.within(polygons.loc[4, 'geometry']).sum())
        south_outside.append(points.within(polygons.loc[5, 'geometry']).sum())
        time.append(t)

    d = {
        'time':time,
        'inside_bay':inside_bay,
        'inside_bar':inside_bar,
        'south_side':south_side,
        'north_side':north_side,
        'north_outside':north_outside,
        'south_outside':south_outside,
        'num_part':total
    }

    df_locs = pd.DataFrame(d)
    for col in shapes:
        df_locs[str(col+"_norm")] = df_locs[col]/df_locs['num_part'] * 100
    
    return df_locs


def load_polygons():
    """ Create and return a geopandas dataframe with each of the polygons if interest
    polygons are generated at https://www.keene.edu/campus/maps/tool/

    Returns:
        geopandas.GeoDataFrame: dataframe with each of the polygons of AOI that are geographically references to WGS84
        
    """
    inside_bar = [(-122.4809793, 37.8260567),
        (-122.4937819, 37.8195480),
        (-122.5071711, 37.8187344),
        (-122.5167839, 37.8181920),
        (-122.5222776, 37.8168359),
        (-122.5290621, 37.8097840),
        (-122.5514566, 37.8306667),
        (-122.5655352, 37.8396146),
        (-122.5772065, 37.8287685),
        (-122.5819853, 37.8214464),
        (-122.5833591, 37.8162935),
        (-122.5837025, 37.8073427),
        (-122.5830156, 37.8008324),
        (-122.5826722, 37.7973058),
        (-122.5771770, 37.7853681),
        (-122.5737425, 37.7799413),
        (-122.5692777, 37.7774991),
        (-122.5644694, 37.7753282),
        (-122.5548529, 37.7715289),
        (-122.5373370, 37.7682722),
        (-122.5266901, 37.7680008),
        (-122.5198212, 37.7688150),
        (-122.5208515, 37.7845541),
        (-122.4954364, 37.7940502),
        (-122.4820419, 37.8030026),
        (-122.4789509, 37.8103264),
        (-122.4799812, 37.8195480),
        (-122.4809793, 37.8260567)]

    inside_bay = [(-122.4775811, 37.8271414),
        (-122.4748337, 37.8095127),
        (-122.4709512, 37.8062577),
        (-122.4682051, 37.8049014),
        (-122.4627114, 37.8042233),
        (-122.4539551, 37.8053083),
        (-122.4330135, 37.8062577),
        (-122.4103527, 37.8058508),
        (-122.4057202, 37.8001542),
        (-122.3890696, 37.7878101),
        (-122.3753342, 37.7967632),
        (-122.3658927, 37.8077496),
        (-122.3185154, 37.8226668),
        (-122.2996319, 37.8232092),
        (-122.2880077, 37.8312090),
        (-122.3096438, 37.9122423),
        (-122.3986816, 37.9332314),
        (-122.4587639, 37.9404069),
        (-122.5137137, 37.9485293),
        (-122.5047809, 37.9288987),
        (-122.4795423, 37.9034387),
        (-122.4421073, 37.8832540),
        (-122.4506911, 37.8749889),
        (-122.4628823, 37.8760729),
        (-122.4663143, 37.8702462),
        (-122.4692338, 37.8728886),
        (-122.4722379, 37.8767504),
        (-122.4683755, 37.8827121),
        (-122.4748990, 37.8880636),
        (-122.4776445, 37.8919246),
        (-122.4851983, 37.8954467),
        (-122.4890602, 37.8984269),
        (-122.4921480, 37.9024905),
        (-122.5037341, 37.9119715),
        (-122.5325844, 37.8887410),
        (-122.5174753, 37.8730919),
        (-122.5058863, 37.8664518),
        (-122.4948124, 37.8743114),
        (-122.4803286, 37.8664518),
        (-122.4762074, 37.8626572),
        (-122.4720863, 37.8515434),
        (-122.4696823, 37.8428681),
        (-122.4672782, 37.8333783),
        (-122.4703691, 37.8282261),
        (-122.4738034, 37.8271414),
        (-122.4775811, 37.8271414)]

    south_side = [(-122.5771770, 37.7853681),
        (-122.5788832, 37.7889971),
        (-122.6798632, 37.7462860),
        (-122.7763386, 37.7017506),
        (-122.7674109, 37.6865373),
        (-122.7591720, 37.6707774),
        (-122.7461247, 37.6539268),
        (-122.7330786, 37.6392474),
        (-122.7192858, 37.6218459),
        (-122.6526420, 37.5734264),
        (-122.5990521, 37.5587311),
        (-122.5277686, 37.5581868),
        (-122.5277690, 37.5761474),
        (-122.5305148, 37.5859423),
        (-122.5325771, 37.5949198),
        (-122.5270842, 37.6019923),
        (-122.5212487, 37.6074321),
        (-122.5179865, 37.6087920),
        (-122.5135231, 37.6085201),
        (-122.5102606, 37.6120557),
        (-122.5080285, 37.6183107),
        (-122.5064846, 37.6267405),
        (-122.5040820, 37.6490340),
        (-122.5054556, 37.6753970),
        (-122.5109488, 37.7006641),
        (-122.5130089, 37.7131586),
        (-122.5150687, 37.7267372),
        (-122.5174717, 37.7405848),
        (-122.5195317, 37.7549727),
        (-122.5219351, 37.7682722),
        (-122.5266901, 37.7680008),
        (-122.5373370, 37.7682722),
        (-122.5548529, 37.7715289),
        (-122.5644694, 37.7753282),
        (-122.5692777, 37.7774991),
        (-122.5737425, 37.7799413),
        (-122.5771770, 37.7853681)]
    
    north_side = [(-122.5655368, 37.8430037),
        (-122.5662231, 37.8439526),
        (-122.5881954, 37.8528989),
        (-122.6026154, 37.8607599),
        (-122.6153176, 37.8668584),
        (-122.6302520, 37.8720078),
        (-122.6378057, 37.8783763),
        (-122.6458739, 37.8842024),
        (-122.6525679, 37.8901635),
        (-122.6642417, 37.8939566),
        (-122.6760864, 37.8905699),
        (-122.6829529, 37.8829830),
        (-122.7049255, 37.8797313),
        (-122.7227783, 37.8862346),
        (-122.7406311, 37.8938212),
        (-122.7523041, 37.9079083),
        (-122.7632904, 37.9203678),
        (-122.7783966, 37.9268676),
        (-122.7893829, 37.9106171),
        (-122.7983093, 37.8867766),
        (-122.8072357, 37.8656387),
        (-122.8154755, 37.8396146),
        (-122.8223419, 37.8173783),
        (-122.8257751, 37.8000186),
        (-122.8250885, 37.7761423),
        (-122.8223419, 37.7566013),
        (-122.8182220, 37.7441142),
        (-122.8089517, 37.7270087),
        (-122.7880087, 37.7012074),
        (-122.5806423, 37.7924224),
        (-122.5826722, 37.7973058),
        (-122.5830156, 37.8008324),
        (-122.5837025, 37.8073427),
        (-122.5833591, 37.8162935),
        (-122.5819853, 37.8214464),
        (-122.5772065, 37.8287685),
        (-122.5655352, 37.8396146),
        (-122.5655368, 37.8430037)]
    
    north_outside = [(-123.2020569, 37.5129939),
        (-123.2006836, 37.5140832),
        (-122.7790816, 37.6990342),
        (-122.8022578, 37.7226639),
        (-122.8110131, 37.7390915),
        (-122.8185659, 37.7568728),
        (-122.8240582, 37.7761423),
        (-122.8216545, 37.8054439),
        (-122.8165045, 37.8287685),
        (-122.8082652, 37.8591335),
        (-122.8000254, 37.8808152),
        (-122.7900679, 37.9041159),
        (-122.7735876, 37.9279508),
        (-122.8031152, 37.9534023),
        (-122.8113537, 37.9647714),
        (-122.8312683, 37.9799275),
        (-122.8587341, 37.9988682),
        (-122.8958121, 38.0086072),
        (-122.9383850, 38.0118533),
        (-122.9425049, 37.9977860),
        (-122.9562378, 37.9718086),
        (-123.0290222, 37.9853396),
        (-123.0564880, 38.0307857),
        (-123.3956909, 37.9831748),
        (-123.2789612, 37.6713209),
        (-123.2020569, 37.5129939)]

    south_outside = [(-123.0359714, 37.2565661),
        (-122.4617484, 37.2631241),
        (-122.4330097, 37.2633973),
        (-122.4295801, 37.2838873),
        (-122.4254600, 37.3084679),
        (-122.4233996, 37.3434132),
        (-122.4261468, 37.3614255),
        (-122.4295796, 37.3777966),
        (-122.4508639, 37.4173452),
        (-122.4587604, 37.4604159),
        (-122.4738689, 37.4887527),
        (-122.4972174, 37.4827592),
        (-122.5188435, 37.4914768),
        (-122.5270851, 37.5089088),
        (-122.5305184, 37.5244308),
        (-122.5305180, 37.5410385),
        (-122.5277686, 37.5581868),
        (-122.5990521, 37.5587311),
        (-122.6527405, 37.5750590),
        (-122.6959989, 37.6077041),
        (-122.7344513, 37.6425098),
        (-122.7571114, 37.6696904),
        (-122.7735909, 37.7044670),
        (-123.2048022, 37.5151725),
        (-123.0359714, 37.2565661)]

    crs = {'init': 'epsg:4326'}
    shapes = ['inside_bay', 'inside_bar','south_side','north_side','north_outside','south_outside']
    d = {'box': 
        ['inside_bay', 'inside_bar','south_side','north_side','north_outside','south_outside'],
     'geometry': 
        [Polygon(inside_bay),Polygon(inside_bar),Polygon(south_side),Polygon(north_side),Polygon(north_outside),Polygon(south_outside)]}
    return gpd.GeoDataFrame(d, crs=crs)

def make_bar_chart(ax, df, index):   
    """Save a frame for to be patched for a gif"""
    shapes = ['inside_bay', 'inside_bar','south_side','north_side','north_outside','south_outside']
    sns.set_style("ticks")
    temp = pd.DataFrame(df.loc[index][shapes])
    sns.barplot(x=temp.index, y=index, data=temp,ax=ax)
    ax.set_ylabel("% of particles")
    ax.set_ylim(0,100)
    ax.set_xticklabels(shapes)
    # ax.set_title("TimeStep: {}".format(temp.columns[0]))
    sns.despine(ax=ax)
    plt.xticks(rotation=45)


def make_animation(model_output,base_folder='/home/pdaniel/SurfaceCurrentMaps/DailyModelRuns/model_output/'):
    hfr2 = xr.open_dataset("http://hfrnet-tds.ucsd.edu/thredds/dodsC/HFR/USWC/2km/hourly/RTV/HFRADAR_US_West_Coast_2km_Resolution_Hourly_RTV_best.ncd")
    
    ds = xr.open_dataset(os.path.join(base_folder,model_output))
    start_time = ds['time'].values[0]
    df = ds['status'].to_dataframe()
    df = df[df['status'] >= 0  ] # remove data rows were trajectory is not yet deployed
    df['start_group'] = 0

    # Make data for barplots in areas of interest
    polygons = load_polygons()
    aoi_df = make_barplot_data(df=df, ds=ds, polygons=polygons)

    # Load Tide Data
    matplotlib.rcParams['contour.negative_linestyle'] = 'solid'      
    try:
        bathy = xr.open_dataset('https://www.ngdc.noaa.gov/thredds/dodsC/crm/crm_vol7.nc')
        bathy = bathy.sel(y=slice(36.244221, 38.233120),x=slice(-123.99,  -122.177032))
    except:
        bathy=None

    # Load Tide Data
    time_df = ds['time'].to_pandas().dt.strftime('%Y%m%d')
    tides = pd.read_csv("https://api.tidesandcurrents.noaa.gov/api/prod/datagetter?begin_date={}&end_date={}&station=9414290&product=water_level&datum=MLLW&time_zone=gmt&units=metric&format=csv".format(time_df.iloc[0],time_df.iloc[-1]))
    tides['dateTime'] = pd.to_datetime(tides['Date Time'])
    tides.index = tides.dateTime
    # Preformat the pandas dataframe with the timesteps
    for i in range(1,501):
        time_diff = df.loc[(i)].index[0] - start_time
        time_diff = int(time_diff.total_seconds() / 3600)
        df.loc[(i)]['start_group'] = time_diff
    last_pos = df.query('status == 1') # Last time and postition (ie washed up?) for each particle
    norm = mcolors.Normalize(vmin=0, vmax=12)
    start_time_ts = df.index.get_level_values(1)[0]

    ## Loop through each timestep and make an image for each frame.
    #print(ds['time'].values)
    for j, t in enumerate(tqdm(pd.to_datetime(ds['time'].values))):
        print(j)
        print(t)
        current_time = t
        tail_time = current_time - timedelta(hours=3)
        bathy=None
        fig, ax, ax_tide = make_bathy_map(bathy=bathy)
        ax_inset = inset_axes(ax, width=1.5, height=1.5)
        make_bar_chart(ax=ax_inset, df=aoi_df, index=j)
        plot_radars(t, hfr2, ax)
        polygons.plot(ax=ax, alpha=.25,lw=3,edgecolor='k')
        polygons.plot(ax=ax,lw=1,edgecolor='k',facecolor='None')

        for i in range(1,501):
            current_df = df.loc[(i)]
            
            if (current_time - start_time_ts).total_seconds() <= (6 * 60):
                plot_df = current_df.query("time <= @current_time")
            else:
                plot_df = current_df.query("time <= @current_time & time >= @tail_time")

            if plot_df.size != 0:
                # Particles that are done
                if np.any(plot_df['status'] == 1):
                    ax.scatter(plot_df.iloc[-1]['lon'], plot_df.iloc[-1]['lat'],marker='x', c=np.array([cm.cm.algae(norm(plot_df['start_group'].iloc[0]))]));
                # plot the past 6 hours and the current location
                else:
                    ax.plot(plot_df['lon'], plot_df['lat'], color=cm.cm.algae((norm(plot_df['start_group'].iloc[0]))),alpha=.75);
                    im = ax.scatter(plot_df.iloc[-1]['lon'], plot_df.iloc[-1]['lat'], c=np.array([cm.cm.algae(norm(plot_df['start_group'].iloc[0]))]),zorder=210, edgecolors='k');
        out_of_model = last_pos.query('time <= @current_time')
        ax.scatter(out_of_model['lon'],out_of_model['lat'],c=np.array([cm.cm.algae(norm(out_of_model['start_group']))])[0],zorder=211, edgecolors='k',marker='x')
        plt.text(.02,.95,"time step: {}".format(j), transform=ax.transAxes)
        plt.text(.02,.925,"time: {}".format(current_time), transform=ax.transAxes)
        # cbar = plt.colorbar(im,cmap=cm.cm.algae, norm=norm)
        # cbar.set_clim(1,24)
        ax_cb, _ = mcbar.make_axes(ax, shrink=.5,orientation='horizontal',pad=0,anchor=(.1,1.75))
        cbar = mcbar.ColorbarBase(ax_cb, cmap=cm.cm.algae,norm=norm,orientation='horizontal',)

        ax_tide.plot(tides['dateTime'],tides[' Water Level'])
        current_tide = (tides.index.get_loc(current_time, method='nearest'))
        ax_tide.scatter(tides.index[current_tide],tides[' Water Level'].iloc[current_tide],c='r',marker='x',s=60,zorder=120)
        ax_tide.set_ylabel('water level [m]')
        out_name = os.path.join(base_folder,'animation-temp',f'sfbay_01_frame_{j:04}.png')
        print(out_name)
        plt.savefig(out_name)
        plt.close()
    
    fname = model_output.split("_")[3] + "_continuous.gif"
    fp_out = os.path.join(base_folder,'animations',fname)
    build_gif(os.path.join(base_folder,'animation-temp'), fp_out)
    copy_file_to_webserver(fname)


def copy_file_to_webserver(fname):
    """Copy images from model runs to webserver where they can be viewed publically."""
    try:
        ncmd=f"scp -i /etc/ssh/keys/pdaniel/scp_rsa /home/pdaniel/SurfaceCurrentMaps/DailyModelRuns/model_output/animations/{fname} skyrocket8.mbari.org:/var/www/html/data/hfr-particle-tracking-sfbay/animations/continuous_oldstyle.gif"
        os.system(ncmd)
        os.system('scp -i /etc/ssh/keys/pdaniel/scp_rsa /home/pdaniel/SurfaceCurrentMaps/DailyModelRuns/model_output/animations/{}  skyrocket8.mbari.org:/var/www/html/data/hfr-particle-tracking-sfbay/animations'.format(fname))
    except:
        pass


if __name__ ==  "__main__":
    model_output = "concave_hrf_20210316T230000_continuous.nc"
    make_animation(model_output)
    
    
