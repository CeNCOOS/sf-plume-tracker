import xarray as xr
import datetime as dt
import os
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pandas as pd
import seaborn as sns

def make_map():
    fig = plt.figure(figsize=(16,8))
    cart_proj = ccrs.PlateCarree()
    ax = plt.axes(projection=cart_proj)
    ax.coastlines('10m', linewidth=0.8,zorder=200)
    ax.add_feature(cfeature.LAND)
    ax.add_feature(cfeature.OCEAN)
    ax.set_xlim(-123,-122.26)
    ax.set_ylim(37.5,38.1)
    return(fig, ax)

def make_static_plot(fname):
    mask_gg = [(37.78008, 37.83338),(-122.50854, -122.47301)]
    SAVE_NAME = os.path.basename(fname).split("_")[2] + "_continuous.png"
    SAVE_DIR = "/home/pdaniel/SurfaceCurrentMaps/DailyModelRuns/model_output/static/"
    start_time_dt = pd.to_datetime(os.path.basename(fname).split("_")[2])

    ds = xr.open_dataset(fname)
    df = ds['status'].to_dataframe()
    df['start_time'] = 0
    experimentStart = df.iloc[0].name[1]
    df_flat = df.reset_index()
    df_flat = df_flat.query("status >= 0")
    first_mapping = df_flat.groupby(by=['trajectory'])['time'].first()
    first_hours = first_mapping - experimentStart
    first_hours = first_hours.apply(lambda x: x.seconds/3600)
    df_flat['start_hour'] = df_flat['trajectory'].map(first_hours)
    
    fig, ax = make_map()
    sns.set_theme(style="darkgrid")
    sns.set_context('talk')
    normalize = mcolors.Normalize(vmin=0, vmax=2)
    for i in range(1, df_flat.trajectory.max()+1):
        new_df = df_flat.query("trajectory == @i")
        if i == 1:
            sLat = [new_df.iloc[0].lat]
            sLon = [new_df.iloc[0].lon]
            ax.scatter(sLon, sLat, zorder=220, c='r', s=70, label='Seed Start')
        if (i % 2) == 0:

            eLat = [new_df.iloc[-1].lat][0]
            eLon = [new_df.iloc[-1].lon][0]
            if not ((eLat > mask_gg[0][0]) & (eLat < mask_gg[0][1]) & (eLon > mask_gg[1][0]) & (eLon < mask_gg[1][1])):
                
                start_hours = new_df.iloc[0].start_hour 
                if start_hours > 3:
                    break
                cax = ax.scatter(eLon, eLat,
                                 zorder=200,
                                 c=start_hours,
                                 norm=normalize,
                                 marker='X',
                                 edgecolor='.25',
                                 cmap='binary',
                                 s=150)
                ax.plot(new_df['lon'],new_df['lat'],color='.5',lw=.5)
    plt.colorbar(cax,label='Released Time [Hours]')
    plt.text(0,-0.03,"Release Time: "+str(start_time_dt),
        horizontalalignment='left',
        verticalalignment='center', 
        transform=ax.transAxes)
    plt.text(0,-0.071,"End Time: "+str(start_time_dt + dt.timedelta(hours=48)),
        horizontalalignment='left',
        verticalalignment='center', 
        transform=ax.transAxes)
    plt.text(0,1.02,"SF Outflow Tracker, 48 Hour Trajectories",
        horizontalalignment='left',
        verticalalignment='center', 
        fontweight='bold',
        transform=ax.transAxes)

    plt.text(0,-.115,"Particles are released every 15 minutes for the first two hours and forced for 48 hours using 2km surface current maps measured by HFR.",
        horizontalalignment='left',
        verticalalignment='center', 
        fontsize=10,
        transform=ax.transAxes)


    
    colors = ['red']
    labels = ['Start']
    plt.legend(
        bbox_to_anchor=(0.87,-0.08,.1,.1),
        markerfirst=False,
        frameon=False,
        fancybox=False
              )
    plt.savefig(fname=os.path.join(SAVE_DIR,SAVE_NAME), dpi=200, bbox_inches='tight', pad_inches=0.15)
    plt.close()
    # Copy file to webserver
    copy_file_to_webserver(SAVE_NAME)
    

def copy_file_to_webserver(fname):
    """Copy images from model runs to webserver where they can be viewed publically."""
    try:
        os.system('scp -i /etc/ssh/keys/pdaniel/scp_rsa /home/pdaniel/SurfaceCurrentMaps/DailyModelRuns/model_output/static/{}  skyrocket8.mbari.org:/var/www/html/data/hfr-particle-tracking-sfbay/static'.format(fname))
    except:
        pass