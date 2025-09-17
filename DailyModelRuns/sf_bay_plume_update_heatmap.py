import matplotlib.pyplot as plt
import os
import xarray as xr
import numpy as np
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point, Polygon
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from shapely.geometry import box
import cmocean
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter


def create_grid(ds):

    # 1. Create a grid of 1km x 1km cells over the area of interest

    # Flatten all points across time and space into x/y arrays
    x = ds['lon'].values.ravel()
    y = ds['lat'].values.ravel()

    # Get fill values
    fill_lon = ds['lon']._FillValue if '_FillValue' in ds['lon'].attrs else 9.969209968386869e+36
    fill_lat = ds['lat']._FillValue if '_FillValue' in ds['lat'].attrs else 9.969209968386869e+36

    # Mask invalid values using x and y, not lon/lat
    mask = (
        np.isfinite(x) &
        np.isfinite(y) &
        (x != fill_lon) &
        (y != fill_lat)
    )
    x = x[mask]
    y = y[mask]
    #print(f"Kept {x.size} points after filtering")

    # Convert to points GeoSeries
    points = gpd.GeoSeries(gpd.points_from_xy(x, y), crs="EPSG:4326").to_crs("EPSG:32610")  # or your UTM zone
    
    # Compute bounding box in projected coords
    xmin, ymin, xmax, ymax = points.total_bounds
    cell_size = 1000  # meters
    grid_cells = []

    for x0 in np.arange(xmin, xmax, cell_size):
        for y0 in np.arange(ymin, ymax, cell_size):
            x1 = x0 + cell_size
            y1 = y0 + cell_size
            grid_cells.append(Polygon([(x0, y0), (x1, y0), (x1, y1), (x0, y1)]))

    grid = gpd.GeoDataFrame({'geometry': grid_cells}, crs=points.crs)

    # 2. Count the number of points in each grid cell and compute percentage

    join = gpd.sjoin(gpd.GeoDataFrame(geometry=points), grid)#, predicate='within')

    # Count how many points per cell over the entire time domain 
    counts = join['index_right'].value_counts()

    # Add counts to grid and compute percentage
    grid['count'] = grid.index.map(counts).fillna(0)
    grid['pct'] = (grid['count'] / grid['count'].sum()) * 100

    return grid, points



def heat_map(grid,start_date):
    # Import bathymetry layer for contours 
    ds_o = xr.open_dataset('/home/pdaniel/SurfaceCurrentMaps/DailyModelRuns/data/sf_bay_topo.nc')
    elv = ds_o['band_data'].values
    xx = ds_o.x.values
    yy = ds_o.y.values

    # Reproject grid back to WGS84 for plotting
    grid_deg = grid.to_crs("EPSG:4326")

    # Create axes with a Cartopy CRS
    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw={'projection': ccrs.PlateCarree()}, dpi=200)

    # Mask zero values for better color scaling
    grid_deg['pct_masked'] = grid_deg['pct'].mask(grid_deg['pct'] == 0)

    # Create plot
    grid_deg.plot(
        column='pct_masked',
        ax=ax,
        legend=True,
        cmap='magma_r',
        legend_kwds={'label': "%", 'shrink': 0.5},
        edgecolor='white',
        zorder=0
    )

    # Add coastline
    coastline = cfeature.NaturalEarthFeature(
        'physical', 'coastline', '10m',
        edgecolor='black', facecolor='slategrey', linewidth=0.8
    )
    ax.add_feature(coastline, zorder=2)

    # Set extent to focus on the area of interest
    ax.set_xlim(-123.05, -122.35)
    ax.set_ylim(37.5, 38.1)
    # Add tick marks for lon/lat
    xticks = [-123.0, -122.7, -122.4]
    yticks = [37.5, 37.8, 38.1]
    ax.set_xticks(xticks, crs=ccrs.PlateCarree())
    ax.set_yticks(yticks, crs=ccrs.PlateCarree())
    ax.xaxis.set_major_formatter(LongitudeFormatter())
    ax.yaxis.set_major_formatter(LatitudeFormatter())

    # Add bathymetry contours for context
    levels = [0,10,20,50,100,200,500,1000]
    ax.contourf(xx,yy,-1*elv[0],zorder=-2,cmap=cmocean.cm.ice_r,levels=levels)
    cont = ax.contour(xx,yy,-1*elv[0],levels=levels[1:],colors='k',linewidths=0.4,linestyles='solid')
    #ax.clabel(cont, inline=True, fontsize=10, fmt='%1.0f') # Note I removed contour labels

    ax.text(1,1.05,'48h Model Run Start Time:',fontweight='bold',fontsize=14,transform=ax.transAxes,ha='right',va='bottom')
    ax.text(1,1,start_date,fontweight='bold',fontsize=14,transform=ax.transAxes,ha='right',va='bottom')

    plt.title(
    '% of Trajectories Landed Within Grid Cells During Model Run',
    fontsize=14,
    pad=60 
    ) 

    plt.savefig('/home/pdaniel/SurfaceCurrentMaps/DailyModelRuns/model_output/heatmaps/heatmap_{}.png'.format(start_date.replace(' ','_').replace(':','-')) , bbox_inches='tight')