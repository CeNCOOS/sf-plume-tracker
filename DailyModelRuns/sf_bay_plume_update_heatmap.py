import matplotlib.pyplot as plt
import os
import xarray as xr
import numpy as np
import pandas as pd
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cmocean
from matplotlib.animation import FuncAnimation
from matplotlib.colors import ListedColormap

def get_latest_ds():
    output_dir = "../DailyModelRuns/model_output/netcdf_model_v2/"
    output_files = os.listdir(output_dir)
    output_files = [f for f in output_files if f.endswith('.nc')]
    ds = xr.open_dataset(os.path.join(output_dir, output_files[-1]))
    return ds


def compute_heatmap(ds):
    # Shapes: (trajectory, time)
    lon = ds['lon'].values.copy()
    lat = ds['lat'].values.copy()
    status = ds['status'].values.copy()

    n_particles, n_times = lon.shape

    # Define grid edges (lon/lat)
    lon_edges = np.linspace(-123.05, -122.35, 100)
    lat_edges = np.linspace(37.5, 38.1, 100)

    # Prepare 3D array to hold counts: (time, lat_bins, lon_bins)
    counts = np.zeros((n_times, len(lat_edges)-1, len(lon_edges)-1))

    # Compute first stranded index for each particle
    first_strand_idx = np.argmax(status==1, axis=1)  # per particle
    never_stranded = np.all(status==0, axis=1)
    first_strand_idx[never_stranded] = -1  # mark particles that never stranded

    for t_idx in range(n_times):
        # Extract positions at this time
        lon_t = lon[:, t_idx].copy()
        lat_t = lat[:, t_idx].copy()
        
        # Carry-forward stranded positions
        stranded_mask = (first_strand_idx >= 0) & (first_strand_idx <= t_idx)
        traj_idx = np.where(stranded_mask)[0]
        lon_t[traj_idx] = lon[traj_idx, first_strand_idx[traj_idx]]
        lat_t[traj_idx] = lat[traj_idx, first_strand_idx[traj_idx]]

        # Count number of stranded particles in this frame
        num_stranded_frame = np.sum((first_strand_idx >= 0) & (first_strand_idx <= t_idx))
        
        # Remove NaNs
        valid_mask = np.isfinite(lon_t) & np.isfinite(lat_t)
        lon_t = lon_t[valid_mask]
        lat_t = lat_t[valid_mask]
        
        # 2D histogram
        H, _, _ = np.histogram2d(lat_t, lon_t, bins=[lat_edges, lon_edges])
        counts[t_idx, :, :] = H

    # Wrap as xarray DataArray
    heat_da = xr.DataArray(
        counts,
        dims=('time', 'lat_bin', 'lon_bin'),
        coords={
            'time': ds['time'],
            'lat_bin': (lat_edges[:-1] + lat_edges[1:]) / 2,
            'lon_bin': (lon_edges[:-1] + lon_edges[1:]) / 2
        }
    )

    # Mask zeros
    heat_da_masked = heat_da.where(heat_da != 0)

    return heat_da_masked, counts, lat_edges, lon_edges, first_strand_idx

def plot_heatmap_aggregated(ds, lon_edges, lat_edges):
    """
    Aggregate particle counts over the entire time domain of the model run, and plot % of trajectories per grid cell.

    Parameters:
    - ds: xarray.Dataset with dimensions ('trajectory', 'time') and variables 'lon' and 'lat'
    - lon_edges, lat_edges: 1D arrays defining the grid edges
    """
    # Mask invalid values
    lon = ds['lon'].where(np.isfinite(ds['lon']))
    lat = ds['lat'].where(np.isfinite(ds['lat']))

    n_time = ds.sizes['time']
    n_particles = ds.sizes['trajectory']

    # Prepare array to hold counts per cell
    counts = np.zeros((len(lat_edges)-1, len(lon_edges)-1))

    # Loop over time to accumulate counts
    for t_idx in range(n_time):
        lon_t = lon.isel(time=t_idx).values
        lat_t = lat.isel(time=t_idx).values
        mask = np.isfinite(lon_t) & np.isfinite(lat_t)
        lon_t, lat_t = lon_t[mask], lat_t[mask]

        # 2D histogram
        H, _, _ = np.histogram2d(lat_t, lon_t, bins=[lat_edges, lon_edges])
        counts += H  # sum over time

    # Convert to % of trajectories (normalize by total particle-time points)
    total_points = n_particles * n_time
    pct = (counts / total_points) * 100

    # Wrap as DataArray
    heat_da = xr.DataArray(
        pct,
        dims=('lat_bin', 'lon_bin'),
        coords={
            'lat_bin': (lat_edges[:-1] + lat_edges[1:])/2,
            'lon_bin': (lon_edges[:-1] + lon_edges[1:])/2
        }
    )

    # Mask zeros
    heat_da_masked = heat_da.where(heat_da != 0)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw={'projection': ccrs.PlateCarree()})
    mesh = heat_da_masked.plot(ax=ax, x='lon_bin', y='lat_bin', cmap='magma_r', add_colorbar=False)
    cbar = fig.colorbar(mesh, ax=ax, label='% of Trajectories in Grid Cell', shrink=0.7) 

    # Add coastline
    coastline = cfeature.NaturalEarthFeature(
        'physical', 'coastline', '10m',
        edgecolor='black', facecolor='slategrey', linewidth=0.8
    )
    ax.add_feature(coastline, zorder=1)

    # Add bathymetry contours
    ds_o = xr.open_dataset('/home/pdaniel/SurfaceCurrentMaps/DailyModelRuns/data/sf_bay_topo.nc')
    elv = ds_o['band_data'].values
    xx = ds_o.x.values
    yy = ds_o.y.values
    levels = [0,5,10,20,50,100,200,500,1000]
    cmap = cmocean.cm.ice_r
    cmap_trunc = ListedColormap(cmap(np.linspace(0.1,1,256)))
    ax.contourf(xx, yy, -1*elv[0], zorder=-1, cmap=cmap_trunc, levels=levels)
    ax.contour(xx, yy, -1*elv[0], levels=levels[1:], colors='k', linewidths=0.5, linestyles='solid')

    # Set extent
    ax.set_xlim(-123.05, -122.35)
    ax.set_ylim(37.5, 38.1)

    # add title with start date
    start_date = ds.isel(time=0).time.values.astype(str).split('.')[0]
    ax.set_title(f'48H Aggregated Heatmap of Particle Trajectories\n(Start Date: {start_date})', fontsize=14)

    plt.savefig('/home/pdaniel/SurfaceCurrentMaps/DailyModelRuns/model_output/heatmaps/heatmap_{}.png'.format(start_date.replace(' ','_').replace(':','-')) , bbox_inches='tight')

    plt.show()
    return heat_da_masked



def animate_heatmap(heat_da_masked, ds, first_strand_idx):
    """
    Create an animation of the heatmap over time.

    Parameters:
    - heat_da_masked: xarray.DataArray with dimensions ('time', 'lat_bin', 'lon_bin')
    - ds: xarray.Dataset with dimensions ('trajectory', 'time') and variables 'lon', 'lat', 'status'
    - first_strand_idx: 1D array indicating the first time index each particle stranded
    """
    # Create figure and axis with Cartopy projection

    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw={'projection': ccrs.PlateCarree()})

    # Add bathymetry contours
    ds_o = xr.open_dataset('/home/pdaniel/SurfaceCurrentMaps/DailyModelRuns/data/sf_bay_topo.nc')
    elv = ds_o['band_data'].values
    xx = ds_o.x.values
    yy = ds_o.y.values
    levels = [0,5,10,20,50,100,200,500,1000]
    cmap = cmocean.cm.ice_r
    cmap_trunc = ListedColormap(cmap(np.linspace(0.1,1,256)))
    ax.contourf(xx, yy, -1*elv[0], zorder=-1, cmap=cmap_trunc, levels=levels)
    ax.contour(xx, yy, -1*elv[0], levels=levels[1:], colors='k', linewidths=0.5, linestyles='solid')

    # Add coastline
    coastline = cfeature.NaturalEarthFeature(
        'physical', 'coastline', '10m',
        edgecolor='black', facecolor='slategrey', linewidth=0.8
    )
    ax.add_feature(coastline, zorder=1)

    # Set extent
    ax.set_xlim(-123.05, -122.35)
    ax.set_ylim(37.5, 38.1)

    # Create initial QuadMesh
    mesh = ax.pcolormesh(
        heat_da_masked.lon_bin,
        heat_da_masked.lat_bin,
        heat_da_masked.isel(time=0),
        cmap='magma_r',
        zorder=2
    )
    cbar = fig.colorbar(mesh, ax=ax, label='% of Trajectories in Grid Cell',shrink=0.7)
    n_particles = ds.sizes['trajectory']
    # Update function
    def update(frame):
        mesh.set_array(heat_da_masked.isel(time=frame).values.ravel())
        time_str = pd.to_datetime(heat_da_masked.time.values[frame]).strftime('%Y-%m-%d %H:%M')
        ax.set_title(f"Time: {time_str}")

        stranded_mask = (first_strand_idx >= 0) & (first_strand_idx <= frame)

        num_stranded_frame = (stranded_mask.sum() / n_particles * 100).round(2)
        # Remove previous text (so it doesn’t overplot)
        for txt in ax.texts:
            txt.remove()

        # Add new text
        ax.text(
            0.02, 0.95,  # location in axis coordinates
            f"Particles that have reached the coastline: {num_stranded_frame} %",
            transform=ax.transAxes,
            fontsize=14,
            color='black'
        )

        return mesh,

    # Create animation
    ani = FuncAnimation(fig, update, frames=heat_da_masked.sizes['time'], blit=True, interval=200)

    # Save as GIF
    ani.save("../DailyModelRuns/model_output/heatmaps/trajectories_animation.gif", writer="pillow", fps=5)

    plt.show()
