import sf_bay_plume_update_modelrun as model
import sf_bay_plume_update_visualizations as viz
import sf_bay_plume_update_heatmap as heatmap
import datetime as dt
 
def main():
    ###### Run plume tracker model ######
    #tides = model.get_high_tides()
    #recent_tides = model.check_recent_tides(tides)
    #start_date = dt.datetime(recent_tides.year,recent_tides.month,recent_tides.day,recent_tides.hour,recent_tides.minute)
    #o = model.particle_tracking(start_date)
    #print('model ran with start date: {}'.format(start_date))
    
    ###### Create static and animated visualizations of projections ######
    #viz.generate_static_plot(o,start_date) 
    #viz.generate_animation_img_stack(o, start_date, add_current_vectors=True)
    #viz.generate_animation_img_stack(o, start_date, add_current_vectors=False)
    #print('visualizations created')

    ###### Create raster heat map ######
    ds = heatmap.get_latest_ds()
    heat_da_masked, counts, lat_edges, lon_edges, first_strand_idx = heatmap.compute_heatmap(ds)
    heatmap.plot_heatmap_aggregated(ds, lon_edges, lat_edges)
    #start_time = ds.isel(time=0).time.values.astype(str).split('.')[0] 
    #end_time = ds.isel(time=-1).time.values.astype(str).split('.')[0]
    heatmap.animate_heatmap_with_tides(heat_da_masked, ds, first_strand_idx)

if __name__ == "__main__":
    main()