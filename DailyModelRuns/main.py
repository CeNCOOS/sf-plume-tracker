import sf_bay_plume_update_model as model
import sf_bay_plume_update_visualizations as viz
import sf_bay_plume_update_heatmap as heatmap
import datetime as dt
import os
import xarray as xr

def main():
    ###### Run plume tracker model ######
    #tides=model.get_high_tides()
    #recent_tides=model.check_recent_tides(tides)
    #start_date=dt.datetime(recent_tides.year,recent_tides.month,recent_tides.day,recent_tides.hour,recent_tides.minute)
    #o = model.particle_tracking(start_date)
    #print('model ran with stard date: {}'.format(start_date))
    
    ###### Create visualizations ######
    ## Note: since this code is currently running under the main branch, I am getting permission
    ## errors because I cannot overwrite. This issue shouldn't persist once we merge branches.
    #viz.generate_static_plot(o,start_date)
    #viz.generate_animation_img_stack(o, start_date, add_current_vectors=True)
    #viz.generate_animation_img_stack(o, start_date, add_current_vectors=False)
    #print('visualizations created')

    # Make heat map
    # # Define the directory where model output files are stored
    output_dir = "../DailyModelRuns/model_output"
    # List all files in the output directory
    output_files = os.listdir(output_dir)
    # Print the list of files with .nc extension
    output_files = [f for f in output_files if f.endswith('.nc')]
    ds = xr.open_dataset(os.path.join(output_dir, output_files[-1]))
    start_date = ds.isel(time=0).time.values.astype(str).split('.')[0] 
    grid,points = heatmap.create_grid(ds)
    heatmap.heat_map(grid, start_date)

if __name__ == "__main__":
    main()