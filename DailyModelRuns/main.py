import sf_bay_plume_update_model as model
import sf_bay_plume_update_visualizations as viz
import datetime as dt

def main():
    ###### Run plume tracker model ######
    tides=model.get_high_tides()
    recent_tides=model.check_recent_tides(tides)
    start_date=dt.datetime(recent_tides.year,recent_tides.month,recent_tides.day,recent_tides.hour,recent_tides.minute)
    o = model.particle_tracking(start_date)
    print('model ran with stard date: {}'.format(start_date))
    
    ###### Create visualizations ######
    ## Note: I am coming across permission errors when trying to save plots and animations
    ## to the server. This is likely bc this is running under pdaniel
    viz.generate_static_plot(o,start_date)
    viz.generate_animation_img_stack(o, start_date, add_current_vectors=True)
    viz.generate_animation_img_stack(o, start_date, add_current_vectors=False)
    print('visualizations created')

if __name__ == "__main__":
    main()