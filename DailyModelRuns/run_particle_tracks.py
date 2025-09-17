from datetime import datetime, timedelta
import os
from opendrift.readers import reader_netCDF_CF_generic
from opendrift.readers import reader_global_landmask
from animate_tracks import make_animation
from static_plots import make_static_plot
from opendrift.models.oceandrift import OceanDrift
import logging
logging.basicConfig(filename='particle_tracking.log', level=logging.DEBUG)

class ModelTracker():
    """
    Wrapper for configuring and running open drift particle tracking models
    Should be used to configure the spatial extent, temporal extent, and how to
    output the data
    """

    def __init__(self, configuration_file):
        self.bbox = None
        self.dataset_urls = None
        self.output_folder = None
        self.start_date = None
        self.seed_coords = None
        self.seed_number = None
        self.seed_radius = None
        self.continuous = False
        self.read_configuration(configuration_file)
        self.o = OceanDrift(loglevel=50)
        self.model_setup()
        self.base_folder = "/home/pdaniel/SurfaceCurrentMaps/DailyModelRuns/model_output/"
        self.fname = "concave_hrf_" + self.start_date.strftime("%Y%m%dT%H%M%S")

    def read_configuration(self, config):
        for key in config:
            if key  ==  'bbox':
                self.bbox = config[key]

            elif key == 'start_date':
                self.start_date = datetime.fromisoformat(config[key])

            elif key == 'duration':
                self.duration = config[key]
        
            elif key == 'seed_coords':
                self.seed_coords = config[key]
            
            elif key == 'seed_number':
                self.seed_number = config[key]
            
            elif key == 'seed_radius':
                self.seed_radius = config[key]
            
            elif key == "continuous":
                self.continuous = config[key]
            
        self.dataset_urls = ["https://dods.ndbc.noaa.gov/thredds/dodsC/hfradar_uswc_{}".format(res,res) for res in config['resolution']]
        #self.dataset_urls = ["http://hfrnet-tds.ucsd.edu/thredds/dodsC/HFR/USWC/{}/hourly/RTV/HFRADAR_US_West_Coast_{}_Resolution_Hourly_RTV_best.ncd".format(res,res) for res in config['resolution']]

    def model_setup(self):
        # reader_landmask = reader_global_landmask.Reader(extent=self.bbox)
        readers = [reader_netCDF_CF_generic.Reader(url) for url in self.dataset_urls]
        # readers.append(reader_landmask)
        self.o.add_reader(readers)
                ### Boiler Plate
        # self.o.set_config('general:coastline_action', 'stranding') 
        self.o.set_config('general:coastline_action', 'previous') 
        #self.o.set_config('drift:advection_scheme', 'runge-kutta4')
        self.o.set_config('drift:scheme','runge-kutta4')
        self.o.set_config('general:time_step_minutes', 15)
        self.o.set_config('drift:stokes_drift', False)
        self.o.set_config('drift:current_uncertainty_uniform', .5)

        if self.continuous:
            time_step = timedelta(minutes=60)
            num_steps = 14
            for i in range(num_steps+1):
                self.o.seed_elements(
                    lon=self.seed_coords[0], 
                    lat=self.seed_coords[1], 
                    number=self.seed_number, 
                    radius=self.seed_radius,
                    time=self.start_date + i*time_step
                )

        else:
            self.o.seed_elements(
                    lon=self.seed_coords[0], 
                    lat=self.seed_coords[1], 
                    number=self.seed_number, 
                    radius=self.seed_radius,
                    time=self.start_date
                    )
        


    def run(self, save_trajectories=False, plot=False, animation=False):
        if self.continuous:
            self.o.run(
                duration=timedelta(hours=self.duration), 
                outfile=os.path.join(self.base_folder,self.fname+"_continuous.nc")
                )
            make_animation(os.path.join(self.base_folder,self.fname+"_continuous.nc"))
            make_static_plot(os.path.join(self.base_folder,self.fname+"_continuous.nc"))
        elif save_trajectories:
            self.o.run(
                duration=timedelta(hours=self.duration), 
                outfile=os.path.join(self.base_folder,self.fname+".nc")
                )
        else:
            self.o.run(
                duration=timedelta(hours=self.duration)
                )
        
        if plot:
            # logging.INFO('Saving plot to: {}'.format(os.path.join(self.base_folder,'plots',self.fname+'.png')))
            self.o.plot(filename=os.path.join(self.base_folder,'plots',self.fname+'.png'))

        if animation:
            # logging.INFO('Saving animation to: {}'.format(os.path.join(self.base_folder,'animations',self.fname+'.gif')))
            self.o.animation(
                background=['x_sea_water_velocity', 'y_sea_water_velocity'], 
                legend_loc='upper center', 
                fast=False,  
                show_trajectories=True, 
                density=False,
                filename=os.path.join(self.base_folder,self.fname+'continuous.gif')
            )

if __name__ == "__main__":
    config = {'bbox' : [-123.99,  -122.177032, 37.244221, 38.233120],
            'seed_coords' : (-122.55183,37.8016),
            'seed_number': 50,
            'seed_radius': 500,
            'start_date' : "2020-09-11T13:00",
            'duration' : 48,
            'resolution': ['2km','6km'],
            'continuous': True
            }

    model = ModelTracker(config)
    print(model.fname)
    model.run(animation=False)
