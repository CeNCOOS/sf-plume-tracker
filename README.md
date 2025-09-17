# SF Bay Plume Tracker #

## Introduction ##
Efforts through the Integrated Ocean Observing System (IOOS) to make surface currents off the coast 
of US in real time, has led to the creation of a robust, high resolution dataset of surface current maps. The 
longevity and resolution of surface current maps makes them an ideal source of studying and monitoring freshwater 
outflow from rivers and bays (see Tijuana River Plume tracker). Using a particle tracking model, the trajectory of 
surface water masses can be estimated by seeding a surface map with “virtual particles” which are advected by 
measured surface currents.


Measured through high frequency radar, surface current maps are spatially resolved at 500 meters, 1 km, 2km, and 
6km, depending on the configuration of instruments. Within the San Francisco Bay radars are configured at higher 
resolutions (500m and 1km). Through the golden gate and the outside of the bay, radar coverage is from 1km, 2km and 
6km maps.

## Goal/Outcomes ## 
- Demonstrate the accuracy and value in a HF-Radar and/or model forced particle tracking model of SF 
Bay outflow. 
- Develop a statistical metric for the destination of the SF Bay surface outflow.


## Model Details ##
The model is a particle tracking model that is seeded at the mouth of the Golden Gate at the beginning of each Ebb tide. Particles are advected using an RK4 schemem at 15 minute intervals. Vectors are estimated using a bilinear intepolation of the nearest surface current measurments. Particles are advected for 48 hours or untill they are removed from the model.

Additionally, a continuous seeding of particles at each hour following Ebb tide is done for 24 hours and particles are advected for 48 hours. These model runs are saved as animations.

## Running on Concave ##
A CRON job is setup to check every four hours if there is enough data to run the model. If there is, meaning there is at least 48 hours of data following a high tide, the model will run `/home/pdaniel/SurfaceCurrentMaps/DailyModelRuns/san_francisco_plume.py*`

Plots and animations are then moved to the skyrocket8 webserver via `scp`. 



### Running Jupyter Notebooks on VM ###
On the VM (concave): `jupyter notebook --no-browser --ip=0.0.0.0 --port=8080` 
Something about SF Bay outflow. SSH the local machine and forward the port 8080: `ssh -L 8080:localhost:<port> 
<remote_user>@<remote_host>`

Open `http://localhost:8080/` with a web browser
