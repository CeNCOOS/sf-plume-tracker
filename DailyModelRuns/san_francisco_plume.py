
#"end_date=today&range=48"
#"time_zone=gmt"
#"https://api.tidesandcurrents.noaa.gov/api/prod/datagetter?date=recent&station=9414290&product=water_level&datum=mllw&units=metric&time_zone=gmt&application=web_services&format=json"
import logging, requests, csv, os, time
from logging import exception
import pandas as pd
import datetime as dt
from run_particle_tracks import ModelTracker
# logging.basicConfig(filename='retreive_tide_info.log', level=logging.INFO)
import logging
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

logging.basicConfig(filename='/home/pdaniel/SurfaceCurrentMaps/DailyModelRuns/sf-bay.log', level=logging.INFO)

def get_high_tides():

    try:
        noaa_api_request = "https://api.tidesandcurrents.noaa.gov/api/prod/datagetter?date=recent&station=9414290&product=predictions&interval=hilo&datum=mllw&units=metric&time_zone=gmt&application=web_services&format=json"
        r = requests.get(noaa_api_request)
        if r.ok:
            df = pd.DataFrame(r.json()['predictions'])
            df['t'] = pd.to_datetime(df["t"])
            high_tides = df.query("type=='H'")['t']
            return high_tides
        else:
            raise Exception("Bad request response")
        
    except requests.exceptions.ConnectionError:
        logging.ERROR('Trouble connecting to NOAA Tides and Currents API')

def check_recent_tides(tides_df):
    """
        Get high tides that were at least 48 hours previous to now. THis will allow two days of model to run
    """
    elapsed_time = dt.datetime.utcnow() - tides_df
    ix = elapsed_time[elapsed_time > dt.timedelta(days=2)].index # get index where tides are over 48 hours old
    ts = pd.to_datetime(list(tides_df[ix].values))
    ts = ts + dt.timedelta(hours=1.5)
    date_str = ts.strftime('%Y-%m-%dT%H:%M')
    return date_str.tolist()

def check_log(fname):
    """Add filename from model run to the log and check it against previous runs. Also create a log 

    Args:
        fname (str): The filename for the current model run. Check it against a list in the log
    """
    df = pd.read_csv('/home/pdaniel/SurfaceCurrentMaps/DailyModelRuns/sf-bay-model-log.csv')
    return (fname in df['model_run_fname'].values)


def update_log(fname,start_date_str):
    """Add the recently run file name to the log file, 

    Args:
        fname (str): The filename for the current model run.
    """
    with open('/home/pdaniel/SurfaceCurrentMaps/DailyModelRuns/sf-bay-model-log.csv', mode='a') as csvfile:
        writer = csv.writer(csvfile, delimiter=',', quotechar="'")
        writer.writerow([fname, start_date_str, dt.datetime.now()])
    
# Changed run time to 24 hours
def run_model(continuous=False):
    tides = get_high_tides()
    recent_tides = check_recent_tides(tides)
    #print(tides)
    for i, start_date_str in enumerate(recent_tides):
        #print(start_date_str)
        if continuous:
            config = {'bbox' : [-123.99,  -122.177032, 37.244221, 38.233120],
                'seed_coords' : (-122.531,37.8016),
                'seed_number': 50,
                'seed_radius': 500,
                'start_date' : start_date_str,
                'duration' : 24,
                'resolution': ['2km','6km'],
                'continuous': True
                }
            model_cont = ModelTracker(config)

        config = {'bbox' : [-123.99,  -122.177032, 37.244221, 38.233120],
                'seed_coords' : (-122.531,37.8016), #(-122.531,37.8016), (-122.55183,37.8016)
                'seed_number': 100,
                'seed_radius': 500,
                'start_date' : start_date_str,
                'duration' : 24,
                'resolution': ['2km','6km']
                }
        model = ModelTracker(config)
        if check_log(model.fname) == False:
            logging.info('RUNNING: model on: {}'.format(model.start_date))
            model.run(save_trajectories=False, plot=False, animation=False)
            if continuous:
                model_cont.run(animation=False)

            update_log(model.fname, model.start_date)
        
        else:
            logging.info("SKIPPING: model already ran for: {}".format(model.start_date))
        
        

def copy_file_to_webserver(fname,continuous=False):
    """Copy images from model runs to webserver where they can be viewed publically."""
    try:
        if continuous:
            os.system('scp -i /etc/ssh/keys/pdaniel/scp_rsa /home/pdaniel/SurfaceCurrentMaps/DailyModelRuns/model_output/plots/{}  skyrocket8.mbari.org:/var/www/html/data/hfr-particle-tracking-sfbay/ '.format(fname + 'continuous.gif'))
            logging.info('MOVING: {} to skyrocket8'.format(fname))
        else:
            os.system('scp -i /etc/ssh/keys/pdaniel/scp_rsa /home/pdaniel/SurfaceCurrentMaps/DailyModelRuns/model_output/plots/{}  skyrocket8.mbari.org:/var/www/html/data/hfr-particle-tracking-sfbay/ '.format(fname + '.png'))
            logging.info('MOVING: {} to skyrocket8'.format(fname))
    except:
        logging.debug('Unabled to move {} to skyrocket8'.format(fname))


if __name__ == "__main__":
    run_model(continuous=True)
    #run_model(continuous=False)
    # logging.info('RUNNING on {}'.format(dt.datetime.now()))
    # try:
    #     run_model(continuous=True)
        
    # except Exception as ex:
    #     template = "An exception of type {0} occurred. Arguments:\n{1!r}"
    #     message = template.format(type(ex).__name__, ex.args)
    #     logging.debug(message)
    #     # time.sleep(60*15)
    #     run_model()
