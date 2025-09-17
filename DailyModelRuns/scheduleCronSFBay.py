from crontab import CronTab


my_cron = CronTab(user='pdaniel')

# job = my_cron.new(command='/home/pdaniel/anaconda3/bin/python /home/pdaniel/SurfaceCurrentMaps/DailyModelRuns/san_francisco_plume.py')
# job.hour.every(6)

for job in my_cron:
    print(job)

    