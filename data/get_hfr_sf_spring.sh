#!/bin/bash

# Add Date customization

url='https://hfrnet-tds.ucsd.edu/thredds/ncss/HFR/USWC/2km/hourly/RTV/HFRADAR_US_West_Coast_2km_Resolution_Hourly_RTV_best.ncd?var=dopx&var=dopy&var=hdop&var=u&var=v&north=38.5&west=-123.4&east=-122&south=37&horizStride=1&time_start=2023-03-01T00%3A00%3A00Z&time_end=2023-06-01T00%3A00%3A00Z&timeStride=1&accept=netcdf4' 
wget -v $url -O hfr-sfbay-2023_spring.nc
