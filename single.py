#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 16:19:33 2024

@author: gruma-r
"""

import netCDF4 as nc
import geopandas as gpd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from cartopy.feature import ShapelyFeature
import os
import numpy as np
import xarray as xr
from datetime import datetime

data_single_directory = '/home/gruma-r/Área de Trabalho/nc_era5_data/single_level'
data_pressure_directory = '/home/gruma-r/Área de Trabalho/nc_era5_data/vort_geop'

# Load Brazil shapefile
shapefile_path = '/home/gruma-r/Área de Trabalho/Lim_america_do_sul_2021.shp'
brazil_states = gpd.read_file(shapefile_path)
brazil_feature = ShapelyFeature(brazil_states['geometry'], ccrs.PlateCarree(), edgecolor='silver', facecolor='none', zorder = 2)

# Create output folder
output_folder = '/home/gruma-r/Área de Trabalho/nc_era5_data/' # 1000-500 (blue)

os.makedirs(output_folder, exist_ok=True)

pressure_level_1 = 500   # Define the first pressure level in hPa
pressure_level_2 = 1000  # Define the second pressure level in hPa
#pressure_level_3 = 850   # Define the third pressure level in hPa

start_date_range = datetime(2024, 1, 15)
end_date_range = datetime(2024, 1, 18)

# Ensure files are sorted and paired correctly
netcdf_files_single = sorted([f for f in os.listdir(data_single_directory) if f.endswith('.nc')])
netcdf_files_pressure = sorted([f for f in os.listdir(data_pressure_directory) if f.endswith('.nc')])

for file_name_single, file_name_pressure in zip(netcdf_files_single, netcdf_files_pressure):
    file_path_single = os.path.join(data_single_directory, file_name_single)
    file_path_pressure = os.path.join(data_pressure_directory, file_name_pressure)

    print(f"Opening single level file: {file_path_single}")
    print(f"Opening pressure level file: {file_path_pressure}")

    # try:
    with nc.Dataset(file_path_single, 'r') as single_nc, nc.Dataset(file_path_pressure, 'r') as pressure_nc:
         xarray_data1 = xr.open_dataset(file_path_single)
         xarray_data2 = xr.open_dataset(file_path_pressure)

         latitude = single_nc.variables['latitude'][:]
         longitude = single_nc.variables['longitude'][:]
         time_variable = single_nc.variables['time']
         mslp_variable = single_nc.variables['msl']
         pressure_levels = xarray_data2['level'][:]  # Pressure levels in hPa
         geopotential = xarray_data2['z']  # Geopotential in m^2/s^2

         # Find indices corresponding to the provided pressure levels
         idx_1 = np.where(pressure_levels == pressure_level_1)[0][0] # 500
         idx_2 = np.where(pressure_levels == pressure_level_2)[0][0] # 1000
         #idx_3 = np.where(pressure_levels == pressure_level_3)[0][0] # 850

         # Calculate layer thickness in decameters (dam)
         layer_thickness1 = (geopotential[:, idx_1, :, :] - geopotential[:, idx_2, :, :]) / 100 # 1000-500 (blue)
         # layer_thickness2 = (geopotential[:, idx_3, :, :] - geopotential[:, idx_2, :, :]) / 100 # 1000-850 (red)

         # Convert time units to datetime
         time_values = time_variable[:]
         time_units = time_variable.units
         calendar = 'gregorian'
         custom_time = nc.num2date(time_values, units=time_units, calendar=calendar)

         start_index = np.argmin(np.abs(custom_time - start_date_range))
         end_index = np.argmin(np.abs(custom_time - end_date_range))

         # Loop through each time step
         for time_step in range(start_index, end_index + 1):
             # Get the current datetime
             current_time = custom_time[time_step]

             # Select mean sea level pressure for the current time step
             mslp = mslp_variable[time_step, :, :]
             
             # Select layer thickness for the current time step
             thickness = layer_thickness1[time_step, :, :] # 1000-500
             # thickness = layer_thickness2[time_step, :, :] # 1000-850

             # Convert pressure from Pa to hPa
             mslp_hpa = mslp / 100

             # Plot mean sea level pressure and thickness
             plt.figure(figsize=(10, 8))
             ax = plt.axes(projection=ccrs.PlateCarree())
             ax.add_feature(brazil_feature)
             ax.coastlines(linewidth=0.5, color='silver', zorder=2)

             
             # Plot MSLP contours
             mslp_levels = np.arange(900, 1100, 4)  # Adjust according to your preference
             cs_mslp = ax.contour(longitude[:], latitude[:], mslp_hpa[:, :], levels=mslp_levels, colors='black', linewidths=1, transform=ccrs.PlateCarree(), zorder=4)
             ax.clabel(cs_mslp, inline=True, fontsize=8, fmt='%d')
             
             
             #Thickness in filled contours
             thickness_levels = np.arange(480, 581, 4)  # Adjust according to your preference  -> 1000-500
             # thickness_levels = np.arange(80, 180, 3)  # Adjust according to your preference -> 1000-850
             thickness = np.minimum(580, np.maximum(thickness, 480))

             #############################################################################################################################################
             #1000-500   
             cf_thickness = ax.contourf(longitude[:], latitude[:], thickness[:, :], levels=thickness_levels, cmap='jet', transform=ccrs.PlateCarree(), zorder=1)
             #1000-850
             # cf_thickness = ax.contourf(longitude[:], latitude[:], thickness[:, :], levels=thickness_levels, cmap='jet_r', transform=ccrs.PlateCarree(), zorder=1)
             #############################################################################################################################################
             cs_thickness = ax.contour(longitude[:], latitude[:], thickness[:, :], levels=thickness_levels, colors='dimgrey', linewidths=1, linestyles='dashed', transform=ccrs.PlateCarree(), zorder=3)
             ax.clabel(cs_thickness, inline=True, fontsize=8, fmt='%d')
             #############################################################################################################################################
             # Add the specific 546 hPa contour line in blue (entre 1000-500)
             blueline = ax.contour(longitude[:], latitude[:], thickness[:, :], levels=[546], colors='dodgerblue', linestyles='dashed', transform=ccrs.PlateCarree(), zorder=1)
             # Add the specific 132 hPa contour line in red (1000-850)
             # red_line = ax.contour(longitude[:], latitude[:], thickness[:, :], levels=[132], colors='red', linestyles='dashed', transform=ccrs.PlateCarree(), zorder=1)
             #############################################################################################################################################
             
             # Add labels to the pressure lines
             # ax.clabel(cs, inline=True, fontsize=8, fmt='%d')

             # Format the datetime information for the filename
             formatted_time = current_time.strftime("%d_%m_%Y_%H:%M:%S")

             # Save the figure
             
             #para 1000-500
             output_filename = f"{output_folder}/thickness_mslp_{formatted_time}.png"
             #para 1000-850
             # output_filename = f"{output_folder}/thickness850_mslp_{formatted_time}.png"
             plt.title(f'Espessura e pressão em superfície às {formatted_time}')

             plt.show()
             plt.savefig(output_filename, dpi=300)
             plt.close()