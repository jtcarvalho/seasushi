#%%
import os
import yaml
import numpy as np
import xarray as xr
import pandas as pd
from glob import glob
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from cartopy.mpl.ticker import (LongitudeFormatter, LatitudeFormatter)

class Sushi:
    def __init__(self, fish_type, sushi_type):
        self.fish_type = fish_type
        self.sushi_type = sushi_type

    def make_sushi(self):
        return self.SushiPiece(self.fish_type, self.sushi_type)

    class SushiPiece:
        def __init__(self, fish, type):
            self.fish = fish
            self.type = type

        def get_details(self):
            return f"Fish: {self.fish}, Type: {self.type}"


def get_vect_components(direction):
    """
    Calculate the vector components from a given direction
    for plotting purposes.

    Parameters:
    direction (float): The direction in radians.

    Returns:
    tuple: The vector components (x, y).
    """
    return -np.sin(direction), np.cos(direction)

def replace_comma(file_in, file_out):
    """
    Replace commas with periods in a text file.

    Parameters:
    file_in (str): The input file path.
    file_out (str): The output file path.
    """
    with open(file_in, "r") as text:
        content = text.read().replace(",", ".")
    with open(file_out, "w") as output:
        output.write(content)

def resample_1h(ds):
    """
    Resample the dataset to 1-hour intervals.

    Parameters:
    ds (xarray.Dataset): The input dataset.

    Returns:
    xarray.Dataset: The resampled dataset.
    """
    return ds.resample(time="1h").mean()

def sel_date(ds, time_ini, time_end):
    """
    Select a subset of the dataset between two dates.

    Parameters:
    ds (xarray.Dataset): The input dataset.
    time_ini (str): The start date.
    time_end (str): The end date.

    Returns:
    xarray.Dataset: The subset of the dataset.
    """
    return ds.sel(time=slice(time_ini, time_end))

def sel_pto_regular(ds, x, y):
    """
    Select the nearest point in the dataset to the given coordinates
    in a netcdf regular grid file
    
    Parameters:
    ds (xarray.Dataset): The input dataset.
    x (float): The longitude.
    y (float): The latitude.

    Returns:
    xarray.Dataset: The subset of the dataset at the nearest point.
    """
    return ds.sel(lat=y, lon=x, method='nearest')

def box_files(filenames, lon_min, lon_max, lat_min, lat_max):
    """
    Select a geographical box from multiple dataset files.

    Parameters:
    filenames (list of str): The list of dataset file paths.
    lon_min (float): The minimum longitude.
    lon_max (float): The maximum longitude.
    lat_min (float): The minimum latitude.
    lat_max (float): The maximum latitude.

    Returns:
    xarray.Dataset: The subset of the dataset within the specified box.
    """
    return xr.open_mfdataset(filenames).sel(latitude=slice(lat_min, lat_max), longitude=slice(lon_min, lon_max))

def box_file(filename, lon_min, lon_max, lat_min, lat_max):
    """
    Select a geographical box from one single dataset file.

    Parameters:
    filename (str):  Dataset file path.
    lon_min (float): The minimum longitude.
    lon_max (float): The maximum longitude.
    lat_min (float): The minimum latitude.
    lat_max (float): The maximum latitude.

    Returns:
    xarray.Dataset: The subset of the dataset within the specified box.
    """
    return xr.open_dataset(filename).sel(latitude=slice(lat_min, lat_max), longitude=slice(lon_min, lon_max))

def save_time_step(input_file, output_file, time_index=None):
    """
    Save a specific time step from a NetCDF file into another NetCDF file.
    
    Parameters:
    - input_file (str): Path to the input NetCDF file.
    - output_file (str): Path to the output NetCDF file.
    - time_index (int, optional): Index of the time step to save. If None, the last time step is used.

    Raises:
    - ValueError: If the 'time' variable is not found in the input file.
    - IndexError: If the provided time index is out of the valid range.
    """
    ds = xr.open_dataset(input_file)

    # Check if the 'time' variable exists in the file
    if "time" not in ds:
        raise ValueError("The 'time' variable was not found in the NetCDF file.")

    # Ensure the time variable contains data
    time_size = ds["time"].size
    if time_size == 0:
        raise ValueError("The 'time' variable is empty in the NetCDF file.")
    
    # Set the time index
    if time_index is None:
        time_index = time_size - 1  # If not specified, use the last time index

    # Check if the time index is within the valid range
    if time_index < 0 or time_index >= time_size:
        raise IndexError(f"The time index {time_index} is out of the valid range. "
                          f"Valid range is 0 to {time_size - 1}.")

    # Select the desired time
    selected_time = ds.isel(time=time_index)

    # Create a new Dataset with the selected time
    new_ds = selected_time.expand_dims("time")

    # Copy attributes of the 'time' variable
    new_ds["time"].attrs = ds["time"].attrs

    # Save the new NetCDF file
    new_ds.to_netcdf(output_file)
    print(f"Time saved in: {output_file}, Time index: {time_index}")

# Function to load the YAML file
def load_config(yaml_file):
    """
    Load a YAML configuration file.

    Parameters:
    yaml_file (str): Path to the YAML file.

    Returns:
    dict: The loaded configuration as a dictionary.
    """
    with open(yaml_file, 'r') as file:
        return yaml.safe_load(file)

# Function to create the plots
def plot_variable(var, datetime_iso, formatted_datetime, title, output_file, cmap, vmin, vmax, label):
    """
    Create and save a plot of a variable.

    Parameters:
    var (xarray.DataArray): The variable to plot.
    datetime_iso (str): The date/time in ISO format.
    formatted_datetime (str): The formatted date/time for the output file name.
    title (str): The title of the plot.
    output_file (str): The path to save the plot.
    cmap (str): The colormap to use.
    vmin (float): The minimum value for normalization.
    vmax (float): The maximum value for normalization.
    label (str): The label for the colorbar.
    """
    plt.figure(figsize=(12, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())
    norm = Normalize(vmin=vmin, vmax=vmax)
    im = var.plot(
        ax=ax,
        transform=ccrs.PlateCarree(),
        cmap=cmap,
        norm=norm,
        add_colorbar=False
    )
    ax.coastlines()
    
    # Update the title with the date/time in simplified ISO format
    plt.title(f"{title} at {datetime_iso}", fontsize=14)
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    
    # Add the horizontal colorbar
    cbar = plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.05, aspect=50)
    cbar.set_label(label, fontsize=12)
    
    # Add ">" symbol for values greater than the maximum
    cbar_ticks = list(range(vmin, vmax + 1, 2))
    cbar_ticks[-1] = f'>{vmax}'
    cbar.set_ticks(range(vmin, vmax + 1, 2))
    cbar.ax.set_xticklabels(cbar_ticks)
    
    # Save the figure with the simplified date format
    plt.savefig(output_file.replace("{datetime}", formatted_datetime), bbox_inches='tight')
    plt.close()

# Function to ensure the output directory exists
def prepare_output_directory(output_file):
    """
    Ensure the output directory exists.

    Parameters:
    output_file (str): The path to the output file.
    """
    output_dir = os.path.dirname(output_file)
    os.makedirs(output_dir, exist_ok=True)

# Function to open the dataset
def open_dataset(file_path):
    """
    Open a dataset from a file.

    Parameters:
    file_path (str): The path to the dataset file.

    Returns:
    xarray.Dataset: The opened dataset.
    """
    return xr.open_dataset(file_path)

# Function to extract date and time information
def extract_datetime_info(datetime_raw):
    """
    Extract date and time information from a raw datetime string.

    Parameters:
    datetime_raw (str): The raw datetime string.

    Returns:
    tuple: The ISO formatted datetime and the formatted datetime for the output file name.
    """
    datetime_raw = str(datetime_raw)  # Extract the date and time of the last time step
    datetime_iso = datetime_raw.split('.')[0][:-3]  # Ex.: '2003-07-01T00:00'
    formatted_datetime = datetime_iso.replace("T", "-").replace(":", "")[:13] + "h"  # Ex.: '2003-07-01-00h'
    return datetime_iso, formatted_datetime

# %%
