#%%
import numpy as np
import xarray as xr
import pandas as pd
from glob import glob
import matplotlib.pyplot as plt
import cartopy.crs as crs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import (LongitudeFormatter, LatitudeFormatter)


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
    # Example usage:
    # save_time_step("input.nc", "last_time.nc")
    # save_time_step("input.nc", "specific_time.nc", time_index=2)

    Parameters:
    - input_file (str): Path to the input NetCDF file.
    - output_file (str): Path to the output NetCDF file.
    - time_index (int, optional): Index of the time step to save. If None, the last time step is used.

    Raises:
    - ValueError: If the 'time' variable is not found in the input file.
    - IndexError: If the provided time index is out of the valid range.
    """
    ds = xr.open_dataset(input_file)

    if "time" not in ds:
        raise ValueError("The 'time' variable was not found in the NetCDF file.")

    if time_index is None:
        time_index = -1

    if time_index < 0 or time_index >= ds["time"].size:
        raise IndexError("The time index is out of the valid range.")

    selected_time = ds["time"].isel(time=time_index)

    new_ds = xr.Dataset(
        {"time": ("time", [selected_time.values])},
        coords={"time": [selected_time.values]},
    )
    new_ds["time"].attrs = ds["time"].attrs

    new_ds.to_netcdf(output_file)
    print(f"Time saved in: {output_file}, Time index: {time_index}")



# %%
