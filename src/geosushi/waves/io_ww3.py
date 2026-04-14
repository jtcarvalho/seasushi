import numpy as np
import xarray as xr
import pandas as pd


def load_ww3_global(file_path, variables=None):
    """
    Load a WW3 global/regional NetCDF output file.

    Parameters
    ----------
    file_path : str
        Path to the WW3 NetCDF file.
    variables : list of str, optional
        List of variable names to load. If None, all variables are loaded.

    Returns
    -------
    xarray.Dataset
        Dataset with the requested variables and coordinates.
    """
    ds = xr.open_dataset(file_path)
    if variables is not None:
        ds = ds[variables]
    return ds


def load_ww3_fields(file_path, time_index=None, variables=None):
    """
    Load WW3 field output at a specific time step.

    Parameters
    ----------
    file_path : str
        Path to the WW3 NetCDF file.
    time_index : int, optional
        Time index to select. If None, all time steps are returned.
    variables : list of str, optional
        Variables to extract. If None, all variables are loaded.

    Returns
    -------
    xarray.Dataset
        Dataset at the selected time step (or full time series if time_index is None).
    """
    ds = xr.open_dataset(file_path)
    if variables is not None:
        ds = ds[variables]
    if time_index is not None:
        ds = ds.isel(time=time_index)
    return ds


def load_ww3_point(file_path, variables=None):
    """
    Load a WW3 point output NetCDF file (e.g. ww3_<stationid>.nc).

    Parameters
    ----------
    file_path : str
        Path to the WW3 point NetCDF file.
    variables : list of str, optional
        Variables to extract (e.g. ['hs', 'fp', 'dir']).
        If None, all variables are returned.

    Returns
    -------
    xarray.Dataset
        Dataset with time as the primary dimension.
    """
    ds = xr.open_dataset(file_path)
    if variables is not None:
        ds = ds[variables]
    return ds


def load_ww3_hs(file_path, time_index=None):
    """
    Load significant wave height (hs) from a WW3 NetCDF file.

    Parameters
    ----------
    file_path : str
        Path to the WW3 NetCDF file.
    time_index : int, optional
        Time index to select. If None, all time steps are returned.

    Returns
    -------
    tuple
        (hs, lon, lat) — arrays of significant wave height, longitude and latitude.
    """
    ds = xr.open_dataset(file_path)

    lon = ds['longitude'].values if 'longitude' in ds else ds['lon'].values
    lat = ds['latitude'].values if 'latitude' in ds else ds['lat'].values

    hs_var = None
    for candidate in ['hs', 'VHM0', 'swh', 'HTSGW']:
        if candidate in ds:
            hs_var = candidate
            break
    if hs_var is None:
        raise KeyError(
            f"No significant wave height variable found in {file_path}. "
            f"Available variables: {list(ds.data_vars)}"
        )

    if time_index is not None:
        hs = ds[hs_var].isel(time=time_index).values
    else:
        hs = ds[hs_var].values

    ds.close()
    return hs, lon, lat


def extract_ww3_point_series(file_path, variable='hs'):
    """
    Extract a time series of a variable from a WW3 point output file.

    Parameters
    ----------
    file_path : str
        Path to the WW3 point NetCDF file.
    variable : str
        Variable name to extract (default: 'hs').

    Returns
    -------
    tuple
        (time, values) — arrays of time and variable values.
    """
    ds = xr.open_dataset(file_path)
    if variable not in ds:
        raise KeyError(
            f"Variable '{variable}' not found. "
            f"Available: {list(ds.data_vars)}"
        )
    time = ds['time'].values
    values = ds[variable].values.squeeze()
    ds.close()
    return time, values


def extract_ww3_at_point(file_path, lat, lon, variables=None,
                         start_date=None, end_date=None):
    """
    Extract a WW3 grid time series at the point nearest to (lat, lon).

    Parameters
    ----------
    file_path : str
        Path to the WW3 NetCDF file (field output with lat/lon dimensions).
    lat : float
        Target latitude in decimal degrees (negative for South).
    lon : float
        Target longitude in decimal degrees (negative for West).
    variables : list of str, optional
        Variables to extract. If None, all variables are returned.
    start_date : str or datetime-like, optional
        Inclusive start of the time range filter.
    end_date : str or datetime-like, optional
        Inclusive end of the time range filter.

    Returns
    -------
    xarray.Dataset
        Dataset with time as the primary dimension at the selected grid point.
    """
    ds = xr.open_dataset(file_path)

    lat_name = 'latitude' if 'latitude' in ds.coords else 'lat'
    lon_name = 'longitude' if 'longitude' in ds.coords else 'lon'

    # Normalize longitude to the dataset's convention (0-360 or -180/180)
    lon_vals = ds[lon_name].values
    if lon_vals.min() >= 0 and lon < 0:
        lon = lon + 360.0
    elif lon_vals.min() < 0 and lon > 180:
        lon = lon - 360.0

    ds = ds.sel({lat_name: lat, lon_name: lon}, method='nearest')

    if variables is not None:
        ds = ds[variables]

    if start_date is not None:
        ds = ds.sel(time=slice(pd.to_datetime(start_date), None))
    if end_date is not None:
        ds = ds.sel(time=slice(None, pd.to_datetime(end_date)))

    return ds
