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

    # Verificando se a variável 'time' existe no arquivo
    if "time" not in ds:
        raise ValueError("The 'time' variable was not found in the NetCDF file.")

    # Garantir que a variável de tempo contém dados
    time_size = ds["time"].size
    if time_size == 0:
        raise ValueError("The 'time' variable is empty in the NetCDF file.")
    
    # Definir o índice de tempo
    if time_index is None:
        time_index = time_size - 1  # Se não for especificado, pega o último índice de tempo

    # Verificação se o índice de tempo está dentro do intervalo válido
    if time_index < 0 or time_index >= time_size:
        raise IndexError(f"The time index {time_index} is out of the valid range. "
                          f"Valid range is 0 to {time_size - 1}.")

    # Selecionar o tempo desejado
    selected_time = ds.isel(time=time_index)

    # Criando um novo Dataset com o tempo selecionado
    new_ds = selected_time.expand_dims("time")

    # Copiar atributos da variável 'time'
    new_ds["time"].attrs = ds["time"].attrs

    # Salvar o novo arquivo NetCDF
    new_ds.to_netcdf(output_file)
    print(f"Time saved in: {output_file}, Time index: {time_index}")


# Função para carregar o arquivo YAML
def load_config(yaml_file):
    with open(yaml_file, 'r') as file:
        return yaml.safe_load(file)

# Função para criar os plots
def plot_variable(var, datetime_iso, formatted_datetime, title, output_file, cmap, vmin, vmax, label):
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
    
    # Atualiza o título com a data/hora no formato ISO simplificado
    plt.title(f"{title} at {datetime_iso}", fontsize=14)
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    
    # Adiciona o colorbar horizontal
    cbar = plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.05, aspect=50)
    cbar.set_label(label, fontsize=12)
    
    # Adiciona símbolo ">" para valores maiores que o máximo
    cbar_ticks = list(range(vmin, vmax + 1, 2))
    cbar_ticks[-1] = f'>{vmax}'
    cbar.set_ticks(range(vmin, vmax + 1, 2))
    cbar.ax.set_xticklabels(cbar_ticks)
    
    # Salva a figura com o formato de data simplificado
    plt.savefig(output_file.replace("{datetime}", formatted_datetime), bbox_inches='tight')
    plt.close()

# Função para garantir que o diretório de saída existe
def prepare_output_directory(output_file):
    output_dir = os.path.dirname(output_file)
    os.makedirs(output_dir, exist_ok=True)

# Função para abrir o dataset
def open_dataset(file_path):
    return xr.open_dataset(file_path)

# Função para extrair informações de data e hora
def extract_datetime_info(datetime_raw):
    datetime_raw = str(datetime_raw)  # Extrai a data e hora do último passo de tempo
    datetime_iso = datetime_raw.split('.')[0][:-3]  # Ex.: '2003-07-01T00:00'
    formatted_datetime = datetime_iso.replace("T", "-").replace(":", "")[:13] + "h"  # Ex.: '2003-07-01-00h'
    return datetime_iso, formatted_datetime

# %%
