##%
import os
import xarray as xr
import tools
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

# Create the directory if it doesn't exist
output_dir = '../figs'
os.makedirs(output_dir, exist_ok=True)

# Open the NetCDF file using xarray
file_path = '../data/ww3-glo025.nc'
dataset = xr.open_dataset(file_path)

# Get the last time step
time_var = dataset['time']
last_time_index = -1  # Last time step
last_time = tools.get_time_step(time_var, last_time_index)

# Get the hs and t01 variables
hs_var = dataset['hs'].isel(time=last_time_index)
t01_var = dataset['t01'].isel(time=last_time_index)

# Plot hs
plt.figure(figsize=(10, 6))
ax = plt.axes(projection=ccrs.PlateCarree())
hs_plot = hs_var.plot(ax=ax, transform=ccrs.PlateCarree(), cmap='viridis', vmin=0, vmax=12)
plt.colorbar(hs_plot, ax=ax, label='Significant Wave Height (m)', ticks=range(0, 13, 1))
ax.coastlines()
plt.title(f'Significant Wave Height at {last_time}')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.savefig(os.path.join(output_dir, 'hs_plot.png'))
plt.close()

# Plot t01
plt.figure(figsize=(10, 6))
ax = plt.axes(projection=ccrs.PlateCarree())
t01_plot = t01_var.plot(ax=ax, transform=ccrs.PlateCarree(), cmap='viridis', vmin=2, vmax=16)
plt.colorbar(t01_plot, ax=ax, label='Wave Period (s)', ticks=range(2, 17, 2))
ax.coastlines()
plt.title(f'Wave Period at {last_time}')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.savefig(os.path.join(output_dir, 't01_plot.png'))
plt.close()

# Close the dataset
dataset.close()