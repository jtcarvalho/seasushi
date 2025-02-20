#%%
import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import json
from glob import glob
from natsort import natsorted

class WW3:
    def __init__(self):
        pass

    def check_data(self, buoy, model):
        """
        Check for common dates between buoy and satellite data.
        """
        try:
            # Open files
            buoy_data = xr.open_dataset(buoy)
            model_data = xr.open_dataset(model)
            # Check common dates
            valid_dates_buoy = buoy_data['time'].where(~np.isnan(buoy_data['hs']), drop=True)
            valid_dates_mod = model_data['time'].where(~np.isnan(model_data['hs']), drop=True)
            common_dates = pd.Index(valid_dates_mod.values).intersection(pd.Index(buoy_data['time'].values))

            return common_dates if not common_dates.empty else None

        except Exception as e:
            print(f"An error occurred while checking the NetCDF files: {e}")
            return None

        finally:
            buoy_data.close()
            model_data.close()

    def load_data(self, buoy_file, model_file):
        """
        Load data from NetCDF files.
        """
        buoy_data = xr.open_dataset(buoy_file)
        model_data = xr.open_dataset(model_file)
        
        return buoy_data, model_data

    def filter_data_by_dates(self, data, dates):
        """
        Filter data to include only the specified dates.
        """
        return data.sel(time=dates)

    def calculate_metrics(self, buoy_data, model_data):
        """
        Calculate RMSE, BIAS, Normalized RMSE, and Normalized BIAS using sums and number of observations.
        """
        buoy_values = buoy_data['hs'].values
        model_values = model_data['hs'].values
        
        # Remove NaN values
        mask = ~np.isnan(buoy_values) & ~np.isnan(model_values)
        buoy_values = buoy_values[mask]
        model_values = model_values[mask]
        
        if len(buoy_values) == 0 or len(model_values) == 0:
            return np.nan, np.nan, np.nan, np.nan, np.nan
        
        # Number of observations
        nobs = len(buoy_values)
        
        # Sum of observed values
        obs_sum = np.sum(buoy_values)
        
        # Calculating BIAS
        bias = np.mean(model_values - buoy_values)
        
        # Calculating RMSE
        rmse = np.sqrt(np.mean((model_values - buoy_values) ** 2))
        
        # Normalized BIAS (using sum)
        normalized_bias = np.sum(model_values - buoy_values) / obs_sum
        
        # Normalized RMSE (using sum)
        normalized_rmse = np.sqrt(np.sum((model_values - buoy_values) ** 2) / obs_sum)
        
        return rmse, bias, normalized_rmse, normalized_bias, nobs

    def plot_time_series(self, buoy_data, model_data, common_dates, exp, output_file):
        """
        Plot time series data and save as PNG.
        """
        # Filter data by common dates
        buoy_filtered = self.filter_data_by_dates(buoy_data, common_dates)
        model_filtered = self.filter_data_by_dates(model_data, common_dates)
        
        # Calculate statistics
        rmse, bias, normalized_rmse, normalized_bias, nobs = self.calculate_metrics(buoy_filtered, model_filtered)
        
        # Plot the data
        plt.figure(figsize=(10, 6))
        plt.plot(buoy_filtered['time'], buoy_filtered['hs'], label='Buoy', color='r')
        plt.plot(model_filtered['time'], model_filtered['hs'], label='WW3', color='k', linestyle=':', linewidth=2.0)
        
        # Add title and labels
        plt.title(f'Comparison for Buoy x WW3 ({exp}) point at East of Martinique')
        plt.xlabel('Date')
        plt.ylabel('Hs')
        plt.ylim(0, 5)  # Set y-axis range
        plt.legend()
        
        # Add statistics to the plot
        stats_text = (f'RMSE: {rmse:.2f}\n'
                      f'BIAS: {bias:.2f}\n'
                      f'NRMSE: {normalized_rmse:.2f}\n'
                      f'NBIAS: {normalized_bias:.2f}\n'
                      f'Number of Observations: {nobs}')
        plt.gca().text(0.09, 0.95, stats_text, transform=plt.gca().transAxes,
                       fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
        
        # Save the plot as a PNG file
        plt.savefig(output_file)
        plt.close()

    def buoy_extraction(self, base, x, y, outname, variables):
        buffer = []
        for f in natsorted(glob(base)):
            print(f)
            ds = xr.open_dataset(f)
            ctrl = self.check_data(f)
            if ctrl:
                ds = ds.isel(node=self.nearest(ds, x, y))[variables]
                print(x, y)
                print('#-----#')
                buffer.append(ds)
            else:
                continue
        out = xr.concat(buffer, dim='time')
        out.to_netcdf(f'{outname}.nc')

    def nearest(self, ds, x, y):
        return np.argmin(np.abs(ds.longitude.values - x) + np.abs(ds.latitude.values - y))

# Main program
if __name__ == "__main__":
    ww3 = WW3()
    # File paths
    buoy_file = f'{pathBuoy}/41040.nc'
    
    exp = 'expb2_143_fct'
    model_file = f'{pathModel}/{exp}/output/points/ww3_41040.nc'

    # Check common dates
    common_dates = ww3.check_data(buoy_file, model_file)
    
    if common_dates is not None:
        # Load data
        buoy_data, model_data = ww3.load_data(buoy_file, model_file)
        
        # Output file
        output_file = f'/work/cmcc/jc11022/projects/ww3Tools/myBuoytools/figs/ww3xbuoy_{exp}.png'

        # Plot and save the time series comparison
        ww3.plot_time_series(buoy_data, model_data, common_dates, exp, output_file)
        
        print(f"Saved as {output_file}")
    else:
        print("No common dates")

# %%
