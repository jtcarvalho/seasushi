#%%
import os
import argparse
import unittest
from seasushi.tools import load_config, plot_variable, prepare_output_directory, open_dataset, extract_datetime_info

# Main function
def main():
    """
    Main function to plot variables from WW3 Global NetCDF using a YAML configuration file.
    This function performs the following steps:
    1. Parses terminal arguments to get the path to the YAML configuration file.
    2. Loads configurations from the specified YAML file.
    3. Ensures that the output directories for the plots exist.
    4. Opens the WW3 Global NetCDF dataset.
    5. Generates the plots as defined in the YAML configuration.
    6. Closes the dataset.
    Args:
        -c (--config): Path to the YAML configuration file.
    Returns:
        None
    """
    # Terminal arguments
    parser = argparse.ArgumentParser(description="Plot variables from WW3 Global NetCDF using YAML configuration.")
    parser.add_argument(
        "-c", "--config", required=True, 
        help="Path to the YAML configuration file."
    )
    args = parser.parse_args()

    # Load configurations from YAML
    config = load_config(args.config)

    # Ensure the output directory exists
    for plot in config["plots"]:
        prepare_output_directory(plot["output_file"])

    # Open the dataset
    dataset = open_dataset(config["dataset"]["file_path"])

    # Generate the plots defined in YAML
    for plot in config["plots"]:
        var = dataset[plot["variable"]].isel(time=-1)
        datetime_iso, formatted_datetime = extract_datetime_info(var.time.values)
        
        plot_variable(
            var,
            datetime_iso=datetime_iso,
            formatted_datetime=formatted_datetime,
            title=plot["title"],
            output_file=plot["output_file"],
            cmap=plot["cmap"],
            vmin=plot["vmin"],
            vmax=plot["vmax"],
            label=plot["label"]
        )

    # Close the dataset
    dataset.close()

if __name__ == "__main__":
    main()



# %%
