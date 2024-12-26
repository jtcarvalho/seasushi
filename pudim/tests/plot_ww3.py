#%%
import os
import argparse
from pudim.tools import load_config, plot_variable, prepare_output_directory, open_dataset, extract_datetime_info

# Principal
def main():
    # Argumentos do terminal
    parser = argparse.ArgumentParser(description="Plot variables from NetCDF using YAML configuration.")
    parser.add_argument(
        "-c", "--config", required=True, 
        help="Path to the YAML configuration file."
    )
    args = parser.parse_args()

    # Carregar as configurações do YAML
    config = load_config(args.config)

    # Garantir que o diretório de saída existe
    for plot in config["plots"]:
        prepare_output_directory(plot["output_file"])

    # Abrir o dataset
    dataset = open_dataset(config["dataset"]["file_path"])

    # Gerar os plots definidos no YAML
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

    # Fechar o dataset
    dataset.close()

if __name__ == "__main__":
    main()

# %%
