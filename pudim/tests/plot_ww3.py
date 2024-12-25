#%%
import os
import yaml
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from matplotlib.colors import Normalize
import argparse  # Para receber argumentos no terminal

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
        output_dir = os.path.dirname(plot["output_file"])
        os.makedirs(output_dir, exist_ok=True)

    # Abrir o dataset
    dataset = xr.open_dataset(config["dataset"]["file_path"])

    # Gerar os plots definidos no YAML
    for plot in config["plots"]:
        var = dataset[plot["variable"]].isel(time=-1)
        datetime_raw = str(var.time.values)  # Extrai a data e hora do último passo de tempo
        
        datetime_iso = datetime_raw.split('.')[0][:-3]  # Ex.: '2003-07-01T00:00'
        formatted_datetime = datetime_iso.replace("T", "-").replace(":", "")[:13] + "h"  # Ex.: '2003-07-01-00h'


        
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
