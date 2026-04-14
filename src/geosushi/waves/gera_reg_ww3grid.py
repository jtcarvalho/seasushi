"""
Geração de grade regular WW3 (WAVE WATCH III).

Este módulo é derivado de myWW3tools/scripts/genWW3grid_fromCoords.py e
expõe as funções principais como uma API importável dentro do pacote geosushi.

Funções principais
------------------
gera_grade_regular(...)
    Gera os arquivos de grade WW3 (.dep, .mask, .obs, .meta) a partir de
    limites geográficos e uma batimetria local (NetCDF).

Funções auxiliares (também exportadas)
---------------------------------------
load_bathy_from_file      — carrega batimetria de NetCDF local
load_bathy_pygmt          — baixa batimetria via pygmt/GMT
load_bathymetry           — roteador inteligente entre as fontes
interpolate_to_ww3_grid   — interpolação bilinear para grade regular
make_mask                 — cria máscara terra/oceano
write_dep                 — escreve arquivo .dep
write_mask                — escreve arquivo .mask
write_obs                 — escreve arquivo .obs (obstáculos zerados)
write_meta                — escreve arquivo .meta para grade RECT

Exemplo de uso
--------------
>>> from geosushi.waves.gera_reg_ww3grid import gera_grade_regular
>>> gera_grade_regular(
...     lon_min=-46.0, lon_max=-38.0,
...     lat_min=-26.0, lat_max=-18.0,
...     dx=0.05, dy=0.05,
...     bathy_file='/data/gebco.nc',
...     output_dir='ww3_grid/',
...     prefix='rio',
... )
"""

import os
import numpy as np
import xarray as xr
from scipy.interpolate import RegularGridInterpolator, griddata

try:
    import yaml
    _HAS_YAML = True
except ImportError:
    _HAS_YAML = False


# ============================================================================
# Batimetria — carregamento
# ============================================================================

def _normalize_coords(da):
    """Renomeia dims para 'lat'/'lon' de forma consistente."""
    rename = {}
    for dim in da.dims:
        d = dim.lower()
        if d in ('latitude', 'y'):
            rename[dim] = 'lat'
        elif d in ('longitude', 'x'):
            rename[dim] = 'lon'
    if rename:
        da = da.rename(rename)
    coord_rename = {}
    for coord in da.coords:
        c = coord.lower()
        if c in ('latitude', 'y') and coord not in da.dims:
            coord_rename[coord] = 'lat'
        elif c in ('longitude', 'x') and coord not in da.dims:
            coord_rename[coord] = 'lon'
    if coord_rename:
        da = da.rename(coord_rename)
    return da


def _sort_coords(da):
    """Garante que lat e lon sejam monotonicamente crescentes."""
    if 'lat' in da.dims and da['lat'].values[0] > da['lat'].values[-1]:
        da = da.isel(lat=slice(None, None, -1))
    if 'lon' in da.dims and da['lon'].values[0] > da['lon'].values[-1]:
        da = da.isel(lon=slice(None, None, -1))
    return da


def load_bathy_from_file(filepath, elev_var=None,
                         lon_min=None, lon_max=None,
                         lat_min=None, lat_max=None, pad=1.0):
    """
    Carrega batimetria de um arquivo NetCDF local (GEBCO, ETOPO, etc.).

    Faz subset da região pedida (+ pad graus) por eficiência.

    Parameters
    ----------
    filepath : str
        Caminho para o arquivo NetCDF.
    elev_var : str, optional
        Nome da variável de elevação. Auto-detectada se None.
    lon_min, lon_max, lat_min, lat_max : float, optional
        Limites do domínio para subset.
    pad : float
        Margem adicional em graus ao redor do domínio.

    Returns
    -------
    xarray.DataArray
        DataArray com dims ('lat', 'lon') e elevação em metros
        (negativo = oceano, positivo = terra).
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Arquivo de batimetria não encontrado: {filepath}")

    print(f"  Abrindo batimetria: {filepath}")
    ds = xr.open_dataset(filepath)

    if elev_var is None:
        for candidate in ['elevation', 'z', 'altitude', 'depth',
                          'topo', 'height', 'Band1']:
            if candidate in ds.data_vars:
                elev_var = candidate
                break
    if elev_var is None:
        raise ValueError(
            f"Variável de elevação não detectada. Disponíveis: {list(ds.data_vars)}\n"
            "Defina 'elev_var' explicitamente."
        )

    da = ds[elev_var].squeeze()
    da = _normalize_coords(da)
    da = _sort_coords(da)

    if lat_min is not None:
        lat_lo = max(float(da['lat'].min()), lat_min - pad)
        lat_hi = min(float(da['lat'].max()), lat_max + pad)
        da = da.sel(lat=slice(lat_lo, lat_hi))

    if lon_min is not None:
        src_lon_min = float(da['lon'].min())
        src_lon_max = float(da['lon'].max())
        req_lon_min = lon_min - pad
        req_lon_max = lon_max + pad

        if src_lon_min >= 0 and req_lon_min < 0:
            req_lon_min += 360
            req_lon_max += 360
        elif src_lon_min < 0 and req_lon_min > 180:
            req_lon_min -= 360
            req_lon_max -= 360

        req_lon_min = max(src_lon_min, req_lon_min)
        req_lon_max = min(src_lon_max, req_lon_max)
        da = da.sel(lon=slice(req_lon_min, req_lon_max))

    print(f"  Variável '{elev_var}' | "
          f"lat [{float(da.lat.min()):.2f}, {float(da.lat.max()):.2f}]  "
          f"lon [{float(da.lon.min()):.2f}, {float(da.lon.max()):.2f}]")
    return da


def load_bathy_pygmt(lon_min, lon_max, lat_min, lat_max, resolution='01m'):
    """
    Baixa batimetria via pygmt (requer pygmt + GMT instalados).

    Parameters
    ----------
    lon_min, lon_max, lat_min, lat_max : float
        Limites do domínio.
    resolution : str
        Resolução GMT, ex.: '01m' (1 arc-min), '30s', '15s'.

    Returns
    -------
    xarray.DataArray
        DataArray com dims ('lat', 'lon') e elevação em metros.
    """
    try:
        import pygmt
    except ImportError:
        raise ImportError(
            "pygmt não está instalado. Instale com: pip install pygmt\n"
            "Também requer GMT (https://www.generic-mapping-tools.org/)"
        )
    pad = 0.5
    region = [lon_min - pad, lon_max + pad, lat_min - pad, lat_max + pad]
    print(f"  Baixando batimetria via pygmt (resolução={resolution}, região={region})")
    grid = pygmt.datasets.load_earth_relief(resolution=resolution, region=region)
    da = _normalize_coords(grid)
    da = _sort_coords(da)
    return da


def load_bathymetry(bathy_file=None, lon_min=None, lon_max=None,
                    lat_min=None, lat_max=None,
                    source='file', elev_var=None,
                    pygmt_resolution='01m'):
    """
    Carrega batimetria a partir da fonte configurada.

    Parameters
    ----------
    bathy_file : str, optional
        Caminho para arquivo NetCDF local (GEBCO, ETOPO, etc.).
    lon_min, lon_max, lat_min, lat_max : float
        Limites do domínio.
    source : {'file', 'pygmt'}
        Fonte da batimetria. Se 'file' e bathy_file é None, tenta pygmt.
    elev_var : str, optional
        Nome da variável de elevação no arquivo NetCDF.
    pygmt_resolution : str
        Resolução para o pygmt (padrão '01m').

    Returns
    -------
    xarray.DataArray
        DataArray com dims ('lat', 'lon') e elevação em metros.
    """
    if source == 'file' and bathy_file is not None:
        return load_bathy_from_file(
            bathy_file, elev_var, lon_min, lon_max, lat_min, lat_max)
    elif source == 'pygmt' or (source == 'file' and bathy_file is None):
        return load_bathy_pygmt(
            lon_min, lon_max, lat_min, lat_max, pygmt_resolution)
    else:
        raise ValueError(f"Fonte de batimetria desconhecida: '{source}'")


# ============================================================================
# Interpolação
# ============================================================================

def interpolate_to_ww3_grid(bathy_da, lon_ww3, lat_ww3):
    """
    Interpolação bilinear da batimetria para uma grade WW3 regular.

    Parameters
    ----------
    bathy_da : xarray.DataArray
        Batimetria com dims ('lat', 'lon') em metros.
    lon_ww3 : numpy.ndarray
        Vetor de longitudes 1-D da grade WW3.
    lat_ww3 : numpy.ndarray
        Vetor de latitudes 1-D da grade WW3.

    Returns
    -------
    numpy.ndarray
        Array 2-D de elevação (ny, nx) em metros.
        Pontos fora do domínio preenchidos com 999999 (sentinela de terra).
    """
    src_lat = bathy_da['lat'].values.astype(float)
    src_lon = bathy_da['lon'].values.astype(float)
    src_values = bathy_da.values.astype(float)

    src_lon_0360 = src_lon[0] >= 0
    ww3_lon_neg = lon_ww3[0] < 0

    interp_lon_ww3 = lon_ww3.copy()
    if src_lon_0360 and ww3_lon_neg:
        interp_lon_ww3 = lon_ww3 + 360.0
    elif not src_lon_0360 and not ww3_lon_neg and lon_ww3[0] > 180:
        interp_lon_ww3 = lon_ww3 - 360.0

    interp_lon_ww3 = np.clip(interp_lon_ww3, src_lon.min(), src_lon.max())
    interp_lat_ww3 = np.clip(lat_ww3, src_lat.min(), src_lat.max())

    interpolator = RegularGridInterpolator(
        (src_lat, src_lon), src_values,
        method='linear', bounds_error=False, fill_value=999999.0
    )

    lon_grid, lat_grid = np.meshgrid(interp_lon_ww3, interp_lat_ww3)
    pts = np.column_stack([lat_grid.ravel(), lon_grid.ravel()])
    elev = interpolator(pts).reshape(len(lat_ww3), len(lon_ww3))
    return elev


# ============================================================================
# Máscara
# ============================================================================

def make_mask(elevation, zdep=-0.10):
    """
    Cria máscara oceano/terra a partir da elevação.

    Parameters
    ----------
    elevation : numpy.ndarray
        Array 2-D de elevação em metros (negativo = oceano).
    zdep : float
        Profundidade limite para discriminar terra/oceano (padrão -0.10 m).

    Returns
    -------
    numpy.ndarray
        Array 2-D de inteiros: 0 = terra, 1 = oceano.
    """
    return np.where(elevation < zdep, 1, 0).astype(int)


# ============================================================================
# Escrita de arquivos WW3
# ============================================================================

_LAND_VALUE_MM = 999999000


def write_dep(filepath, elevation_m, nx, ny):
    """
    Escreve arquivo de batimetria WW3 (.dep).

    Os valores são armazenados em milímetros inteiros (fator de escala 0.001).
    Layout IDLA=1: linha a linha, de sul para norte.

    Parameters
    ----------
    filepath : str
        Caminho do arquivo de saída.
    elevation_m : numpy.ndarray
        Elevação em metros, shape (ny, nx).
    nx, ny : int
        Dimensões da grade.
    """
    data_mm = np.round(elevation_m * 1000).astype(np.int64)
    data_mm[data_mm > 900000000] = _LAND_VALUE_MM

    with open(filepath, 'w') as f:
        for j in range(ny):
            line = ''.join(f'{data_mm[j, i]:12d}' for i in range(nx))
            f.write(line + '\n')


def write_mask(filepath, mask, nx, ny):
    """
    Escreve arquivo de máscara WW3 (.mask).

    0 = terra, 1 = oceano. Layout IDLA=1 (sul para norte).

    Parameters
    ----------
    filepath : str
        Caminho do arquivo de saída.
    mask : numpy.ndarray
        Máscara inteira 2-D, shape (ny, nx).
    nx, ny : int
        Dimensões da grade.
    """
    with open(filepath, 'w') as f:
        for j in range(ny):
            line = ''.join(f'{mask[j, i]:3d}' for i in range(nx))
            f.write(line + '\n')


def write_obs(filepath, nx, ny):
    """
    Escreve arquivo de obstáculos sub-grade WW3 (.obs) com todos zeros.

    Shape: (2*NY, NX) — linhas 0..NY-1 para faces E-O, NY..2NY-1 para N-S.

    Parameters
    ----------
    filepath : str
        Caminho do arquivo de saída.
    nx, ny : int
        Dimensões da grade.
    """
    with open(filepath, 'w') as f:
        for j in range(2 * ny):
            line = ''.join(f'{0:3d}' for _ in range(nx))
            f.write(line + '\n')


_META_HEADER = """\
$ Define grid -------------------------------------- $
$ Five records containing :
$  1 Type of grid, coordinate system and type of closure: GSTRG, FLAGLL,
$    CSTRG. Grid closure can only be applied in spherical coordinates.
$      GSTRG  : 'RECT' (rectilinear) or 'CURV' (curvilinear)
$      FLAGLL : T = Spherical (lon/lat degrees), F = Cartesian (m)
$      CSTRG  : 'NONE', 'SMPL' (periodic i), 'TRPL' (tripole)
$  2 NX, NY
$  3 Grid increments SX, SY and scaling (division) factor.
$  4 Coordinates of (1,1) and scaling factor.
$  5 Bottom depth parameters, file unit, scale, IDLA, IDFM, format, FROM, name.
$
"""


def write_meta(filepath, nx, ny, dx, dy, lon1, lat1, prefix,
               zdep=-0.10, zmin=2.50, closure='NONE'):
    """
    Escreve o arquivo meta de grade WW3 para grade RETANGULAR (.meta).

    Os incrementos são armazenados em arc-minutos (SX = dx * 60) com
    SCALE=60, seguindo a convenção do WW3 gridgen.

    Grades globais (span de lon ≈ 360°) recebem automaticamente closure='SMPL'.

    Parameters
    ----------
    filepath : str
        Caminho do arquivo de saída.
    nx, ny : int
        Número de pontos em lon e lat.
    dx, dy : float
        Espaçamento de grade em graus.
    lon1, lat1 : float
        Coordenadas do canto inferior esquerdo (primeiro ponto).
    prefix : str
        Prefixo dos arquivos de saída (usado nas referências internas do .meta).
    zdep : float
        Profundidade limite terra/oceano (padrão -0.10 m).
    zmin : float
        Profundidade mínima permitida no WW3 (padrão 2.50 m).
    closure : str
        Tipo de fechamento: 'NONE', 'SMPL' ou 'TRPL'.
    """
    sx = dx * 60.0
    sy = dy * 60.0
    scale_xy = 60.0

    lon_span = (nx - 1) * dx
    if abs(lon_span - 360.0) < dx:
        closure = 'SMPL'

    with open(filepath, 'w') as f:
        f.write(_META_HEADER)
        f.write(f"   'RECT'  T  '{closure}'\n")
        f.write(f"{nx:7d}  {ny:7d}\n")
        f.write(f"{sx:8.2f}   {sy:8.2f}   {scale_xy:.2f}\n")
        f.write(f"{lon1:.4f}         {lat1:.4f}         1.00\n")
        f.write(f"$ Bottom Bathymetry\n")
        f.write(f"{zdep:.2f}   {zmin:.2f}  40  0.001000  1  1 "
                f"'(....)'  NAME  './{prefix}.dep'\n")
        f.write(f"$ Sub-grid information\n")
        f.write(f"50  0.010000  1  1  '(....)'  NAME  './{prefix}.obs'\n")
        f.write(f"$ Mask Information\n")
        f.write(f"60  1  1  '(....)'  NAME  './{prefix}.mask'\n")


# ============================================================================
# Função principal
# ============================================================================

def gera_grade_regular(lon_min, lon_max, lat_min, lat_max,
                       dx, dy,
                       bathy_file=None,
                       output_dir='ww3_grid_output',
                       prefix='grid',
                       zdep=-0.10,
                       zmin=2.50,
                       bathy_source='file',
                       elev_var=None,
                       pygmt_resolution='01m'):
    """
    Gera os arquivos de grade regular WW3 a partir de limites geográficos.

    Cria os seguintes arquivos no diretório de saída:
      {prefix}.dep   — batimetria (mm, escala 0.001 → m)
      {prefix}.mask  — máscara terra/oceano (0=terra, 1=oceano)
      {prefix}.obs   — obstáculos sub-grade (todos zeros)
      {prefix}.meta  — definição da grade WW3 (para ww3_grid.inp)

    Parameters
    ----------
    lon_min, lon_max : float
        Limites de longitude em graus (convenção -180/180 ou 0/360).
    lat_min, lat_max : float
        Limites de latitude em graus.
    dx, dy : float
        Espaçamento da grade em graus.
    bathy_file : str, optional
        Caminho para arquivo NetCDF de batimetria (GEBCO, ETOPO, etc.).
        Se None e bathy_source='file', tenta usar pygmt automaticamente.
    output_dir : str
        Diretório de saída (criado automaticamente se não existir).
    prefix : str
        Prefixo para os arquivos gerados (padrão: 'grid').
    zdep : float
        Profundidade limite (m) para classificar terra/oceano (padrão -0.10).
    zmin : float
        Profundidade mínima permitida no WW3 (padrão 2.50 m).
    bathy_source : {'file', 'pygmt'}
        Fonte de batimetria. 'file' usa bathy_file; 'pygmt' baixa via GMT.
    elev_var : str, optional
        Nome da variável de elevação no NetCDF. Auto-detectada se None.
    pygmt_resolution : str
        Resolução para baixar batimetria via pygmt (padrão '01m').

    Returns
    -------
    dict
        Dicionário com os caminhos dos arquivos gerados:
        {'dep': ..., 'mask': ..., 'obs': ..., 'meta': ...}

    Examples
    --------
    >>> from geosushi.waves.gera_reg_ww3grid import gera_grade_regular
    >>> gera_grade_regular(
    ...     lon_min=-46.0, lon_max=-38.0,
    ...     lat_min=-26.0, lat_max=-18.0,
    ...     dx=0.05, dy=0.05,
    ...     bathy_file='/data/gebco_2023.nc',
    ...     output_dir='saida/grade_rio',
    ...     prefix='rio',
    ... )
    """
    print("\n" + "=" * 70)
    print(" GEOSUSHI — GERAÇÃO DE GRADE REGULAR WW3")
    print("=" * 70)

    os.makedirs(output_dir, exist_ok=True)

    # ── Grade WW3 ──────────────────────────────────────────────────────────
    nx = int(round((lon_max - lon_min) / dx)) + 1
    ny = int(round((lat_max - lat_min) / dy)) + 1

    lon_ww3 = lon_min + np.arange(nx) * dx
    lat_ww3 = lat_min + np.arange(ny) * dy

    print(f"\n→ Grade WW3 regular")
    print(f"  Domínio : lon [{lon_ww3[0]:.4f}, {lon_ww3[-1]:.4f}]°  "
          f"lat [{lat_ww3[0]:.4f}, {lat_ww3[-1]:.4f}]°")
    print(f"  Tamanho : {nx} × {ny}  (lon × lat)")
    print(f"  Espaçamento: {dx:.6f}° × {dy:.6f}°")

    # ── Batimetria ─────────────────────────────────────────────────────────
    print("\n→ Carregando batimetria")
    bathy_da = load_bathymetry(
        bathy_file=bathy_file,
        lon_min=lon_min, lon_max=lon_max,
        lat_min=lat_min, lat_max=lat_max,
        source=bathy_source,
        elev_var=elev_var,
        pygmt_resolution=pygmt_resolution,
    )

    print("\n→ Interpolando para a grade WW3")
    elevation = interpolate_to_ww3_grid(bathy_da, lon_ww3, lat_ww3)
    print(f"  Elevação: [{elevation.min():.1f}, {elevation.max():.1f}] m")

    # ── Máscara ────────────────────────────────────────────────────────────
    print("\n→ Calculando máscara terra/oceano")
    mask = make_mask(elevation, zdep=zdep)
    n_ocean = int((mask == 1).sum())
    n_land = int((mask == 0).sum())
    pct = 100.0 * n_ocean / (nx * ny)
    print(f"  Células oceano : {n_ocean}  ({pct:.1f}%)")
    print(f"  Células terra  : {n_land}")

    if n_ocean == 0:
        print("\n  AVISO: Todas as células foram classificadas como terra!")
        print("  Verifique os limites do domínio e a fonte de batimetria.")

    # ── Escrita dos arquivos ───────────────────────────────────────────────
    print("\n→ Escrevendo arquivos de saída")

    dep_file  = os.path.join(output_dir, f'{prefix}.dep')
    mask_file = os.path.join(output_dir, f'{prefix}.mask')
    obs_file  = os.path.join(output_dir, f'{prefix}.obs')
    meta_file = os.path.join(output_dir, f'{prefix}.meta')

    print(f"  {dep_file} …",  end=' ', flush=True)
    write_dep(dep_file, elevation, nx, ny)
    print("ok")

    print(f"  {mask_file} …", end=' ', flush=True)
    write_mask(mask_file, mask, nx, ny)
    print("ok")

    print(f"  {obs_file} …",  end=' ', flush=True)
    write_obs(obs_file, nx, ny)
    print("ok")

    print(f"  {meta_file} …", end=' ', flush=True)
    write_meta(meta_file, nx, ny, dx, dy, lon_ww3[0], lat_ww3[0], prefix,
               zdep=zdep, zmin=zmin)
    print("ok")

    print("\n" + "=" * 70)
    print(" GRADE GERADA COM SUCESSO")
    print("=" * 70)
    print(f"\n  Diretório de saída : {os.path.abspath(output_dir)}/")
    print(f"  Grade             : {nx} × {ny}  (NX × NY)")
    print(f"  Faixa batimetria  : [{elevation.min():.1f}, {elevation.max():.1f}] m")
    print(f"  Células oceano    : {n_ocean}  |  terra: {n_land}\n")

    return {
        'dep':  dep_file,
        'mask': mask_file,
        'obs':  obs_file,
        'meta': meta_file,
    }
