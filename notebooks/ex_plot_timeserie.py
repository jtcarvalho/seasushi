
import sys
import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np

# Allow importing geosushi from the src layout
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from geosushi.waves.io_pnboia import load_pnboia
from geosushi.waves.io_ww3 import extract_ww3_at_point

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')

PNBOIA_FILE = os.path.join(DATA_DIR, 'pnboia_fortaleza.csv')
WW3_FILE    = os.path.join(DATA_DIR, 'norte_2018.nc')

# PNBOIA Fortaleza buoy position (BSce — Boia Semiestacionária do Ceará)
BUOY_LAT = -3.7
BUOY_LON = -38.5

START_DATE = '2018-01-01'
END_DATE   = '2018-05-20'

# ---------------------------------------------------------------------------
# Load PNBOIA observations
# ---------------------------------------------------------------------------
obs = load_pnboia(
    PNBOIA_FILE,
    variables=['Hs', 'Tp', 'Dm'],
    start_date=START_DATE,
    end_date=END_DATE,
)

# ---------------------------------------------------------------------------
# Load WW3 model at the nearest grid point to the buoy
# ---------------------------------------------------------------------------
ww3 = extract_ww3_at_point(
    WW3_FILE,
    lat=BUOY_LAT,
    lon=BUOY_LON,
    variables=['hs'],
    start_date=START_DATE,
    end_date=END_DATE,
)
ww3_time = pd.to_datetime(ww3['time'].values)
ww3_hs   = ww3['hs'].values.squeeze()

# ---------------------------------------------------------------------------
# Align on common timestamps
# ---------------------------------------------------------------------------
obs_series  = obs['Hs'].dropna()
ww3_series  = pd.Series(ww3_hs, index=ww3_time).dropna()

common_idx  = obs_series.index.intersection(ww3_series.index)
obs_common  = obs_series.loc[common_idx]
ww3_common  = ww3_series.loc[common_idx]

# ---------------------------------------------------------------------------
# Time-series plot
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(14, 5))

ax.plot(obs_common.index, obs_common.values,
        label='PNBOIA Fortaleza', color='steelblue',
        marker='o', markersize=2, linestyle=':', linewidth=1.2)
ax.plot(ww3_common.index, ww3_common.values,
        label='WW3 norte_2018', color='firebrick',
        linestyle='-', linewidth=1.4)

ax.set_xlabel('Data')
ax.set_ylabel('Hs [m]')
ax.set_title(f'Altura Significativa — PNBOIA Fortaleza vs WW3\n'
             f'{START_DATE} a {END_DATE}')
ax.legend()
ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%b'))
ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('timeserie_fortaleza_ww3.png', dpi=150, bbox_inches='tight')
plt.show()
print("Plot salvo em timeserie_fortaleza_ww3.png")
