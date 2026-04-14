import numpy as np
from geosushi.ocean import load_oscar
import matplotlib.pyplot as plt

u, v, mag, dir, lon, lat  = load_oscar("/work/cmcc/jc11022/projects/geosushi/data/oscar_currents_interim_20260120.nc", 0)

print(mag.shape)
print(lon.shape)
print(lat.shape)

lons,lats = np.meshgrid(lon,lat)

print(lons.shape)
print(lats.shape)
plt.figure()
plt.contourf(lons,lats,mag)
plt.savefig('teste.png')



