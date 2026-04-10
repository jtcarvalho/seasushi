import xarray as xr
import numpy as np

def load_oscar(file_path, time_index):
    ds = xr.open_dataset(file_path)

    u = ds.u[time_index, :, :].values
    v = ds.v[time_index, :, :].values    
    mag = np.sqrt(u**2 + v**2)
    dir = np.arctan2(v, u) * (180 / np.pi) 
    lon = ds.lon.values
    lat = ds.lat.values

    # # Load wind data if available
    # wnd = ds.wnd[time_index, 0].values if 'wnd' in ds else None
    # wnddir = ds.wnddir[time_index, 0].values if 'wnddir' in ds else None

    ds.close()

    return u,v,mag,dir,lon,lat




# from scipy.interpolate import griddata
# arq  = xr.open_dataset('/work/cmcc/jc11022/projects/geosushi/data/oscar_currents_interim_20260120.nc')


# int=arq.variables['int'][:]
# int1=int[0,:,:]

# lon=arq.variables['lon'][:]
# lat=arq.variables['lat'][:]

# x,y=np.meshgrid(lon,lat)

# arq2=nc.Dataset('/home/jonas/cou_09a12abr2014.nc','r')

# x2=arq2.variables['lon_rho'][:]
# y2=arq2.variables['lat_rho'][:]


# x=x.ravel()              
# #x=list(x[x!=np.isnan])
# y=y.ravel()              
# #y=list(y[y!=np.isnan])
# int1=int1.ravel()
# #int1=list(int1[int1!=np.isnan])

# scipy.interpolate.RectBivariateSpline(x2, y2,int1 , bbox=[None, None, None, None], kx=3, ky=3, s=0)[source]



# z1 = griddata(  (x, y), int1, (x2, y2), method='linear' )  #melhor
# z2 = griddata(  (x, y), int1, (x2, y2), method='nearest' ) #ruim, pois pega o nan da costa
# z3 = griddata(  (x, y), int1, (x2, y2), method='cubic' )   #ruim

# plt.figure(1)
# plt.contourf(z1)
# plt.figure(2)
# plt.pcolormesh(x2,y2,z2,shading='gouraud')
# plt.contourf(z2)
# plt.figure(3)
# plt.contourf(z3)



# #x = np.loadtxt('data.txt',usecols=[0])
# #y = np.loadtxt('data.txt',usecols=[1])
# #s = np.loadtxt('data.txt',usecols=[2])

# lim2=x2.max()
# lim1=x2.min()
# xi=np.linspace(lim1,lim2,361)

# lim2=y2.max()
# lim1=y2.min()
# yi=np.linspace(lim1,lim2,271)

# xi2,yi2=np.meshgrid(xi,yi)





