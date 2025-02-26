import xarray
import numpy as np

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import zarr
#import metview as mv


#import metpy.calc
#from metpy.units import units

from pycontrails.datalib.ecmwf.model_levels import model_level_pressure

from pycontrails import Flight
from pycontrails.datalib.ecmwf import ERA5
from pycontrails.models.cocip import Cocip
from pycontrails.models.apcemm import APCEMM
from pycontrails.models.dry_advection import DryAdvection
from pycontrails.core import met_var, GeoVectorDataset, models, vector
from pycontrails.physics import constants, thermo, units
from pycontrails.datalib.ecmwf import ERA5ARCO
from pycontrails.datalib.ecmwf import ERA5ModelLevel
from pycontrails import MetDataArray, MetDataset, MetVariable

from pycontrails import Flight
from pycontrails.datalib.ecmwf import ERA5
from pycontrails.models.cocip import Cocip
from pycontrails.models.dry_advection import DryAdvection
from pycontrails.models.humidity_scaling import ConstantHumidityScaling

plt.rcParams["figure.figsize"] = (10, 6)

ds = xarray.open_zarr(
    'gs://gcp-public-data-arco-era5/ar/model-level-1h-0p25deg.zarr-v1',
    chunks=None,
    storage_options=dict(token='anon'),
)

time_start = np.datetime64('2020-02-25T01:00')
time_end   = np.datetime64('2020-02-25T04:30')

ds = ds.sel(time=slice(time_start, time_end))
ds = ds.sel(longitude=slice(45, 50))
ds = ds.sel(latitude=slice(90, 80))
ds = ds.drop_vars(("divergence", "ozone_mass_mixing_ratio", "specific_rain_water_content", "specific_snow_water_content", "vorticity"))

ds = ds.rename({"temperature":"air_temperature","u_component_of_wind":"eastward_wind","v_component_of_wind":"northward_wind", "vertical_velocity":"lagrangian_tendency_of_air_pressure"})

print(ds)

ds2 = model_level_pressure()

met = MetDataset(ds)

flight_attrs = {
    "flight_id": "test",
    # set constants along flight path
    "true_airspeed": 226.099920796651,  # true airspeed, m/s
    "thrust": 0.22,  # thrust_setting
    "nvpm_ei_n": 1.897462e15,  # non-volatile emissions index
    "aircraft_type": "E190",
    "wingspan": 48,  # m
    "n_engine": 2,
}

# Example flight
df = pd.DataFrame()
df["longitude"] = np.linspace(47, 47, 1)
df["latitude"] = np.linspace(85, 85, 1)
df["altitude"] = np.linspace(10900, 10900, 1)
df["engine_efficiency"] = np.linspace(0.34, 0.34, 1)
df["fuel_flow"] = np.linspace(2.1, 2.1, 1)  # kg/s
df["aircraft_mass"] = np.linspace(154445, 154445, 1)  # kg
df["time"] = pd.date_range("2020-02-25T01:00", "2020-02-25T01:00", periods=1)

fl = Flight(df, attrs=flight_attrs)

dt_integration = pd.Timedelta(minutes=2)
max_age = pd.Timedelta(hours=6)

params = {
    "dt_integration": dt_integration,
    "max_age": max_age,
    "depth": 1.0,  # initial plume depth, [m]
    "width": 1.0,  # initial plume width, [m]
}
dry_adv = DryAdvection(met, params)
dry_adv_df = dry_adv.eval(fl).dataframe

ax = plt.axes()

ax.scatter(fl["longitude"], fl["latitude"], s=3, color="red", label="Flight path")
ax.scatter(
    dry_adv_df["longitude"], dry_adv_df["latitude"], s=0.1, color="purple", label="Plume evolution"
)
ax.legend()
ax.set_title("Flight path and plume evolution under dry advection")

plt.show()