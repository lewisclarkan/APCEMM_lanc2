import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import xarray as xr
import zarr

from pycontrails import Flight
from pycontrails.datalib.ecmwf import ERA5
from pycontrails.models.cocip import Cocip
from pycontrails.models.apcemm import APCEMM
from pycontrails.models.dry_advection import DryAdvection
from pycontrails.core import met_var, GeoVectorDataset, models, vector
from pycontrails.physics import constants, thermo, units
from pycontrails.datalib.ecmwf import ERA5ARCO
from pycontrails.datalib.ecmwf import ERA5, ERA5ModelLevel

from pycontrails import Flight
from pycontrails.datalib.ecmwf import ERA5
from pycontrails.models.cocip import Cocip
from pycontrails.models.dry_advection import DryAdvection
from pycontrails.models.humidity_scaling import ConstantHumidityScaling
from pycontrails.datalib.ecmwf import ERA5ModelLevel
from pycontrails import MetDataset


def open_dataset_from_sample(sample):

    # Takes an input sample point of [Index, Longitude, Latitude, Altitude, Time, Aircraft Type]

    s_index, s_longitude, s_latitude, s_altitude, s_time, s_type = sample

    max_life = 6

    time = (str(s_time), str(s_time + np.timedelta64(max_life, 'h')))

    era5ml = ERA5ModelLevel(
        time=time,
        variables=("t", "q", "u", "v", "w", "ciwc"),
        grid=1,  # horizontal resolution, 0.25 by default
        model_levels=range(70, 91),
        pressure_levels=np.arange(170, 400, 10),
    )
    met_t = era5ml.open_metdataset()

    geopotential = met_t.data.coords["altitude"].data * 9.8 

    ds = met_t.data.assign_coords(geopotential=("geopotential", geopotential))

    met = MetDataset(ds)

    return met

if __name__ == "__main__":

    sample = np.array([0, 47.750093016881074, 32.08399195343702, 10668.0, np.datetime64('2019-05-01T00'), 'A359'])

    met = open_dataset_from_sample(sample)

    met