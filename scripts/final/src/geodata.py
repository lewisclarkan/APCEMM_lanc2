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

from pycontrails import Flight
from pycontrails.datalib.ecmwf import ERA5
from pycontrails.models.cocip import Cocip
from pycontrails.models.dry_advection import DryAdvection
from pycontrails.models.humidity_scaling import ConstantHumidityScaling
from pycontrails.datalib.ecmwf import ERA5ModelLevel









def open_dataset_from_sample(sample):

    # Takes an input sample point of [Index, Longitude, Latitude, Altitude, Time, Aircraft Type]

    s_index, s_longitude, s_latitude, s_altitude, s_time, s_type = sample

    #max_life = 1

    time = (str(s_time), str(s_time + np.timedelta64(1, 'h')))

    print(time)

    #time = ("2019-05-01T00", "2019-05-01T12")

    era5 = ERA5ARCO(
        time = time,
        variables = (*Cocip.met_variables, *Cocip.optional_met_variables, *APCEMM.met_variables),
        cachestore=None,
    )
    
    met = era5.open_metdataset()

    return met


    

if __name__ == "__main__":

    sample = np.array([0, 47.750093016881074, 32.08399195343702, 10668.0, np.datetime64('2019-05-01T00'), 'A359'])

    print("Started")

    met = open_dataset_from_sample(sample)

    print("Done")

    print(met)

