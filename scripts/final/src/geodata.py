import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import xarray as xr
import zarr
import yaml

#from aircraft import set_flight_parameters

from pycontrails import Flight
from pycontrails.models.dry_advection import DryAdvection
from pycontrails.core import met_var, GeoVectorDataset, models
from pycontrails.physics import constants, thermo, units
from pycontrails.datalib.ecmwf import ERA5ModelLevel
from pycontrails import MetDataset
from pycontrails.models.apcemm import utils

def open_dataset(sample):

    s_index, s_longitude, s_latitude, s_altitude, s_time, s_type = sample

    max_life = 12

    time = (str(s_time), str(s_time + np.timedelta64(max_life, 'h')))

    era5ml = ERA5ModelLevel(
        time=time,
        variables=("t", "q", "u", "v", "w", "ciwc"),
        grid=1,  # horizontal resolution, 0.25 by default
        model_levels=range(70, 91),
        pressure_levels=np.arange(170, 400, 10),
    )
    met_t = era5ml.open_metdataset()

    geopotential = met_t.data.coords["altitude"].data

    temp1 = np.repeat(geopotential, len(met_t.data.coords["time"]))
    temp2 = np.tile(temp1, len(met_t.data.coords["longitude"])*len(met_t.data.coords["latitude"]))

    geopotential_4d = np.reshape(temp2, (len(met_t.data.coords["longitude"]),len(met_t.data.coords["latitude"]),len(met_t.data.coords["level"]),len(met_t.data.coords["time"])))

    ds = met_t.data.assign(geopotential_height=(met_t.data["air_temperature"].dims, geopotential_4d))

    met = MetDataset(ds)

    return met

def advect(met, fl):

    dt_input_met = np.timedelta64(6, "m")

    dt_integration = np.timedelta64(2, 'm')
    max_age = np.timedelta64(12, 'h')

    params = {
        "dt_integration": dt_integration,
        "max_age": max_age,
        "depth": 1.0,  # initial plume depth, [m]
        "width": 1.0,  # initial plume width, [m]
    }

    dry_adv = DryAdvection(met, params)
    dry_adv_df = dry_adv.eval(fl).dataframe
    v_wind = dry_adv_df["v_wind"].values
    level = dry_adv_df["level"].values
    vertical_velocity = dry_adv_df["vertical_velocity"].values
    air_pressure = dry_adv_df["air_pressure"].values
    air_temperature = dry_adv_df["air_temperature"].values
    lon = dry_adv_df["longitude"].values
    age = dry_adv_df["age"].values
    u_wind = dry_adv_df["u_wind"].values
    lat = dry_adv_df["latitude"].values
    azimuth = dry_adv_df["azimuth"].values
    depth = dry_adv_df["depth"].values
    time = dry_adv_df["time"].values

    n_profiles = int(max_age / dt_input_met) + 1
    tick = np.timedelta64(1, "s")
    target_elapsed = np.linspace(
        0, (n_profiles - 1) * dt_input_met / tick, n_profiles
    )
    target_time = time[0] + target_elapsed * tick
    elapsed = (dry_adv_df["time"] - dry_adv_df["time"][0]) / tick

    min_pos = np.min(lon[lon>0], initial = np.inf)
    max_neg = np.max(lon[lon<0], initial=-np.inf)
    if (180 - min_pos) + (180 + max_neg) < 180 and min_pos < np.inf and max_neg > -np.inf:
        lon = np.where(lon > 0, lon - 360, lon)
    interp_lon = np.interp(target_elapsed, elapsed, lon)
    interp_lon = np.where(interp_lon > 180, interp_lon - 360, interp_lon)

    interp_lat = np.interp(target_elapsed, elapsed, lat)
    interp_az = np.interp(target_elapsed, elapsed, azimuth)

    altitude = met["altitude"].values

    ds = utils.generate_apcemm_input_met(
        time=target_time,
        longitude=interp_lon,
        latitude=interp_lat,
        azimuth=interp_az,
        altitude=altitude,
        met=met,
        humidity_scaling=None,
        dz_m=200,
        interp_kwargs={'method':'linear'})

    return ds, air_pressure[0]
