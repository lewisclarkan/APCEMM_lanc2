import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import xarray as xr
import zarr

from pycontrails import Flight
from pycontrails.models.dry_advection import DryAdvection
from pycontrails.core import met_var, GeoVectorDataset, models
from pycontrails.physics import constants, thermo, units
from pycontrails.datalib.ecmwf import ERA5ModelLevel
from pycontrails import MetDataset
from pycontrails.models.apcemm import utils


def open_dataset_from_sample(sample):

    # Takes an input sample point of [Index, Longitude, Latitude, Altitude, Time, Aircraft Type]

    s_index, s_longitude, s_latitude, s_altitude, s_time, s_type = sample

    max_life = 7

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

def run_DryAdvection_and_met(sample, met):

    ################################################
    # Let's artificially set the flight parameters #
    ################################################

    dt_input_met = np.time_delta64 = np.timedelta64(1, "h")

    flight_attrs = {
        "flight_id": "test",
        "true_airspeed": 230,
        "thrust": 0.22, 
        "nvpm_ei_n": 1.897462e15, 
        "aircraft_type": "E190",
        "wingspan": 48,
        "n_engine": 2,
    }

    df_fl = pd.DataFrame()
    df_fl["longitude"]          = np.linspace(sample["longitude"], sample["longitude"], 1)
    df_fl["latitude"]           = np.linspace(sample["latitude"], sample["latitude"], 1)
    df_fl["altitude"]           = np.linspace(10900, 10900, 1)
    df_fl["engine_efficiency"]  = np.linspace(0.34, 0.34, 1)
    df_fl["fuel_flow"]          = np.linspace(2.1, 2.1, 1)  # kg/s
    df_fl["aircraft_mass"]      = np.linspace(154445, 154445, 1)  # kg
    df_fl["time"]               = pd.date_range(sample["time"], sample["time"], periods=1)

    fl = Flight(df_fl, attrs=flight_attrs)

    ################################################
    #                                              #
    ################################################

    dt_integration = np.timedelta64(2, 'm')
    max_age = np.timedelta64(6, 'h')

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


    path = "mets/input.nc"
    ds.to_netcdf(path)

if __name__ == "__main__":
    
    df = pd.read_csv("samples/samples.csv", sep='\t')
    df_samples_by_time = df.sort_values('time')
    sample = df_samples_by_time.iloc[0,1:]

    sample['index'] = int(sample['index'])
    sample['longitude'] = float(sample['longitude'])
    sample['latitude'] = float(sample['latitude'])
    sample['altitude'] = float(sample['altitude'])
    sample['time'] = np.datetime64(sample['time'])

    met = open_dataset_from_sample(sample.values)

    print(met["altitude"])

    run_DryAdvection_and_met(sample, met)