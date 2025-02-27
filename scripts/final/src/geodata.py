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

    geopotential1 = met_t.data.coords["altitude"].data * 9.8 
    geopotential2 = np.repeat(geopotential1, [len(met_t.data.coords["longitude"])*len(met_t.data.coords["latitude"])*len(met_t.data.coords["time"])])
    geopotential3 = np.reshape(geopotential2, (len(met_t.data.coords["longitude"]),len(met_t.data.coords["latitude"]),len(met_t.data.coords["level"]),len(met_t.data.coords["time"])))
    ds = met_t.data.assign(geopotential=(met_t.data["air_temperature"].dims, geopotential3))

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

    time = target_time
    lon = interp_lon
    lat = interp_lat
    azimuth = interp_az
    altitude = altitude
    humidity_scaling = None
    dz_m = 200

    def normal_wind_shear(u_hi,u_lo, v_hi,v_lo,azimuth,dz: float):    
        du_dz = (u_hi - u_lo) / dz
        dv_dz = (v_hi - v_lo) / dz
        az_radians = units.degrees_to_radians(azimuth)
        sin_az = np.sin(az_radians)
        cos_az = np.cos(az_radians)
        return sin_az * dv_dz - cos_az * du_dz
    
    # generate_apcemm_input_met

    # Ensure that altitudes are sorted ascending
    altitude = np.sort(altitude)

    # Check for required fields in met
    vars = (met_var.AirTemperature,
            met_var.SpecificHumidity,
            met_var.Geopotential,
            met_var.EastwardWind,
            met_var.NorthwardWind,
            met_var.VerticalVelocity)

    met.ensure_vars(vars)
    met.standardize_variables(vars)

    # Flatten input arrays
    time = time.ravel()
    lon = lon.ravel()
    lat = lat.ravel()
    azimuth = azimuth.ravel()
    altitude = altitude.ravel()

    # Estimate pressure levels close to target altitudes
    # (not exact because this assumes the ISA temperature profile)
    pressure = units.m_to_pl(altitude) * 1e2

    # Broadcast to required shape and create vector for intial interpolation
    # onto original pressure levels at target horizontal location
    shape = (time.size, altitude.size)
    time = np.broadcast_to(time[:, np.newaxis], shape).ravel()
    lon = np.broadcast_to(lon[:, np.newaxis], shape).ravel()
    lat = np.broadcast_to(lat[:, np.newaxis], shape).ravel()
    azimuth = np.broadcast_to(azimuth[:, np.newaxis], shape).ravel()
    level = np.broadcast_to(pressure[np.newaxis, :] / 1e2, shape).ravel()

    vector = GeoVectorDataset(data = {"azimuth": azimuth}, longitude=lon, latitude=lat, level=level, time=time)

    # Downselect met beofre interpolation
    met = vector.downselect_met(met)

    # Interpolate meterology data onto vector
    scale_humidity = humidity_scaling is not None and "specfic_humidity" not in vector
    for met_key in (
        "air_temperature",
        "eastward_wind",
        "geopotential",
        "northward_wind",
        "specific_humidity",
        "lagrangian_tendency_of_air_pressure",
    ):

        #models.interpolate_met(met, vector, met_key, None, method='nearest')
        print(vector)
        vector[met_key] = vector.intersect_met(met[met_key])


    # Interpolate winds at lower level for shear calculation
    air_pressure_lower = thermo.pressure_dz(vector["air_temperature"], vector.air_pressure, dz_m)
    lower_level = air_pressure_lower / 100.0
    for met_key in ("eastward_wind", "northward_wind"):
        vector_key = f"{met_key}_lower"
        models.interpolate_met(met, vector, met_key, vector_key, level=lower_level)

    # Apply humidity scaling
    if scale_humidity and humidity_scaling is not None:
        humidity_scaling.eval(vector, copy_source=False)

    # Compute RHi and segment-normal shear
    vector.setdefault(
        "rhi",
        thermo.rhi(vector["specific_humidity"], vector["air_temperature"], vector.air_pressure),
    )

    vector.setdefault(
        "normal_shear",
        normal_wind_shear(
                vector["eastward_wind"],
                vector["eastward_wind_lower"],
                vector["northward_wind"],
                vector["northward_wind_lower"],
                vector["azimuth"],
                dz_m,            
        ),
    )

    # Reshape interpolated fields to (time, level)

    nlev = altitude.size
    ntime = len(vector) // nlev
    shape = (ntime, nlev)
    time = np.unique(vector["time"])
    time = (time - time[0]) / np.timedelta64(1, "h")
    temperature = vector["air_temperature"].reshape(shape)
    qv = vector["specific_humidity"].reshape(shape)
    z = vector["geopotential"].reshape(shape)
    rhi = vector["rhi"].reshape(shape)
    shear = vector["normal_shear"].reshape(shape)
    shear[:, -1] = shear[:, -2]
    omega = vector["lagrangian_tendency_of_air_pressure"].reshape(shape)
    virtual_temperature = temperature * (1 + qv / constants.epsilon) / (1 + qv)
    density = pressure[np.newaxis, :] / (constants.R_d * virtual_temperature)
    w = -omega / (density * constants.g)

    # Interpolate fields to target altitudes profile-by-profile
    # to obtain 2D arrays with dimensions (time, altitude)
    temperature_on_z = np.zeros(shape, dtype=temperature.dtype)
    rhi_on_z = np.zeros(shape, dtype=rhi.dtype)
    shear_on_z = np.zeros(shape, dtype=shear.dtype)
    w_on_z = np.zeros(shape, dtype=w.dtype)

    # Fields should already be on pressure levels close to target
    # altitudes, so this just uses linear interpolation and const.
    # extrapolation on field expected by APCEMM. 
    # NaNs are preserved at the start and end of interpolated
    # profiles but are removed in interiors.

    def interp(z: np.ndarray, z0: np.ndarray, f0: np.ndarray) -> np.ndarray:
        # mask nans
        mask = np.isnan(z0) | np.isnan(f0)
        if np.all(mask):
            msg = (
                "Found all-NaN profile during APCEMM meterology input file creation. "
                "MetDataset may have insufficient spatiotemporal coverage."
            )
            raise ValueError(msg)
        z0 = z0[~mask]
        f0 = f0[~mask]

        # interpolate
        #assert np.all(np.diff(z0) > 0)  # expect increasing altitudes
        fi = np.interp(z, z0, f0, left=f0[0], right=f0[-1])

        # restore nans at start and end of profile
        if mask[0]:  # nans at top of profile
            fi[z > z0.max()] = np.nan
        if mask[-1]:  # nans at end of profile
            fi[z < z0.min()] = np.nan
        return fi

    # The manual for loop is unlikely to be a bottleneck since a
    # substantial amount of work is done within each iteration.
    for i in range(ntime):

        print(i)

        #print(altitude)
        #print(temperature[i, :])
        #print(z[i, :])

        temperature_on_z[i, :] = interp(altitude, z[i, :], temperature[i, :])

        rhi_on_z[i, :] = interp(altitude, z[i, :], rhi[i, :])

        shear_on_z[i, :] = interp(altitude, z[i, :], shear[i, :])
        w_on_z[i, :] = interp(altitude, z[i, :], w[i, :])

    # APCEMM also requires initial pressure profile
    pressure_on_z = interp(altitude, z[0, :], pressure)

    # Create APCEMM input dataset.
    # Transpose require because APCEMM expects (altitude, time) arrays.

    ds = xr.Dataset(
        data_vars={
            "pressure": (("altitude",), pressure_on_z.astype("float32") / 1e2, {"units": "hPa"}),
            "temperature": (
                ("altitude", "time"),
                temperature_on_z.astype("float32").T,
                {"units": "K"},
            ),
            "relative_humidity_ice": (
                ("altitude", "time"),
                1e2 * rhi_on_z.astype("float32").T,
                {"units": "percent"},
            ),
            "shear": (("altitude", "time"), shear_on_z.astype("float32").T, {"units": "s**-1"}),
            "w": (("altitude", "time"), w_on_z.astype("float32").T, {"units": "m s**-1"}),
        },
        coords={
            "altitude": ("altitude", altitude.astype("float32") / 1e3, {"units": "km"}),
            "time": ("time", time, {"units": "hours"}),
        },
    )

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

    run_DryAdvection_and_met(sample, met)