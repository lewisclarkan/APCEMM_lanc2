import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import xarray as xr
import cartopy.crs as ccrs

import random
import time

import dask.config

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
from alive_progress import alive_bar

from src.sampling import calcTotalDistance, samplePoint, generateFlight
from src.geodata import open_dataset_from_sample

if __name__ == "__main__":

    print("Starting final.py...\n")

    # Read in dataframe
    """df = pd.read_csv('flight_data/flightlist_20190101_20190131.csv.gz')
    df.drop('number', axis=1, inplace=True)
    df.drop('registration', axis=1, inplace=True)
    df.drop('icao24', axis=1, inplace=True)
    df.to_pickle('flight_data/flightlist_20190101_20190131.pkl')"""

    ######################################################################################################################
    #                                                Sampling module                                                     # 
    ######################################################################################################################

    # Set the number of samples and flights
    n_samples = 100
    n_flights = 100

    # TODO: read in multiple picked files and combine them
    df = pd.read_pickle('flight_data/flightlist_20190101_20190131.pkl')

    # Randomise the samples
    """df = df.sample(frac=1)"""

    samples = np.empty((n_samples,2),int)
    flights = []

    print("Converting to list of flight objects...")
    with alive_bar(n_flights) as bar:
        for i in range(0, n_flights):
            flights.append(generateFlight(df.iloc[i]))
            bar()

    total_distance = calcTotalDistance(flights)

    print("\nTaking samples...")
    with alive_bar(n_samples) as bar:
        for i in range(0, n_samples):
            samples[i] = samplePoint(flights, total_distance)
            bar()

    sample_indices = np.arange(0, n_samples, 1)

    longitudes   = np.empty(n_samples)
    latitudes    = np.empty(n_samples)
    altitudes    = np.empty(n_samples)
    times        = np.empty(n_samples, dtype = 'datetime64[s]')
    aircrafts    = np.empty(n_samples, dtype = object)

    print("\nDetermining sample characteristics...")
    with alive_bar(n_samples) as bar:
        for i in range(0, n_samples):
            longitudes[i]   = flights[samples[i][0]]['longitude'][samples[i][1]]
            latitudes[i]    = flights[samples[i][0]]['latitude'][samples[i][1]]
            altitudes[i]    = flights[samples[i][0]]['altitude'][samples[i][1]]
            times[i]        = flights[samples[i][0]]['time'][samples[i][1]]
            aircrafts[i]    = flights[samples[i][0]].attrs['aircraft_type']
            bar()

    print(f"\nTotal distance flown in dataset was {total_distance/1000:.2f} km.\n")


    df_samples = pd.DataFrame(data = np.array([sample_indices, longitudes, latitudes, altitudes, times, aircrafts]).transpose(), columns = ["index", "longitude", "latitude", "altitude", "time", "aircraft type"])
    df_samples.to_csv('samples/samples.csv', sep='\t')

    df_samples_by_time = df_samples.sort_values('time')

    ######################################################################################################################
    #                                           Meteo data and advection                                                 # 
    ######################################################################################################################

    sample = df_samples_by_time.iloc[0,:]

    print("Downloading and opening dataset...\n")
    met = open_dataset_from_sample(sample)

    ################################################
    # Let's artificially set the flight parameters #
    ################################################

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

    dt_integration = pd.Timedelta(minutes=2)
    max_age = pd.Timedelta(hours=6)

    params = {
        "dt_integration": dt_integration,
        "max_age": max_age,
        "depth": 1.0,  # initial plume depth, [m]
        "width": 1.0,  # initial plume width, [m]
    }

    print("Running DryAdvection model...\n")
    dry_adv = DryAdvection(met, params)
    dry_adv_df = dry_adv.eval(fl).dataframe

    ax = plt.axes()

    ax.scatter(fl["longitude"], fl["latitude"], s=3, color="red", label="Flight path")
    ax.scatter(
        dry_adv_df["longitude"], dry_adv_df["latitude"], s=0.1, color="purple", label="Plume evolution"
    )
    ax.legend()
    ax.set_title("Flight path and plume evolution under dry advection")

    plt.savefig("test.png")