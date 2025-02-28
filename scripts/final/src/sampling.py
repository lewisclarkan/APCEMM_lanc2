import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import xarray as xr
import random
import time

from alive_progress import alive_bar
from pycontrails import Flight

def calcTotalDistance(flights) -> float:
    """Calculates total distance flown"""
    
    total_distance = 0 

    for flight in flights:
        total_distance += flight.length

    return total_distance

def samplePoint(flights, total_distance):
    """Generate a random index i for flight index, and j 
    for segment index. Inputs are a list of pycontrails flight objects"""

    sample_distance = random.randint(0, np.round(total_distance,0))
    cumulative_distance = 0

    flag = True
    i=0; j=0

    while flag:
        if ((sample_distance - cumulative_distance) < flights[i].length):
            flag = False
        else:
            cumulative_distance += flights[i].length
            i += 1

    remaining_dist = sample_distance - cumulative_distance
    flights[i] = flights[i].resample_and_fill('1min')
    lengths = flights[i].segment_length()
    
    flag = True
    while flag:
        if (j == len(lengths)-1):
            flag = False
        elif ((remaining_dist) < lengths[j]):
            flag = False
        else:
            remaining_dist -= lengths[j]
            j += 1

    return [i,j]

def generateFlight(flight):

    flight_attrs = {
        "flight_id":        flight["callsign"],
        "aircraft_type":    flight["typecode"]
    }

    df=pd.DataFrame()

    # TODO: see if there is a better way to do this, especially altitude
    df["latitude"] = np.array([flight["latitude_1"], flight["latitude_2"]])
    df["longitude"] = np.array([flight["longitude_1"], flight["longitude_2"]])
    df["time"] = np.array([flight["firstseen"], flight["lastseen"]])
    df["altitude_ft"] = np.array([35_000.0, 35_000.0])

    return Flight(df, attrs=flight_attrs)

def generateDfSamples(df, n_samples, n_flights):

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

    return df_samples