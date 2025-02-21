import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import xarray as xr
import random
import time

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

    # TO-DO: see if there is a better way to do this, especially altitude
    df["latitude"] = np.array([flight["latitude_1"], flight["latitude_2"]])
    df["longitude"] = np.array([flight["longitude_1"], flight["longitude_2"]])
    df["time"] = np.array([flight["firstseen"], flight["lastseen"]])
    df["altitude_ft"] = np.array([35_000.0, 35_000.0])

    return Flight(df, attrs=flight_attrs)
