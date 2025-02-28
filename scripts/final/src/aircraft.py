import numpy as np
import pandas as pd
from pycontrails import Flight


def set_flight_parameters(sample):

    # FOR DEVELOPMENT ONLY
    flight_attrs = {
        "flight_id": "test",
        "true_airspeed": 230,
        "thrust": 0.22, 
        "nvpm_ei_n": 1.897462e15, 
        "aircraft_type": sample["aircraft type"],
        "wingspan": 48,
        "n_engine": 2,
    }

    df = pd.DataFrame()
    df["longitude"]          = np.linspace(sample["longitude"], sample["longitude"], 1)
    df["latitude"]           = np.linspace(sample["latitude"], sample["latitude"], 1)
    df["altitude"]           = np.linspace(10900, 10900, 1)
    df["engine_efficiency"]  = np.linspace(0.34, 0.34, 1)
    df["fuel_flow"]          = np.linspace(2.1, 2.1, 1)  # kg/s
    df["aircraft_mass"]      = np.linspace(154445, 154445, 1)  # kg
    df["time"]               = pd.date_range(sample["time"], sample["time"], periods=1)

    return Flight(df, attrs=flight_attrs)

