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

    return Flight(df_fl, attrs=flight_attrs)

