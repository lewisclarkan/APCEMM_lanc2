import pandas as pd
import numpy as np
import argparse
import pickle
import xarray as xr

from src.radiative_forcing import read_apcemm_data, apce_data_struct, calc_sample

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Runs the main computation')
    parser.add_argument("--start", required=True, type=int)
    parser.add_argument("--end", required=True, type=int)
    parser.add_argument("--month", required=True, type=int)
    arg = parser.parse_args()

    start_index = arg.start
    end_index = arg.end
    month = arg.month

    ds = xr.load_dataset('gribs/albedo.grib', engine="cfgrib")
    ds = ds.expand_dims({'level':[-1]})

    met_albedo = ds

    df = pd.read_csv(f"outputs/{month}_{start_index}_{end_index}.txt", sep = ",", header = 0)
    
    print(df)

    df.columns = ["Index", "Status", "Latitude", "Longitude", "Altitude", "Time", "Age"]

    for i in range(1, df.shape[0]):

        apce_data_file_name = (f"results/month_{month}/{start_index}_{end_index}/sample_{i}_apce_data.pkl")
        sample_file_name = (f"results/month_{month}/{start_index}_{end_index}/sample_{i}_sample.pkl")
        ds_temp_file_name = (f"results/month_{month}/{start_index}_{end_index}/sample_{i}_ds_temp.pkl")

        path_start = (f"results/month_{month}/{start_index}_{end_index}/sample_{i}")

        if df.iloc[i, 1] == "Contrail_formed":
            with open(apce_data_file_name, "rb") as f:
                apce_data = pickle.load(f)
            with open(sample_file_name, "rb") as f:
                sample = pickle.load(f)
            with open(ds_temp_file_name, "rb") as f:
                ds_temp = pickle.load(f)

            j_per_m, age = calc_sample(apce_data, sample, met_albedo, ds_temp, path_start)

            print(j_per_m, age)