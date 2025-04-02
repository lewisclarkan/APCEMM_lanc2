
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import yaml
from subprocess import call
import os

from pycontrails import Flight

from src.aircraft import set_flight_parameters, clean_flight_data, get_aircraft_properties
from src.generate_yaml import generate_yaml_d
from src.geodata import open_dataset, advect, get_albedo, get_temperature_and_clouds_met
from src.sampling import generateDfSamples
from src.radiative_forcing import read_apcemm_data, apce_data_struct, calc_sample
from src.file_management import write_output_header, write_output

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
    #n_samples = 100
    #n_flights = 100

    # TODO: read in multiple pickled files and combine them
    df = pd.read_pickle("flight_data/flightlist_20190101_20190131.pkl")

    # Randomise the samples
    """df = df.sample(frac=1)"""

    # Generate the samples and save them to samples.csv
    #df_samples = generateDfSamples(df, n_samples, n_flights)
    #df_samples.to_pickle("samples/samples.pkl")

    df_samples = pd.read_pickle("samples/samples.pkl")
    df_aircraft = pd.read_csv('flight_data/aircraft.csv', index_col = 0)
    df_samples = clean_flight_data(df_samples,df_aircraft)

    # Sort values by time
    df_samples_by_time = df_samples.sort_values("time")

    ######################################################################################################################
    #                                             Meteorology data module                                                # 
    ######################################################################################################################

    write_output_header()

    met_albedo = get_albedo('gribs/albedo.grib')


    for i in range(62, 65): #len(df_samples_by_time)):

        identifier = i
        sample = df_samples_by_time.iloc[i,:]

        # Download the datasets from CDS and open them
        print("Downloading and opening dataset...\n")
        met = open_dataset(sample)
        met_temp = get_temperature_and_clouds_met(sample)
        
    ######################################################################################################################
    #                                           Aircraft performance module                                              # 
    ######################################################################################################################

        # Create the pycontrails flight object 
        altitude = 10900
        fl = set_flight_parameters(sample, df_aircraft, altitude, i)

    ######################################################################################################################
    #                                     Advection model and input files module                                         # 
    ######################################################################################################################

        # Run DryAdvection model and generate .nc input ds and output to file

        try: os.makedirs("mets")
        except: pass
        try:  os.makedirs("yamls")
        except: pass

        print("Running DryAdvection model...\n")
        ds, ds_temp, pressure, temperature, flag = advect(met, met_temp, fl)

        if flag == False:

            try: os.remove(f"mets/input{i}.nc")
            except FileNotFoundError: pass
            try: os.remove(f"mets/input_temp{i}.nc")
            except FileNotFoundError: pass

            ds.to_netcdf(f"mets/input{i}.nc")
            ds_temp.to_netcdf(f"mets/input_temp{i}.nc")

            properties_dict = get_aircraft_properties(sample, df_aircraft, temperature)

            # Generate the .yaml input dictionary and output to file
            d = generate_yaml_d(identifier, sample, fl, float(pressure/100), properties_dict)
            with open(f"yamls/input{i}.yaml", "w") as yaml_file:
                yaml.dump(d, yaml_file, default_flow_style=False, sort_keys=False)

    ######################################################################################################################
    #                                               APCEMM module                                                        # 
    ######################################################################################################################

            try: os.makedirs("APCEMM_results")
            except: pass

            apcemm_file_path = "../../build/APCEMM"
            call(["./../../build/APCEMM", f"yamls/input{i}.yaml"])#

            print("APCEMM done")

    ######################################################################################################################
    #                                          Radiative forcing module                                                  # 
    ######################################################################################################################

            apce_data = read_apcemm_data(f"APCEMM_results/APCEMM_out_{i}/")

            try:
                with open(f"APCEMM_results/APCEMM_out_{i}/status_case0", "r") as f:
                    status = f.readline()

                if (str(status) == "NoPersistence\n"):
                    status = "No persistence "
                    print("No persistence\n")
                    j_per_m = 0
                    age = 0
                else:
                    j_per_m, age = calc_sample(apce_data, sample, met_albedo, ds_temp)
                    status = "Contrail formed"

            except FileNotFoundError:
                status = "Error          "
                j_per_m = 0
                age = 0
                continue

        else:
            status = "Error          "
            j_per_m = 0
            age = 0

        write_output(sample, j_per_m, age, status)
    