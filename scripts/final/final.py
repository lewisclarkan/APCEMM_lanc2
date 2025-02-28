
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import yaml

from pycontrails import Flight

from src.aircraft import set_flight_parameters
from src.generate_yaml import generate_yaml_d
from src.geodata import open_dataset, advect
from src.sampling import generateDfSamples


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

    # TODO: read in multiple pickled files and combine them
    df = pd.read_pickle('flight_data/flightlist_20190101_20190131.pkl')

    # Randomise the samples
    """df = df.sample(frac=1)"""

    # Generate the samples and save them to samples.csv
    df_samples = generateDfSamples(df, n_samples, n_flights)
    df_samples.to_csv('samples/samples.csv', sep='\t')

    # Sort values by time
    df_samples_by_time = df_samples.sort_values('time')

    ######################################################################################################################
    #                                     Meteorology data and advection model                                           # 
    ######################################################################################################################

    for i in range(0,10):

        identifier = i
        sample = df_samples_by_time.iloc[i,:]


        # Download the dataset from CDS and open it
        print("Downloading and opening dataset...\n")
        met = open_dataset(sample)

        # Create the pycontrails flight object (and set EIs etc.)
        fl = set_flight_parameters(sample)

        # Run DryAdvection model and generate the input .nc file
        print("Running DryAdvection model...\n")
        ds, pressure = advect(met, fl)
        ds.to_netcdf(f"mets/input{i}.nc")

        # Generate the .yaml input file dictionary and output to file
        d = generate_yaml_d(identifier, sample, fl, float(pressure/100))
        with open(f'yamls/input{i}.yaml', 'w') as yaml_file:
            yaml.dump(d, yaml_file, default_flow_style=False, sort_keys=False)

