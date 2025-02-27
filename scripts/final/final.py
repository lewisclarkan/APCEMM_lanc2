
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from alive_progress import alive_bar

from src.sampling import calcTotalDistance, samplePoint, generateFlight
from src.geodata import open_dataset_from_sample, run_DryAdvection_and_met

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
    #                                     Meteorology data and advection model                                           # 
    ######################################################################################################################

    sample = df_samples_by_time.iloc[0,:]

    print("Downloading and opening dataset...\n")
    met = open_dataset_from_sample(sample)

    print("Running DryAdvection model...\n")
    run_DryAdvection_and_met(sample, met)