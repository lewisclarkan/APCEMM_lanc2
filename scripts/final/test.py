from pycontrails.datalib.ecmwf import ERA5ARCO
from datetime import datetime

time = datetime(2022, 3, 26, 0), datetime(2022, 3, 26, 2)

arco = ERA5ARCO(
    time=time,
    variables=["t", "q", "u", "v", "w", "z"],
)

met_pl = arco.open_metdataset()

with open('out.txt', 'w') as f:
    print(met_pl)