
import os

def write_output_header():

    try:
        os.remove("output.txt")
    except FileNotFoundError:
        print("output.txt does not yet exist")

    with open("output.txt", "w") as f:
        f.write("Index  Status            Latitude  Longitude  Altitude  Time                   Age    J_per_m\n")

    return

def write_output(sample, j_per_m, age, status):

    with open("output.txt", "a") as f:
        f.write(f"{sample['index']}     {status}   {sample['latitude']:.2f}     {sample['longitude']:.2f}      {sample['altitude']:.1f}   {sample['time']}    {age}     {j_per_m:.2f}\n")

    return



