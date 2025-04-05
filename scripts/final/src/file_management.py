
import os

def write_output_header(file_name):

    try:
        os.remove(file_name)
    except FileNotFoundError:
        print("output.txt does not yet exist")

    with open(file_name, "w") as f:
        f.write("Index  Status            Latitude  Longitude  Altitude  Time                   Age\n")

    return

def write_output(file_name, sample, age, status):

    with open(file_name, "a") as f:
        f.write(f"{sample['index']}     {status}   {sample['latitude']:.2f}     {sample['longitude']:.2f}      {sample['altitude']:.1f}   {sample['time']}    {age}\n")

    return



