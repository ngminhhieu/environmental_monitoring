from datetime import datetime
import os
import csv

def write_log(path, filename, error, input_feature = []):
    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    if isinstance(error, list):
        error.insert(0, dt_string)
    if os.path.exists(path) == False:
        os.makedirs(path)
    with open(path + filename, 'a') as file:
        writer = csv.writer(file)
        writer.writerow(error)
        writer.writerow(input_feature)