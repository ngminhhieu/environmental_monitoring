from pandas import read_csv
import numpy as np
import yaml
from datetime import datetime
import os
import csv

def generate_data(all_input_features, dataset, output_dir):
    set_config(all_input_features, "GA_taiwan")
    dataset = read_csv(dataset, usecols=all_input_features)
    np.savez(output_dir, monitoring_data = dataset)

def set_config(all_input_features, name="GA"):
    with open('config/taiwan/{}.yaml'.format(name), 'r') as f:
        config = yaml.load(f)

    config['model']['input_dim'] = len(all_input_features)

    with open('config/taiwan/{}.yaml'.format(name), 'w') as f:
        yaml.dump(config, f)

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