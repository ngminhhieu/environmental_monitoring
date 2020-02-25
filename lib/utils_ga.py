from pandas import read_csv
import numpy as np
import yaml

def generate_data(input_features, dataset, output_dir):
    set_config(input_features)
    dataset = read_csv(dataset, usecols=input_features)
    np.savez(output_dir, monitoring_data = dataset)

def set_config(input_features, name="GA"):
    with open('config/taiwan/{}.yaml'.format(name), 'r') as f:
        config = yaml.load(f)

    config['model']['input_dim'] = len(input_features)

    with open('config/taiwan/{name}.yaml'.format(name), 'w') as f:
        yaml.dump(config, f)
