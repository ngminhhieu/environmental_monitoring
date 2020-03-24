from pandas import read_csv
import numpy as np
import pandas as pd
import yaml

def generate_npz(all_input_features, dataset, output_dir, config_path):
    set_config(all_input_features, config_path)
    dataset = read_csv(dataset, usecols=all_input_features)
    dataset['date'] = pd.to_datetime(dataset['date'])
    dataset['date'] = dataset['date'].values.astype(float)
    print(dataset)
    np.savez(output_dir, monitoring_data = dataset)

def set_config(all_input_features, config_path):
    with open(config_path, 'r') as f:
        config = yaml.load(f)

    config['model']['input_dim'] = len(all_input_features)

    with open(config_path, 'w') as f:
        yaml.dump(config, f)

def preprocessing_original_data(dataset, output_dir):
    
    # replace by median values 
    dataset.fillna(dataset.mean(), inplace=True)
    dataset.to_csv(output_dir, index=False)

if __name__ == "__main__":
    cols = ['date', 'power']
    dataset = 'data/csv/evn.csv'
    output_dir = 'data/npz/evn_seq2seq.npz'
    config_path = 'config/evn/seq2seq.yaml'
    generate_npz(cols, dataset, output_dir, config_path)
