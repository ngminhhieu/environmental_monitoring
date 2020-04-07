from pandas import read_csv
import numpy as np
import pandas as pd
import yaml
from lib import constant

def generate_npz(all_input_features, dataset, output_dir, config_path):
    set_config(all_input_features, config_path)
    dataset = read_csv(dataset, usecols=all_input_features)
    np.savez(output_dir, monitoring_data = dataset)

def set_config(all_input_features, config_path):
    with open(config_path, 'r') as f:
        config = yaml.load(f)

    config['model']['input_dim'] = len(all_input_features)

    with open(config_path, 'w') as f:
        yaml.dump(config, f)

def preprocessing_original_data(dataset, output_dir):
    # dataset['TIME'] = pd.to_datetime(dataset['TIME'])
    # dataset['TIME'] = dataset['TIME'].values.astype(float)
    # replace by median values 
    dataset.fillna(dataset.mean(), inplace=True)
    dataset.to_csv(output_dir, index=False)

def preprocessing_comparison_data():
    cols = ['AMB_TEMP', 'CO','NO', 'NO2', 'NOx', 'O3', 'PM10', 'PM2.5', 'RH', 'SO2', 'WD_HR', 'WIND_DIREC', 'WIND_SPEED', 'WS_HR']
    len_cols = len(cols)    
    comparison_data = read_csv('data/csv/raw_taiwan_data.csv', usecols=[i for i in range(0,26)])
    comparison_data = comparison_data.drop(columns=['factor'])
    
    # Fill NAN to mean data
    # convert dataframe to numeric to define whether values are a number or not
    for i in range(1,25):
        comparison_data.iloc[:,i] = pd.to_numeric(comparison_data.iloc[:,i], errors='coerce')
    # replace missing values by mean data
    comparison_data.fillna(comparison_data.median(), inplace=True)

    # transform data    
    comparison_data = comparison_data.to_numpy()
    factor_data = np.zeros(shape=(int(len(comparison_data)/len_cols * 24), len_cols), dtype='float')
    time_data = np.zeros([int(len(comparison_data)/len_cols * 24), 1], dtype=object)
    index = -1
    for row in range(len(comparison_data)):
        if row%len_cols == 0:
            index += 1
            for j in range (24):
                time_data[24*index+j] = str(comparison_data[row,0]) + " " + str(j) + ":00"
        factor_data[24*index:24*(index+1),row%len_cols] = comparison_data[row, 1:25]

    # merge 2 array to panda
    # correlation
    new_data = np.concatenate((time_data, factor_data), axis=1)
    dataset = pd.DataFrame(new_data)
    dataset.columns = ['TIME','AMB_TEMP', 'CO','NO', 'NO2', 'NOx', 'O3', 'PM10', 'PM2.5', 'RH', 'SO2', 'WD_HR', 'WIND_DIREC', 'WIND_SPEED', 'WS_HR']
    columnsTitles = ['TIME','AMB_TEMP', 'CO','NO', 'NO2', 'NOx', 'O3', 'RH', 'SO2', 'WD_HR', 'WIND_DIREC', 'WIND_SPEED', 'WS_HR', 'PM10', 'PM2.5']
    dataset=dataset.reindex(columns=columnsTitles)
    dataset.to_csv('data/csv/taiwan_data_mean.csv', encoding='utf-8', index=False)

if __name__ == "__main__":
    # taiwan
    cols = ['MONTH', 'HOUR', 'AMB_TEMP', 'NO', 'NOx', 'RH', 'SO2', 'WIND_SPEED', 'WS_HR', 'PM10', 'PM2.5']
    dataset = 'data/csv/taiwan_data_mean.csv'
    output_dir = 'data/npz/ga_seq2seq.npz'
    config_path = 'config/taiwan/GA.yaml'
    generate_npz(cols, dataset, output_dir, config_path)


    # hanoi
    target_feature = ['PM2.5']
    dataset_hanoi = 'data/csv/hanoi_data_median.csv'
    config_path_full_hanoi = 'config/hanoi/full_hanoi.yaml'
    generate_npz(constant.hanoi_features + target_feature, dataset_hanoi, 'data/npz/full_hanoi.npz', config_path_full_hanoi)

    config_path_pm25_hanoi = 'config/hanoi/pm25_hanoi.yaml'
    generate_npz(target_feature, dataset_hanoi, 'data/npz/pm25_hanoi.npz', config_path_pm25_hanoi)

    ga_hanoi_features = ['WIND_SPEED', 'TEMP', 'RADIATION', 'PM10']
    config_path_ga_hanoi = 'config/hanoi/ga_hanoi.yaml'
    generate_npz(ga_hanoi_features, dataset_hanoi, 'data/npz/ga_hanoi.npz', config_path_ga_hanoi)
