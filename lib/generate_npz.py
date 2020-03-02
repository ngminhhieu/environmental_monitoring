from pandas import read_csv
import numpy as np
import yaml

def generate_data(cols, dataset, output_dir):
    dataset = read_csv(dataset, usecols=cols)
    np.savez(output_dir, monitoring_data = dataset)

def generate_taiwan_data():
    cols = ['WIND_DIREC', 'WIND_SPEED', 'AMB_TEMP','RH','PM10', 'PM2.5']
    dataset = read_csv('data/csv/taiwan_data_mean.csv', usecols=cols)
    np.savez('data/npz/taiwan_data.npz', monitoring_data = dataset)

def generate_hanoi_data_fi_xgboost():
    features = np.load('data/npz/feature_engineering/hanoi_data_xgboost.npz')
    cols = features['features']
    cols = np.append(cols, ['PM10', 'PM2.5'])
    print(cols)
    # cols = ['TIME','WIND_SPEED','WIND_DIR','TEMP','RH','BAROMETER','RADIATION','INNER_TEMP','PM2.5']
    # cols = ['TEMP', 'PM10', 'PM2.5']
    set_input_dim(len(cols), 'hanoi')
    dataset = read_csv('data/csv/hanoi_data_mean.csv', usecols=cols)
    np.savez('data/npz/hanoi_data_xgboost.npz', monitoring_data = dataset)

def generate_taiwan_data_fi_xgboost():
    features = np.load('data/npz/feature_engineering/taiwan_data_xgboost.npz')
    cols = features['features']
    cols = np.append(cols, ['PM10', 'PM2.5'])
    print(cols)
    # cols = ['AMB_TEMP', 'CO', 'O3', 'SO2', 'WS_HR', 'PM10', 'PM2.5']
    set_input_dim(len(cols), 'taiwan')
    dataset = read_csv('data/csv/taiwan_data_mean.csv', usecols=cols)
    np.savez('data/npz/taiwan_data_xgboost.npz', monitoring_data = dataset)

def set_input_dim(number_of_input_dim,name):
    if name == 'hanoi':
        # update config for inputdim = 24
        with open('config/hanoi/horizon_1_xgboost.yaml', 'r') as f:
            config = yaml.load(f)

        config['model']['input_dim'] = number_of_input_dim

        with open('config/hanoi/horizon_1_xgboost.yaml', 'w') as f:
            yaml.dump(config, f)
    else:
        # update config for inputdim = 24
        with open('config/taiwan/l_24_horizon_1_xgboost.yaml', 'r') as f:
            config = yaml.load(f)

        config['model']['input_dim'] = number_of_input_dim

        with open('config/taiwan/l_24_horizon_1_xgboost.yaml', 'w') as f:
            yaml.dump(config, f)
        
        # update config for inputdim = 48
        for i in range(1,7):
            with open('config/{}/horizon_{}_xgboost.yaml'.format(name, str(i)), 'r') as f:
                config = yaml.load(f)

            config['model']['input_dim'] = number_of_input_dim

            with open('config/{}/horizon_{}_xgboost.yaml'.format(name, str(i)), 'w') as f:
                yaml.dump(config, f)

if __name__ == "__main__":
    dataset = 'data/csv/taiwan_data_mean.csv'    
    features = ['MONTH', 'DAY', 'YEAR', 'HOUR', 'AMB_TEMP', 'CO', 'NO', 'NO2',
    'NOx', 'O3', 'RH', 'SO2', 'WD_HR', 'WIND_DIREC', 'WIND_SPEED', 'WS_HR', 'PM10', 'PM2.5']  
    output_dir = 'data/npz/taiwan_data_mean.npz'
    generate_data(features, dataset, output_dir)

    features = ['PM2.5']  
    output_dir = 'data/npz/taiwan_data_mean_pm25.npz'
    generate_data(features, dataset, output_dir)

    features = ['MONTH', 'HOUR', 'AMB_TEMP', 'NO',
    'NOx', 'RH', 'SO2', 'WIND_SPEED', 'WS_HR', 'PM10', 'PM2.5']  
    output_dir = 'data/npz/5_years.npz'
    generate_data(features, dataset, output_dir)