from pandas import read_csv
import numpy as np
import yaml

def generate_data(dataset, output_name):    
    np.savez('data/npz/{}.npz'.format(output_name), monitoring_data = dataset)

def generate_hanoi_data_fi_xgboost():
    features = np.load('data/npz/feature_engineering/hanoi_data_xgboost.npz')
    cols = features['features']
    cols = np.append(cols, ['PM10', 'PM2.5'])
    set_input_dim(len(cols), 'hanoi')
    dataset = read_csv('data/csv/hanoi_data_mean.csv', usecols=cols)
    np.savez('data/npz/hanoi_data_xgboost.npz', monitoring_data = dataset)

def generate_taiwan_data_fi_xgboost(dataset_url, dir_url, output_name):
    features = np.load('data/npz/feature_engineering/taiwan_data_xgboost.npz')
    cols = features['features']
    cols = np.append(cols, ['PM10', 'PM2.5'])
    set_input_dim(len(cols), dir_url)
    dataset = read_csv('data/csv/{}.csv'.format(dataset_url), usecols=cols)
    np.savez('data/npz/{}.npz'.format(output_name), monitoring_data = dataset)

def set_input_dim(number_of_input_dim,name):
    if name == 'hanoi':
        # update config for inputdim = 24
        with open('config/hanoi/horizon_1_xgboost.yaml', 'r') as f:
            config = yaml.load(f)

        config['model']['input_dim'] = number_of_input_dim

        with open('config/hanoi/horizon_1_xgboost.yaml', 'w') as f:
            yaml.dump(config, f)
    else:
        # update config for inputdim = 48
        for i in range(1,7):
            with open('config/{}/horizon_{}_xgboost.yaml'.format(name, str(i)), 'r') as f:
                config = yaml.load(f)

            config['model']['input_dim'] = number_of_input_dim

            with open('config/{}/horizon_{}_xgboost.yaml'.format(name, str(i)), 'w') as f:
                yaml.dump(config, f)

if __name__ == "__main__":
    # hanoi data without feature selection
    cols = ['TIME','WIND_SPEED','WIND_DIR','TEMP','RH','BAROMETER','RADIATION','INNER_TEMP','PM10', 'PM2.5']
    dataset_hanoi = read_csv('data/csv/hanoi_data_mean.csv', usecols=cols)
    generate_data(dataset_hanoi, "hanoi_data")

    # taiwan data without feature selection
    cols_taiwan = ['WIND_DIREC', 'WIND_SPEED', 'AMB_TEMP','RH','PM10', 'PM2.5']

    dataset_taiwan_yilan_yilan = read_csv('data/csv/taiwan_yilan_yilan_mean.csv', usecols=cols)
    generate_data(dataset_taiwan_yilan_yilan, "taiwan_yilan_yilan_data")

    dataset_taiwan_yilan_dongshan = read_csv('data/csv/taiwan_yilan_dongshan_mean.csv', usecols=cols)
    generate_data(dataset_taiwan_yilan_yilan, "taiwan_yilan_dongshan_data")

    # hanoi data with feature selection using xgboost
    generate_hanoi_data_fi_xgboost()

    # taiwan data feature selection using xgboost
    generate_taiwan_data_fi_xgboost('data/csv/taiwan_yilan_yilan_mean.csv', 'taiwan/yilan/yilan', 'taiwan_yilan_yilan_data_xgboost')
    generate_taiwan_data_fi_xgboost('data/csv/taiwan_yilan_dongshan_mean.csv', 'taiwan/yilan/dongshan', 'taiwan_yilan_dongshan_data_xgboost')
