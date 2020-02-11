from pandas import read_csv
import numpy as np
import yaml

def generate_hanoi_data(cols, dataset, output_dir):
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
    dataset = 'data/csv/hanoi_data_mean.csv'    

    # cols = ['WIND_SPEED', 'RADIATION', 'PM10', 'PM2.5']    
    # output_dir = 'data/npz/random_search/1.npz'
    # generate_hanoi_data(cols, dataset, output_dir)

    # cols = ['TIME', 'WIND_SPEED', 'TEMP', 'BAROMETER', 'INNER_TEMP', 'PM10', 'PM2.5']    
    # output_dir = 'data/npz/random_search/2.npz'
    # generate_hanoi_data(cols, dataset, output_dir)
    
    # cols = ['TEMP', 'BAROMETER', 'RADIATION','INNER_TEMP','PM10', 'PM2.5']    
    # output_dir = 'data/npz/random_search/3.npz'
    # generate_hanoi_data(cols, dataset, output_dir)

    # cols = ['WIND_SPEED', 'TEMP', 'RADIATION','INNER_TEMP', 'PM2.5']    

    # output_dir = 'data/npz/random_search/4.npz'
    # generate_hanoi_data(cols, dataset, output_dir)

    # cols = ['TIME', 'WIND_DIR', 'TEMP', 'RADIATION', 'PM10', 'PM2.5']    
    # output_dir = 'data/npz/random_search/5.npz'
    # generate_hanoi_data(cols, dataset, output_dir)

    # cols = ['TIME','WIND_SPEED','WIND_DIR','TEMP', 'RADIATION', 'INNER_TEMP','PM10', 'PM2.5']    
    # dataset = 'data/csv/hanoi_data_mean.csv'
    # output_dir = 'data/npz/random_search/6.npz'
    # generate_hanoi_data(cols, dataset, output_dir)

    # cols = ['WIND_DIR', 'TEMP', 'RH','BAROMETER', 'INNER_TEMP', 'PM2.5']    
    # output_dir = 'data/npz/random_search/7.npz'
    # generate_hanoi_data(cols, dataset, output_dir)

    # cols = ['TIME','WIND_SPEED', 'RH', 'BAROMETER','RADIATION', 'PM2.5']    
    # dataset = 'data/csv/hanoi_data_mean.csv'
    # output_dir = 'data/npz/random_search/8.npz'
    # generate_hanoi_data(cols, dataset, output_dir)

    # cols = ['WIND_SPEED','WIND_DIR','TEMP','RH', 'PM10', 'PM2.5']    
    # output_dir = 'data/npz/random_search/9.npz'
    # generate_hanoi_data(cols, dataset, output_dir)

    # cols = ['RH', 'RADIATION', 'INNER_TEMP', 'PM2.5']    
    # output_dir = 'data/npz/random_search/10.npz'
    # generate_hanoi_data(cols, dataset, output_dir)

    # cols = ['PM2.5']    
    # output_dir = 'data/npz/random_search/11.npz'
    # generate_hanoi_data(cols, dataset, output_dir)

    dataset = 'data/csv/monthly_check/train_data.csv' 
    cols = ['TIME', 'WIND_SPEED', 'WIND_DIR', 'TEMP', 'RH', 'BAROMETER', 'RADIATION', 'INNER_TEMP', 'PM10', 'PM2.5']    
    output_dir = 'data/npz/monthly_check/train_data.npz'
    generate_hanoi_data(cols, dataset, output_dir)
    for i in range(3,13):
        dataset = 'data/csv/monthly_check/test_data_{}.csv'.format(str(i))
        output_dir = 'data/npz/monthly_check/test_data_{}.npz'.format(str(i))
        generate_hanoi_data(cols, dataset, output_dir)
