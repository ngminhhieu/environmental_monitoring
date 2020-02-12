from pandas import read_csv
import numpy as np
import yaml
from shutil import copyfile

def generate_hanoi_data(cols, dataset, output_dir):
    dataset = read_csv(dataset, usecols=cols)
    np.savez(output_dir, monitoring_data = dataset)

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
    src = 'config/random_search/sample.yaml'
    dataset = 'data/csv/monthly_check/train_data.csv'
    times_random_search = 10
    features = ['MONTH', 'DAY', 'YEAR', 'HOUR', 'WIND_SPEED', 'WIND_DIR', 'TEMP', 'RH', 'BAROMETER', 'RADIATION', 'INNER_TEMP', 'PM10']
    input_features = ['PM2.5']
    for time in range(12, 12+times_random_search):
        # find input_features by random search
        binary_features = np.random.randint(2, size=9)
        for index, value in enumerate(binary_features, start=0):
            if value == 1:
                input_features.append(features[index])

        # generate npz file
        output_dir = 'data/npz/random_search/{}.npz'.format(str(time))
        generate_hanoi_data(input_features, dataset, output_dir)

        # create file config        
        des = 'config/random_search/{}.yaml'.format(str(time))
        copyfile(src, des)

        # update config
        with open(des, 'r') as f:
            config = yaml.load(f)

        config['model']['input_dim'] = len(input_features)
        config['data']['dataset'] = 'data/npz/random_search/{}.npz'.format(str(time))
        config['base_dir'] = 'log/seq2seq/random_search/{}'.format(str(time))

        with open(des, 'w') as f:
            f.write('# ' + str(input_features) + '\n')
            yaml.dump(config, f)
        
        # reset input_features
        input_features = ['PM2.5']

        # update npz test for each month from March to December 2017
        for i in range(3,13):
            dataset = 'data/csv/monthly_check/test_data_{}.csv'.format(str(i))
            output_dir = 'data/npz/monthly_check/test_data_{}.npz'.format(str(i))
            generate_hanoi_data(input_features, dataset, output_dir)

        
                


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

    # dataset = 'data/csv/monthly_check/train_data.csv' 
    # cols = ['TIME', 'WIND_SPEED', 'WIND_DIR', 'TEMP', 'RH', 'BAROMETER', 'RADIATION', 'INNER_TEMP', 'PM10', 'PM2.5']    
    # output_dir = 'data/npz/monthly_check/train_data.npz'
    # generate_hanoi_data(cols, dataset, output_dir)
    
