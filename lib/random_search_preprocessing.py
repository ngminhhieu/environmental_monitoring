from pandas import read_csv
import numpy as np
import yaml
from shutil import copyfile
import argparse
import os
import sys

def generate_hanoi_data(cols, dataset, output_dir):
    dataset = read_csv(dataset, usecols=cols)
    np.savez(output_dir, monitoring_data = dataset)

def set_config(name):
    with open('config/taiwan/{}.yaml'.format(name), 'r') as f:
        config = yaml.load(f)

    config['test']['test_monthly'] = False

    with open('config/taiwan/{name}.yaml'.format(name), 'w') as f:
        yaml.dump(config, f)
    

def random_search():
    src = 'config/random_search/sample.yaml'
    dataset_train = 'data/csv/taiwan_data_mean.csv'
    times_random_search = 10
    features = ['MONTH', 'DAY', 'YEAR', 'HOUR', 'AMB_TEMP', 'CO', 'NO', 'NO2',
    'NOx', 'O3', 'RH', 'SO2', 'WD_HR', 'WIND_DIREC', 'WIND_SPEED', 'WS_HR', 'PM10']

    # generate train npz for only pm2.5
    # also genereate test npz 
    input_features = ['PM2.5']
    output_dir = 'data/npz/random_search/taiwan/pm25.npz'
    generate_hanoi_data(input_features, dataset_train, output_dir)
    # update npz test for each month from March to December 2017
    for i in range(1,13):
            dataset_test = 'data/csv/monthly_check_taiwan/test_data_{}.csv'.format(str(i))
            output_dir = 'data/npz/monthly_check_taiwan/test_data_{}.npz'.format(str(i))
            generate_hanoi_data(input_features, dataset_test, output_dir)

    # random search
    for time in range(1, 1+times_random_search):
        # find input_features by random search
        binary_features = np.random.randint(2, size=len(features))
        for index, value in enumerate(binary_features, start=0):
            if value == 1:
                input_features.append(features[index])

        # generate npz file
        output_dir = 'data/npz/random_search/taiwan/{}.npz'.format(str(time))
        generate_hanoi_data(input_features, dataset_train, output_dir)

        # create file config        
        des = 'config/random_search/taiwan/{}.yaml'.format(str(time))
        copyfile(src, des)

        # update config
        with open(des, 'r') as f:
            config = yaml.load(f)

        config['model']['input_dim'] = len(input_features)
        config['data']['dataset'] = 'data/npz/random_search/taiwan/{}.npz'.format(str(time))
        config['base_dir'] = 'log/seq2seq/random_search/taiwan/{}'.format(str(time))
        config['input_features'] = str(input_features)

        with open(des, 'w') as f:
            # f.write('# ' + str(input_features) + '\n')
            yaml.dump(config, f)
        
        # reset input_features
        input_features = ['PM2.5']

if __name__ == "__main__":
    sys.path.append(os.getcwd())
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_cpu_only', default=False, type=str, help='Whether to run tensorflow on cpu.')
    parser.add_argument('--config_file', default='config/config.yaml', type=str,
                        help='Config file for pretrained model.')
    parser.add_argument('--func', default='random_search', type=str)
    args = parser.parse_args()
    
    if args.func == 'random_search':
        random_search()
    elif args.func == 'set_config':
        set_config()
    else:
        raise RuntimeError("No function!")
   
    
