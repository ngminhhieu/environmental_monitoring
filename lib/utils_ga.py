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

    # +1 la vi input_features chua co PM2.5
    config['model']['input_dim'] = len(input_features)+1

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