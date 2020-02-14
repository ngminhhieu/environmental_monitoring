import yaml

for index in range(1,11):
    des = 'config/random_search/taiwan/{}.yaml'.format(str(index))
    # update config
    with open(des, 'r') as f:
        config = yaml.load(f)

    config['test']['test_monthly'] = False

    with open(des, 'w') as f:
        yaml.dump(config, f)