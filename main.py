import argparse
import os
import sys
import tensorflow as tf
import numpy as np
import yaml
import random as rn
from model.supervisor import EncoderDecoder
from lib.GABinary import evolution

def seed():
    # The below is necessary for starting Numpy generated random numbers
    # in a well-defined initial state.
    np.random.seed(42)
    # The below is necessary for starting core Python generated random numbers
    # in a well-defined state.
    rn.seed(12345)
    # The below tf.set_random_seed() will make random number generation
    # in the TensorFlow backend have a well-defined initial state.
    # For further details, see:
    # https://www.tensorflow.org/api_docs/python/tf/set_random_seed
    tf.set_random_seed(1234)
    

def print_lstm_info(mode, config):
    print('----------------------- INFO -----------------------')

    print('|--- MODE:\t{}'.format(mode))
    print('|--- ALG:\t{}'.format(config['alg']))
    print('|--- BASE_DIR:\t{}'.format(config['base_dir']))
    print('|--- LOG_LEVEL:\t{}'.format(config['log_level']))
    print('|--- GPU:\t{}'.format(config['gpu']))

    print('----------------------- DATA -----------------------')
    print('|--- BATCH_SIZE:\t{}'.format(config['data']['batch_size']))
    print('|--- DATASET:\t{}'.format(config['data']['dataset']))
    print('|--- TEST_SIZE:\t{}'.format(config['data']['test_size']))

    print('----------------------- MODEL -----------------------')
    print('|--- SEQ_LEN:\t{}'.format(config['model']['seq_len']))
    print('|--- HORIZON:\t{}'.format(config['model']['horizon']))
    print('|--- INPUT_DIM:\t{}'.format(config['model']['input_dim']))
    print('|--- L1_DECAY:\t{}'.format(config['model']['l1_decay']))
    print('|--- OUTPUT_DIMS:\t{}'.format(config['model']['output_dim']))
    print('|--- RNN_UNITS:\t{}'.format(config['model']['rnn_units']))
    print('|--- RNN_LAYERS:\t{}'.format(config['model']['rnn_layers']))

    if mode == 'train':
        print('----------------------- TRAIN -----------------------')
        print('|--- DROPOUT:\t{}'.format(config['train']['dropout']))
        print('|--- EPOCHS:\t{}'.format(config['train']['epochs']))
        print('|--- OPTIMIZER:\t{}'.format(config['train']['optimizer']))
        print('|--- PATIENCE:\t{}'.format(config['train']['patience']))

    else:
        print('----------------------- TEST -----------------------')
        print('|--- RUN_TIMES:\t{}'.format(config['test']['run_times']))

    print('----------------------------------------------------')
    infor_correct = input('Is the information correct? y(Yes)/n(No):')
    if infor_correct != 'y' and infor_correct != 'yes' and infor_correct != 'YES' and infor_correct != 'Y':
        raise RuntimeError('Information is not correct!')


def train_lstm_ed(config):
    with tf.device('/device:GPU:{}'.format(config['gpu'])):
        model = EncoderDecoder(is_training=True, **config)
        model.plot_models()
        model.train()


def test_lstm_ed(config):
    with tf.device('/device:GPU:{}'.format(config['gpu'])):
        model = EncoderDecoder(is_training=False, **config)
        model.test()
        model.plot_series()


def evaluate_lstm_ed(config):
    with tf.device('/device:GPU:{}'.format(config['gpu'])):
        model = EncoderDecoder(is_training=False, **config)
        model.evaluate()


if __name__ == '__main__':
    np.random.seed(1)
    sys.path.append(os.getcwd())
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_cpu_only', default=False, type=str, help='Whether to run tensorflow on cpu.')
    parser.add_argument('--config_file', default='config/config.yaml', type=str,
                        help='Config file for pretrained model.')
    parser.add_argument('--mode', default='train', type=str,
                        help='Run mode.')
    args = parser.parse_args()

    with open(args.config_file) as f:
        config = yaml.load(f)

    # print_lstm_info(args.mode, config)

    if args.mode == 'train':
        train_lstm_ed(config)
    elif args.mode == 'evaluate' or args.mode == 'evaluation':
        evaluate_lstm_ed(config)
    elif args.mode == "test":
        test_lstm_ed(config)
    elif args.mode == "ga":
        
    else:
        raise RuntimeError("Mode needs to be train/evaluate/test!")
