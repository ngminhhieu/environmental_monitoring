import argparse
import os
import sys
import tensorflow as tf
import numpy as np
import yaml
import random as rn
from model.supervisor import EncoderDecoder
from lib.GABinary import evolution
from lib import utils_ga
from lib import constant
from model.supervisor import EncoderDecoder
from lib import preprocessing_data

def seed():
    # The below is necessary for starting Numpy generated random numbers
    # in a well-defined initial state.
    np.random.seed(2)
    # The below is necessary for starting core Python generated random numbers
    # in a well-defined state.
    rn.seed(12345)
    # The below tf.set_random_seed() will make random number generation
    # in the TensorFlow backend have a well-defined initial state.
    # For further details, see:
    # https://www.tensorflow.org/api_docs/python/tf/set_random_seed
    tf.set_random_seed(1234)
    
if __name__ == '__main__':
    seed()
    sys.path.append(os.getcwd())
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_cpu_only', default=False, type=str, help='Whether to run tensorflow on cpu.')
    parser.add_argument('--config_file', default=False, type=str,
                        help='Config file for pretrained model.')
    parser.add_argument('--mode', default='ga_seq2seq', type=str,
                        help='Run mode.')
    args = parser.parse_args()

    # load config for seq2seq model
    if args.config_file != False:
        with open(args.config_file) as f:
            config = yaml.load(f)

    if args.mode == 'ga_seq2seq':
        evo = evolution(total_feature=len(constant.hanoi_features), pc=0.8, pm=0.2, population_size=30, max_gen=40)
        fitness = [evo["gen"], evo["fitness"]]
        utils_ga.write_log(path="log/GA/", filename="result_binary.csv", error=fitness)
    elif args.mode == 'seq2seq_train':
        model = EncoderDecoder(is_training=True, **config)
        model.train()
    elif args.mode == 'seq2seq_test':
        # predict
        model = EncoderDecoder(is_training=False, **config)
        model.test()
        model.plot_series()
    elif args.mode == 'preprocessing':
        preprocessing_data.preprocess_all()
    else:
        raise RuntimeError("Mode needs to be train/evaluate/test!")
