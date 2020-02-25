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
    
if __name__ == '__main__':
    np.random.seed(1)
    sys.path.append(os.getcwd())
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_cpu_only', default=False, type=str, help='Whether to run tensorflow on cpu.')
    parser.add_argument('--mode', default='ga', type=str,
                        help='Run mode.')
    args = parser.parse_args()

    if args.mode == 'ga':
        evo = evolution(total_feature=len(constant.features), pc=0.2, pm=0.2, population_size=40, max_gen=10)
        fitness = [evo["gen"], evo["fitness"]]
        utils_ga.write_log(path="log/GA/", filename="result_binary.csv", error=fitness)
    else:
        raise RuntimeError("Mode needs to be train/evaluate/test!")
