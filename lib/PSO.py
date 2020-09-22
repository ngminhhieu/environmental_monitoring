# ignore warnings
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
import random
from operator import itemgetter
import copy
import yaml
import math
from sklearn.metrics import mean_absolute_error
import pandas as pd

from lib import constant
from lib import utils_ga
from lib import preprocessing_data
from model.supervisor import EncoderDecoder
import numpy as np
from datetime import datetime

w = 1
c1 = 1
c2 = 1

target_feature = ['PM2.5']
dataset = 'data/csv/hanoi_data.csv'
output_dir = 'data/npz/hanoi/ga_hanoi.npz'
config_path_ga = 'config/hanoi/ga_hanoi.yaml'
log_path = "log/pso/"



def get_input_features(gen_array):
    input_features = []    
    for index, value in enumerate(gen_array, start=0):
        if value == 1:
            input_features.append(constant.hanoi_features[index])
    return input_features

def load_config():
    with open(config_path_ga) as f:
        config = yaml.load(f)
    return config

def fitness(arr):
    input_features = get_input_features(arr)
    preprocessing_data.generate_npz(input_features+target_feature, dataset, output_dir, config_path_ga)
    config = load_config()
    # train
    model = EncoderDecoder(is_training=True, **config)
    training_time = model.train()

    # predict
    model = EncoderDecoder(is_training=False, **config)
    mae = model.test()
    return mae, np.sum(np.array(training_time))


def individual(total_feature):
    location = np.random.rand(total_feature)
    velocity = np.zeros(total_feature)
    fit, time_training = fitness(np.rint(location))
    return {"location": location, "velocity": velocity, "fitness": fit, "pbest": location, "fit_best": fit, "time": time_training}

def multi(arr1, arr2):
    return arr1 * arr2


def add(arr1, arr2):
    return arr1 + arr2 - arr1 * arr2


def minus(arr):
    a = np.random.randint(1, size=len(arr)) + 1
    return a - arr


def gbest(popu):
    id_min = min(range(len(popu)), key=lambda index: popu[index]['fitness'])
    return popu[id_min]["location"]


def repair(arr):
    if np.max(arr) == np.min(arr):
        if arr[0] > 1:
            return np.ones(len(arr))
        elif arr[0] < 0:
            return np.zeros(len(arr))
        else:
            return arr
    return (arr - np.min(arr)) / (np.max(arr) - np.min(arr))


def evolution(pop_size, total_feature, max_time):
    pop = []
    first_training_time = 0
    start_time = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    for _ in range(pop_size):
        indi = individual(total_feature=total_feature)
        pop.append(indi)
        first_training_time += indi["time"]
    utils_ga.write_log(path=log_path, filename="fitness_gen.csv", error=[start_time, first_training_time])

    t = 1
    while t <= max_time:
        
        g = gbest(popu=pop)
        for i, _ in enumerate(pop):
            r1 = np.random.rand(total_feature)
            r2 = np.random.rand(total_feature)
            loc = c1 * r1 * (pop[i]["pbest"] - pop[i]["location"])
            glo = c2 * r2 * (g - pop[i]["location"])
            pop[i]["velocity"] = w * pop[i]["velocity"] + loc + glo
            pop[i]["location"] = pop[i]["location"] + pop[i]["velocity"]
            pop[i]["location"] = repair(pop[i]["location"])
            pop[i]["fitness"], pop[i]["time"] = fitness(np.rint(pop[i]["location"]))
            if pop[i]["fitness"] < pop[i]["fit_best"]:
                pop[i]["pbest"] = pop[i]["location"]
                pop[i]["fit_best"] = pop[i]["fitness"]

            print("Timeeeeee: ", i+t+1) 
        g = gbest(popu=pop)
        fitness_mae, time_training = fitness(np.rint(g))     
        fitness_error = [t, np.rint(g), fitness_mae, time_training]
        utils_ga.write_log(path=log_path, filename="fitness_gen.csv", error=fitness_error)
        print("t =", t, "fitness =", fitness_mae, "time =", time_training)

        t = t + 1

    return np.rint(g), fitness_mae
