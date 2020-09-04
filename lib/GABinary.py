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

target_feature = ['PM2.5']
dataset = 'data/csv/hanoi_data.csv'
output_dir = 'data/npz/hanoi/ga_hanoi.npz'
config_path_ga = 'config/hanoi/ga_hanoi.yaml'

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

def fitness(gen_array):
    input_features = get_input_features(gen_array)
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
    a = [0 for _ in range(total_feature)]
    for i in range(total_feature):
        r = random.random()
        if r < 0.5:
            a[i] = 1
    indi = {"gen": a, "fitness": 0, "time": 0}
    indi["fitness"], indi["time"] = fitness(indi["gen"])
    return indi


def crossover(father, mother, total_feature):
    cutA = random.randint(1, total_feature-1)
    cutB = random.randint(1, total_feature-1)
    while cutB == cutA:
        cutB = random.randint(1, total_feature - 1)
    start = min(cutA, cutB)
    end = max(cutA, cutB)
    child1 = {"gen": [0 for _ in range(total_feature)], "fitness": 0, "time": 0}
    child2 = {"gen": [0 for _ in range(total_feature)], "fitness": 0, "time": 0}

    child1["gen"][:start] = father["gen"][:start]
    child1["gen"][start:end] = mother["gen"][start:end]
    child1["gen"][end:] = father["gen"][end:]
    child1["fitness"], child1["time"] = fitness(child1["gen"])

    child2["gen"][:start] = mother["gen"][:start]
    child2["gen"][start:end] = father["gen"][start:end]
    child2["gen"][end:] = mother["gen"][end:]
    child2["fitness"], child1["time"] = fitness(child2["gen"])
    return child1, child2


def mutation(father, total_feature):
    a = copy.deepcopy(father["gen"])
    i = random.randint(0, total_feature-1)
    if a[i] == 0:
        a[i] = 1
    else:
        a[i] = 0
    child = {"gen": a, "fitness": 0, "time": 0}
    child["fitness"], child["time"] = fitness(child["gen"])
    return child


def selection(popu, population_size):
    new_list = sorted(popu, key=itemgetter("fitness"), reverse=False)
    return new_list[:population_size]
    # n = math.floor(population_size / 2)
    # temp = sorted(popu, key=itemgetter("fitness"), reverse=False)
    # new_list = temp[:n]
    # while len(new_list) < population_size:
    #     i = random.randint(n, len(temp)-1)
    #     new_list.append(temp[i])
    #     temp.remove(temp[i])
    # return new_list


def evolution(total_feature, population_size, pc=0.8, pm=0.2, max_gen=1000):
    population = []
    for _ in range(population_size):
        population.append(individual(total_feature=total_feature))
    t = 0
    while t < max_gen:
        training_time_gen = 0
        temp_population = []
        for i, _ in enumerate(population):
            r = random.random()
            if r < pc:
                j = random.randint(0, population_size-1)
                while j == i:
                    j = random.randint(0, population_size - 1)
                f_child, m_child = crossover(population[i], population[j], total_feature=total_feature)
                temp_population.append(f_child)
                temp_population.append(m_child)
                training_time_gen += f_child["time"] + m_child["time"]
            if r < pm:
                off = mutation(population[i], total_feature=total_feature)
                temp_population.append(off)
                training_time_gen += off["time"]

        population = selection(population+temp_population, population_size)
        fitness = [t, population[0]["gen"], population[0]["fitness"], training_time_gen]
        utils_ga.write_log(path="log/GA_pop_50/", filename="fitness_gen.csv", error=fitness)
        print("t =", t, "fitness =", population[0]["fitness"], "time =", training_time_gen)
        t = t + 1
    return population[0]
