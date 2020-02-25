# ignore warnings
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
import random
from operator import itemgetter
import copy
import constant
import utils
from sklearn.metrics import mean_absolute_error
import pandas as pd
import utils_ga
from model.supervisor import EncoderDecoder

target_feature = ['PM2.5']
def get_input_features(gen_array):
    input_features = []    
    for index, value in enumerate(gen_array, start=0):
        if value == 1:
            input_features.append(constant.features[index])
    return input_features

def preprocessing_config(input_features):
    path = 'data/csv/taiwan_test.csv'    
    output_dir = 'data/npz/ga.npz'
    utils_ga.generate_data(input_features+target_feature, path, output_dir)

def fitness(gen_array, model_x):
    input_features = get_input_features(gen_array)
    preprocessing_config(input_features)
    # train
    model = EncoderDecoder(is_training=True, **config)
    model.train()

    # predict
    model = EncoderDecoder(is_training=False, **config)
    mae = model.test()
    return mae


def individual(total_feature):
    a = [0 for _ in range(total_feature)]
    for i in range(total_feature):
        r = random.random()
        if r < 0.5:
            a[i] = 1
    indi = {"gen": a, "fitness": 0}
    indi["fitness"] = fitness(indi["gen"], xgb)
    return indi


def crossover(father, mother, total_feature):
    cutA = random.randint(1, total_feature-1)
    cutB = random.randint(1, total_feature-1)
    while cutB == cutA:
        cutB = random.randint(1, total_feature - 1)
    start = min(cutA, cutB)
    end = max(cutA, cutB)
    child1 = {"gen": [0 for _ in range(total_feature)], "fitness": 0}
    child2 = {"gen": [0 for _ in range(total_feature)], "fitness": 0}

    child1["gen"][:start] = father["gen"][:start]
    child1["gen"][start:end] = mother["gen"][start:end]
    child1["gen"][end:] = father["gen"][end:]
    child1["fitness"] = fitness(child1["gen"], xgb)

    child2["gen"][:start] = mother["gen"][:start]
    child2["gen"][start:end] = father["gen"][start:end]
    child2["gen"][end:] = mother["gen"][end:]
    child2["fitness"] = fitness(child2["gen"], xgb)
    return child1, child2


def mutation(father, total_feature):
    a = copy.deepcopy(father["gen"])
    i = random.randint(0, total_feature-1)
    if a[i] == 0:
        a[i] = 1
    else:
        a[i] = 0
    child = {"gen": a, "fitness": 0}
    child["fitness"] = fitness(child["gen"], xgb)
    return child


def selection(popu, population_size):
    new_list = sorted(popu, key=itemgetter("fitness"), reverse=False)
    return new_list[:population_size]


def evolution(total_feature, population_size, pc=0.8, pm=0.2, max_gen=1000):
    population = []
    for _ in range(population_size):
        population.append(individual(total_feature=total_feature))
    t = 0
    while t < max_gen:
        for i, _ in enumerate(population):
            r = random.random()
            if r < pc:
                j = random.randint(0, population_size-1)
                while j == i:
                    j = random.randint(0, population_size - 1)
                f_child, m_child = crossover(population[i], population[j], total_feature=total_feature)
                population.append(f_child)
                population.append(m_child)
            if r < pm:
                off = mutation(population[i], total_feature=total_feature)
                population.append(off)
        population = selection(population, population_size)
        fitness = [t, population[0]["gen"], population[0]["fitness"]]
        utils_ga.write_log(path="log/GA/", filename="fitness_gen.csv", error=fitness)
        print("t =", t, "fitness =", population[0]["fitness"])
        t = t + 1
    return population[0]
