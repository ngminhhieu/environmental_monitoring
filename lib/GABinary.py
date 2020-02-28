import random
from operator import itemgetter
import copy
import constant
import utils
from sklearn.metrics import mean_absolute_error
import pandas as pd
import math

# ignore warnings
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

# get models
models = utils.get_models("SVR", "Lasso", "ElasticNet", 
                        "KernelRidge", "GradientBoostingRegressor", 
                        "LGBMRegressor", "XGBRegressor", "RandomForestRegressor",
                        "DecisionTreeRegressor", "AdaBoostRegressor",
                        "MLPRegressor", "KNeighborsRegressor", "ExtraTreesRegressor")

decisionTree = models["DecisionTreeRegressor"]
extraTree = models["ExtraTreesRegressor"]
GBoost = models["GradientBoostingRegressor"]
randomForest = models["RandomForestRegressor"]
xgb = models["XGBRegressor"]
target_feature = 'PM2.5'

def get_input_features(gen_array):
    input_features = []    
    for index, value in enumerate(gen_array, start=0):
        if value == 1:
            input_features.append(constant.features[index])
    return input_features

def get_dataset(input_features):
    path = 'data/csv/taiwan_data_mean.csv'
    taiwan_dataset = pd.read_csv(path, usecols=input_features+[target_feature])
    new_dataset = utils.data_preprocessing(taiwan_dataset, input_features, target_feature)
    X_train, y_train, X_valid, y_valid, X_test, y_test = utils.split_data(new_dataset, 0.6, 0.2)
    return X_train, y_train, X_valid, y_valid, X_test, y_test

def fitness(gen_array, model_x):
    input_features = get_input_features(gen_array)
    X_train, y_train, X_valid, y_valid, X_test, y_test  = get_dataset(input_features)
    # fit and predict
    model_x.fit(X_train, y_train, eval_metric="mae", eval_set=[(X_valid, y_valid)], verbose=False, early_stopping_rounds = 15)
    prediction_values = model_x.predict(X_test)
    mae = mean_absolute_error(y_test, prediction_values)
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
    n = math.floor(population_size / 2)
    temp = sorted(popu, key=itemgetter("fitness"), reverse=False)
    new_list = temp[:n]
    while len(new_list) < population_size:
        i = random.randint(n, len(temp)-1)
        new_list.append(temp[i])
        temp.remove(temp[i])
    return new_list


def evolution(total_feature, population_size, pc=0.8, pm=0.2, max_gen=1000):
    print("Starting...")
    print("Preparing :)")
    population = []
    for _ in range(population_size):
        population.append(individual(total_feature=total_feature))
    t = 0
    print("Lets go!!!")
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
        utils.write_log(path="log/GA/", filename="fitness_gen.csv", error=fitness)
        print("t =", t, "fitness =", population[0]["fitness"])
        t = t + 1
    print("Done")
    return population[0]
