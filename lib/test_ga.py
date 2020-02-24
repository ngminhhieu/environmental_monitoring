import GA
import utils
import constant

for i in range(5, len(constant.features)):
    print(i)
    evo = GA.evolution(nb_feature=i, total_feature=len(constant.features), pc=0.2, pm=0.2, population_size=50, max_gen=100)
    fitness = [i, evo[0]["gen"], evo[0]["fitness"]]
    utils.write_log(path="log/GA/", filename="result.csv", error=fitness)