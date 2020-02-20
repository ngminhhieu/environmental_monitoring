import GA
import utils
import constant

for i in range(1, len(constant.features)+1):
    evo = GA.evolution(nb_feature=i, total_feature=len(constant.features), pc=0.2, pm=0.2, population_size=50, max_gen=200)
    utils.write_log(path="log/GA/", filename="result.csv", error=evo)