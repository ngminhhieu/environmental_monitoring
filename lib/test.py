import GA
import utils
import constant
evo = GA.evolution(nb_feature=8, total_feature=len(constant.features), pc=0.2, pm=0.2, population_size=20, max_gen=30)
utils.write_log(path="log/GA/", filename="result.csv", error=evo)