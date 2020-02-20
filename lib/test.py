import GA
import utils
import constant
evo = GA.evolution(nb_feature=5, total_feature=len(constant.features), pc=0.2, pm=0.2, population_size=50, max_gen=1)
utils.write_log(path="log/GA/", filename="result.csv", error=evo)