import GA
import utils
import constant
evo = GA.evolution(nb_feature=5, total_feature=len(constant.features), pc=0.2, pm=0.2, population_size=50, max_gen=30)
utils.write_log("log/GA/test.csv", error=evo)