import GA
import utils
import constant
import GABinary

evo=GABinary.evolution(total_feature=len(constant.features), pc=0.2, pm=0.2, population_size=50, max_gen=1010)
fitness = [evo["gen"], evo["fitness"]]
utils.write_log(path="log/GA/", filename="result_binary.csv", error=fitness)