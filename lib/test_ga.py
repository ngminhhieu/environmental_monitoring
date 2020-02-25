import GA
import utils
import constant
import GABinary

evo=GABinary.evolution(total_feature=len(constant.features), pc=0.2, pm=0.2, population_size=50, max_gen=5)
fitness = [evo["gen"], evo["fitness"]]
utils.write_log(path="log/GA/", filename="result_binary.csv", error=fitness)

# for i in range(len(constant.features)-1, 1, -1):
#     print(i)
#     evo = GA.evolution(nb_feature=i, total_feature=len(constant.features), pc=0.2, pm=0.2, population_size=50, max_gen=50)
#     fitness = [i, evo["gen"], evo["fitness"]]
#     utils.write_log(path="log/GA/", filename="result.csv", error=fitness)