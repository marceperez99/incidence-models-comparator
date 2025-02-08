from evolutionary_algorithm import GeneticAlgorithm
from models.exponential import  exponential_evaluator, exponential_model
import numpy as np
import pandas as pd

from models.subexponential import subexponential_model, subexponential_evaluator

POPULATION = 100
GENERATIONS = 50
data_csv = pd.read_csv('./case_data.csv')

df_0 = data_csv[(data_csv['disease'] != "DENGUE,CHIKUNGUNYA") & (data_csv['disease'] != "ARBOVIROSIS")]
# Update the 'i_cases' column in the DataFrame 'df_0' where the value is 0 with a random exponential value between 1e-9 and 1.
df_0.loc[df_0['i_cases'] == 0, 'i_cases'] = np.exp(np.random.uniform(np.log(1e-9), np.log(1)))

# Group by specified columns and apply mutations
df_0 = df_0.groupby(
    ['name', 'level', 'disease', 'classification'],
    group_keys=False
).apply(lambda group: (
    group.assign(id_proy=group['name'] + '-' + group['disease'] + '-' + group['classification'],
                 i_cases=group['i_cases'] + np.exp(np.random.uniform(np.log(1e-9), np.log(1), size=len(group))),
                 csum=group['i_cases'].cumsum())
)
        ).reset_index(drop=True)
# create columns of time
df_0['date'] = pd.to_datetime(df_0['date'])
df_0['t'] = (df_0['date'] - df_0['date'].min() + (1 * np.timedelta64(1, 'D'))) / np.timedelta64(1, 'D')
dataset = df_0


def get_initial_population(size):
    return [(np.random.randint(low=2, high=dataset.shape[0] // 2), np.random.randint(low=2, high=6)) for _ in
            range(size - 1)]

# Exponential model
# exponential_model(dataset, 2, 1, plot=True)
# genetic_agent = GeneticAlgorithm(POPULATION, GENERATIONS, 0.1, exponential_evaluator(dataset), get_initial_population)
# best_individual, error = genetic_agent.run()
# print(best_individual, error)

# Subexponential

genetic_agent = GeneticAlgorithm(POPULATION, GENERATIONS, 0.1, subexponential_evaluator(dataset), get_initial_population)
best_individual, error = genetic_agent.run()
print(best_individual, error)
# print(subexponential_model(dataset, 4, 3, plot=True))