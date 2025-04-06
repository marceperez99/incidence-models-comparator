import random
import pandas as pd

NUMBER_OF_BITS = 7
C = 1.5


def decode(individual):
    return tuple([int(individual[i:i + NUMBER_OF_BITS], 2) for i in range(0, len(individual), NUMBER_OF_BITS)])


class GeneticAlgorithm:
    def __init__(self, population_size, generations, mutation_rate, eval_function,
                 initial_population_function):
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate

        initial = initial_population_function(self.population_size)
        self.population = ["".join([format(p, f'0{NUMBER_OF_BITS}b') for p in i]) for i in initial]
        self.eval_function = lambda a: eval_function(decode(a))

        self.C = C

    def select_parents(self):
        # Selección por torneo
        m = len(self.population)
        p = [0] * m
        p[0] = (2 - self.C) / m
        p[m - 1] = self.C / m

        for i in range(1, m - 1):
            p[i] = p[0] + (p[m - 1] - p[0]) * (i) / (m - 1)

        assert all(x < y for x, y in zip(p, p[1:])), f"The list is not in strict ascending order{p}"

        return random.choices(self.population, k=2, weights=p)

    def mutate(self, individual):
        # Aplicar mutación a un individuo con cierta probabilidad
        if random.random() < self.mutation_rate:
            individual = list(individual)
            bit = random.randint(0, len(individual) - 1)
            individual[bit] = '1' if individual[bit] == '0' else '0'
        return "".join(individual)

    def crossover(self, a, b):

        k1 = random.randint(0, len(a) - 2)
        k2 = random.randint(k1 + 1, len(a) - 1)
        offspring = b[:k1] + a[k1:k2] + b[k2:]

        return self.mutate(offspring)

    def run(self):
        best_individual = None

        for generation in range(self.generations):

            new_population = []
            for _ in range(self.population_size):
                parent1, parent2 = self.select_parents()
                offspring1 = self.crossover(parent1, parent2)
                new_population.append(offspring1)
            if best_individual and best_individual not in new_population:
                new_population.append(best_individual)
            assert not best_individual or best_individual in new_population
            self.population = sorted(new_population, key=self.eval_function, reverse=True)

            if len(self.population) > self.population_size:
                self.population = self.population[-self.population_size:]

            assert not best_individual or self.eval_function(best_individual) >= self.eval_function(self.population[-1])
            best_individual = self.population[-1]
            print(f'best in generation {generation}', decode(best_individual), self.eval_function(best_individual))


        # Devuelve el mejor individuo encontrado
        best_individual = min(self.population, key=self.eval_function)

        return decode(best_individual), self.eval_function(best_individual)
