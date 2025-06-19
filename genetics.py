import random
import numpy as np


def cost(subset, full_set, s):
    return abs(s - np.dot(subset, full_set))


# Funkcja tworzy jeden chromosom (jednego osobnika)
def generate_chromosome(full_set):
    return [random.randint(0, 1) for _ in range(len(full_set))]


# Tworzę początkową populację - 20 chromosomów (osobników)
def generate_init_population(full_set):
    return [generate_chromosome(full_set) for _ in range(20)]


def fitness_factory(full_set, s):
    def fitness(chromosome):
        return 1.0 / (cost(chromosome, full_set, s) + 0.001)

    return fitness


def roulette_selection(fitness_values):
    ret_indexes = []
    sum_fitness = sum(fitness_values)
    for _ in range(len(fitness_values)):
        pick = random.uniform(0, sum_fitness)
        current = 0
        for idx, fit in enumerate(fitness_values):
            # dodajemy po kolei wartości funkcji fitness żeby sprawdzić w którym miejscu
            # ruletki wylądował nasz pick.
            current += fit
            # jeśli nasz pick to 5, a wartości fitness to 1, 2, 3 - to dopiero dla osobnika
            # o indeksie 2 dodamy go do listy.
            if current >= pick:
                ret_indexes.append(idx)
                break
    return ret_indexes


def one_point_crossover(selected_indexes, population, p_crossover=0.25):
    new_population = []
    for i in range(0, len(selected_indexes), 2):
        parent_1 = population[selected_indexes[i]].copy()
        parent_2 = population[selected_indexes[i]].copy()
        # losujemy żeby spojrzeć czy w ogóle krzyżujemy (krzyżujemy tylko część populacji
        if random.random() < p_crossover:
            # population[0] - 1 robimy po to, że musimy znać długość podzbioru. Równie dobrze
            # mogłoby być population[1] [2] itd.
            cp1 = random.randint(0, len(population[0]) - 1)
            kid_1 = parent_1[0:cp1] + parent_2[cp1:]
            kid_2 = parent_2[0:cp1] + parent_1[cp1:]
            new_population.append(kid_1)
            new_population.append(kid_2)
        else:
            new_population.append(parent_1)
            new_population.append(parent_2)
    return new_population


def two_point_crossover(selected_indexes, population, p_crossover=0.25):
    new_population = []
    for i in range(0, len(selected_indexes), 2):
        parent_1 = population[selected_indexes[i]].copy()
        parent_2 = population[selected_indexes[i]].copy()
        if random.random() < p_crossover:
            cp1 = random.randint(0, len(population[0]) - 1)
            cp2 = random.randint(0, len(population[0]) - 1)
            if cp1 > cp2:
                # w pythonie można robić zamianę w ten sposób, bez zmiennej temp
                cp1, cp2 = cp2, cp1
            kid_1 = parent_1[0:cp1] + parent_2[cp1:cp2] + parent_1[cp2:]
            kid_2 = parent_2[0:cp1] + parent_1[cp1:cp2] + parent_2[cp1:]
            new_population.append(kid_1)
            new_population.append(kid_2)
        else:
            new_population.append(parent_1)
            new_population.append(parent_2)
    return new_population


# do uzupełnienia - warunek zakończenia
def term_condition_iterations(population):
    return True


def genetic_algorithm(full_set, crossover, mutation, term_condition):
    population = generate_init_population(full_set)
    while term_condition(population):
        # liczymy jak "dobry" jest każdy osobnik
        population_fit = [ fitness(o) for o in population ]
        # wybieramy ruletką osobniki do populacji
        selected_indexes = roulette_selection(population_fit)
        # krzyżowanie
        offspring = crossover(selected_indexes, population.copy())
        # mutacja
        offspring = mutation(offspring)
        # nowa populacja po krzyżowaniu i mutacji
        population = offspring




x = generate_init_population([1, 2, 3, 6, 8, 2, 0, -3, 20, 15, 7, 3])
fitness = fitness_factory([1, 2, 3, 6, 5, 2, 0, -3, 20, 15, 7, 3], 5)
print(x)
print("-------")
print(roulette_selection(fitness(x)))
print("-------")
print(two_point_crossover(roulette_selection(fitness(x)), x, 0.25))


# print(x)
# print(fitness(x))
