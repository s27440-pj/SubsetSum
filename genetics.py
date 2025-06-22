import random
import numpy as np
import argparse

from main import read_set_from_file


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
        parent_2 = population[selected_indexes[i+1]].copy()
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
        parent_2 = population[selected_indexes[i+1]].copy()
        if random.random() < p_crossover:
            cp1 = random.randint(0, len(population[0]) - 1)
            cp2 = random.randint(0, len(population[0]) - 1)
            if cp1 > cp2:
                # w pythonie można robić zamianę w ten sposób, bez zmiennej temp
                cp1, cp2 = cp2, cp1
            kid_1 = parent_1[0:cp1] + parent_2[cp1:cp2] + parent_1[cp2:]
            kid_2 = parent_2[0:cp1] + parent_1[cp1:cp2] + parent_2[cp2:]
            new_population.append(kid_1)
            new_population.append(kid_2)
        else:
            new_population.append(parent_1)
            new_population.append(parent_2)
    return new_population


def mutation_bit_flip(offspring, p_mutation=0.01):
    new_population = []
    for chromosome in offspring:
        new_chromosome = []
        for bit in chromosome:
            if random.random() < p_mutation:
                new_chromosome.append(1 - bit)
            else:
                new_chromosome.append(bit)
        new_population.append(new_chromosome)
    return new_population


def mutation_normal(offspring, p_mutation=0.01):
    n = len(offspring[0])
    new_population = []
    # wybieramy ile bitów zmieniamy (ograniczenie żeby nie więcej niż tyle ile jest bitów, ale nie mniej niż 0)
    indexes = max(0, min(n, int(random.gauss(n/2, n/6))))
    indexes_to_flip = random.sample(range(n), indexes)
    for chromosome in offspring:
        if random.random() < p_mutation:
            new_chromosome = chromosome.copy()
            for i in indexes_to_flip:
                new_chromosome[i] = 1 - new_chromosome[i]
            new_population.append(new_chromosome)
        else:
            new_population.append(chromosome)
    return new_population


def best_element(population, fitness):
    best = population[0]
    for element in population:
        if fitness(element) >= fitness(best):
            best = element
    return best


max_iterations = 100
iteration = 0


def termination_max_iterations(population):
    global iteration
    iteration += 1
    return iteration < max_iterations


def genetic_algorithm(full_set, s, crossover, mutation, term_condition):
    population = generate_init_population(full_set)
    fitness = fitness_factory(full_set, s)
    while term_condition(population):
        # liczymy jak "dobry" jest każdy osobnik
        population_fit = [fitness(o) for o in population]
        # wybieramy ruletką osobniki do populacji
        selected_indexes = roulette_selection(population_fit)
        # krzyżowanie
        offspring = crossover(selected_indexes, population.copy())
        # mutacja
        offspring = mutation(offspring)
        # nowa populacja po krzyżowaniu i mutacji
        population = offspring
        if cost(best_element(population, fitness), full_set, s) == 0:
            return best_element(population, fitness), 0, iteration
    return best_element(population, fitness), cost(best_element(population, fitness), full_set, s), max_iterations


# full_set = [1, 2, 3, 6, 8, 2, 0, -3, 20, 15, 7, 3]
# s = 3
# x = generate_init_population(full_set)
# fitness = fitness_factory(full_set, s)
# print(x)
# print("-------")
# print(roulette_selection(fitness(x)))
# print("-------")
# print(two_point_crossover(roulette_selection(fitness(x)), x, 0.25))
# print("-------")
# print(genetic_algorithm(full_set, s, two_point_crossover, mutation_bit_flip, termination_max_iterations))

# print(mutation_bit_flip([[1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1], [1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1], [0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1], [0, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0], [0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1], [0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0], [1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1], [0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0], [0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0], [1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0], [1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0], [0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1], [0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0], [1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1], [0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1], [1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1], [0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1], [0, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 0], [0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0], [0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1]]))


# print(x)
# print(fitness(x))

def main(full_set, s, crossover, mutation):
    if crossover == "one-point":
        if mutation == "bit-flip":
            print(genetic_algorithm(full_set, s, one_point_crossover,
                                mutation_bit_flip, termination_max_iterations))
        elif mutation == "normal":
            print(genetic_algorithm(full_set, s, one_point_crossover,
                                mutation_normal, termination_max_iterations))
    elif crossover == "two-point":
        if mutation == "bit-flip":
            print(genetic_algorithm(full_set, s, two_point_crossover,
                                mutation_bit_flip, termination_max_iterations))
        elif mutation == "normal":
            print(genetic_algorithm(full_set, s, two_point_crossover,
                                mutation_normal, termination_max_iterations))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Subset sum problem solver - genetic algorithm.")

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--set", nargs="+", type=int, help="The set of numbers provided directly.")
    group.add_argument("--file", type=str, help="Path to a file containing the set of numbers.")

    parser.add_argument("--s", type=int, help="The target sum.")
    parser.add_argument("--crossover", choices=["one-point", "two-point"],
                        help="choose crossover method")
    parser.add_argument("--mutation", choices=["bit-flip", "normal"],
                                               help="choose mutation method")

    args = parser.parse_args()
    if args.file:
        input_set = read_set_from_file(args.file)
    else:
        input_set = args.set

    main(input_set, args.s, args.crossover, args.mutation)


