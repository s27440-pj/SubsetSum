import itertools
import math
import random
import numpy as np
import argparse


# objective function - cost function (want to minimalize it)
def cost(subset, full_set, s):
    return abs(s - np.dot(subset, full_set))


def generate_neighbors(subset):
    neighbors = []
    for i in range(len(subset)):
        neighbor = subset.copy()
        neighbor[i] = 1 - neighbor[i]  # Changing 0 to 1, 1 to 0
        neighbors.append(neighbor)
    return neighbors


def random_subset(full_set):
    return [random.choice([0, 1]) for _ in range(len(full_set))]


def random_solution(full_set, s, max_iterations):
    subset = random_subset(full_set)
    for i in range(max_iterations):
        potential_subset = random_subset(full_set)
        if cost(potential_subset, full_set, s) < cost(subset, full_set, s):
            subset = potential_subset
        if cost(subset, full_set, s) == 0:
            return subset, 0, i

    return subset, cost(subset, full_set, s), max_iterations


def brute_force(full_set, s):
    n = len(full_set)
    best_subset = None
    best_cost = float('inf')

    for iteration, subset in enumerate(itertools.product([0, 1], repeat=n)):
        current_cost = cost(subset, full_set, s)
        if current_cost < best_cost:
            best_cost = current_cost
            best_subset = subset

            if best_cost == 0:
                return best_subset, 0, iteration

    return best_subset, best_cost, 2**len(full_set)


def hill_climbing(full_set, s, max_iterations):
    subset = random_subset(full_set)

    for i in range(max_iterations):
        neighbours = generate_neighbors(subset)
        # best neighbour - equals to loop for j in neighbours[1:] and comparing costs
        best_neighbour = min(neighbours, key=lambda neighbour: cost(neighbour, full_set, s))

        if cost(best_neighbour, full_set, s) < cost(subset, full_set, s):
            subset = best_neighbour
        else:
            return subset, cost(subset, full_set, s), i

        if cost(subset, full_set, s) == 0:
            return subset, 0, i

    return subset, cost(subset, full_set, s), max_iterations


def hill_climbing_stochastic(full_set, s, max_iterations):
    subset = random_subset(full_set)

    for i in range(max_iterations):
        neighbours = generate_neighbors(subset)
        random_neighbour = random.choice(neighbours)

        if cost(random_neighbour, full_set, s) <= cost(subset, full_set, s):
            subset = random_neighbour

        if cost(subset, full_set, s) == 0:
            return subset, 0, i

    return subset, cost(subset, full_set, s), max_iterations


def tabu_search(full_set, s, max_iterations, tabu_size):
    subset = random_subset(full_set)
    # on the lecture we were keeping all global best, but I think we don't need it
    global_best = subset.copy()
    tabu_list = [subset.copy()]

    for i in range(max_iterations):
        # all_neighbours = generate_neighbours(subset, full_set)
        # for n in neighbours:
        # if n not in tabu_list: neighbours.append(n)
        # neighbours, but only ones that are not on tabu_list
        neighbours = [neighbour for neighbour in generate_neighbors(subset)
                      if neighbour not in tabu_list]

        # if there are no neighbours - no sense in looking further
        if len(neighbours) == 0:
            break

        # best_neighbour = neighbours[0]
        # for j in neighbours[1:]:
        # if loss(j) < loss(best_neighbour): best_neighbour = j
        best_neighbour = min(neighbours, key=lambda neighbour: cost(neighbour, full_set, s))
        subset = best_neighbour
        tabu_list.append(best_neighbour)
        if isinstance(tabu_size, int) and tabu_size > 0:
            tabu_list = tabu_list[-tabu_size:]
        if cost(subset, full_set, s) < cost(global_best, full_set, s):
            global_best = subset.copy()
        if cost(subset, full_set, s) == 0:
            return subset, 0, i
    return global_best, cost(global_best, full_set, s), max_iterations


def tabu_search_returning(full_set, s, max_iterations, tabu_size):
    subset = random_subset(full_set)
    global_best = subset.copy()
    tabu_list = [subset.copy()]
    backup_list = []

    for i in range(max_iterations):
        neighbours = [neighbour for neighbour in generate_neighbors(subset)
                      if neighbour not in tabu_list]

        # if there are no neighbours - checking backup list
        if len(neighbours) == 0:
            print("backups: ", backup_list)
            if len(backup_list) > 1:
                subset = backup_list.pop()
                continue
            else:
                break

        if len(neighbours) > 1:
            backup_list.append(subset)

        best_neighbour = min(neighbours, key=lambda neighbour: cost(neighbour, full_set, s))
        subset = best_neighbour
        tabu_list.append(best_neighbour)

        if isinstance(tabu_size, int) and tabu_size > 0:
            tabu_list = tabu_list[-tabu_size:]
        if cost(subset, full_set, s) < cost(global_best, full_set, s):
            global_best = subset.copy()
        if cost(subset, full_set, s) == 0:
            return subset, 0, i

    return global_best, cost(global_best, full_set, s), max_iterations


# simulated annealing
def temperature(k):
    return 100.0/(k+1)


def neighbor_normal(subset):
    n = len(subset)
    # middle of subset, and n/6 - 99% of index will be between 0 and n, 3 sigma rule
    indexes_number = int(random.gauss(n/2, n/6))
    # number chosen from gauss can be smaller than 0 and greater than length of subset. This is to prevent it.
    indexes_number = max(0, min(n, indexes_number))
    neighbor = subset.copy()
    indices_to_flip = random.sample(range(len(subset)), indexes_number)
    for idx in indices_to_flip:
        neighbor[idx] = 1 - neighbor[idx]
    return neighbor


def sim_annealing(full_set, s, max_iterations):
    subset = random_subset(full_set)
    best_subset = subset.copy()

    for i in range(max_iterations):
        random_neighbour = neighbor_normal(subset)
        cost_subset = cost(subset, full_set, s)
        cost_neighbour = cost(random_neighbour, full_set, s)

        if cost_neighbour < cost_subset:
            subset = random_neighbour
        else:
            # (0,1)
            if random.random() < math.exp(-abs(cost_neighbour - cost_subset) / temperature(i)):
                subset = random_neighbour

        if cost(subset, full_set, s) < cost(best_subset, full_set, s):
            best_subset = subset

        if cost(subset, full_set, s) == 0:
            return subset, 0, i

    return best_subset, cost(subset, full_set, s), max_iterations


def read_set_from_file(file_path):
    with open(file_path, "r") as file:
        numbers = [int(num) for num in file.read().split()]
    return numbers


# print(sim_annealing([1, 2, 2, 2, 3, 4], 1, 100))
# print(brute_force([1, 2, 3], -1))


def main(full_set, s, algorithm, iterations, tabu_size=None):
    if algorithm == "random":
        print(random_solution(full_set, s, iterations))
    elif algorithm == "brute_force":
        print(brute_force(full_set, s))
    elif algorithm == "hill_climbing":
        print(hill_climbing(full_set, s, iterations))
    elif algorithm == "hill_climbing_stochastic":
        print(hill_climbing_stochastic(full_set, s, iterations))
    elif algorithm == "tabu":
        print(tabu_search(full_set, s, iterations, tabu_size))
    elif algorithm == "tabu_returning":
        print(tabu_search_returning(full_set, s, iterations, tabu_size))
    elif algorithm == "sim_annealing":
        print(sim_annealing(full_set, s, iterations))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Subset sum problem solver with metaheuristics.")

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--set", nargs="+", type=int, help="The set of numbers provided directly.")
    group.add_argument("--file", type=str, help="Path to a file containing the set of numbers.")

    parser.add_argument("--s", type=int, help="The target sum.")
    parser.add_argument("--algorithm", choices=["random", "brute_force", "hill_climbing", "hill_climbing_stochastic",
                                                "tabu", "tabu_returning", "sim_annealing"],
                        help="Algorithm to use.")
    parser.add_argument("--iterations", type=int, help="Maximum iterations in algorithm")
    parser.add_argument("--tabu_size", type=int, help="Maximum tabu size - can be none")

    args = parser.parse_args()
    if args.file:
        input_set = read_set_from_file(args.file)
    else:
        input_set = args.set

    main(input_set, args.s, args.algorithm, args.iterations,
         args.tabu_size if args.algorithm in ["tabu", "tabu_returning"] else None)
