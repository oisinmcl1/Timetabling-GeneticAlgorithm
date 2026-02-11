"""
Genetic Algorithm for Exam Scheduling Problem
Oisin Mc Laughlin - 22441106
Ciaran Gray - 22427722
"""

import sys
import random
from typing import List, Tuple


def read_instance(filename: str) -> Tuple[int, int, int, List[List[int]]]:
    """
    Reads the exam scheduling instance from a file.
    :param filename: Instance file path.
    :return: A tuple containing: (num_exams, num_timeslots, num_students, student_exams)
    """
    student_exams: List[List[int]] = []

    with open(filename, "r", encoding="utf-8") as f:
        header_line = None

        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            header_line = line
            break

        if header_line is None:
            raise ValueError("File is empty or contains only comments/blank lines")

        parts = header_line.split()

        if len(parts) < 3:
            raise ValueError("Header must contain at least three integers: n k m")

        try:
            num_exams, num_timeslots, num_students = map(int, parts[:3])
        except ValueError as e:
            raise ValueError("Header values must be integers: n k m") from e

        # read the student rows
        for raw in f:
            line = raw.strip()

            if not line or line.startswith("#"):
                continue
            vals = line.split()

            try:
                ints = [int(v) for v in vals]
            except ValueError:
                raise ValueError(f"Student row contains non-integer values: {line}")

            exams = [idx for idx, val in enumerate(ints[:num_exams]) if val == 1]
            student_exams.append(exams)

    return num_exams, num_timeslots, num_students, student_exams


def initialize_population(pop_size: int, num_exams: int, num_timeslots: int) -> List[List[int]]:
    """
    Creates an initial population of random solutions. Each solution is a list of length `num_exams`
    :param pop_size: number of individuals in the population
    :param num_exams: number of exams (length of each solution)
    :param num_timeslots: number of timeslots (range of values for each gene in the solution)
    :return: A list of `pop_size` solutions, where each solution is a list of length `num_exams` with values in [0, num_timeslots-1]
    """
    if pop_size < 1:
        raise ValueError("pop_size must be >= 1")
    if num_exams < 1:
        raise ValueError("num_exams must be >= 1")
    if num_timeslots < 1:
        raise ValueError("num_timeslots must be >= 1")

    rng = random.Random()
    population: List[List[int]] = []

    for _ in range(pop_size):
        individual = [rng.randrange(num_timeslots) for _ in range(num_exams)]
        population.append(individual)

    return population


def evaluate_fitness(solution: List[int], student_exams: List[List[int]], weight) -> int:
    """
    Evaluates the fitness of a solution based on the number of hard violations and soft penalties.
    For each student, we count:
      - hard_violations: number of exams scheduled in the same timeslot (duplicates)
      - soft_penalty: number of adjacent occupied timeslot pairs (slot[i+1]
      hard violation will be weighted more than soft violations by multiplying hard_violations by `weight`)
    :param solution: A list of length `num_exams` where each value is the assigned timeslot for that exam.
    :param student_exams: A list of lists, where each inner list contains the exam indices that a student is taking.
    :param weight: The weight to apply to hard violations when calculating the fitness. Higher weight means hard violations are more costly.
    :return: The fitness value where higher is better.
    """
    hard_violations = 0
    soft_penalty = 0

    if not solution:
        return 0

    for exams in student_exams:
        if not exams:
            continue
        # get assigned slots for this student's exams

        try:
            slots = [solution[e] for e in exams]
        except IndexError:
            raise ValueError("Student exam index out of range of number of timeslots")

        # hard violations: duplicate slots
        hard_violations += len(slots) - len(set(slots))

        # soft penalty: count adjacent occupied timeslots in sorted distinct slots
        distinct_sorted = sorted(set(slots))

        for i in range(len(distinct_sorted) - 1):
            if distinct_sorted[i + 1] == distinct_sorted[i] + 1:
                soft_penalty += 1

    cost = weight * hard_violations + soft_penalty
    return -cost


def select_parents(population: List[List[int]], fitnesses: List[int], rng: random.Random, tournament_size: int) -> Tuple[List[int], List[int]]:
    """
    Selects two parents from the population using tournament selection.
    :param population: A list of solutions (individuals) in the current population.
    :param fitnesses: A list of fitness values corresponding to each individual in the population. Must be the same length as `population`.
    :param rng: A random number generator instance to use for selection randomness.
    :param tournament_size: The number of competitors to sample for each parent selection. Higher values increase selection pressure.
    :return: A tuple containing two selected parents (copies of the individuals from the population).
    """
    if not population:
        raise ValueError("Population is empty")
    pop_n = len(population)

    #radomly sample competitors for selection, and then pick the best fitness out of random samples
    def pick_one() -> List[int]:
        """
        Helper function to select one parent using tournament selection.
        :return: A copy of the selected parent solution.
        """
        k = min(tournament_size, pop_n)
        competitors = rng.sample(range(pop_n), k=k)
        best_idx = max(competitors, key=lambda idx: fitnesses[idx])
        return population[best_idx].copy()

    parent1 = pick_one()
    parent2 = pick_one()
    return parent1, parent2


def single_point_crossover(parent1: List[int], parent2: List[int], rng: random.Random) -> Tuple[List[int], List[int]]:
    """
    Performs single-point crossover between two parents to produce two children.
    :param parent1: A list of length `num_exams` representing the first parent solution.
    :param parent2: A list of length `num_exams` representing the second parent solution.
    :param rng: A random number generator instance to use for selecting the crossover point.
    :return: A tuple containing two child solutions resulting from the crossover. Each child is a list of length `num_exams`.
    """
    n = len(parent1)
    if n <= 1 or parent1 == parent2:
        return parent1.copy(), parent2.copy()

    point = rng.randrange(1, n)  # crossover point in [1, n-1]

    child1 = parent1[:point] + parent2[point:]
    child2 = parent2[:point] + parent1[point:]
    return child1, child2


def mutate(individual: List[int], num_timeslots: int, mutation_rate: float, rng: random.Random) -> None:
    """
    Performs in-place per-gene mutation on an individual. Each gene (exam assignment) has a chance to mutate based on `mutation_rate`.
    :param individual: A list of length `num_exams` representing a solution, where each value is the assigned timeslot for that exam. This list will be modified in-place.
    :param num_timeslots: The number of available timeslots. Mutated gene values will be in the range [0, num_timeslots-1].
    :param mutation_rate: The probability (between 0 and 1) that each gene will mutate. Higher values lead to more mutations.
    :param rng: A random number generator instance to use for mutation randomness.
    :return: None. The `individual` list is modified in-place.
    """
    if num_timeslots <= 1 or mutation_rate <= 0.0:
        return

    for i in range(len(individual)):
        if rng.random() < mutation_rate:
            old = individual[i]

            # pick a new timeslot different from current when possible
            new = rng.randrange(num_timeslots - 1)

            # map to value in [0, num_timeslots-1] skipping `old`
            if new >= old:
                new += 1
            individual[i] = new

def run_ga(num_exams: int,
           num_timeslots: int,
           student_exams: List[List[int]],
           pop_size: int,
           generations: int,
           tournament_size: int,
           elitism: bool,
           crossover_rate: float,
           mutation_rate: float) -> Tuple[List[int], int, List[int]]:
    """
    Runs the genetic algorithm for the exam scheduling problem.
    :param num_exams: The number of exams to schedule (length of each solution).
    :param num_timeslots: The number of available timeslots (range of values for each gene in the solution).
    :param student_exams: A list of lists, where each inner list contains the exam indices that a student is taking. Used for fitness evaluation.
    :param pop_size: The number of individuals in the population. Higher values increase diversity but also increase computation time.
    :param generations: The number of generations to run the algorithm for. More generations allow for more optimization but take more time.
    :param tournament_size: The number of competitors to sample for tournament selection. Higher values increase selection pressure towards fitter individuals.
    :param elitism: If True, the best solution from each generation is guaranteed to survive to the next generation. This can help preserve good solutions but may reduce diversity.
    :param crossover_rate: The probability (between 0 and 1) that crossover will be applied to selected parents. Higher values lead to more crossover and potentially faster exploration of the solution space.
    :param mutation_rate: The probability (between 0 and 1) that each gene in the offspring will mutate. Higher values lead to more mutations and increased diversity, but too high can disrupt good solutions.
    :return: A tuple containing: (best_solution, best_fitness, history)
    """
    # GA main loop with single-point crossover and per-gene mutation.
    rng = random.Random()

    # initialize
    population = initialize_population(pop_size, num_exams, num_timeslots)
    fitnesses = [evaluate_fitness(ind, student_exams, 100) for ind in population]

    best_idx = max(range(len(population)), key=lambda i: fitnesses[i])
    best_solution = population[best_idx].copy()
    best_fitness = fitnesses[best_idx]

    history: List[int] = [best_fitness]

    for gen in range(1, generations + 1):
        new_population: List[List[int]] = []

        # keep best solution if elitism flag is set to true
        if elitism:
            new_population.append(best_solution.copy())

        # fill the rest by selecting parents, applying crossover and mutation
        while len(new_population) < pop_size:
            p1, p2 = select_parents(population, fitnesses, rng, tournament_size=tournament_size)

            if rng.random() < crossover_rate:
                c1, c2 = single_point_crossover(p1, p2, rng)
            else:
                # no crossover: children are copies of parents
                c1, c2 = p1.copy(), p2.copy()

            # mutate children in-place
            mutate(c1, num_timeslots, mutation_rate, rng)
            if len(new_population) < pop_size:
                new_population.append(c1)
            if len(new_population) < pop_size:
                mutate(c2, num_timeslots, mutation_rate, rng)
                new_population.append(c2)

        # replace population and evaluate
        population = new_population
        fitnesses = [evaluate_fitness(ind, student_exams, 100) for ind in population]

        # update best
        gen_best_idx = max(range(len(population)), key=lambda i: fitnesses[i])
        gen_best_fitness = fitnesses[gen_best_idx]
        gen_best_solution = population[gen_best_idx].copy()

        if gen_best_fitness > best_fitness:
            best_fitness = gen_best_fitness
            best_solution = gen_best_solution.copy()

        history.append(best_fitness)

        print(f"Gen {gen}: best_fitness={best_fitness}")
        print("Number of unique individuals= ", len(set(tuple(ind) for ind in population)))

    return best_solution, best_fitness, history

if __name__ == "__main__":
    # instance = "test_case1.txt"
    instance = "small-2.txt"
    # instance = "medium-1.txt"

    # Genetic algorithm parameters
    pop = 200            # population size
    gens = 500           # number of generations
    cx = 0.9            # crossover rate
    mut = 0.1           # mutation rate (per gene)
    tour = 3             # tournament size
    elitism = True       # keep elite

    try:
        num_exams, num_timeslots, num_students, student_exams = read_instance(instance)
    except Exception as e:
        print(f"Error reading instance '{instance}': {e}")
        sys.exit(1)

    print(f"Instance: exams={num_exams}, timeslots={num_timeslots}, students={num_students}")
    print(f"Parsed {len(student_exams)} student rows")
    print(f"GA params: pop={pop}, gens={gens}, cx={cx}, mut={mut}, tour={tour}, elitism={elitism}")

    # run GA
    best_solution, best_fitness, history = run_ga(
        num_exams=num_exams,
        num_timeslots=num_timeslots,
        student_exams=student_exams,
        pop_size=pop,
        generations=gens,
        tournament_size=tour,
        elitism=elitism,
        crossover_rate=cx,
        mutation_rate=mut,
    )

    print("\n--- GA Result ---")
    print(f"Best fitness: {best_fitness}")
    print(f"Best solution (exam -> slot): {best_solution}")
    final_cost = -evaluate_fitness(best_solution, student_exams, 100)
    print(f"Final cost (100*hard + soft) = {final_cost}")


    def compute_violations(solution, student_exams):
        """
        Computes the number of hard violations and soft penalties for a given solution.
        :param solution: A list of length `num_exams` where each value is the assigned timeslot for that exam.
        :param student_exams: A list of lists, where each inner list contains the exam indices that a student is taking.
        :return: A tuple (hard_violations, soft_penalty)
        """
        hard = 0
        soft = 0
        for exams in student_exams:
            if not exams:
                continue
            slots = [solution[e] for e in exams]
            hard += len(slots) - len(set(slots))
            ds = sorted(set(slots))
            for i in range(len(ds) - 1):
                if ds[i + 1] == ds[i] + 1:
                    soft += 1
        return hard, soft

    hard_v, soft_p = compute_violations(best_solution, student_exams)
    print(f"Hard violations: {hard_v}, Soft penalty: {soft_p}")
    print("Fitness history (last 10):", history[-10:])
