import sys
import argparse
import random
from typing import List, Tuple


def read_instance(filename: str) -> Tuple[int, int, int, List[List[int]]]:
    """Read an instance file and return (num_exams, num_timeslots, num_students, student_exams).
    student_exams is a list of lists where each inner list contains exam indices (0-based)
    that the student is registered for.
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



def parse_args():
    p = argparse.ArgumentParser(description="Exam Timetabling GA")
    p.add_argument("instance", nargs="?", default="/Users/ciarangray/Desktop/4th year/AI/A1/test_case1.txt", help="instance file (default: /Users/ciarangray/Desktop/4th year/AI/A1/test_case1.txt)")
    return p.parse_args()


def initialize_population(pop_size: int, num_exams: int, num_timeslots: int, seed: int = None) -> List[List[int]]:
    """Create a population of random solutions.
      seed: optional RNG seed for reproducibility

    Returns:
      A list of `pop_size` individuals. Each individual is a list of length
      `num_exams` where each gene is an int in [0, num_timeslots-1].
    """
    if pop_size < 1:
        raise ValueError("pop_size must be >= 1")
    if num_exams < 1:
        raise ValueError("num_exams must be >= 1")
    if num_timeslots < 1:
        raise ValueError("num_timeslots must be >= 1")

    rng = random.Random(seed)
    population: List[List[int]] = []
    for _ in range(pop_size):
        individual = [rng.randrange(num_timeslots) for _ in range(num_exams)]
        population.append(individual)
    return population


def evaluate_fitness(solution: List[int], student_exams: List[List[int]], weight) -> int:
    """Evaluate a solution and return fitness following the pseudocode.

    For each student:
      - hard_violations: number of exams scheduled in the same timeslot (duplicates)
      - soft_penalty: number of adjacent occupied timeslot pairs (slot[i+1] == slot[i] + 1)

      *Hard violation will be weighted more than soft violations
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
            raise ValueError("Student exam index out of range of solution length")

        # hard violations: duplicate slots
        hard_violations += len(slots) - len(set(slots))

        # soft penalty: count adjacent occupied timeslots in sorted distinct slots
        distinct_sorted = sorted(set(slots))
        for i in range(len(distinct_sorted) - 1):
            if distinct_sorted[i + 1] == distinct_sorted[i] + 1:
                soft_penalty += 1

    cost = weight * hard_violations + soft_penalty
    return -cost


def select_parents(population: List[List[int]], fitnesses: List[int], rng: random.Random, tournament_size: int = 3) -> Tuple[List[int], List[int]]:
    """Select two parents using tournament selection.

    Picks `tournament_size` competitors for each parent and returns deep copies of
    the two winners. If the population is small, sampling is adjusted accordingly.
    """
    if not population:
        raise ValueError("Population is empty")
    pop_n = len(population)

    #radomly sample competitors for selection, and then pick the best fitness out of random samples
    def pick_one() -> List[int]:
        k = min(tournament_size, pop_n)
        competitors = rng.sample(range(pop_n), k=k)
        best_idx = max(competitors, key=lambda idx: fitnesses[idx])
        return population[best_idx].copy()

    parent1 = pick_one()
    parent2 = pick_one()
    return parent1, parent2


def run_ga(num_exams: int,
           num_timeslots: int,
           student_exams: List[List[int]],
           pop_size: int,
           generations: int,
           tournament_size: int,
           elitism: bool,
           seed: int) -> Tuple[List[int], int, List[int]]:
    """Basic GA main loop (first iteration):

    - Initializes a random population
    - Evaluates fitness
    - Repeatedly builds new populations by selecting parents and copying them
      into the new population (no crossover or mutation implemented in this
      first iteration).

    Returns: (best_solution, best_fitness, fitness_history)
    """
    rng = random.Random(seed)

    # initialize
    population = initialize_population(pop_size, num_exams, num_timeslots, seed=seed)
    fitnesses = [evaluate_fitness(ind, student_exams, 100) for ind in population]

    best_idx = max(range(len(population)), key=lambda i: fitnesses[i])
    best_solution = population[best_idx].copy()
    best_fitness = fitnesses[best_idx]

    history: List[int] = [best_fitness]

    for gen in range(1, generations + 1):
        new_population: List[List[int]] = []

        # keep elite
        if elitism:
            new_population.append(best_solution.copy())

        # fill the rest by selecting parents and copying one of them
        while len(new_population) < pop_size:
            p1, p2 = select_parents(population, fitnesses, rng, tournament_size=tournament_size)
            #no crossover or mtation, just pik a parent
            chosen = p1 if rng.random() < 0.5 else p2
            new_population.append(chosen.copy())

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

    return best_solution, best_fitness, history


if __name__ == "__main__":
    args = parse_args()

    try:
        num_exams, num_timeslots, num_students, student_exams = read_instance(args.instance)
    except Exception as e:
        print(f"Error reading instance '{args.instance}': {e}")
        sys.exit(1)

    print(f"Instance: exams={num_exams}, timeslots={num_timeslots}, students={num_students}")
    print(f"Parsed {len(student_exams)} student rows")
