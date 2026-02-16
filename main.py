"""
Genetic Algorithm for Exam Scheduling Problem
Oisin Mc Laughlin - 22441106
Ciaran Gray - 22427722
"""

import sys
import random
from typing import List, Tuple
import matplotlib.pyplot as plt
from collections import Counter
import statistics


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
        gen_best_idx = max(range(len(population)), key=lambda i, fits=fitnesses: fits[i])
        gen_best_fitness = fitnesses[gen_best_idx]
        gen_best_solution = population[gen_best_idx].copy()

        if gen_best_fitness > best_fitness:
            best_fitness = gen_best_fitness
            best_solution = gen_best_solution.copy()

        history.append(best_fitness)

        print(f"Gen {gen}: best_fitness={best_fitness}")
        print("Number of unique individuals= ", len(set(tuple(ind) for ind in population)))

    return best_solution, best_fitness, history


def multiple_runs_analysis(instance_files: List[str],
                         num_runs: int = 10,
                         pop_size: int = 200,
                         generations: int = 500,
                         tournament_size: int = 3,
                         elitism: bool = True,
                         crossover_rate: float = 0.9,
                         mutation_rate: float = 0.1) -> dict:
    """
    Performs multiple runs of the genetic algorithm on different instances and computes summary statistics.

    :param instance_files: List of instance file names to test
    :param num_runs: Number of independent runs to perform for each instance
    :param pop_size: Population size for GA
    :param generations: Number of generations for GA
    :param tournament_size: Tournament size for selection
    :param elitism: Whether to use elitism
    :param crossover_rate: Crossover probability
    :param mutation_rate: Mutation probability per gene
    """
    print(f"Running consistency analysis with {num_runs} runs per instance...")
    print(f"GA Parameters: pop={pop_size}, gens={generations}, cx={crossover_rate}, mut={mutation_rate}")
    print("=" * 80)

    all_results = {}

    for instance in instance_files:
        print(f"\nAnalyzing instance: {instance}")
        print("-" * 40)

        try:
            num_exams, num_timeslots, num_students, student_exams = read_instance(instance)
            print(f"Instance details: exams={num_exams}, timeslots={num_timeslots}, students={num_students}")
        except Exception as e:
            print(f"Error reading instance '{instance}': {e}")
            continue

        # Store results for this instance
        fitness_results = []
        cost_results = []
        hard_violations_results = []
        soft_penalties_results = []

        print(f"Running {num_runs} independent trials...")

        for run in range(num_runs):
            print(f"  Run {run + 1}/{num_runs}...", end=" ")

            # Run GA with different random seed each time
            random.seed(run * 42)  # Different seed for each run

            best_solution, best_fitness, _ = run_ga(
                num_exams=num_exams,
                num_timeslots=num_timeslots,
                student_exams=student_exams,
                pop_size=pop_size,
                generations=generations,
                tournament_size=tournament_size,
                elitism=elitism,
                crossover_rate=crossover_rate,
                mutation_rate=mutation_rate,
            )

            # Compute detailed metrics
            final_cost = -best_fitness
            hard_violations, soft_penalties = compute_violations(best_solution, student_exams)

            # Store results
            fitness_results.append(best_fitness)
            cost_results.append(final_cost)
            hard_violations_results.append(hard_violations)
            soft_penalties_results.append(soft_penalties)

            print(f"Fitness: {best_fitness}, Cost: {final_cost}")

        # Compute statistics
        fitness_stats = {
            'mean': statistics.mean(fitness_results),
            'best': max(fitness_results),
            'worst': min(fitness_results),
            'std': statistics.stdev(fitness_results) if len(fitness_results) > 1 else 0.0,
            'median': statistics.median(fitness_results)
        }

        cost_stats = {
            'mean': statistics.mean(cost_results),
            'best': min(cost_results),  # Lower cost is better
            'worst': max(cost_results),
            'std': statistics.stdev(cost_results) if len(cost_results) > 1 else 0.0,
            'median': statistics.median(cost_results)
        }

        hard_stats = {
            'mean': statistics.mean(hard_violations_results),
            'best': min(hard_violations_results),
            'worst': max(hard_violations_results),
            'std': statistics.stdev(hard_violations_results) if len(hard_violations_results) > 1 else 0.0,
            'median': statistics.median(hard_violations_results)
        }

        soft_stats = {
            'mean': statistics.mean(soft_penalties_results),
            'best': min(soft_penalties_results),
            'worst': max(soft_penalties_results),
            'std': statistics.stdev(soft_penalties_results) if len(soft_penalties_results) > 1 else 0.0,
            'median': statistics.median(soft_penalties_results)
        }

        # Store all results
        all_results[instance] = {
            'fitness': fitness_stats,
            'cost': cost_stats,
            'hard_violations': hard_stats,
            'soft_penalties': soft_stats,
            'raw_data': {
                'fitness': fitness_results,
                'cost': cost_results,
                'hard_violations': hard_violations_results,
                'soft_penalties': soft_penalties_results
            }
        }

        # Print summary for this instance
        print(f"\nSummary Statistics for {instance}:")
        print(f"  Fitness    - Mean: {fitness_stats['mean']:.2f}, Best: {fitness_stats['best']}, Worst: {fitness_stats['worst']}, Std: {fitness_stats['std']:.2f}")
        print(f"  Total Cost - Mean: {cost_stats['mean']:.2f}, Best: {cost_stats['best']}, Worst: {cost_stats['worst']}, Std: {cost_stats['std']:.2f}")
        print(f"  Hard Viol. - Mean: {hard_stats['mean']:.2f}, Best: {hard_stats['best']}, Worst: {hard_stats['worst']}, Std: {hard_stats['std']:.2f}")
        print(f"  Soft Pen.  - Mean: {soft_stats['mean']:.2f}, Best: {soft_stats['best']}, Worst: {soft_stats['worst']}, Std: {soft_stats['std']:.2f}")

    # Print overall comparison
    print("\n" + "=" * 80)
    print("OVERALL COMPARISON ACROSS INSTANCES")
    print("=" * 80)

    # Create comparison table
    print(f"{'Instance':<15} {'Fitness (Best)':<15} {'Cost (Best)':<12} {'Hard Viol.':<12} {'Std Dev':<10}")
    print("-" * 70)

    for instance, results in all_results.items():
        print(f"{instance:<15} {results['fitness']['best']:<15} {results['cost']['best']:<12} {results['hard_violations']['best']:<12} {results['fitness']['std']:<10.2f}")

    # Plot comparison charts
    plot_multiple_runs_comparison(all_results)

    return all_results


def compute_violations(solution: List[int], student_exams: List[List[int]]) -> Tuple[int, int]:
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


def plot_multiple_runs_comparison(results_dict: dict) -> None:
    """
    Creates comparison plots for multiple runs analysis.
    :param results_dict: Dictionary containing results from multiple_runs_analysis
    """
    instances = list(results_dict.keys())

    # Create subplots for different metrics
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Multiple Runs Analysis - Consistency Comparison', fontsize=16)

    metrics = ['fitness', 'cost', 'hard_violations', 'soft_penalties']
    titles = ['Fitness Distribution', 'Cost Distribution', 'Hard Violations Distribution', 'Soft Penalties Distribution']

    for i, (metric, title) in enumerate(zip(metrics, titles)):
        ax = axes[i // 2, i % 2]

        # Box plot for each instance
        data_for_plot = []
        labels = []

        for instance in instances:
            data_for_plot.append(results_dict[instance]['raw_data'][metric])
            labels.append(instance.replace('.txt', ''))

        box_plot = ax.boxplot(data_for_plot, tick_labels=labels, patch_artist=True)

        # Color the boxes
        colors = ['blue', 'green', 'red']
        for patch, color in zip(box_plot['boxes'], colors[:len(instances)]):
            patch.set_facecolor(color)

        ax.set_title(title)
        ax.set_ylabel(metric.replace('_', ' ').title())
        ax.grid(True, alpha=0.3)

        # Add mean markers
        for j, instance in enumerate(instances):
            mean_val = results_dict[instance][metric]['mean']
            ax.plot(j + 1, mean_val, markersize=8, label='Mean' if i == 0 and j == 0 else "")

    # Add legend to the first subplot
    axes[0, 0].legend()

    plt.tight_layout()
    plt.show()

    # Create a separate convergence plot showing fitness evolution for best runs
    plt.figure(figsize=(12, 8))

    for instance in instances:
        # For demonstration, we would need to store fitness histories from each run
        # This is a simplified version showing the concept
        plt.plot(range(10), [results_dict[instance]['fitness']['best']] * 10,
                label=f"{instance.replace('.txt', '')} (Best: {results_dict[instance]['fitness']['best']})",
                linewidth=2)

    plt.title('Best Fitness Achieved by Instance')
    plt.xlabel('Generation (Illustrative)')
    plt.ylabel('Fitness')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_fitness_history(history: List[int], title: str = "Genetic Algorithm Fitness Evolution") -> None:
    """
    Plots the fitness evolution over generations.
    :param history: List of best fitness values for each generation
    :param title: Title for the plot
    """
    plt.figure(figsize=(10, 6))
    plt.plot(history, linewidth=2, color='blue')
    plt.title(title)
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_detailed_analysis(history: List[int],
                         best_solution: List[int],
                         instance_name: str) -> None:
    """
    Plots detailed analysis including fitness evolution and solution statistics.
    :param history: List of best fitness values for each generation
    :param best_solution: The best solution found
    :param student_exams: Student exam assignments for computing violations
    :param instance_name: Name of the instance being solved
    """

    # Plot 1: Fitness evolution
    plt.figure(figsize=(10, 6))
    plt.plot(history, linewidth=2, color='blue')
    plt.title(f'Fitness Evolution - {instance_name}')
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # Plot 2: Solution distribution (timeslot usage)
    plt.figure(figsize=(10, 6))
    slot_counts = Counter(best_solution)
    slots = list(slot_counts.keys())
    counts = list(slot_counts.values())

    plt.bar(slots, counts, color='lightcoral', alpha=0.7)
    plt.title(f'Timeslot Usage Distribution - {instance_name}')
    plt.xlabel('Timeslot')
    plt.ylabel('Number of Exams')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # Plot 3: Convergence analysis (last 50% of generations)
    plt.figure(figsize=(10, 6))
    mid_point = len(history) // 2
    if mid_point > 0:
        convergence_data = history[mid_point:]
        plt.plot(range(mid_point, len(history)), convergence_data, linewidth=2, color='green')
        plt.title(f'Convergence Analysis (Last 50%) - {instance_name}')
        plt.xlabel('Generation')
        plt.ylabel('Fitness')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    instance_files = ["test_case1.txt", "small-2.txt", "medium-1.txt"]

    # Genetic algorithm parameters
    pop = 200            # population size
    gens = 500           # number of generations
    cx = 0.9            # crossover rate
    mut = 0.1           # mutation rate (per gene)
    tour = 3             # tournament size
    elitism = True       # keep elite

    # print("Multiple Runs Consistency Analysis")
    # print("Testing algorithm consistency across multiple runs and instances")
    # print()

    # Run consistency analysis on all three instances
    # all_results = multiple_runs_analysis(
    #     instance_files=instance_files,
    #     num_runs=3,  # 3 runs per instance for demonstration
    #     pop_size=pop,
    #     generations=gens,
    #     tournament_size=tour,
    #     elitism=elitism,
    #     crossover_rate=cx,
    #     mutation_rate=mut
    # )

    print("\nIndividual Instance Analysis")
    print("Running detailed analysis on individual instances...")

    # Run detailed analysis on one instance for demonstration
    # instance = "test_case1.txt"
    instance = "small-2.txt"
    # instance = "medium-1.txt"

    try:
        num_exams, num_timeslots, num_students, student_exams = read_instance(instance)
    except Exception as e:
        print(f"Error reading instance '{instance}': {e}")
        sys.exit(1)

    print(f"\nDetailed analysis for: {instance}")
    print(f"Instance: exams={num_exams}, timeslots={num_timeslots}, students={num_students}")
    print(f"Parsed {len(student_exams)} student rows")
    print(f"GA params: pop={pop}, gens={gens*5}, cx={cx}, mut={mut}, tour={tour}, elitism={elitism}")

    # run GA with more generations for detailed analysis
    best_solution, best_fitness, history = run_ga(
        num_exams=num_exams,
        num_timeslots=num_timeslots,
        student_exams=student_exams,
        pop_size=pop,
        generations=gens*5,  # More generations for detailed analysis
        tournament_size=tour,
        elitism=elitism,
        crossover_rate=cx,
        mutation_rate=mut,
    )

    print("\n--- Detailed GA Result ---")
    print(f"Best fitness: {best_fitness}")
    print(f"Best solution (exam -> slot): {best_solution}")
    final_cost = -evaluate_fitness(best_solution, student_exams, 100)
    print(f"Final cost (100*hard + soft) = {final_cost}")

    hard_v, soft_p = compute_violations(best_solution, student_exams)
    print(f"Hard violations: {hard_v}, Soft penalty: {soft_p}")
    print("Fitness history (last 10):", history[-10:])

    # Plot the detailed results
    plot_detailed_analysis(history, best_solution, instance)
