# Exam Scheduling using Genetic Algorithm

Oisin Mc Laughlin - 22441106 <br> Ciaran Gray 22427722

## Problem Description

The exam scheduling problem involves:
- **Exams**: Each exam must be scheduled in exactly one timeslot
- **Timeslots**: Limited number of available time periods
- **Students**: Each student is enrolled in multiple exams
- **Constraints**:
  - **Hard**: No student can have multiple exams in the same timeslot
  - **Soft**: Minimize adjacent timeslot assignments for each student

## Code Structure

#### Input/Output Functions
- `read_instance(filename)`: Parses exam scheduling instance files
  - Returns: (num_exams, num_timeslots, num_students, student_exams)
  - Input format: Header line with "n k m", followed by student-exam binary matrix

#### Genetic Algorithm Components
- `initialize_population(pop_size, num_exams, num_timeslots)`: Creates random initial solutions
- `evaluate_fitness(solution, student_exams, weight)`: Calculates fitness score
  - Hard violations weighted by `weight` parameter (default: 100)
  - Soft penalties for adjacent timeslots
- `select_parents(population, fitnesses, rng, tournament_size)`: Tournament selection
- `single_point_crossover(parent1, parent2, rng)`: Single-point crossover operator
- `mutate(individual, num_timeslots, mutation_rate, rng)`: Per-gene mutation

#### Main Algorithm
- `run_ga(...)`: Main genetic algorithm loop with configurable parameters
  - Population size, generations, selection pressure, crossover/mutation rates
  - Returns best solution, fitness, and convergence history

#### Analysis Functions
- `multiple_runs_analysis(...)`: Runs algorithm multiple times for consistency analysis
- `compute_violations(solution, student_exams)`: Detailed violation counting
- `plot_fitness_history(...)`: Visualizes algorithm convergence
- `plot_detailed_analysis(...)`: Creates comprehensive analysis plots
- `plot_multiple_runs_comparison(...)`: Compares results across instances

### File Format

Instance files contain:<br>
n k m           # num_exams, num_timeslots, num_students<br>
0 1 0 1 ...     # Student 1's exam enrollments <br>
1 0 1 0 ...     # Student 2's exam enrollments <br>
...

## Output

The algorithm provides:
- Best solution found (exam-to-timeslot assignments)
- Fitness score and cost breakdown
- Hard violations and soft penalties
- Convergence plots and analysis charts
- Statistical summaries for multiple runs

## Reference

Based on research from: https://dl.acm.org/doi/fullHtml/10.1145/3644479.3644500
