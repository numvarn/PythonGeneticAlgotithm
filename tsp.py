import pygad
import numpy as np
import matplotlib.pyplot as plt


# Create coordinates
city = 20
coords = np.random.rand(city, 2) * 2

last_fitness = 0

# Define Fitness Function


def fitness_func(solution, solution_idx):
    global coords

    fitness = 0
    for i in range(0, len(solution) - 1):
        coor1 = coords[solution[i]]
        coor2 = coords[solution[i + 1]]
        fitness += np.sqrt((coor1[0] - coor2[0]) **
                           2 + (coor1[1] - coor2[1]) ** 2)
    return fitness


def on_generation(ga_instance):
    global last_fitness
    print("Generation = {generation}".format(
        generation=ga_instance.generations_completed))

    print("Fitness    = {fitness}".format(fitness=ga_instance.best_solution(
        pop_fitness=ga_instance.last_generation_fitness)[1]))

    print("Change     = {change}".format(change=ga_instance.best_solution(
        pop_fitness=ga_instance.last_generation_fitness)[1] - last_fitness))

    last_fitness = ga_instance.best_solution(
        pop_fitness=ga_instance.last_generation_fitness)[1]


# User defined crossover function
def crossover_func(parents, offspring_size, ga_instance):
    offspring = []
    idx = 0
    while len(offspring) != offspring_size[0]:
        parent1 = parents[idx % parents.shape[0], :].copy()
        parent2 = parents[(idx + 1) % parents.shape[0], :].copy()

        random_split_point = np.random.choice(range(offspring_size[0]))

        tmp = np.array(
            [x for x in parent2 if x not in parent1[random_split_point:]])

        new_gene = np.append(parent1[random_split_point:], tmp)
        offspring.append(new_gene)

        idx += 1

    return np.array(offspring)


def plotGraph(solution):
    global coords
    x = []
    y = []

    for i in solution:
        x.append(coords[i][0])
        y.append(coords[i][1])

    labels = [str(val)+"("+str(key+1)+")" for key, val in enumerate(solution)]

    fix, ax = plt.subplots()
    ax.scatter(x, y)
    ax.plot(x, y)

    for i, txt in enumerate(labels):
        ax.annotate(txt, (x[i], y[i]))

    plt.show()


# Configure GA
num_generations = 300
num_parents_mating = 20  # city
sol_per_pop = 100  # Number of solutions in each generation
num_genes = city
gene_type = [int] * num_genes
keep_parents = 10

# Selection method
parent_selection_type = "sss"

# user defined crossover function
crossover_type = crossover_func

mutation_percent_genes = 20
mutation_type = "Swap"

# Initialize Population
pop_init = []
gene_space = [x for x in range(city)]
for i in range(sol_per_pop):
    pop_init.append(np.random.permutation(gene_space))

# Create GA object
ga_instance = pygad.GA(num_generations=num_generations,
                       num_parents_mating=num_parents_mating,
                       sol_per_pop=sol_per_pop,
                       initial_population=pop_init,
                       num_genes=num_genes,
                       gene_type=gene_type,
                       parent_selection_type=parent_selection_type,
                       keep_parents=keep_parents,
                       crossover_type=crossover_type,
                       mutation_type=mutation_type,
                       mutation_percent_genes=mutation_percent_genes,
                       fitness_func=fitness_func,
                       on_generation=on_generation)

print("\nInitialize Population")
print(ga_instance.initial_population)
print("\n---------------\n\n")

# Start GA
ga_instance.run()

# Returning the details of the best solution.
solution, solution_fitness, solution_idx = ga_instance.best_solution(
    ga_instance.last_generation_fitness)

print("\n\nParameters of the best solution : {solution}".format(
    solution=solution))
print("Fitness value of the best solution = {solution_fitness}".format(
    solution_fitness=solution_fitness))
print("Index of the best solution : {solution_idx}".format(
    solution_idx=solution_idx))

ga_instance.plot_fitness()
plotGraph(solution)
