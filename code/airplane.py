import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import sys
from scipy import stats
import statistics
import random
import copy
import math
import time


# Structure to store all the data needed per population member
class PopulationMember:
    def __init__(self, airplane):
        self.airplane = airplane
        self.cost = 0
        self.invert_cost = 0
        self.probability = 0


routes = np.array([0, 1, 2, 3, 4])
revenue_lost_per_passenger_turned_away = np.array([13, 20, 7, 7, 15])

passenger_demand_per_route = np.array([800, 900, 700, 650, 380])

number_of_aircraft_available_per_type = [10, 19, 25, 16]

aircraft_capacity_per_spot = np.matrix([[16, 10, 30, 23], [16, 10, 30, 23],
                                        [16, 10, 30, 23], [16, 10, 30, 23],
                                        [16, 10, 30, 23]])

operational_cost_per_spot = np.matrix([[12, 20, 30, 19], [2, 34, 10, 20],
                                       [43, 63, 40, 12], [32, 10, 6, 34],
                                       [20, 30, 10, 87]])

number_of_aircraft_per_spot = np.matrix([[0, 0, 0, 0], [0, 0, 0, 0],
                                         [0, 0, 0, 0], [0, 0, 0, 0],
                                         [0, 0, 0, 0]])


# Function to generate 1 column of a population member
def generate_column(row_nb, aircraft_type_count):
    column = np.zeros((row_nb))
    local_total = 0
    for index in range(0, row_nb):
        # pick a random number between 0 and the aircraft count
        val = random.randint(0, aircraft_type_count - local_total)
        local_total += val
        # if we are at the end but the total is not yet reached
        if aircraft_type_count != local_total and index == row_nb - 1:
            column[index] = aircraft_type_count - local_total
        # else, we add the random value to the airplane matrix
        else:
            column[index] = val
    return column


# Function to generate a population member
def generate_population_member():
    # get the dimension (row*column) of the number_of_aircraft_per_spot matrix)
    row_nb, column_nb = number_of_aircraft_per_spot.shape
    distribution = np.zeros((row_nb, column_nb))
    for index in range(0, column_nb):
        # get the aircraft count per type
        aircraft_type_count = number_of_aircraft_available_per_type[index]
        # generate the corresponding column and assign
        distribution[:, index] = generate_column(row_nb, aircraft_type_count)
    return distribution

# Function to generate the initial population
def generate_first_population(pop_count):
    all_population = []
    for index in range(0, pop_count):
        # generate a random distribution
        distribution = generate_population_member()
        all_population.append(PopulationMember(distribution))
    return all_population


# Function to computes the total revenu lost of one route
def get_revenu_lost_per_route(airplanes):
    revenu_lost = 0
    for index in routes:
        revenu_lost += max(
            ((passenger_demand_per_route[index] - np.multiply(
                airplanes[index], aircraft_capacity_per_spot[index]).sum()) *
             revenue_lost_per_passenger_turned_away[index]), 0)
    return revenu_lost

# The cost function
def cost_function(member):
    total_operating_cost = np.multiply(operational_cost_per_spot,
                                       member.airplane).sum()
    total_revenue_lost = get_revenu_lost_per_route(member.airplane)
    total_lost = total_operating_cost + total_revenue_lost
    return total_lost

# Function to compute the cost of all the population
def compute_costs(population):
    for index in range(0, len(population)):
        cost = cost_function(population[index])
        population[index].cost = cost
        population[index].invert_cost = 1 / cost
    return population

# Function to compute the picking probability according to the cost
def compute_probabilities(all_population):
    sum_cost = sum(p.invert_cost for p in all_population)
    for index in range(0, len(all_population)):
        probability = math.exp(
            math.exp((1 / all_population[index].cost) / sum_cost))
        all_population[index].probability = probability
    return all_population

# Function to pick the two best individual among the population
def pick_parents_elitist(all_population):
    parents = sorted(all_population, key=lambda k: k.cost)
    return [parents[0], parents[1]]

# Function to pick the two best individual among the population according to their probability
def pick_parents_proba(all_population):
    all_proba = list(map(lambda x: x.probability, all_population))
    return random.choices(all_population, weights=all_proba, k=2)


# Cross function
def cross(parent1, parent2, cross_threshold):
    # Copy both parents
    children = [
        PopulationMember(copy.deepcopy(parent1.airplane)),
        PopulationMember(copy.deepcopy(parent2.airplane))
    ]
    row_nb, column_nb = number_of_aircraft_per_spot.shape
    for index in range(0, column_nb):
        np.random.seed(10)
        rand = random.uniform(0, 1)
        # If the random is lower than the threshold, the two column are switched
        if rand <= cross_threshold:
            children[0].airplane[:, index] = copy.deepcopy(
                parent2.airplane[:, index])
            children[1].airplane[:, index] = copy.deepcopy(
                parent1.airplane[:, index])
    return children[0], children[1]


# Mutation function by recreating the whole column
def mutation_generate_column(child, mutation_threshold):
    row_nb, column_nb = number_of_aircraft_per_spot.shape
    for index in range(0, column_nb):
        np.random.seed(10)
        rand = random.uniform(0, 1)
        if rand <= mutation_threshold:
            aircraft_type_count = number_of_aircraft_available_per_type[index]
            child.airplane[:, index] = generate_column(row_nb,
                                                       aircraft_type_count)
    return child

# Mutation function by switching two values of the same column
def mutation_switch_one_element(child, mutation_threshold):
    row_nb, column_nb = number_of_aircraft_per_spot.shape
    child_copy = copy.deepcopy(child)
    for index in range(0, column_nb):
        np.random.seed(10)
        rand = random.uniform(0, 1)
        if rand <= mutation_threshold:
            index1 = random.randint(0, row_nb - 1)
            index2 = random.randint(0, row_nb - 1)
            while index1 == index2:
                index2 = random.randint(0, row_nb - 1)
            tmp = child_copy.airplane[index1, index]
            child_copy.airplane[index1, index] = child_copy.airplane[index2,
                                                                     index]
            child_copy.airplane[index2, index] = tmp
    return child_copy


# Function to regenerate a new population
def generate_new_population(all_population, cross_threshold,
                            mutation_threshold):
    new_pop = []
    for index in range(0, len(all_population)):
        parents = pick_parents_elitist(all_population)
        parent1 = copy.deepcopy(parents[0])
        parent2 = copy.deepcopy(parents[1])
        child1, child2 = cross(parent1, parent2, cross_threshold)
        np.random.seed(10)
        rand_cross = random.uniform(0, 1)
        rand_mutation = random.uniform(0, 1)
        if rand_cross >= 0.5:
            picked_child = child1
        else:
            picked_child = child2
        if rand_mutation >= 0.5:
            picked_child = mutation_switch_one_element(picked_child, mutation_threshold)
        new_pop.append(copy.deepcopy(picked_child))
    return new_pop


def generate_plot(generation, all_mins, initial_population_count,
                  cross_threshold, mutation_threshold):
    results = list(map(lambda x: x.cost, all_mins))
    print(all_mins[generation - 1].airplane)
    print(all_mins[generation - 1].cost)
    plt.plot(results)
    title = 'Total costs : ' + str(
        all_mins[generation - 1].cost) + '$' + ' | Generation : ' + str(
            generation) + ', Initial population : ' + str(
                initial_population_count)
    plt.xlabel(title)
    plt.savefig('.outputs/airplanes_g' + str(generation) + '_i' +
                str(initial_population_count) + '_cT' + str(cross_threshold) +
                '_mT' + str(mutation_threshold) + '.png')


def draw_table(mean_of_mins, std_of_mins, times_of_mins, row_labels,
               col_labels, title):

    fig = plt.figure(figsize=(12, 4))
    gs = GridSpec(nrows=2, ncols=2)

    ax1 = fig.add_subplot(gs[:, 1])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[0, 0])

    plt.set_cmap('Spectral')
    im = ax1.imshow(mean_of_mins, aspect='auto')
    im2 = ax2.imshow(mean_of_mins, aspect='auto')
    im3 = ax3.imshow(mean_of_mins, aspect='auto')

    ax1.set_title("Mean of cost function call to reach 19200")
    ax2.set_title("Standard deviation")
    ax3.set_title("Time (sec.)")

    # We want to show all ticks...
    ax1.set_xticks(np.arange(len(col_labels)))
    ax1.set_yticks(np.arange(len(row_labels)))
    # We want to show all ticks...
    ax2.set_xticks(np.arange(len(col_labels)))
    ax2.set_yticks(np.arange(len(row_labels)))
    # We want to show all ticks...
    ax3.set_yticks(np.arange(len(row_labels)))
    # ... and label them with the respective list entries
    ax1.set_xticklabels(list(map(lambda x: "p=" + str(x), col_labels)))
    ax1.set_yticklabels(list(map(lambda x: "g=" + str(x), row_labels)))
    # ... and label them with the respective list entries
    ax2.set_xticklabels(list(map(lambda x: "p=" + str(x), col_labels)))
    ax2.set_yticklabels(list(map(lambda x: "g=" + str(x), row_labels)))
    # ... and label them with the respective list entries
    ax3.set_yticklabels(list(map(lambda x: "g=" + str(x), row_labels)))

    # Rotate the tick labels and set their alignment.
    plt.setp(ax1.get_xticklabels(),
             rotation=45,
             ha="right",
             rotation_mode="anchor")
    # Rotate the tick labels and set their alignment.
    plt.setp(ax2.get_xticklabels(),
             rotation=45,
             ha="right",
             rotation_mode="anchor")
    # Rotate the tick labels and set their alignment.
    plt.setp(ax3.get_xticklabels(), visible=False)

    # Loop over data dimensions and create text annotations.
    for i in range(len(row_labels)):
        for j in range(len(col_labels)):
            text = ax1.text(j,
                            i,
                            mean_of_mins[i][j],
                            ha="center",
                            va="center",
                            color="black")
            text2 = ax2.text(j,
                             i,
                             std_of_mins[i][j],
                             ha="center",
                             va="center",
                             color="black")
            text3 = ax3.text(j,
                             i,
                             times_of_mins[i][j],
                             ha="center",
                             va="center",
                             color="black")
    plt.suptitle(title)
    print('matplotlib-table_' + str(title))
    plt.savefig('./outputs/matplotlib-table_' + str(title) + '.png',
                bbox_inches='tight',
                pad_inches=0.05,
                dpi=400)


def main():
    mins_of_mins = []
    std_of_mins = []
    times_of_mins = []
    calls_of_mins = []
    calls_of_mins_std = []
    generations = [10, 15]
    init_pops = [10, 15, 20, 25]
    reach = 19200
    cross_threshold = float(sys.argv[1])
    mutation_threshold = float(sys.argv[2])
    runs = 15
    # For each generations
    for gen in range(0, len(generations)):
        generation = generations[gen]
        all_mins_mean = []
        all_mins_std = []
        all_mins_times = []
        all_calls_of_pop = []
        all_calls_of_pop_std = []
        # For each population
        for init_p in range(0, len(init_pops)):
            initial_population_count = init_pops[init_p]
            start = time.time()
            mins_of_pop = []
            calls_of_pop = []
            # Run n times
            for n in range(0, runs):
                all_mins_of_the_current_gen = []
                # initial population
                all_population_of_the_current_gen = generate_first_population(
                    initial_population_count)
                compute_cost_call = 0
                print('Time :', n, 'and generation :', generation, 'for pop :',
                      initial_population_count)
                # From 0 to the size of the generation
                for gen_2 in range(0, generation):
                    # Compute the cost of all the population
                    all_population_of_the_current_gen = compute_costs(
                        all_population_of_the_current_gen)
                    # Get the minimum cost and add it into the list
                    min_cost = min(all_population_of_the_current_gen,
                                   key=lambda x: x.cost)
                    all_mins_of_the_current_gen.append(min_cost)
                    if (min_cost.cost >= reach):
                        compute_cost_call += 1
                    # Compute probabilities
                    all_population_of_the_current_gen = compute_probabilities(
                        all_population_of_the_current_gen)
                    # Pick parents
                    all_population_of_the_current_gen = generate_new_population(
                        all_population_of_the_current_gen, cross_threshold,
                        mutation_threshold)
                # Pick the last min (the best we get) and put it into the min of mins
                mins_of_pop.append(all_mins_of_the_current_gen[generation -
                                                               1].cost)
                # Pick the number of time the cost function have been called to reach the REACH value
                calls_of_pop.append(compute_cost_call)
            # After 10 tried, only keep the mean of all the mins
            print("###############")
            all_mins_mean.append(round(statistics.mean(mins_of_pop)))
            all_mins_std.append(round(statistics.stdev(mins_of_pop), 2))
            all_calls_of_pop.append(round(statistics.mean(calls_of_pop), 2))
            all_calls_of_pop_std.append(
                round(statistics.stdev(calls_of_pop), 2))
            all_mins_times.append(round((time.time() - start) / runs, 2))
        # Put the array of mins into the array containing all array of mins
        mins_of_mins.append(all_mins_mean)
        std_of_mins.append(all_mins_std)
        times_of_mins.append(all_mins_times)
        calls_of_mins.append(all_calls_of_pop)
        calls_of_mins_std.append(all_calls_of_pop_std)
        #generate_plot(generation, all_mins, initial_population_count, cross_threshold, mutation_threshold)
    draw_table(mins_of_mins,
               std_of_mins,
               times_of_mins,
               row_labels=generations,
               col_labels=init_pops,
               title='cT=' + str(cross_threshold) + ' & mT=' +
               str(mutation_threshold) + " (" + str(runs) + " runs)")


main()