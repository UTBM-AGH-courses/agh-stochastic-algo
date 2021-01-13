import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import statistics
import random
import copy
import math


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


def generate_initial_population():
    # get the dimension (row*column) of the number_of_aircraft_per_spot matrix)
    row_nb, column_nb = number_of_aircraft_per_spot.shape
    distribution = np.zeros((row_nb, column_nb))
    for index in range(0, column_nb):
        # get the aircraft count per type
        aircraft_type_count = number_of_aircraft_available_per_type[index]
        # generate the corresponding column and assign
        distribution[:, index] = generate_column(row_nb, aircraft_type_count)
    return distribution


def get_revenu_lost_per_route(airplanes):
    revenu_lost = 0
    string = ''
    for index in routes:
        revenu_lost_temp = ((passenger_demand_per_route[index] - np.multiply(
            airplanes[index], aircraft_capacity_per_spot[index]).sum()) *
                            revenue_lost_per_passenger_turned_away[index])
        if revenu_lost_temp <= 0:
            revenu_lost_temp = 0
        revenu_lost += revenu_lost_temp
        string += 'Revenu lost on ' + str(index + 1) + ' : ' + str(
            revenu_lost_temp) + ' => (' + str(
                passenger_demand_per_route[index]) + ' - ' + str(
                    np.multiply(airplanes[index],
                                aircraft_capacity_per_spot[index]).sum()
                ) + ')' + ' * ' + str(
                    revenue_lost_per_passenger_turned_away[index]) + ' \n'
    return revenu_lost


def cost_function(member):
    total_operating_cost = np.multiply(operational_cost_per_spot,
                                       member.airplane).sum()
    total_revenue_lost = get_revenu_lost_per_route(member.airplane)
    total_lost = total_operating_cost + total_revenue_lost
    return total_lost


def generate_first_population(pop_count):
    all_population = []
    for index in range(0, pop_count):
        # generate a random distribution
        distribution = generate_initial_population()
        all_population.append(PopulationMember(distribution))
    return all_population


def compute_costs(population):
    for index in range(0, len(population)):
        cost = cost_function(population[index])
        population[index].cost = cost
        population[index].invert_cost = 1 / cost
    return population


def compute_probabilities(all_population):
    sum_cost = sum(p.invert_cost for p in all_population)
    for index in range(0, len(all_population)):
        probability = math.exp(
            math.exp((1 / all_population[index].cost) / sum_cost))
        all_population[index].probability = probability
    return all_population


def pick_parents(all_population):
    #all_proba = list(map(lambda x: x.probability, all_population))
    #return random.choices(all_population, weights=all_proba, k=2)
    parents = sorted(all_population, key=lambda k: k.cost)
    return [parents[0], parents[1]]


def cross(parent1, parent2, cross_threshold):
    children = [
        PopulationMember(copy.deepcopy(parent1.airplane)),
        PopulationMember(copy.deepcopy(parent2.airplane))
    ]
    row_nb, column_nb = number_of_aircraft_per_spot.shape
    for index in range(0, column_nb):
        np.random.seed(10)
        rand = random.uniform(0, 1)
        if rand <= cross_threshold:
            children[0].airplane[:, index] = copy.deepcopy(
                parent2.airplane[:, index])
            children[1].airplane[:, index] = copy.deepcopy(
                parent1.airplane[:, index])
    return children[0], children[1]


def mutation2(child, mutation_threshold):
    row_nb, column_nb = number_of_aircraft_per_spot.shape
    for index in range(0, column_nb):
        np.random.seed(10)
        rand = random.uniform(0, 1)
        if rand <= mutation_threshold:
            aircraft_type_count = number_of_aircraft_available_per_type[index]
            child.airplane[:, index] = generate_column(row_nb,
                                                       aircraft_type_count)
    return child


def mutation(child, mutation_threshold):
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


def generate_new_population(all_population, cross_threshold,
                            mutation_threshold):
    new_pop = []
    for index in range(0, len(all_population)):
        parents = pick_parents(all_population)
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
            picked_child = mutation(picked_child, mutation_threshold)
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
    plt.savefig('./airplanes_g' + str(generation) + '_i' +
                str(initial_population_count) + '_cT' + str(cross_threshold) +
                '_mT' + str(mutation_threshold) + '.png')


def draw_table2(table_vals, row_labels, col_labels, title):
    fig, ax = plt.subplots()
    plt.set_cmap('Spectral')
    im = ax.imshow(table_vals, aspect='auto')

    # We want to show all ticks...
    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_yticks(np.arange(len(row_labels)))
    # ... and label them with the respective list entries
    ax.set_xticklabels(list(map(lambda x: "p=" + str(x), col_labels)))
    ax.set_yticklabels(list(map(lambda x: "g=" + str(x), row_labels)))

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(),
             rotation=45,
             ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(len(row_labels)):
        for j in range(len(col_labels)):
            text = ax.text(j,
                           i,
                           table_vals[i][j],
                           ha="center",
                           va="center",
                           color="w")
    plt.title(title)
    print('matplotlib-table_' + str(title))
    plt.savefig('matplotlib-table_' + str(title) + '.png',
                bbox_inches='tight',
                pad_inches=0.05,
                dpi=400)


def draw_table(table_vals, row_labels, col_labels, title):
    the_table = plt.table(
        cellText=table_vals,
        rowLabels=list(map(lambda x: "p=" + str(x), row_labels)),
        colLabels=list(map(lambda x: "g=" + str(x), col_labels)),
        loc='center')
    the_table.auto_set_font_size(False)
    the_table.set_fontsize(24)
    the_table.scale(6, 6)
    for pos in ['right', 'top', 'bottom', 'left']:
        plt.gca().spines[pos].set_visible(False)
    plt.tick_params(axis='x',
                    which='both',
                    bottom=False,
                    top=False,
                    labelbottom=False)
    plt.tick_params(axis='y',
                    which='both',
                    right=False,
                    left=False,
                    labelleft=False)
    plt.title(title)
    print('matplotlib-table_' + str(title))
    plt.savefig('matplotlib-table_' + str(title) + '.png',
                bbox_inches='tight',
                pad_inches=0.05)


def main():
    mins_of_mins = []
    generations = [5, 10, 20, 100, 200]
    init_pops = [5, 10, 20, 50, 80]
    cross_threshold = 0.7
    mutation_threshold = 0.25
    runs = 10
    for gen in range(0, len(generations)):
        all_mins_mean = []
        generation = generations[gen]
        mins_of_pop = []
        for init_p in range(0, len(init_pops)):
            initial_population_count = init_pops[init_p]
            for n in range(0, runs):
                all_mins_of_the_current_gen = []
                # initial population
                all_population_of_the_current_gen = generate_first_population(
                    initial_population_count)
                print('Time :', n, 'and generation :', generation, 'for pop :',
                      initial_population_count)
                for gen_2 in range(0, generation):
                    # Compute the cost of all the population
                    all_population_of_the_current_gen = compute_costs(
                        all_population_of_the_current_gen)
                    # Get the minimum cost and add it into the list
                    all_mins_of_the_current_gen.append(
                        min(all_population_of_the_current_gen,
                            key=lambda x: x.cost))
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
            # After 10 tried, only keep the mean of all the mins
            print("###############")
            all_mins_mean.append(round(statistics.mean(mins_of_pop)))
        # Put the array of mins into the array containing all array of mins
        mins_of_mins.append(all_mins_mean)
        #generate_plot(generation, all_mins, initial_population_count, cross_threshold, mutation_threshold)
    draw_table2(mins_of_mins,
                row_labels=generations,
                col_labels=init_pops,
                title='cT=' + str(cross_threshold) + ' & mT=' +
                str(mutation_threshold) + " (" + str(runs) + " runs)")


main()

# Reapeat 30 times and get the mean and standard deviation
# Classical vauss sucessing
# Algo ISS
# For generation:
#    pick 2 parents
#    cross them to get children
#    choose them by uniform proba
#    mutation it
#    add them in the next generation
# Increase threshold for c and m
# STart greater to cross dynamicly
# Then increase mutation if a big area without nothing dynamicly
# Check mistake in my code
# Compare when increase the population (previous version) ==> get the generation with the best fitness and compare
# Increase selection presure (better get better and worst get worst) ==> Scale values linear (x10000) OR exp (VERY DELICATE)
