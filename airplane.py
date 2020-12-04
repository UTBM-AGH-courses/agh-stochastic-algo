import numpy as np
import matplotlib.pyplot as plt
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

aircraft_capacity_per_spot = np.matrix([[16, 10, 30, 23],
                                        [16, 10, 30, 23],
                                        [16, 10, 30, 23],
                                        [16, 10, 30, 23],
                                        [16, 10, 30, 23]])

operational_cost_per_spot = np.matrix([[12, 20, 30, 19],
                                       [2, 34, 10, 20],
                                       [43, 63, 40, 12],
                                       [32, 10, 6, 34],
                                       [20, 30, 10, 87]])

number_of_aircraft_per_spot = np.matrix([[0, 0, 0, 0],
                                         [0, 0, 0, 0],
                                         [0, 0, 0, 0],
                                         [0, 0, 0, 0],
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
            airplanes[index], aircraft_capacity_per_spot[index]).sum()) * revenue_lost_per_passenger_turned_away[index])
        if revenu_lost_temp <= 0:
            revenu_lost_temp = 0
        revenu_lost += revenu_lost_temp
        string += 'Revenu lost on ' + str(index + 1) + ' : ' + str(revenu_lost_temp) + ' => (' + str(
            passenger_demand_per_route[index]) + ' - ' + str(
            np.multiply(airplanes[index], aircraft_capacity_per_spot[index]).sum()) + ')' + ' * ' + str(
            revenue_lost_per_passenger_turned_away[index]) + ' \n'
    return revenu_lost


def cost_function(member):
    total_operating_cost = np.multiply(
        operational_cost_per_spot, member.airplane).sum()
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
        probability = math.exp(math.exp((1 / all_population[index].cost) / sum_cost))
        all_population[index].probability = probability
    return all_population


def pick_parents(all_population):
    #all_proba = list(map(lambda x: x.probability, all_population))
    #return random.choices(all_population, weights=all_proba, k=2)
    parents = sorted(all_population, key=lambda k: k.cost)
    return [parents[0], parents[1]]

def cross(parent1, parent2, cross_threshold):
    children = [PopulationMember(copy.deepcopy(parent1.airplane)), PopulationMember(copy.deepcopy(parent2.airplane))]
    row_nb, column_nb = number_of_aircraft_per_spot.shape
    for index in range(0, column_nb):
        np.random.seed(10)
        rand = random.uniform(0, 1)
        if rand <= cross_threshold:
            children[0].airplane[:, index] = copy.deepcopy(parent2.airplane[:, index])
            children[1].airplane[:, index] = copy.deepcopy(parent1.airplane[:, index])
    return children[0], children[1]


def mutation2(child, mutation_threshold):
    row_nb, column_nb = number_of_aircraft_per_spot.shape
    for index in range(0, column_nb):
        np.random.seed(10)
        rand = random.uniform(0, 1)
        if rand <= mutation_threshold:
            aircraft_type_count = number_of_aircraft_available_per_type[index]
            child.airplane[:, index] = generate_column(
                row_nb, aircraft_type_count)
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
            child_copy.airplane[index1, index] = child_copy.airplane[index2, index]
            child_copy.airplane[index2, index] = tmp
    return child_copy


def generate_new_population(all_population, cross_threshold, mutation_threshold):
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


def main():
    initial_population_count = 30
    generation = 400
    cross_threshold = 0.8
    mutation_threshold = 0.8
    all_mins = []
    all_costs = []

    # initial population
    all_population = generate_first_population(initial_population_count)

    for gen in range(0, generation):
        # Get the cost
        all_population = compute_costs(all_population)
        all_costs.append(np.array(list(map(lambda x: x.cost, all_population))))
        all_mins.append(min(all_population, key=lambda x: x.cost))
        # Compute probabilities
        all_population = compute_probabilities(all_population)
        # Pick parents
        all_population = generate_new_population(
            all_population, cross_threshold, mutation_threshold)
        print(gen)

    results = list(map(lambda x: x.cost, all_mins))
    print(all_mins[generation - 1].airplane)
    print(all_mins[generation - 1].cost)
    plt.plot(results)
    title = 'Total costs : ' + str(all_mins[generation - 1].cost) + '$' + ' | Generation : ' + str(
        generation) + ', Initial population : ' + str(initial_population_count)
    plt.xlabel(title)
    plt.savefig('./airplanes_g' + str(generation) + '_i' + str(initial_population_count) +
                '_cT' + str(cross_threshold) + '_mT' + str(mutation_threshold) + '.png')

#main()

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
