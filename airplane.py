import numpy as np
import matplotlib.pyplot as plt
import random

routes = np.array([0, 1, 2, 3, 4])
revenu_lost_per_passenger_turned_away = np.array([13, 20, 7, 7, 15])

passenger_demand_per_route = np.array([800, 900, 700, 650, 380])

number_of_aircraft_avalaible_per_type = [10, 19, 25, 16]

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
        val = random.randint(0, aircraft_type_count-local_total)
        local_total += val
        # if we are at the end but the total is not yet reached
        if (aircraft_type_count != local_total and index == row_nb-1):
            column[index] = aircraft_type_count-local_total
        # else, we add the random value to the airplane matrix
        else:
            column[index] = val
    return column

def generate_aircraft_distribution():
    # get the dimension (row*column) of the number_of_aircraft_per_spot matrix)
    row_nb, column_nb = number_of_aircraft_per_spot.shape
    distribution = np.zeros((row_nb, column_nb))
    for index in range(0, column_nb):
        # get the aircraft count per type
        aircraft_type_count = number_of_aircraft_avalaible_per_type[index]
        # generate the coresponding column and assign
        distribution[:, index] = generate_column(row_nb, aircraft_type_count)
    return distribution

def get_revenu_lost_per_route(airplanes):
    revenu_lost = 0
    string = ''
    for index in routes:
        revenu_lost_temp = ((passenger_demand_per_route[index] - np.multiply(
            airplanes[index], aircraft_capacity_per_spot[index]).sum()) * revenu_lost_per_passenger_turned_away[index])
        revenu_lost += revenu_lost_temp
        string += 'Revenu lost on ' + str(index+1) + ' : ' + str(revenu_lost_temp) + ' => (' + str(passenger_demand_per_route[index]) + ' - ' + str(np.multiply(airplanes[index], aircraft_capacity_per_spot[index]).sum()) + ')' + ' * ' + str(revenu_lost_per_passenger_turned_away[index]) + ' \n'
    return revenu_lost

def cost_function(airplanes):
    total_operating_cost = np.multiply(
        operational_cost_per_spot, airplanes).sum()
    total_revenu_lost = get_revenu_lost_per_route(airplanes)
    total_lost = total_operating_cost + total_revenu_lost
    return total_lost

def generate_first_population(pop_count):
    all_population = []
    for index in range(0, pop_count):
        # generate a random distribution
        distribution = generate_aircraft_distribution()
        all_population.append(distribution)
    return all_population

def compute_costs(population):
    all_costs = []
    all_invert_costs = []
    for index in range(0, len(population)):
        cost = cost_function(population[index])
        all_invert_costs.append(1/cost)
        all_costs.append(cost)
    return all_costs, all_invert_costs

def compute_probabilitis(all_costs, all_invert_costs):
    sum_cost = sum(all_invert_costs)
    all_proba = []
    for index in range(0, len(all_costs)):
        proba = (1/all_costs[index]) / sum_cost
        all_proba.append(proba)
    return all_proba

def pick_parents(all_population, all_proba):
    return random.choices(all_population, weights=all_proba, k=2)

def cross(parent1, parent2):
    childs = [parent1.copy(), parent2.copy()]
    row_nb, column_nb = number_of_aircraft_per_spot.shape
    for index in range(0, column_nb):
        rand = random.uniform(0, 1)
        cross_threshold = 0.1
        if (rand >= cross_threshold):
            #print('Switch column', index)
            childs[0][:, index] = parent2[:, index]
            childs[1][:, index] = parent1[:, index]
    return childs[0], childs[1]

def mutation(child):
    row_nb, column_nb = number_of_aircraft_per_spot.shape
    for index in range(0, column_nb):
        rand = random.uniform(0, 1)
        mutation_threshold = 0.8
        if (rand >= mutation_threshold):
            #print('Regenerate column', index)
            aircraft_type_count = number_of_aircraft_avalaible_per_type[index]
            child[:, index] = generate_column(row_nb, aircraft_type_count)
    return child

def mutation_all(population):
    row_nb, column_nb = number_of_aircraft_per_spot.shape
    for pop_index in range(0, len(population)):
        chance = (random.randint(1,100))/100
        pop_threshold = 0.30
        if (chance <= pop_threshold):
            for index in range(0, column_nb):
                rand = random.gauss(0, 0.4)
                mutation_threshold = 0
                if (rand >= mutation_threshold):
                    #print('Regenerate column', index)
                    aircraft_type_count = number_of_aircraft_avalaible_per_type[index]
                    population[pop_index][:, index] = generate_column(row_nb, aircraft_type_count)
    return population



def main():
    initial_population_count = 200
    generation = 100
    all_population = []
    all_costs = []
    all_invert_costs = []
    all_proba = []
    all_mins = []

    # initial population
    all_population = generate_first_population(initial_population_count)

    for gen in range(0, generation):
        # Get the cost
        all_costs, all_invert_costs = compute_costs(all_population)
        all_mins.append(min(all_costs))
        # Compute probalities
        all_proba = compute_probabilitis(all_costs, all_invert_costs)
        # Pick parents
        parents = pick_parents(all_population, all_proba)
        parent1 = parents[0]
        parent2 = parents[1]
        # Cross parents
        child1, child2 = cross(parent1, parent2)
        # Mutations 
        child1 = mutation(child1)
        child2 = mutation(child2)
        all_population.append(child1)
        all_population.append(child2)
        print(gen)

    plt.plot(all_mins)
    plt.ylabel('Total costs')
    plt.savefig('airplanes.png')

main()



