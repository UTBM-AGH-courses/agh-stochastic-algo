import numpy as np

routes = np.array([0, 1, 2, 3, 4])
revenu_lost_per_passenger_turned_away = np.array([13, 20, 7, 7, 15])

passenger_demand_per_route = np.array([800, 900, 700, 650, 380])

number_of_aircraft_avalaible_per_type = [10, 19, 25, 16]

aircraft_capacity_per_spot = np.matrix([[16, 10, 16, 23],
                                        [20, 10, 14, 15],
                                        [57, 5, 20, 29],
                                        [9, 11, 22, 17],
                                        [35, 34, 34, 9]])

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


def generate_aircraft_population():
    # get the dimension (row*column) of the number_of_aircraft_per_spot matrix)
    row_nb, column_nb = number_of_aircraft_per_spot.shape
    for index in range(0, column_nb):
        aircraft_type_count = number_of_aircraft_avalaible_per_type[index]
        local_total = 0
        for index_2 in range(0, row_nb):
            # pick a random number between 0 and the aircraft count
            val = np.random.randint(0, aircraft_type_count)
            local_total += val
            # if we are at the end but the total is not yet reached
            if (aircraft_type_count != 0 and index_2 == row_nb-1):
                number_of_aircraft_per_spot[index_2,
                                            index] = number_of_aircraft_avalaible_per_type[index]-local_total
            # else, we add the random value to the airplane matrix
            else:
                number_of_aircraft_per_spot[index_2, index] = val
            # substract the total aircraft count with the random value picked
            aircraft_type_count -= val


def get_revenu_lost_per_route(airplanes):
    revenu_lost = 0
    string = ''
    for index in routes:
        revenu_lost_temp = ((passenger_demand_per_route[index] - np.multiply(
            airplanes[index], aircraft_capacity_per_spot[index]).sum()) * revenu_lost_per_passenger_turned_away[index])
        revenu_lost += revenu_lost_temp
        string += 'Revenu lost on ' + str(index+1) + ' : ' + str(revenu_lost_temp) + ' => (' + str(passenger_demand_per_route[index]) + ' - ' + str(np.multiply(airplanes[index], aircraft_capacity_per_spot[index]).sum()) + ')' + ' * ' + str(revenu_lost_per_passenger_turned_away[index]) + ' \n'
    print(string)
    return revenu_lost


def cost_function(airplanes):
    print('#########################################')
    print('Aircraft distribution : ')
    print(airplanes)
    print('Aircraft capacity : ')
    print(aircraft_capacity_per_spot)
    total_operating_cost = np.multiply(
        operational_cost_per_spot, airplanes).sum()
    total_revenu_lost = get_revenu_lost_per_route(airplanes)
    total_lost = total_operating_cost + total_revenu_lost
    print('Total revenu lost for people turned away from all the routes')
    print(total_revenu_lost)
    print('Total cost operation for the ' + str(airplanes.sum()) + ' aircraft')
    print(total_operating_cost.sum())
    print('\n')
    print('Total lost')
    print(total_lost)
    return total_lost

generate_aircraft_population()


total_cost = []
for tirage in range(0, 50000):
    rand1 = np.random.randint(0, 4)
    rand2 = np.random.randint(0, 4)
    while rand1 == rand2:
        rand2 = np.random.randint(0, 4)
    print('#########################################')
    print('Pick number', tirage, '| Switch line', rand1, 'and', rand2)
    number_of_aircraft_per_spot[[rand1, rand2]
                                ] = number_of_aircraft_per_spot[[rand2, rand1]]
    total_cost.append(cost_function(number_of_aircraft_per_spot))
    print('\n')


print('#########################################')
print('Min cost ==> ' + str(min(total_cost)))
print('#########################################')


# cost_function()
