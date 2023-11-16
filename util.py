import random

TOP, BOTTOM, LEFT, RIGHT = 0, 1, 2, 3
DIRECTIONS = [0, 1, 2, 3]
opposite = [1, 0, 3, 2]

drdc = [(-1, 0), (1, 0), (0, -1), (0, 1)]

def dir_index(index, dir):
    dr, dc = drdc[dir]
    return (index[0]+dr, index[1]+dc)


def weighted_choice(weights, values):
    total_weight = sum([weights[val] for val in values])
    rand_num = random.randint(0, total_weight)
    weight_sum = 0 
    for val in values:
        weight_sum += weights[val]
        if rand_num <= weight_sum:
            return val
