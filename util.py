import random

drdc = {'top': (-1, 0), 'bottom': (1, 0), 'left': (0, -1), 'right': (0, 1)}

def dir_index(index, dir):
    dr, dc = drdc[dir]
    return (index[0]+dr, index[1]+dc)

    
def opposite(dir):
    if dir == 'top':
        return 'bottom'
    elif dir =='bottom':
        return 'top'
    elif dir =='left':
        return 'right'
    else:
        return 'left'
    
def weighted_choice(weights, values):
    total_weight = sum([weights[val] for val in values])
    rand_num = random.randint(0, total_weight)
    weight_sum = 0
    for val in values:
        weight_sum += weights[val]
        if rand_num <= weight_sum:
            return val