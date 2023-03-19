import random

def weighted_choice(weights, values):
    """
    This function takes in two arguments:
    weights: a dictionary with keys as strings and values as integers, representing the weights for each value
    values: a list of values, each associated with a weight in the weights dictionary

    The function returns one randomly selected value from the values list, based on the weights provided in the weights dictionary.
    """
    total_weight = sum(weights.values())
    rand_num = random.uniform(0, total_weight)
    weight_sum = 0
    for i, val in enumerate(values):
        weight_sum += weights[val]
        if rand_num <= weight_sum:
            return val

weights = {'apple': 3, 'banana': 2, 'orange': 5}
values = ['apple', 'banana', 'orange']

selected_value = weighted_choice(weights, values)
print(selected_value)