import numpy as np

tileset = [1, 2, 3]
width = 5
height = 3

arr = np.empty((height, width), dtype=object)
for i in range(height):
    for j in range(width):
        arr[i, j] = tileset.copy()

print(arr)