import matplotlib.pyplot as plt
import random, time
from PIL import Image
from wfc import *

s = time.time()
tiles = get_tiles('sample/Flowers.png', (4, 4), flip_horizontal=True)
print(time.time()-s)

s = time.time()
adj = get_adjacent_tiles(tiles)
print(time.time()-s)

tile_hash = random.choice(list(tiles.keys()))

plt.imshow(tiles[tile_hash])
plt.show()

for adj_tile_hash in adj[tile_hash]['bottom']:
    plt.imshow(tiles[adj_tile_hash])
    plt.show()