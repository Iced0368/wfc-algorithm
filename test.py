import matplotlib.pyplot as plt
import random
from PIL import Image
from wfc import *

tiles = get_tiles('sample/Flowers.png', 5, 5)

print(tiles.keys())

im = Image.fromarray(random.choice(list(tiles.values())))
plt.imshow(im)
plt.show()