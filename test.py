#%%matplotlib inline
import matplotlib.pyplot as plt
import random, time
from PIL import Image
from wfc import *

model = WFCModel('sample/Knot.png', (5, 5), flip_horizontal=False, flip_vertical=False, rotate=True)

print(len(model.tileset))

"""
adj_size = [len(model.adjacency[tile_hash]['right'])for tile_hash in list(model.tileset.keys())]
adj_size.sort()
print(adj_size)

"""
s = time.time()
result = model.generate((10, 10), False)
print(time.time() - s)

if result:
    img = model.overwrite_tile()
    plt.imshow(img)
    plt.show()
    print(img)

else:
    print("TT_TT")



# %%
