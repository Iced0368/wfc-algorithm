#%%matplotlib inline
import matplotlib.pyplot as plt
import random, time
from PIL import Image
from wfc import *

model = WFCModel('sample/Knot.png', (5,5), flip_horizontal=True, flip_vertical=True, rotate=True)

print(len(model.tileset))
"""
print(random.choice(list(model.tileset.values())))
print(average_tiles(list(model.tileset.values())))

"""

model.generate((32, 32), True)

img = model.overwrite_tile()
if img is None:
    print("TT_TT")

else:
    print(img)
    plt.imshow(img)
    plt.show()


# %%
