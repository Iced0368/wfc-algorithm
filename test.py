#%%matplotlib inline
import matplotlib.pyplot as plt
import random, time
from PIL import Image
from wfc import *


s = time.time()
model = WFCModel('sample/MagicOffice.png', 
                 (3, 3), 
                 flip_horizontal=False, 
                 flip_vertical=False, 
                 rotate=True)
print(time.time()-s)

print(len(model.tileset))

# %%

show_process = 30
show_prop = False
seed = None

s = time.time()
result = model.generate((128, 64), show_process, show_prop, seed)
print(time.time() - s)

if result:
    img = model.overwrite_tile()
    if not show_process:
        plt.imshow(img)
        plt.axis('off')
        plt.show()
    print(img)

else:
    print("TT_TT")



# %%
model.view_performance()
model.view_log()


# %%
