import numpy as np
import matplotlib.pyplot as plt
import random
import sys
import time
from IPython.display import clear_output
from PIL import Image

from wfc_tiling import get_tiles, get_adjacent_tiles
from util import opposite, dir_index, weighted_choice, DIRECTIONS

sys.setrecursionlimit(10**7)


class WFCModel:
    def __init__(self, image_path, tile_size, 
                 flip_horizontal=False, 
                 flip_vertical=False, 
                 rotate=False):
        
        self.tile_size = tile_size

        self.tileset, self.weights = get_tiles(image_path, tile_size, flip_horizontal, flip_vertical, rotate)
        self.adjacency = get_adjacent_tiles(self.tileset)

        self.tileset_middle = set([i for i in range(len(self.tileset))])
        self.tileset_edge = {dir: set() for dir in DIRECTIONS}

        for i in range(len(self.tileset)):
            for dir in DIRECTIONS:
                if len(self.adjacency[i][dir]) == 0:
                    self.tileset_middle.remove(i)
                    self.tileset_edge[dir].add(i)

        self.average_color = np.array([self.tileset[i][0][0] for i in range(len(self.tileset))]).mean(axis=0)

        self.superposition = None
        self.possible_patterns = None
        self.grid_size = None
        self.prop_state = None

        self.performance = None
        self.log = None
        print('model initialization finised')


    def init_superposition(self, size):
        width, height = size
        self.grid_size = (height, width)
        
        superpos = np.empty((height, width), dtype=object)
        pos_pat = np.empty((height, width), dtype=object)

        for i in range(height):
            for j in range(width):
                superpos[i, j] = self.tileset_middle.copy()
                pos_pat[i, j] = {
                    'top': self.tileset_middle.copy(), 
                    'bottom': self.tileset_middle.copy(), 
                    'left': self.tileset_middle.copy(), 
                    'right': self.tileset_middle.copy()
                }
                """
                if i == 0:
                    pos_pat[i, j]['top'] |= self.tileset_edge['top']
                if i == height-1:
                    pos_pat[i, j]['bottom'] |= self.tileset_edge['bottom']
                if j == 0:
                    pos_pat[i, j]['left'] |= self.tileset_edge['left']
                if j == width-1:
                    pos_pat[i, j]['right'] |= self.tileset_edge['right']
                """
        
        self.superposition = superpos
        self.possible_patterns = pos_pat
        self.prop_state = {}
        self.performance = {'Propagate': [], 'Visualize': []}
        self.log = {'Observed': []}
        print('superposition initialization finised')


    def is_valid_index(self, r, c):
        return r >= 0 and c >= 0 and r < self.grid_size[0] and c < self.grid_size[1]

    def hash_index(self, r, c):
        return self.grid_size[1] * r + c
    
    def decode_index(self, n):
        return (n // self.grid_size[1], n % self.grid_size[1])

    def get_minimum_entropy(self):
        min_entropy = 999999
        observed_cnt = 0
        result = []
        for r in range(self.grid_size[0]):
            for c in range(self.grid_size[1]):
                if len(self.superposition[r, c]) == 1:
                    observed_cnt += 1
                    continue
                if len(self.superposition[r, c]) < min_entropy:
                    result = [(r, c)]
                    min_entropy = len(self.superposition[r][c])
                elif len(self.superposition[r, c]) == min_entropy:
                    result.append((r, c))
        self.log['Observed'].append(observed_cnt)
        return (min_entropy, result)
    

    def propagate(self, r, c, dir):
        ar, ac = dir_index((r, c), dir)
        if not self.is_valid_index(ar, ac):
            return                      
        index_hash = self.hash_index(ar, ac)
        if index_hash in self.prop_state:
            self.prop_state[index_hash].append(opposite(dir))
        else:
            self.prop_state[index_hash] = [opposite(dir)]


    def collapse(self, r, c, index):
        self.superposition[r, c] = set([index])
        for dir in DIRECTIONS:
            self.possible_patterns[r, c][dir] = self.adjacency[index][dir]
        for dir in DIRECTIONS:
            self.propagate(r, c, dir)

        
    def update_superposition(self, r, c, updated_from):
        prev_len = len(self.superposition[r, c])

        changed = False
        # Superposition Update
        for dir in updated_from:
            ar, ac = dir_index((r, c), dir)
            if self.is_valid_index(ar, ac):
                self.superposition[r, c] &= self.possible_patterns[ar, ac][opposite(dir)]
        
        if len(self.superposition[r, c]) < prev_len:
            changed = True

        if not changed: # If nothing changed in Superposition, Not need to update Adjacency
            return
        
        changed = False
        # Adjacency Update
        for dir in DIRECTIONS:
            pos_pat = set()
            for tile_hash in self.superposition[r, c]:
                pos_pat |= self.adjacency[tile_hash][dir]
            if len(pos_pat) < len(self.possible_patterns[r, c][dir]):
                changed = True
                self.possible_patterns[r, c][dir] = pos_pat

        if not changed: # If nothing changed in Adjacency, Not need to propagate
            return

        # Propagate
        for dir in DIRECTIONS:
            #if dir in updated_from: # Not need to propagte to Updated-from
            #    continue
            self.propagate(r, c, dir)
    

    def update_wave(self):
        items = list(self.prop_state.items())
        self.prop_state = {}
        for index_hash, updated_from in items:
            r, c = self.decode_index(index_hash)
            if self.is_valid_index(r, c):
                self.update_superposition(r, c, updated_from)
        
        #self.show_entropy()


    def next_step(self, show_prop=False):
        entropy, indices = self.get_minimum_entropy()
        if entropy == 0:
            return -1
        if entropy == 999999:
            return 1
        
        index = random.choice(indices)
        tile_index = weighted_choice(self.weights, list(self.superposition[index[0], index[1]]))

        self.collapse(index[0], index[1], tile_index)

        while len(self.prop_state) > 0:
            self.update_wave()

            if show_prop:
                clear_output(wait=True)
                self.view_prop_wave()
             
        return 0


    def generate(self, size, show_process=False, show_prop=False):
        self.init_superposition(size)
        result = 0
        step = 0

        while result == 0:
            step += 1
            s = time.time()
            result = self.next_step(show_prop)
            self.performance['Propagate'].append(time.time()-s)

            if show_process and (step % show_process == 0 or result == 1):
                s = time.time()
                img = self.overwrite_tile()
                if img is None:
                    return False
                clear_output(wait=True)
                plt.imshow(img)
                plt.axis('off')
                plt.show()
                self.performance['Visualize'].append(time.time()-s)

        return result > 0
    

    def overwrite_tile(self):
        # Create the output image with the specified size
        #output_size = ((self.tile_size[0]-1)*(self.grid_size[0])+1, (self.tile_size[1]-1)*(self.grid_size[1])+1)
        output_size = (self.grid_size[0]+self.tile_size[0]-1, self.grid_size[1]+self.tile_size[1]-1)
        output_image = Image.new('RGB', output_size, (0, 0, 0))

        output_size = (self.grid_size[0], self.grid_size[1], 3)
        output_image = np.zeros(output_size, dtype=np.uint8)

        # Loop over the grid and paste each tile onto the output image
        for i in range(self.grid_size[0]):
            for j in range(self.grid_size[1]):
                superpos = list(self.superposition[i, j])
                if len(superpos) == 0:
                    return None
                elif len(superpos) > 1:
                    if len(superpos) >= 0.9 * len(self.tileset):
                        color = self.average_color
                    else:
                        color = np.array([self.tileset[i][0][0] for i in superpos]).mean(axis=0)
                else:
                    color = self.tileset[superpos[0]][0][0]

                output_image[i][j] = color
        return Image.fromarray(output_image)


    def show_entropy(self):
        for r in range(self.grid_size[0]):
            for c in range(self.grid_size[1]):
                print(str(len(self.superposition[r, c])).rjust(4), end='')
            print()
        print()


    def view_performance(self):
        for key, value in self.performance.items():
            plt.plot(value, label=key)
        plt.legend()
        plt.show()

        for key, value in self.performance.items():
            print(f"{key} : {np.array(value).sum()}s")


    def view_prop_wave(self):
        arr= np.zeros((self.grid_size[0], self.grid_size[1]), dtype=bool)
        for index in list(self.prop_state):
            arr[self.decode_index(index)] = True
        plt.imshow(arr)
        plt.show()
        

    def view_log(self):
        for key, value in self.log.items():
            plt.plot(value, label=key)
        plt.legend()
        plt.show()