import numpy as np
import matplotlib.pyplot as plt
import hashlib
import random
import sys
import time
import asyncio
from queue import PriorityQueue
from IPython.display import clear_output
from PIL import Image
from typing import Tuple


sys.setrecursionlimit(10**7)

DIRECTIONS = ['top', 'bottom', 'left', 'right']

class PropWave:
    def __init__(self, epicenter, grid_size):
        self.updated_from = {}
        self.heapq = PriorityQueue()
        self.epicenter = epicenter
        self.grid_size = grid_size

    def size(self):
        return len(self.updated_from)

    def add(self, r, c, dir):
        index_hash = self.grid_size[1]*r + c
        if index_hash in self.updated_from:
            self.updated_from[index_hash].append(opposite(dir))
        else:
            self.updated_from[index_hash] = [opposite(dir)]
            self.heapq.put((ManhattanDistance((r, c), (self.epicenter)), index_hash))
            

    def pop(self): # return index_hash, updated_from
        if self.size() == 0:
            return None
        dist, index_hash = self.heapq.get()
        return (index_hash, self.updated_from.pop(index_hash))



class WFCModel:
    def __init__(self, image_path, tile_size: Tuple, flip_horizontal=False, flip_vertical=False, rotate=False):
        self.tile_size = tile_size
        self.tileset = get_tiles(image_path, tile_size, flip_horizontal, flip_vertical, rotate)
        self.adjacency = get_adjacent_tiles(self.tileset)

        cnt = 0
        for key in list(self.tileset.keys()):
            if len(self.adjacency[key]['top']) == 0 or len(self.adjacency[key]['left']) == 0 or len(self.adjacency[key]['right']) == 0 or len(self.adjacency[key]['bottom']) == 0:
                del self.tileset[key]
                cnt +=1
        self.adjacency = get_adjacent_tiles(self.tileset)
        print(f'{cnt} Eliminated')

        self.average_color = np.array([self.tileset[hash][0][0] for hash in list(self.tileset.keys())]).mean(axis=0)[:3]

        self.superposition = None
        self.possible_patterns = None
        self.grid_size = None
        self.prop_state = None

        self.performance = None
        self.log = None
        print('model initialization finised')


    def init_superposition(self, size):
        self.grid_size = size
        height, width = size
        superpos = np.empty((height, width), dtype=object)
        pos_pat = np.empty((height, width), dtype=object)
        for i in range(height):
            for j in range(width):
                superpos[i, j] = set(self.tileset)
                pos_pat[i, j] = {'top': set(self.tileset), 'bottom': set(self.tileset), 'left': set(self.tileset), 'right': set(self.tileset)}
        self.superposition = superpos
        self.possible_patterns = pos_pat
        self.performance = {'Propagate': [], 'Visualize': []}
        self.log = {'Observed': []}
        print('superposition initialization finised')


    def is_valid_index(self, r, c):
        return r >= 0 and c >= 0 and r < self.grid_size[0] and c < self.grid_size[1]

    def hash_index(self, r, c):
        return self.grid_size[1] * r + c
    
    def decode_index(self, n):
        return (n // self.grid_size[1], n % self.grid_size[1])


    def propagate(self, r, c, dir):
        ar, ac = dir_index((r, c), dir)
        if not self.is_valid_index(ar, ac):
            return
        self.prop_state.add(ar, ac, opposite(dir))


    def collapse(self, r, c, value):
        self.prop_state = PropWave(epicenter=(r, c), grid_size=self.grid_size)
        self.superposition[r, c] = set([value])

        for dir in DIRECTIONS:
            self.possible_patterns[r, c][dir] = self.adjacency[value][dir]
        for dir in DIRECTIONS:
            self.propagate(r, c, dir)

        
    def update_superposition(self, r, c, updated_from):
        changed = False
        # Superposition Update
        for dir in DIRECTIONS:
            ar, ac = dir_index((r, c), dir)
            if self.is_valid_index(ar, ac):
                prev_len = len(self.superposition[r, c])
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
    

    def next_step(self, show_prop=False, seed=random.random()):
        random.seed(seed)
        entropy, indices = self.get_minimum_entropy()
        if entropy == 0:
            return -1
        if entropy == 999999:
            return 1

        index = random.choice(indices)
        tile_hash = random.choice(list(self.superposition[index[0], index[1]]))
        self.collapse(index[0], index[1], tile_hash)

        while self.prop_state.size() > 0:
            index_hash, updated_from = self.prop_state.pop()
            r, c = self.decode_index(index_hash)
            if self.is_valid_index(r, c):
                self.update_superposition(r, c, updated_from)
             
        return 0


    def generate(self, size, show_process=False, show_prop=False, seed=random.random()):
        self.init_superposition(size)
        result = 0
        step = 0

        while result == 0:
            step += 1
            s = time.time()
            result = self.next_step(show_prop, seed)
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

        print('generate seed:', seed)

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
                tile_hash_list = list(self.superposition[i, j])
                if len(tile_hash_list) == 0:
                    return None
                elif len(tile_hash_list) > 1:
                    if len(tile_hash_list) >= 1 * len(self.tileset):
                        color = self.average_color
                    else:
                        color = np.array([self.tileset[hash][0][0] for hash in tile_hash_list]).mean(axis=0)
                else:
                    color = self.tileset[tile_hash_list[0]][0][0]

                output_image[i][j] = color[:3]
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




drdc = {'top': (-1, 0), 'bottom': (1, 0), 'left': (0, -1), 'right': (0, 1)}

def dir_index(index, dir):
    dr, dc = drdc[dir]
    return (index[0]+dr, index[1]+dc)

def hash_tile(tile):
    return hashlib.sha256(tile.tobytes()).hexdigest()

def ManhattanDistance(p1, p2):
    return abs(p1[0]-p2[0]) + abs(p1[1]-p2[1])
    
def opposite(dir):
    if dir == 'top':
        return 'bottom'
    elif dir =='bottom':
        return 'top'
    elif dir =='left':
        return 'right'
    else:
        return 'left'


def get_tiles(image_path, tile_size, flip_horizontal=False, flip_vertical=False, rotate=False):
    image = Image.open(image_path)
    tiles = {}
    width, height = image.size

    for i in range(0, width - tile_size[0] + 1):
        for j in range(0, height - tile_size[1] + 1):
            box = (i, j, i+tile_size[0], j+tile_size[1])
            tile_img = image.crop(box)

            if flip_horizontal:
                tile_img_h = tile_img.transpose(Image.FLIP_LEFT_RIGHT)
                tile_hash_h = hash_tile(tile_img_h)
                tiles[tile_hash_h] = np.array(tile_img_h)

            if flip_vertical:
                tile_img_v = tile_img.transpose(Image.FLIP_TOP_BOTTOM)
                tile_hash_v = hash_tile(tile_img_v)
                tiles[tile_hash_v] = np.array(tile_img_v)

            if rotate:
                for k in range(3):
                    tile_img_r = tile_img.rotate((k+1)*90)
                    tile_hash_r = hash_tile(tile_img_r)
                    tiles[tile_hash_r] = np.array(tile_img_r)

            tile_hash = hash_tile(tile_img)
            tiles[tile_hash] = np.array(tile_img)

    return tiles


def get_adjacent_tiles(tiles):
    adjacent_tiles = {}
    for tile_hash, tile in tiles.items():
        # Initialize the list of adjacent tiles for this tile
        adjacent_tiles[tile_hash] = {'top': set(), 'bottom': set(), 'left': set(), 'right': set()}

        # Iterate over all other tiles to check for adjacency
        for other_hash, other_tile in tiles.items():
            # Check if other tile can be placed on top of this tile
            if np.array_equal(tile[:-1, :], other_tile[1:, :]):
                adjacent_tiles[tile_hash]['top'].add(other_hash)

            # Check if other tile can be placed below this tile
            if np.array_equal(tile[1:, :], other_tile[:-1, :]):
                adjacent_tiles[tile_hash]['bottom'].add(other_hash)

            # Check if other tile can be placed to the left of this tile
            if np.array_equal(tile[:, :-1], other_tile[:, 1:]):
                adjacent_tiles[tile_hash]['left'].add(other_hash)

            # Check if other tile can be placed to the right of this tile
            if np.array_equal(tile[:, 1:], other_tile[:, :-1]):
                adjacent_tiles[tile_hash]['right'].add(other_hash)

    return adjacent_tiles
