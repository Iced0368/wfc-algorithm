import numpy as np
from IPython.display import clear_output
import matplotlib.pyplot as plt
import hashlib
import random
import sys
from PIL import Image
from typing import Tuple

sys.setrecursionlimit(10**7)

class WFCModel:
    def __init__(self, image_path, tile_size: Tuple, flip_horizontal=False, flip_vertical=False, rotate=False):
        self.tileset = get_tiles(image_path, tile_size, flip_horizontal, flip_vertical, rotate)
        self.tile_size = tile_size
        self.adjacency = get_adjacent_tiles(self.tileset)
        self.superposition = None
        self.possible_patterns = None
        self.grid_size = None
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
        print('superposition initialization finised')

    def is_valid_index(self, r, c):
        return r >= 0 and c >= 0 and r < self.grid_size[0] and c < self.grid_size[1]

    def collapse(self, r, c, value):
        self.superposition[r, c] = set([value])
        for dir in ['top', 'bottom', 'left', 'right']:
            self.possible_patterns[r, c][dir] = self.adjacency[value][dir]
        for dir in ['top', 'bottom', 'left', 'right']:
            ar, ac = dir_index((r, c), dir)
            if self.is_valid_index(ar, ac):
                self.update_superposition(ar, ac, updated_from=[opposite(dir)])

        
    def update_superposition(self, r, c, updated_from=['top', 'bottom', 'left', 'right']):
        changed = False
        # Superposition Update
        for dir in updated_from:
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
        for dir in ['top', 'bottom', 'left', 'right']:
            pos_pat = set()
            for tile_hash in self.superposition[r, c]:
                pos_pat |= self.adjacency[tile_hash][dir]
            if len(pos_pat) < len(self.possible_patterns[r, c][dir]):
                changed = True
                self.possible_patterns[r, c][dir] = pos_pat

        if not changed: # If nothing changed in Adjacency, Not need to propagate
            return

        for dir in ['top', 'bottom', 'left', 'right']:
            if dir in updated_from: # No need to propagte to Updated-from
                continue
            ar, ac = dir_index((r, c), dir)
            if self.is_valid_index(ar, ac):
                self.update_superposition(ar, ac, updated_from=[opposite(dir)])


    def get_minimum_entropy(self):
        min_entropy = 99999
        result = []
        for r in range(self.grid_size[0]):
            for c in range(self.grid_size[1]):
                if len(self.superposition[r, c]) == 1:
                    continue
                if len(self.superposition[r, c]) < min_entropy:
                    result = [(r, c)]
                    min_entropy = len(self.superposition[r][c])
                elif len(self.superposition[r, c]) == min_entropy:
                    result.append((r, c))
        return (min_entropy, result)


    def generate(self, size, show_process=False):
        self.init_superposition(size)

        entropy, indices = self.get_minimum_entropy()
        while entropy > 1 and entropy != 99999:
            index = random.choice(indices)
            tile_hash = random.choice(list(self.superposition[index[0], index[1]]))
            self.collapse(index[0], index[1], tile_hash)

            if show_process:
                clear_output(wait=True)
                plt.imshow(self.overwrite_tile())
                plt.show()
            #print('entropy=', entropy)
            #self.show_entropy()
            entropy, indices = self.get_minimum_entropy()

    def overwrite_tile(self):
        # Create the output image with the specified size
        #output_size = ((self.tile_size[0]-1)*(self.grid_size[0])+1, (self.tile_size[1]-1)*(self.grid_size[1])+1)
        output_size = (self.grid_size[0]+self.tile_size[0]-1, self.grid_size[1]+self.tile_size[1]-1)
        output_image = Image.new('RGB', output_size, (0, 0, 0))

        # Loop over the grid and paste each tile onto the output image
        for i in range(self.grid_size[0]):
            for j in range(self.grid_size[1]):
                tile_hash_list = list(self.superposition[i, j])
                if len(tile_hash_list) == 0:
                    return None
                elif len(tile_hash_list) > 1:
                    tile = average_tiles([self.tileset[hash] for hash in tile_hash_list])
                else:
                    tile = self.tileset[tile_hash_list[0]]
                tile_image = Image.fromarray(tile, mode='RGB')

                output_image.paste(tile_image, (i, j))
        cropped_image = output_image.crop((0, 0, self.grid_size[0], self.grid_size[1]))
        return cropped_image.transpose(Image.ROTATE_270)


    def show_entropy(self):
        for r in range(self.grid_size[0]):
            for c in range(self.grid_size[1]):
                print(str(len(self.superposition[r, c])).rjust(5), end='')
            print()
        print()




drdc = {'top': (-1, 0), 'bottom': (1, 0), 'left': (0, -1), 'right': (0, 1)}

def dir_index(index, dir):
    dr, dc = drdc[dir]
    return (index[0]+dr, index[1]+dc)

def hash_tile(tile):
    return hashlib.sha256(tile.tobytes()).hexdigest()
    
def opposite(dir):
    if dir == 'top':
        return 'bottom'
    elif dir =='bottom':
        return 'top'
    elif dir =='left':
        return 'right'
    else:
        return 'left'
    

def average_tiles(tiles):
    sum_arr = np.zeros_like(tiles[0], dtype=np.uint64)
    for tile in tiles:
        sum_arr += tile.astype(np.uint64) 
    avg_arr = sum_arr / len(tiles)
    return np.uint8(avg_arr)


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
            if tile_hash != other_hash:
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
# %%
