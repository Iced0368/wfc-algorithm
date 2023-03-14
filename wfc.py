import numpy as np
from PIL import Image
import hashlib
from typing import Tuple


class WFCModel:
    def __init__(self, image_path, tile_size: Tuple, flip_horizontal=False, flip_vertical=False, rotate=False):
        self.tileset = get_tiles(image_path, tile_size, flip_horizontal, flip_vertical, rotate)
        self.adjacency = get_adjacent_tiles(self.tileset)
        self.superposition = None

    def init_superposition(self, size):
        height, width = size
        arr = np.empty((height, width), dtype=object)
        for i in range(height):
            for j in range(width):
                arr[i, j] = set(self.tileset)

    def collapse(self, r, c, value):
        self.superposition[r][c] = set([value])
        
    def update_superposition(self, r, c):
        



def hash_tile(tile):
    return hashlib.md5(tile.tobytes()).hexdigest()

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
        adjacent_tiles[tile_hash] = {'top': [], 'bottom': [], 'left': [], 'right': []}

        # Iterate over all other tiles to check for adjacency
        for other_hash, other_tile in tiles.items():
            if tile_hash != other_hash:
                # Check if other tile can be placed on top of this tile
                if np.array_equal(tile[0, :], other_tile[-1, :]):
                    adjacent_tiles[tile_hash]['top'].append(other_hash)

                # Check if other tile can be placed below this tile
                if np.array_equal(tile[-1, :], other_tile[0, :]):
                    adjacent_tiles[tile_hash]['bottom'].append(other_hash)

                # Check if other tile can be placed to the left of this tile
                if np.array_equal(tile[:, 0], other_tile[:, -1]):
                    adjacent_tiles[tile_hash]['left'].append(other_hash)

                # Check if other tile can be placed to the right of this tile
                if np.array_equal(tile[:, -1], other_tile[:, 0]):
                    adjacent_tiles[tile_hash]['right'].append(other_hash)

    return adjacent_tiles