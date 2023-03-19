import numpy as np
import hashlib
from PIL import Image


def hash_tile(tile):
    return hashlib.sha256(tile.tobytes()).hexdigest()

def get_tiles(image_path, tile_size, flip_horizontal=False, flip_vertical=False, rotate=False):
    image = Image.open(image_path).convert("RGB")
    tiles = {}
    width, height = image.size
    weights = {}

    for i in range(0, width - tile_size[0] + 1):
        for j in range(0, height - tile_size[1] + 1):
            box = (i, j, i+tile_size[0], j+tile_size[1])
            tile_img = image.crop(box)

            if flip_horizontal:
                tile_img_h = tile_img.transpose(Image.FLIP_LEFT_RIGHT)
                tile_hash_h = hash_tile(tile_img_h)
                tiles[tile_hash_h] = np.array(tile_img_h)
                if tile_hash_h in weights:
                    weights[tile_hash_h] += 1
                else:
                    weights[tile_hash_h] = 1

            if flip_vertical:
                tile_img_v = tile_img.transpose(Image.FLIP_TOP_BOTTOM)
                tile_hash_v = hash_tile(tile_img_v)
                tiles[tile_hash_v] = np.array(tile_img_v)
                if tile_hash_v in weights:
                    weights[tile_hash_v] += 1
                else:
                    weights[tile_hash_v] = 1

            if rotate:
                for k in range(3):
                    tile_img_r = tile_img.rotate((k+1)*90)
                    tile_hash_r = hash_tile(tile_img_r)
                    tiles[tile_hash_r] = np.array(tile_img_r)
                    if tile_hash_r in weights:
                        weights[tile_hash_r] += 1
                    else:
                        weights[tile_hash_r] = 1

            tile_hash = hash_tile(tile_img)
            tiles[tile_hash] = np.array(tile_img)
            if tile_hash in weights:
                weights[tile_hash] += 1
            else:
                weights[tile_hash] = 1

    return tiles, weights


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
