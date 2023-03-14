import numpy as np
from PIL import Image
import hashlib

def get_tiles(image_file, width, height):
    tiles = {}
    image = Image.open(image_file)
    img_width, img_height = image.size
    for x in range(0, img_width - width + 1, width):
        for y in range(0, img_height - height + 1, height):
            tile = image.crop((x, y, x+width, y+height))
            tile_hash = hashlib.md5(tile.tobytes()).hexdigest()
            tiles[tile_hash] = np.array(tile)
    return tiles