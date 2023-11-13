
DIRECTIONS = ['top', 'bottom', 'left', 'right']

drdc = {'top': (-1, 0), 'bottom': (1, 0), 'left': (0, -1), 'right': (0, 1)}

def dir_index(index, dir):
    dr, dc = drdc[dir]
    return (index[0]+dr, index[1]+dc)

def opposite(dir):
    if dir == 'top':
        return 'bottom'
    elif dir =='bottom':
        return 'top'
    elif dir =='left':
        return 'right'
    else:
        return 'left'