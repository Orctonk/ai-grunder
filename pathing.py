import numpy as np

def get_closest(pos, path):
    closest = path[0]
    for point in path[1:-1]:
        if np.linalg.norm(point - pos) < np.linalg.norm(closest - pos):
            closest = point
    return closest

def get_target(radius, pos, path):
    closest = get_closest(pos, path)
    
    index = 0
    while not np.allclose(closest, path[index]):
        index += 1
    
    for point in path[index:-1]:        
        target = point
        if np.linalg.norm(point - pos) > radius:
            break
    return target