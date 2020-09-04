import numpy as np
import Robot
from Path import Path

path = Path("Path-around-table.json").getPath()

def get_closest(pos, path):
    closest = path[0]
    for point in path[1:-1]:
        if np.linalg.norm(point - pos) < np.linalg.norm(closest - pos):
            closest = point
    return closest

def get_target(radius, pos, path):
    closest = get_closest(pos, path)
    target = closest
    # index = np.where(np.allclose(path, closest)
    # index
    for point in path[index:-1]:
        target = point
        if np.linalg.norm(point - pos) > radius:
            break
    return target

get_target(2, [0,0], path)