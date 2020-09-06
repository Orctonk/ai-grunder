import numpy as np
import Robot
from Path import Path

def get_closest(pos, path):
    closest = path[0]
    for point in path[1:-1]:
        if np.linalg.norm(point - pos) < np.linalg.norm(closest - pos):
            closest = point
    return closest

def get_target(radius, pos, path):
    closest = get_closest(pos, path)
    target = [np.nan, np.nan]
    for point in path:
        if (np.array_equal(target, [np.nan, np.nan]) and np.allclose(closest, point)):
            target = point
            continue

        target = point
        if np.linalg.norm(point - pos) > radius:
            break
    return target
