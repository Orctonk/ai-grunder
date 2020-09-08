import numpy as np

def get_closest(pos, path):
    """
    Get the point in path closest to pos.
    """
    closest = path[0]
    for point in path[1:-1]:
        if np.linalg.norm(point - pos) < np.linalg.norm(closest - pos):
            closest = point
    return closest

def get_target(radius, pos, path):
    """
    Get a target point in path that is approx radius distance ahead of pos.
    """
    closest = get_closest(pos, path)
    
    index = 0
    while not np.allclose(closest, path[index]):
        index += 1
    
    for point in path[index:-1]:        
        target = point
        if np.linalg.norm(point - pos) > radius:
            break
    return target

def target_dist(pos, target):
    """
    Get the distance to the target
    """
    return np.linalg.norm(target - pos)

def target_ang(pos, target, heading):
    """
    Get the angle between the robots heading and the target in radians.
    """
    x_axis = np.array([1, 0])
    vect = target - pos
    vect_norm = vect / np.linalg.norm(vect)
    ang = np.arccos(np.clip(np.dot(x_axis, vect_norm), -1.0, 1.0))
    ang -= abs(heading)
    return ang if target[1] > pos[1] else -ang

def vector_to_robospace(robopos, roborot, vector):
    originzero = vector - robopos
    rotation = np.array([  [np.cos(roborot), np.sin(roborot)],
                [np.sin(roborot),-np.cos(roborot)]])
    return np.matmul(rotation,originzero)

def target_turn_factor(robotarget):
    return -(2 * robotarget[1]) / (robotarget[0] ** 2 + robotarget[1] ** 2) 