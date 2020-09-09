import numpy as np

def get_closest(pos, path, start_index = 0):
    """
    Get the point in path closest to pos.
    """
    closest = path[start_index]
    for point in path[start_index + 1:-1]:
        if np.linalg.norm(point - pos) < np.linalg.norm(closest - pos):
            closest = point
    return closest

def get_target(radius, pos, path):
    """
    Get a target point in path that is approx radius distance ahead of pos.
    """
    if "passed" not in get_target.__dict__:
        get_target.passed = 0

    closest = get_closest(pos, path, get_target.passed)
    
    while not np.allclose(closest, path[get_target.passed]):
        get_target.passed += 1
    
    for point in path[get_target.passed:-1]:        
        target = point
        if np.linalg.norm(point - pos) > radius:
            break
    return target

def target_dist(pos, target):
    """
    Get the distance to the target
    """
    return np.linalg.norm(target - pos)

def vector_to_robospace(robopos, roborot, vector):
    """
    Transform vector into robotspace
    """
    originzero = vector - robopos
    rotation = np.array([[np.cos(roborot), np.sin(roborot)],
                [np.sin(roborot), -np.cos(roborot)]])
    return np.matmul(rotation, originzero)

def target_turn_factor(robotarget):
    """
    Calculate the robot turnfactor
    """
    return -(2 * robotarget[1]) / (robotarget[0] ** 2 + robotarget[1] ** 2)