import numpy as np

def calculateAvoidanceVector(robotarget, lasers, angles):
    """ 
    Calculates a vector which points in collision avoidance direction
    """
    dist = np.linalg.norm(robotarget)
    targetangle = np.math.atan2(robotarget[1],robotarget[0])
    avoidanceVector = np.array([0.0,0.0])
    for (i, l) in enumerate(lasers['Echoes']):
        laserlength = min(l,dist * 2)
        angle = angles[i]
        laservec = (1/(1+np.abs(targetangle - angle)) * np.array([np.math.cos(angle),np.math.sin(angle)]))
        avoidanceVector += laservec
    avoidanceVector = avoidanceVector / np.linalg.norm(avoidanceVector)
    projed = np.dot(avoidanceVector,robotarget)/ np.linalg.norm(robotarget)**2 * robotarget
    return avoidanceVector - projed
