from Robot import *
from Path import *
from ShowPath import *
from pathing import *
from avoidance import *

# Best test, 1, 0.75
lookahead_dist = 2
speed_factor = 1.25

# load a path file
path = Path("Path-around-table.json").getPath()

# plot the path
sp = ShowPath(path)

print("Path length = " + str(len(path)))
print("First point = " + str(path[0][0]) + ", " + str(path[0][1]))

# make a robot to move around
robot = Robot()

# robot.setMotion(0.2PASS_THRESHOLD, 0)
dist = float('inf')
while dist > 0.2:
    time.sleep(0.02)
    pos = robot.getPosition()
    heading = robot.getHeading()
    target = get_target(lookahead_dist, pos, path)
    dist = target_dist(pos, target) 
    robotarget = vector_to_robospace(pos, heading, target)
    av = calculateAvoidanceVector(robotarget,robot.getLaser(),robot.getLaserAngles())
    turnfactor = target_turn_factor(robotarget - av)  
    speed = speed_factor / (1 + abs(turnfactor))
    robot.setMotion(speed, speed * turnfactor)
    av = vector_to_robospace(np.array([0,0]),-heading,av)
    sp.update(pos, target - av)

robot.setMotion(0, 0)

