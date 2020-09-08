from Robot import *
from Path import *
from ShowPath import *
from pathing import *

# Best test, 1, 0.75
lookahead_dist = 1
speed_factor = 0.75

# load a path file
path = Path("Path-around-table.json").getPath()

# plot the path
sp = ShowPath(path)

print("Path length = " + str(len(path)))
print("First point = " + str(path[0][0]) + ", " + str(path[0][1]))

# make a robot to move around
robot = Robot()

# robot.setMotion(0.2, 0)
dist = float('inf')
while dist > 0.1:
    time.sleep(0.02)
    pos = robot.getPosition()
    heading = robot.getHeading()
    target = get_target(lookahead_dist, pos, path)
    dist = target_dist(pos, target) 
    robotarget = vector_to_robospace(pos, heading, target)
    turnfactor = target_turn_factor(robotarget)  
    speed = speed_factor / (1 + abs(turnfactor))
    robot.setMotion(speed, speed * turnfactor)
    sp.update(pos, target)

robot.setMotion(0, 0)
