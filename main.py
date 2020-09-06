from Robot import *
from Path import *
from ShowPath import *
from pathing import *

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
while dist > 0.01:
    time.sleep(0.2)
    pos = robot.getPosition()
    heading = robot.getHeading()
    target = get_target(0.5, pos, path)
    dist = target_dist(pos, target)
    ang = target_ang(pos, target, heading)    
    robot.setMotion(dist / ((1 + abs(ang))) , ang * 2)
    sp.update(pos, target)

robot.setMotion(0, 0)
