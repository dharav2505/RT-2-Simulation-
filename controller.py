import pybullet as p
import time

class RobotController:
    def __init__(self, env):
        self.env = env

    def pick_and_place(self, cube_name):
        for name, cube_id in self.env.cubes:
            if name == cube_name:
                pos, _ = p.getBasePositionAndOrientation(cube_id)
                self.move_to(pos)
                time.sleep(1)
                print(f"Picked {name}")
                self.move_to([pos[0], pos[1]+0.2, pos[2]])
                print(f"Placed {name}")
                break

    def move_to(self, pos):
        p.resetBasePositionAndOrientation(self.env.robot, pos, [0,0,0,1])
