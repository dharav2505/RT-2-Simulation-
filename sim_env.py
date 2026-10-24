import pybullet as p
import pybullet_data
import numpy as np
import cv2

class SimulationEnvironment:
    def __init__(self):
        p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.resetSimulation()
        p.setGravity(0, 0, -9.8)
        self.plane = p.loadURDF("plane.urdf")
        self.robot = p.loadURDF("ur5.urdf", [0,0,0], useFixedBase=True)
        self.cubes = []
        colors = {"red": [1,0,0,1], "green": [0,1,0,1], "blue": [0,0,1,1]}
        pos = [[0.5, 0, 0.02], [0.6, 0.2, 0.02], [0.6, -0.2, 0.02]]
        for c, p_ in zip(colors, pos):
            cube_id = p.loadURDF("cube_small.urdf", basePosition=p_)
            p.changeVisualShape(cube_id, -1, rgbaColor=colors[c])
            self.cubes.append((c, cube_id))

    def get_camera_image(self):
        width, height, view, proj, _, _, rgb, _, _ = p.getDebugVisualizerCamera()
        _, _, rgbImg, _, _ = p.getCameraImage(width, height, view, proj)
        rgbImg = np.array(rgbImg, dtype=np.uint8)
        rgbImg = cv2.cvtColor(rgbImg, cv2.COLOR_BGRA2RGB)
        return rgbImg
