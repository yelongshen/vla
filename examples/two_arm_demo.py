"""Two-arm PyBullet demo: spawn two KUKA arms and do a simple coordinated motion.

Run: python examples/two_arm_demo.py
"""
import time
import numpy as np

try:
    import pybullet as p
    import pybullet_data
except Exception:
    raise ImportError("pybullet is required. Install with `pip install pybullet` or use conda-forge.")


def main(render=True):
    cid = p.connect(p.GUI if render else p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.resetSimulation()
    p.setGravity(0, 0, -9.81)

    plane = p.loadURDF("plane.urdf")

    # Load two KUKA arms (kuka_iiwa/model.urdf exists in pybullet_data)
    startPosA = [0, -0.5, 0]
    startPosB = [0, 0.5, 0]
    orn = p.getQuaternionFromEuler([0, 0, 0])
    kukaA = p.loadURDF("kuka_iiwa/model.urdf", startPosA, orn)
    kukaB = p.loadURDF("kuka_iiwa/model.urdf", startPosB, orn)

    # collect revolute joints
    def get_control_joints(robot):
        ids = []
        for i in range(p.getNumJoints(robot)):
            info = p.getJointInfo(robot, i)
            if info[2] == p.JOINT_REVOLUTE:
                ids.append(i)
        return ids

    jointsA = get_control_joints(kukaA)
    jointsB = get_control_joints(kukaB)

    # simple coordinated motion: sin waves with phase offset
    t = 0.0
    while t < 6.28 * 2:
        for i, j in enumerate(jointsA):
            targetA = 0.5 * np.sin(t + i * 0.2)
            p.setJointMotorControl2(kukaA, j, p.POSITION_CONTROL, targetPosition=targetA)
        for i, j in enumerate(jointsB):
            targetB = 0.5 * np.sin(t + i * 0.2 + 1.57)
            p.setJointMotorControl2(kukaB, j, p.POSITION_CONTROL, targetPosition=targetB)

        p.stepSimulation()
        time.sleep(1.0 / 240.0)
        t += 1.0 / 240.0

    p.disconnect()


if __name__ == "__main__":
    main(render=True)
