"""Simple Reachy2 PyBullet wrapper.

This wrapper expects the Reachy URDF to be provided by path (environment variable REACHY_URDF or argument).
It exposes reset(), step(action), get_observation(), and close().

This is a minimal starting point — adapt joint mappings and controllers to your Reachy model.
"""
import os
import numpy as np


class ReachyEnv:
    def __init__(self, urdf_path: str = None, render: bool = False, time_step: float = 1.0 / 240.0):
        try:
            import pybullet as p
            import pybullet_data
        except Exception as e:
            raise ImportError("pybullet is required for ReachyEnv. Install with `pip install pybullet`.") from e

        self._p = p
        self._pbdata = pybullet_data
        self.render = render
        self.physics_client = p.connect(p.GUI if render else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(time_step)
        self.time_step = time_step
        self.robot = None
        self.joint_ids = []

        if urdf_path is None:
            urdf_path = os.environ.get("REACHY_URDF")
        if urdf_path is None:
            raise ValueError("Provide Reachy URDF path via urdf_path or REACHY_URDF environment variable")

        self.urdf_path = urdf_path

    def reset(self):
        p = self._p
        p.resetSimulation()
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(self.time_step)
        plane = p.loadURDF("plane.urdf")

        start_pos = [0, 0, 0]
        start_orn = p.getQuaternionFromEuler([0, 0, 0])
        self.robot = p.loadURDF(self.urdf_path, start_pos, start_orn, useFixedBase=False)

        # collect controllable/revolute joints
        self.joint_ids = []
        for i in range(p.getNumJoints(self.robot)):
            info = p.getJointInfo(self.robot, i)
            joint_type = info[2]
            # consider revolute/prismatic/continuous as actuated
            if joint_type in (getattr(p, 'JOINT_REVOLUTE', None), getattr(p, 'JOINT_PRISMATIC', None), getattr(p, 'JOINT_CONTINUOUS', None)):
                self.joint_ids.append(i)

        return self.get_observation()

    def step(self, action: np.ndarray):
        p = self._p
        # action: joint position targets len == len(joint_ids)
        if len(action) != len(self.joint_ids):
            if len(action) < len(self.joint_ids):
                action = np.concatenate([action, np.zeros(len(self.joint_ids) - len(action))])
            else:
                action = action[: len(self.joint_ids)]

        for idx, j in enumerate(self.joint_ids):
            p.setJointMotorControl2(self.robot, j, p.POSITION_CONTROL, targetPosition=float(action[idx]))

        for _ in range(10):
            p.stepSimulation()

        return self.get_observation(), 0.0, False, {}

    def get_observation(self):
        p = self._p
        joint_states = [p.getJointState(self.robot, j)[0] for j in self.joint_ids]
        base_pos, base_orn = p.getBasePositionAndOrientation(self.robot)
        return {"joint_positions": np.array(joint_states), "base_pos": np.array(base_pos), "base_orn": np.array(base_orn)}

    def close(self):
        try:
            self._p.disconnect()
        except Exception:
            pass
