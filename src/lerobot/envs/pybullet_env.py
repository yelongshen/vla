"""Minimal PyBullet continuous-control environment for stepping/climbing demo.

This is a lightweight illustrative environment that builds a staircase from boxes and spawns
an existing URDF robot (e.g., Kuka or a simple quadruped/humanoid if available in pybullet_data).
It provides a reset(), step(action), and get_observation() API. Actions are interpreted as
joint position targets for simplicity.

Note: This demo is educational — for real locomotion research use Brax, MuJoCo, or a tuned pybullet setup.
"""
import os
import numpy as np


class PyBulletStairsEnv:
    def __init__(self, render: bool = False, time_step: float = 1.0 / 240.0):
        try:
            import pybullet as p
            import pybullet_data
        except Exception as e:
            raise ImportError("pybullet is required for PyBulletStairsEnv. Install with `pip install pybullet`.") from e

        self._p = p
        self._pbdata = pybullet_data
        self.render = render
        self.physics_client = p.connect(p.GUI if render else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(time_step)
        self.time_step = time_step
        # optional robot selector (name or path)
        self.robot_urdf = None
        self.robot = None
        self.stairs_ids = []
        self.joint_ids = []

    def reset(self):
        p = self._p
        p.resetSimulation()
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(self.time_step)
        plane = p.loadURDF("plane.urdf")

        # build simple stairs using boxes
        stair_height = 0.08
        stair_depth = 0.2
        num_steps = 6
        for i in range(num_steps):
            z = stair_height * (i + 0.5)
            pos = [1.0 + stair_depth * i, 0, z - 0.01]
            col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[stair_depth / 2, 0.5, stair_height / 2])
            vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[stair_depth / 2, 0.5, stair_height / 2], rgbaColor=[0.6, 0.6, 0.6, 1])
            idb = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=col, baseVisualShapeIndex=vis, basePosition=pos)
            self.stairs_ids.append(idb)

        # load a simple robot URDF (use r2d2.urdf in pybullet_data as stand-in)
        start_pos = [0, 0, 0.5]
        start_orn = p.getQuaternionFromEuler([0, 0, 0])
        # load a robot URDF. If user supplied a robot_urdf, try to resolve it; otherwise use builtin humanoid
        robot_path = None
        user_spec = getattr(self, 'robot_urdf', None)
        if user_spec:
            # resolve user-specified URDF: absolute path, relative path, or name under pybullet_data
            if os.path.isabs(user_spec) and os.path.exists(user_spec):
                robot_path = user_spec
            elif os.path.exists(user_spec):
                robot_path = user_spec
            else:
                candidate = user_spec
                if not candidate.endswith('.urdf'):
                    candidate = candidate + '.urdf'
                candidate1 = os.path.join(self._pbdata.getDataPath(), candidate)
                candidate2 = os.path.join(self._pbdata.getDataPath(), user_spec)
                if os.path.exists(candidate1):
                    robot_path = candidate1
                elif os.path.exists(candidate2):
                    robot_path = candidate2

        if robot_path is None:
            # default fallback to pybullet_data humanoid if available, else r2d2
            humanoid_path = os.path.join(self._pbdata.getDataPath(), "humanoid/humanoid.urdf")
            if os.path.exists(humanoid_path):
                robot_path = humanoid_path
            else:
                robot_path = os.path.join(self._pbdata.getDataPath(), "r2d2.urdf")
        self.robot = p.loadURDF(robot_path, start_pos, start_orn)

        # collect controllable joints (robust to pybullet API differences)
        self.joint_ids = []
        allowed_joint_types = {getattr(p, 'JOINT_REVOLUTE', None), getattr(p, 'JOINT_PRISMATIC', None)}
        jc = getattr(p, 'JOINT_CONTINUOUS', None)
        if jc is not None:
            allowed_joint_types.add(jc)
        # remove any None values
        allowed_joint_types = {t for t in allowed_joint_types if t is not None}

        for i in range(p.getNumJoints(self.robot)):
            info = p.getJointInfo(self.robot, i)
            joint_type = info[2]
            if joint_type in allowed_joint_types:
                self.joint_ids.append(i)

        return self.get_observation()

    def step(self, action: np.ndarray):
        """Action: array of joint position targets (clipped to [-1,1] mapped to joint limits).
        Returns obs, reward, done, info where reward is negative deviation from step progress.
        """
        p = self._p
        if len(action) != len(self.joint_ids):
            # pad or clip
            if len(action) < len(self.joint_ids):
                action = np.concatenate([action, np.zeros(len(self.joint_ids) - len(action))])
            else:
                action = action[: len(self.joint_ids)]
        # simple mapping: map [-1,1] to joint limits
        for idx, j in enumerate(self.joint_ids):
            info = p.getJointInfo(self.robot, j)
            lower = info[8]
            upper = info[9]
            if lower >= upper:
                target = 0.0
            else:
                target = float((action[idx] + 1.0) / 2.0) * (upper - lower) + lower
            p.setJointMotorControl2(self.robot, j, p.POSITION_CONTROL, targetPosition=target)

        # step simulation for a few frames
        for _ in range(10):
            p.stepSimulation()

        obs = self.get_observation()
        base_pos, _ = p.getBasePositionAndOrientation(self.robot)
        # reward: encourage forward progress in x direction (towards stairs)
        reward = base_pos[0]
        done = False
        info = {"base_pos": base_pos}
        return obs, reward, done, info

    def get_observation(self):
        p = self._p
        # joint states
        joint_states = [p.getJointState(self.robot, j)[0] for j in self.joint_ids]
        base_pos, base_orn = p.getBasePositionAndOrientation(self.robot)
        obs = {"joint_positions": np.array(joint_states), "base_pos": np.array(base_pos)}
        return obs

    def close(self):
        try:
            self._p.disconnect()
        except Exception:
            pass
