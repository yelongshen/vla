"""Minimal AI2-THOR wrapper for a single-manipulation example.

This wrapper exposes a small Gym-like API: reset(goal_instruction), step(action_str), get_observation().
Actions are high-level strings mapped to AI2-THOR controller.step commands.

Note: AI2-THOR must be installed and the required scenes will be downloaded on demand.
"""
from typing import Dict, Any, Optional
import numpy as np


class AI2ThorEnv:
    def __init__(self, scene: str = "FloorPlan1", width: int = 300, height: int = 300, headless: bool = True):
        try:
            import ai2thor.controller as controller
        except Exception as e:
            raise ImportError("ai2thor is required for AI2ThorEnv. Install with `pip install ai2thor`.") from e

        self.controller = controller.Controller(scene=scene, width=width, height=height, renderDepthImage=True, agentMode="default")
        self._last_event = None
        self.goal_instruction = ""

    def reset(self, goal_instruction: str = "put apple in microwave") -> Dict[str, Any]:
        self.controller.reset(self.controller.scene_name)
        self._last_event = self.controller.last_event
        self.goal_instruction = goal_instruction
        return self.get_observation()

    def step(self, action: str) -> Dict[str, Any]:
        """Perform a high-level action.

        Supported high-level strings (examples):
          - 'LookDown', 'LookUp', 'RotateLeft', 'RotateRight', 'MoveAhead'
          - 'OpenMicrowave', 'CloseMicrowave', 'PickupObject:Apple', 'PutObject:Microwave'

        Returns observation dict with keys: rgb, depth, metadata
        """
        ev = self._last_event
        # simple mapping for demo purposes
        try:
            if action == "RotateLeft":
                ev = self.controller.step(dict(action="RotateLeft"))
            elif action == "RotateRight":
                ev = self.controller.step(dict(action="RotateRight"))
            elif action == "MoveAhead":
                ev = self.controller.step(dict(action="MoveAhead"))
            elif action == "LookDown":
                ev = self.controller.step(dict(action="LookDown"))
            elif action == "OpenMicrowave":
                ev = self.controller.step(dict(action="OpenObject", objectId=self._find_object_by_name("Microwave")))
            elif action.startswith("PickupObject:"):
                name = action.split(":", 1)[1]
                objId = self._find_object_by_name(name)
                ev = self.controller.step(dict(action="PickupObject", objectId=objId))
            elif action.startswith("PutObject:"):
                target = action.split(":", 1)[1]
                receptacle_id = self._find_object_by_name(target)
                # Place the held object in the receptacle
                ev = self.controller.step(dict(action="PutObject", receptacleObjectId=receptacle_id))
            else:
                # fallback: try direct action mapping
                ev = self.controller.step(dict(action=action))
        except Exception as e:
            # Keep previous event on error
            ev = self.controller.last_event

        self._last_event = ev
        return self.get_observation()

    def get_observation(self) -> Dict[str, Any]:
        ev = self._last_event or self.controller.last_event
        rgb = ev.frame if hasattr(ev, "frame") else None
        depth = ev.depth_frame if hasattr(ev, "depth_frame") else None
        metadata = dict(objects=[{"objectId": o.objectId, "name": o.objectType} for o in ev.metadata["objects"]]) if ev and ev.metadata else {}
        return {"rgb": rgb, "depth": depth, "metadata": metadata, "goal": self.goal_instruction}

    def _find_object_by_name(self, name: str) -> Optional[str]:
        # naive case-insensitive match to objectType or objectId
        ev = self._last_event or self.controller.last_event
        if not ev or "objects" not in ev.metadata:
            return None
        for o in ev.metadata["objects"]:
            if name.lower() in o.get("objectType", "").lower() or name.lower() in o.get("objectId", "").lower():
                return o.get("objectId")
        return None

    def close(self):
        try:
            self.controller.stop()
        except Exception:
            pass
