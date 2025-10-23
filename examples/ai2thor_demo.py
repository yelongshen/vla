"""AI2-THOR demo runner: performs a tiny scripted manipulation episode.

This demo will:
 - Reset scene
 - Rotate to find an object named 'Apple'
 - Pickup the apple
 - Rotate to find 'Microwave' and open it
 - Put the apple in the microwave

Run: python examples/ai2thor_demo.py
"""
from lerobot.envs.ai2thor_env import AI2ThorEnv
import time


def main():
    env = AI2ThorEnv(scene="FloorPlan1", headless=True)
    obs = env.reset("put apple in microwave")
    print("Reset obs metadata objects:", len(obs.get("metadata", {}).get("objects", [])))

    # Basic scripted loop: scan and try pickup
    actions = ["RotateRight"] * 4 + ["MoveAhead"] * 2
    for a in actions:
        obs = env.step(a)
        time.sleep(0.2)

    # Try to pickup apple
    obs = env.step("PickupObject:Apple")
    print("After pickup metadata objects:", len(obs.get("metadata", {}).get("objects", [])))

    # Rotate towards microwave then open and place
    for _ in range(6):
        obs = env.step("RotateRight")
        time.sleep(0.1)

    obs = env.step("OpenMicrowave")
    time.sleep(0.2)
    obs = env.step("PutObject:Microwave")
    print("Done. Final metadata objects:", len(obs.get("metadata", {}).get("objects", [])))

    env.close()


if __name__ == "__main__":
    main()
