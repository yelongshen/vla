"""Minimal Omniverse Isaac Sim demo.

This script boots an Omniverse Kit session (via Isaac Sim), spawns a ground plane and
one dynamic cube, advances the simulation for a few seconds, and then shuts down.
Run it with the Isaac Sim python entry point, for example:

    ./python.sh examples/omniverse_minimal_demo.py --kit

The script assumes it is executed within an Omniverse Isaac Sim installation.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Optional

import numpy as np

try:
    from omni.isaac.kit import SimulationApp  # type: ignore
except ImportError as exc:  # pragma: no cover - informative error for non-Omniverse envs
    raise RuntimeError(
        "This demo must be run from within an Omniverse Isaac Sim Python environment.\n"
        "Launch it using Isaac Sim's python interpreter, e.g. ./python.sh examples/omniverse_minimal_demo.py"
    ) from exc


DEFAULT_STAGE_PATH = "/World"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Minimal Omniverse demo (Isaac Sim)")
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run the simulation without rendering (useful for CI or remote sessions).",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=240,
        help="Number of physics steps to simulate (default: 240, roughly 4 seconds at 60 Hz).",
    )
    parser.add_argument(
        "--stage",
        type=str,
        default=DEFAULT_STAGE_PATH,
        help="USD prim path for the world root (default: /World).",
    )
    return parser.parse_args()


def build_world(world, stage_path: str) -> None:
    """Populate the world with a ground plane and a single dynamic cube."""
    from omni.isaac.core.objects import DynamicCuboid  # type: ignore
    from omni.isaac.core.prims import GroundPlane  # type: ignore

    world.scene.add_default_ground_plane()

    cube_size = 0.2
    cube_mass = 1.0

    world.scene.add(
        DynamicCuboid(
            prim_path=f"{stage_path}/Cube",
            name="demo_cube",
            position=np.array([0.0, 0.0, cube_size * 2.0], dtype=np.float32),
            scale=np.array([cube_size] * 3, dtype=np.float32),
            color=np.array([0.2, 0.5, 0.9], dtype=np.float32),
            mass=cube_mass,
        )
    )

    GroundPlane(prim_path=f"{stage_path}/GroundPlane", name="ground", size=20.0)


def main() -> None:
    args = parse_args()

    sim_app = SimulationApp({"headless": args.headless})

    from omni.isaac.core import World  # type: ignore

    world = World(stage_units_in_meters=1.0)
    world.reset()

    build_world(world, args.stage)

    print("Starting simulation loop...")
    for step in range(max(args.steps, 0)):
        world.step(render=not args.headless)
        if step % 60 == 0:
            cube_prim = world.scene.get_object("demo_cube")
            if cube_prim is not None:
                pose = cube_prim.get_pose()
                print(
                    f"Step {step:04d} | Cube position: {pose.p.tolist()} orientation (quat): {pose.q.tolist()}"
                )

    print("Simulation complete. Shutting down Omniverse...")
    sim_app.close()


if __name__ == "__main__":
    main()
