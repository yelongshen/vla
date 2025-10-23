"""List URDF files available in the workspace and in pybullet_data.

Usage:
  python tools/list_urdfs.py

This script prints:
 - URDF files found under the repository folder
 - URDF files available in pybullet_data (if pybullet is installed)
"""
import os
from pathlib import Path


def list_repo_urdfs(root: Path):
    urdfs = list(root.rglob('*.urdf'))
    return urdfs


def list_pybullet_data_urdfs():
    try:
        import pybullet_data
    except Exception:
        return None
    data_path = Path(pybullet_data.getDataPath())
    urdfs = list(data_path.rglob('*.urdf'))
    return data_path, urdfs


def main():
    repo_root = Path(__file__).resolve().parent.parent
    print(f"Searching for URDFs under repo: {repo_root}")
    repo_urdfs = list_repo_urdfs(repo_root)
    if repo_urdfs:
        print("\nURDF files in repository:")
        for u in repo_urdfs:
            print(" -", u.relative_to(repo_root))
    else:
        print("No URDF files found in repository.")

    pb = list_pybullet_data_urdfs()
    if pb is None:
        print("\npybullet_data not available (pybullet not installed). Install pybullet to list built-in URDFs.")
    else:
        data_path, urdfs = pb
        print(f"\nURDF files in pybullet_data: {data_path}")
        for u in sorted(urdfs):
            print(" -", u.relative_to(data_path))

if __name__ == '__main__':
    main()
