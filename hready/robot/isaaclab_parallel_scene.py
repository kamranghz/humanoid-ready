"""Scene pieces for many parallel copies of the SMPL-X-skeleton humanoid on Isaac Lab + Newton.

Import this only after Isaac Lab's ``AppLauncher`` has started the application (it imports Isaac Lab at module level).
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg

NJMAX = 600  # contact constraints per world, as in the single-copy stand check
NCONMAX = 150


def newton_sim_cfg(dt: float) -> SimulationCfg:
    return SimulationCfg(dt=dt, physics=NewtonCfg(solver_cfg=MJWarpSolverCfg(
        njmax=NJMAX, nconmax=NCONMAX, cone="pyramidal", impratio=1.0, integrator="implicitfast", iterations=100), num_substeps=1))


def primed_usd_copy(asset_dir: Path) -> tuple[str, str]:
    """A private copy of the newest importer output (the importer writes numbered folders under usd/); the copy gets primed."""
    usd_root = asset_dir / "usd"
    target = usd_root / "smplx_humanoid_primed"
    if not target.is_dir():
        sources = sorted(usd_root.glob("smplx_humanoid_sim_*"), key=lambda p: p.stat().st_mtime)
        if not sources:
            raise FileNotFoundError(f"no importer output under {usd_root}; run hready/robot/isaaclab_humanoid_stand.py once")
        shutil.copytree(sources[-1], target)
        source = str(sources[-1].name)
    else:
        source = "existing copy"
    files = sorted(target.glob("*.usd*"))
    if len(files) != 1:
        raise FileNotFoundError(f"expected one top-level USD file in {target}, found {[f.name for f in files]}")
    return str(files[0]), source


def make_scene_cfg(meta: dict, usd_path: str, n: int, env_spacing: float):
    groups: dict[str, list[str]] = {}
    for name, g in meta["group_of_joint"].items():
        if name != "pelvis":
            groups.setdefault(g, []).append(f"{name}_(x:[0-2]|[xyz])")
    actuators = {
        g: ImplicitActuatorCfg(joint_names_expr=exprs, stiffness=meta["pd"][g]["kp"], damping=meta["pd"][g]["kd"],
                               effort_limit_sim=meta["pd"][g]["torque_limit"])
        for g, exprs in groups.items()
    }
    robot = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(usd_path=usd_path),
        init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 1.5)), actuators=actuators)
    mat = sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0)

    @configclass
    class _Scene(InteractiveSceneCfg):
        ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg(physics_material=mat))
        light = AssetBaseCfg(prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=2000.0))

    cfg = _Scene(num_envs=n, env_spacing=env_spacing, replicate_physics=True)
    cfg.robot = robot
    return cfg


def hinge_index_of_dofs(dof_names: list[str], meta: dict) -> list[int]:
    """For every simulation DOF, its index in the 63-vector of SMPL-X-ordered hinge angles.

    The importer merges the three hinges of a body into one D6 joint named "<body>_x" with DOFs ":0", ":1", ":2"; the DOF order
    is taken as x, y, z (the stand check verified the resulting pose, not the axis order)."""
    out = []
    for n in dof_names:
        m = re.fullmatch(r"(.+)_([xyz])(?::([0-2]))?", n)
        if m is None:
            raise ValueError(f"unexpected joint name {n!r}")
        base, axis, dof = m.group(1), m.group(2), m.group(3)
        out.append(3 * (meta["joint_names"].index(base) - 1) + (int(dof) if dof is not None else "xyz".index(axis)))
    return out
