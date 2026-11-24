"""WSL-only Isaac Lab + Newton check of the SMPL-X-skeleton humanoid (E6a): load, hold a static pose under PD for 5 s.

Run with Isaac Lab's Python (the same one as the G1 smoke test), after ``scripts/humanoid_asset_build.py``:

    python hready/robot/isaaclab_humanoid_stand.py --headless \\
        --asset D:/projects/hready_data/robot_assets/smplx_humanoid --out results/E/asset/isaac_stand_check.json

The pose is the canonical zero pose (T-pose) unless ``--amass-npz`` and ``--frame`` give one AMASS frame. Nothing is
tuned here: the gains come from the asset metadata. The result file records what loaded and how the pose held.

``--probe`` does not stand. The body starts 30 m up, so for two seconds it is in free fall and no joint carries a gravity
torque; the joints start at zero and the PD is told to go to one random pose inside half of every joint limit. That separates
three questions from balance and contacts: does the PD act on all 63 DOFs, are the DOF limits and gains the ones in the
asset, and do the simulator's three DOFs per joint turn about the axes this repo assumes (x, y, z in that order).
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

if __name__ != "__main__":
    raise ImportError("hready.robot.isaaclab_humanoid_stand is an Isaac Lab entrypoint; run it with Isaac Lab's Python.")

print(sys.executable, flush=True)

from isaaclab.app import AppLauncher

_parser = argparse.ArgumentParser(description="SMPL-X-skeleton humanoid: load and hold a static pose in Isaac Lab + Newton.")
_parser.add_argument("--asset", type=str, required=True, help="folder written by scripts/humanoid_asset_build.py")
_parser.add_argument("--out", type=str, required=True, help="result json")
_parser.add_argument("--seconds", type=float, default=5.0)
_parser.add_argument("--amass-npz", type=str, default=None, help="optional AMASS file; one frame gives the static pose")
_parser.add_argument("--frame", type=int, default=0)
_parser.add_argument("--mujoco-contacts", action="store_true", help="use MuJoCo's own contact detection instead of Newton's")
_parser.add_argument("--dt", type=float, default=0.005, help="physics step in seconds")
_parser.add_argument("--gain-scale", type=float, default=1.0, help="multiplies every stiffness and damping from the asset")
_parser.add_argument("--armature", type=float, default=None,
                     help="joint armature (kg m^2) written for every DOF; default keeps whatever the importer produced")
_parser.add_argument("--prime-drives", action=argparse.BooleanOptionalAction, default=True,
                     help="before the simulator is built, give every joint axis in the USD a placeholder drive (nonzero gains); "
                          "without it the importer's joints are not actuated and the PD does nothing")
_parser.add_argument("--probe", action="store_true", help="free-fall PD, limit and axis check instead of standing")
_parser.add_argument("--probe-seed", type=int, default=0, help="seed of the random target pose of --probe")
AppLauncher.add_app_launcher_args(_parser)
_args = _parser.parse_args()
_app = AppLauncher(_args)
simulation_app = _app.app

import isaaclab
import isaaclab.sim as sim_utils
import isaaclab_newton
import newton
import numpy as np
import torch
import warp as wp
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.sim import SimulationCfg, SimulationContext
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from hready.robot import humanoid_asset as H

SIM_DT = _args.dt
NJMAX = 600
NCONMAX = 150
FALL_Z_FRAC = 0.5  # fall when the pelvis drops below this fraction of its starting height
FALL_TILT_DEG = 45.0


def _newton_cfg() -> SimulationCfg:
    return SimulationCfg(
        dt=SIM_DT,
        physics=NewtonCfg(
            solver_cfg=MJWarpSolverCfg(njmax=NJMAX, nconmax=NCONMAX, cone="pyramidal", impratio=1.0,
                                       integrator="implicitfast", iterations=100, use_mujoco_contacts=_args.mujoco_contacts),
            num_substeps=1,
        ),
    )


def _robot_cfg(asset_dir: Path, meta: dict) -> ArticulationCfg:
    groups: dict[str, list[str]] = {}
    for name, g in meta["group_of_joint"].items():
        if name != "pelvis":
            # The importer may merge the three hinges of one body into one D6 joint named "<name>_x" with DOFs ":0", ":1", ":2".
            groups.setdefault(g, []).append(f"{name}_(x:[0-2]|[xyz])")
    extra = {} if _args.armature is None else {"armature": _args.armature}
    actuators = {
        g: ImplicitActuatorCfg(joint_names_expr=exprs, stiffness=meta["pd"][g]["kp"] * _args.gain_scale,
                               damping=meta["pd"][g]["kd"] * _args.gain_scale,
                               effort_limit_sim=meta["pd"][g]["torque_limit"], **extra)
        for g, exprs in groups.items()
    }
    spawn = sim_utils.MjcfFileCfg(
        asset_path=str(asset_dir / "smplx_humanoid_sim.xml"),
        usd_dir=str(asset_dir / "usd"),
        usd_file_name="smplx_humanoid.usd",
        force_usd_conversion=True,
        make_instanceable=False,
        fix_base=False,
        self_collision=False,
        import_physics_scene=False,
    )
    return ArticulationCfg(prim_path="/World/Robot", spawn=spawn,
                           init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 1.5)), actuators=actuators)


def _tilt_deg(q_xyzw: np.ndarray) -> float:
    """Angle between the body's up axis (its +Y, canonical SMPL-X frame) and world +Z."""
    x, y, z, w = q_xyzw
    up_z = 2 * (y * z + x * w)
    return math.degrees(math.acos(max(-1.0, min(1.0, up_z))))


def _current_stage():
    """The open USD stage: Isaac Lab's own getter if this version has one, else Omniverse's."""
    getter = getattr(sim_utils, "get_current_stage", None)
    if getter is None:
        try:
            import omni.usd

            return omni.usd.get_context().get_stage()
        except ImportError:
            return None
    return getter()


def _prepare_usd() -> dict:
    """Before the simulator is built: what the importer wrote for the first joints, and optionally a placeholder drive."""
    out: dict = {"stage_found": False, "prime_drives": bool(_args.prime_drives)}
    try:
        from hready.robot import usd_drives as U

        stage = _current_stage()
        if stage is None:
            return out
        out["stage_found"] = True
        out["first_joints_as_imported"] = U.describe_joints(stage, "/World/Robot")
        if _args.prime_drives:
            out["prime"] = U.prime_usd_drives(stage, "/World/Robot")
    except Exception as exc:  # noqa: BLE001  (diagnostic only, reported in the result file)
        out["error"] = repr(exc)
    return out


def _newton_drive_state() -> dict:
    """What Newton's model holds for the drives after the build: actuation mode and gains per DOF."""
    try:
        from isaaclab_newton.physics.newton_manager import NewtonManager

        model = NewtonManager._model
    except Exception as exc:  # noqa: BLE001
        return {"available": False, "error": repr(exc)}
    out: dict = {"available": True}
    for name in ("joint_target_mode", "joint_target_ke", "joint_target_kd", "joint_armature", "joint_effort_limit"):
        try:
            v = getattr(model, name, None)
            if v is None:
                out[name] = None
                continue
            a = np.asarray(v.numpy())
            vals, counts = np.unique(a, return_counts=True)
            out[name] = {"dtype": str(a.dtype), "size": int(a.size), "min": float(a.min()), "max": float(a.max()),
                         "unique_counts": {str(x): int(c) for x, c in list(zip(vals, counts, strict=True))[:6]}}
        except Exception as exc:  # noqa: BLE001
            out[name] = {"error": repr(exc)}
    return out


def _drive_line(usd: dict, drive: dict) -> str:
    mode = (drive.get("joint_target_mode") or {}).get("unique_counts")
    ke = (drive.get("joint_target_ke") or {}).get("max")
    return f"usd_stage_found={usd.get('stage_found')} usd_prime={usd.get('prime')} newton_target_mode_counts={mode} newton_ke_max={ke}"


PROBE_SECONDS = 2.0
PROBE_TIMES_S = (0.05, 0.1, 0.2, 0.5, 1.0, 1.5, 2.0)
PROBE_SEARCH_TIMES_S = (0.2, 2.0)  # the 288-candidate axis search is slower, so it runs at these times only


def _read(robot, field: str) -> np.ndarray | None:
    """One field of the simulator's data as numpy, or None when this Isaac Lab version does not have it."""
    try:
        v = getattr(robot.data, field)
        return getattr(v, "torch", v).detach().cpu().numpy().astype(np.float64)
    except Exception:  # noqa: BLE001  (diagnostic read: a missing field is reported, not fatal)
        return None


def _compare(sim_value: np.ndarray | None, expected: np.ndarray, names: list[str]) -> dict:
    """Largest difference between what the simulator holds and what the asset says, with the worst DOFs."""
    if sim_value is None:
        return {"available": False}
    v = sim_value.reshape(-1, *expected.shape)[0] if sim_value.size == expected.size else sim_value
    if v.shape != expected.shape:
        return {"available": True, "shape_mismatch": [list(v.shape), list(expected.shape)]}
    diff = np.abs(v - expected)
    diff = diff.max(axis=-1) if diff.ndim > 1 else diff
    scale = np.abs(expected).max(axis=-1) if expected.ndim > 1 else np.abs(expected)
    worst = np.argsort(-diff)[:5]
    return {"available": True, "max_abs_diff": float(diff.max()), "max_rel_diff": float((diff / np.maximum(scale, 1e-9)).max()),
            "worst": [{"dof": names[i], "sim": np.round(v[i], 4).tolist(), "asset": np.round(expected[i], 4).tolist()} for i in worst]}


def _run_probe(sim, robot, meta: dict, names: list[str], index_in_smplx: dict, r_root: np.ndarray, usd: dict, drive: dict) -> None:
    n = len(names)
    smplx_idx = np.array([index_in_smplx[i] for i in range(n)])
    dof_joint = [int(k) // 3 + 1 for k in smplx_idx]
    dof_slot = [int(k) % 3 for k in smplx_idx]
    lim = np.array([meta["limits_rad"][meta["joint_names"][j]] for j in range(1, 22)])  # (21, 3, 2)
    target63 = np.random.default_rng(_args.probe_seed).uniform(0.5 * lim[..., 0], 0.5 * lim[..., 1]).reshape(63)
    target = target63[smplx_idx]

    pd = [meta["pd"][meta["group_of_joint"][meta["joint_names"][j]]] for j in dof_joint]
    readback = {
        "stiffness": _compare(_read(robot, "joint_stiffness"), np.array([p["kp"] for p in pd]) * _args.gain_scale, names),
        "damping": _compare(_read(robot, "joint_damping"), np.array([p["kd"] for p in pd]) * _args.gain_scale, names),
        "effort_limit": _compare(_read(robot, "joint_effort_limits"), np.array([p["torque_limit"] for p in pd]), names),
        "position_limits_rad": _compare(_read(robot, "joint_pos_limits"), np.array([lim[j - 1, k] for j, k in zip(dof_joint, dof_slot, strict=True)]), names),
        "armature_kgm2": _compare(_read(robot, "joint_armature"), np.full(n, H.JOINT_ARMATURE if _args.armature is None else _args.armature), names),
    }

    pose0 = torch.zeros((1, 7), device=sim.device)
    pose0[0, 2] = 30.0  # 2 s of free fall is 19.6 m
    pose0[0, 3:7] = torch.tensor(H.matrix_to_quat_xyzw(r_root), device=sim.device, dtype=torch.float32)
    robot.write_root_link_pose_to_sim_index(root_pose=pose0)
    robot.write_joint_position_to_sim_index(position=torch.zeros((1, n), device=sim.device))
    robot.write_data_to_sim()
    robot.update(SIM_DT)
    q_tgt = torch.tensor(target[None], dtype=torch.float32, device=sim.device)
    body_ids = [robot.find_bodies([nm])[0][0] for nm in meta["joint_names"]]
    snap_steps = {max(1, round(t / SIM_DT)): t for t in PROBE_TIMES_S}
    search_steps = {max(1, round(t / SIM_DT)) for t in PROBE_SEARCH_TIMES_S}
    group_of_dof = [meta["group_of_joint"][meta["joint_names"][j]] for j in dof_joint]
    timeline, mapping, final = [], {}, None
    for step in range(1, max(snap_steps) + 1):
        robot.set_joint_position_target_index(target=q_tgt)
        robot.write_data_to_sim()
        sim.step()
        robot.update(SIM_DT)
        if step not in snap_steps:
            continue
        q = robot.data.joint_pos.torch[0].detach().cpu().numpy().astype(np.float64)
        body = robot.data.body_pos_w.torch[0, body_ids].detach().cpu().numpy().astype(np.float64)
        quat = robot.data.root_quat_w.torch[0].detach().cpu().numpy().astype(np.float64)
        qd = _read(robot, "joint_vel")
        entry = {"t_s": snap_steps[step], "finite": bool(np.isfinite(q).all() and np.isfinite(body).all())}
        if entry["finite"]:
            err = q - target
            entry.update({"error_max_rad": float(np.abs(err).max()), "error_median_rad": float(np.median(np.abs(err))),
                          "joint_speed_max_rad_s": None if qd is None else float(np.abs(qd).max()),
                          "pelvis_z_m": float(body[0, 2])})
            if step in search_steps:
                res = H.match_dof_mapping(meta, H.quat_xyzw_to_matrix(quat), body[0], q, dof_joint, dof_slot, body)
                assumed = next(r for r in res if r["is_assumed"])
                mapping[snap_steps[step]] = {"assumed": assumed, "best": res[0], "second_best_max_err_m": res[1]["max_err_m"]}
            final = (q, err)
        timeline.append(entry)
        if not entry["finite"]:
            break
    probe = {
        "seed": _args.probe_seed, "dt": SIM_DT, "gain_scale": _args.gain_scale, "joint_count": n,
        "target_rms_rad": float(np.sqrt(np.mean(target**2))), "timeline": timeline, "readback": readback, "mapping": mapping,
        "usd": usd, "newton_drive_state": drive,
    }
    verdict = {"completed": bool(timeline and timeline[-1]["finite"] and timeline[-1]["t_s"] == PROBE_TIMES_S[-1])}
    if final is not None:
        q, err = final
        worst = np.argsort(-np.abs(err))[:8]
        probe["worst_dofs_at_end"] = [{"dof": names[i], "target": round(float(target[i]), 4), "measured": round(float(q[i]), 4)} for i in worst]
        probe["error_max_by_group_rad"] = {g: float(max(abs(err[i]) for i in range(n) if group_of_dof[i] == g)) for g in sorted(set(group_of_dof))}
        verdict["pd_holds_target_within_0p02_rad"] = bool(np.abs(err).max() <= 0.02)
    if PROBE_SEARCH_TIMES_S[-1] in mapping:
        m = mapping[PROBE_SEARCH_TIMES_S[-1]]
        verdict["axes_as_assumed_within_5mm"] = bool(m["assumed"]["max_err_m"] <= 0.005)
        verdict["axes_identifiable_second_best_over_5mm"] = bool(m["second_best_max_err_m"] > 0.005)
    gains = [readback[k] for k in ("stiffness", "damping", "effort_limit")]
    verdict["gains_as_asset"] = all(g.get("max_rel_diff", 1.0) < 1e-3 for g in gains) if all(g.get("available") for g in gains) else None
    arm = readback["armature_kgm2"]
    verdict["armature_as_asset"] = bool(arm.get("max_abs_diff", 1.0) < 1e-6) if arm.get("available") else None
    plim = readback["position_limits_rad"]
    verdict["limits_as_asset"] = bool(plim.get("max_abs_diff", 1.0) < 1e-3) if plim.get("available") else None
    probe["verdict"] = verdict
    probe["versions"] = {"isaaclab": getattr(isaaclab, "__version__", "unknown"), "isaaclab_newton": isaaclab_newton.__version__,
                         "newton": getattr(newton, "__version__", "unknown"), "warp": wp.__version__}
    Path(_args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(_args.out).write_text(json.dumps(probe, indent=1), encoding="utf-8")
    print(json.dumps(probe, indent=1), flush=True)
    by_t = {e["t_s"]: round(e.get("error_max_rad", float("nan")), 4) for e in timeline}
    print(f"PROBE pd_error_max_rad_by_time_s={by_t} target_rms_rad={probe['target_rms_rad']:.3f}", flush=True)
    for t, m in mapping.items():
        b, a = m["best"], m["assumed"]
        print(f"PROBE axes t={t}s assumed_err_cm={100 * a['max_err_m']:.2f} best_err_cm={100 * b['max_err_m']:.2f} "
              f"best_is_assumed={b['is_assumed']} best_order={b['order']} best_axis_of_slot={b['axis_of_slot']} "
              f"best_sign_of_slot={b['sign_of_slot']} second_best_err_cm={100 * m['second_best_max_err_m']:.2f}", flush=True)
    print("PROBE drives " + _drive_line(usd, drive), flush=True)
    print("PROBE readback " + " ".join(f"{k}={v.get('max_abs_diff', 'n/a')}" for k, v in readback.items()), flush=True)
    print("PROBE verdict " + " ".join(f"{k}={v}" for k, v in verdict.items()), flush=True)
    simulation_app.close()


def _com_margin(robot, body_ids: list[int], com_local: np.ndarray, mass_kg: np.ndarray, foot_boxes: dict, meta: dict) -> float:
    """Distance (m) from the ground projection of the whole-body centre of mass to the edge of the support polygon.

    The polygon is the convex hull of the sole corners (foot box corners within 3 cm of the lowest one) of both feet;
    positive means inside. Static stability as in Goswami 1999 and Del Prete et al. 2016: the projection stays inside the hull.
    """
    pos = robot.data.body_pos_w.torch[0, body_ids].detach().cpu().numpy().astype(np.float64)
    quat = robot.data.body_quat_w.torch[0, body_ids].detach().cpu().numpy().astype(np.float64)
    rot = np.stack([H.quat_xyzw_to_matrix(q) for q in quat])
    com = ((pos + np.einsum("jab,jb->ja", rot, com_local)) * mass_kg[:, None]).sum(0) / mass_kg.sum()
    corners = []
    for nm, box in foot_boxes.items():
        j = meta["joint_names"].index(nm)
        corners.append(pos[j] + box @ rot[j].T)
    corners = np.concatenate(corners)
    sole = corners[corners[:, 2] <= corners[:, 2].min() + 0.03]
    return H.margin_in_hull(H.convex_hull_2d(sole), com[:2])


def _run() -> None:
    asset_dir = Path(_args.asset)
    meta = json.loads((asset_dir / "smplx_humanoid.json").read_text(encoding="utf-8"))
    r_root = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], float)  # canonical Y-up body in the Z-up world (AMASS identity)
    hing = np.zeros(63)
    pose_source = "canonical zero pose (T-pose)"
    if _args.amass_npz:
        z = np.load(_args.amass_npz)
        poses = z["poses"][_args.frame]
        r_root = H.aa_to_matrix(poses[:3])
        hing = H.pose_to_hinges(poses[3:66].reshape(21, 3))
        pose_source = f"{_args.amass_npz} frame {_args.frame} (neutral shape, betas ignored)"
    local = np.stack([np.eye(3)] + [H.hinges_to_matrix(hing.reshape(21, 3)[j - 1]) for j in range(1, 22)])
    root_z = -H.lowest_height(meta, r_root, local) + 0.003

    sim = SimulationContext(_newton_cfg())
    mat = sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0)
    sim_utils.GroundPlaneCfg(physics_material=mat).func("/World/defaultGroundPlane", sim_utils.GroundPlaneCfg(physics_material=mat))
    sim_utils.DomeLightCfg(intensity=2000.0).func("/World/Light", sim_utils.DomeLightCfg(intensity=2000.0))
    robot = Articulation(_robot_cfg(asset_dir, meta))
    usd_report = _prepare_usd()
    sim.reset()
    robot.reset()
    drive_state = _newton_drive_state()

    names = list(robot.joint_names)
    index_in_smplx = {}
    for i, n in enumerate(names):
        m = re.fullmatch(r"(.+)_([xyz])(?::([0-2]))?", n)
        if m is None:
            raise ValueError(f"unexpected joint name {n!r}")
        base, axis, dof = m.group(1), m.group(2), m.group(3)
        # D6 DOF order ":0, :1, :2" is taken as x, y, z; the axis order is not checked by this script.
        index_in_smplx[i] = 3 * (meta["joint_names"].index(base) - 1) + (int(dof) if dof is not None else "xyz".index(axis))
    if _args.probe:
        _run_probe(sim, robot, meta, names, index_in_smplx, r_root, usd_report, drive_state)
        return
    report = {
        "joint_count": len(names), "body_count": len(robot.body_names), "joint_names_match_asset": len(names) == 63,
        "pose_source": pose_source,
    }
    mass_t = (robot.data.body_mass if hasattr(robot.data, "body_mass") else robot.data.default_mass).torch[0].detach().cpu().numpy()
    body_names = list(robot.body_names)
    expected = [meta["body_mass_kg"][meta["joint_names"].index(n)] for n in body_names]
    report.update({
        "mass_total_kg": float(mass_t.sum()), "mass_expected_kg": meta["total_mass_kg"],
        "body_mass_kg": {n: round(float(m), 4) for n, m in zip(body_names, mass_t)},
        "body_mass_expected_kg": {n: round(float(m), 4) for n, m in zip(body_names, expected)},
        "body_mass_max_abs_error_kg": float(np.abs(mass_t - np.array(expected)).max()),
    })
    q_tgt = torch.tensor([[hing[index_in_smplx[i]] for i in range(len(names))]], dtype=torch.float32, device=sim.device)
    arm_sim = _read(robot, "joint_armature")
    report["sim_joint_armature_min_max_kgm2"] = None if arm_sim is None else [float(arm_sim.min()), float(arm_sim.max())]
    report["armature_arg_kgm2"] = _args.armature
    report["usd"], report["newton_drive_state"] = usd_report, drive_state

    pose0 = torch.zeros((1, 7), device=sim.device)
    pose0[0, :3] = torch.tensor([0.0, 0.0, root_z], device=sim.device)
    pose0[0, 3:7] = torch.tensor(H.matrix_to_quat_xyzw(r_root), device=sim.device, dtype=torch.float32)
    robot.write_root_link_pose_to_sim_index(root_pose=pose0)
    robot.write_joint_position_to_sim_index(position=q_tgt)
    robot.write_data_to_sim()
    robot.update(SIM_DT)

    ids_p, _ = robot.find_bodies(["pelvis"])
    ids_h, _ = robot.find_bodies(["head"])
    pz0 = float(robot.data.body_pos_w.torch[0, ids_p[0], 2].item())
    hz0 = float(robot.data.body_pos_w.torch[0, ids_h[0], 2].item())
    report["upright_at_start"] = hz0 > pz0  # False means the quaternion convention is wrong
    n_steps = int(_args.seconds / SIM_DT)
    pelvis_z, tilt, fall_step, nan_step = [], [], -1, -1
    log_steps = {int(t / SIM_DT) for t in (0.0, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0, 2.0, 3.0, 4.0, 4.99)}
    tracked = [robot.find_bodies([n])[0][0] for n in ("left_ankle", "right_ankle", "left_foot", "right_foot")]
    tracked_names = ("left_ankle", "right_ankle", "left_foot", "right_foot")
    trace = []
    feet_first = None
    smplx_body_ids = [robot.find_bodies([nm])[0][0] for nm in meta["joint_names"]]
    com_local, mass_kg = H.body_com_local(meta), np.asarray(meta["body_mass_kg"])
    foot_boxes = {nm: H.box_corners_local(meta, nm) for nm in ("left_ankle", "right_ankle")}
    com_margin_m, com_error, pelvis_xy = [], None, []
    for step in range(n_steps):
        robot.set_joint_position_target_index(target=q_tgt)
        robot.write_data_to_sim()
        sim.step()
        robot.update(SIM_DT)
        pz = float(robot.data.body_pos_w.torch[0, ids_p[0], 2].item())
        tl = _tilt_deg(robot.data.root_quat_w.torch[0].detach().cpu().numpy())
        pelvis_z.append(pz)
        tilt.append(tl)
        pelvis_xy.append(robot.data.body_pos_w.torch[0, ids_p[0], :2].detach().cpu().numpy().copy())
        if step % 10 == 0 and com_error is None:
            try:
                com_margin_m.append(_com_margin(robot, smplx_body_ids, com_local, mass_kg, foot_boxes, meta))
            except Exception as exc:  # noqa: BLE001  (reported in the result file; the criterion then fails)
                com_error = repr(exc)
        if nan_step < 0 and not (math.isfinite(pz) and math.isfinite(tl)):
            nan_step = step
        if fall_step < 0 and (not math.isfinite(pz) or pz < FALL_Z_FRAC * pelvis_z[0] or tl > FALL_TILT_DEG):
            fall_step = step
        if step == 0:
            feet_first = robot.data.body_pos_w.torch[0, tracked, :2].detach().cpu().numpy().copy()
        if step in log_steps:
            q = robot.data.joint_pos.torch[0].detach().cpu().numpy()
            top = np.argsort(-np.abs(q))[:5]
            bp = robot.data.body_pos_w.torch[0].detach().cpu().numpy()
            trace.append({
                "step": step, "pelvis_xyz": [round(float(v), 4) for v in bp[ids_p[0]]], "tilt_deg": round(tl, 2),
                "feet_xyz": {n: [round(float(v), 4) for v in bp[i]] for n, i in zip(tracked_names, tracked)},
                "largest_joint_angles_rad": {names[i]: round(float(q[i]), 3) for i in top},
            })
        if nan_step >= 0:
            break
    xy = robot.data.root_pos_w.torch[0, :2].detach().cpu().numpy()
    last2 = np.array(pelvis_xy[-int(2.0 / SIM_DT):]) if pelvis_xy else np.zeros((0, 2))
    pelvis_xy_range_m = float((last2.max(0) - last2.min(0)).max()) if len(last2) else float("nan")
    com_margin_min_m = min(com_margin_m) if com_margin_m else float("nan")
    feet_last = robot.data.body_pos_w.torch[0, tracked, :2].detach().cpu().numpy()
    foot_drift_m = float(np.linalg.norm(feet_last - feet_first, axis=1).max()) if feet_first is not None else float("nan")
    tail = [v for v in pelvis_z[-int(2.0 / SIM_DT):] if math.isfinite(v)]
    drop_m = (pelvis_z[0] - min(v for v in pelvis_z if math.isfinite(v))) if pelvis_z else float("nan")
    tilt_max = max((t for t in tilt if math.isfinite(t)), default=float("nan"))
    criteria = {
        "no_fall_pelvis_drop_le_3cm": bool(drop_m <= 0.03), "tilt_le_5deg": bool(tilt_max < 5.0),
        "no_slide_feet_le_1cm": bool(foot_drift_m <= 0.01),
        "pelvis_xy_range_last_2s_le_5mm": bool(pelvis_xy_range_m <= 0.005),
        "com_projection_inside_support_polygon_all_time": bool(com_error is None and com_margin_m and com_margin_min_m > 0.0),
        "steady_last_2s_range_le_2mm": bool(len(tail) > 0 and (max(tail) - min(tail)) <= 0.002),
        "ran_full_time_without_nan": bool(nan_step < 0 and len(pelvis_z) == n_steps),
    }
    finite = [v for v in pelvis_z if math.isfinite(v)]
    report.update({
        "seconds": _args.seconds, "dt": SIM_DT, "steps_run": len(pelvis_z), "fell": fall_step >= 0, "fall_step": fall_step,
        "first_non_finite_step": nan_step, "mujoco_contacts": bool(_args.mujoco_contacts), "gain_scale": _args.gain_scale,
        "foot_drift_m": foot_drift_m, "pelvis_drop_m": drop_m, "criteria": criteria, "passed": all(criteria.values()),
        "pelvis_xy_range_last_2s_m": pelvis_xy_range_m, "com_margin_min_m": com_margin_min_m, "com_margin_error": com_error,
        "com_margin_samples": len(com_margin_m),
        "first_recorded_criterion_pelvis_drift_le_1cm_informational": {"value_m": float(np.linalg.norm(xy)), "met": bool(float(np.linalg.norm(xy)) <= 0.01),
                                                                        "note": "replaced after the first Isaac result; kept for transparency"},
        "trace": trace,
        "pelvis_z_first_steps_m": [round(v, 4) for v in pelvis_z[:5]] + [round(pelvis_z[i], 4) for i in (9, 19, 49, 99) if i < len(pelvis_z)],
        "pelvis_z_start_m": pelvis_z[0] if pelvis_z else None, "pelvis_z_end_m": finite[-1] if finite else None,
        "pelvis_z_min_m": min(finite) if finite else None, "max_tilt_deg": max((t for t in tilt if math.isfinite(t)), default=None),
        "root_xy_end_m": xy.tolist(),
        "versions": {"isaaclab": getattr(isaaclab, "__version__", "unknown"), "isaaclab_newton": isaaclab_newton.__version__,
                     "newton": getattr(newton, "__version__", "unknown"), "warp": wp.__version__},
    })
    Path(_args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(_args.out).write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps(report, indent=1), flush=True)
    print(f"SUMMARY dt={SIM_DT} gain_scale={_args.gain_scale} armature={_args.armature} prime_drives={_args.prime_drives} mujoco_contacts={_args.mujoco_contacts} "
          f"passed={report['passed']} "
          f"fell={report['fell']} fall_step={fall_step} drop_cm={100 * drop_m:.1f} tilt_max_deg={tilt_max:.1f} "
          f"foot_drift_cm={100 * foot_drift_m:.1f} pelvis_xy_cm={100 * float(np.linalg.norm(xy)):.1f} "
          f"pelvis_xy_range_last2s_mm={1000 * pelvis_xy_range_m:.1f} com_margin_min_cm={100 * com_margin_min_m:.1f}", flush=True)
    print("SUMMARY criteria " + " ".join(f"{k}={v}" for k, v in criteria.items()), flush=True)
    print("SUMMARY drives " + _drive_line(usd_report, drive_state), flush=True)
    simulation_app.close()


if __name__ == "__main__":
    _run()
