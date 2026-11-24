"""SMPL-X-skeleton humanoid asset: 22 body joints, 3 hinge DoF per joint, generated from the SMPL-X body model.

The asset is written as MJCF (MuJoCo XML). It keeps the SMPL-X joint order, so ``body_pose`` axis-angle rows map to
hinge triples and back (``pose_to_hinges`` / ``hinges_to_pose``). Body frames are kept in the canonical SMPL-X frame
(Y up, +Z forward); the root free joint carries the AMASS ``global_orient``, so an AMASS frame is loaded by
``qpos = [transl, quat(global_orient), hinges]``. No mesh is used in the simulator: collision shapes are capsules and
boxes fitted to the SMPL-X vertices that each body moves.

Choices (all documented here, none tuned to a result):

- Joint positions: the SMPL-X joint regressor on the neutral shape (or on the given betas).
- Masses: de Leva (1996) segment fractions from ``hready.losses._constants`` (trunk split over pelvis, spine1-3 and
  neck); the foot mass is split 85 % foot, 15 % toe body; the two collar bodies take 0.15 % of the total each from spine3
  because a body needs a positive mass. Total mass is a parameter (default 75 kg) and the fractions sum to it exactly.
- Joint limits: knees and elbows follow the item-3 table (``hready.losses.biomech``: flexion 0 to 150 deg on the
  hinge axis of that table, +-``_OFF_AXIS`` on the other two axes). All other joints use the generous per-group
  ranges in ``_LIMITS_DEG``; these are not anatomical limits, they only keep the Euler triple away from its singularity.
- Hinge order: the three hinges of a joint are x, y, z in the parent-aligned frame and compose as
  ``R = Rx(a) Ry(b) Rz(c)``; ``hinges_to_pose`` and ``pose_to_hinges`` are exact inverses for ``|b| < pi/2``.
- PD: position actuators, gains and torque limits per joint group in ``_GROUPS``.
- Collision: every body collides with the ground plane only (no self collision), as in most tracking setups.
"""

from __future__ import annotations

import itertools
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from hready.body.joint_indices import JOINT_NAMES_22
from hready.losses._constants import COM_MASS_FRAC_22

NUM_JOINTS = 22
DEFAULT_TOTAL_MASS_KG = 75.0
JOINT_ARMATURE = 0.01  # kg m^2 added to every hinge, as in the MJCF; the stand check can pass it to a simulator that dropped it
JOINT_PASSIVE_DAMPING = 0.5  # Nm s/rad on every hinge in the MJCF
_OFF_AXIS = np.deg2rad(20.0)

# (min, max) degrees on the three hinges (a, b, c) = (x, y, z) per joint name; default for joints not listed.
_LIMITS_DEG = {
    "left_hip": ((-120, 50), (-60, 60), (-60, 60)),
    "right_hip": ((-120, 50), (-60, 60), (-60, 60)),
    "left_knee": ((0, 150), None, None),  # item-3 table; off axes filled with +-_OFF_AXIS
    "right_knee": ((0, 150), None, None),
    "left_ankle": ((-50, 50), (-30, 30), (-30, 30)),
    "right_ankle": ((-50, 50), (-30, 30), (-30, 30)),
    "left_foot": ((-45, 45), (-20, 20), (-20, 20)),
    "right_foot": ((-45, 45), (-20, 20), (-20, 20)),
    "spine1": ((-45, 45), (-40, 40), (-40, 40)),
    "spine2": ((-45, 45), (-40, 40), (-40, 40)),
    "spine3": ((-45, 45), (-40, 40), (-40, 40)),
    "neck": ((-60, 60), (-60, 60), (-45, 45)),
    "head": ((-60, 60), (-60, 60), (-45, 45)),
    "left_collar": ((-30, 30), (-30, 30), (-30, 30)),
    "right_collar": ((-30, 30), (-30, 30), (-30, 30)),
    "left_shoulder": ((-120, 120), (-90, 90), (-120, 120)),
    "right_shoulder": ((-120, 120), (-90, 90), (-120, 120)),
    "left_elbow": (None, (-150, 0), None),  # item-3 table: left flexion = -b
    "right_elbow": (None, (0, 150), None),  # right flexion = +b
    "left_wrist": ((-80, 80), (-60, 60), (-80, 80)),
    "right_wrist": ((-80, 80), (-60, 60), (-80, 80)),
}
_DEFAULT_ELBOW_OFF_DEG = ((-90, 90), None, (-60, 60))

# joint-name group -> (kp [Nm/rad], kd [Nm s/rad], torque limit [Nm]); one PD pair per group, set from body mass and
# lever arm (stiffness above m g h at the ankle), not from a rollout result.
_GROUPS = {
    "leg": (1000.0, 100.0, 400.0),
    "foot": (1200.0, 100.0, 300.0),  # ankle stiffness must exceed m g h (about 75 * 9.81 * 0.9 = 660 Nm/rad)
    "spine": (1000.0, 100.0, 300.0),
    "head": (200.0, 20.0, 60.0),
    "arm": (400.0, 40.0, 150.0),
    "hand": (100.0, 10.0, 40.0),
}
_GROUP_OF = {
    "pelvis": "spine", "left_hip": "leg", "right_hip": "leg", "left_knee": "leg", "right_knee": "leg",
    "left_ankle": "foot", "right_ankle": "foot", "left_foot": "foot", "right_foot": "foot",
    "spine1": "spine", "spine2": "spine", "spine3": "spine", "neck": "head", "head": "head",
    "left_collar": "arm", "right_collar": "arm", "left_shoulder": "arm", "right_shoulder": "arm",
    "left_elbow": "arm", "right_elbow": "arm", "left_wrist": "hand", "right_wrist": "hand",
}
_FOOT_BODIES = {7: 10, 8: 11}  # ankle body -> toe body
_FOOT_TOE_SHARE = 0.15
_COLLAR_FRAC = 0.0015
_MIN_RADIUS, _MAX_RADIUS = 0.02, 0.12


# ---------------------------------------------------------------------------------------------- rotations
def _rx(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def _ry(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def _rz(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def aa_to_matrix(aa: np.ndarray) -> np.ndarray:
    """Rodrigues, aa (...,3) -> (...,3,3)."""
    aa = np.asarray(aa, dtype=np.float64)
    th = np.linalg.norm(aa, axis=-1, keepdims=True)
    k = aa / np.maximum(th, 1e-12)
    kx, ky, kz = k[..., 0], k[..., 1], k[..., 2]
    z = np.zeros_like(kx)
    K = np.stack([z, -kz, ky, kz, z, -kx, -ky, kx, z], axis=-1).reshape(aa.shape[:-1] + (3, 3))
    t = th[..., None]
    return np.eye(3) + np.sin(t) * K + (1 - np.cos(t)) * (K @ K)


def matrix_to_aa(r: np.ndarray) -> np.ndarray:
    r = np.asarray(r, dtype=np.float64)
    cos = np.clip((np.trace(r, axis1=-2, axis2=-1) - 1) / 2, -1, 1)
    th = np.arccos(cos)
    v = np.stack([r[..., 2, 1] - r[..., 1, 2], r[..., 0, 2] - r[..., 2, 0], r[..., 1, 0] - r[..., 0, 1]], axis=-1)
    s = np.sin(th)[..., None]
    return np.where(s > 1e-8, v / (2 * np.maximum(s, 1e-12)) * th[..., None], v / 2)


def matrix_to_hinges(r: np.ndarray) -> np.ndarray:
    """R = Rx(a) Ry(b) Rz(c) -> (a, b, c); valid for |b| < pi/2."""
    b = np.arcsin(np.clip(r[..., 0, 2], -1, 1))
    a = np.arctan2(-r[..., 1, 2], r[..., 2, 2])
    c = np.arctan2(-r[..., 0, 1], r[..., 0, 0])
    return np.stack([a, b, c], axis=-1)


def hinges_to_matrix(h: np.ndarray) -> np.ndarray:
    h = np.asarray(h, dtype=np.float64)
    out = np.empty(h.shape[:-1] + (3, 3))
    for idx in np.ndindex(h.shape[:-1]):
        out[idx] = _rx(h[idx][0]) @ _ry(h[idx][1]) @ _rz(h[idx][2])
    return out


def pose_to_hinges(body_pose_aa: np.ndarray) -> np.ndarray:
    """SMPL-X body_pose axis-angle (21,3) -> hinge angles (63,) in the asset's qpos order (joints 1..21, a b c each)."""
    return matrix_to_hinges(aa_to_matrix(np.asarray(body_pose_aa).reshape(21, 3))).reshape(63)


def hinges_to_pose(hinges: np.ndarray) -> np.ndarray:
    return matrix_to_aa(hinges_to_matrix(np.asarray(hinges).reshape(21, 3))).reshape(21, 3)


def forward_kinematics(rest: np.ndarray, parents: np.ndarray, root_rot: np.ndarray, root_pos: np.ndarray,
                       local_rot: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Standard kinematic chain: rest joints (22,3), local rotations (22,3,3) with the root entry unused.
    Returns world joint positions (22,3) and world rotations (22,3,3)."""
    world = np.zeros((NUM_JOINTS, 3, 3))
    pos = np.zeros((NUM_JOINTS, 3))
    world[0] = root_rot
    pos[0] = root_pos
    for j in range(1, NUM_JOINTS):
        p = parents[j]
        pos[j] = pos[p] + world[p] @ (rest[j] - rest[p])
        world[j] = world[p] @ local_rot[j]
    return pos, world


def forward_joints(rest: np.ndarray, parents: np.ndarray, root_rot: np.ndarray, root_pos: np.ndarray,
                   local_rot: np.ndarray) -> np.ndarray:
    return forward_kinematics(rest, parents, root_rot, root_pos, local_rot)[0]


def _axis_rot(axis: int, angle: np.ndarray) -> np.ndarray:
    """Rotation matrices about the x (0), y (1) or z (2) axis for an array of angles."""
    c, s = np.cos(angle), np.sin(angle)
    i, j = (axis + 1) % 3, (axis + 2) % 3
    out = np.zeros(angle.shape + (3, 3))
    out[..., axis, axis] = 1.0
    out[..., i, i] = c
    out[..., j, j] = c
    out[..., j, i] = s
    out[..., i, j] = -s
    return out


def matrix_to_quat_xyzw(r: np.ndarray) -> np.ndarray:
    """Rotation matrix to quaternion (x, y, z, w), the layout Isaac Lab's Newton backend uses."""
    t = np.trace(r)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        return np.array([(r[2, 1] - r[1, 2]) / s, (r[0, 2] - r[2, 0]) / s, (r[1, 0] - r[0, 1]) / s, 0.25 * s])
    i = int(np.argmax(np.diag(r)))
    j, k = (i + 1) % 3, (i + 2) % 3
    s = np.sqrt(1.0 + r[i, i] - r[j, j] - r[k, k]) * 2
    q = np.zeros(4)
    q[i] = 0.25 * s
    q[j] = (r[j, i] + r[i, j]) / s
    q[k] = (r[k, i] + r[i, k]) / s
    q[3] = (r[k, j] - r[j, k]) / s
    return q


def quat_xyzw_to_matrix(q: np.ndarray) -> np.ndarray:
    x, y, z, w = np.asarray(q, float) / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def match_dof_mapping(meta: dict, root_rot: np.ndarray, root_pos: np.ndarray, dof_values: np.ndarray,
                      dof_joint: list[int], dof_slot: list[int], body_pos: np.ndarray) -> list[dict]:
    """Which reading of the simulator's three DOFs per joint reproduces the measured body positions.

    ``dof_values[i]`` is the measured angle of one simulator DOF, ``dof_joint[i]`` the SMPL-X joint (1..21) it belongs to and
    ``dof_slot[i]`` its position (0, 1, 2) among that joint's three DOFs. ``body_pos`` holds the simulator's body positions in
    SMPL-X joint order. A candidate reading says which axis (x, y, z) each slot turns about (``axis_of_slot``), with which sign
    (``sign_of_slot``), and in which order the three rotations are applied (``order``, slot numbers, first = outermost). The
    reading this asset was written for is slot k about axis k, positive, order (0, 1, 2): ``is_assumed`` marks it. All
    6 x 6 x 8 = 288 candidates are tried and returned sorted by the largest body-position error (m).
    """
    rest, par = np.asarray(meta["rest_joints"]), np.asarray(meta["parents"])
    vals = np.zeros((NUM_JOINTS - 1, 3))
    for v, j, k in zip(np.asarray(dof_values, float), dof_joint, dof_slot, strict=True):
        vals[j - 1, k] = v
    target = np.asarray(body_pos, float)[1:]
    out = []
    for order in itertools.permutations(range(3)):
        for axes in itertools.permutations(range(3)):
            for signs in itertools.product((1.0, -1.0), repeat=3):
                rots = [_axis_rot(axes[k], signs[k] * vals[:, k]) for k in range(3)]
                local = np.concatenate([np.eye(3)[None], rots[order[0]] @ rots[order[1]] @ rots[order[2]]])
                err = np.linalg.norm(forward_joints(rest, par, root_rot, root_pos, local)[1:] - target, axis=1)
                out.append({"order": list(order), "axis_of_slot": list(axes), "sign_of_slot": [int(x) for x in signs],
                            "is_assumed": order == (0, 1, 2) and axes == (0, 1, 2) and signs == (1.0, 1.0, 1.0),
                            "max_err_m": float(err.max()), "mean_err_m": float(err.mean())})
    out.sort(key=lambda r: (round(r["max_err_m"], 7), not r["is_assumed"]))  # an exact tie goes to the assumed reading
    return out


def body_com_local(meta: dict) -> np.ndarray:
    """Centre of mass of every body in its own frame (22, 3): the middle of a capsule, the centre of a box, the origin of a toe."""
    out = np.zeros((NUM_JOINTS, 3))
    for j, name in enumerate(meta["joint_names"]):
        g = meta["geoms"].get(name)
        if g is not None:
            out[j] = (np.asarray(g["a"]) + np.asarray(g["b"])) / 2 if g["type"] == "capsule" else np.asarray(g["pos"])
    return out


def convex_hull_2d(points: np.ndarray) -> np.ndarray:
    """Convex hull of 2-D points, counter-clockwise, no repeated first point (monotone chain)."""
    pts = sorted({(float(x), float(y)) for x, y in np.asarray(points)[:, :2]})
    if len(pts) < 3:
        return np.array(pts)

    def half(seq):
        h: list[tuple[float, float]] = []
        for q in seq:
            while len(h) >= 2 and (h[-1][0] - h[-2][0]) * (q[1] - h[-2][1]) - (h[-1][1] - h[-2][1]) * (q[0] - h[-2][0]) <= 0:
                h.pop()
            h.append(q)
        return h[:-1]

    return np.array(half(pts) + half(pts[::-1]))


def margin_in_hull(hull: np.ndarray, point: np.ndarray) -> float:
    """Smallest distance from ``point`` to an edge of the counter-clockwise ``hull``; negative when the point is outside."""
    d = []
    for a, b in zip(hull, np.roll(hull, -1, axis=0), strict=True):
        e = b - a
        d.append(float((e[0] * (point[1] - a[1]) - e[1] * (point[0] - a[0])) / np.linalg.norm(e)))
    return min(d)


def box_corners_local(meta: dict, joint_name: str) -> np.ndarray:
    g = meta["geoms"][joint_name]
    c, h = np.asarray(g["pos"]), np.asarray(g["size"])
    return np.array([c + np.array([sx * h[0], sy * h[1], sz * h[2]]) for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])


def lowest_height(meta: dict, root_rot: np.ndarray, local_rot: np.ndarray, up: int = 2) -> float:
    """Lowest point of all collision shapes (world coordinate ``up``) for the root at the origin; spawn the root at
    ``-lowest_height + clearance`` to stand on the floor in any pose."""
    rest = np.asarray(meta["rest_joints"])
    pos, rot = forward_kinematics(rest, np.asarray(meta["parents"]), root_rot, np.zeros(3), local_rot)
    low = np.inf
    for j, name in enumerate(meta["joint_names"]):
        g = meta["geoms"].get(name)
        if g is None:
            continue
        if g["type"] == "capsule":
            pts = [np.asarray(g["a"]), np.asarray(g["b"])]
            r = g["radius"]
        else:
            c, h = np.asarray(g["pos"]), np.asarray(g["size"])
            pts = [c + np.array([sx * h[0], sy * h[1], sz * h[2]]) for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]
            r = 0.0
        for q in pts:
            low = min(low, float((pos[j] + rot[j] @ q)[up]) - r)
    return low


# ---------------------------------------------------------------------------------------------- SMPL-X data
@dataclass
class SmplxRest:
    joints: np.ndarray  # (22, 3) canonical frame, metres
    parents: np.ndarray  # (22,)
    vertices: np.ndarray  # (10475, 3)
    body_of_vertex: np.ndarray  # (10475,) body index 0..21 of the dominant LBS joint


def load_smplx_rest(npz_path: str | Path, betas: np.ndarray | None = None) -> SmplxRest:
    z = np.load(npz_path, allow_pickle=True)
    v = z["v_template"].astype(np.float64)
    if betas is not None:
        b = np.asarray(betas, dtype=np.float64)
        v = v + np.einsum("vcb,b->vc", z["shapedirs"][:, :, : len(b)].astype(np.float64), b)
    full_parents = z["kintree_table"][0].astype(np.int64)
    full_parents[0] = -1
    joints_full = z["J_regressor"].astype(np.float64) @ v
    # ancestor body (< 22) of every one of the 55 SMPL-X joints: hands -> wrist, jaw and eyes -> head
    anc = np.zeros(len(full_parents), dtype=np.int64)
    for j in range(len(full_parents)):
        k = j
        while k >= NUM_JOINTS:
            k = full_parents[k]
        anc[j] = k
    body_of_vertex = anc[np.argmax(z["weights"], axis=1)]
    return SmplxRest(joints_full[:NUM_JOINTS], full_parents[:NUM_JOINTS], v, body_of_vertex)


def smplx_to_model(hinges63: np.ndarray, asset_meta: dict) -> np.ndarray:
    """Reorder a (63,) SMPL-X-ordered hinge vector into the model's qpos / ctrl order."""
    return np.asarray(hinges63)[np.asarray(asset_meta["hinge_perm"])]


# ---------------------------------------------------------------------------------------------- asset
@dataclass
class HumanoidAsset:
    mjcf: str
    meta: dict
    mjcf_sim: str = ""


def _children(parents: np.ndarray) -> list[list[int]]:
    ch = [[] for _ in range(NUM_JOINTS)]
    for j in range(1, NUM_JOINTS):
        ch[parents[j]].append(j)
    return ch


def _axis_distance(pts: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    d = b - a
    L2 = float(d @ d)
    t = np.clip(((pts - a) @ d) / max(L2, 1e-12), 0.0, 1.0)
    return np.linalg.norm(pts - (a + t[:, None] * d), axis=1)


def _limits_rad(name: str) -> np.ndarray:
    spec = _LIMITS_DEG[name]
    out = np.zeros((3, 2))
    for k in range(3):
        if spec[k] is not None:
            out[k] = np.deg2rad(spec[k])
        else:
            out[k] = (-_OFF_AXIS, _OFF_AXIS)
    if name.endswith("elbow"):
        out[0] = np.deg2rad((-90, 90))
        out[2] = np.deg2rad((-60, 60))
    return out


def body_masses(total_mass: float) -> np.ndarray:
    frac = np.array(COM_MASS_FRAC_22, dtype=np.float64)
    for ankle, toe in _FOOT_BODIES.items():
        foot = frac[toe]  # the repo table puts the foot mass on the toe joint index
        frac[toe] = foot * _FOOT_TOE_SHARE
        frac[ankle] = foot * (1.0 - _FOOT_TOE_SHARE)
    for collar in (13, 14):
        frac[collar] = _COLLAR_FRAC
    frac[9] -= 2 * _COLLAR_FRAC
    frac = frac / frac.sum()
    return frac * total_mass


def _inertial_xml(g: dict | None, mass: float) -> str:
    """Explicit <inertial> for one body: mass and inertia of its single collision shape (a sphere of r = 2 cm for the toe).

    Written for every body because Isaac's importer takes the body mass from shape density, not from the geom mass attribute.
    """
    f = lambda x: " ".join(f"{v:.9g}" for v in x)
    if g is None:
        i = 0.4 * mass * 0.02**2
        return f'<inertial pos="0 0 0" mass="{mass:.6f}" diaginertia="{i:.9g} {i:.9g} {i:.9g}"/>'
    if g["type"] == "box":
        sx, sy, sz = 2 * np.asarray(g["size"], float)
        d = mass / 12.0 * np.array([sy**2 + sz**2, sx**2 + sz**2, sx**2 + sy**2])
        return f'<inertial pos="{f(g["pos"])}" mass="{mass:.6f}" diaginertia="{f(d)}"/>'
    a, b, r = np.asarray(g["a"], float), np.asarray(g["b"], float), float(g["radius"])
    length = float(np.linalg.norm(b - a))
    v_cyl, v_sph = np.pi * r * r * length, 4.0 / 3.0 * np.pi * r**3
    mc, ms = mass * v_cyl / (v_cyl + v_sph), mass * v_sph / (v_cyl + v_sph)
    axial = mc * r * r / 2 + ms * 2 * r * r / 5
    perp = mc * (length**2 / 12 + r * r / 4) + ms * (2 * r * r / 5 + length**2 / 4 + 3 * length * r / 8)
    u = (b - a) / length
    m3 = perp * np.eye(3) + (axial - perp) * np.outer(u, u)
    full = [m3[0, 0], m3[1, 1], m3[2, 2], m3[0, 1], m3[0, 2], m3[1, 2]]
    return f'<inertial pos="{f((a + b) / 2)}" mass="{mass:.6f}" fullinertia="{f(full)}"/>'


def build_humanoid(rest: SmplxRest, total_mass: float = DEFAULT_TOTAL_MASS_KG) -> HumanoidAsset:
    J, par = rest.joints, rest.parents
    ch = _children(par)
    mass = body_masses(total_mass)
    ground_lift = -float(rest.vertices[:, 1].min())  # standing: sole at y = min vertex
    lines: list[str] = []
    geoms: dict[int, dict] = {}
    order: list[int] = []  # joints in the order MuJoCo numbers them (depth first), which is not the SMPL-X order

    def f(x): return " ".join(f"{v:.6f}" for v in x)

    for j in range(NUM_JOINTS):
        mine = rest.vertices[rest.body_of_vertex == j]
        if j in _FOOT_BODIES:
            toe = _FOOT_BODIES[j]
            pts = rest.vertices[np.isin(rest.body_of_vertex, (j, toe))]
            lo, hi = pts.min(0) - J[j], pts.max(0) - J[j]
            geoms[j] = {"type": "box", "pos": (lo + hi) / 2, "size": (hi - lo) / 2}
            continue
        if j in _FOOT_BODIES.values():
            geoms[j] = None
            continue
        if j == 0:
            a, b = J[1], J[2]
        elif ch[j]:
            a, b = J[j], np.mean([J[c] for c in ch[j]], axis=0)
        else:  # leaf: head and the two hands run to the farthest dominant vertex along the parent direction
            u = J[j] - J[par[j]]
            u = u / np.linalg.norm(u)
            a, b = J[j], J[j] + u * max(float(((mine - J[j]) @ u).max()), 0.05)
        pts = mine if len(mine) else J[j][None]
        r = float(np.clip(np.median(_axis_distance(pts, a, b)), _MIN_RADIUS, _MAX_RADIUS))
        geoms[j] = {"type": "capsule", "a": a - J[j], "b": b - J[j], "radius": r}

    def emit(j: int, depth: int) -> None:
        ind = "  " * (depth + 2)
        name = JOINT_NAMES_22[j]
        order.append(j)
        if j == 0:
            lines.append(f'{ind}<body name="{name}" pos="0 0 0">')
            lines.append(f'{ind}  <freejoint name="root"/>')
        else:
            lines.append(f'{ind}<body name="{name}" pos="{f(J[j] - J[par[j]])}">')
            lim = _limits_rad(name)
            for k, ax in enumerate(("1 0 0", "0 1 0", "0 0 1")):
                lines.append(
                    f'{ind}  <joint name="{name}_{"xyz"[k]}" type="hinge" axis="{ax}" range="{lim[k, 0]:.6f} {lim[k, 1]:.6f}"'
                    f' limited="true" armature="{JOINT_ARMATURE}" damping="{JOINT_PASSIVE_DAMPING}"/>')
        g = geoms[j]
        lines.append(f"{ind}  " + _inertial_xml(g, float(mass[j])))
        if g is None:  # toe body: mass only, no collision
            pass
        elif g["type"] == "capsule":
            lines.append(f'{ind}  <geom name="{name}_col" type="capsule" fromto="{f(g["a"])} {f(g["b"])}" size="{g["radius"]:.6f}"'
                         f' contype="1" conaffinity="0" friction="1.0 0.005 0.0001"/>')
        else:
            lines.append(f'{ind}  <geom name="{name}_col" type="box" pos="{f(g["pos"])}" size="{f(g["size"])}"'
                         f' contype="1" conaffinity="0" friction="1.0 0.005 0.0001"/>')
        for c in ch[j]:
            emit(c, depth + 1)
        lines.append(f"{ind}</body>")

    emit(0, 0)
    act = []
    for j in order[1:]:
        name = JOINT_NAMES_22[j]
        kp, kd, tq = _GROUPS[_GROUP_OF[name]]
        for k in range(3):
            act.append(f'    <position name="{name}_{"xyz"[k]}" joint="{name}_{"xyz"[k]}" kp="{kp}" kv="{kd}" forcerange="{-tq} {tq}" forcelimited="true"/>')
    head = (
        '<mujoco model="smplx_humanoid">\n'
        '  <compiler angle="radian" inertiafromgeom="auto" balanceinertia="true"/>\n'
        '  <option timestep="0.002" gravity="0 0 -9.81" integrator="implicitfast"/>\n'
        '  <worldbody>\n')
    floor = '    <geom name="floor" type="plane" size="20 20 0.1" contype="0" conaffinity="1" friction="1.0 0.005 0.0001"/>\n'
    body = "\n".join(lines) + "\n  </worldbody>\n"
    xml = head + floor + body + "  <actuator>\n" + "\n".join(act) + "\n  </actuator>\n</mujoco>\n"
    xml_sim = head + body + "</mujoco>\n"  # for Isaac Lab: no floor and no actuators; the drives come from the actuator config
    meta = {
        "joint_names": list(JOINT_NAMES_22),
        "hinge_perm": [3 * (j - 1) + k for j in order[1:] for k in range(3)],
        # q_model[i] = hinges_smplx[hinge_perm[i]]; same for ctrl
        "parents": par.tolist(), "rest_joints": J.tolist(),
        "total_mass_kg": total_mass, "body_mass_kg": mass.tolist(), "mass_source": "de Leva 1996 via hready.losses._constants",
        "limits_rad": {JOINT_NAMES_22[j]: _limits_rad(JOINT_NAMES_22[j]).tolist() for j in range(1, NUM_JOINTS)},
        "pd": {g: {"kp": v[0], "kd": v[1], "torque_limit": v[2]} for g, v in _GROUPS.items()},
        "group_of_joint": _GROUP_OF, "root_height_standing_m": float(J[0][1]) + ground_lift,
        "joint_armature_kgm2": JOINT_ARMATURE, "joint_passive_damping": JOINT_PASSIVE_DAMPING,
        "frame": "canonical SMPL-X (Y up, +Z forward); root free joint carries the AMASS global_orient",
        "geoms": {JOINT_NAMES_22[j]: (None if g is None else {k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in g.items()})
                  for j, g in geoms.items()},
    }
    return HumanoidAsset(xml, meta, xml_sim)


def write_asset(asset: HumanoidAsset, out_dir: str | Path) -> tuple[Path, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    x, m = out / "smplx_humanoid.xml", out / "smplx_humanoid.json"
    x.write_text(asset.mjcf, encoding="utf-8")
    (out / "smplx_humanoid_sim.xml").write_text(asset.mjcf_sim, encoding="utf-8")
    m.write_text(json.dumps(asset.meta, indent=1), encoding="utf-8")
    return x, m
