"""SMPL-X-skeleton humanoid asset (numpy only; the MuJoCo check is skipped when mujoco is not installed)."""

import xml.etree.ElementTree as ET

import numpy as np
import pytest

from hready.losses._constants import COM_MASS_FRAC_22
from hready.robot import humanoid_asset as H

# neutral SMPL-X rest joints (m, canonical frame, rounded) for the 22 body joints
_REST = np.array([
    [0.003, -0.351, 0.012], [0.061, -0.444, -0.014], [-0.060, -0.455, -0.009], [0.0, -0.242, -0.016],
    [0.116, -0.823, -0.023], [-0.104, -0.818, -0.026], [0.010, -0.110, -0.022], [0.073, -1.226, -0.055],
    [-0.089, -1.228, -0.046], [-0.002, -0.057, 0.007], [0.120, -1.284, 0.063], [-0.128, -1.287, 0.073],
    [-0.014, 0.108, -0.025], [0.045, 0.028, 0.0], [-0.049, 0.027, -0.006], [0.011, 0.269, -0.004],
    [0.164, 0.085, -0.016], [-0.152, 0.080, -0.019], [0.418, 0.013, -0.058], [-0.423, 0.044, -0.046],
    [0.670, 0.036, -0.061], [-0.672, 0.039, -0.061]])
_PARENTS = np.array([-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19])


def _rest():
    """Joints from the table above; vertices are points around each bone, labelled with their body."""
    rng = np.random.default_rng(0)
    verts, label = [], []
    for j in range(22):
        end = _REST[j] + (_REST[j] - _REST[_PARENTS[j]]) * 0.5 if _PARENTS[j] >= 0 else _REST[j] + [0, 0.1, 0]
        pts = _REST[j] + rng.uniform(0, 1, (30, 1)) * (end - _REST[j]) + rng.normal(0, 0.03, (30, 3))
        verts.append(pts)
        label += [j] * 30
    return H.SmplxRest(_REST, _PARENTS, np.concatenate(verts), np.array(label))


def test_hinge_round_trip():
    rng = np.random.default_rng(1)
    aa = rng.normal(0, 0.4, (21, 3))
    back = H.hinges_to_pose(H.pose_to_hinges(aa))
    assert np.abs(H.aa_to_matrix(back) - H.aa_to_matrix(aa)).max() < 1e-9


def test_masses_sum_and_de_leva_fractions():
    m = H.body_masses(75.0)
    assert abs(m.sum() - 75.0) < 1e-9
    assert (m > 0).all()
    # head and thigh keep their de Leva fraction (collars take 0.3 % of the total from spine3 only)
    assert abs(m[15] / 75.0 - COM_MASS_FRAC_22[15]) < 1e-3
    assert abs(m[1] / 75.0 - COM_MASS_FRAC_22[1]) < 1e-3


def test_mjcf_structure_and_joint_order():
    asset = H.build_humanoid(_rest())
    root = ET.fromstring(asset.mjcf)
    assert len(root.findall(".//joint[@type='hinge']")) == 63
    assert len(root.findall(".//position")) == 63
    perm = asset.meta["hinge_perm"]
    assert sorted(perm) == list(range(63))
    assert abs(sum(asset.meta["body_mass_kg"]) - asset.meta["total_mass_kg"]) < 1e-9
    # knee flexion limit is the item-3 table range, elbows are mirrored
    assert np.allclose(np.degrees(asset.meta["limits_rad"]["left_knee"][0]), [0, 150])
    assert np.allclose(np.degrees(asset.meta["limits_rad"]["left_elbow"][1]), [-150, 0])
    assert np.allclose(np.degrees(asset.meta["limits_rad"]["right_elbow"][1]), [0, 150])


def test_mujoco_forward_kinematics_matches():
    mujoco = pytest.importorskip("mujoco")
    asset = H.build_humanoid(_rest())
    model = mujoco.MjModel.from_xml_string(asset.mjcf)
    data = mujoco.MjData(model)
    rng = np.random.default_rng(2)
    hing = rng.uniform(-0.4, 0.4, 63)
    hing[[3 * (4 - 1), 3 * (5 - 1)]] = 0.8  # knees flexed inside their range
    data.qpos[:3] = [0, 0, 1]
    data.qpos[3:7] = [np.cos(np.pi / 4), np.sin(np.pi / 4), 0, 0]
    data.qpos[7:] = H.smplx_to_model(hing, asset.meta)
    mujoco.mj_forward(model, data)
    aa = H.hinges_to_pose(hing)
    local = np.stack([np.eye(3)] + [H.aa_to_matrix(aa[j - 1]) for j in range(1, 22)])
    r_root = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], float)
    fk = H.forward_joints(_REST, _PARENTS, r_root, np.array([0, 0, 1.0]), local)
    names = asset.meta["joint_names"]
    mj = np.stack([data.xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n)] for n in names])
    assert np.abs(fk - mj).max() < 1e-5


def test_lowest_height_puts_the_feet_on_the_floor():
    mujoco = pytest.importorskip("mujoco")
    asset = H.build_humanoid(_rest())
    r_root = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], float)
    low = H.lowest_height(asset.meta, r_root, np.stack([np.eye(3)] * 22))
    model = mujoco.MjModel.from_xml_string(asset.mjcf)
    data = mujoco.MjData(model)
    for dz, touching in ((0.002, False), (-0.002, True)):
        data.qpos[:] = 0
        data.qpos[:3] = [0, 0, -low + dz]
        data.qpos[3:7] = [np.cos(np.pi / 4), np.sin(np.pi / 4), 0, 0]
        mujoco.mj_forward(model, data)
        assert (data.ncon > 0) == touching


def test_sim_variant_has_no_floor_and_no_actuators():
    asset = H.build_humanoid(_rest())
    root = ET.fromstring(asset.mjcf_sim)
    assert root.find(".//geom[@type='plane']") is None
    assert root.find("actuator") is None
    assert len(root.findall(".//joint[@type='hinge']")) == 63


def _fake_simulator(meta, values, order, axes, signs, root_rot, root_pos):
    """Body positions of a simulator whose three DOFs per joint are read as (order, axes, signs); values (21, 3) in slot order."""
    rots = [H._axis_rot(axes[k], signs[k] * values[:, k]) for k in range(3)]
    local = np.concatenate([np.eye(3)[None], rots[order[0]] @ rots[order[1]] @ rots[order[2]]])
    return H.forward_joints(np.asarray(meta["rest_joints"]), np.asarray(meta["parents"]), root_rot, root_pos, local)


def test_dof_mapping_search_recovers_an_unusual_reading():
    meta = H.build_humanoid(_rest()).meta
    rng = np.random.default_rng(5)
    values = rng.uniform(-0.45, 0.45, (21, 3))
    dof_joint = [j for j in range(1, 22) for _ in range(3)]
    slot = [k for _ in range(1, 22) for k in range(3)]
    r_root = H.aa_to_matrix(np.array([0.3, -0.5, 0.2]))
    pos = np.array([0.1, 0.2, 1.0])
    for order, axes, signs in (((0, 1, 2), (0, 1, 2), (1, 1, 1)), ((2, 0, 1), (1, 2, 0), (-1, 1, -1)), ((2, 1, 0), (2, 1, 0), (1, 1, 1))):
        body = _fake_simulator(meta, values, order, axes, signs, r_root, pos)
        res = H.match_dof_mapping(meta, r_root, pos, values.reshape(63), dof_joint, slot, body)
        assert res[0]["order"] == list(order) and res[0]["axis_of_slot"] == list(axes) and res[0]["sign_of_slot"] == list(signs)
        assert res[0]["max_err_m"] < 1e-9 and res[1]["max_err_m"] > 1e-2
        assert res[0]["is_assumed"] == (order == (0, 1, 2))
        assert sum(r["is_assumed"] for r in res) == 1 and len(res) == 288


def test_dof_mapping_search_agrees_with_mujoco():
    mujoco = pytest.importorskip("mujoco")
    asset = H.build_humanoid(_rest())
    model = mujoco.MjModel.from_xml_string(asset.mjcf)
    data = mujoco.MjData(model)
    rng = np.random.default_rng(3)
    hing = np.zeros(63)
    for j in range(1, 22):
        lim = np.array(asset.meta["limits_rad"][asset.meta["joint_names"][j]])
        hing[3 * (j - 1): 3 * j] = rng.uniform(lim[:, 0] * 0.5, lim[:, 1] * 0.5)
    r_root = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], float)
    data.qpos[:3] = [0.0, 0.0, 1.0]
    data.qpos[3:7] = [np.cos(np.pi / 4), np.sin(np.pi / 4), 0, 0]
    data.qpos[7:] = H.smplx_to_model(hing, asset.meta)
    mujoco.mj_forward(model, data)
    body = np.stack([data.xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n)] for n in asset.meta["joint_names"]])
    dof_joint = [j for j in range(1, 22) for _ in range(3)]
    slot = [k for _ in range(1, 22) for k in range(3)]
    best = H.match_dof_mapping(asset.meta, r_root, data.qpos[:3], hing, dof_joint, slot, body)[0]
    assert best["is_assumed"] and best["max_err_m"] < 1e-4


def test_quaternion_helpers_round_trip():
    rng = np.random.default_rng(1)
    for _ in range(50):
        r = H.aa_to_matrix(rng.normal(size=3) * 2.0)
        assert np.allclose(H.quat_xyzw_to_matrix(H.matrix_to_quat_xyzw(r)), r, atol=1e-9)
    r = H.aa_to_matrix(np.array([np.pi, 0.0, 0.0]))  # the branch with a non-positive trace
    assert np.allclose(H.quat_xyzw_to_matrix(H.matrix_to_quat_xyzw(r)), r, atol=1e-9)
    up = H.quat_xyzw_to_matrix(H.matrix_to_quat_xyzw(np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], float)))[:, 1]
    assert np.allclose(up, [0, 0, 1])  # the canonical body +Y points along world +Z


def test_support_polygon_helpers():
    hull = H.convex_hull_2d(np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0.5, 0.5], [0.2, 0.9]], float))
    assert len(hull) == 4
    assert abs(H.margin_in_hull(hull, np.array([0.5, 0.5])) - 0.5) < 1e-12
    assert abs(H.margin_in_hull(hull, np.array([0.9, 0.5])) - 0.1) < 1e-12
    assert H.margin_in_hull(hull, np.array([1.2, 0.5])) < 0


def test_centre_of_mass_from_bodies_matches_mujoco():
    mujoco = pytest.importorskip("mujoco")
    asset = H.build_humanoid(_rest())
    model = mujoco.MjModel.from_xml_string(asset.mjcf)
    data = mujoco.MjData(model)
    rng = np.random.default_rng(8)
    hing = np.zeros(63)
    for j in range(1, 22):
        lim = np.array(asset.meta["limits_rad"][asset.meta["joint_names"][j]])
        hing[3 * (j - 1): 3 * j] = rng.uniform(lim[:, 0] * 0.5, lim[:, 1] * 0.5)
    data.qpos[:3] = [0.1, 0.2, 1.0]
    data.qpos[3:7] = [np.cos(np.pi / 4), np.sin(np.pi / 4), 0, 0]
    data.qpos[7:] = H.smplx_to_model(hing, asset.meta)
    mujoco.mj_forward(model, data)
    local = np.stack([np.eye(3)] + [H.hinges_to_matrix(hing.reshape(21, 3)[j - 1]) for j in range(1, 22)])
    pos, rot = H.forward_kinematics(np.asarray(asset.meta["rest_joints"]), np.asarray(asset.meta["parents"]),
                                    np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], float), data.qpos[:3], local)
    com_b = pos + np.einsum("jab,jb->ja", rot, H.body_com_local(asset.meta))
    m = np.asarray(asset.meta["body_mass_kg"])
    assert np.allclose((m[:, None] * com_b).sum(0) / m.sum(), data.subtree_com[1], atol=1e-4)
