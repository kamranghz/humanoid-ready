"""Ego heading from the camera look axis: unit length, rotation invariance, fallback order (numpy only)."""

import numpy as np

from hready.data.ego_heading import (
    SOURCE_BACKFILL,
    SOURCE_LOOK,
    SOURCE_WORLD_Y,
    heading_sequence,
)


def _camera_rotations(look):
    """Rotation-like matrices whose third row is the given look direction (other rows unused)."""
    r = np.zeros((len(look), 3, 3))
    r[:, 2] = look
    return r


def _rot_z(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def test_heading_is_unit_and_rotates_with_the_world():
    rng = np.random.default_rng(0)
    look = rng.normal(size=(20, 3))
    look[:, 2] *= 0.3
    head, pelvis = rng.normal(size=(20, 3)), rng.normal(size=(20, 3))
    h, src = heading_sequence(_camera_rotations(look), head, pelvis)
    assert np.allclose(np.linalg.norm(h, axis=1), 1.0)
    rz = _rot_z(0.9)
    h_rot, src_rot = heading_sequence(_camera_rotations(look @ rz.T), head @ rz.T, pelvis @ rz.T)
    assert np.array_equal(src, src_rot)
    assert np.allclose(h_rot, h @ rz[:2, :2].T)


def test_fallback_backfills_clip_start_and_uses_world_axis_only_without_any_heading():
    look = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])  # vertical, vertical, then +x
    same = np.zeros((3, 3))  # pelvis == head: no pelvis->head direction
    h, src = heading_sequence(_camera_rotations(look), same, same)
    assert list(src) == [SOURCE_BACKFILL, SOURCE_BACKFILL, SOURCE_LOOK]
    assert np.allclose(h, [[1.0, 0.0]] * 3)
    h2, src2 = heading_sequence(_camera_rotations(look[:2]), same[:2], same[:2])
    assert list(src2) == [SOURCE_WORLD_Y, SOURCE_WORLD_Y] and np.allclose(h2, [[0.0, 1.0]] * 2)
