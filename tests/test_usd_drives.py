import pytest

pytest.importorskip("pxr")
from pxr import Usd, UsdGeom, UsdPhysics

from hready.robot import usd_drives as U


def _stage():
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage, "/World")
    UsdGeom.Xform.Define(stage, "/World/Robot")
    return stage


def test_d6_joint_gets_a_drive_on_every_axis_and_existing_drives_are_kept():
    stage = _stage()
    d6 = UsdPhysics.Joint.Define(stage, "/World/Robot/hip_x")
    driven = UsdPhysics.Joint.Define(stage, "/World/Robot/knee_x")
    existing = UsdPhysics.DriveAPI.Apply(driven.GetPrim(), "rotX")
    existing.CreateStiffnessAttr(500.0)
    UsdPhysics.FixedJoint.Define(stage, "/World/Robot/weld")
    UsdPhysics.RevoluteJoint.Define(stage, "/World/Robot/elbow")
    report = U.prime_usd_drives(stage, "/World/Robot")
    assert report == {"joints_found": 3, "axes_primed": 3 + 2 + 1, "axes_already_driven": 1}
    for axis in U.D6_AXES:
        drive = UsdPhysics.DriveAPI.Get(d6.GetPrim(), axis)
        assert drive.GetStiffnessAttr().Get() > 0 and drive.GetDampingAttr().Get() > 0
    assert UsdPhysics.DriveAPI.Get(driven.GetPrim(), "rotX").GetStiffnessAttr().Get() == 500.0
    assert UsdPhysics.DriveAPI.Get(stage.GetPrimAtPath("/World/Robot/elbow"), "angular").GetStiffnessAttr().Get() > 0
    assert U.prime_usd_drives(stage, "/World/Robot")["axes_primed"] == 0  # a second call changes nothing


def test_missing_root_is_reported_not_raised():
    assert U.prime_usd_drives(_stage(), "/World/Nothing")["joints_found"] == 0
    assert U.describe_joints(_stage(), "/World/Nothing") == []


def test_prime_usd_file_saves_in_place_and_is_repeatable(tmp_path):
    path = tmp_path / "robot.usda"
    stage = Usd.Stage.CreateNew(str(path))
    root = UsdGeom.Xform.Define(stage, "/Robot")
    stage.SetDefaultPrim(root.GetPrim())
    UsdPhysics.Joint.Define(stage, "/Robot/hip_x")
    stage.GetRootLayer().Save()
    first = U.prime_usd_file(str(path))
    assert first["axes_primed"] == 3 and first["root_prim"] == "/Robot"
    reopened = Usd.Stage.Open(str(path))
    for axis in U.D6_AXES:
        assert UsdPhysics.DriveAPI.Get(reopened.GetPrimAtPath("/Robot/hip_x"), axis).GetStiffnessAttr().Get() > 0
    assert U.prime_usd_file(str(path))["axes_primed"] == 0


def test_prime_usd_file_without_default_prim_raises(tmp_path):
    path = tmp_path / "empty.usda"
    stage = Usd.Stage.CreateNew(str(path))
    stage.GetRootLayer().Save()
    with pytest.raises(ValueError):
        U.prime_usd_file(str(path))
