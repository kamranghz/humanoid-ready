"""USD helpers for the Isaac Lab humanoid check: make sure every joint axis carries a drive when the simulator is built.

The Isaac MJCF importer writes the three hinges of one body as one generic D6 joint. If that joint has no drive (stiffness
and damping both zero) when Newton builds its model, Newton marks the DOFs as not actuated, and gains that Isaac Lab writes
afterwards are never used. A placeholder drive with nonzero gains is enough to mark the DOFs as actuated; Isaac Lab's actuator
configuration then writes the real gains over it.
"""

from __future__ import annotations

from pxr import Usd, UsdPhysics

D6_AXES = ("rotX", "rotY", "rotZ")


def _joint_prims(stage: Usd.Stage, root_path: str) -> list[Usd.Prim]:
    root = stage.GetPrimAtPath(root_path)
    if not root or not root.IsValid():
        return []
    return [p for p in Usd.PrimRange(root) if p.IsA(UsdPhysics.Joint) and not p.IsA(UsdPhysics.FixedJoint)]


def _drive_names(prim: Usd.Prim) -> tuple[str, ...]:
    if prim.GetTypeName() == "PhysicsJoint":  # generic D6
        return D6_AXES
    if prim.IsA(UsdPhysics.PrismaticJoint):
        return ("linear",)
    return ("angular",)


def prime_usd_drives(stage: Usd.Stage, root_path: str, stiffness: float = 1.0, damping: float = 1.0,
                     max_force: float = 1.0e3) -> dict:
    """Give every drive axis under ``root_path`` nonzero gains where both are zero or missing. Returns what was done."""
    joints = _joint_prims(stage, root_path)
    primed = already = 0
    for prim in joints:
        for axis in _drive_names(prim):
            drive = UsdPhysics.DriveAPI.Apply(prim, axis)
            k_attr, d_attr = drive.GetStiffnessAttr(), drive.GetDampingAttr()
            k = k_attr.Get() if k_attr and k_attr.HasAuthoredValue() else 0.0
            d = d_attr.Get() if d_attr and d_attr.HasAuthoredValue() else 0.0
            if (k or 0.0) > 0.0 or (d or 0.0) > 0.0:
                already += 1
                continue
            drive.CreateTypeAttr("force")
            drive.CreateStiffnessAttr(float(stiffness))
            drive.CreateDampingAttr(float(damping))
            drive.CreateMaxForceAttr(float(max_force))
            drive.CreateTargetPositionAttr(0.0)
            primed += 1
    return {"joints_found": len(joints), "axes_primed": primed, "axes_already_driven": already}


def describe_joints(stage: Usd.Stage, root_path: str, limit: int = 3) -> list[dict]:
    """Type, applied schemas and drive/limit attributes of the first few joints, to see what the importer wrote."""
    out = []
    for prim in _joint_prims(stage, root_path)[:limit]:
        attrs = {a.GetName(): (a.Get() if a.HasAuthoredValue() else None) for a in prim.GetAttributes()
                 if "drive" in a.GetName().lower() or "limit" in a.GetName().lower()}
        out.append({"path": str(prim.GetPath()), "type": prim.GetTypeName(),
                    "schemas": [str(s) for s in prim.GetAppliedSchemas()], "attributes": {k: str(v) for k, v in attrs.items()}})
    return out


def prime_usd_file(usd_path: str, stiffness: float = 1.0, damping: float = 1.0, max_force: float = 1.0e3) -> dict:
    """Prime the drives of a USD file in place (under its default prim) and save it. Safe to repeat: driven axes are kept.

    Needed when the robot is spawned from the file by a scene that clones it: the simulator model is built while the scene is
    constructed, so there is no moment between the spawn and the build at which the open stage could still be primed.
    """
    stage = Usd.Stage.Open(str(usd_path))
    root = stage.GetDefaultPrim() if stage else None
    if root is None or not root.IsValid():
        raise ValueError(f"{usd_path} has no default prim")
    report = prime_usd_drives(stage, str(root.GetPath()), stiffness, damping, max_force)
    if report["axes_primed"] > 0:
        stage.GetRootLayer().Save()
    report["root_prim"] = str(root.GetPath())
    return report
