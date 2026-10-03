"""MuJoCo kinematic replay of Newton run archives (Windows / hready-gmr, item 6C)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

REPLAY_LABEL = "kinematic replay of the Newton-simulated trajectory (not a Newton render)"


def _repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "configs" / "paths.example.yaml").is_file():
            return parent
    raise FileNotFoundError("repo root")


def _read_paths_yaml() -> dict[str, str]:
    paths = _repo_root() / "configs" / "paths.yaml"
    out: dict[str, str] = {}
    if paths.is_file():
        for line in paths.read_text(encoding="utf-8").splitlines():
            s = line.strip()
            if not s or s.startswith("#") or ":" not in s:
                continue
            key, val = s.split(":", 1)
            out[key.strip()] = val.strip()
    return out


def _gmr_xml() -> Path:
    cfg = _read_paths_yaml()
    data_root = Path(cfg.get("data_root", "D:/projects/hready_data"))
    gmr_root = Path(cfg.get("gmr_root", str(data_root.parent / "third_party" / "GMR")))
    xml = gmr_root / "assets" / "unitree_g1" / "g1_mocap_29dof.xml"
    if not xml.is_file():
        raise FileNotFoundError(xml)
    return xml


def render_replay_mp4(
    run_npz: Path,
    out_mp4: Path,
    *,
    fps: float | None = None,
    width: int = 1280,
    height: int = 720,
) -> dict[str, str]:
    import imageio.v2 as imageio
    import mujoco

    xml = _gmr_xml()
    data = np.load(run_npz, allow_pickle=True)
    q = np.asarray(data["q"], dtype=np.float64)
    q_ref = np.asarray(data["q_ref"], dtype=np.float64)
    root_pos = np.asarray(data["root_pos"], dtype=np.float64)
    root_quat = np.asarray(data["root_quat_xyzw"], dtype=np.float64)
    root_ref = np.asarray(data["root_pos_ref"], dtype=np.float64)
    quat_ref = np.asarray(data["root_quat_xyzw_ref"], dtype=np.float64)
    t = np.asarray(data["time"], dtype=np.float64)
    if fps is None:
        fps = float(data["fps"]) if "fps" in data else (1.0 / float(data["dt"]) if "dt" in data else 30.0)
        if t.size > 1:
            fps = max(1.0, (t.size - 1) / max(t[-1] - t[0], 1e-6))

    model = mujoco.MjModel.from_xml_path(str(xml))
    sim_data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=height, width=width)
    scene_option = mujoco.MjvOption()
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = True
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.distance = 2.8
    cam.azimuth = 90
    cam.elevation = -15
    cam.lookat[:] = np.array([0.0, 0.0, 0.85])

    out_mp4 = out_mp4.resolve()
    out_mp4.parent.mkdir(parents=True, exist_ok=True)

    def _apply(root_p: np.ndarray, root_q: np.ndarray, dof: np.ndarray) -> None:
        sim_data.qpos[:3] = root_p
        sim_data.qpos[3:7] = root_q
        nq = min(len(dof), model.nq - 7)
        sim_data.qpos[7 : 7 + nq] = dof[:nq]
        mujoco.mj_forward(model, sim_data)

    from PIL import Image, ImageDraw

    frames: list[np.ndarray] = []
    stride = max(1, round(len(q) / min(len(q), fps * 20)))
    for i in range(0, len(q), stride):
        _apply(root_pos[i], root_quat[i], q[i])
        renderer.update_scene(sim_data, camera=cam, scene_option=scene_option)
        img_sim = renderer.render()
        _apply(root_ref[i], quat_ref[i], q_ref[i])
        renderer.update_scene(sim_data, camera=cam, scene_option=scene_option)
        img_ref = renderer.render()
        panel = np.concatenate([img_ref, img_sim], axis=1)
        pil = Image.fromarray(panel)
        draw = ImageDraw.Draw(pil)
        draw.rectangle((0, 0, panel.shape[1], 28), fill=(0, 0, 0))
        draw.text((8, 6), REPLAY_LABEL, fill=(255, 255, 255))
        draw.text((8, panel.shape[0] - 22), "left=reference  right=sim", fill=(220, 220, 220))
        frames.append(np.asarray(pil))

    try:
        writer = imageio.get_writer(str(out_mp4), fps=min(fps, 30.0), codec="libx264", quality=8)
        for fr in frames:
            writer.append_data(fr)
        writer.close()
    except Exception as exc:
        raise RuntimeError(
            f"MP4 encode failed ({exc}). imageio/ffmpeg must be available in hready-gmr."
        ) from exc

    meta = {"label": REPLAY_LABEL, "run_npz": str(run_npz.resolve()), "mp4": str(out_mp4)}
    return meta


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="MuJoCo kinematic replay of a Newton run .npz")
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)
    meta = render_replay_mp4(args.run, args.out)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
