from __future__ import annotations

from pathlib import Path

import mujoco


MODEL_PATH = Path(__file__).with_name("humanoid.xml")


def load_model(timestep: float = 0.001) -> mujoco.MjModel:
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    model.opt.timestep = timestep
    return model


def joint_dofs(model: mujoco.MjModel) -> list[tuple[str, int]]:
    result: list[tuple[str, int]] = []
    for joint_id in range(model.njnt):
        if model.jnt_type[joint_id] != mujoco.mjtJoint.mjJNT_HINGE:
            continue
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        if name:
            result.append((name, int(model.jnt_dofadr[joint_id])))
    return result

