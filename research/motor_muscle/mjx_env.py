from __future__ import annotations

from typing import NamedTuple

import numpy as np

from .config import ExperimentConfig
from .model import joint_dofs, load_model
from .motors import DenseMotorArray


class JAXMotorState(NamedTuple):
    current: object
    temperature: object
    failed: object
    energy_j: object


class JAXMotorParameters(NamedTuple):
    group: object
    group_dof: object
    sign: object
    kt: object
    resistance0: object
    moment_arm: object
    counts: object


class MJXBatchStep(NamedTuple):
    observation: object
    reward: object
    terminated: object
    truncated: object
    info: dict[str, object]


def _require_jax():
    try:
        import jax
        import jax.numpy as jnp
        from mujoco import mjx
    except ImportError as exc:
        raise RuntimeError(
            "MJX backend is not installed; install the 'mjx' extra or "
            "research/requirements-rocm.txt"
        ) from exc
    return jax, jnp, mjx


def _tree_stack(jax, trees):
    return jax.tree_util.tree_map(lambda *values: jax.numpy.stack(values), *trees)


def _antagonists(jnp, signed):
    return jnp.stack((jnp.clip(signed, 0, 1), jnp.clip(-signed, 0, 1)), axis=-1).reshape(
        signed.shape[0], -1
    )


def jax_motor_step(
    parameters: JAXMotorParameters,
    state: JAXMotorState,
    group_command,
    allocation_bias,
    joint_velocity,
    dt: float,
):
    """Pure JAX dense electrical/thermal motor update for a batch of worlds."""
    jax, jnp, _ = _require_jax()
    group_count = group_command.shape[1]
    motor_count = state.current.shape[1]
    uniform_layout = motor_count % group_count == 0

    temp_derate = jnp.clip((95.0 - state.temperature) / 20.0, 0.0, 1.0)
    health = (~state.failed).astype(jnp.float32) * temp_derate
    if uniform_layout:
        motors_per_group = motor_count // group_count
        health_grouped = health.reshape(health.shape[0], group_count, motors_per_group)
        raw_grouped = health_grouped * jnp.exp(
            0.35 * allocation_bias[:, :, None]
        )
        weight_sum = jnp.sum(raw_grouped, axis=2, keepdims=True)
        normalized_grouped = (
            raw_grouped
            * parameters.counts[:, :, None]
            / jnp.maximum(1e-6, weight_sum)
        )
        voltage = (
            12.0
            * group_command[:, :, None]
            * normalized_grouped
            * health_grouped
        ).reshape(health.shape)
    else:
        raw_weight = health * jnp.exp(
            0.35
            * jnp.take_along_axis(allocation_bias, parameters.group, axis=1)
        )
        weight_sum = jax.vmap(
            lambda group, weight: jnp.bincount(
                group, weights=weight, length=group_count
            )
        )(parameters.group, raw_weight)
        normalized = (
            raw_weight
            * jnp.take_along_axis(parameters.counts, parameters.group, axis=1)
            / jnp.maximum(
                1e-6,
                jnp.take_along_axis(weight_sum, parameters.group, axis=1),
            )
        )
        voltage = (
            12.0
            * jnp.take_along_axis(group_command, parameters.group, axis=1)
            * normalized
            * health
        )
    resistance = parameters.resistance0 * (1 + 0.0039 * (state.temperature - 24.0))
    if uniform_layout:
        motors_per_group = motor_count // group_count
        motor_velocity = jnp.broadcast_to(
            jnp.repeat(joint_velocity, 2, axis=1)[:, :, None],
            (joint_velocity.shape[0], group_count, motors_per_group),
        ).reshape(joint_velocity.shape[0], motor_count)
    else:
        motor_velocity = jnp.take_along_axis(
            joint_velocity, parameters.group_dof, axis=1
        )
    back_emf = 0.08 * motor_velocity * parameters.sign
    target_current = jnp.clip((voltage - back_emf) / resistance, 0.0, 4.0)
    alpha = 1.0 - jnp.exp(-dt / 0.012)
    current = state.current + (target_current - state.current) * alpha
    current = jnp.where(state.failed, 0.0, current)
    motor_torque = parameters.kt * current * parameters.moment_arm * parameters.sign
    dof_count = joint_velocity.shape[1]
    if uniform_layout:
        grouped_torque = motor_torque.reshape(
            motor_torque.shape[0], group_count, motor_count // group_count
        ).sum(axis=2)
        generalized = grouped_torque.reshape(
            motor_torque.shape[0], dof_count, 2
        ).sum(axis=2)
    else:
        generalized = jax.vmap(
            lambda group_dof, torque: jnp.bincount(
                group_dof, weights=torque, length=dof_count
            )
        )(parameters.group_dof, motor_torque)
    joule = current * current * resistance
    cooling = 0.42 * (state.temperature - 24.0)
    temperature = state.temperature + (joule - cooling) * dt / 1.25
    energy = state.energy_j + jnp.sum(jnp.abs(voltage * current), axis=1) * dt
    next_state = JAXMotorState(current, temperature, state.failed, energy)
    diagnostics = {
        "peak_temperature_c": jnp.max(temperature, axis=1),
        "mean_temperature_c": jnp.mean(temperature, axis=1),
        "saturated_ratio": jnp.mean(target_current >= 3.999, axis=1),
    }
    return next_state, generalized, diagnostics


class MJXMotorMuscleEnv:
    """Batched accelerator backend with physics and motors resident on device."""

    def __init__(self, config: ExperimentConfig | None = None, batch_size: int = 64):
        self.config = config or ExperimentConfig()
        self.config.validate()
        if not 1 <= batch_size <= 256:
            raise ValueError("batch_size must be in [1, 256]")
        self.batch_size = batch_size
        self.jax, self.jnp, self.mjx = _require_jax()
        self.mj_model = load_model(self.config.physics_dt)
        self.mjx_model = self.mjx.put_model(self.mj_model)
        dofs = joint_dofs(self.mj_model)
        self.dof_names = [name for name, _ in dofs]
        self.dof_indices = np.asarray([index for _, index in dofs], dtype=np.int32)
        import mujoco

        self.qpos_indices = np.asarray(
            [
                int(
                    self.mj_model.jnt_qposadr[
                        mujoco.mj_name2id(
                            self.mj_model, mujoco.mjtObj.mjOBJ_JOINT, name
                        )
                    ]
                )
                for name in self.dof_names
            ],
            dtype=np.int32,
        )
        self.pelvis_body = mujoco.mj_name2id(
            self.mj_model, mujoco.mjtObj.mjOBJ_BODY, "pelvis"
        )
        self.floor_geom = mujoco.mj_name2id(
            self.mj_model, mujoco.mjtObj.mjOBJ_GEOM, "floor"
        )
        self.left_foot_geom = mujoco.mj_name2id(
            self.mj_model, mujoco.mjtObj.mjOBJ_GEOM, "left_foot_geom"
        )
        self.right_foot_geom = mujoco.mj_name2id(
            self.mj_model, mujoco.mjtObj.mjOBJ_GEOM, "right_foot_geom"
        )
        self.seeds = np.arange(batch_size, dtype=np.int64)
        self.data = None
        self.motor_parameters = None
        self.motor_state = None
        self.previous_action = None
        self.delay_buffer = None
        self.delay_cursor = None
        self.initial_height = None
        self._compiled_step = self.jax.jit(self._step_impl)
        self._compiled_rollout = self.jax.jit(self._rollout_impl)
        self._compiled_trace_rollout = self.jax.jit(self._trace_rollout_impl)

    @property
    def action_size(self) -> int:
        return len(self.dof_indices) * 3

    def _host_initialization(self, seeds: np.ndarray):
        parameters = []
        states = []
        data_rows = []
        for seed in seeds.tolist():
            motors = DenseMotorArray(
                self.config.motor_count,
                self.dof_indices,
                int(seed),
                self.config.parameter_variation,
            )
            motors.reset(
                int(seed) + 101,
                self.config.failure_ratio,
                self.config.concentrated_failure,
            )
            parameters.append(
                JAXMotorParameters(
                    motors.group,
                    motors.group_dof,
                    motors.sign,
                    motors.kt,
                    motors.resistance0,
                    motors.moment_arm,
                    motors._counts,
                )
            )
            states.append(
                JAXMotorState(
                    motors.current,
                    motors.temperature,
                    motors.failed,
                    np.float32(0),
                )
            )
            row = self.mjx.make_data(self.mj_model)
            rng = np.random.default_rng(int(seed))
            qpos = np.asarray(row.qpos).copy()
            qpos[2] = 0.95
            qpos[3] = 1.0
            qpos[4:7] = 0.0
            qpos[self.qpos_indices] = rng.normal(0, 0.003, len(self.qpos_indices))
            row = row.replace(qpos=self.jnp.asarray(qpos))
            data_rows.append(row)
        return (
            _tree_stack(self.jax, parameters),
            _tree_stack(self.jax, states),
            _tree_stack(self.jax, data_rows),
        )

    def reset_batch(
        self,
        seeds: np.ndarray | list[int] | None = None,
        config: ExperimentConfig | None = None,
    ):
        if config is not None:
            self.__init__(config, self.batch_size)
        seeds = np.arange(self.batch_size) if seeds is None else np.asarray(seeds)
        if seeds.shape != (self.batch_size,):
            raise ValueError(f"expected {self.batch_size} seeds")
        self.seeds = seeds.astype(np.int64)
        self.motor_parameters, self.motor_state, self.data = self._host_initialization(self.seeds)
        self.data = self.jax.vmap(self.mjx.forward, in_axes=(None, 0))(self.mjx_model, self.data)
        self.previous_action = self.jnp.zeros(
            (self.batch_size, len(self.dof_indices)), dtype=self.jnp.float32
        )
        delay_size = max(1, self.config.control_delay_steps + 1)
        self.delay_buffer = self.jnp.zeros(
            (delay_size, self.batch_size, self.action_size), dtype=self.jnp.float32
        )
        self.delay_cursor = self.jnp.asarray(0, dtype=self.jnp.int32)
        self.initial_height = self.data.xpos[:, self.pelvis_body, 2]
        return self._observation(
            self.data, self.motor_parameters, self.motor_state
        ), self._info(self.data, self.motor_state, {})

    def reset(self, seed: int | None = None, config: ExperimentConfig | None = None):
        if self.batch_size != 1:
            raise ValueError("reset is only available when batch_size=1; use reset_batch")
        obs, info = self.reset_batch(
            [self.config.seed if seed is None else seed], config
        )
        return obs[0], {key: value[0] for key, value in info.items()}

    def _group_summary(self, parameters, state):
        group_count = len(self.dof_indices) * 2
        motor_count = state.temperature.shape[1]
        if motor_count % group_count == 0:
            grouped_shape = (
                state.temperature.shape[0],
                group_count,
                motor_count // group_count,
            )
            temp = state.temperature.reshape(grouped_shape).mean(axis=2)
            available = (~state.failed).reshape(grouped_shape).mean(axis=2)
            return temp, available.astype(self.jnp.float32)
        temp = self.jax.vmap(
            lambda group, value: self.jnp.bincount(
                group, weights=value, length=group_count
            )
        )(parameters.group, state.temperature)
        available = self.jax.vmap(
            lambda group, value: self.jnp.bincount(
                group, weights=value, length=group_count
            )
        )(parameters.group, (~state.failed).astype(self.jnp.float32))
        return temp / parameters.counts, available / parameters.counts

    def _observation(self, data, parameters, motor_state):
        temperature, availability = self._group_summary(parameters, motor_state)
        quat = data.qpos[:, 3:7]
        w, x, y, z = [quat[:, index] for index in range(4)]
        roll = self.jnp.arctan2(
            2 * (w * x + y * z), 1 - 2 * (x * x + y * y)
        )
        pitch = self.jnp.arcsin(self.jnp.clip(2 * (w * y - z * x), -1, 1))
        contact = data._impl.contact
        active = contact.dist <= 0
        floor_pair = (contact.geom1 == self.floor_geom) | (
            contact.geom2 == self.floor_geom
        )
        left_pair = (contact.geom1 == self.left_foot_geom) | (
            contact.geom2 == self.left_foot_geom
        )
        right_pair = (contact.geom1 == self.right_foot_geom) | (
            contact.geom2 == self.right_foot_geom
        )
        foot_contact = self.jnp.stack(
            (
                self.jnp.any(active & floor_pair & left_pair, axis=1),
                self.jnp.any(active & floor_pair & right_pair, axis=1),
            ),
            axis=1,
        )
        parts = (
            data.qpos[:, self.qpos_indices],
            data.qvel[:, self.dof_indices],
            self.jnp.stack((roll, pitch), axis=1),
            data.qpos[:, :2],
            data.qvel[:, :3],
            data.qvel[:, 3:6],
            foot_contact.astype(self.jnp.float32),
            (temperature - 24.0) / 60.0,
            availability,
        )
        return self.jnp.concatenate(parts, axis=1).astype(self.jnp.float32)

    def _physics_substep(self, carry, _):
        data, parameters, motor_state, group_command, allocation = carry
        joint_velocity = data.qvel[:, self.dof_indices]
        motor_state, torque, diagnostics = jax_motor_step(
            parameters,
            motor_state,
            group_command,
            allocation,
            joint_velocity,
            self.config.physics_dt,
        )
        qfrc = self.jnp.zeros_like(data.qfrc_applied)
        qfrc = qfrc.at[:, self.dof_indices].set(torque)
        data = data.replace(qfrc_applied=qfrc)
        data = self.jax.vmap(self.mjx.step, in_axes=(None, 0))(self.mjx_model, data)
        return (data, parameters, motor_state, group_command, allocation), diagnostics

    def _step_impl(
        self,
        data,
        motor_parameters,
        motor_state,
        previous_action,
        delay_buffer,
        delay_cursor,
        action,
        initial_height,
    ):
        action = self.jnp.clip(action, -1, 1)
        delay_buffer = delay_buffer.at[delay_cursor].set(action)
        read_cursor = (
            delay_cursor + 1
        ) % delay_buffer.shape[0] if self.config.control_delay_steps else delay_cursor
        delayed_action = delay_buffer[read_cursor]
        delay_cursor = (delay_cursor + 1) % delay_buffer.shape[0]
        n = len(self.dof_indices)
        signed = delayed_action[:, :n]
        allocation = delayed_action[:, n:]
        group_command = _antagonists(self.jnp, signed)
        action_delta = self.jnp.mean((signed - previous_action) ** 2, axis=1)
        carry = (data, motor_parameters, motor_state, group_command, allocation)
        carry, diagnostics = self.jax.lax.scan(
            self._physics_substep, carry, None, length=self.config.control_stride
        )
        data, _, motor_state, _, _ = carry
        observation = self._observation(data, motor_parameters, motor_state)
        quat = data.qpos[:, 3:7]
        w, x, y, z = [quat[:, index] for index in range(4)]
        roll = self.jnp.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
        pitch = self.jnp.arcsin(self.jnp.clip(2 * (w * y - z * x), -1, 1))
        tilt = self.jnp.sqrt(roll * roll + pitch * pitch)
        height = data.xpos[:, self.pelvis_body, 2]
        terminated = (height < initial_height * 0.75) | (
            self.jnp.maximum(self.jnp.abs(roll), self.jnp.abs(pitch))
            > self.jnp.deg2rad(45)
        )
        truncated = data.time >= self.config.duration_s
        reward = 1.0 - 1.4 * tilt * tilt - 0.02 * action_delta - 10.0 * terminated
        info = {
            "time": data.time,
            "pelvis_height": height,
            "torso_tilt_rad": tilt,
            "energy_j": motor_state.energy_j,
            "peak_temperature_c": diagnostics["peak_temperature_c"][-1],
            "mean_temperature_c": diagnostics["mean_temperature_c"][-1],
            "saturated_ratio": diagnostics["saturated_ratio"][-1],
            "action_delta": action_delta,
        }
        return (
            data,
            motor_state,
            signed,
            delay_buffer,
            delay_cursor,
            MJXBatchStep(observation, reward, terminated, truncated, info),
        )

    def step_batch(self, actions):
        actions = self.jnp.asarray(actions, dtype=self.jnp.float32)
        if actions.shape != (self.batch_size, self.action_size):
            raise ValueError(
                f"expected action shape {(self.batch_size, self.action_size)}, got {actions.shape}"
            )
        (
            self.data,
            self.motor_state,
            self.previous_action,
            self.delay_buffer,
            self.delay_cursor,
            result,
        ) = self._compiled_step(
            self.data,
            self.motor_parameters,
            self.motor_state,
            self.previous_action,
            self.delay_buffer,
            self.delay_cursor,
            actions,
            self.initial_height,
        )
        return (
            result.observation,
            result.reward,
            result.terminated,
            result.truncated,
            result.info,
        )

    def _rollout_impl(
        self,
        data,
        motor_parameters,
        motor_state,
        previous_action,
        delay_buffer,
        delay_cursor,
        actions,
        initial_height,
    ):
        def control_step(carry, action):
            data, motor_state, previous_action, delay_buffer, delay_cursor = carry
            outputs = self._step_impl(
                data,
                motor_parameters,
                motor_state,
                previous_action,
                delay_buffer,
                delay_cursor,
                action,
                initial_height,
            )
            return outputs[:5], outputs[5]

        initial = (
            data,
            motor_state,
            previous_action,
            delay_buffer,
            delay_cursor,
        )
        final, results = self.jax.lax.scan(control_step, initial, actions)
        return (*final, results)

    def rollout_batch(self, actions):
        """Run [time, batch, action] fully on device with one JIT boundary."""
        actions = self.jnp.asarray(actions, dtype=self.jnp.float32)
        if actions.ndim != 3 or actions.shape[1:] != (
            self.batch_size,
            self.action_size,
        ):
            raise ValueError(
                "expected rollout actions shaped "
                f"[time, {self.batch_size}, {self.action_size}], got {actions.shape}"
            )
        (
            self.data,
            self.motor_state,
            self.previous_action,
            self.delay_buffer,
            self.delay_cursor,
            results,
        ) = self._compiled_rollout(
            self.data,
            self.motor_parameters,
            self.motor_state,
            self.previous_action,
            self.delay_buffer,
            self.delay_cursor,
            actions,
            self.initial_height,
        )
        return (
            results.observation,
            results.reward,
            results.terminated,
            results.truncated,
            results.info,
        )

    def _trace_rollout_impl(
        self,
        data,
        motor_parameters,
        motor_state,
        previous_action,
        delay_buffer,
        delay_cursor,
        actions,
        initial_height,
    ):
        def control_step(carry, action):
            data, motor_state, previous_action, delay_buffer, delay_cursor = carry
            outputs = self._step_impl(
                data,
                motor_parameters,
                motor_state,
                previous_action,
                delay_buffer,
                delay_cursor,
                action,
                initial_height,
            )
            next_carry = outputs[:5]
            next_data = next_carry[0]
            trace = (
                next_data.qpos,
                next_data.qvel,
                outputs[5].terminated,
                outputs[5].truncated,
                outputs[5].info,
            )
            return next_carry, trace

        initial = (
            data,
            motor_state,
            previous_action,
            delay_buffer,
            delay_cursor,
        )
        final, trace = self.jax.lax.scan(control_step, initial, actions)
        return (*final, trace)

    def rollout_trace_batch(self, actions):
        """Run on device and return qpos/qvel traces at control-rate boundaries."""
        actions = self.jnp.asarray(actions, dtype=self.jnp.float32)
        if actions.ndim != 3 or actions.shape[1:] != (
            self.batch_size,
            self.action_size,
        ):
            raise ValueError(
                "expected rollout actions shaped "
                f"[time, {self.batch_size}, {self.action_size}], got {actions.shape}"
            )
        (
            self.data,
            self.motor_state,
            self.previous_action,
            self.delay_buffer,
            self.delay_cursor,
            trace,
        ) = self._compiled_trace_rollout(
            self.data,
            self.motor_parameters,
            self.motor_state,
            self.previous_action,
            self.delay_buffer,
            self.delay_cursor,
            actions,
            self.initial_height,
        )
        qpos, qvel, terminated, truncated, info = trace
        return {
            "qpos": qpos,
            "qvel": qvel,
            "terminated": terminated,
            "truncated": truncated,
            "info": info,
        }

    def step(self, action):
        if self.batch_size != 1:
            raise ValueError("step is only available when batch_size=1; use step_batch")
        outputs = self.step_batch(self.jnp.asarray(action)[None, :])
        return tuple(
            {key: value[0] for key, value in item.items()}
            if isinstance(item, dict)
            else item[0]
            for item in outputs
        )

    def _info(self, data, motor_state, diagnostics):
        return {
            "time": data.time,
            "pelvis_height": data.xpos[:, self.pelvis_body, 2],
            "energy_j": motor_state.energy_j,
        }
