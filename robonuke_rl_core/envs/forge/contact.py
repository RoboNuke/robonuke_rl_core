"""`ForgeContactSensorWrapper`: per-axis in-contact flags from a real contact sensor.

Two pieces, like the recorder camera:

* :func:`install_contact_sensor` runs **before** ``gym.make`` and adds an Isaac Lab
  ``ContactSensor`` on the held asset, filtered against the fixed asset, so
  ``force_matrix_w`` (the per-pair contact force) is populated.
* :class:`ForgeContactSensorWrapper` reads it each step, rotates the force into the
  end-effector frame, thresholds each axis, publishes the flags on the step metric channel
  and — when asked — appends them to the observation.

When a controller with a selection block is configured, the wrapper also publishes
``infos["in_contact"]``: the same flags, restricted to the force-eligible axes **in
selection order**, which is the per-transition target the supervised selection loss learns
against. That alignment is why the controller config reaches this wrapper at all.

Reads from the env: ``scene.sensors``, ``fingertip_midpoint_quat``, ``cfg.observation_space``
/ ``cfg.state_space``, and ``_get_observations``, which it wraps.

The sensor's prim path needs care: ``activate_contact_sensors=True`` puts the PhysX contact
API on the asset's *rigid body*, which for the Factory and Forge assets is a child of the
articulation root and whose name is asset-specific. So the installer walks the cloned stage
for the first descendant that carries the API rather than hard-coding a name.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import torch

from .compat import require_forge_env

__all__ = [
    "install_contact_sensor",
    "ForgeContactSensorWrapper",
    "SENSOR_KEY",
    "SELECTION_FLAGS_KEY",
    "ENV_READS",
]

#: top-level ``infos`` key the selection-ordered contact flags are published under
SELECTION_FLAGS_KEY = "in_contact"

#: scene-sensor name the contact sensor is registered under
SENSOR_KEY = "peg_contact_sensor"

ENV_READS = (
    f"scene.sensors[{SENSOR_KEY!r}]",
    "fingertip_midpoint_quat",
    "cfg.observation_space/state_space",
    "_get_observations",
)


import re

#: ``/World/envs/env_0/...`` -> ``/World/envs/env_.*/...``
_ENV_INDEX = re.compile(r"/env_\d+/")


def resolve_contact_body(root_expr: str) -> str:
    """The prim under ``root_expr`` that actually carries the PhysX contact-report API.

    ``activate_contact_sensors=True`` puts that API on the asset's **rigid body**, which for
    the Factory and Forge held and fixed assets is a child of the articulation root with an
    asset-specific name (``factory_peg_8mm``, ``factory_gear_medium``, ...). A sensor pointed
    at the root finds no reporter and raises at init, so the already-cloned stage is walked
    for the first descendant that has the API and its path is rewritten back to the
    cross-env form. Falls back to the root, whose own error message is a good one.
    """
    import isaaclab.sim as sim_utils
    from pxr import PhysxSchema

    matches = sim_utils.find_matching_prims(root_expr)
    if not matches:
        return root_expr
    queue = list(matches[0:1])
    while queue:
        prim = queue.pop(0)
        if prim.HasAPI(PhysxSchema.PhysxContactReportAPI):
            return _ENV_INDEX.sub("/env_.*/", prim.GetPath().pathString)
        queue.extend(prim.GetChildren())
    return root_expr


def install_contact_sensor(task_name: str, task_cfg: Any, cfg: Any) -> None:
    """Add the held-vs-fixed contact sensor. Call **before** ``gym.make``.

    The sensor cannot simply be declared on the scene cfg: ``InteractiveScene`` builds its
    sensors when the scene is constructed, which is before the task has spawned the peg and
    the hole, and the prim that carries the PhysX contact API is an asset-specific *child* of
    those roots. So the sensor is constructed after ``clone_environments`` instead — the same
    one-shot shim the recorder camera uses — when the stage holds the assets and the child
    body can be found by walking it.
    """
    import importlib

    import gymnasium as gym

    require_forge_env(task_name, "install_contact_sensor", ENV_READS)
    if getattr(task_cfg.scene, "clone_in_fabric", False):
        raise ValueError(
            "wrappers.contact needs task.cfg.scene.clone_in_fabric: false — a contact reporter "
            "has to exist on real per-env prims, and under Fabric cloning the cloned bodies "
            "carry no PhysX contact API, so the sensor fails to initialize. Add it to the "
            "experiment:\n  task:\n    cfg:\n      scene:\n        clone_in_fabric: false"
        )

    entry_point = str(gym.spec(task_name).entry_point)
    module_name, _, class_name = entry_point.partition(":")
    env_class = getattr(importlib.import_module(module_name), class_name)
    original_setup_scene = env_class._setup_scene

    def patched_setup_scene(self):
        original_clone = self.scene.clone_environments

        def shim_clone(*args, **kwargs):
            self.scene.clone_environments = original_clone
            result = original_clone(*args, **kwargs)
            from isaaclab.sensors import ContactSensor, ContactSensorCfg

            held = resolve_contact_body(cfg.held_prim_expr)
            fixed = resolve_contact_body(cfg.fixed_prim_expr)
            sensor_cfg = ContactSensorCfg(
                prim_path=held,
                update_period=0.0,  # every physics step
                history_length=0,  # only the latest reading
                debug_vis=False,
                track_air_time=False,
                # without a filter there is no per-pair force matrix, only the net force
                filter_prim_paths_expr=[fixed],
            )
            self.scene._sensors[SENSOR_KEY] = ContactSensor(sensor_cfg)
            print(f"[contact] sensor on {held} filtered against {fixed}", flush=True)
            return result

        self.scene.clone_environments = shim_clone
        try:
            return original_setup_scene(self)
        finally:
            env_class._setup_scene = original_setup_scene

    env_class._setup_scene = patched_setup_scene


class ForgeContactSensorWrapper(gym.Wrapper):
    """Publish per-axis in-contact flags, and optionally append them to the observation."""

    #: one flag per task-space translation axis
    FLAGS = 3

    def __init__(self, env: Any, cfg: Any, task_name: str, controller_cfg: Any = None) -> None:
        require_forge_env(task_name, type(self).__name__, ENV_READS)
        super().__init__(env)
        unwrapped = env.unwrapped
        self.cfg = cfg
        self.device = unwrapped.device
        self.num_envs = int(unwrapped.num_envs)
        self._selection_axes = self._resolve_selection_axes(controller_cfg)

        sensors = getattr(getattr(unwrapped, "scene", None), "sensors", {}) or {}
        if SENSOR_KEY not in sensors:
            raise RuntimeError(
                f"{type(self).__name__}: no contact sensor at scene.sensors[{SENSOR_KEY!r}]. "
                "install_contact_sensor(...) must run before gym.make() when "
                "wrappers.contact.enabled is true."
            )
        self.sensor = sensors[SENSOR_KEY]

        # the flags live on the env so other wrappers (fragile's loss-of-contact) can read them
        unwrapped.in_contact = torch.zeros(
            (self.num_envs, self.FLAGS), dtype=torch.bool, device=self.device
        )
        self._appended = int(cfg.append_to_policy_obs) * self.FLAGS
        if cfg.append_to_policy_obs:
            unwrapped.cfg.observation_space += self.FLAGS
        if cfg.append_to_critic_state and getattr(unwrapped.cfg, "state_space", 0):
            unwrapped.cfg.state_space += self.FLAGS
        if cfg.append_to_policy_obs or cfg.append_to_critic_state:
            native_actions = unwrapped.actions
            unwrapped._configure_gym_env_spaces()
            unwrapped.actions = torch.zeros_like(native_actions)  # see the controller wrapper
            self.observation_space = unwrapped.observation_space

        self._original_get_observations = unwrapped._get_observations
        unwrapped._get_observations = self._get_observations

    # ------------------------------------------------------------------ the selection order
    def _resolve_selection_axes(self, controller_cfg: Any):
        """The force-eligible axes, in selection order, or None without a selection block.

        A rotation axis is not negotiable: the sensor gives a contact *force*, so there is
        one flag per translation axis and none for a torque. Publishing a tensor whose
        columns did not line up with the selection dims would mis-supervise the axis
        silently, so a force-eligible rotation axis raises here instead.
        """
        if controller_cfg is None or not getattr(controller_cfg, "enabled", False):
            return None
        from ..interface import AXES, ActionLayout

        layout = ActionLayout(controller_cfg)
        if not layout.selection_dim:
            return None
        beyond = [AXES[axis] for axis in layout.force_axes if axis >= self.FLAGS]
        if beyond:
            raise ValueError(
                f"wrappers.contact has one in-contact flag per translation axis "
                f"{AXES[: self.FLAGS]}, but controller.force_axes makes {beyond} "
                "force-eligible too, and a contact force says nothing about a torque axis. "
                f"Narrow controller.force_axes to the first {self.FLAGS} axes, or disable "
                "wrappers.contact."
            )
        return list(layout.force_axes)

    # ------------------------------------------------------------------ the flags
    def refresh(self) -> torch.Tensor:
        """Recompute the per-axis flags from the latest contact force."""
        from isaacsim.core.utils.torch import quat_rotate_inverse

        unwrapped = self.env.unwrapped
        matrix = self.sensor.data.force_matrix_w
        if matrix is None:
            raise RuntimeError(
                f"{type(self).__name__}: the contact sensor has no force_matrix_w; it needs "
                "filter_prim_paths_expr (install_contact_sensor sets it)"
            )
        # (num_envs, bodies, filters, 3) -> one force per env
        world_force = matrix.reshape(self.num_envs, -1, 3).sum(dim=1)
        local = quat_rotate_inverse(unwrapped.fingertip_midpoint_quat, world_force)
        unwrapped.in_contact = local.abs() > float(self.cfg.force_threshold)
        self._world_force = world_force
        return unwrapped.in_contact

    def _get_observations(self):
        observations = self._original_get_observations()
        flags = self.refresh().to(torch.float32)
        if self.cfg.append_to_policy_obs and isinstance(observations, dict):
            observations["policy"] = torch.cat((observations["policy"], flags), dim=-1)
        if (
            self.cfg.append_to_critic_state
            and isinstance(observations, dict)
            and "critic" in observations
        ):
            observations["critic"] = torch.cat((observations["critic"], flags), dim=-1)
        return observations

    def step(self, action):
        observations, rewards, terminated, truncated, infos = self.env.step(action)
        if isinstance(infos, dict):
            flags = self.env.unwrapped.in_contact.to(torch.float32)
            metrics = infos.setdefault("metrics_to_log", {})
            for index, axis in enumerate("xyz"):
                metrics[f"contact/in_contact_{axis}"] = flags[:, index]
            metrics["contact/in_contact_any"] = flags.amax(dim=1)
            if self._selection_axes is not None:
                # a top-level key, not a metric: the learner stores it per transition for
                # the supervised selection loss, which needs it aligned with the selection
                infos[SELECTION_FLAGS_KEY] = flags[:, self._selection_axes]
        return observations, rewards, terminated, truncated, infos
