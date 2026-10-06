"""Composing an env: what runs before `gym.make`, and what wraps the env after.

Two entry points, both called by every script so that what you train, evaluate and watch is
the same stack:

* :func:`prepare_task` edits the task cfg **before** ``gym.make`` — the contact sensor and the
  orientation representation both have to exist before the env sizes its spaces.
* :func:`build_env` wraps the env **after** ``gym.make`` and before skrl's wrapper, so models
  are built from the wrapped spaces.
"""

from __future__ import annotations

from typing import Any, List

__all__ = ["WRAPPER_ORDER", "prepare_task", "build_env", "describe", "rewrite_orientation_order"]

#: The order wrappers are applied in, innermost first. Each edge is deliberate:
#:
#: 1. ``controller`` is innermost because it replaces the env's torque law and widens the
#:    action space; everything outside it sees the final action vector.
#: 2. ``efficient_reset`` wraps ``_reset_idx``, which must still be the env's own reset
#:    chain — so it goes on before anything that reacts to a reset.
#: 3. ``fragile`` wraps ``_get_dones`` and terminates episodes, so it sits above the reset
#:    wrapper whose partial path its terminations trigger.
#: 4. ``contact`` is outermost because it is the one that appends to the observation: it must
#:    wrap after anything else that edits obs, or its flags would not be last.
#: 5. ``task_metrics`` reads what the env did and publishes it per agent; it changes no
#:    behaviour at all, so it goes outermost where it sees the finished step.
WRAPPER_ORDER = ("controller", "efficient_reset", "fragile", "contact", "task_metrics")


def prepare_task(cfg: Any, task_name: str, task_cfg: Any) -> None:
    """Edit the task cfg in place for whatever the config enables. Call before ``gym.make``."""
    wrappers = getattr(cfg, "wrappers", None)
    if wrappers is None:
        return

    if wrappers.contact.enabled:
        from .forge.contact import install_contact_sensor

        install_contact_sensor(task_name, task_cfg, wrappers.contact)

    if wrappers.orientation.mode != "quat":
        apply_orientation_mode(task_name, task_cfg, wrappers.orientation.mode)


def rewrite_orientation_order(task_cfg: Any, obs_dims: dict, state_dims: dict) -> List[str]:
    """Swap every ``*_quat`` channel for its 6-D form, registering the new dims.

    Pure bookkeeping on dicts and lists, so it is testable without Isaac Lab. Returns the
    quaternion channels it replaced.
    """
    from .orientation import ROT6D_DIM, rot6d_suffix

    channels = [
        name
        for name in (*getattr(task_cfg, "obs_order", ()), *getattr(task_cfg, "state_order", ()))
        if name.endswith("_quat")
    ]
    for name in channels:
        obs_dims[rot6d_suffix(name)] = ROT6D_DIM
        state_dims[rot6d_suffix(name)] = ROT6D_DIM

    def swap(order):
        return [rot6d_suffix(name) if name.endswith("_quat") else name for name in order]

    if hasattr(task_cfg, "obs_order"):
        task_cfg.obs_order = swap(task_cfg.obs_order)
    if hasattr(task_cfg, "state_order"):
        task_cfg.state_order = swap(task_cfg.state_order)
    return channels


def apply_orientation_mode(task_name: str, task_cfg: Any, mode: str) -> None:
    """Swap the quaternion channels in ``obs_order`` / ``state_order`` for their 6-D form.

    Forge-family envs size their observation from those orders
    (``observation_space = sum(OBS_DIM_CFG[k] for k in obs_order)``), so the new channels have
    to be registered in the shared dim dicts first, and the env has to publish them — which is
    why this runs before ``gym.make`` and patches the env class's observation assembly rather
    than wrapping it.
    """
    from isaaclab_tasks.direct.factory import factory_env_cfg

    from .forge.compat import require_forge_env

    require_forge_env(task_name, "wrappers.orientation.mode", ("obs_order", "state_order"))

    quat_channels = rewrite_orientation_order(
        task_cfg, factory_env_cfg.OBS_DIM_CFG, factory_env_cfg.STATE_DIM_CFG
    )
    _publish_rot6d(quat_channels)
    print(f"[orientation] {mode}: {sorted(set(quat_channels))} -> *_rot6d", flush=True)


def _publish_rot6d(quat_channels: List[str]) -> None:
    """Make every quaternion channel publish its 6-D counterpart beside itself.

    Both Factory and Forge assemble their observation as a dict and then flatten it with
    ``factory_utils.collapse_obs_dict(obs_dict, order)``, so augmenting the dict inside that
    one function covers both envs and every channel, without copying either env's observation
    method into this package (which is what the port this grew from does — and what would
    then have to be re-copied at every Isaac Lab release).
    """
    from isaaclab_tasks.direct.factory import factory_utils

    from .orientation import quat_to_rot6d, rot6d_suffix

    if getattr(factory_utils, "_rot6d_installed", False):
        return
    original = factory_utils.collapse_obs_dict

    def collapse(obs_dict, obs_order):
        for name in [key for key in obs_dict if key.endswith("_quat")]:
            obs_dict.setdefault(rot6d_suffix(name), quat_to_rot6d(obs_dict[name]))
        return original(obs_dict, obs_order)

    factory_utils.collapse_obs_dict = collapse
    factory_utils._rot6d_installed = True


def build_env(cfg: Any, env: Any, task_name: str) -> Any:
    """Apply the wrappers the config enables, innermost first. Unchanged when none are."""
    controller = getattr(cfg, "controller", None)
    wrappers = getattr(cfg, "wrappers", None)

    for name in WRAPPER_ORDER:
        if name == "controller" and controller is not None and controller.enabled:
            from .forge.controller import ForgeControllerWrapper

            env = ForgeControllerWrapper(env, controller, task_name)
        elif wrappers is None:
            continue
        elif name == "efficient_reset" and wrappers.efficient_reset.enabled:
            from .forge.reset import ForgeEfficientResetWrapper

            env = ForgeEfficientResetWrapper(env, wrappers.efficient_reset, task_name)
        elif name == "fragile" and wrappers.fragile.enabled:
            from .forge.fragile import ForgeFragileObjectWrapper

            env = ForgeFragileObjectWrapper(env, wrappers.fragile, task_name)
        elif name == "contact" and wrappers.contact.enabled:
            from .forge.contact import ForgeContactSensorWrapper

            # the controller decides which axes the selection covers, and the flags are
            # published in that order for the supervised selection loss
            env = ForgeContactSensorWrapper(env, wrappers.contact, task_name, controller)
        elif name == "task_metrics" and wrappers.task_metrics.enabled:
            from .forge.compat import is_forge_task
            from .forge.metrics import ForgeTaskMetricsWrapper

            # on by default, so a non-Forge task skips it rather than failing the run; it
            # reads Forge's own metric hooks and has nothing to read elsewhere
            if is_forge_task(task_name):
                env = ForgeTaskMetricsWrapper(env, wrappers.task_metrics, task_name)
            else:
                print(
                    f"[envs] task metrics skipped: {task_name} is not a Forge-family task",
                    flush=True,
                )
    return env


def describe(cfg: Any) -> List[str]:
    """Names of the wrappers this config enables, in the order they are applied."""
    controller = getattr(cfg, "controller", None)
    wrappers = getattr(cfg, "wrappers", None)
    enabled = []
    for name in WRAPPER_ORDER:
        if name == "controller":
            if controller is not None and controller.enabled:
                enabled.append(name)
        elif wrappers is not None and getattr(wrappers, name).enabled:
            enabled.append(name)
    return enabled
