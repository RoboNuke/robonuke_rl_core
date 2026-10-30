"""PARTS-style per-phase residuals (``sac.route_labels``): one complete SAC per listed label, routed per env.

Su et al., "From Pretraining to Proficiency" (arXiv:2609.21788): each bottleneck subtask of a long-horizon task gets its
own residual policy, its own replay buffer and its own local reward. Here a "phase" is a ``sac.std_label_names`` label
the env publishes for every observation (``sac.std_by_label``, e.g. one_leg's ``eval_state/phase/id``), and
``sac.route_labels`` lists the phases that get a residual. Each is an ordinary :class:`~robonuke_rl_core.learners.sac.SAC`
with every setting of the ``sac`` section and ``route_label`` set (its own actor, critics, targets, optimizers, replay
memory; see :meth:`SAC._route_signals` for its local reward and attempt ends), so every SAC option (n-step, PARTS'
terms, success-reweighted retraining, demos) applies to each residual separately.

:class:`RoutedSAC` is what the trainer drives: it hands every env step to every residual (each stores only the
transitions starting in its own phase), and for each env takes the action of the residual of the phase its observation
is in, or zeros (the base alone) in a phase without one. The phase of the observation an action is chosen for is the
one the env published with it (the previous step's ``infos``); before any, every env runs the base alone.

Logging: the first residual forwards the env's and the episodes' metrics (``episode/*``, ``phase/*``, ``residual/*``,
``reward/*``), every residual its own learner metrics under ``<phase>/`` (``insert/loss/critic``,
``insert/policy/std``, ...). Checkpoints: each residual writes its own, under ``<run dir>/<phase>/checkpoints``.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import torch

__all__ = ["RoutedSAC", "ENV_METRIC_PREFIXES", "route_hook"]

#: metric names that describe the env and its episodes, not a learner: forwarded once (by the first residual), as is
ENV_METRIC_PREFIXES = ("episode/", "phase/", "residual/", "reward/")


def route_hook(hook: Any, route: str, first: bool):
    """An ``on_log`` hook for one residual: env and episode metrics pass through from the first residual only, its
    learner metrics go out as ``<route>/<name>``."""

    def routed(agent_idx: int, metrics: Dict[str, Any], step: int) -> None:
        out = {}
        for name, value in metrics.items():
            if name.startswith(ENV_METRIC_PREFIXES):
                if first:
                    out[name] = value
            else:
                out[f"{route}/{name}"] = value
        if out:
            hook(agent_idx, out, step)

    return routed


class RoutedSAC:
    """Routes every env step to one SAC per phase (see the module docstring). ``learners`` maps each route label to
    its SAC, in ``sac.route_labels`` order; ``names`` is ``sac.std_label_names``."""

    def __init__(self, learners: Dict[str, Any], names: Sequence[str]) -> None:
        if not learners:
            raise ValueError("RoutedSAC needs at least one routed learner")
        self.learners = dict(learners)
        self.names = list(names)
        self.routes = list(self.learners)
        self.route_index = torch.tensor([self.names.index(route) for route in self.routes])
        first = next(iter(self.learners.values()))
        self.device = first.device
        self.num_envs = first.num_envs
        self.training = False
        self.env_reset_requested = False
        #: each env's label at the observation the next action is chosen for (None before the env published one)
        self._labels: Optional[torch.Tensor] = None

    # ------------------------------------------------------------------ the trainer's interface
    def init(self, *, trainer_cfg: Optional[Dict[str, Any]] = None) -> None:
        for learner in self.learners.values():
            learner.init(trainer_cfg=trainer_cfg)

    def enable_training_mode(self, enabled: bool = True, **kwargs) -> None:
        self.training = enabled
        for learner in self.learners.values():
            learner.enable_training_mode(enabled, **kwargs)

    def enable_models_training_mode(self, enabled: bool = True) -> None:
        for learner in self.learners.values():
            learner.enable_models_training_mode(enabled)

    def track_data(self, tag: str, value: float) -> None:
        pass  # metrics leave through the residuals' on_log hooks

    def pre_interaction(self, *, timestep: int, timesteps: int) -> None:
        for learner in self.learners.values():
            learner.pre_interaction(timestep=timestep, timesteps=timesteps)

    def act(self, observations, states, *, timestep: int, timesteps: int):
        """Each env's action from the residual of its observation's phase; zeros in a phase without one."""
        actions = None
        outputs: Dict[str, Any] = {}
        for route, learner in self.learners.items():
            routed, out = learner.act(observations, states, timestep=timestep, timesteps=timesteps)
            if actions is None:
                actions = torch.zeros_like(routed)
            if self._labels is not None:
                mask = (self._labels == self.names.index(route)).reshape(-1, 1).to(routed.dtype)
                actions = actions + mask * routed
            outputs[route] = out
        return actions, outputs

    def record_transition(self, *, infos, **kwargs) -> None:
        for learner in self.learners.values():
            learner.record_transition(infos=infos, **kwargs)
        self._labels = self._labels_from(infos)

    def post_interaction(self, *, timestep: int, timesteps: int) -> None:
        for learner in self.learners.values():
            learner.post_interaction(timestep=timestep, timesteps=timesteps)

    def forced_env_reset(self) -> None:
        self.env_reset_requested = False
        self._labels = None
        for learner in self.learners.values():
            learner.forced_env_reset()

    # ------------------------------------------------------------------ helpers
    def _labels_from(self, infos: Any) -> Optional[torch.Tensor]:
        """The label channel every residual reads (``sac.std_by_label``), from the first residual's parser."""
        first = next(iter(self.learners.values()))
        return first._std_labels_from(infos)

    def route_of(self, labels: torch.Tensor) -> List[Optional[str]]:
        """Per env, the route its label selects (None: no residual)."""
        out = []
        for value in labels.reshape(-1).tolist():
            name = self.names[value] if 0 <= value < len(self.names) else None
            out.append(name if name in self.learners else None)
        return out
