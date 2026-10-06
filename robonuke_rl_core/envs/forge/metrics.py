"""`ForgeTaskMetricsWrapper`: what the task itself did, per agent.

Reads from the env: ``_log_factory_metrics`` and ``_log_forge_metrics`` (both of which it
wraps — they are handed exactly the per-env quantities this needs), ``ep_succeeded``,
``ep_success_times``, ``episode_length_buf``, ``first_pred_success_tx`` and
``cfg_task.success_threshold`` / ``engage_threshold``.

**Why a wrapper and not the env's own numbers.** Factory and Forge already log success rates,
reward terms and prediction quality — but into ``extras`` as scalars *already averaged over
every env*. With several agents training side by side on disjoint env blocks, that average
mixes them, and a mixed metric cannot tell you that agent 1 has stopped learning. So this
reads the same quantities one step upstream, while they are still per env, and publishes
them through the two metric channels, which slice per agent.

What it publishes, all through the episode channel unless noted:

* ``episode/success`` — did this episode ever succeed (the rate, once averaged)
* ``episode/success_step`` — the step it first succeeded on; **NaN** for episodes that never
  did, so the mean is over successful episodes only
* ``episode/engaged`` — engagement by the task's own looser threshold
* ``termination/<cause>`` — one 0/1 per cause, so each becomes a rate
* ``reward/<term>`` — the per-term reward decomposition, on the step channel
* ``Success_Prediction/<metric>`` — how good Forge's success-prediction action is
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import gymnasium as gym
import torch

from .compat import require_forge_env

__all__ = ["ForgeTaskMetricsWrapper", "ENV_READS", "SUCCESS_PREDICTION"]

ENV_READS = (
    "_log_factory_metrics / _log_forge_metrics",
    "ep_succeeded",
    "ep_success_times",
    "episode_length_buf",
    "first_pred_success_tx (Forge)",
    "cfg_task.success_threshold/engage_threshold",
)

#: the prefix Forge's success-prediction metrics are published under
SUCCESS_PREDICTION = "Success_Prediction"


class ForgeTaskMetricsWrapper(gym.Wrapper):
    """Per-agent task outcomes, reward decomposition and success-prediction quality."""

    def __init__(self, env: Any, cfg: Any, task_name: str) -> None:
        require_forge_env(task_name, type(self).__name__, ENV_READS)
        super().__init__(env)
        unwrapped = env.unwrapped
        self.cfg = cfg
        self.device = unwrapped.device
        self.num_envs = int(unwrapped.num_envs)

        self._terms: Dict[str, torch.Tensor] = {}
        self._unloggable: set = set()
        self._successes: Optional[torch.Tensor] = None
        self._prediction: Optional[torch.Tensor] = None
        #: "did this episode ever engage", latched like the env's own ``ep_succeeded``.
        #: Engagement is the only quantity here read from live scene state rather than from
        #: a buffer, so it is sampled in the ``_log_factory_metrics`` tap (inside
        #: ``_get_rewards``, before ``_reset_idx``) and accumulated, never read at the end of
        #: a step. See :meth:`engaged_now`.
        self._engaged = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        threshold = getattr(getattr(unwrapped, "cfg_task", None), "engage_threshold", None)
        self._engage_threshold = (
            float(threshold)
            if threshold is not None and hasattr(unwrapped, "_get_curr_successes")
            else None
        )

        self._original_factory_log = getattr(unwrapped, "_log_factory_metrics", None)
        if self._original_factory_log is None:
            raise RuntimeError(
                f"{type(self).__name__}: the env has no _log_factory_metrics, which is where "
                "the per-env reward terms and successes are still per env"
            )
        unwrapped._log_factory_metrics = self._log_factory_metrics
        self._original_forge_log = getattr(unwrapped, "_log_forge_metrics", None)
        if self._original_forge_log is not None:
            unwrapped._log_forge_metrics = self._log_forge_metrics

    # ------------------------------------------------------------------ the taps
    def _log_factory_metrics(self, rew_dict, curr_successes):
        """Capture the per-env reward terms and successes, then let the env log its own.

        Also the one safe moment to sample engagement: this runs inside ``_get_rewards``,
        which is before ``DirectRLEnv.step`` calls ``_reset_idx``. A step later the scene is
        already back at its initial condition for every env that finished.
        """
        self._terms.update({name: value.detach() for name, value in rew_dict.items()})
        self._successes = curr_successes.detach()
        engaged = self.engaged_now()
        if engaged is not None:
            self._engaged |= engaged.reshape(-1).to(torch.bool)
        return self._original_factory_log(rew_dict, curr_successes)

    def _log_forge_metrics(self, rew_dict, policy_success_pred):
        """Same for Forge's extra terms and its success-prediction action."""
        self._terms.update({name: value.detach() for name, value in rew_dict.items()})
        self._prediction = policy_success_pred.detach()
        return self._original_forge_log(rew_dict, policy_success_pred)

    # ------------------------------------------------------------------ the metrics
    def engaged_now(self) -> Optional[torch.Tensor]:
        """Is the peg engaged **right now**, by the task's looser threshold?

        Instantaneous, so it is only meaningful while the scene still holds the state it is
        asked about. None when the task defines no ``engage_threshold``.
        """
        if self._engage_threshold is None:
            return None
        return self.env.unwrapped._get_curr_successes(
            success_threshold=self._engage_threshold, check_rot=False
        )

    def step_metrics(self, dtype: torch.dtype) -> Dict[str, torch.Tensor]:
        """Per-step, per-env: the reward decomposition.

        A term is normally one value per env, but a task is free to publish one that is
        already reduced (a scalar) or that carries trailing dimensions. Both are useful to
        log, so a scalar is broadcast to every env and trailing dimensions are averaged; a
        term that is neither is named once and skipped rather than crashing a training run
        over a logging detail.
        """
        out: Dict[str, torch.Tensor] = {}
        for name, value in self._terms.items():
            flat = value.reshape(value.shape[0], -1) if value.dim() > 1 else value.reshape(-1)
            if flat.dim() > 1 and flat.shape[0] == self.num_envs:
                out[f"reward/{name}"] = flat.mean(dim=1).to(dtype)
            elif flat.numel() == self.num_envs:
                out[f"reward/{name}"] = flat.to(dtype)
            elif flat.numel() == 1:
                # the task reduced it already: the same number for every env
                out[f"reward/{name}"] = flat.reshape(()).expand(self.num_envs).to(dtype)
            elif name not in self._unloggable:
                self._unloggable.add(name)
                print(
                    f"[task-metrics] reward term {name!r} is {tuple(value.shape)}, neither "
                    f"per-env ({self.num_envs}) nor scalar: not logged",
                    flush=True,
                )
        return out

    def episode_metrics(self, terminated, truncated, dtype: torch.dtype) -> Dict[str, torch.Tensor]:
        """Per-episode, per-env: outcome, timing, cause and prediction quality."""
        unwrapped = self.env.unwrapped
        nan = torch.full((self.num_envs,), float("nan"), dtype=dtype, device=self.device)

        succeeded = getattr(unwrapped, "ep_succeeded", None)
        if succeeded is None:
            return {}
        succeeded = succeeded.reshape(-1).to(torch.bool)
        metrics: Dict[str, torch.Tensor] = {"episode/success": succeeded.to(dtype)}

        times = getattr(unwrapped, "ep_success_times", None)
        if times is not None:
            # NaN where it never succeeded, so the mean is "how long a success took"
            metrics["episode/success_step"] = torch.where(succeeded, times.to(dtype), nan)

        if self._engage_threshold is not None:
            # the LATCH, not a live reading: "did this episode ever engage". Success implies
            # engagement by a looser threshold, so latched this way the engagement rate is
            # always >= the success rate, which is the only way the pair means anything.
            metrics["episode/engaged"] = self._engaged.to(dtype)

        metrics.update(self._termination_causes(succeeded, terminated, truncated, dtype))
        metrics.update(self._prediction_metrics(succeeded, dtype, nan))
        return metrics

    def _termination_causes(self, succeeded, terminated, truncated, dtype) -> Dict[str, torch.Tensor]:
        """One 0/1 per cause, so each averages into its own rate.

        ``success`` wins over ``timeout`` when both are true on the same step: the episode
        ended because it was finished, not because the clock ran out.

        **``timeout`` is the clock, read off ``truncated`` alone.** Not "any done that was
        not a success" — that counted every broken peg as a timeout, because the fragile
        wrapper ends an episode by raising ``terminated``. Factory and Forge themselves end
        only on the clock and raise both flags there, so ``truncated`` is exactly the clock
        and ``terminated & ~truncated`` is exactly an early termination. The early causes are
        owned by the wrapper that decides them: ``fragile/broke`` and ``fragile/cause_*``.
        So ``termination/success``, ``termination/timeout`` and ``fragile/broke`` together
        account for every episode, across the two namespaces.
        """
        truncated = truncated.reshape(-1).to(torch.bool)
        success = succeeded
        timeout = truncated & ~success
        return {
            "termination/success": success.to(dtype),
            "termination/timeout": timeout.to(dtype),
        }

    def _prediction_metrics(self, succeeded, dtype, nan) -> Dict[str, torch.Tensor]:
        """How good Forge's success-prediction action was, per env.

        The policy's 7th action is a success prediction in [0, 1]. Over the episodes that end
        this step: did it predict a success that happened (precision), did it catch the
        successes that happened (recall), how many steps late was it (delay), and how far off
        was the value itself (error). Every one is NaN where the question does not apply, so
        the averages are over the episodes that could answer it.
        """
        unwrapped = self.env.unwrapped
        if self._prediction is None:
            return {}
        prediction = self._prediction.reshape(-1).to(dtype)
        metrics = {f"{SUCCESS_PREDICTION}/error": (succeeded.to(dtype) - prediction).abs()}

        thresholds = getattr(unwrapped, "first_pred_success_tx", None)
        times = getattr(unwrapped, "ep_success_times", None)
        if not isinstance(thresholds, dict) or times is None:
            return metrics

        for threshold, first_predicted in thresholds.items():
            predicted = first_predicted.reshape(-1) > 0
            actual_step = times.reshape(-1).to(dtype)
            actual = succeeded
            tag = f"{SUCCESS_PREDICTION}/{threshold}"
            # precision: of the episodes it called, how many really succeeded (and did so
            # before the call); recall: of the successes, how many it called at all
            correct = predicted & actual & (actual_step < first_predicted.reshape(-1).to(dtype))
            metrics[f"{tag}_precision"] = torch.where(predicted, correct.to(dtype), nan)
            metrics[f"{tag}_recall"] = torch.where(actual, predicted.to(dtype), nan)
            delay = first_predicted.reshape(-1).to(dtype) - actual_step
            both = predicted & actual
            metrics[f"{tag}_delay_all"] = torch.where(both, delay, nan)
            metrics[f"{tag}_delay_correct"] = torch.where(both & (delay > 0), delay, nan)
        return metrics

    # ------------------------------------------------------------------ the step
    def step(self, action):
        observations, rewards, terminated, truncated, infos = self.env.step(action)
        if isinstance(infos, dict):
            dtype = rewards.dtype if torch.is_floating_point(rewards) else torch.float32
            infos.setdefault("metrics_to_log", {}).update(self.step_metrics(dtype))
            infos.setdefault("episode_metrics_to_log", {}).update(
                self.episode_metrics(terminated, truncated, dtype)
            )
        self._terms = {}
        # the env clears its own ep_succeeded on the NEXT step's _pre_physics_step; the
        # engagement latch is ours, so it is cleared here, once the episode is reported
        done = torch.logical_or(terminated.reshape(-1), truncated.reshape(-1))
        self._engaged &= ~done
        return observations, rewards, terminated, truncated, infos
