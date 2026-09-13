from __future__ import annotations

from math import prod

import torch

from rsl_rl.modules import EmpiricalNormalization

from mjlab.rl.runner import MjlabOnPolicyRunner


class MoMoMaskedEmpiricalNormalization(EmpiricalNormalization):
    """EmpNorm that leaves one contiguous MoMo slice as an exact identity."""

    def __init__(
        self,
        source: EmpiricalNormalization,
        momo_slice: slice,
    ) -> None:
        super().__init__(
            shape=tuple(source._mean.shape[1:]),
            eps=source.eps,
            until=source.until,
        )
        self.load_state_dict(source.state_dict())
        self.momo_start = int(momo_slice.start)
        self.momo_stop = int(momo_slice.stop)
        self.reset_momo_slice()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = super().forward(x)
        y[..., self.momo_start : self.momo_stop] = x[..., self.momo_start : self.momo_stop]
        return y

    @torch.jit.unused
    def update(self, x: torch.Tensor) -> None:
        super().update(x)
        self.reset_momo_slice()

    @torch.jit.unused
    def reset_momo_slice(self) -> None:
        with torch.inference_mode():
            self._mean[..., self.momo_start : self.momo_stop] = 0.0
            self._var[..., self.momo_start : self.momo_stop] = 1.0
            self._std[..., self.momo_start : self.momo_stop] = 1.0


class MoMoEffortScaleRunner(MjlabOnPolicyRunner):
    """Runner that keeps effort-scaled MoMo observations out of EmpNorm."""

    momo_obs_sets = ("actor", "critic")
    masked_obs_term = "momo_disturbance"
    masked_obs_terms: dict[str, str] = {}

    def __init__(
        self,
        env,
        train_cfg: dict,
        log_dir: str | None = None,
        device: str = "cpu",
    ) -> None:
        super().__init__(env, train_cfg, log_dir, device)
        self.momo_normalizer_slices = {
            obs_set: self._install_momo_masked_normalizer(
                getattr(self.alg, obs_set),
                obs_set,
            )
            for obs_set in self.momo_obs_sets
        }

    def load(
        self,
        path: str,
        load_cfg: dict | None = None,
        strict: bool = True,
        map_location: str | None = None,
    ) -> dict:
        infos = super().load(
            path,
            load_cfg=load_cfg,
            strict=strict,
            map_location=map_location,
        )
        self.reset_momo_normalizer_slices()
        return infos

    def save(self, path: str, infos=None) -> None:
        self.reset_momo_normalizer_slices()
        super().save(path, infos)

    def reset_momo_normalizer_slices(self) -> None:
        for obs_set in self.momo_obs_sets:
            model = getattr(self.alg, obs_set)
            normalizer = model.obs_normalizer
            if not isinstance(normalizer, MoMoMaskedEmpiricalNormalization):
                raise RuntimeError(
                    "MoMoEffortScaleRunner requires MoMoMaskedEmpiricalNormalization"
                )
            normalizer.reset_momo_slice()

    def _install_momo_masked_normalizer(self, model, obs_set: str) -> slice:
        momo_slice = self._find_momo_slice(model, obs_set)
        if not isinstance(model.obs_normalizer, EmpiricalNormalization):
            raise RuntimeError(
                f"{obs_set} obs_normalization must be enabled for MoMoEffortScaleRunner"
            )
        model.obs_normalizer = MoMoMaskedEmpiricalNormalization(
            model.obs_normalizer,
            momo_slice,
        ).to(self.device)
        return momo_slice

    def _find_momo_slice(self, model, obs_set: str) -> slice:
        obs_manager = self.env.unwrapped.observation_manager
        masked_obs_term = self.masked_obs_terms.get(
            obs_set,
            self.masked_obs_term,
        )
        offset = 0
        for group_name in model.obs_groups:
            if not obs_manager.group_obs_concatenate[group_name]:
                raise RuntimeError(
                    "MoMoEffortScaleRunner only supports concatenated observation groups"
                )

            term_offset = 0
            term_names = obs_manager.active_terms[group_name]
            term_dims = obs_manager.group_obs_term_dim[group_name]
            for term_name, term_dim in zip(term_names, term_dims, strict=True):
                flat_dim = int(prod(term_dim))
                if term_name == masked_obs_term:
                    return slice(offset + term_offset, offset + term_offset + flat_dim)
                term_offset += flat_dim

            group_dim = obs_manager.group_obs_dim[group_name]
            if not isinstance(group_dim, tuple):
                raise RuntimeError("MoMoEffortScaleRunner only supports tensor observation groups")
            offset += int(prod(group_dim))

        raise RuntimeError(
            f"Could not find {masked_obs_term} in {obs_set} observation groups {model.obs_groups}"
        )
