from __future__ import annotations

import torch


class VectorizedMoMo:
    """Vectorized Momentum Observer for parallel environments.

    Maintains independent MoMo state for each environment and supports batch updates.
    Uses the ideal robot model's momentum and bias forces for the disturbance-free
    prediction.

    Attributes:
        num_envs: Number of parallel environments.
        nv: Velocity degrees of freedom (6 for base + num_joints).
        dt: Time step for discrete integration.
        device: PyTorch device (cuda/cpu).
        K_O: Observer gain matrix (num_envs, nv).
        r: Current disturbance residual / estimate (num_envs, nv).
        p_prev: Previous-step generalized momentum (num_envs, nv).
    """

    def __init__(
        self,
        num_envs: int,
        nv: int,
        K_O: torch.Tensor | list[float] | float,
        dt: float,
        device: str,
        nominal_mg: float = 0.0,
        butter_order: int = 0,
        butter_cutoff: float = 1.0,
    ):
        """Initialize vectorized MoMo.

        Args:
            num_envs: Number of parallel environments.
            nv: Velocity degrees of freedom (typically 6 + num_joints).
            K_O: Observer gains. Can be:
                - float: Same gain for all DOFs and environments
                - list[float]: Per-DOF gains (length must equal nv)
                - torch.Tensor: Custom gain matrix (num_envs, nv) or (nv,)
            dt: Integration time step (typically env.step_dt).
            device: PyTorch device string ('cuda' or 'cpu').
            nominal_mg: Nominal m*g for baseline subtraction.
            butter_order: Butterworth post-filter order (0=disabled, 2 or 4).
            butter_cutoff: Butterworth cutoff frequency in Hz.

        Example:
            >>> momo = VectorizedMoMo(num_envs=4096, nv=35, K_O=3.0, dt=0.02, device='cuda')
        """
        self.num_envs = num_envs
        self.nv = nv
        self.dt = dt
        self.device = device
        self.nominal_mg = nominal_mg

        # Process gain parameter
        if isinstance(K_O, (int, float)):
            # Scalar gain: same for all DOFs and environments
            self.K_O = torch.full((num_envs, nv), float(K_O), device=device)
        elif isinstance(K_O, list):
            # List of per-DOF gains
            if len(K_O) != nv:
                raise ValueError(f"K_O list length {len(K_O)} must equal nv {nv}")
            gains_tensor = torch.tensor(K_O, dtype=torch.float32, device=device)
            self.K_O = gains_tensor.unsqueeze(0).expand(num_envs, -1).clone()
        elif isinstance(K_O, torch.Tensor):
            # Tensor gains
            K_O = K_O.to(device)
            if K_O.dim() == 1:
                # (nv,) -> (num_envs, nv)
                if K_O.shape[0] != nv:
                    raise ValueError(f"K_O tensor length {K_O.shape[0]} must equal nv {nv}")
                self.K_O = K_O.unsqueeze(0).expand(num_envs, -1).clone()
            elif K_O.dim() == 2:
                # (num_envs, nv)
                if K_O.shape != (num_envs, nv):
                    raise ValueError(f"K_O shape {K_O.shape} must be ({num_envs}, {nv})")
                self.K_O = K_O.clone()
            else:
                raise ValueError(f"K_O tensor must be 1D or 2D, got {K_O.dim()}D")
        else:
            raise TypeError(f"K_O must be float, list, or torch.Tensor, got {type(K_O)}")

        # Internal state: residual (disturbance estimate)
        self.r = torch.zeros((num_envs, nv), device=device)

        # Previous-step generalized momentum
        self.p_prev = torch.zeros((num_envs, nv), device=device)

        # Filtered output (observations read this, not raw r)
        self.r_filtered = torch.zeros((num_envs, nv), device=device)

        # Butterworth post-filter (causal IIR, matches standalone MomentumObserver)
        self.butter_order = butter_order
        self.butter_cutoff = butter_cutoff
        if butter_order > 0:
            from scipy.signal import butter as sp_butter, sosfilt_zi

            fs = 1.0 / dt
            sos = sp_butter(butter_order, butter_cutoff, btype="low", fs=fs, output="sos")
            n_sections = sos.shape[0]
            self.n_sections = n_sections
            # SOS coefficients: b (numerator), a (denominator, skip a0=1)
            self.sos_b = torch.tensor(sos[:, :3], dtype=torch.float32, device=device)
            self.sos_a = torch.tensor(sos[:, 4:6], dtype=torch.float32, device=device)
            # Filter state: (n_sections, 2, num_envs, nv)
            self.butter_zi = torch.zeros(n_sections, 2, num_envs, nv, device=device)
            # Unit steady-state zi for warm-start (n_sections, 2)
            self._zi_unit = torch.tensor(sosfilt_zi(sos), dtype=torch.float32, device=device)
        else:
            self.n_sections = 0
            self.sos_b = None
            self.sos_a = None
            self.butter_zi = None
            self._zi_unit = None

        self.initialized = False

    def _butter_filter_step(self, x: torch.Tensor) -> torch.Tensor:
        """Apply one step of causal Butterworth SOS filter (transposed direct form II).

        Equivalent to scipy.signal.sosfilt processing one sample at a time.
        Each call processes a single time-step for all envs and DOFs in parallel.

        Args:
            x: Input signal (num_envs, nv).

        Returns:
            Filtered signal (num_envs, nv).
        """
        y = x
        for s in range(self.n_sections):
            x_in = y
            b0 = self.sos_b[s, 0]
            b1 = self.sos_b[s, 1]
            b2 = self.sos_b[s, 2]
            a1 = self.sos_a[s, 0]
            a2 = self.sos_a[s, 1]
            y = b0 * x_in + self.butter_zi[s, 0]
            self.butter_zi[s, 0] = b1 * x_in - a1 * y + self.butter_zi[s, 1]
            self.butter_zi[s, 1] = b2 * x_in - a2 * y
        return y

    def initialize_state(self):
        """Warm-start the vertical residual at nominal weight.

        Previous momentum starts at zero; the environment reset anchors it to
        the reset state before the next update.
        """
        self.r.zero_()
        self.r[:, 2] = self.nominal_mg
        self.r_filtered.zero_()
        self.r_filtered[:, 2] = self.nominal_mg
        self.p_prev.zero_()
        if self.butter_order > 0:
            self.butter_zi.zero_()
            # Warm-start z-axis at nominal_mg
            self.butter_zi[:, :, :, 2] = self._zi_unit.unsqueeze(-1) * self.nominal_mg
        self.initialized = True

    def reset(self, env_ids: torch.Tensor):
        """Reset MoMo state for specific environments.

        Args:
            env_ids: Environment indices to reset (K,).
        """
        if len(env_ids) == 0:
            return

        if env_ids.dim() == 0:
            env_ids = env_ids.unsqueeze(0)

        self.r[env_ids] = 0.0
        self.r[env_ids, 2] = self.nominal_mg
        self.r_filtered[env_ids] = 0.0
        self.r_filtered[env_ids, 2] = self.nominal_mg
        self.p_prev[env_ids] = 0.0
        if self.butter_order > 0:
            self.butter_zi[:, :, env_ids] = 0.0
            self.butter_zi[:, :, env_ids, 2] = self._zi_unit.unsqueeze(-1) * self.nominal_mg

    def update(
        self,
        p_current: torch.Tensor,
        beta: torch.Tensor,
        tau_v: torch.Tensor,
        tau_passive: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Update MoMo and compute disturbance residual.

        This should be called once per environment step, before computing observations.

        Args:
            p_current: Current generalized momentum M(q) * qdot (num_envs, nv).
            beta: Bias term C^T qdot - g, approximated as -qfrc_bias (num_envs, nv).
            tau_v: Actuator generalized forces (num_envs, nv).
            tau_passive: Passive forces (damping, spring, etc.) from qfrc_passive
                (num_envs, nv). Optional; when provided, these are included in the
                known-forces term so they are not lumped into the disturbance estimate.

        Returns:
            r: Estimated disturbance residual (num_envs, nv).

        Theory (discrete recursion):
            delta_p = p_current - p_prev
            known_forces = tau_v + beta + r  (+ tau_passive if provided)
            r = r + K_O * delta_p - K_O * dt * known_forces
            p_prev = p_current
        """
        if p_current.shape != (self.num_envs, self.nv):
            raise ValueError(
                f"p_current shape {p_current.shape} must be ({self.num_envs}, {self.nv})"
            )
        if beta.shape != (self.num_envs, self.nv):
            raise ValueError(f"beta shape {beta.shape} must be ({self.num_envs}, {self.nv})")
        if tau_v.shape != (self.num_envs, self.nv):
            raise ValueError(f"tau_v shape {tau_v.shape} must be ({self.num_envs}, {self.nv})")
        if tau_passive is not None and tau_passive.shape != (self.num_envs, self.nv):
            raise ValueError(
                f"tau_passive shape {tau_passive.shape} must be ({self.num_envs}, {self.nv})"
            )

        # 1. Momentum difference
        delta_p = p_current - self.p_prev

        # 2. Compute known forces
        known_forces = tau_v + beta + self.r
        if tau_passive is not None:
            known_forces = known_forces + tau_passive

        # 3. Update residual: r = r + K_O * delta_p - K_O * dt * known_forces
        self.r = self.r + self.K_O * delta_p - self.K_O * self.dt * known_forces

        # 4. Store current momentum for next step
        self.p_prev = p_current.clone()

        # 5. Apply Butterworth post-filter (or pass-through)
        if self.butter_order > 0:
            self.r_filtered = self._butter_filter_step(self.r)
        else:
            self.r_filtered.copy_(self.r)

        return self.r_filtered
