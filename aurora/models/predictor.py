"""Slow predictor for Aurora replay and transition scoring."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
from torch import nn

from aurora.core.config import PredictorConfig
from aurora.core.types import ExperienceFrame, PredictorPeek, PredictorState


class _PredictorNet(nn.Module):
    def __init__(self, config: PredictorConfig):
        super().__init__()
        self.config = config
        self.input_dim = config.latent_dim * 2 + config.action_dim + 1
        self.hidden_dim = config.hidden_dim
        if config.hidden_dim > 0:
            self.proposal = nn.Linear(self.input_dim, config.hidden_dim)
            self.gate = nn.Linear(self.input_dim, config.hidden_dim)
            self.mu_head = nn.Linear(config.hidden_dim, config.latent_dim)
            self.log_sigma_head = nn.Linear(config.hidden_dim, config.latent_dim)
        else:
            self.linear_mu = nn.Linear(self.input_dim, config.latent_dim)
            self.linear_log_sigma = nn.Parameter(torch.zeros(config.latent_dim))

    def forward(
        self,
        workspace: torch.Tensor,
        frontier: torch.Tensor,
        action: torch.Tensor,
        delta_t: torch.Tensor,
        h_prev: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dt = delta_t.reshape(-1, 1)
        inp = torch.cat([workspace, frontier, action, dt], dim=-1)
        if self.hidden_dim > 0:
            if h_prev is None:
                h_prev = torch.zeros(inp.shape[0], self.hidden_dim, dtype=inp.dtype, device=inp.device)
            gate = torch.sigmoid(self.gate(inp))
            proposal = torch.tanh(self.proposal(inp))
            h = (1.0 - gate) * h_prev + gate * proposal
            mu = self.mu_head(h)
            log_sigma = torch.clamp(
                self.log_sigma_head(h),
                min=self.config.min_log_sigma,
                max=self.config.max_log_sigma,
            )
            return mu, log_sigma, h
        mu = self.linear_mu(inp)
        log_sigma = torch.clamp(
            self.linear_log_sigma.unsqueeze(0).expand_as(mu),
            min=self.config.min_log_sigma,
            max=self.config.max_log_sigma,
        )
        h = torch.zeros(inp.shape[0], 0, dtype=inp.dtype, device=inp.device)
        return mu, log_sigma, h


class SlowPredictor:
    def __init__(self, config: PredictorConfig, *, lr: float = 3e-3, weight_decay: float = 1e-5):
        self.config = config
        self.device = torch.device(config.device)
        self.model = _PredictorNet(config).to(self.device)
        self.target_model = _PredictorNet(config).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.h = (
            torch.zeros(1, config.hidden_dim, device=self.device)
            if config.hidden_dim > 0
            else torch.zeros(1, 0, device=self.device)
        )
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.training_steps = 0

    def reset_state(self) -> None:
        self.h = (
            torch.zeros(1, self.config.hidden_dim, device=self.device)
            if self.config.hidden_dim > 0
            else torch.zeros(1, 0, device=self.device)
        )

    def _to_tensor(self, x: np.ndarray | Sequence[float]) -> torch.Tensor:
        arr = np.asarray(x, dtype=np.float64)
        return torch.as_tensor(arr, dtype=torch.float32, device=self.device)

    def peek(
        self,
        workspace_vec: np.ndarray | Sequence[float],
        frontier_vec: np.ndarray | Sequence[float],
        action_vec: np.ndarray | Sequence[float],
        delta_t: float = 1.0,
    ) -> PredictorPeek:
        workspace = self._to_tensor(workspace_vec).reshape(1, -1)
        frontier = self._to_tensor(frontier_vec).reshape(1, -1)
        action = self._to_tensor(action_vec).reshape(1, -1)
        dt = torch.tensor([float(delta_t)], dtype=torch.float32, device=self.device)
        with torch.no_grad():
            mu, log_sigma, h_next = self.model(workspace, frontier, action, dt, self.h)
        return PredictorPeek(
            mu=mu.squeeze(0).cpu().numpy().astype(np.float64),
            sigma_diag=torch.exp(log_sigma.squeeze(0)).cpu().numpy().astype(np.float64),
            h=h_next.squeeze(0).cpu().numpy().astype(np.float64),
        )

    def step(
        self,
        workspace_vec: np.ndarray | Sequence[float],
        frontier_vec: np.ndarray | Sequence[float],
        action_vec: np.ndarray | Sequence[float],
        delta_t: float = 1.0,
    ) -> PredictorPeek:
        workspace = self._to_tensor(workspace_vec).reshape(1, -1)
        frontier = self._to_tensor(frontier_vec).reshape(1, -1)
        action = self._to_tensor(action_vec).reshape(1, -1)
        dt = torch.tensor([float(delta_t)], dtype=torch.float32, device=self.device)
        with torch.no_grad():
            mu, log_sigma, h_next = self.model(workspace, frontier, action, dt, self.h)
        self.h = h_next.detach()
        return PredictorPeek(
            mu=mu.squeeze(0).cpu().numpy().astype(np.float64),
            sigma_diag=torch.exp(log_sigma.squeeze(0)).cpu().numpy().astype(np.float64),
            h=h_next.squeeze(0).cpu().numpy().astype(np.float64),
        )

    def update_target(self, ema: float = 0.995) -> None:
        with torch.no_grad():
            for target_param, param in zip(self.target_model.parameters(), self.model.parameters()):
                target_param.mul_(ema).add_(param, alpha=1.0 - ema)

    def _batch_tensors(self, frames: Sequence[ExperienceFrame]) -> tuple[torch.Tensor, ...]:
        workspace = np.stack([np.asarray(frame.workspace_vec, dtype=np.float64) for frame in frames], axis=0)
        frontier = np.stack([np.asarray(frame.frontier_vec, dtype=np.float64) for frame in frames], axis=0)
        action = np.stack([np.asarray(frame.action_vec, dtype=np.float64) for frame in frames], axis=0)
        delta_t = np.asarray(
            [1.0 if frame.next_ts is None else max(float(frame.next_ts - frame.ts), 1e-3) for frame in frames],
            dtype=np.float64,
        )
        target = np.stack([np.asarray(frame.next_x, dtype=np.float64) for frame in frames], axis=0)
        return (
            torch.as_tensor(workspace, dtype=torch.float32, device=self.device),
            torch.as_tensor(frontier, dtype=torch.float32, device=self.device),
            torch.as_tensor(action, dtype=torch.float32, device=self.device),
            torch.as_tensor(delta_t, dtype=torch.float32, device=self.device),
            torch.as_tensor(target, dtype=torch.float32, device=self.device),
        )

    def fit_batch(
        self,
        frames: Sequence[ExperienceFrame],
        *,
        train_steps: int = 1,
        target_ema: float = 0.995,
        drift_penalty: float = 1e-5,
    ) -> dict[str, float]:
        frames = [frame for frame in frames if frame.next_x is not None]
        if not frames:
            return {"loss": 0.0, "nll": 0.0, "drift": 0.0}
        workspace, frontier, action, delta_t, target = self._batch_tensors(frames)
        last_loss = 0.0
        last_nll = 0.0
        last_drift = 0.0
        for _ in range(max(int(train_steps), 1)):
            h0 = (
                torch.zeros(workspace.shape[0], self.config.hidden_dim, dtype=torch.float32, device=self.device)
                if self.config.hidden_dim > 0
                else torch.zeros(workspace.shape[0], 0, dtype=torch.float32, device=self.device)
            )
            mu, log_sigma, _ = self.model(workspace, frontier, action, delta_t, h0)
            var = torch.exp(log_sigma)
            nll = 0.5 * (((target - mu) ** 2) / var + log_sigma).sum(dim=-1).mean()
            drift = torch.tensor(0.0, device=self.device)
            if drift_penalty > 0.0:
                for param, target_param in zip(self.model.parameters(), self.target_model.parameters()):
                    drift = drift + ((param - target_param.detach()) ** 2).mean()
            loss = nll + drift_penalty * drift
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            self.update_target(ema=target_ema)
            self.training_steps += 1
            last_loss = float(loss.detach().cpu().item())
            last_nll = float(nll.detach().cpu().item())
            last_drift = float(drift.detach().cpu().item())
        return {"loss": last_loss, "nll": last_nll, "drift": last_drift}

    def score_transition(self, frame: ExperienceFrame) -> float:
        if frame.next_x is None:
            return 0.0
        workspace = self._to_tensor(frame.workspace_vec).reshape(1, -1)
        frontier = self._to_tensor(frame.frontier_vec).reshape(1, -1)
        action = self._to_tensor(frame.action_vec).reshape(1, -1)
        delta_t = torch.tensor(
            [1.0 if frame.next_ts is None else max(float(frame.next_ts - frame.ts), 1e-3)],
            dtype=torch.float32,
            device=self.device,
        )
        target = self._to_tensor(frame.next_x).reshape(1, -1)
        h0 = (
            torch.zeros(1, self.config.hidden_dim, dtype=torch.float32, device=self.device)
            if self.config.hidden_dim > 0
            else torch.zeros(1, 0, dtype=torch.float32, device=self.device)
        )
        with torch.no_grad():
            mu, log_sigma, _ = self.model(workspace, frontier, action, delta_t, h0)
            var = torch.exp(log_sigma)
            nll = 0.5 * (((target - mu) ** 2) / var + log_sigma).sum(dim=-1).mean()
        return float(nll.cpu().item())

    def export_state(self) -> PredictorState:
        theta = {k: v.detach().cpu().numpy().astype(np.float64) for k, v in self.model.state_dict().items()}
        theta_target = {
            k: v.detach().cpu().numpy().astype(np.float64) for k, v in self.target_model.state_dict().items()
        }
        h = self.h.detach().cpu().numpy().reshape(-1).astype(np.float64)
        return PredictorState(h=h, theta=theta, theta_target=theta_target)

    def restore_state(self, state: PredictorState) -> None:
        model_state = {k: torch.as_tensor(v, dtype=torch.float32, device=self.device) for k, v in state.theta.items()}
        target_state = {
            k: torch.as_tensor(v, dtype=torch.float32, device=self.device) for k, v in state.theta_target.items()
        }
        self.model.load_state_dict(model_state)
        self.target_model.load_state_dict(target_state)
        if self.config.hidden_dim > 0:
            self.h = torch.as_tensor(state.h, dtype=torch.float32, device=self.device).reshape(1, -1)
        else:
            self.h = torch.zeros(1, 0, dtype=torch.float32, device=self.device)
