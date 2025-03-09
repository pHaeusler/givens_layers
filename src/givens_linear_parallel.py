import torch
import torch.nn as nn


class GivensLinearParallel(nn.Module):
    def __init__(self, dim, bias=True):
        super().__init__()
        self.dim = dim
        self.num_rotations = dim * (dim - 1) // 2
        self.angles = nn.Parameter(torch.zeros(self.num_rotations))
        self.bias = nn.Parameter(torch.zeros(dim)) if bias else None
        pairs = [(i, j) for i in range(dim) for j in range(i + 1, dim)]
        pair_to_idx = {pair: idx for idx, pair in enumerate(pairs)}

        # Schedule pairs into steps with disjoint indices
        schedule = []
        remaining_pairs = set(pairs)
        while remaining_pairs:
            step = []
            used_indices = set()
            for pair in list(remaining_pairs):
                i, j = pair
                if i not in used_indices and j not in used_indices:
                    step.append(pair)
                    used_indices.update([i, j])
            schedule.append(step)
            remaining_pairs -= set(step)

        self.num_steps = len(schedule)
        for step_idx, step in enumerate(schedule):
            I_step = torch.tensor([p[0] for p in step], dtype=torch.long)
            J_step = torch.tensor([p[1] for p in step], dtype=torch.long)
            angle_idx_step = torch.tensor([pair_to_idx[p] for p in step], dtype=torch.long)
            self.register_buffer(f"I_step_{step_idx}", I_step)
            self.register_buffer(f"J_step_{step_idx}", J_step)
            self.register_buffer(f"angle_idx_step_{step_idx}", angle_idx_step)

    def forward(self, x):
        for step_idx in range(self.num_steps):
            I_step = getattr(self, f"I_step_{step_idx}")
            J_step = getattr(self, f"J_step_{step_idx}")
            angle_idx_step = getattr(self, f"angle_idx_step_{step_idx}")
            angles_step = self.angles[angle_idx_step]
            cosθ, sinθ = torch.cos(angles_step), torch.sin(angles_step)
            x_I, x_J = x[:, I_step], x[:, J_step]
            x[:, I_step] = cosθ * x_I - sinθ * x_J
            x[:, J_step] = sinθ * x_I + cosθ * x_J
        return x + self.bias if self.bias is not None else x

    def weight_matrix(self):
        W = torch.eye(self.dim, device=self.angles.device)
        for step_idx in range(self.num_steps):
            I_step = getattr(self, f"I_step_{step_idx}")
            J_step = getattr(self, f"J_step_{step_idx}")
            angle_idx_step = getattr(self, f"angle_idx_step_{step_idx}")
            for k in range(len(I_step)):
                i, j = I_step[k], J_step[k]
                angle_idx = angle_idx_step[k]
                theta = self.angles[angle_idx]
                cos_theta, sin_theta = torch.cos(theta), torch.sin(theta)
                Wi, Wj = W[:, i].clone(), W[:, j].clone()
                W[:, i] = cos_theta * Wi - sin_theta * Wj
                W[:, j] = sin_theta * Wi + cos_theta * Wj
        return W
