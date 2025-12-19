import torch
from torch import nn, Tensor, cat

class TimeDelayedAggregation(nn.Module):
    def __init__(self, in_channels: int, n_taps: int):
        super().__init__()
        self.n_taps = n_taps
        self.weight = nn.Parameter(torch.Tensor(n_taps, in_channels, in_channels))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x_curr: Tensor, x_prev: Tensor, S: Tensor) -> Tensor:
        """
        x_curr: [B * N, 128]
        x_prev: [B, N, T, 128]
        S: [B, N, N]
        """
        if x_prev.shape[2] < self.n_taps - 1:
            raise ValueError("x_prev.shape[2] must be >= self.n_taps - 1")
        B_N, C = x_curr.shape
        B = S.shape[0]
        N = S.shape[1]
        y_time = [torch.matmul(x_curr, self.weight[0])] # [B*N, 128]
        
        for t in range(self.n_taps - 1):
            prev_feat = x_prev[:, :, t, :] # [B, N, 128]
            filtered = torch.matmul(S, prev_feat) # [B, N, 128]
            filtered = filtered.view(B_N, C)
            y_time.append(torch.matmul(filtered, self.weight[t+1]))
        return cat(y_time, dim=1)