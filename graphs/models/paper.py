import logging
from torch import Tensor
import torch
from torch.nn import (
    Conv2d,
    Linear,
    Flatten,
    ReLU,
    MaxPool2d,
    Module,
    Sequential,
    BatchNorm2d,
    Dropout2d,
)
from torch_geometric.nn import GCNConv
from torch.cuda import is_available
from torch import isnan, ones_like, device


class PaperArchitecture(Module):
    """
    #### Test 1

    Try to replicate original paper architecture

    input tensor: [t, n agents, channels, h vision, w vision]
    """

    def __init__(self, config) -> None:
        super().__init__()
        self.logger = logging.getLogger("Agent")
        self.config = config
        self.device = device("cuda" if config["cuda"] and is_available() else "cpu")

        self.n_agents = config["num_agents"]
        self.FOV = config["FOV"]
        self.wl, self.hl = self.FOV + 2, self.FOV + 2  # from paper
        self.CHANNELS: int = 3  # agents, obstacles, goals
        self.n_actions: int = 5  # stay, up, down, left, right
        self.E = 1  # Number of edge features
        self.nGraphFilterTaps = self.config["nGraphFilterTaps"]

        self.conv1 = self._conv_block(self.CHANNELS, 128)
        self.conv2 = self._conv_block(128, 128)
        self.conv3 = self._conv_block(128, 128)
        self.drop = Dropout2d(p=0.2)
        self.cgnn = GCNConv(128, 128)
        self.fc = Sequential(
            Flatten(),
            Linear(128 * (self.wl // 8) * (self.hl // 8), 256),
            ReLU(),
            Linear(256, self.n_actions),
        )

    def _dense_to_edge_index(self, S: Tensor):
        B, N, _ = S.shape
        rows, cols = [], []
        for b in range(B):
            idx = (S[b] > 0).nonzero(as_tuple=False)  # coppie (i,j)
            if idx.numel() == 0:
                continue
            rows.append(idx[:, 0] + b * N)
            cols.append(idx[:, 1] + b * N)
        if not rows:
            return torch.empty(2, 0, dtype=torch.long, device=S.device)
        return torch.stack([torch.cat(rows), torch.cat(cols)], dim=0)

    def addGSO(self, S: Tensor) -> None:
        """
        Training is made on different steps of different maps so we need to get the GSO everytime

        this function refers to the original one in the repository
        """
        if self.E == 1:  # It is B x T x N x N
            assert len(S.shape) == 3
            self.S = S.unsqueeze(1)  # B x E x N x N
        else:
            assert len(S.shape) == 4
            assert S.shape[1] == self.E
            self.S = S

        # Remove nan data
        self.S[isnan(self.S)] = 0
        if self.config["GSO_mode"] == "dist_GSO_one":
            self.S[self.S > 0] = 1
        elif self.config["GSO_mode"] == "full_GSO":
            self.S = ones_like(self.S).to(self.device)

    def _conv_block(
        self,
        inp: int,
        output: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
    ) -> Sequential:
        return Sequential(
            Conv2d(
                in_channels=inp,
                out_channels=output,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            ),
            BatchNorm2d(output),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2),
            Conv2d(
                in_channels=output,
                out_channels=output,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            ),
            BatchNorm2d(output),
            ReLU(),
        )

    def _format_to_conv2d(self, x: Tensor) -> Tensor:
        """
        Conv2d expects (B, C, W, H), we have (B, N, C, W, H),
        so we merge B and N dimensions to avoid looping
        """
        (B, N, C, W, H) = x.shape
        x = x.view(B * N, C, W, H)
        return x

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass through the network
        x: (B, N, C, W, H)
        return: (B, N, n_actions)
        """
        # convolutional layers
        x = self._format_to_conv2d(x).to(self.device)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        # graph conv layer

        # fully connected layers
        x = self.fc(x)
        return x
