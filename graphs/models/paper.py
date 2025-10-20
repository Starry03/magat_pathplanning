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
    Dropout,
    AdaptiveAvgPool2d,
)
from torch_geometric.nn import GCNConv, Sequential as GSequential
from torch.cuda import is_available
from torch import isnan, ones_like

from graphs.weights_initializer import weights_init


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
        self.device = torch.device(
            "cuda" if config["cuda"] and is_available() else "cpu"
        )

        self.n_agents = config["num_agents"]
        self.FOV = config["fov"]
        self.wl, self.hl = self.FOV + 2, self.FOV + 2  # from paper
        self.CHANNELS: int = 3  # agents, obstacles, goals
        self.n_actions: int = 5  # stay, up, down, left, right
        self.E = 1  # Number of edge features
        self.nGraphFilterTaps = self.config["nGraphFilterTaps"]

        # layers
        self.pool = AdaptiveAvgPool2d((1, 1))
        self.drop = Dropout(p=0.2)
        self.activation = ReLU(inplace=True)
        self.conv1 = self._conv_block(self.CHANNELS, 32)
        self.conv2 = self._conv_block(32, 64)
        self.conv3 = self._conv_block(64, 128)
        self.compress = Sequential(
            Conv2d(128, 128, kernel_size=1, stride=1, padding=0),
            BatchNorm2d(128),
            self.activation,
        )
        self.flatten = Flatten()
        self.cgnn = self._graph_block(128, 128, k=self.nGraphFilterTaps)
        self.fc = Linear(128, self.n_actions)
        self.to(self.device)
        self.S = torch.ones(1, self.E, self.n_agents, self.n_agents, device=self.device)
        # self.apply(weights_init) TODO: make this work

    def _agents_to_edge_index(self, S: Tensor):
        """

        return: edge_index in shape [2, E]
        """
        B, _, N, _ = S.shape
        rows, cols = [], []
        for b in range(B):
            idx = (S[b] > 0).nonzero(as_tuple=False)  # coppie (i,j)
            if idx.numel() == 0:
                continue

            # geometric offset to have a single big graph
            rows.append(idx[:, 0] + b * N)
            cols.append(idx[:, 1] + b * N)
        if not rows:
            return torch.empty(2, 0, dtype=torch.long, device=S.device)

        # concatenate all batches
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
            self.S = ones_like(self.S)
        self.S = self.S.to(self.device)

    def _conv_block(
        self,
        inp: int,
        output: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
        dilation: int = 1,
        ceiling_mode: bool = False,
    ) -> Sequential:
        return Sequential(
            Conv2d(
                in_channels=inp,
                out_channels=output,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            ),
            BatchNorm2d(output, eps, momentum, affine, track_running_stats),
            self.activation,
            MaxPool2d(
                kernel_size=2,
                stride=2,
                padding=0,
                dilation=dilation,
                ceil_mode=ceiling_mode,
            ),
            Conv2d(
                in_channels=output,
                out_channels=output,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            ),
            BatchNorm2d(output, eps, momentum, affine, track_running_stats),
            self.activation,
        )

    def _graph_block(self, inp: int, output: int, k: int = 3) -> GSequential:
        """
        k graph convolutional layers with ReLU activation
        node will have info from k-hop neighbors
        """
        layers = []
        for _ in range(k):
            layers.append((GCNConv(inp, output, improved=True), "x, edge_index -> x"))
            layers.append(self.activation)
            layers.append(self.drop)
            inp = output
        return GSequential(input_args="x, edge_index", modules=layers)

    def _format_to_conv2d(self, x: Tensor) -> Tensor:
        """
        Conv2d expects (B, C, W, H), we have (B, N, C, W, H),
        so we merge B and N dimensions to avoid looping
        """
        (B, N, C, W, H) = x.shape
        return x.view(B * N, C, W, H)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass through the network
        x: (B, N, C, W, H)
        return: (B, N, n_actions)
        """
        # convolutional layers
        x = self._format_to_conv2d(x).to(self.device)

        # x = self.conv1(x)
        # x = self.drop(x)
        # x = self.conv2(x)
        # x = self.drop(x)
        # x = self.conv3(x)  # output [B*N, 128, wl//8, hl//8]
        # x = self.pool(x)  # output [B*N, 128, 1, 1]
        # x = self.compress(x)
        # x = self.flatten(x)  # output [B*N, 128]

        x = self.flatten(
            self.compress(
                self.pool(self.conv3(self.drop(self.conv2(self.drop(self.conv1(x))))))
            )
        )

        # graph convolutional layer
        edge_index = self._agents_to_edge_index(self.S).to(self.device)
        # x = self.cgnn(x, edge_index)  # [B*N, 128]

        # fully connected layers
        # x = self.activation(x)
        # x = self.fc(x)
        return self.fc(self.activation(self.cgnn(x, edge_index)))
