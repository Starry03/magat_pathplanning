from torch import Tensor, stack, empty, cat, long, isnan, ones, Size
import torch
from torch.nn import (
    Module,
    Conv2d,
    Linear,
    Flatten,
    ReLU,
    MaxPool2d,
    Sequential,
    BatchNorm2d,
    Dropout,
    AdaptiveAvgPool2d,
)
from torch_geometric.nn import GCNConv, ChebConv, Sequential as GSequential


class Model(Module):
    def __init__(self) -> None:
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.n_agents = 10
        self.FOV = 4
        self.wl, self.hl = self.FOV + 2, self.FOV + 2
        self.CHANNELS: int = 3
        self.n_actions: int = 5
        self.E = 1
        self.nGraphFilterTaps = 3

        # layers
        self.pool = AdaptiveAvgPool2d((1, 1))
        self.drop = Dropout(p=0.2)
        self.activation = ReLU(inplace=False)
        self.conv1 = self._conv_block(self.CHANNELS, 32)
        self.conv2 = self._conv_block(32, 64)
        self.conv3 = self._conv_block(64, 128)
        self.compress = Sequential(
            Conv2d(128, 128, kernel_size=1, stride=1, padding=0),
            BatchNorm2d(128),
            self.activation,
        )
        self.flatten = Flatten()
        self.flatten = Flatten()
        # self.cgnn = ChebConv(
        #     in_channels=128,
        #     out_channels=128,
        #     K=self.nGraphFilterTaps,  # K-hop polynomial filter
        #     normalization="sym",  # Symmetric normalization (I - D^{-1/2} A D^{-1/2})
        # )
        # self.cgnn = self._graph_block_paper()
        self.fc = Linear(128 * self.nGraphFilterTaps, self.n_actions)
        self.S = ones(1, self.E, self.n_agents, self.n_agents, device=self.device)

    def _agents_to_edge_index(self, S: Tensor):
        """
        Vectorized edge index computation
        return: edge_index in shape [2, E]
        """
        B, _, N, _ = S.shape
        
        # Find all non-zero entries in S [B, E, N, N]
        # indices will be [num_edges, 4] -> (b, e, i, j)
        indices = S.nonzero(as_tuple=False)
        
        if indices.numel() == 0:
            return empty(2, 0, dtype=long, device=S.device)
            
        b_idx = indices[:, 0]
        row_idx = indices[:, 2]
        col_idx = indices[:, 3]
        
        # Calculate global indices for the big graph
        # rows = i + b * N
        # cols = j + b * N
        rows = row_idx + b_idx * N
        cols = col_idx + b_idx * N
        
        return stack([rows, cols], dim=0)

    def addGSO(self, S: Tensor) -> None:
        """
        Training is made on different steps of different maps so we need to get the GSO everytime

        this function refers to the original one in the repository
        """
        if S.shape == Size([1, 1]):
            self.S = ones(1, self.E, self.n_agents, self.n_agents, device=self.device)
            return
        if self.E == 1:
            assert len(S.shape) == 3
            self.S = S.unsqueeze(1)
        else:
            assert len(S.shape) == 4
            assert S.shape[1] == self.E
            self.S = S

        self.S[isnan(self.S)] = 0
        self.S[self.S > 0] = 1
        self.S = self.S.float()
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

    @DeprecationWarning
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
        # if (len(x.shape) == 4):
        #     return x
        (B, N, C, W, H) = x.shape
        return x.view(B * N, C, W, H)

    def forward(self, x: Tensor, x_prev: Tensor | None = None) -> tuple[Tensor, Tensor]:
        """
        Forward pass through the network
        x: (B, N, C, W, H)
        x_prev: (B, N, T, 128) or None
        return: (B, N, n_actions), (B, N, 128)
        """
        (B, N, C, W, H) = x.shape
        # convolutional layers
        x = self._format_to_conv2d(x)
        x = self.flatten(
            self.compress(
                self.pool(self.conv3(self.drop(self.conv2(self.drop(self.conv1(x))))))
            )
        )  # output [B*N, 128]
        
        current_feature = x.view(B, N, -1) # [B, N, 128]

        # time delayed graph
        # if x_prev is None, we assume it's the first step, so we pad with zeros
        # x_prev shape expected: [B, N, T, 128]
        # we need T = nGraphFilterTaps - 1
        
        if x_prev is None:
             x_prev = torch.zeros(B, N, self.nGraphFilterTaps - 1, 128, device=self.device)

        y_time = [x] # x is [B*N, 128]
        if len(self.S.shape) == 4:
             S = self.S.squeeze(1) # [B, N, N]
        else:
             S = self.S
             
        if len(S.shape) == 4 and S.shape[1] == 1:
             S = S.squeeze(1)
        
        for t in range(self.nGraphFilterTaps - 1):
            prev_feat = x_prev[:, :, t, :]
            
            filtered = torch.matmul(S, prev_feat)
            
            filtered = filtered.view(B * N, 128)
            y_time.append(filtered)
        x_concatenated = cat(y_time, dim=1) # [B*N, 128 * K]
        
        x = x_concatenated
        x = self.activation(x)
        x = self.fc(x)  # [B*N, n_actions]

        return x, current_feature

    @staticmethod
    def load_from_checkpoint(path: str) -> "Model":
        """
        Load a model from a PyTorch Lightning checkpoint file (.ckpt)

        Args:
            path: Path to the checkpoint file

        Returns:
            Model instance with loaded weights
        """
        model = Model()
        checkpoint = torch.load(path, map_location=model.device)

        # PyTorch Lightning checkpoints have 'state_dict' key
        if "state_dict" in checkpoint:
            # Remove 'model.' prefix if present (common in Lightning checkpoints)
            state_dict = checkpoint["state_dict"]
            new_state_dict = {}
            for key, value in state_dict.items():
                new_key = key.replace("model.", "") if key.startswith("model.") else key
                new_state_dict[new_key] = value
            model.load_state_dict(new_state_dict)
        else:
            model.load_state_dict(checkpoint)

        return model

    def save_to_checkpoint(self, path: str) -> None:
        """
        Save the model's state dictionary to a PyTorch Lightning checkpoint file (.ckpt)

        Args:
            path: Path to save the checkpoint file
        """
        torch.save(self.state_dict(), path)