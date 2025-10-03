import logging
from torch.nn import Conv2d, Linear, Flatten, ReLU, MaxPool2d, Module, Sequential, BatchNorm2d
from torch_geometric.nn import GCNConv

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

        self.n_agents = config["num_agents"]
        self.FOV = config["FOV"]
        self.wl, self.hl = self.FOV + 2, self.FOV + 2 # from paper

        self.conv1 = None
        self.conv2 = None
        self.conv3 = None
        self.cgnn = GCNConv(128, 128)

    def addGSO(self, x):
        pass

    @staticmethod
    def conv_block(inp: int, output: int, kernel_size: int = 3, stride: int = 1, padding: int = 0) -> Sequential:
        return Sequential(
            Conv2d(in_channels=inp, out_channels=output, kernel_size=kernel_size, stride=stride, padding=padding),
            BatchNorm2d(output),
            ReLU(),
            MaxPool2d(kernel_size=kernel_size, stride=stride, padding=padding),
            Conv2d(in_channels=output, out_channels=output, kernel_size=kernel_size, stride=stride, padding=padding),
            BatchNorm2d(output),
            ReLU(),
        )
    
    def forward(self, x):
        return Exception("Not implemented yet")