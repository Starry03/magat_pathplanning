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
from torch.optim import Adam
from torch_geometric.nn import GCNConv, ChebConv, Sequential as GSequential
from lightning.pytorch import LightningModule
from torch.optim import lr_scheduler
from torch.nn import CrossEntropyLoss
from torchmetrics import Accuracy
from utils.new_simulator import multiRobotSimNew
from utils.metrics import MonitoringMultiAgentPerformance
from dataloader.IL_DataLoader import IL_DataLoader
from logger import logger
from tqdm import tqdm
import time
from test.renderer import Renderer
from layer.time_graph import TimeDelayedAggregation

class Model(LightningModule):
    def __init__(self, config) -> None:
        super().__init__()
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.config = config
        self.config.device = self.dev
        self.n_agents = 10
        self.FOV = 4
        self.wl, self.hl = self.FOV + 2, self.FOV + 2
        self.CHANNELS: int = 3
        self.n_actions: int = 5
        self.E = 1
        self.nGraphFilterTaps = 3

        self.loss = CrossEntropyLoss()
        self.train_acc = Accuracy(task="multiclass", num_classes=self.n_actions)
        self.val_acc = Accuracy(task="multiclass", num_classes=self.n_actions)
        self.robot = multiRobotSimNew(config)
        self.renderer = Renderer(self.robot)
        self.recorder = MonitoringMultiAgentPerformance(self.config)


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
        self.tgnn = TimeDelayedAggregation(128, self.nGraphFilterTaps)
        self.flatten = Flatten()
        # self.cgnn = ChebConv(
        #     in_channels=128,
        #     out_channels=128,
        #     K=self.nGraphFilterTaps,  # K-hop polynomial filter
        #     normalization="sym",  # Symmetric normalization (I - D^{-1/2} A D^{-1/2})
        # )
        # self.cgnn = self._graph_block_paper()
        self.fc = Linear(128 * self.nGraphFilterTaps, self.n_actions)
        self.S = ones(1, self.E, self.n_agents, self.n_agents, device=self.dev)

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
            self.S = ones(1, self.E, self.n_agents, self.n_agents, device=self.dev)
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
        self.S = self.S.to(self.dev)

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
        # print(x.shape)
        x = self.conv1(x)
        # print(x.shape)
        x = self.drop(x)
        x = self.conv2(x)
        # print(x.shape)
        x = self.drop(x)
        x = self.conv3(x)
        # print(x.shape)
        x = self.drop(x)
        x = self.pool(x)
        # print(x.shape)
        x = self.compress(x)
        # print(x.shape)
        x = self.flatten(x) # output [B*N, 128]
        # print(x.shape)
        
        current_feature = x.view(B, N, -1) # [B, N, 128]

        # time delayed graph
        # if x_prev is None, we assume it's the first step, so we pad with zeros
        # x_prev shape expected: [B, N, T, 128]
        # we need T = nGraphFilterTaps - 1
        
        if x_prev is None:
            x_prev = torch.zeros(B, N, self.nGraphFilterTaps - 1, 128, device=self.dev)

        y_time = [x] # x is [B*N, 128]
        if len(self.S.shape) == 4:
            S = self.S.squeeze(1) # [B, N, N]
        else:
            S = self.S
             
        if len(S.shape) == 4 and S.shape[1] == 1:
            S = S.squeeze(1)
        
        # for t in range(self.nGraphFilterTaps - 1):
        #     prev_feat = x_prev[:, :, t, :]
            
        #     filtered = torch.matmul(S, prev_feat)
            
        #     filtered = filtered.view(B * N, 128)
        #     y_time.append(filtered)
        # x_concatenated = cat(y_time, dim=1) # [B*N, 128 * K]
        
        # x = x_concatenated
        x = self.tgnn(x, x_prev, S)
        # print(x.shape)
        x = self.activation(x)
        # print(x.shape)
        x = self.fc(x)  # [B*N, n_actions]
        # print(x.shape)
        # exit(0)
        return x, current_feature

    @staticmethod
    def load_from_checkpoint(path: str, config) -> "Model":
        """
        Load a model from a PyTorch Lightning checkpoint file (.ckpt)

        Args:
            path: Path to the checkpoint file

        Returns:
            Model instance with loaded weights
        """
        model = Model(config)
        checkpoint = torch.load(path, map_location=model.dev)

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
    
    def training_step(self, batch, batch_idx):
        batch_input, batch_target, _, batch_GSO, _ = batch
        (B, N, _, _, _) = batch_input.shape
        batch_target = batch_target.reshape(B * N, self.n_actions)
        self.addGSO(batch_GSO)
        logits, _ = self(batch_input)
        targets = torch.max(batch_target, 1)[1]
        loss = self.loss(logits, targets)
        self.train_acc(logits, targets)
        self.log(
            "train_acc", self.train_acc, prog_bar=True, on_step=True, on_epoch=True
        )
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        # batch_input, batch_target, _, batch_GSO, _ = batch
        # (B, N, _, _, _) = batch_input.shape
        # batch_target = batch_target.reshape(B * N, self.n_actions)
        # self.addGSO(batch_GSO)
        # logits, _ = self(batch_input)
        # targets = torch.max(batch_target, 1)[1]
        # loss = self.loss(logits, targets)
        # self.val_acc(logits, targets)
        # self.log("val_acc", self.val_acc, prog_bar=True, on_step=False, on_epoch=True)
        # self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        # return loss
        return self.test_step(batch, batch_idx, False)

    def multiAgent_ActionPolicy(
        self, input, load_target, makespanTarget, tensor_map, ID_dataset, mode
    ):
        self.robot.setup(
            input, load_target, makespanTarget, tensor_map, ID_dataset, mode
        )
        maxstep = self.robot.getMaxstep()
        allReachGoal = False
        noReachGoalbyCollsionShielding = False
        check_collisionFreeSol = False
        check_CollisionHappenedinLoop = False
        check_CollisionPredictedinLoop = False
        findOptimalSolution = False
        compare_makespan, compare_flowtime = self.robot.getOptimalityMetrics()
        currentStep = 0
        
        # Initialize Renderer
        logger.debug("New map")

        Case_start = time.time()
        Time_cases_ForwardPass = []
        for step in range(maxstep):
            currentStep = step + 1
            currentState = self.robot.getCurrentState()
            currentStateGPU = currentState.to(self.config.device)

            gso = self.robot.getGSO(step)
            gsoGPU = gso.to(self.config.device)
            self.addGSO(gsoGPU)
            step_start = time.time()
            actionVec_predict, _ = self(currentStateGPU)  # B x N X 5
            if self.config.batch_numAgent:
                actionVec_predict = actionVec_predict.detach().cpu()
            else:
                actionVec_predict = [ts.detach().cpu() for ts in actionVec_predict]
            time_ForwardPass = time.time() - step_start
            Time_cases_ForwardPass.append(time_ForwardPass)
            allReachGoal, check_moveCollision, check_predictCollision = self.robot.move(
                actionVec_predict, currentStep
            )
            
            # Render step
            if self.renderer:                 
                self.renderer.update_stats(actionVec_predict, [check_moveCollision]) 
                self.renderer.render()

            if check_moveCollision:
                check_CollisionHappenedinLoop = True
            if check_predictCollision:
                check_CollisionPredictedinLoop = True
            if allReachGoal:
                break
            elif currentStep >= (maxstep):
                break

        num_agents_reachgoal = self.robot.count_numAgents_ReachGoal()
        store_GSO, store_communication_radius = self.robot.count_GSO_communcationRadius(
            currentStep
        )

        if allReachGoal and not check_CollisionHappenedinLoop:
            check_collisionFreeSol = True
            noReachGoalbyCollsionShielding = False
            findOptimalSolution, compare_makespan, compare_flowtime = (
                self.robot.checkOptimality(True)
            )
            if self.config.log_anime and self.config.mode == "test":
                self.robot.save_success_cases("success")

        if currentStep >= (maxstep):
            findOptimalSolution, compare_makespan, compare_flowtime = (
                self.robot.checkOptimality(False)
            )

        if currentStep >= (maxstep) and not allReachGoal:
            if self.config.log_anime and self.config.mode == "test":
                self.robot.save_success_cases("failure")

        if (
            currentStep >= (maxstep)
            and not allReachGoal
            and check_CollisionPredictedinLoop
            and not check_CollisionHappenedinLoop
        ):
            findOptimalSolution, compare_makespan, compare_flowtime = (
                self.robot.checkOptimality(False)
            )
            # print("### Case - {} -Step{} exceed maxstep({})- ReachGoal: {} due to CollsionShielding \n".format(ID_dataset,currentStep,maxstep, allReachGoal))
            noReachGoalbyCollsionShielding = True
            if self.config.log_anime and self.config.mode == "test":
                self.robot.save_success_cases("failure")
        time_record = time.time() - Case_start

        if self.config.mode == "test":
            exp_status = (
                "################## {} - End of loop ################## ".format(
                    self.config.exp_name
                )
            )
            case_status = "####### Case{} \t Computation time:{} \t Step{}/{}\t- AllReachGoal-{}\n".format(
                ID_dataset, time_record, currentStep, maxstep, allReachGoal
            )

            logger.info("{} \n {}".format(exp_status, case_status))

        return (
            allReachGoal,
            noReachGoalbyCollsionShielding,
            findOptimalSolution,
            check_collisionFreeSol,
            check_CollisionPredictedinLoop,
            compare_makespan,
            compare_flowtime,
            num_agents_reachgoal,
            store_GSO,
            store_communication_radius,
            time_record,
            Time_cases_ForwardPass,
        )

    def test_step(self, batch, batch_idx, log=True):
        """

        single dim batch
        """
        if log:
            logger.info("test started")
        self.recorder.reset()
        (
            batch_input,
            batch_target,
            batch_info_tuple,
            batch_GSO,
            batch_map_tensor,
        ) = batch

        log_result = self.multiAgent_ActionPolicy(
            batch_input,
            batch_target,
            batch_info_tuple,
            batch_map_tensor.cpu(),
            self.recorder.count_validset,
            mode="test",
        )
        self.recorder.update(self.robot.getMaxstep(), log_result)
        if log:
            logger.info(
                "Accurracy(reachGoalnoCollision): {} \n  "
                "DeteriorationRate(MakeSpan): {} \n  "
                "DeteriorationRate(FlowTime): {} \n  "
                "Rate(collisionPredictedinLoop): {} \n  "
                "Rate(FailedReachGoalbyCollisionShielding): {} \n ".format(
                    round(self.recorder.rateReachGoal, 4),
                    round(self.recorder.avg_rate_deltaMP, 4),
                    round(self.recorder.avg_rate_deltaFT, 4),
                    round(self.recorder.rateCollisionPredictedinLoop, 4),
                    round(self.recorder.rateFailedReachGoalSH, 4),
                )
            )
        return self.recorder.rateReachGoal

    def configure_optimizers(self):
        optimizer = Adam(
            self.parameters(),
            lr=self.config.get("learning_rate", 1e-3),
            weight_decay=self.config.get("weight_decay", 0),
        )
        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer,
            T_max=self.config.get("max_epochs", 10),
            eta_min=1e-6,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
        }

    def test_single(self, mode, data_loader: IL_DataLoader, limit: int = -1):
        """
        One cycle of model validation
        :return:
        """
        logger.info("test started")
        self.eval()
        if mode == "test":
            dataloader = data_loader.test_loader
            label = "test"
        elif mode == "test_trainingSet":
            dataloader = data_loader.test_trainingSet_loader
            label = "test_training"
        else:
            dataloader = data_loader.valid_loader
            label = "valid"

        count: int = 0
        self.recorder.reset()
        with torch.no_grad():
            for input, target, makespan, _, tensor_map in tqdm(
                dataloader, desc=f"Testing on {label} set", total=len(dataloader)
            ):
                inputGPU = input.to(self.dev)
                targetGPU = target.to(self.dev)
                log_result = self.multiAgent_ActionPolicy(
                    inputGPU,
                    targetGPU,
                    makespan,
                    tensor_map,
                    self.recorder.count_validset,
                    mode,
                )
                self.recorder.update(self.robot.getMaxstep(), log_result)
                logger.info("current rateReachGoal: {}".format(self.recorder.rateReachGoal))
                count += 1
                if limit != -1 and count >= limit:
                    break

        logger.info(
            "Accurracy(reachGoalnoCollision): {} \n  "
            "DeteriorationRate(MakeSpan): {} \n  "
            "DeteriorationRate(FlowTime): {} \n  "
            "Rate(collisionPredictedinLoop): {} \n  "
            "Rate(FailedReachGoalbyCollisionShielding): {} \n ".format(
                round(self.recorder.rateReachGoal, 4),
                round(self.recorder.avg_rate_deltaMP, 4),
                round(self.recorder.avg_rate_deltaFT, 4),
                round(self.recorder.rateCollisionPredictedinLoop, 4),
                round(self.recorder.rateFailedReachGoalSH, 4),
            )
        )
        return self.recorder.rateReachGoal